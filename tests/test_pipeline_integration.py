from __future__ import annotations

import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from flare.detection.detector import ChannelWindowManager, IsolationForestDetector
from flare.detection.events import AnomalyDetected, WindowStats
from flare.ingestion.frame import TelemetryFrame
from flare.llm.backend import LLMBackend
from flare.llm.retrieval import KnowledgeBase
from flare.pipeline import FLAREPipeline

# ---------------------------------------------------------------------------
# Constants and skip marker
# ---------------------------------------------------------------------------

MODEL_PATH = Path("models/isolation_forest.joblib")
REGISTRY_PATH = "data/knowledge_base/reference_registry.json"
DOCS_FOLDER = Path("data/knowledge_base")
REGISTRY_CONFIG = Path("config/registry_config.json")
PROFILES_PATH = Path("config/mission_profiles.yaml")
MISSION_ID = "SKYROOT_M1"

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="No trained model — run: python -m flare.detection.train "
           "--csv data/telemetry.csv --mission config/mission_profiles.yaml "
           "--mission-id SKYROOT_M1 --output models/",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mission_profile() -> dict:
    with PROFILES_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)[MISSION_ID]


class _MockLLMBackend(LLMBackend):
    @property
    def backend_id(self) -> str:
        return "mock:test"

    def generate(self, system: str, user: str) -> dict:
        return {"text": "Test recommendation.", "logprobs": []}


class _TrackingLLMBackend(LLMBackend):
    """Mock LLM backend that records how many times generate() is called."""

    def __init__(self) -> None:
        self.generate_call_count: int = 0

    @property
    def backend_id(self) -> str:
        return "tracking:test"

    def generate(self, system: str, user: str) -> dict:
        self.generate_call_count += 1
        return {"text": "Test recommendation.", "logprobs": []}


def _make_pipeline(tmp_path: Path, mission_profile: dict) -> FLAREPipeline:
    detector = IsolationForestDetector(
        model_path=str(MODEL_PATH),
        mission_profile=mission_profile,
    )
    kb = KnowledgeBase(
        docs_folder=DOCS_FOLDER,
        registry_config_path=REGISTRY_CONFIG,
    )
    return FLAREPipeline(
        mission_id=MISSION_ID,
        registry_path=REGISTRY_PATH,
        hot_storage_path=str(tmp_path / "hot_storage"),
        cold_storage_path=str(tmp_path / "cold_storage"),
        llm_backend=_MockLLMBackend(),
        detector=detector,
        knowledge_base=kb,
        incident_log_path=tmp_path / "incidents.jsonl",
        mission_profile=mission_profile,
        top_k=3,
    )


def _make_frame(value: float, row_index: int = 0) -> TelemetryFrame:
    return TelemetryFrame(
        apid=100,
        sequence_count=row_index % 16384,
        sequence_flag="standalone",
        timestamp=float(row_index),
        channel_id="BUS_VOLTAGE",
        value=value,
        unit="V",
        subsystem="EPS",
        source_file="test.csv",
        row_index=row_index,
    )


def _make_window_stats(value: float, profile: dict) -> WindowStats:
    ranges = profile["nominal_ranges"]["BUS_VOLTAGE"]
    return WindowStats(
        mean=value,
        std=0.5,
        min=value - 1.0,
        max=value + 1.0,
        nominal_low=float(ranges[0]),
        nominal_high=float(ranges[1]),
        window_size=1,
    )


# ---------------------------------------------------------------------------
# Test 1 — pipeline initialises
# ---------------------------------------------------------------------------

@requires_model
def test_pipeline_initialises(tmp_path: Path) -> None:
    profile = _load_mission_profile()
    pipeline = _make_pipeline(tmp_path, profile)
    assert pipeline._state_handler is not None
    assert pipeline.repository is not None


# ---------------------------------------------------------------------------
# Test 2 — nominal frame produces no incident
# ---------------------------------------------------------------------------

@requires_model
def test_process_frame_nominal(tmp_path: Path) -> None:
    profile = _load_mission_profile()
    pipeline = _make_pipeline(tmp_path, profile)

    # BUS_VOLTAGE=30.0 V scores ~-0.41 with the StandardScaler-normalised model,
    # which is above the calibrated anomaly_threshold=-0.58 → is_anomaly=False.
    frame = _make_frame(value=30.0, row_index=0)
    context_window: tuple[TelemetryFrame, ...] = ()
    window_stats = _make_window_stats(30.0, profile)

    pipeline.process_frame(frame, context_window, window_stats)

    # is_anomaly=False → pipeline returns early → no incident opened
    assert pipeline.repository.open_incidents() == []


# ---------------------------------------------------------------------------
# Test 3 — anomalous frame processed end-to-end
# ---------------------------------------------------------------------------

@requires_model
def test_process_frame_anomaly_end_to_end(tmp_path: Path) -> None:
    profile = _load_mission_profile()
    pipeline = _make_pipeline(tmp_path, profile)

    # Use ChannelWindowManager for realism (as specified)
    channel_manager = ChannelWindowManager(window_size=60)
    # TANK_PRESSURE=150000 Pa is outside nominal [200000, 350000] and scores
    # ~-0.71 with the calibrated model (below anomaly_threshold=-0.58). BUS_VOLTAGE
    # values cannot reach -0.58 with the StandardScaler-normalised model.
    # Inject MIN_CONSECUTIVE_FRAMES=3 consecutive anomalous frames so the
    # persistence check passes. A single frame is pruned by design.
    for i in range(3):
        frame = TelemetryFrame(
            apid=201, sequence_count=i % 16384, sequence_flag="standalone",
            timestamp=float(i), channel_id="TANK_PRESSURE",
            value=150000.0, unit="Pa", subsystem="PROP",
            source_file="test.csv", row_index=i,
        )
        context_window, window_stats = channel_manager.update_with_profile(frame, profile)
        pipeline.process_frame(frame, context_window, window_stats)

    # Anomaly must have been escalated — incident log has entries.
    # Incident state may be INVESTIGATING (FLAGGED verdict) or RESOLVED (SAFE verdict).
    incident_log = tmp_path / "incidents.jsonl"
    assert incident_log.exists() and incident_log.stat().st_size > 0, \
        "Expected incident log with entries — anomaly not detected or not escalated"

    # At least one report file written (SAFE/FLAGGED/ERROR all produce a file)
    hot_root = tmp_path / "hot_storage" / MISSION_ID
    reports = list((hot_root / "reports").rglob("*.json")) if (hot_root / "reports").exists() else []
    errors = list((hot_root / "reports" / "errors").rglob("*.json")) if (hot_root / "reports" / "errors").exists() else []
    assert reports or errors, (
        f"Expected at least one report file under {hot_root}/reports/ or reports/errors/"
    )


# ---------------------------------------------------------------------------
# Test 4 — incident replayed after restart
# ---------------------------------------------------------------------------

@requires_model
def test_incident_replay_after_restart(tmp_path: Path) -> None:
    profile = _load_mission_profile()

    # First pipeline run — inject MIN_CONSECUTIVE_FRAMES=3 anomalous frames so
    # the persistence check passes and an incident is created.
    pipeline1 = _make_pipeline(tmp_path, profile)
    channel_manager = ChannelWindowManager(window_size=60)
    for i in range(3):
        frame = TelemetryFrame(
            apid=201, sequence_count=i % 16384, sequence_flag="standalone",
            timestamp=float(i), channel_id="TANK_PRESSURE",
            value=150000.0, unit="Pa", subsystem="PROP",
            source_file="test.csv", row_index=i,
        )
        context_window, window_stats = channel_manager.update_with_profile(frame, profile)
        pipeline1.process_frame(frame, context_window, window_stats)

    import json as _json

    # Incident log must exist — anomaly was detected and escalated.
    # Incident may be RESOLVED (SAFE verdict) or INVESTIGATING (FLAGGED).
    incident_log = tmp_path / "incidents.jsonl"
    assert incident_log.exists() and incident_log.stat().st_size > 0, \
        "No incident log written by pipeline1 — anomaly not detected or not escalated"

    # Read original_id and final state from log
    with incident_log.open(encoding="utf-8") as fh:
        original_id = _json.loads(fh.readline())["incident_id"]
    final_state = "DETECTED"
    with incident_log.open(encoding="utf-8") as fh:
        for line in fh:
            rec = _json.loads(line.strip())
            if rec["incident_id"] == original_id:
                final_state = rec["to_state"]

    # Second pipeline run — replay from same incident_log_path
    pipeline2 = _make_pipeline(tmp_path, profile)

    # Incident must have been replayed (any state)
    replayed = pipeline2.repository.get(original_id)
    assert replayed is not None, (
        f"Incident {original_id} not found after replay"
    )
    assert replayed.current_state == final_state, (
        f"Replayed state {replayed.current_state!r} != logged final state {final_state!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — env var check fires before any pipeline work
# ---------------------------------------------------------------------------

def test_env_var_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from run_pipeline import check_env_vars

    monkeypatch.delenv("ASTR_O_REPORT_SECRET", raising=False)

    with pytest.raises(SystemExit):
        check_env_vars(backend="openai")


# ---------------------------------------------------------------------------
# Test 6 — Stage 1 None guard prevents Stage 2 from running
# ---------------------------------------------------------------------------

@requires_model
def test_stage1_none_guard(tmp_path: Path) -> None:
    profile = _load_mission_profile()
    pipeline = _make_pipeline(tmp_path, profile)

    # Replace Stage 1 state_handler with one that returns None to verify
    # process_frame() stops before audit when Stage 1 fails.
    pipeline._state_handler = lambda e: None

    channel_manager = ChannelWindowManager(window_size=60)
    frame = _make_frame(value=5.0, row_index=0)
    context_window, window_stats = channel_manager.update_with_profile(frame, profile)

    pipeline.process_frame(frame, context_window, window_stats)

    # Stage 2 never ran — no files written to hot_storage
    hot_root = tmp_path / "hot_storage"
    all_json = list(hot_root.rglob("*.json")) if hot_root.exists() else []
    assert all_json == [], (
        f"Stage 2 should not have run, but found files: {all_json}"
    )


# ---------------------------------------------------------------------------
# Test 7 — pruning suppresses noise and sustains on real anomalies
# ---------------------------------------------------------------------------

def test_pruning_suppresses_noise(tmp_path: Path) -> None:
    """
    Verifies the three-stage pruning logic in process_frame():
      A — single anomalous frame is suppressed (persistence count < 3)
      B — three consecutive anomalous frames escalate
      C — sustained anomaly keeps signaling (counter not reset after escalation)
      D — nominal frame resets counter; next single anomalous frame is pruned again
    """
    from flare.pipeline import FLAREPipeline

    profile = _load_mission_profile()
    ranges = profile["nominal_ranges"]["BUS_VOLTAGE"]
    nominal_low, nominal_high = float(ranges[0]), float(ranges[1])

    # Frame/stats used for every anomalous injection
    anomalous_value = nominal_low - 10.0  # clearly outside spec
    anomalous_frame = TelemetryFrame(
        apid=100, sequence_count=0, sequence_flag="standalone",
        timestamp=0.0, channel_id="BUS_VOLTAGE",
        value=anomalous_value, unit="V", subsystem="EPS",
        source_file="test.csv", row_index=0,
    )
    anomalous_stats = WindowStats(
        mean=anomalous_value, std=0.5,
        min=anomalous_value - 1.0, max=anomalous_value + 1.0,
        nominal_low=nominal_low, nominal_high=nominal_high,
        window_size=10,
    )

    # Frame/stats used for the nominal reset injection
    nominal_frame = TelemetryFrame(
        apid=100, sequence_count=99, sequence_flag="standalone",
        timestamp=99.0, channel_id="BUS_VOLTAGE",
        value=30.0, unit="V", subsystem="EPS",
        source_file="test.csv", row_index=99,
    )
    nominal_stats = WindowStats(
        mean=30.0, std=0.5, min=29.0, max=31.0,
        nominal_low=nominal_low, nominal_high=nominal_high,
        window_size=10,
    )

    # Mock detector: returns a controlled AnomalyDetected based on frame value
    mock_detector = MagicMock()

    def _mock_score(frame, context_window, window_stats, mission_id, incident_id):
        is_anom = (frame.value < nominal_low or frame.value > nominal_high)
        return AnomalyDetected(
            event_id=str(uuid.uuid4()),
            incident_id=incident_id,
            mission_id=mission_id,
            frame=frame,
            anomaly_score=-0.5 if is_anom else 0.2,
            decision_threshold=-0.1,
            is_anomaly=is_anom,
            context_window=context_window,
            window_stats=window_stats,
            detector_version="test",
            detection_method="isolation_forest",
            detected_at=0.0,
        )

    mock_detector.score.side_effect = _mock_score

    tracking_llm = _TrackingLLMBackend()
    kb = KnowledgeBase(docs_folder=DOCS_FOLDER, registry_config_path=REGISTRY_CONFIG)

    pipeline = FLAREPipeline(
        mission_id=MISSION_ID,
        registry_path=REGISTRY_PATH,
        hot_storage_path=str(tmp_path / "hot_storage"),
        cold_storage_path=str(tmp_path / "cold_storage"),
        llm_backend=tracking_llm,
        detector=mock_detector,
        knowledge_base=kb,
        incident_log_path=tmp_path / "incidents.jsonl",
        mission_profile=profile,
        top_k=3,
    )

    ctx: tuple[TelemetryFrame, ...] = ()

    # --- Part A: single anomalous frame is suppressed ---
    pipeline.process_frame(anomalous_frame, ctx, anomalous_stats)
    assert pipeline.repository.open_incidents() == [], (
        "Part A: single anomalous frame should be pruned (count=1, need 3)"
    )

    # --- Part B: three consecutive frames escalate ---
    pipeline.process_frame(anomalous_frame, ctx, anomalous_stats)  # count=2
    pipeline.process_frame(anomalous_frame, ctx, anomalous_stats)  # count=3 → escalate
    # Incident may be immediately RESOLVED by a SAFE verdict; check LLM was called instead
    assert tracking_llm.generate_call_count >= 1, (
        "Part B: LLM must be called on third consecutive anomalous frame — escalation failed"
    )

    # --- Part C: sustained anomaly keeps signaling ---
    pipeline.process_frame(anomalous_frame, ctx, anomalous_stats)  # count=4 → escalate again
    assert tracking_llm.generate_call_count >= 2, (
        "Part C: LLM must be called for both frame 3 and frame 4 — "
        "counter reset after escalation would silence frame 4 (safety failure)"
    )

    # --- Part D: nominal frame resets counter; next anomalous frame is pruned ---
    pipeline.process_frame(nominal_frame, ctx, nominal_stats)      # resets counter to 0
    incidents_after_c = len(pipeline.repository.open_incidents())
    pipeline.process_frame(anomalous_frame, ctx, anomalous_stats)  # count=1 → pruned
    assert len(pipeline.repository.open_incidents()) == incidents_after_c, (
        "Part D: single anomalous frame after counter reset should be pruned again"
    )


# ---------------------------------------------------------------------------
# Shared helpers for audit resolution tests
# ---------------------------------------------------------------------------

_AUDIT_METRICS = {
    "hallucination_score": 0.05,
    "retrieval_quality": 0.90,
    "faithfulness": 0.95,
    "token_confidence": 0.85,
    "latency_ms": 180.0,
    "cost_usd": 0.0001,
    "overall_safety_score": 0.92,
}

_FULL_MAP = {k: "complete" for k in (
    "registry_lookup", "contradiction_detector", "token_confidence",
    "source_verifier", "groundedness", "criteria_gate",
    "causal_chain_mapper", "integrity_signer", "storage",
)}


def _make_investigating_incident(tmp_path: Path) -> tuple:
    """Return (repository, incident_id) with incident in INVESTIGATING state."""
    from flare.state.incident import IncidentRepository
    repo = IncidentRepository(log_path=tmp_path / "incidents.jsonl")
    incident_id = str(uuid.uuid4())
    incident = repo.open(
        incident_id=incident_id,
        mission_id=MISSION_ID,
        channel_id="BUS_VOLTAGE",
        subsystem="EPS",
        opened_at=0.0,
    )
    incident.transition_to("INVESTIGATING", trigger="test_setup", evidence={})
    repo.save(incident)
    return repo, incident_id


def _make_reco_event(incident_id: str) -> object:
    from flare.llm.events import LogprobToken, RecommendationGenerated, RetrievedChunk
    chunk = RetrievedChunk(
        chunk_id="doc_001_chunk_000",
        source="voltage_spec.txt",
        text="BUS_VOLTAGE nominal range: 28–32 V.",
        rank=1,
        dense_score=0.91,
        sparse_score=0.75,
        rrf_score=0.031,
        source_tier="CRITICAL",
    )
    return RecommendationGenerated(
        event_id=str(uuid.uuid4()),
        incident_id=incident_id,
        parent_event_id=str(uuid.uuid4()),
        mission_id=MISSION_ID,
        llm_text="Bus voltage is within safe operating range.",
        logprobs=(LogprobToken(token="Bus", logprob=-0.05),),
        retrieved_chunks=(chunk,),
        query="BUS_VOLTAGE anomaly: value 21.5 V, threshold -0.58, subsystem EPS",
        retrieval_method="hybrid",
        top_k=3,
        all_ranked_chunks=(
            {"chunk_id": "doc_001_chunk_000", "rank": 1,
             "dense_score": 0.91, "sparse_score": 0.75, "rrf_score": 0.031},
        ),
        llm_backend="mock:test",
        generated_at=1748000000.0,
    )


# ---------------------------------------------------------------------------
# Test 8 — incident resolves on SAFE audit verdict
# ---------------------------------------------------------------------------

def test_incident_resolves_on_safe_verdict(tmp_path: Path) -> None:
    from flare.audit.handlers import make_audit_handler

    repo, incident_id = _make_investigating_incident(tmp_path)

    mock_pipeline = MagicMock()
    mock_pipeline.hot_storage_path = str(tmp_path / "hot_storage")
    mock_pipeline.mission_id = MISSION_ID
    mock_pipeline.process_span.return_value = {
        "span_id": str(uuid.uuid4()),
        "status": "SAFE",
        "signed_report": {"data": "signed_content", "signature": "abc123"},
        "signed_error_record": None,
        "causal_chain": None,
        "metrics": _AUDIT_METRICS,
        "pipeline_completion_map": _FULL_MAP,
        "degraded_confidence": False,
        "error": None,
    }

    handler = make_audit_handler(mock_pipeline, repo)
    handler(_make_reco_event(incident_id))

    resolved = repo.get(incident_id)
    assert resolved is not None
    assert resolved.current_state == "RESOLVED"
    assert resolved.closed_at is not None
    open_ids = {i.incident_id for i in repo.open_incidents()}
    assert incident_id not in open_ids, (
        f"Incident {incident_id} must not appear in open_incidents() after SAFE verdict"
    )


# ---------------------------------------------------------------------------
# Test 9 — FLAGGED verdict does NOT resolve the incident
# ---------------------------------------------------------------------------

def test_incident_not_resolved_on_flagged(tmp_path: Path) -> None:
    from flare.audit.handlers import make_audit_handler

    repo, incident_id = _make_investigating_incident(tmp_path)

    mock_pipeline = MagicMock()
    mock_pipeline.hot_storage_path = str(tmp_path / "hot_storage")
    mock_pipeline.mission_id = MISSION_ID
    mock_pipeline.process_span.return_value = {
        "span_id": str(uuid.uuid4()),
        "status": "FLAGGED",
        "signed_report": {"data": "flagged_report", "signature": "def456"},
        "signed_error_record": None,
        "causal_chain": {"root_cause_chunks": ["doc_001_chunk_000"], "pattern": "out_of_range"},
        "metrics": {**_AUDIT_METRICS, "hallucination_score": 0.75, "overall_safety_score": 0.30},
        "pipeline_completion_map": _FULL_MAP,
        "degraded_confidence": False,
        "error": None,
    }

    handler = make_audit_handler(mock_pipeline, repo)
    handler(_make_reco_event(incident_id))

    still_open = repo.get(incident_id)
    assert still_open is not None
    assert still_open.current_state == "INVESTIGATING", (
        "FLAGGED verdict must not resolve the incident — fault may still be active"
    )
