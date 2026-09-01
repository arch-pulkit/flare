from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from flare.detection.events import AnomalyDetected, WindowStats
from flare.ingestion.frame import TelemetryFrame
from flare.state.handlers import make_state_machine_handler
from flare.state.incident import (
    Incident,
    IncidentRepository,
    InvalidTransitionError,
    StateTransitionEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_incident(state: str = "DETECTED") -> Incident:
    inc = Incident(
        incident_id=str(uuid.uuid4()),
        mission_id="TEST_M1",
        channel_id="BUS_VOLTAGE",
        subsystem="EPS",
        current_state=state,
        opened_at=1000.0,
    )
    return inc


def _make_anomaly_event(incident_id: str | None = None, detected_at: float = 1000.0) -> AnomalyDetected:
    frame = TelemetryFrame(
        apid=100,
        sequence_count=0,
        sequence_flag="standalone",
        timestamp=detected_at,
        channel_id="BUS_VOLTAGE",
        value=45.0,
        unit="V",
        subsystem="EPS",
        source_file="test.csv",
        row_index=0,
    )
    stats = WindowStats(
        mean=30.0, std=1.0, min=28.0, max=32.0,
        nominal_low=27.5, nominal_high=32.5, window_size=10,
    )
    return AnomalyDetected(
        event_id=str(uuid.uuid4()),
        incident_id=incident_id or str(uuid.uuid4()),
        mission_id="TEST_M1",
        frame=frame,
        anomaly_score=-0.3,
        decision_threshold=-0.1,
        is_anomaly=True,
        context_window=(frame,),
        window_stats=stats,
        detector_version="abc123",
        detection_method="isolation_forest",
        detected_at=detected_at,
    )


# ---------------------------------------------------------------------------
# Incident entity tests
# ---------------------------------------------------------------------------

def test_valid_transition():
    inc = _make_incident("DETECTED")
    te = inc.transition_to("INVESTIGATING", trigger="test", evidence={"k": "v"})

    assert inc.current_state == "INVESTIGATING"
    assert isinstance(te, StateTransitionEvent)
    assert te.from_state == "DETECTED"
    assert te.to_state == "INVESTIGATING"
    assert te.trigger == "test"
    assert te.transition_id  # non-empty uuid
    assert len(inc.transitions) == 1
    assert inc.transitions[0] is te


def test_invalid_transition():
    inc = _make_incident("DETECTED")
    with pytest.raises(InvalidTransitionError):
        inc.transition_to("MITIGATING", trigger="test", evidence={})


def test_terminal_state():
    # MITIGATING removed from transition table (2026-06-27). Use INVESTIGATING
    # as the pre-RESOLVED state; intent is unchanged: verify RESOLVED is terminal.
    inc = _make_incident("INVESTIGATING")
    inc.transition_to("RESOLVED", trigger="manual", evidence={})
    assert inc.is_resolved()
    with pytest.raises(InvalidTransitionError):
        inc.transition_to("INVESTIGATING", trigger="retry", evidence={})


# ---------------------------------------------------------------------------
# Repository tests
# ---------------------------------------------------------------------------

def test_repository_idempotent_open(tmp_path: Path):
    repo = IncidentRepository(tmp_path / "incidents.jsonl")
    iid = str(uuid.uuid4())
    inc1 = repo.open(iid, "TEST_M1", "BUS_VOLTAGE", "EPS", opened_at=1000.0)
    inc2 = repo.open(iid, "TEST_M1", "BUS_VOLTAGE", "EPS", opened_at=1000.0)
    assert inc1 is inc2
    assert len([i for i in repo._incidents.values()]) == 1


def test_jsonl_persistence(tmp_path: Path):
    log_path = tmp_path / "incidents.jsonl"
    repo = IncidentRepository(log_path)
    iid = str(uuid.uuid4())
    inc = repo.open(iid, "TEST_M1", "BUS_VOLTAGE", "EPS", opened_at=1000.0)
    te = inc.transition_to("INVESTIGATING", trigger="test", evidence={"score": -0.3})
    repo.save(inc)

    # Reconstruct from log
    repo2 = IncidentRepository(log_path)
    repo2.load_from_log(log_path)
    replayed = repo2.get(iid)

    assert replayed is not None
    assert replayed.current_state == "INVESTIGATING"
    assert len(replayed.transitions) == 1
    assert replayed.transitions[0].transition_id == te.transition_id
    assert replayed.transitions[0].from_state == "DETECTED"
    assert replayed.transitions[0].to_state == "INVESTIGATING"
    assert replayed.opened_at == 1000.0


def test_replay_full_lifecycle(tmp_path: Path):
    log_path = tmp_path / "incidents.jsonl"
    repo = IncidentRepository(log_path)

    detected_at = 1234567890.0
    iid = str(uuid.uuid4())
    inc = repo.open(iid, "TEST_M1", "BUS_VOLTAGE", "EPS", opened_at=detected_at)

    te1 = inc.transition_to("INVESTIGATING", trigger="anomaly_detected", evidence={"score": -0.3})
    repo.save(inc)
    te2 = inc.transition_to("RESOLVED", trigger="manual_close", evidence={"operator": "test"})
    repo.save(inc)

    assert inc.is_resolved()
    assert inc.closed_at is not None

    # Replay on fresh repository
    repo2 = IncidentRepository(log_path)
    repo2.load_from_log(log_path)
    replayed = repo2.get(iid)

    assert replayed is not None
    assert replayed.current_state == "RESOLVED"
    assert len(replayed.transitions) == 2
    assert replayed.closed_at is not None
    # opened_at must equal the detected_at from the original event (audit chain)
    assert replayed.opened_at == detected_at
    # transition_ids preserved
    assert replayed.transitions[0].transition_id == te1.transition_id
    assert replayed.transitions[1].transition_id == te2.transition_id


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------

def test_handler_creates_incident(tmp_path: Path):
    repo = IncidentRepository(tmp_path / "incidents.jsonl")
    handler = make_state_machine_handler(repo)

    detected_at = 9999.0
    event = _make_anomaly_event(detected_at=detected_at)
    returned = handler(event)

    assert returned is event  # same object returned unchanged
    incident = repo.get(event.incident_id)
    assert incident is not None
    assert incident.current_state == "INVESTIGATING"
    assert incident.opened_at == detected_at  # opened_at == event.detected_at


def test_load_from_log_skips_corrupted_line(tmp_path: Path):
    """Corrupted JSONL lines must be skipped; valid surrounding incidents must be loaded."""
    import json as _json

    log_path = tmp_path / "incidents.jsonl"

    # Build two valid incidents manually so we control the exact JSON written
    def _valid_line(incident_id: str, mission_id: str = "TEST_M1") -> str:
        return _json.dumps({
            "transition_id": str(uuid.uuid4()),
            "incident_id": incident_id,
            "mission_id": mission_id,
            "channel_id": "BUS_VOLTAGE",
            "subsystem": "EPS",
            "opened_at": 1000.0,
            "from_state": "DETECTED",
            "to_state": "INVESTIGATING",
            "trigger": "anomaly_detected",
            "evidence": {},
            "transitioned_at": 1001.0,
        })

    id_a = str(uuid.uuid4())
    id_b = str(uuid.uuid4())
    lines = [
        _valid_line(id_a),
        "THIS IS NOT JSON {{{",          # corrupted
        _valid_line(id_b),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    repo = IncidentRepository(log_path)
    repo.load_from_log(log_path)  # must not raise

    inc_a = repo.get(id_a)
    inc_b = repo.get(id_b)
    assert inc_a is not None, "incident A (before corrupted line) must be loaded"
    assert inc_b is not None, "incident B (after corrupted line) must be loaded"
    assert inc_a.current_state == "INVESTIGATING"
    assert inc_b.current_state == "INVESTIGATING"


def test_handler_idempotent_on_resolved(tmp_path: Path):
    log_path = tmp_path / "incidents.jsonl"
    repo = IncidentRepository(log_path)
    iid = str(uuid.uuid4())

    # Manually advance to RESOLVED (MITIGATING removed 2026-06-27)
    inc = repo.open(iid, "TEST_M1", "BUS_VOLTAGE", "EPS", opened_at=1000.0)
    inc.transition_to("INVESTIGATING", trigger="setup", evidence={})
    repo.save(inc)
    inc.transition_to("RESOLVED", trigger="setup", evidence={})
    repo.save(inc)

    transition_count_before = len(inc.transitions)

    handler = make_state_machine_handler(repo)
    event = _make_anomaly_event(incident_id=iid)
    returned = handler(event)

    assert returned is event
    # No new transition was added
    assert len(inc.transitions) == transition_count_before
