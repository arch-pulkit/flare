from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from flare.detection.events import AnomalyDetected, WindowStats
from flare.ingestion.frame import TelemetryFrame
from flare.llm.events import RetrievedChunk
from flare.llm.prompt import build_query
from flare.llm.retrieval import KnowledgeBase

DOCS_FOLDER = Path("data/knowledge_base")
REGISTRY_CONFIG = Path("config/registry_config.json")


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase(DOCS_FOLDER, REGISTRY_CONFIG)


def _make_anomaly(
    channel_id: str = "BUS_VOLTAGE",
    value: float = 31.2,
    unit: str = "V",
    subsystem: str = "EPS",
) -> AnomalyDetected:
    frame = TelemetryFrame(
        apid=100,
        sequence_count=0,
        sequence_flag="standalone",
        timestamp=1000.0,
        channel_id=channel_id,
        value=value,
        unit=unit,  # type: ignore[arg-type]
        subsystem=subsystem,  # type: ignore[arg-type]
        source_file="test.csv",
        row_index=0,
    )
    stats = WindowStats(
        mean=30.0, std=1.0, min=28.0, max=32.0,
        nominal_low=27.5, nominal_high=32.5, window_size=10,
    )
    return AnomalyDetected(
        event_id=str(uuid.uuid4()),
        incident_id=str(uuid.uuid4()),
        mission_id="TEST_M1",
        frame=frame,
        anomaly_score=-0.3,
        decision_threshold=-0.1,
        is_anomaly=True,
        context_window=(frame,),
        window_stats=stats,
        detector_version="abc123",
        detection_method="isolation_forest",
        detected_at=1000.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_knowledge_base_loads(kb: KnowledgeBase) -> None:
    assert len(kb._chunks) > 0


def test_retrieve_returns_top_k(kb: KnowledgeBase) -> None:
    chunks, _ = kb.retrieve("bus voltage anomaly EPS", top_k=3)
    assert len(chunks) == 3
    for c in chunks:
        assert isinstance(c, RetrievedChunk)


def test_rrf_scores_positive(kb: KnowledgeBase) -> None:
    chunks, _ = kb.retrieve("voltage anomaly", top_k=5)
    for c in chunks:
        assert c.rrf_score > 0
        assert c.dense_score >= 0
        assert c.sparse_score >= 0


def test_source_tier_populated(kb: KnowledgeBase) -> None:
    chunks, _ = kb.retrieve("pressure tank anomaly", top_k=5)
    valid_tiers = {"CRITICAL", "SUPPORTING", "REFERENCE"}
    for c in chunks:
        assert c.source_tier in valid_tiers


def test_chunk_source_is_filename_only(kb: KnowledgeBase) -> None:
    chunks, _ = kb.retrieve("thruster temperature limit", top_k=5)
    for c in chunks:
        # Must be a bare filename — no path separators
        assert "/" not in c.source
        assert "\\" not in c.source
        assert c.source.endswith(".txt")


def test_all_ranked_chunks_complete(kb: KnowledgeBase) -> None:
    top_k = 3
    chunks, all_ranked = kb.retrieve("voltage current EPS", top_k=top_k)
    assert len(all_ranked) >= top_k
    assert len(all_ranked) == len(kb._chunks)
    required_keys = {"chunk_id", "rank", "dense_score", "sparse_score", "rrf_score"}
    for entry in all_ranked:
        assert required_keys == set(entry.keys())


def test_build_query_format(kb: KnowledgeBase) -> None:
    anomaly = _make_anomaly(channel_id="BUS_VOLTAGE", value=31.2, unit="V", subsystem="EPS")
    query = build_query(anomaly)
    assert "BUS_VOLTAGE" in query
    assert "31.2" in query
    assert "V" in query
    assert "EPS" in query


def test_voltage_query_retrieves_voltage_spec(kb: KnowledgeBase) -> None:
    query = "BUS_VOLTAGE anomaly: value 31.2 V, threshold -0.1, subsystem EPS"
    chunks, _ = kb.retrieve(query, top_k=3)
    sources = [c.source for c in chunks]
    assert "voltage_spec.txt" in sources, (
        f"voltage_spec.txt not in top-3 results. Got: {sources}\n"
        "Fix: voltage_spec.txt content may be too sparse or chunked poorly."
    )
