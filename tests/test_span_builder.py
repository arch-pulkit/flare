from __future__ import annotations

import pytest

from flare.audit.span_builder import build_span
from flare.llm.events import LogprobToken, RecommendationGenerated, RetrievedChunk

# --- fixture ---

@pytest.fixture()
def sample_event() -> RecommendationGenerated:
    chunks = (
        RetrievedChunk(
            chunk_id="doc_001_chunk_000",
            source="voltage_spec.txt",
            text="BUS_VOLTAGE nominal range: 28–32 V.",
            rank=1,
            dense_score=0.91,
            sparse_score=0.75,
            rrf_score=0.031,
            source_tier="CRITICAL",
        ),
        RetrievedChunk(
            chunk_id="doc_001_chunk_001",
            source="voltage_spec.txt",
            text="Undervoltage fault threshold: 22 V.",
            rank=2,
            dense_score=0.80,
            sparse_score=0.60,
            rrf_score=0.028,
            source_tier="CRITICAL",
        ),
    )
    all_ranked = (
        {"chunk_id": "doc_001_chunk_000", "rank": 1, "dense_score": 0.91, "sparse_score": 0.75, "rrf_score": 0.031},
        {"chunk_id": "doc_001_chunk_001", "rank": 2, "dense_score": 0.80, "sparse_score": 0.60, "rrf_score": 0.028},
        {"chunk_id": "doc_002_chunk_000", "rank": 3, "dense_score": 0.50, "sparse_score": 0.30, "rrf_score": 0.015},
    )
    return RecommendationGenerated(
        event_id="evt-llm-001",
        incident_id="inc-001",
        parent_event_id="evt-anomaly-001",
        mission_id="SKYROOT_M1",
        llm_text="ASSESSMENT: Bus voltage is critically low...",
        logprobs=(
            LogprobToken(token="ASSESSMENT", logprob=-0.05),
            LogprobToken(token=":", logprob=-0.01),
        ),
        retrieved_chunks=chunks,
        query="BUS_VOLTAGE anomaly: value 21.5 V, threshold -0.1, subsystem EPS",
        retrieval_method="hybrid",
        top_k=2,
        all_ranked_chunks=all_ranked,
        llm_backend="openai:gpt-4o-mini",
        generated_at=1748000000.0,
    )


# --- tests ---

def test_span_shape_complete(sample_event: RecommendationGenerated) -> None:
    span = build_span(sample_event)
    for key in ("span_id", "trace_id", "retrieved_chunks", "retrieval_metadata", "llm_response"):
        assert key in span, f"missing top-level key: {key!r}"


def test_span_id_equals_event_id(sample_event: RecommendationGenerated) -> None:
    span = build_span(sample_event)
    assert span["span_id"] == sample_event.event_id


def test_trace_id_equals_incident_id(sample_event: RecommendationGenerated) -> None:
    span = build_span(sample_event)
    assert span["trace_id"] == sample_event.incident_id


def test_retrieved_chunks_flat_list_present(sample_event: RecommendationGenerated) -> None:
    span = build_span(sample_event)
    flat = span["retrieval_metadata"]["retrieved_chunks"]
    assert isinstance(flat, list), "flat list must be a list"
    assert all(isinstance(item, str) for item in flat), "flat list must contain strings (chunk_ids)"
    assert flat == [c.chunk_id for c in sample_event.retrieved_chunks]


def test_all_ranked_chunks_complete(sample_event: RecommendationGenerated) -> None:
    span = build_span(sample_event)
    assert len(span["retrieval_metadata"]["all_ranked_chunks"]) == len(sample_event.all_ranked_chunks)


def test_logprobs_shape(sample_event: RecommendationGenerated) -> None:
    span = build_span(sample_event)
    logprobs = span["llm_response"]["logprobs"]
    assert isinstance(logprobs, list)
    for entry in logprobs:
        assert "token" in entry and "logprob" in entry


def test_span_builder_is_pure(sample_event: RecommendationGenerated, monkeypatch: pytest.MonkeyPatch) -> None:
    write_calls: list[str] = []

    from pathlib import Path
    monkeypatch.setattr(Path, "write_text", lambda self, *a, **kw: write_calls.append(str(self)))

    span1 = build_span(sample_event)
    span2 = build_span(sample_event)

    assert span1 == span2, "build_span must be deterministic"
    assert write_calls == [], "build_span must not perform any file I/O"
