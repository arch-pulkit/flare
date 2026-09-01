from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flare.audit.events import AuditCompleted
from flare.audit.handlers import make_audit_handler
from flare.llm.events import LogprobToken, RecommendationGenerated, RetrievedChunk

# --- shared helpers ---

_FULL_COMPLETION_MAP = {
    "registry_lookup": "complete",
    "contradiction_detector": "complete",
    "token_confidence": "complete",
    "source_verifier": "complete",
    "groundedness": "complete",
    "criteria_gate": "complete",
    "causal_chain_mapper": "complete",
    "integrity_signer": "complete",
    "storage": "complete",
}

_METRICS = {
    "hallucination_score": 0.05,
    "retrieval_quality": 0.90,
    "faithfulness": 0.95,
    "token_confidence": 0.85,
    "latency_ms": 180.0,
    "cost_usd": 0.0001,
    "overall_safety_score": 0.92,
}


def _make_rec_event() -> RecommendationGenerated:
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
        event_id="evt-llm-001",
        incident_id="inc-001",
        parent_event_id="evt-anomaly-001",
        mission_id="SKYROOT_M1",
        llm_text="ASSESSMENT: Bus voltage is low.",
        logprobs=(LogprobToken(token="ASSESSMENT", logprob=-0.05),),
        retrieved_chunks=(chunk,),
        query="BUS_VOLTAGE anomaly: value 21.5 V, threshold -0.1, subsystem EPS",
        retrieval_method="hybrid",
        top_k=5,
        all_ranked_chunks=(
            {"chunk_id": "doc_001_chunk_000", "rank": 1, "dense_score": 0.91, "sparse_score": 0.75, "rrf_score": 0.031},
        ),
        llm_backend="openai:gpt-4o-mini",
        generated_at=1748000000.0,
    )


def _make_mock_pipeline(tmp_path: Path) -> MagicMock:
    mock = MagicMock()
    mock.hot_storage_path = str(tmp_path)
    mock.mission_id = "TEST_MISSION"
    return mock


def _make_audit_completed(
    *,
    status: str = "SAFE",
    signed_report: dict | None = None,
    signed_error_record: dict | None = None,
    causal_chain: dict | None = None,
    degraded_confidence: bool = False,
    report_path: str = "",
) -> AuditCompleted:
    return AuditCompleted(
        event_id="evt-audit-001",
        incident_id="inc-001",
        parent_event_id="evt-llm-001",
        status=status,
        signed_report=signed_report,
        signed_error_record=signed_error_record,
        causal_chain=causal_chain,
        metrics=_METRICS,
        pipeline_completion_map=_FULL_COMPLETION_MAP,
        degraded_confidence=degraded_confidence,
        report_path=report_path,
        audited_at=1748000001.0,
    )


# --- audit_handler tests ---

def test_audit_handler_safe(tmp_path: Path) -> None:
    mock_pipeline = _make_mock_pipeline(tmp_path)
    mock_pipeline.process_span.return_value = {
        "status": "SAFE",
        "signed_report": {"data": "signed_content", "signature": "abc123"},
        "signed_error_record": None,
        "causal_chain": None,
        "metrics": _METRICS,
        "pipeline_completion_map": _FULL_COMPLETION_MAP,
        "degraded_confidence": False,
        "error": None,
    }

    handler = make_audit_handler(mock_pipeline)
    result = handler(_make_rec_event())

    assert result.status == "SAFE"
    assert result.signed_report is not None
    assert result.causal_chain is None
    assert result.degraded_confidence is False


def test_audit_handler_flagged(tmp_path: Path) -> None:
    causal_chain = {"root_cause_chunks": ["doc_001_chunk_000"], "pattern": "out_of_range"}
    mock_pipeline = _make_mock_pipeline(tmp_path)
    mock_pipeline.process_span.return_value = {
        "status": "FLAGGED",
        "signed_report": {"data": "flagged_report", "signature": "def456"},
        "signed_error_record": None,
        "causal_chain": causal_chain,
        "metrics": _METRICS,
        "pipeline_completion_map": _FULL_COMPLETION_MAP,
        "degraded_confidence": False,
        "error": None,
    }

    handler = make_audit_handler(mock_pipeline)
    result = handler(_make_rec_event())

    assert result.status == "FLAGGED"
    assert result.causal_chain is not None
    assert result.degraded_confidence is False


def test_audit_handler_partial_evaluation(tmp_path: Path) -> None:
    partial_map = {**_FULL_COMPLETION_MAP, "source_verifier": "failed"}
    mock_pipeline = _make_mock_pipeline(tmp_path)
    mock_pipeline.process_span.return_value = {
        "status": "PARTIAL_EVALUATION",
        "signed_report": None,
        "signed_error_record": None,
        "causal_chain": None,
        "metrics": _METRICS,
        "pipeline_completion_map": partial_map,
        "degraded_confidence": True,
        "error": None,
    }

    handler = make_audit_handler(mock_pipeline)
    result = handler(_make_rec_event())

    assert result.status == "PARTIAL_EVALUATION"
    assert result.degraded_confidence is True
    assert result.report_path == ""

    # verify write_report was NOT called — signed_report was None so no file should exist
    reports_dir = tmp_path / "TEST_MISSION" / "reports"
    if reports_dir.exists():
        json_files = list(reports_dir.glob("*.json"))
        assert json_files == [], f"write_report must not be called with signed_report=None; found: {json_files}"


def test_audit_handler_error(tmp_path: Path) -> None:
    mock_pipeline = _make_mock_pipeline(tmp_path)
    mock_pipeline.process_span.return_value = {
        "status": "ERROR",
        "signed_report": None,
        "signed_error_record": {"error_id": "err-001", "reason": "registry_lookup failed"},
        "causal_chain": None,
        "metrics": _METRICS,
        "pipeline_completion_map": {**_FULL_COMPLETION_MAP, "registry_lookup": "failed"},
        "degraded_confidence": True,
        "error": "registry_lookup failed: document not found",
    }

    handler = make_audit_handler(mock_pipeline)
    result = handler(_make_rec_event())

    assert result.status == "ERROR"
    assert result.signed_report is None
    assert result.signed_error_record is not None
    assert result.degraded_confidence is True

    # audit_handler owns the error-record write (storage_handler was removed as dead code)
    error_file = tmp_path / "TEST_MISSION" / "reports" / "errors" / f"{result.event_id}.json"
    assert error_file.exists(), "audit_handler must write signed_error_record to errors/ path"

