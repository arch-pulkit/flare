from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from flare.detection.events import AnomalyDetected, WindowStats
from flare.ingestion.frame import TelemetryFrame
from flare.llm.handlers import make_llm_handler


def _make_anomaly() -> AnomalyDetected:
    frame = TelemetryFrame(
        apid=100,
        sequence_count=0,
        sequence_flag="standalone",
        timestamp=1000.0,
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


def test_llm_handler_backend_failure_returns_degraded_event() -> None:
    """When the LLM backend raises, handler must return a valid degraded event, not propagate."""
    kb = MagicMock()
    kb.retrieve.return_value = ([], [])

    backend = MagicMock()
    backend.generate.side_effect = RuntimeError("API timeout")
    backend.backend_id = "openai:gpt-4o-mini"

    handler = make_llm_handler(
        knowledge_base=kb,
        llm_backend=backend,
        mission_id="TEST_M1",
        top_k=3,
    )
    event = _make_anomaly()
    result = handler(event)

    # Must return a RecommendationGenerated, not raise
    assert result is not None
    assert result.incident_id == event.incident_id
    assert result.logprobs == ()

    parsed = json.loads(result.llm_text)
    assert parsed["ASSESSMENT"] == "LLM_UNAVAILABLE"
    assert parsed["LIKELY_CAUSE"] == "NOT PROVIDED"
    assert parsed["RECOMMENDED_ACTION"] == "NOT PROVIDED"
    assert parsed["URGENCY"] == "NOT PROVIDED"
