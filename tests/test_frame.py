from __future__ import annotations

import pytest

from flare.ingestion.frame import TelemetryFrame, validate_frame


def _make_frame(**overrides) -> TelemetryFrame:
    defaults = dict(
        apid=100,
        sequence_count=0,
        sequence_flag="standalone",
        timestamp=1000.0,
        channel_id="BUS_VOLTAGE",
        value=30.0,
        unit="V",
        subsystem="EPS",
        source_file="test.csv",
        row_index=0,
    )
    defaults.update(overrides)
    return TelemetryFrame(**defaults)


def test_telemetry_frame_is_frozen():
    frame = _make_frame()
    with pytest.raises((AttributeError, TypeError)):
        frame.value = 99.0  # type: ignore[misc]


def test_validate_frame_apid_bounds():
    frame = _make_frame(apid=2048)
    with pytest.raises(ValueError, match="APID"):
        validate_frame(frame)


def test_validate_frame_sequence_count_bounds():
    frame = _make_frame(sequence_count=16384)
    with pytest.raises(ValueError, match="sequence_count"):
        validate_frame(frame)


def test_frame_equality_by_value():
    f1 = _make_frame()
    f2 = _make_frame()
    assert f1 == f2
    f3 = _make_frame(value=31.0)
    assert f1 != f3
