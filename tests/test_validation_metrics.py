from __future__ import annotations

import math
import time

import pytest

from flare.ingestion.frame import TelemetryFrame
from flare.metrics.validation import (
    FLAREResult,
    LegacyBaseline,
    LegacyResult,
    ValidationMatrix,
    build_validation_matrix,
    compute_fpr,
    compute_mttr,
    compute_ttd,
    format_matrix_report,
)


def _frame(
    channel_id: str,
    value: float,
    row_index: int = 0,
    subsystem: str = "EPS",
    unit: str = "V",
    apid: int = 100,
) -> TelemetryFrame:
    return TelemetryFrame(
        apid=apid,
        sequence_count=0,
        sequence_flag="standalone",
        timestamp=0.0,
        channel_id=channel_id,
        value=value,
        unit=unit,  # type: ignore[arg-type]
        subsystem=subsystem,  # type: ignore[arg-type]
        source_file="test",
        row_index=row_index,
    )


def _minimal_legacy(alarm_rate: float, total: int = 1000) -> LegacyResult:
    alarm_count = round(alarm_rate * total)
    return LegacyResult(
        total_frames=total,
        alarm_count=alarm_count,
        alarm_frames=tuple(range(alarm_count)),
        alarm_rate=alarm_rate,
        channel_alarm_counts={},
    )


def _minimal_flare(escalation_rate: float, total: int = 1000) -> FLAREResult:
    esc_count = round(escalation_rate * total)
    return FLAREResult(
        total_frames=total,
        escalation_count=esc_count,
        escalation_frames=tuple(range(esc_count)),
        escalation_rate=escalation_rate,
        channel_escalation_counts={},
        incident_timings={},
    )


# ── 1 ──────────────────────────────────────────────────────────────────────────


def test_legacy_baseline_alarms_outside_range() -> None:
    baseline = LegacyBaseline({"BUS_VOLTAGE": (22.0, 34.0)})
    assert baseline.check(_frame("BUS_VOLTAGE", 35.0)) is True
    assert baseline.check(_frame("BUS_VOLTAGE", 28.0)) is False


# ── 2 ──────────────────────────────────────────────────────────────────────────


def test_legacy_baseline_unknown_channel() -> None:
    baseline = LegacyBaseline({"BUS_VOLTAGE": (22.0, 34.0)})
    assert baseline.check(_frame("TX_FREQUENCY", 437_050_000.0)) is False


# ── 3 ──────────────────────────────────────────────────────────────────────────


def test_legacy_run_on_sequence() -> None:
    baseline = LegacyBaseline({"BUS_VOLTAGE": (22.0, 34.0)})
    frames = [_frame("BUS_VOLTAGE", 28.0, i) for i in range(8)] + [
        _frame("BUS_VOLTAGE", 35.0, 8),
        _frame("BUS_VOLTAGE", 21.0, 9),
    ]
    result = baseline.run_on_sequence(frames)
    assert result.total_frames == 10
    assert result.alarm_count == 2
    assert result.alarm_rate == pytest.approx(0.2)
    assert 8 in result.alarm_frames
    assert 9 in result.alarm_frames


# ── 4 ──────────────────────────────────────────────────────────────────────────


def test_compute_fpr_ratio() -> None:
    legacy = _minimal_legacy(0.18)
    flare = _minimal_flare(0.015)
    _, _, ratio = compute_fpr(legacy, flare)
    assert ratio == pytest.approx(12.0, abs=0.01)


# ── 5 ──────────────────────────────────────────────────────────────────────────


def test_compute_fpr_zero_flare_escalations() -> None:
    legacy = _minimal_legacy(0.18)
    flare = _minimal_flare(0.0)
    _, _, ratio = compute_fpr(legacy, flare)
    assert math.isinf(ratio)


# ── 6 ──────────────────────────────────────────────────────────────────────────


def test_compute_ttd_flare_earlier() -> None:
    ttd_flare, ttd_legacy, ttd_advantage = compute_ttd(
        injection_frame=100,
        flare_escalation_frame=120,
        legacy_alarm_frame=475,
        frame_interval_seconds=0.1,
    )
    assert ttd_flare == pytest.approx(2.0)
    assert ttd_legacy == pytest.approx(37.5)
    assert ttd_advantage == pytest.approx(35.5)


# ── 7 ──────────────────────────────────────────────────────────────────────────


def test_compute_ttd_flare_later() -> None:
    _, _, ttd_advantage = compute_ttd(
        injection_frame=100,
        flare_escalation_frame=200,
        legacy_alarm_frame=150,
        frame_interval_seconds=0.1,
    )
    assert ttd_advantage < 0


# ── 8 ──────────────────────────────────────────────────────────────────────────


def test_compute_mttr() -> None:
    flare = FLAREResult(
        total_frames=100,
        escalation_count=2,
        escalation_frames=(0, 1),
        escalation_rate=0.02,
        channel_escalation_counts={},
        incident_timings={
            "inc_1": (0.0, 8.0),
            "inc_2": (0.0, 9.0),
        },
    )
    mttr_flare, mttr_legacy, mttr_reduction = compute_mttr(flare, mttr_legacy_seconds=300.0)
    assert mttr_flare == pytest.approx(8.5)
    assert mttr_reduction == pytest.approx(291.5)


# ── 9 ──────────────────────────────────────────────────────────────────────────


def test_compute_mttr_empty_incidents() -> None:
    flare = _minimal_flare(0.0)
    mttr_flare, _, _ = compute_mttr(flare)
    assert mttr_flare == 0.0


# ── 10 ─────────────────────────────────────────────────────────────────────────


def test_build_validation_matrix() -> None:
    legacy = _minimal_legacy(0.18, total=600)
    flare = FLAREResult(
        total_frames=600,
        escalation_count=9,
        escalation_frames=tuple(range(9)),
        escalation_rate=0.015,
        channel_escalation_counts={},
        incident_timings={"inc_1": (0.0, 8.3)},
    )
    before = time.time()
    matrix = build_validation_matrix(
        legacy=legacy,
        flare=flare,
        injection_frame=100,
        flare_escalation_frame=120,
        legacy_alarm_frame=475,
        frame_interval_seconds=0.1,
        mttr_legacy_seconds=300.0,
        mission_id="SKYROOT_M1",
    )
    after = time.time()

    assert matrix.fpr_legacy == pytest.approx(0.18)
    assert matrix.fpr_flare == pytest.approx(0.015)
    assert matrix.fpr_reduction_ratio == pytest.approx(12.0, abs=0.01)
    assert matrix.ttd_flare_seconds == pytest.approx(2.0)
    assert matrix.ttd_legacy_seconds == pytest.approx(37.5)
    assert matrix.ttd_advantage_seconds == pytest.approx(35.5)
    assert matrix.mttr_flare_seconds == pytest.approx(8.3)
    assert matrix.mttr_legacy_seconds == pytest.approx(300.0)
    assert matrix.mttr_reduction_seconds == pytest.approx(291.7)
    assert matrix.total_frames == 600
    assert matrix.mission_id == "SKYROOT_M1"
    assert before <= matrix.computed_at <= after


# ── 11 ─────────────────────────────────────────────────────────────────────────


def test_format_matrix_report_contains_required_fields() -> None:
    legacy = _minimal_legacy(0.183, total=600)
    flare = FLAREResult(
        total_frames=600,
        escalation_count=9,
        escalation_frames=tuple(range(9)),
        escalation_rate=0.015,
        channel_escalation_counts={},
        incident_timings={"inc_1": (0.0, 8.3)},
    )
    matrix = build_validation_matrix(
        legacy=legacy,
        flare=flare,
        injection_frame=100,
        flare_escalation_frame=120,
        legacy_alarm_frame=475,
        frame_interval_seconds=0.1,
    )
    report = format_matrix_report(matrix)

    assert "Legacy" in report
    assert "FLARE" in report
    assert "Reduction" in report or "reduction" in report
    assert "Time to Detect" in report or "TIME TO DETECT" in report or "TTD" in report
    assert "Resolution" in report or "MTTR" in report
