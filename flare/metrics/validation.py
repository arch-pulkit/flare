# BOUNDARY: imports from flare.ingestion.frame, flare.detection.events, and stdlib only.
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from flare.ingestion.frame import TelemetryFrame


@dataclass(frozen=True)
class LegacyResult:
    total_frames: int
    alarm_count: int
    alarm_frames: tuple[int, ...]
    alarm_rate: float
    channel_alarm_counts: dict[str, int]


@dataclass(frozen=True)
class FLAREResult:
    total_frames: int
    escalation_count: int
    escalation_frames: tuple[int, ...]
    escalation_rate: float
    channel_escalation_counts: dict[str, int]
    # key: incident_id, value: (detected_at, audited_at)
    incident_timings: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class ValidationMatrix:
    # FPR
    fpr_legacy: float
    fpr_flare: float
    fpr_reduction_ratio: float          # fpr_legacy / fpr_flare; inf when fpr_flare == 0

    # TTD
    ttd_flare_seconds: float
    ttd_legacy_seconds: float
    ttd_advantage_seconds: float        # positive = FLARE detected earlier

    # MTTR
    mttr_flare_seconds: float
    mttr_legacy_seconds: float
    mttr_reduction_seconds: float       # mttr_legacy - mttr_flare

    # Metadata
    total_frames: int
    frame_interval_seconds: float
    mission_id: str
    computed_at: float                  # time.time() at construction

    # Dual-track MTTR legacy baselines (supplementary — primary is mttr_legacy_seconds)
    mttr_legacy_simple_seconds: float = 300.0   # known pattern, no spec lookup — floor
    mttr_legacy_novel_seconds: float = 900.0    # spec retrieval + recommendation — FLARE's case


class LegacyBaseline:
    """Static threshold alarm system — no ML, no context window, no persistence."""

    def __init__(self, nominal_ranges: dict[str, tuple[float, float]]) -> None:
        self._nominal_ranges = nominal_ranges

    def check(self, frame: TelemetryFrame) -> bool:
        """True if frame.value is outside the nominal range for its channel."""
        if frame.channel_id not in self._nominal_ranges:
            return False
        lo, hi = self._nominal_ranges[frame.channel_id]
        return frame.value < lo or frame.value > hi

    def run_on_sequence(self, frames: list[TelemetryFrame]) -> LegacyResult:
        alarm_frames: list[int] = []
        channel_alarm_counts: dict[str, int] = {}
        for frame in frames:
            if self.check(frame):
                alarm_frames.append(frame.row_index)
                channel_alarm_counts[frame.channel_id] = (
                    channel_alarm_counts.get(frame.channel_id, 0) + 1
                )
        total = len(frames)
        alarm_count = len(alarm_frames)
        return LegacyResult(
            total_frames=total,
            alarm_count=alarm_count,
            alarm_frames=tuple(alarm_frames),
            alarm_rate=alarm_count / total if total > 0 else 0.0,
            channel_alarm_counts=channel_alarm_counts,
        )


def compute_fpr(
    legacy: LegacyResult,
    flare: FLAREResult,
) -> tuple[float, float, float]:
    """Returns (fpr_legacy, fpr_flare, fpr_reduction_ratio)."""
    fpr_legacy = legacy.alarm_rate
    fpr_flare = flare.escalation_rate
    ratio = math.inf if fpr_flare == 0.0 else fpr_legacy / fpr_flare
    return fpr_legacy, fpr_flare, ratio


def compute_ttd(
    injection_frame: int,
    flare_escalation_frame: int,
    legacy_alarm_frame: int,
    frame_interval_seconds: float,
) -> tuple[float, float, float]:
    """Returns (ttd_flare_s, ttd_legacy_s, ttd_advantage_s).

    ttd_advantage_s is positive when FLARE detected earlier, negative otherwise.
    """
    ttd_flare = (flare_escalation_frame - injection_frame) * frame_interval_seconds
    ttd_legacy = (legacy_alarm_frame - injection_frame) * frame_interval_seconds
    ttd_advantage = ttd_legacy - ttd_flare
    return ttd_flare, ttd_legacy, ttd_advantage


def compute_mttr(
    flare: FLAREResult,
    mttr_legacy_seconds: float = 900.0,
) -> tuple[float, float, float]:
    """Returns (mttr_flare_s, mttr_legacy_s, mttr_reduction_s).

    mttr_flare_s is 0.0 when no incidents are recorded.

    # Default 900s = novel anomaly with spec retrieval (FLARE's actual automation target).
    # Use 300s for simple/known anomaly lower bound.
    # Source: Hundman et al. KDD 2018; NASA NTRS 20210007857.
    """
    if not flare.incident_timings:
        mttr_flare = 0.0
    else:
        durations = [
            audited_at - detected_at
            for detected_at, audited_at in flare.incident_timings.values()
        ]
        mttr_flare = sum(durations) / len(durations)
    return mttr_flare, mttr_legacy_seconds, mttr_legacy_seconds - mttr_flare


def build_validation_matrix(
    legacy: LegacyResult,
    flare: FLAREResult,
    injection_frame: int,
    flare_escalation_frame: int,
    legacy_alarm_frame: int,
    frame_interval_seconds: float,
    mttr_legacy_seconds: float = 900.0,
    mission_id: str = "SKYROOT_M1",
) -> ValidationMatrix:
    fpr_legacy, fpr_flare, fpr_ratio = compute_fpr(legacy, flare)
    ttd_flare, ttd_legacy, ttd_advantage = compute_ttd(
        injection_frame, flare_escalation_frame, legacy_alarm_frame, frame_interval_seconds
    )
    mttr_flare, mttr_legacy, mttr_reduction = compute_mttr(flare, mttr_legacy_seconds)
    return ValidationMatrix(
        fpr_legacy=fpr_legacy,
        fpr_flare=fpr_flare,
        fpr_reduction_ratio=fpr_ratio,
        ttd_flare_seconds=ttd_flare,
        ttd_legacy_seconds=ttd_legacy,
        ttd_advantage_seconds=ttd_advantage,
        mttr_flare_seconds=mttr_flare,
        mttr_legacy_seconds=mttr_legacy,
        mttr_reduction_seconds=mttr_reduction,
        total_frames=legacy.total_frames,
        frame_interval_seconds=frame_interval_seconds,
        mission_id=mission_id,
        computed_at=time.time(),
    )


def format_matrix_report(matrix: ValidationMatrix) -> str:
    """Investor-readable validation summary for CLI output."""
    width = 51
    sep = "│" + " " * (width - 2) + "│"

    ratio_str = (
        "∞" if math.isinf(matrix.fpr_reduction_ratio)
        else f"{matrix.fpr_reduction_ratio:.1f}×"
    )
    legacy_count = round(matrix.fpr_legacy * matrix.total_frames)
    flare_count = round(matrix.fpr_flare * matrix.total_frames)
    n = matrix.total_frames

    def row(text: str) -> str:
        return f"│ {text:<{width - 4}} │"

    advantage_label = "earlier" if matrix.ttd_advantage_seconds >= 0 else "later (FLARE slower)"

    if matrix.mttr_flare_seconds > 0:
        simple_ratio = f"{matrix.mttr_legacy_simple_seconds / matrix.mttr_flare_seconds:.0f}×"
        novel_ratio = f"{matrix.mttr_legacy_novel_seconds / matrix.mttr_flare_seconds:.0f}×"
    else:
        simple_ratio = "∞"
        novel_ratio = "∞"

    lines = [
        "┌─ FLARE Validation Matrix " + "─" * (width - 27) + "┐",
        row(f"Mission: {matrix.mission_id}"),
        sep,
        row("FALSE POSITIVE RATE"),
        row(f"  Legacy:  {matrix.fpr_legacy * 100:5.1f}%  ({legacy_count} / {n} frames)"),
        row(f"  FLARE:   {matrix.fpr_flare * 100:5.1f}%  ({flare_count} / {n} frames)"),
        row(f"  Reduction: {ratio_str}"),
        sep,
        row("TIME TO DETECT"),
        row(f"  FLARE:   {matrix.ttd_flare_seconds:6.1f} s"),
        row(f"  Legacy:  {matrix.ttd_legacy_seconds:6.1f} s"),
        row(f"  Advantage: {abs(matrix.ttd_advantage_seconds):.1f} s {advantage_label}"),
        sep,
        row("MEAN TIME TO RESOLUTION (MTTR)"),
        row(f"  FLARE:          {matrix.mttr_flare_seconds:5.1f}s  (end-to-end pipeline)"),
        row(f"  Legacy simple:  {matrix.mttr_legacy_simple_seconds:.0f}s  (known pattern,"),
        row(f"                           no spec lookup)"),
        row(f"  Legacy novel:   {matrix.mttr_legacy_novel_seconds:.0f}s  (spec retrieval +"),
        row(f"                           recommendation)"),
        row(f"  Reduction:  {simple_ratio}  (simple) / {novel_ratio} (novel — FLARE's case)"),
        row(f"  Source: Hundman et al. KDD 2018;"),
        row(f"  NASA NTRS 20210007857"),
        "└" + "─" * (width - 2) + "┘",
    ]
    return "\n".join(lines)
