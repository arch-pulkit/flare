from __future__ import annotations

from pathlib import Path

import pytest

# Skip all tests if telemetry CSV is not present
_CSV = Path("data/telemetry.csv")
pytestmark = pytest.mark.skipif(
    not _CSV.exists(),
    reason="data/telemetry.csv not found",
)

import yaml

from scripts.inject_anomalies import InjectionSpec, build_injected_sequence
from scripts.simulation_harness import SCENARIO_MTTR, SCENARIO_TTD_INVISIBLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tp_spec(
    injection_type: str = "collective",
    start_frame: int = 0,
    duration_frames: int = 5,
    target_value: float = 140000.0,
    nominal_value: float = 250000.0,
) -> InjectionSpec:
    return InjectionSpec(
        channel_id="TANK_PRESSURE",
        injection_type=injection_type,
        start_frame=start_frame,
        duration_frames=duration_frames,
        target_value=target_value,
        nominal_value=nominal_value,
    )


# ---------------------------------------------------------------------------
# 1 — point injection modifies exactly one frame
# ---------------------------------------------------------------------------

def test_point_injection_single_frame() -> None:
    spec = InjectionSpec(
        channel_id="TANK_PRESSURE",
        injection_type="point",
        start_frame=50,
        duration_frames=1,
        target_value=99999.0,
        nominal_value=250000.0,
    )
    frames, injection_frame_index = build_injected_sequence(_CSV, spec)

    injected = [f for f in frames if abs(f.value - 99999.0) < 0.01]
    assert len(injected) == 1, f"Expected 1 injected frame, got {len(injected)}"
    assert injected[0].row_index == injection_frame_index


# ---------------------------------------------------------------------------
# 2 — degradation produces linear drift
# ---------------------------------------------------------------------------

def test_degradation_injection_linear() -> None:
    spec = InjectionSpec(
        channel_id="TANK_PRESSURE",
        injection_type="degradation",
        start_frame=100,
        duration_frames=10,
        target_value=130000.0,
        nominal_value=250000.0,
    )
    frames, injection_frame_index = build_injected_sequence(_CSV, spec)

    # Collect injected TANK_PRESSURE frames by row_index proximity to injection start
    tp_frames = sorted(
        [f for f in frames if f.channel_id == "TANK_PRESSURE"],
        key=lambda f: f.row_index,
    )
    # Find injection window within channel frames
    start_idx = next(i for i, f in enumerate(tp_frames) if f.row_index == injection_frame_index)
    window = tp_frames[start_idx : start_idx + 10]

    assert len(window) == 10
    # First ≈ nominal_value (k=0 → value = nominal + 0 = nominal)
    assert abs(window[0].value - 250000.0) < 1.0
    # Last ≈ target_value (k=9 → value = nominal + (target-nominal) * 9/9 = target)
    assert abs(window[-1].value - 130000.0) < 1.0
    # Intermediate values strictly decreasing (nominal → target, target < nominal)
    values = [f.value for f in window]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))


# ---------------------------------------------------------------------------
# 3 — collective injection sets all frames to target_value
# ---------------------------------------------------------------------------

def test_collective_injection_uniform() -> None:
    spec = _tp_spec(injection_type="collective", start_frame=0, duration_frames=20,
                    target_value=140000.0)
    frames, injection_frame_index = build_injected_sequence(_CSV, spec)

    tp_frames = sorted(
        [f for f in frames if f.channel_id == "TANK_PRESSURE"],
        key=lambda f: f.row_index,
    )
    start_idx = next(i for i, f in enumerate(tp_frames) if f.row_index == injection_frame_index)
    window = tp_frames[start_idx : start_idx + 20]

    assert len(window) == 20
    assert all(abs(f.value - 140000.0) < 0.01 for f in window)

    # Frames outside window are unchanged
    outside = [f for f in tp_frames if f.row_index not in {w.row_index for w in window}]
    assert all(abs(f.value - 140000.0) > 0.01 or f.value != 140000.0 for f in outside), (
        "Frames outside injection window should not be 140000.0"
    )


# ---------------------------------------------------------------------------
# 4 — sequence length unchanged
# ---------------------------------------------------------------------------

def test_sequence_length_unchanged() -> None:
    # Load original count
    from scripts.inject_anomalies import _CHANNEL_MAP
    from flare.ingestion.reader import SMAPReader
    reader = SMAPReader(csv_path=str(_CSV), mission_profile={}, channel_map=_CHANNEL_MAP)
    original_frames = list(reader.stream())
    original_total = len(original_frames)

    for injection_type in ("point", "degradation", "collective"):
        spec = _tp_spec(injection_type=injection_type, duration_frames=5)
        injected_frames, _ = build_injected_sequence(_CSV, spec)
        assert len(injected_frames) == original_total, (
            f"Length changed for {injection_type}: "
            f"{len(injected_frames)} vs {original_total}"
        )


# ---------------------------------------------------------------------------
# 5 — injection_frame_index is row_index, not spec.start_frame
# ---------------------------------------------------------------------------

def test_injection_frame_index_is_row_index() -> None:
    # start_frame=50 is well below the T-2 block (rows 300-399 in the 6-channel CSV)
    spec = _tp_spec(start_frame=50, duration_frames=3)
    frames, injection_frame_index = build_injected_sequence(_CSV, spec)

    # Find the first modified frame (TANK_PRESSURE frame with value == target)
    modified = [
        f for f in frames
        if f.channel_id == "TANK_PRESSURE" and abs(f.value - 140000.0) < 0.01
    ]
    assert modified, "No modified TANK_PRESSURE frames found"

    first_modified = min(modified, key=lambda f: f.row_index)
    # injection_frame_index must equal the row_index of the first modified frame
    assert injection_frame_index == first_modified.row_index
    # And it must NOT equal spec.start_frame (which is 50; T-2 starts at row_index 300)
    assert injection_frame_index != spec.start_frame


# ---------------------------------------------------------------------------
# 6 — non-injected channels are completely unchanged
# ---------------------------------------------------------------------------

def test_non_injected_channels_unchanged() -> None:
    from scripts.inject_anomalies import _CHANNEL_MAP
    from flare.ingestion.reader import SMAPReader

    reader = SMAPReader(csv_path=str(_CSV), mission_profile={}, channel_map=_CHANNEL_MAP)
    original = {f.row_index: f for f in reader.stream()}

    spec = _tp_spec(duration_frames=30)
    injected_frames, _ = build_injected_sequence(_CSV, spec)

    bv_injected = {f.row_index: f for f in injected_frames if f.channel_id == "BUS_VOLTAGE"}
    bv_original = {ri: f for ri, f in original.items() if f.channel_id == "BUS_VOLTAGE"}

    assert set(bv_injected.keys()) == set(bv_original.keys()), "BUS_VOLTAGE row_index set changed"
    for ri in bv_original:
        assert bv_injected[ri].value == bv_original[ri].value, (
            f"BUS_VOLTAGE value changed at row_index {ri}"
        )


# ---------------------------------------------------------------------------
# 7 — SCENARIO_MTTR clears the persistence check
# ---------------------------------------------------------------------------

def test_mttr_scenario_clears_persistence_check() -> None:
    from flare.pipeline import MIN_CONSECUTIVE_FRAMES

    assert SCENARIO_MTTR.duration_frames >= MIN_CONSECUTIVE_FRAMES, (
        f"SCENARIO_MTTR duration_frames={SCENARIO_MTTR.duration_frames} "
        f"< MIN_CONSECUTIVE_FRAMES={MIN_CONSECUTIVE_FRAMES}; "
        "persistence check will suppress the anomaly and ASTR-O will not run"
    )

    frames, injection_frame_index = build_injected_sequence(_CSV, SCENARIO_MTTR)
    injected_tp = [
        f for f in frames
        if f.channel_id == "TANK_PRESSURE"
        and f.row_index >= injection_frame_index
        and abs(f.value - SCENARIO_MTTR.target_value) < 0.01
    ]
    assert len(injected_tp) == SCENARIO_MTTR.duration_frames, (
        f"Expected {SCENARIO_MTTR.duration_frames} injected T-2 frames, "
        f"got {len(injected_tp)}"
    )


# ---------------------------------------------------------------------------
# 8 — SCENARIO_TTD_INVISIBLE target_value is inside nominal range
# ---------------------------------------------------------------------------

def test_invisible_scenario_config() -> None:
    """target_value must be inside nominal range — it is an invisible anomaly,
    not a threshold-crossing one.  If this fails the scenario proves nothing."""
    profile = yaml.safe_load(
        Path("config/mission_profiles.yaml").read_text(encoding="utf-8")
    )["SKYROOT_M1"]
    rng = profile["nominal_ranges"][SCENARIO_TTD_INVISIBLE.channel_id]
    lo, hi = float(rng[0]), float(rng[1])
    assert lo < SCENARIO_TTD_INVISIBLE.target_value < hi, (
        f"SCENARIO_TTD_INVISIBLE.target_value={SCENARIO_TTD_INVISIBLE.target_value} "
        f"must be inside nominal range ({lo}, {hi}) — "
        "it is an invisible anomaly, not a threshold-crossing one"
    )
