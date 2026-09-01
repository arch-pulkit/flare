# BOUNDARY: imports from flare.ingestion.frame, flare.ingestion.reader, and stdlib only.
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from flare.ingestion.frame import TelemetryFrame
from flare.ingestion.reader import SMAPReader

logger = logging.getLogger(__name__)

from flare.ingestion.reader import _CHANNEL_MAP


@dataclass(frozen=True)
class InjectionSpec:
    channel_id: str
    injection_type: str      # "point" | "degradation" | "collective"
    start_frame: int         # row_index lower bound where injection begins
    duration_frames: int
    target_value: float      # point: injected value; degradation: final value; collective: sustained value
    nominal_value: float     # baseline value before injection (used for degradation start)


def build_injected_sequence(
    csv_path: Path,
    spec: InjectionSpec,
) -> tuple[list[TelemetryFrame], int]:
    """Load base CSV, inject anomaly per spec, return (modified_frames, injection_frame_index).

    injection_frame_index is the row_index of the first modified TelemetryFrame —
    NOT spec.start_frame, which is a row_index lower bound that may not align to a
    channel frame boundary.
    """
    # 1. Load all frames
    reader = SMAPReader(
        csv_path=str(csv_path),
        mission_profile={},
        channel_map=_CHANNEL_MAP,
    )
    all_frames: list[TelemetryFrame] = list(reader.stream())

    # 2. Filter to target channel
    channel_frames = [f for f in all_frames if f.channel_id == spec.channel_id]
    if not channel_frames:
        logger.warning(
            "build_injected_sequence: channel %r not found in %s",
            spec.channel_id,
            csv_path,
        )
        return all_frames, 0

    # Find first channel frame at row_index >= start_frame (row_index lower bound)
    window_start_idx = 0
    for i, f in enumerate(channel_frames):
        if f.row_index >= spec.start_frame:
            window_start_idx = i
            break

    # 3. Build (original_frame, modified_value) pairs for injection window
    injection_window = channel_frames[window_start_idx : window_start_idx + spec.duration_frames]
    if not injection_window:
        logger.warning(
            "build_injected_sequence: empty injection window for channel %r "
            "start_frame=%d duration=%d",
            spec.channel_id,
            spec.start_frame,
            spec.duration_frames,
        )
        return all_frames, 0

    modified_pairs: list[tuple[TelemetryFrame, float]] = []
    n = len(injection_window)
    for k, frame in enumerate(injection_window):
        if spec.injection_type == "point":
            new_value = spec.target_value
        elif spec.injection_type == "degradation":
            denom = spec.duration_frames - 1 if spec.duration_frames > 1 else 1
            new_value = spec.nominal_value + (
                (spec.target_value - spec.nominal_value) * (k / denom)
            )
        elif spec.injection_type == "collective":
            new_value = spec.target_value
        else:
            raise ValueError(f"Unknown injection_type: {spec.injection_type!r}")
        modified_pairs.append((frame, new_value))

    # 4. Construct replacement TelemetryFrame instances (all fields preserved except value)
    modified_by_row_index: dict[int, TelemetryFrame] = {}
    for original, new_value in modified_pairs:
        modified_by_row_index[original.row_index] = TelemetryFrame(
            apid=original.apid,
            sequence_count=original.sequence_count,
            sequence_flag=original.sequence_flag,
            timestamp=original.timestamp,
            channel_id=original.channel_id,
            value=new_value,
            unit=original.unit,
            subsystem=original.subsystem,
            source_file=original.source_file,
            row_index=original.row_index,
        )

    # 5. Merge modified frames back into full frame list
    result: list[TelemetryFrame] = [
        modified_by_row_index.get(f.row_index, f) for f in all_frames
    ]

    # 6. Sort by row_index; read injection_frame_index from the first modified frame
    result.sort(key=lambda f: f.row_index)
    injection_frame_index = injection_window[0].row_index

    return result, injection_frame_index
