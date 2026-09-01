# BOUNDARY: flare/ingestion/reader.py → imports ONLY from flare.ingestion
#no log, no event, just frames flowing out one at a time.
from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flare.ingestion.frame import TelemetryFrame, validate_frame

logger = logging.getLogger(__name__)

_APID_MAP: dict[str, int] = {
    "BUS_VOLTAGE": 100,
    "SOLAR_CURRENT": 101,
    "THRUSTER_TEMP": 200,
    "TANK_PRESSURE": 201,
    "REACTION_WHEEL_RPM": 300,
    "TX_FREQUENCY": 400,
}

_UNIT_MAP: dict[str, str] = {
    "BUS_VOLTAGE": "V",
    "SOLAR_CURRENT": "A",
    "THRUSTER_TEMP": "degC",
    "TANK_PRESSURE": "Pa",
    "REACTION_WHEEL_RPM": "rpm",
    "TX_FREQUENCY": "Hz",
}

_SUBSYSTEM_MAP: dict[str, str] = {
    "BUS_VOLTAGE": "EPS",
    "SOLAR_CURRENT": "EPS",
    "THRUSTER_TEMP": "PROP",
    "TANK_PRESSURE": "PROP",
    "REACTION_WHEEL_RPM": "ADCS",
    "TX_FREQUENCY": "COMMS",
}

# Canonical SMAP raw channel name → FLARE channel_id mapping.
# Single source of truth — import this instead of defining a local copy.
_CHANNEL_MAP: dict[str, str] = {
    "P-1": "BUS_VOLTAGE",
    "P-2": "SOLAR_CURRENT",
    "T-1": "THRUSTER_TEMP",
    "T-2": "TANK_PRESSURE",
    "A-1": "REACTION_WHEEL_RPM",
    "S-1": "TX_FREQUENCY",
}


class SMAPReader:
    def __init__(
        self,
        csv_path: str,
        mission_profile: dict[str, Any],
        channel_map: dict[str, str],
    ) -> None:
        self._csv_path = Path(csv_path)
        self._mission_profile = mission_profile
        self._channel_map = channel_map
        self._warned_unknown_channels: set[str] = set()

#actual ingestion, reads the csv row by row, for each row it checks.
#yields one TelemetryFrame at a time, never holds the whole csv in memory, pipeline calls next() on it in a loop.
    def stream(self) -> Generator[TelemetryFrame, None, None]:
        df = pd.read_csv(self._csv_path)
        for row_index, row in df.iterrows():
            try:
                raw_channel = str(row["channel_id"])
                channel_id = self._channel_map.get(raw_channel, raw_channel)
                if channel_id not in _APID_MAP and channel_id not in self._warned_unknown_channels:
                    logger.warning(
                        "Unknown channel_id '%s' — defaulting to "
                        "apid=0, unit=dimensionless, subsystem=EPS. "
                        "Check channel mapping configuration.",
                        channel_id,
                    )
                    self._warned_unknown_channels.add(channel_id)
                apid = _APID_MAP.get(channel_id, 0)
                unit = _UNIT_MAP.get(channel_id, "dimensionless")
                subsystem = _SUBSYSTEM_MAP.get(channel_id, "EPS")

                seq_count = int(row["sequence_number"]) % 16384

                frame = TelemetryFrame(
                    apid=apid,
                    sequence_count=seq_count,
                    sequence_flag="standalone",
                    timestamp=float(row["timestamp"]),
                    channel_id=channel_id,
                    value=float(row["value"]),
                    unit=unit,  # type: ignore[arg-type]
                    subsystem=subsystem,  # type: ignore[arg-type]
                    source_file=str(row.get("source_file", self._csv_path.name)),
                    row_index=int(row_index),
                )
                validate_frame(frame)
                yield frame
            except Exception as exc:
                logger.warning("Skipping malformed row %s: %s", row_index, exc)


#one time preprocessing, reads raw SMAP.npy fiels from NASA, converts them to FLARE's CSV format
#can run before pipeline starts
#Output is telemetry.csv
def preprocess_smap_npy(
    smap_dir: str,
    output_csv: str,
    channel_map: dict[str, str],
) -> None:
    """Convert raw SMAP .npy files (train/ and test/ subdirs) to FLARE CSV format."""
    smap_path = Path(smap_dir)
    output_path = Path(output_csv)

    rows: list[dict[str, Any]] = []
    sequence_counter: dict[str, int] = {}

    for split in ("train", "test"):
        split_dir = smap_path / split
        if not split_dir.exists():
            logger.warning("SMAP split directory not found: %s", split_dir)
            continue

        for npy_file in sorted(split_dir.glob("*.npy")):
            smap_channel = npy_file.stem
            channel_id = channel_map.get(smap_channel, smap_channel)
            data: np.ndarray = np.load(npy_file)

            # SMAP .npy shape: (n_timesteps,) or (n_timesteps, n_features)
            values = data[:, 0] if data.ndim == 2 else data

            for i, val in enumerate(values):
                seq = sequence_counter.get(channel_id, 0) % 16384
                sequence_counter[channel_id] = seq + 1
                rows.append(
                    {
                        "timestamp": float(i),
                        "channel_id": smap_channel,
                        "value": float(val),
                        "unit": _UNIT_MAP.get(channel_id, "dimensionless"),
                        "subsystem": _SUBSYSTEM_MAP.get(channel_id, "EPS"),
                        "sequence_number": seq,
                        "source_file": npy_file.name,
                    }
                )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)
