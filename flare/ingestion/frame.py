# BOUNDARY: flare/ingestion/frame.py → imports NOTHING from flare.*
#No log, no event, just a frozen container.

#innermostlayer, zero dependencies, everything flows outward from this
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#this is a container, does nothing; holds one telemetry reading, sealed forever. 
@dataclass(frozen=True)
class TelemetryFrame:
    # NOTE: computed from _APID_MAP at read time; not consumed by detection, retrieval, LLM,
    # span building, or incident tracking. Round-tripped only by scripts/inject_anomalies.py.
    # CCSDS scaffolding — kept for future packet-stream ingestion. (audit finding #6, 2026-06-26)
    apid: int
    sequence_count: int
    # NOTE: always "standalone" from SMAPReader (CSV ingestion has no packet framing). Not
    # validated by validate_frame(). Not read downstream. CCSDS framing mode field — kept for
    # future packet-stream ingestion. (audit finding #7, 2026-06-26)
    sequence_flag: Literal["continuation", "first", "last", "standalone"]
    timestamp: float
    channel_id: str
    value: float
    unit: Literal["V", "A", "degC", "Pa", "Hz", "rpm", "W", "dimensionless"]
    subsystem: Literal["EPS", "PROP", "ADCS", "THERMAL", "COMMS"]
    # NOTE: not consumed downstream except by scripts/inject_anomalies.py round-trip.
    # CCSDS scaffolding — same rationale as apid. (audit finding #6, 2026-06-26)
    source_file: str
    row_index: int

#a standalone function, checks appid is 11bit and sequence number is 14 bit
#called by reader.py, not by frame itself.
def validate_frame(frame: TelemetryFrame) -> None:
    """Service-layer validation. Raises ValueError on constraint violation."""
    if not (0 <= frame.apid <= 2047):
        raise ValueError(f"APID {frame.apid} out of 11-bit range [0, 2047]")
    if not (0 <= frame.sequence_count <= 16383):
        raise ValueError(
            f"sequence_count {frame.sequence_count} out of 14-bit range [0, 16383]"
        )
    # unit, subsystem, and sequence_flag are not re-checked here — they are derived
    # from internal maps (_UNIT_MAP, _SUBSYSTEM_MAP) or hardcoded in reader.py and
    # can only hold valid Literal values at this call site.
