# BOUNDARY: flare/audit/events.py → imports from flare.llm.events and stdlib only
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditCompleted:
    event_id: str
    incident_id: str
    parent_event_id: str          # RecommendationGenerated.event_id

    # ASTR-O returns four statuses — all four handled in audit_handler
    status: str                   # "SAFE" | "FLAGGED" | "PARTIAL_EVALUATION" | "ERROR"

    # signed_report is None on ERROR — always check before accessing
    signed_report: dict | None
    signed_error_record: dict | None   # populated on ERROR, None otherwise
    causal_chain: dict | None          # populated on FLAGGED only, None otherwise
    metrics: dict                      # always present
    pipeline_completion_map: dict      # always present — use to debug PARTIAL_EVALUATION
    degraded_confidence: bool

    report_path: str              # absolute path written to hot_storage; empty string on ERROR
    audited_at: float
