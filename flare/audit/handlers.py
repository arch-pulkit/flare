# BOUNDARY: flare/audit/handlers.py → imports from flare.audit.*, flare.llm.events, astr_o, and stdlib only
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Protocol

from astr_o import ASTROPipeline, CrossSpanCorrelator

from flare.audit.events import AuditCompleted
from flare.audit.span_builder import build_span
from flare.llm.events import RecommendationGenerated

logger = logging.getLogger(__name__)


class IncidentStore(Protocol):
    """Structural protocol for incident persistence — keeps audit layer independent of state layer."""
    def get(self, incident_id: str) -> Optional[object]: ...
    def save(self, incident: object) -> None: ...


def _write_report(
    report: dict,
    hot_storage_path: Path,
    mission_id: str,
    event_id: str,
) -> str:
    report_dir = hot_storage_path / mission_id / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"{event_id}.json"
    report_file.write_text(json.dumps(report), encoding="utf-8")
    return str(report_file.resolve())


def make_audit_handler(
    pipeline: ASTROPipeline,
    repository: IncidentStore | None = None,
) -> Callable[[RecommendationGenerated], AuditCompleted]:
    hot_storage_path = Path(pipeline.hot_storage_path)
    mission_id = pipeline.mission_id

    def audit_handler(event: RecommendationGenerated) -> AuditCompleted:
        audit_event_id = str(uuid.uuid4())
        span = build_span(event)
        result = pipeline.process_span(span)
        status = result["status"]
        metrics = result["metrics"]
        completion_map = result["pipeline_completion_map"]

        if status == "SAFE":
            signed_report = result["signed_report"]
            signed_error_record = None
            causal_chain = None
            report_path = _write_report(signed_report, hot_storage_path, mission_id, audit_event_id)
            degraded = False
            logger.info(
                "audit SAFE: incident=%s safety_score=%.4f latency_ms=%.1f",
                event.incident_id,
                metrics.get("overall_safety_score", 0.0),
                metrics.get("latency_ms", 0.0),
            )
            if repository is not None:
                incident = repository.get(event.incident_id)
                if incident is not None and not incident.is_resolved():  # type: ignore[union-attr]
                    incident.transition_to(  # type: ignore[union-attr]
                        new_state="RESOLVED",
                        trigger="astr_o_safe_verdict",
                        evidence={
                            "span_id": result["span_id"],
                            "overall_safety_score": metrics.get("overall_safety_score"),
                            "audited_at": time.time(),
                        },
                    )
                    repository.save(incident)
                    logger.info(
                        "Incident %s resolved — ASTR-O verdict SAFE",
                        event.incident_id,
                    )

        elif status == "FLAGGED":
            signed_report = result["signed_report"]
            signed_error_record = None
            causal_chain = result["causal_chain"]
            report_path = _write_report(signed_report, hot_storage_path, mission_id, audit_event_id)
            degraded = False
            logger.warning(
                "audit FLAGGED: incident=%s root_cause_chunks=%s",
                event.incident_id,
                causal_chain.get("root_cause_chunks") if causal_chain else None,
            )

        elif status == "PARTIAL_EVALUATION":
            signed_report = result["signed_report"]
            signed_error_record = None
            causal_chain = None
            # guard: never call _write_report with None
            report_path = (
                _write_report(signed_report, hot_storage_path, mission_id, audit_event_id)
                if signed_report
                else ""
            )
            degraded = True
            logger.warning(
                "audit PARTIAL_EVALUATION: incident=%s failed_layers=%s",
                event.incident_id,
                {k: v for k, v in completion_map.items() if v == "failed"},
            )

        else:  # ERROR
            signed_report = None
            signed_error_record = result["signed_error_record"]
            causal_chain = None
            report_path = ""
            degraded = True
            metrics = result.get("metrics") or {}
            logger.error(
                "audit ERROR: incident=%s error=%s failed_layers=%s",
                event.incident_id,
                result.get("error"),
                {k: v for k, v in completion_map.items() if v == "failed"},
            )
            if signed_error_record is not None:
                errors_dir = hot_storage_path / mission_id / "reports" / "errors"
                errors_dir.mkdir(parents=True, exist_ok=True)
                error_file = errors_dir / f"{audit_event_id}.json"
                error_file.write_text(json.dumps(signed_error_record), encoding="utf-8")
            else:
                logger.error(
                    "signed_error_record is None on ERROR status — "
                    "incident_id=%s, no error file written, "
                    "pipeline_completion_map not persisted",
                    event.incident_id,
                )

        return AuditCompleted(
            event_id=audit_event_id,
            incident_id=event.incident_id,
            parent_event_id=event.event_id,
            status=status,
            signed_report=signed_report,
            signed_error_record=signed_error_record,
            causal_chain=causal_chain,
            metrics=metrics,
            pipeline_completion_map=completion_map,
            degraded_confidence=degraded,
            report_path=report_path,
            audited_at=time.time(),
        )

    return audit_handler



def make_correlator_handler(
    hot_storage_path: str,
    mission_id: str,
) -> Callable[[AuditCompleted], AuditCompleted]:
    # CrossSpanCorrelator.__init__ is attr-only (no I/O); analyze() does the
    # hot_storage read at call time, so a single instance sees every new span.
    correlator = CrossSpanCorrelator(
        hot_storage_path=hot_storage_path,
        mission_id=mission_id,
    )

    def correlator_handler(event: AuditCompleted) -> AuditCompleted:
        alerts = correlator.analyze(window=20)
        for alert in alerts:
            logger.warning(
                "PATTERN ALERT: %s — %s",
                alert["pattern"],
                alert["recommendation"],
            )
        return event

    return correlator_handler
