# BOUNDARY: flare/state/handlers.py → imports from flare.state.incident and flare.detection.events only
#Where the log is first written in a live pipeline run.
from __future__ import annotations

import logging
from collections.abc import Callable

from flare.detection.events import AnomalyDetected
from flare.state.incident import IncidentRepository

logger = logging.getLogger(__name__)

#the service layer orchestrates, not enforce.
#One sentence, whole architecture : "Handler orchestrates, Entity enforces."
def make_state_machine_handler(
    repository: IncidentRepository,
) -> Callable[[AnomalyDetected], AnomalyDetected]:
    """Return a side-effecting handler closed over the given repository.

    Purpose: open the Incident in the repository and transition it to
    INVESTIGATING. Returns the same AnomalyDetected event unchanged.
    process_frame() calls it for its side effect and passes the original
    event directly to the next handler — the return value is intentionally
    unused by the pipeline.
    """

    #calls the log
    def state_machine_handler(event: AnomalyDetected) -> AnomalyDetected:
        incident = repository.get(event.incident_id)
        if incident is None:
            incident = repository.open(
                incident_id=event.incident_id,
                mission_id=event.mission_id,
                channel_id=event.frame.channel_id,
                subsystem=event.frame.subsystem,
                opened_at=event.detected_at,
            )

        if incident.is_resolved():
            logger.warning(
                "state_machine_handler: incident %s is already RESOLVED — returning event unchanged",
                event.incident_id,
            )
            return event

        # Normal anomaly flow: DETECTED → INVESTIGATING
        if incident.current_state == "DETECTED":
            next_state = "INVESTIGATING"
        else:
            logger.error(
                "Incident %s already %r — re-escalation produces a signed report with "
                "no corresponding state transition. incident_id=%s",
                event.incident_id,
                incident.current_state,
                event.incident_id,
            )
            return event

        # InvalidTransitionError from transition_to() propagates — never caught here.
        # The entity enforces invariants; the handler orchestrates only.
        incident.transition_to(
            new_state=next_state,
            trigger="anomaly_detected",
            evidence={
                "channel_id":         event.frame.channel_id,
                "anomaly_score":      event.anomaly_score,
                "decision_threshold": event.decision_threshold,
                "detected_at":        event.detected_at,
            },
        )
        repository.save(incident)
        return event

    return state_machine_handler
