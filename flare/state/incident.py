# BOUNDARY: flare/state/incident.py → imports stdlib only (no flare.* dependencies)
#The JSNOL log is first written here, save() is the log writer.
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#the law of state machine
#a plain dict, single source of truth for legal state moves
VALID_TRANSITIONS: dict[str, list[str]] = {
    "DETECTED":      ["INVESTIGATING", "RESOLVED"],
    "INVESTIGATING": ["RESOLVED"],
    "RESOLVED":      [],  # terminal — transition_to() always raises here
}


#custom exception, raised by transition_to() | raised by entity, not by handler.

class InvalidTransitionError(Exception):
    pass


#frozen(fact that happened), has uuid; what gets written to JSONL log, one line per transition
@dataclass(frozen=True)
class StateTransitionEvent:
    transition_id: str      # uuid4 — required for audit traceability and replay determinism
    incident_id: str
    from_state: str
    to_state: str
    trigger: str        #"anomaly_detected"
    evidence: dict      #dict: channel_id, score, threshold, detected_at
    transitioned_at: float      #float timestampß


#mutable entity
@dataclass
class Incident:
    """Mutable entity. Enforces all state-transition invariants.
    Never calls the repository. Never does I/O."""
    incident_id: str
    mission_id: str
    channel_id: str
    subsystem: str
    current_state: str      #changes over lifetime
    opened_at: float        #not until RESOLVED
    closed_at: float | None = None
    transitions: list[StateTransitionEvent] = field(default_factory=list)   #list, grows over time

    #checks valid transitions, raises invalidtransitions if bad
    def transition_to(
        self,
        new_state: str,
        trigger: str,
        evidence: dict,
    ) -> StateTransitionEvent:
        if self.current_state == "RESOLVED":
            raise InvalidTransitionError(
                f"Incident {self.incident_id} is already RESOLVED (terminal state)"
            )
        allowed = VALID_TRANSITIONS.get(self.current_state, [])
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition {self.current_state!r} → {new_state!r} "
                f"for incident {self.incident_id}. Allowed: {allowed}"
            )
        te = StateTransitionEvent(
            transition_id=str(uuid.uuid4()),
            incident_id=self.incident_id,
            from_state=self.current_state,
            to_state=new_state,
            trigger=trigger,
            evidence=evidence,
            transitioned_at=time.time(),
        )
        self.current_state = new_state
        self.transitions.append(te)
        if new_state == "RESOLVED":
            self.closed_at = te.transitioned_at
        return te

    def is_resolved(self) -> bool:
        return self.current_state == "RESOLVED"

    def age_seconds(self) -> float:
        return time.time() - self.opened_at


class IncidentRepository:
    """The only component that touches the JSONL log."""

    def __init__(self, log_path: Path) -> None:
        self._incidents: dict[str, Incident] = {}
        self._log_path = log_path

    def open(
        self,
        incident_id: str,
        mission_id: str,
        channel_id: str,
        subsystem: str,
        opened_at: float,
    ) -> Incident:
        """Idempotent. Caller passes opened_at explicitly (from event.detected_at)."""
        if incident_id in self._incidents:
            return self._incidents[incident_id]
        incident = Incident(
            incident_id=incident_id,
            mission_id=mission_id,
            channel_id=channel_id,
            subsystem=subsystem,
            current_state="DETECTED",
            opened_at=opened_at,
        )
        self._incidents[incident_id] = incident
        return incident

    def get(self, incident_id: str) -> Incident | None:
        return self._incidents.get(incident_id)

    def save(self, incident: Incident) -> None:
        """Append the last StateTransitionEvent as a JSON line to the log."""
        if not incident.transitions:
            logger.warning("save() called on incident %s with no transitions", incident.incident_id)
            return
        te = incident.transitions[-1]
        record = {
            "transition_id":  te.transition_id,
            "incident_id":    te.incident_id,
            "mission_id":     incident.mission_id,
            "channel_id":     incident.channel_id,
            "subsystem":      incident.subsystem,
            "opened_at":      incident.opened_at,
            "from_state":     te.from_state,
            "to_state":       te.to_state,
            "trigger":        te.trigger,
            "evidence":       te.evidence,
            "transitioned_at": te.transitioned_at,
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def load_from_log(self, log_path: Path) -> None:
        """Replay every JSONL line to rebuild in-memory state from scratch.

        Directly constructs StateTransitionEvent from stored data to preserve
        transition_id — calling transition_to() would generate new UUIDs.
        First line for an incident_id: call open() (creates DETECTED), then
        apply the stored transition directly.
        Subsequent lines: apply stored transition directly.
        """
        self._incidents = {}
        self._log_path = log_path
        if not log_path.exists():
            return
        with log_path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                    incident_id = record["incident_id"]

                    if incident_id not in self._incidents:
                        # First line for this incident: reconstruct in DETECTED state
                        self.open(
                            incident_id=incident_id,
                            mission_id=record["mission_id"],
                            channel_id=record["channel_id"],
                            subsystem=record["subsystem"],
                            opened_at=record["opened_at"],
                        )

                    incident = self._incidents[incident_id]
                    te = StateTransitionEvent(
                        transition_id=record["transition_id"],
                        incident_id=incident_id,
                        from_state=record["from_state"],
                        to_state=record["to_state"],
                        trigger=record["trigger"],
                        evidence=record["evidence"],
                        transitioned_at=record["transitioned_at"],
                    )
                    incident.transitions.append(te)
                    incident.current_state = record["to_state"]  # direct mutation: transition_to() would generate a new UUID, breaking transition_id preservation
                    if record["to_state"] == "RESOLVED":
                        incident.closed_at = record["transitioned_at"]
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.error(
                        "Skipping corrupted line in %s: %s",
                        log_path, exc,
                    )
                    continue

    def open_incidents(self) -> list[Incident]:
        return [i for i in self._incidents.values() if not i.is_resolved()]
