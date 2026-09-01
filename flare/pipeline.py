# BOUNDARY 5: pipeline.py — only file allowed to import
# across all layers. Nothing imports this file except
# run_pipeline.py.
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from flare.audit.events import AuditCompleted
from flare.audit.handlers import (
    make_audit_handler,
    make_correlator_handler,
)
from flare.detection.detector import BaseDetector
from flare.detection.events import AnomalyDetected, WindowStats
from flare.ingestion.frame import TelemetryFrame
from flare.llm.backend import LLMBackend
from flare.llm.events import RecommendationGenerated
from flare.llm.groundedness import _flare_groundedness
from flare.llm.handlers import make_llm_handler
from flare.llm.retrieval import KnowledgeBase
from flare.state.handlers import make_state_machine_handler
from flare.state.incident import IncidentRepository

logger = logging.getLogger(__name__)

MIN_CONSECUTIVE_FRAMES: int = 3


class FLAREPipeline:
    def __init__(
        self,
        mission_id: str,
        registry_path: str,
        hot_storage_path: str,
        cold_storage_path: str,
        llm_backend: LLMBackend,
        detector: BaseDetector,
        knowledge_base: KnowledgeBase,
        incident_log_path: Path,
        mission_profile: dict,
        top_k: int = 5,
        schema_registry_path: str = "data/schema_registry.db",
    ) -> None:
        # Public attributes — read by audit_handler via pipeline reference
        self.mission_id = mission_id
        self.hot_storage_path = hot_storage_path
        self.cold_storage_path = cold_storage_path

        # decision_threshold: key may be "decision_threshold" or legacy "anomaly_threshold"
        self.decision_threshold: float = float(
            mission_profile.get(
                "decision_threshold",
                mission_profile.get("anomaly_threshold", -0.1),
            )
        )
        self._consecutive_anomaly_counts: dict[str, int] = {}

        self._detector = detector

        # 1. Build domain schema from knowledge base documents
        from config.schema_builder import build_domain_schema
        from astr_o import ASTROPipeline  # type: ignore[import-untyped]
        from astr_o.schema import SchemaRegistry

        docs_folder = Path(registry_path).parent
        domain_schema = build_domain_schema(docs_folder)

        # Wire into SchemaRegistry for version tracking.
        # mission_id is used as project_name — 1:1 mapping.
        schema_registry = SchemaRegistry(schema_registry_path)

        try:
            # First run for this mission — register schema
            version_id = schema_registry.register_schema(
                project_name=mission_id,
                schema=domain_schema,
                description=f"Auto-registered by FLAREPipeline from {docs_folder}",
            )
            logging.info(
                "Schema registered for mission %s: %s",
                mission_id, version_id,
            )
        except ValueError:
            # Schema already exists for this mission_id.
            # Migrate to capture any knowledge base changes.
            # migrate_schema is non-breaking for additions —
            # warnings are logged but do not block init.
            version_id, warnings = schema_registry.migrate_schema(
                project_name=mission_id,
                new_schema=domain_schema,
                description=f"Auto-migrated by FLAREPipeline from {docs_folder}",
            )
            if warnings:
                logging.warning(
                    "Schema migration warnings for mission %s: %s",
                    mission_id, warnings,
                )
            logging.info(
                "Schema migrated for mission %s: %s",
                mission_id, version_id,
            )

        # Store version_id as public attribute — appears in audit logs and can be read by callers
        self.schema_version_id = version_id

        # Use the versioned schema from the registry, not the raw build output —
        # ensures ASTROPipeline uses exactly the schema that was registered,
        # including any normalisation SchemaRegistry applies
        retrieved = schema_registry.get_schema(mission_id)
        if retrieved is None:
            logging.warning(
                "SchemaRegistry returned None for mission %s "
                "— falling back to build_domain_schema() output",
                mission_id,
            )
            retrieved = domain_schema
        domain_schema = retrieved

        # 2. Initialise ASTROPipeline
        self.astr_o_pipeline = ASTROPipeline(
            registry_path=registry_path,
            domain_schema=domain_schema,
            hot_storage_path=hot_storage_path,
            cold_storage_path=cold_storage_path,
            mission_id=mission_id,
            enable_dashboard=False,
            groundedness_fn=_flare_groundedness,
        )

        # 3. Initialise IncidentRepository; replay log if it exists
        self.repository = IncidentRepository(log_path=incident_log_path)
        if incident_log_path.exists():
            self.repository.load_from_log(incident_log_path)

        # 4. Build all handlers via factory functions; stored as instance attributes
        # for direct dispatch in process_frame(). Stage 1: state → llm.
        # Stage 2: audit → correlator. No message bus.
        self._state_handler = make_state_machine_handler(self.repository)
        self._llm_handler = make_llm_handler(
            knowledge_base=knowledge_base,
            llm_backend=llm_backend,
            mission_id=mission_id,
            top_k=top_k,
            max_per_source=1,
        )
        self._audit_handler = make_audit_handler(self.astr_o_pipeline, self.repository)
        self._correlator_handler = make_correlator_handler(
            hot_storage_path=hot_storage_path,
            mission_id=mission_id,
        )

        # Observation slots — set by process_frame() on each escalation so callers
        # (run_pipeline.py, simulation_harness.py) can read results without splicing
        # into the handler chain.
        self.last_recommendation: RecommendationGenerated | None = None
        self.last_audit: AuditCompleted | None = None

    def process_frame(
        self,
        frame: TelemetryFrame,
        context_window: tuple[TelemetryFrame, ...],
        window_stats: WindowStats,
    ) -> None:
        incident_id = str(uuid.uuid4())
        result = self._detector.score(
            frame, context_window, window_stats, self.mission_id, incident_id
        )

        if not result.is_anomaly:
            # Nominal frame: reset persistence counter for this channel.
            # This is the ONLY place the counter is reset.
            self._consecutive_anomaly_counts[result.frame.channel_id] = 0
            return

        # --- Pruning check 1: nominal range sanity ---
        # IsolationForest can flag statistically unusual values that are still
        # physically within spec. Suppress them — escalating a physically safe
        # reading would produce noise for operators.
        if (result.window_stats.nominal_low
                <= result.frame.value
                <= result.window_stats.nominal_high):
            logging.debug(
                "Pruned: %s value %.3f within nominal range "
                "[%.3f, %.3f] — statistically unusual but physically safe",
                result.frame.channel_id,
                result.frame.value,
                result.window_stats.nominal_low,
                result.window_stats.nominal_high,
            )
            self._consecutive_anomaly_counts[result.frame.channel_id] = 0
            return

        # --- Pruning check 2: score magnitude ---
        # Scores close to the decision threshold are noise, not signal.
        # decision_threshold is negative (e.g. -0.1); 0.8× brings the boundary
        # inward: -0.1 * 0.8 = -0.08. Frames with score > -0.08 are too close.
        if result.anomaly_score > (self.decision_threshold * 0.8):
            logging.debug(
                "Pruned: %s score %.4f too close to threshold %.4f "
                "(boundary %.4f) — likely noise",
                result.frame.channel_id,
                result.anomaly_score,
                self.decision_threshold,
                self.decision_threshold * 0.8,
            )
            self._consecutive_anomaly_counts[result.frame.channel_id] = 0
            return

        # --- Pruning check 3: persistence ---
        # Require MIN_CONSECUTIVE_FRAMES consecutive anomalous frames before
        # escalating. Single-frame spikes are noise.
        channel = result.frame.channel_id
        self._consecutive_anomaly_counts[channel] = (
            self._consecutive_anomaly_counts.get(channel, 0) + 1
        )

        if self._consecutive_anomaly_counts[channel] < MIN_CONSECUTIVE_FRAMES:
            logging.debug(
                "Pruned: %s anomalous for %d/%d consecutive frames "
                "— not yet persistent",
                channel,
                self._consecutive_anomaly_counts[channel],
                MIN_CONSECUTIVE_FRAMES,
            )
            return

        # --- All checks passed — escalate ---
        # Counter intentionally NOT reset here. A sustained anomaly continues
        # signaling on every frame until a nominal frame arrives. Silence after
        # first escalation would be a safety failure.

        # Stage 1: state_handler (side effect: opens/transitions incident) → llm_handler
        self.last_recommendation = None
        self.last_audit = None

        self._state_handler(result)
        rec_event = self._llm_handler(result)
        if rec_event is None:
            logger.warning(
                "Pipeline Stage 1 (llm_handler) returned None for incident %s"
                " — skipping audit.",
                result.incident_id,
            )
            return
        self.last_recommendation = rec_event

        # Stage 2: audit_handler → correlator_handler
        audit_event = self._audit_handler(rec_event)
        if audit_event is None:
            logger.warning(
                "Pipeline Stage 2 (audit_handler) returned None for incident %s.",
                result.incident_id,
            )
            return
        self.last_audit = audit_event
        self._correlator_handler(audit_event)
