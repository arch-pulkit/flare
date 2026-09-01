"""FLARE pipeline CLI entrypoint. The only file a user touches directly."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flare.ingestion.frame import TelemetryFrame

logger = logging.getLogger("flare.cli")


# ---------------------------------------------------------------------------
# Environment variable check — called before any pipeline imports
# ---------------------------------------------------------------------------

def check_env_vars(backend: str) -> None:
    """Raise SystemExit with a clear message if any required env var is absent."""
    missing: list[str] = []
    for var in ("ASTR_O_REGISTRY_SECRET", "ASTR_O_REPORT_SECRET"):
        if not os.environ.get(var):
            missing.append(var)
    if backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        print("ERROR: Required environment variables not set:")
        for var in missing:
            print(f"  {var}")
        print("Add them to your .env file and source it before running.")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLARE — Forensic Lifecycle Anomaly Recognition Engine"
    )
    parser.add_argument("--telemetry",    required=True,  help="Path to telemetry CSV")
    parser.add_argument("--mission",      required=True,  help="Mission ID (key in mission_profiles.yaml)")
    parser.add_argument("--backend",      default="openai", choices=["openai", "llama_cpp"],
                        help="LLM backend (default: openai)")
    parser.add_argument("--model",        default=None,   help="Model path (llama_cpp only)")
    parser.add_argument("--top-k",        type=int, default=5, help="Retrieval top_k (default: 5)")
    parser.add_argument("--hot-storage",  default="hot_storage", help="hot_storage root (default: hot_storage)")
    parser.add_argument("--cold-storage", default="cold_storage", help="cold_storage root (default: cold_storage)")
    parser.add_argument("--log-level",    default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"], help="Log level (default: INFO)")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Mission profile loader
# ---------------------------------------------------------------------------

def _load_mission_profile(mission_id: str) -> dict:
    import yaml
    profiles_path = Path("config/mission_profiles.yaml")
    with profiles_path.open(encoding="utf-8") as fh:
        all_profiles = yaml.safe_load(fh)
    if mission_id not in all_profiles:
        print(f"ERROR: Mission ID {mission_id!r} not found in {profiles_path}")
        print(f"Available missions: {list(all_profiles.keys())}")
        raise SystemExit(1)
    return all_profiles[mission_id]


# ---------------------------------------------------------------------------
# Channel map (SMAP raw channel → FLARE channel_id)
# ---------------------------------------------------------------------------

from flare.ingestion.reader import _CHANNEL_MAP


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Step 1: env var check fires before any pipeline work
    check_env_vars(backend=args.backend)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    # Step 2: Load mission profile
    mission_profile = _load_mission_profile(args.mission)

    print(f"FLARE starting — mission={args.mission}  backend={args.backend}  telemetry={args.telemetry}")

    # Step 3: Initialise detector
    from flare.detection.detector import IsolationForestDetector

    model_path = mission_profile.get("model_path", "models/isolation_forest.joblib")
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Run: python -m flare.detection.train --csv data/telemetry.csv "
              "--mission config/mission_profiles.yaml --mission-id SKYROOT_M1 --output models/")
        raise SystemExit(1)

    detector = IsolationForestDetector(model_path=model_path, mission_profile=mission_profile)

    # Step 4: Initialise KnowledgeBase
    from flare.llm.retrieval import KnowledgeBase

    docs_folder = Path(mission_profile.get("docs_folder", "data/knowledge_base"))
    registry_config_path = Path(mission_profile.get("registry_config_path", "config/registry_config.json"))
    knowledge_base = KnowledgeBase(
        docs_folder=docs_folder,
        registry_config_path=registry_config_path,
    )

    # Step 5: Initialise LLM backend
    from flare.llm.backend import LLMBackend

    if args.backend == "openai":
        from flare.llm.backend import OpenAIBackend
        llm_model = mission_profile.get("llm_model", "gpt-4o-mini")
        llm_backend: LLMBackend = OpenAIBackend(model=llm_model)
    else:
        print("ERROR: LlamaCppBackend is post-funding (S10) — not yet implemented.")
        print("Use --backend openai for now.")
        raise SystemExit(1)

    # Step 6: Initialise FLAREPipeline
    from flare.pipeline import FLAREPipeline

    registry_path = mission_profile.get("registry_path", "data/knowledge_base/reference_registry.json")
    incident_log_path = Path(args.hot_storage) / args.mission / "incidents.jsonl"
    top_k = int(mission_profile.get("top_k", args.top_k))

    pipeline = FLAREPipeline(
        mission_id=args.mission,
        registry_path=registry_path,
        hot_storage_path=args.hot_storage,
        cold_storage_path=args.cold_storage,
        llm_backend=llm_backend,
        detector=detector,
        knowledge_base=knowledge_base,
        incident_log_path=incident_log_path,
        mission_profile=mission_profile,
        top_k=top_k,
    )

    # Step 7: Stream frames
    import json
    from flare.detection.detector import ChannelWindowManager
    from flare.ingestion.reader import SMAPReader

    reader = SMAPReader(
        csv_path=args.telemetry,
        mission_profile=mission_profile,
        channel_map=_CHANNEL_MAP,
    )
    channel_manager = ChannelWindowManager(window_size=60)

    frames_processed = 0
    anomalies_detected = 0
    resolved_count = 0
    last_frame: TelemetryFrame | None = None
    # Escalation is detected by a change in the last audit event, not by a change in
    # open-incident count: a SAFE verdict resolves the incident inside the same
    # process_frame() call, so the count is identical before and after.
    seen_audit_id: str | None = None

    try:
        for frame in reader.stream():
            last_frame = frame
            context_window, window_stats = channel_manager.update_with_profile(
                frame, mission_profile
            )
            pipeline.process_frame(frame, context_window, window_stats)
            frames_processed += 1

            audit = pipeline.last_audit
            if audit is not None and audit.event_id != seen_audit_id:
                seen_audit_id = audit.event_id
                anomalies_detected += 1
                if audit.status == "SAFE":
                    resolved_count += 1
                incident_id = audit.incident_id
                print(
                    f"\nANOMALY  channel={frame.channel_id}"
                    f"  value={frame.value:.4f} {frame.unit}"
                    f"  incident={incident_id[:8]}"
                    f"  verdict={audit.status}"
                )
                reco = pipeline.last_recommendation
                if reco is not None:
                    try:
                        parsed = json.loads(reco.llm_text)
                        print(f"  ASSESSMENT:          {parsed['ASSESSMENT']}")
                        print(f"  LIKELY CAUSE:        {parsed['LIKELY_CAUSE']}")
                        print(f"  RECOMMENDED ACTION:  {parsed['RECOMMENDED_ACTION']}")
                        print(f"  URGENCY:             {parsed['URGENCY']}")
                    except (json.JSONDecodeError, KeyError):
                        print(f"  Recommendation: {reco.llm_text}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        logger.error(
            "Pipeline crashed at frame row_index=%s channel=%s: %s",
            last_frame.row_index if last_frame else "N/A",
            last_frame.channel_id if last_frame else "N/A",
            exc,
        )
        print(
            f"\nPipeline error — frames={frames_processed}"
            f"  anomalies={anomalies_detected}"
            f"  resolved={resolved_count}"
            f"  open_incidents={len(pipeline.repository.open_incidents())}"
        )
        raise

    # Step 8: Exit summary
    print(
        f"\nDone — frames={frames_processed}"
        f"  anomalies={anomalies_detected}"
        f"  resolved={resolved_count}"
        f"  open_incidents={len(pipeline.repository.open_incidents())}"
    )
    print(f"Reports written to: {args.hot_storage}/{args.mission}/reports/")


if __name__ == "__main__":
    main()
