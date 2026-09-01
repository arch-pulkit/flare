# BOUNDARY: dashboard/readers.py — reads hot_storage JSON only; no flare package imports.
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def read_recent_reports(
    hot_storage_path: Path,
    mission_id: str,
    n: int = 20,
) -> list[dict]:
    """Read the last n signed_report JSON files from reports/, newest first.

    Returns [] on any error — never raises.
    """
    reports_dir = hot_storage_path / mission_id / "reports"
    if not reports_dir.exists():
        return []
    try:
        files = sorted(
            [f for f in reports_dir.iterdir() if f.is_file() and f.suffix == ".json"],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:n]
        result: list[dict] = []
        for f in files:
            try:
                result.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                logger.exception("read_recent_reports: failed to read %s", f)
        return result
    except Exception:
        logger.exception("read_recent_reports: failed to list %s", reports_dir)
        return []


def read_recent_causal_chains(
    hot_storage_path: Path,
    mission_id: str,
    n: int = 10,
) -> list[dict]:
    """Read the last n causal chain JSON files from causal_chains/, newest first.

    Returns [] on any error — never raises.
    """
    chains_dir = hot_storage_path / mission_id / "causal_chains"
    if not chains_dir.exists():
        return []
    try:
        files = sorted(
            [f for f in chains_dir.iterdir() if f.is_file() and f.suffix == ".json"],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:n]
        result: list[dict] = []
        for f in files:
            try:
                result.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                logger.exception("read_recent_causal_chains: failed to read %s", f)
        return result
    except Exception:
        logger.exception("read_recent_causal_chains: failed to list %s", chains_dir)
        return []


def read_validation_matrix(
    hot_storage_path: Path,
    mission_id: str,
) -> dict | None:
    """Read validation_matrix.json. Returns None if file missing or unreadable."""
    matrix_file = hot_storage_path / mission_id / "validation_matrix.json"
    if not matrix_file.exists():
        return None
    try:
        return json.loads(matrix_file.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("read_validation_matrix: failed to read %s", matrix_file)
        return None


def read_open_incidents(
    hot_storage_path: Path,
    mission_id: str,
) -> list[dict]:
    """Read incidents.jsonl; return summaries of non-RESOLVED incidents.

    Each entry:
      incident_id, channel_id, current_state, opened_at, age_seconds
    Returns [] on any error — never raises.
    """
    incidents_file = hot_storage_path / mission_id / "incidents.jsonl"
    if not incidents_file.exists():
        return []
    try:
        incidents: dict[str, dict] = {}
        for line in incidents_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                iid = entry["incident_id"]
                if iid not in incidents:
                    incidents[iid] = {
                        "incident_id": iid,
                        "channel_id": entry.get("channel_id", ""),
                        "opened_at": entry.get("opened_at", 0.0),
                        "current_state": entry.get("to_state", "DETECTED"),
                    }
                else:
                    incidents[iid]["current_state"] = entry.get(
                        "to_state", incidents[iid]["current_state"]
                    )
            except Exception:
                logger.exception("read_open_incidents: failed to parse line")
        now = time.time()
        return [
            {**v, "age_seconds": now - v["opened_at"]}
            for v in incidents.values()
            if v["current_state"] != "RESOLVED"
        ]
    except Exception:
        logger.exception("read_open_incidents: failed to read %s", incidents_file)
        return []
