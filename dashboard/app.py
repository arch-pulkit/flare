# BOUNDARY: dashboard/app.py — reads hot_storage JSON only; no flare package imports.
# Pure reader of hot_storage JSON files + Streamlit render layer.
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from readers import (
    read_open_incidents,
    read_recent_causal_chains,
    read_recent_reports,
    read_validation_matrix,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

HOT_STORAGE = Path(os.environ.get("FLARE_HOT_STORAGE", "hot_storage"))
HOT_STORAGE_SIM = Path(os.environ.get("FLARE_HOT_STORAGE_SIM", "hot_storage_sim"))
MISSION_ID = os.environ.get("FLARE_MISSION_ID", "SKYROOT_M1")
REFRESH_INTERVAL_S = 2

st.set_page_config(
    page_title="FLARE Mission HUD",
    page_icon="🛰️",
    layout="wide",
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

_STATUS_BADGE: dict[str, str] = {
    "SAFE":               "✅ SAFE",
    "FLAGGED":            "⚠️ FLAGGED",
    "PARTIAL_EVALUATION": "~ PARTIAL",
    "ERROR":              "✗ ERROR",
}

_STATUS_COLOR: dict[str, str] = {
    "SAFE":               "green",
    "FLAGGED":            "orange",
    "PARTIAL_EVALUATION": "goldenrod",
    "ERROR":              "red",
}


def _extract_status(report: dict) -> str:
    try:
        return report["report"]["executive_summary"]["verdict"]
    except (KeyError, TypeError):
        return "ERROR"


def _extract_channel(report: dict) -> str:
    try:
        query: str = report["report"]["executive_summary"]["query"]
        return query.split(" anomaly:")[0]
    except (KeyError, TypeError, IndexError):
        return "—"


def _extract_hmac(report: dict) -> str:
    try:
        return report["signature_metadata"]["signature"][:16]
    except (KeyError, TypeError):
        return "—"


def _extract_timestamp(report: dict) -> str:
    try:
        return report["report"]["generated_at"][:19].replace("T", " ")
    except (KeyError, TypeError):
        return "—"


def _extract_urgency(report: dict) -> str:
    try:
        answer: str = report["report"]["executive_summary"]["answer"]
        try:
            return json.loads(answer).get("URGENCY", "—")
        except (json.JSONDecodeError, AttributeError):
            pass
        m = re.search(r"URGENCY\s*:\s*(\w+)", answer)
        if m:
            return m.group(1).strip()
    except (KeyError, TypeError):
        pass
    return "—"


def _extract_recommended_action(report: dict) -> str | None:
    try:
        answer: str = report["report"]["executive_summary"]["answer"]
        try:
            return json.loads(answer).get("RECOMMENDED_ACTION") or None
        except (json.JSONDecodeError, AttributeError):
            pass
        m = re.search(r"RECOMMENDED_ACTION\s*:\s*(.*?)(?=\nURGENCY:|$)", answer, re.DOTALL)
        if m:
            return m.group(1).strip() or None
    except (KeyError, TypeError):
        pass
    return None


def _criteria_score(report: dict, key: str) -> float:
    """Return 1.0 if named criteria passed, 0.0 otherwise."""
    try:
        return 1.0 if report["report"]["criteria_evaluation"][key]["passed"] else 0.0
    except (KeyError, TypeError):
        return 0.0


def _count_reports(hot_storage_path: Path, mission_id: str) -> int:
    d = hot_storage_path / mission_id / "reports"
    if not d.exists():
        return 0
    try:
        return sum(1 for f in d.iterdir() if f.is_file() and f.suffix == ".json")
    except Exception:
        return 0


# ─── Page render ─────────────────────────────────────────────────────────────

try:
    # ── Panel 1: Pipeline Status ──────────────────────────────────────────────
    st.header(f"🛰️ FLARE Mission HUD — {MISSION_ID}")

    open_incidents = read_open_incidents(HOT_STORAGE, MISSION_ID)
    reports = read_recent_reports(HOT_STORAGE, MISSION_ID, n=20)
    report_count = _count_reports(HOT_STORAGE, MISSION_ID)

    last_status = _extract_status(reports[0]) if reports else "—"
    last_color = _STATUS_COLOR.get(last_status, "gray")

    p1_col1, p1_col2, p1_col3 = st.columns(3)
    p1_col1.metric("Open Incidents", len(open_incidents))
    p1_col2.metric("Reports Written", report_count)
    with p1_col3:
        st.markdown(
            f"**Last Verdict:** "
            f"<span style='color:{last_color};font-weight:bold'>{last_status}</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Panels 2 + 3: Two Doctors | Validation Matrix ─────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("⚕️ The Two Doctors")
        doc1, doc2 = st.columns(2)

        with doc1:
            st.markdown("**FLARE** — Detection")
            if open_incidents:
                for inc in open_incidents[:10]:
                    age = inc["age_seconds"]
                    age_color = "green" if age < 30 else ("orange" if age < 120 else "red")
                    st.markdown(
                        f"• **{inc['channel_id']}** · {inc['current_state']} · "
                        f"<span style='color:{age_color}'>{age:.0f}s</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✓ No active incidents")

        with doc2:
            st.markdown("**ASTR-O** — Verification")
            if reports:
                latest = reports[0]
                status = _extract_status(latest)
                _badge_map: dict[str, tuple[str, str]] = {
                    "SAFE":               ("✓ VERIFIED", "green"),
                    "FLAGGED":            ("⚠ FLAGGED", "orange"),
                    "PARTIAL_EVALUATION": ("~ PARTIAL", "goldenrod"),
                    "ERROR":              ("✗ ERROR", "red"),
                }
                badge_text, badge_color = _badge_map.get(status, ("? UNKNOWN", "gray"))
                st.markdown(
                    f"<span style='color:{badge_color};font-size:1.2em;"
                    f"font-weight:bold'>{badge_text}</span>",
                    unsafe_allow_html=True,
                )
                hmac = _extract_hmac(latest)
                st.code(f"Proof: {hmac}...", language=None)

                # Three bars derived from criteria evaluation (numeric scores not persisted)
                ground_score = _criteria_score(latest, "criteria_4_groundedness")
                retrieval_score = _criteria_score(latest, "criteria_3_source_verification")
                token_score = _criteria_score(latest, "criteria_2_token_confidence")

                st.markdown(f"**Grounding** (1 − hallucination): `{ground_score:.2f}`")
                st.progress(ground_score)
                st.markdown(f"**Retrieval Quality**: `{retrieval_score:.2f}`")
                st.progress(retrieval_score)
                st.markdown(f"**Token Confidence**: `{token_score:.2f}`")
                st.progress(token_score)

                if status == "FLAGGED":
                    causal_chains = read_recent_causal_chains(HOT_STORAGE, MISSION_ID, n=1)
                    if causal_chains:
                        cc = causal_chains[0]
                        ca = cc.get("causal_analysis", {})
                        with st.expander("Root Cause Analysis"):
                            fc = ca.get("failure_category") or cc.get("primary_failure", "—")
                            rcc = cc.get("root_cause_chunks", [])
                            recs = cc.get("recommendations", [])
                            st.markdown(f"**Root cause:** {fc or '—'}")
                            if rcc:
                                st.markdown(
                                    f"**Affected chunks:** {', '.join(str(c) for c in rcc[:3])}"
                                )
                            if recs:
                                st.markdown(f"**Recommendation:** {recs[0]}")
            else:
                st.info("No audit reports yet — run the pipeline first.")

    with col_right:
        st.subheader("📊 Validation Matrix")
        matrix = read_validation_matrix(HOT_STORAGE_SIM, MISSION_ID)

        if matrix is None:
            st.info("Run `python -m scripts.simulation_harness --scenario mttr` to populate metrics.")
            c1, c2, c3 = st.columns(3)
            c1.metric("False Positive Rate", "—")
            c2.metric("Mean Time to Resolution", "—")
            c3.metric("Time to Detect", "—")
        else:
            fpr_ratio = matrix.get("fpr_reduction_ratio") or 1.0
            fpr_legacy = matrix.get("fpr_legacy", 0.0)
            fpr_flare = matrix.get("fpr_flare", 0.0)
            mttr_flare = matrix.get("mttr_flare_seconds", 0.0)
            mttr_legacy = matrix.get("mttr_legacy_seconds", 900.0)
            mttr_legacy_simple = matrix.get("mttr_legacy_simple_seconds", 300.0)
            mttr_legacy_novel = matrix.get("mttr_legacy_novel_seconds", 900.0)
            ttd_flare = matrix.get("ttd_flare_seconds", 0.0)
            ttd_adv = matrix.get("ttd_advantage_seconds", 0.0)

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "False Positive Rate",
                f"{fpr_ratio:.1f}×",
                delta=f"Legacy {fpr_legacy*100:.1f}% → FLARE {fpr_flare*100:.1f}%",
                delta_color="normal",
            )
            c2.metric(
                "Mean Time to Resolution",
                f"{mttr_flare:.1f}s",
                delta=f"vs {mttr_legacy:.0f}s manual (novel)",
                delta_color="inverse",
            )
            with c2:
                st.caption(
                    f"FLARE automates spec retrieval + LLM recommendation + cryptographic audit. "
                    f"Novel anomaly track: {mttr_legacy_novel:.0f}s legacy "
                    f"(Hundman et al. KDD 2018). "
                    f"Simple/known pattern: {mttr_legacy_simple:.0f}s legacy. "
                    f"Both vs FLARE: {mttr_flare:.1f}s measured end-to-end."
                )
            if ttd_adv > 0:
                c3.metric(
                    "Time to Detect",
                    f"{ttd_flare:.1f}s",
                    delta=f"{ttd_adv:.1f}s earlier",
                    delta_color="normal",
                )
            else:
                c3.metric(
                    "Time to Detect",
                    f"{ttd_flare:.1f}s",
                    delta="threshold-crossing anomaly",
                    delta_color="off",
                )

        with st.expander("About these metrics"):
            st.markdown(
                "**FPR:** 3-frame burst → 1 FLARE escalation vs N legacy alarms. "
                "Ratio improves with real SMAP data noise floor.\n\n"
                "**MTTR:** Measured from `AnomalyDetected.detected_at` to "
                "`AuditCompleted.audited_at`. Includes retrieval, LLM call, and ASTR-O "
                "cryptographic audit. Measured FLARE MTTR: ~5.8s.\n\n"
                "Two legacy baselines:\n"
                "- **Simple/known anomaly: 300s** (conservative floor — experienced "
                "engineer, no spec lookup required)\n"
                "- **Novel anomaly (FLARE's case): 900s** (15 min) — consistent with "
                "manual ops burden characterised by Hundman et al. KDD 2018 and NASA NTRS "
                "20210007857. This is the correct comparison for FLARE, which automates "
                "spec retrieval and recommendation generation end-to-end.\n\n"
                "Reductions: 52× (simple) / 155× (novel). "
                "Use `python -m scripts.simulation_harness --scenario mttr --mttr-legacy 300` "
                "for conservative mode.\n\n"
                "**TTD:** Threshold-crossing anomalies only. Statistical deviation "
                "detection (anomalies invisible to legacy) pending empirical "
                "verification on real SMAP data."
            )

    st.divider()

    # ── Panel 4: Audit Log ────────────────────────────────────────────────────
    st.subheader("📋 Audit Log — Last 20 Reports")

    if reports:
        rows = []
        for rep in reports:
            status = _extract_status(rep)
            rows.append({
                "Time": _extract_timestamp(rep),
                "Status": _STATUS_BADGE.get(status, status),
                "Channel": _extract_channel(rep),
                "Urgency": _extract_urgency(rep),
                "Span ID": rep.get("report", {}).get("span_id", "—")[:8] + "...",
                "Sig": _extract_hmac(rep),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("**Select a span to view full report:**")
        span_labels = [
            f"{rep.get('report', {}).get('span_id', '—')[:8]}... ({_extract_channel(rep)})"
            for rep in reports
        ]
        selected_idx = st.selectbox(
            "Span ID", range(len(span_labels)), format_func=lambda i: span_labels[i], key="span_select"
        )
        selected = reports[selected_idx]

        action = _extract_recommended_action(selected)
        if action:
            st.info(f"**RECOMMENDED ACTION:** {action}")

        st.json(selected)
    else:
        st.info("No reports found — run the pipeline first.")

except Exception:
    logger.exception("HUD render error")
    st.error("Dashboard render error — stale data shown. Pipeline may not be running.")

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
time.sleep(REFRESH_INTERVAL_S)
st.rerun()
