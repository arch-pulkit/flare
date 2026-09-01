# BOUNDARY: flare/llm/prompt.py → imports from flare.detection.events, flare.llm.events, and stdlib only
from __future__ import annotations

import re

from flare.detection.events import AnomalyDetected
from flare.llm.events import RetrievedChunk

SYSTEM_PROMPT: str = (
    "You are a spacecraft anomaly analyst for mission\n"
    "operations. A telemetry anomaly has been detected.\n"
    "You will be given the anomaly data and relevant\n"
    "excerpts from spacecraft specification documents.\n\n"
    "Your task: produce a structured recommendation for\n"
    "the mission operations team by selecting and arranging\n"
    "phrases directly from the provided specification excerpts.\n\n"
    "CRITICAL RULES — violation causes audit failure:\n"
    "1. Every numerical limit, threshold, or range you\n"
    "   mention must be copied exactly from the provided\n"
    "   specification excerpts. Do not paraphrase numbers.\n"
    "   Do not round. Do not estimate.\n"
    "2. Every sentence containing a number must cite its\n"
    "   source immediately after the number:\n"
    '     "...below 4200 RPM (source: adcs_spec.txt)..."\n'
    "3. Your RECOMMENDED_ACTION must contain at least one\n"
    "   exact numerical value from the specifications.\n"
    "   The number must appear verbatim as it appears in\n"
    "   the excerpt — same digits, same units.\n"
    "4. Do not speculate beyond what the specifications\n"
    "   state. If the specs do not cover the anomaly,\n"
    "   say so explicitly.\n"
    '5. Do not fabricate values. If you cannot find a\n'
    '   relevant threshold in the excerpts, say "threshold\n'
    '   not found in retrieved context."\n'
    "6. Do NOT mention the detected anomaly value in\n"
    "   RECOMMENDED_ACTION. Never write the measured value\n"
    "   (e.g. \"6200 RPM\" or \"437050000 Hz\") in your\n"
    "   recommendation. Cite ONLY the spec threshold from\n"
    "   the retrieved context (e.g. \"below 4200 RPM per\n"
    "   adcs_spec.txt\"). The anomaly value is irrelevant\n"
    "   to the recommendation — the threshold is what\n"
    "   matters.\n"
    "7. RECOMMENDED_ACTION must use only words and phrases\n"
    "   that appear in the provided specification excerpts.\n"
    "   Do not generate new sentences from scratch — select\n"
    "   the most relevant phrases from the retrieved context\n"
    "   and arrange them into your recommendation. Every\n"
    "   content word must come from the excerpts, not from\n"
    "   your own vocabulary.\n\n"
    "Output format — use these exact labels:\n"
    "ASSESSMENT: [what is wrong, severity, affected subsystem]\n"
    "LIKELY_CAUSE: [root cause hypothesis based on specs]\n"
    "RECOMMENDED_ACTION: [phrases selected from retrieved excerpts only, with exact values and sources]\n"
    "URGENCY: [IMMEDIATE if safety-critical | HIGH if degraded | NOMINAL if monitoring only]"
)

_LABEL_RE = re.compile(
    r"^(ASSESSMENT|LIKELY_CAUSE|RECOMMENDED_ACTION|URGENCY)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

_VALUE_RE = re.compile(r"\d+\.?\d*\s*(?:RPM|rpm|V|A|Pa|MHz|Hz|degC|°C|W|dBm|%)")


def build_query(anomaly: AnomalyDetected) -> str:
    """Pure function — no I/O. Produces retrieval query from anomaly event."""
    return (
        f"{anomaly.frame.channel_id} anomaly: value {anomaly.frame.value} {anomaly.frame.unit}, "
        f"threshold {anomaly.decision_threshold}, subsystem {anomaly.frame.subsystem}"
    )


def build_user_prompt(anomaly: AnomalyDetected, chunks: list[RetrievedChunk]) -> str:
    window_min = anomaly.window_stats.min
    window_max = anomaly.window_stats.max
    window_count = len(anomaly.context_window)

    context_sections = "\n\n".join(
        f"[Source: {chunk.source}]\n{chunk.text}"
        for chunk in chunks
    )

    seen: set[tuple[str, str]] = set()
    value_hints: list[str] = []
    for chunk in chunks:
        for match in _VALUE_RE.finditer(chunk.text):
            key = (match.group(), chunk.source)
            if key not in seen:
                seen.add(key)
                value_hints.append(f"  - {match.group()} (from {chunk.source})")

    hint_section = ""
    if value_hints:
        hint_section = (
            "\n\nKey values to use verbatim in your recommendation"
            " (copy exactly, including units):\n"
            + "\n".join(value_hints)
        )

    return (
        f"ANOMALY REPORT\n"
        f"Channel:         {anomaly.frame.channel_id}\n"
        f"Current value:   {anomaly.frame.value} {anomaly.frame.unit}\n"
        f"Subsystem:       {anomaly.frame.subsystem}\n"
        f"Anomaly score:   {anomaly.anomaly_score:.4f} (threshold: {anomaly.decision_threshold})\n"
        f"Context window:  {window_count} frames, value range "
        f"[{window_min:.3f}, {window_max:.3f}] {anomaly.frame.unit}\n\n"
        f"RETRIEVED SPECIFICATION CONTEXT\n"
        f"{context_sections}"
        f"{hint_section}"
    )


def parse_llm_response(raw_text: str) -> dict[str, str]:
    """Extract the four structured fields from an LLM response.

    Scans for ASSESSMENT / LIKELY_CAUSE / RECOMMENDED_ACTION / URGENCY labels
    at the start of a line (case-insensitive).  Each field's value runs until
    the next label or end of text.  Missing fields are set to "NOT PROVIDED".
    If no labels are found at all, returns the full raw_text as ASSESSMENT.
    """
    result: dict[str, str] = {k: "NOT PROVIDED" for k in ("ASSESSMENT", "LIKELY_CAUSE", "RECOMMENDED_ACTION", "URGENCY")}

    matches = list(_LABEL_RE.finditer(raw_text))
    if not matches:
        result["ASSESSMENT"] = raw_text.strip()
        return result

    for i, match in enumerate(matches):
        key = match.group(1).upper()
        value_start = match.end()
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        value = raw_text[value_start:value_end].strip()
        if key in result:
            result[key] = value

    return result
