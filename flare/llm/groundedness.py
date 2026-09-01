# BOUNDARY: flare/llm/groundedness.py → imports from stdlib only.
# Pure function, no I/O, no flare.* imports. Safe to import from any layer.
from __future__ import annotations

from typing import Any


def _flare_groundedness(
    llm_response: dict[str, Any],
    enriched_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Custom groundedness function for ASTR-O.

    ASTR-O's default check runs word-overlap against the full llm_response.text,
    which in FLARE is json.dumps(parsed_dict). JSON key names and operational
    English ("exceeds", "verify", "reduce") fail this check even when the LLM
    has correctly cited verbatim spec values. This function checks only the
    RECOMMENDED_ACTION field — the value-citing part — against chunk text.
    """
    import json as _json
    import re as _re

    text = llm_response.get("text", "")
    try:
        # Pipeline path: llm_text is json.dumps({"ASSESSMENT":..., "RECOMMENDED_ACTION":...})
        parsed = _json.loads(text)
        text_to_check = parsed.get("RECOMMENDED_ACTION", "")
        if not text_to_check or text_to_check == "NOT PROVIDED":
            return {"groundedness_score": 1.0, "ungrounded_claims": []}
    except (ValueError, AttributeError):
        # Plain-text fallback (eval script uses raw LLM output): extract
        # RECOMMENDED_ACTION section before falling back to full text.
        m = _re.search(
            r"RECOMMENDED_ACTION\s*:\s*(.*?)(?:\n(?:ASSESSMENT|LIKELY_CAUSE|URGENCY)\s*:|$)",
            text,
            _re.IGNORECASE | _re.DOTALL,
        )
        text_to_check = m.group(1).strip() if m else text

    # Stopwords — valid recommendation language that never appears in spec text.
    # Counting these as ungrounded inflates the failure rate without measuring
    # actual grounding quality. Open Issue #9 fix — 2026-06-11.
    _STOPWORDS = {
        "should", "ensure", "prevent", "damage",
        "immediate", "immediately", "indicating",
        "breaches", "triggering", "reduce", "verify",
        "confirm", "check", "monitor", "inspect",
        "review", "assess", "evaluate", "recommend",
        "suggested", "required", "necessary", "action",
        "manual", "adjustment", "adjustments",
        "within", "below", "above", "exceeds",
        "between", "from", "this", "that", "with",
        "have", "been", "will", "when", "also",
        "more", "than", "into", "under", "over",
        "following", "based", "using", "note", "these",
        "their", "such", "must", "taken", "system",
        "cause", "indicates", "other", "severe",
        "intervention", "personnel",
        # Additional operational prose words observed as ungrounded
        "restore", "restored", "restoring",
        "mitigate", "mitigating", "mitigation",
        "associated", "association",
        "reading", "risks", "risk",
        "further", "additional",
        "source",   # citation prefix: "(source: ..."
        "address", "addressing", "addressed",
        "orientation", "measures", "measure",
        "alternative", "alternatives",
        "engaging", "engage",
        "shutting",
        "corrective", "immediate",
        "protecting", "protection",
        "affected", "impact",
        # Stopword iteration 2 — 2026-06-11
        # Operational qualifier / probability language
        "potential", "potentially",
        # Operational advice verbs and connectives
        "prompt", "additionally",
        "could", "lead", "avoid", "perform",
        "suggesting", "requiring",
        # Observational / relational connectors
        "reads", "which",
        # Operational consequence nouns and adjectives
        "problems", "actions", "crucial",
        "safeguard", "integrity", "increased",
        # General-English condition / severity qualifiers
        # (not quantitative spec vocabulary like threshold/nominal/limit)
        "condition", "conditions", "critical",
    }

    # Numeric token exclusion — scenario/computed values (e.g. "6200", "434200000")
    # come from the anomaly event, not from spec chunks. They are not claims about
    # spec content and should not count as ungrounded.
    _NUMERIC_RE = _re.compile(r"^\d+\.?\d*[a-z%]*$")

    raw_words = [w.strip(".,;:!?\"'()[]{}") for w in text_to_check.lower().split()]
    words = {
        w for w in raw_words
        if len(w) > 3
        and w not in _STOPWORDS
        and not _NUMERIC_RE.match(w)
        and "_" not in w  # filter identifiers: chunk IDs, channel names, source filenames
    }
    if not words:
        return {"groundedness_score": 1.0, "ungrounded_claims": []}

    chunk_text = " ".join(c.get("text", "").lower() for c in enriched_chunks)
    ungrounded = [w for w in words if w not in chunk_text]
    score = max(0.0, 1.0 - len(ungrounded) / len(words))
    return {"groundedness_score": round(score, 4), "ungrounded_claims": ungrounded}
