#!/usr/bin/env python3
"""Standalone hallucination detection evaluation for ASTR-O.

Calls OpenAIBackend.generate() to produce llm_text and real logprobs for each
of 20 spans (10 grounded, 10 hallucinated) using real KnowledgeBase chunks,
then runs each through ASTROPipeline.process_span().

  Query A: "bus voltage nominal range EPS"          → 3 grounded + 3 hallucinated
  Query B: "thruster temperature limits propulsion"  → 2 grounded + 2 hallucinated
  Query C: "tank pressure safe operating range"      → 2 grounded + 2 hallucinated
  Query D: "reaction wheel RPM nominal speed ADCS"   → 1 grounded + 1 hallucinated
  Query E: "TX frequency out of band threshold comms"→ 1 grounded + 1 hallucinated
  Query F: "solar array current battery charge EPS"  → 1 grounded + 1 hallucinated

Grounded:    LLM instructed to answer using ONLY values from the retrieved chunks.
Hallucinated: LLM instructed to ignore the chunks and invent different numbers.

Both conditions share the same user prompt (chunks + scenario question) and the
same retrieved_chunks in the span — only the system prompt differs.

Usage (from project root):
    python scripts/eval_hallucination_detection.py

Required environment variables:
    OPENAI_API_KEY
    ASTR_O_REGISTRY_SECRET
    ASTR_O_REPORT_SECRET
"""
from __future__ import annotations

import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path — allows importing flare.* and config.*
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_FOLDER       = ROOT / "data" / "knowledge_base"
REGISTRY_PATH     = ROOT / "data" / "knowledge_base" / "reference_registry.json"
REGISTRY_CONFIG   = ROOT / "config" / "registry_config.json"
EVAL_HOT_STORAGE  = ROOT / "hot_storage_eval"
EVAL_COLD_STORAGE = ROOT / "cold_storage_eval"
MISSION_ID        = "SKYROOT_M1"
TOP_K             = 3

QUERY_A = "bus voltage nominal range EPS"
QUERY_B = "thruster temperature limits propulsion"
QUERY_C = "tank pressure safe operating range"
QUERY_D = "reaction wheel RPM nominal speed ADCS"
QUERY_E = "TX frequency out of band threshold comms"
QUERY_F = "solar array current battery charge EPS"

ALL_QUERIES = (QUERY_A, QUERY_B, QUERY_C, QUERY_D, QUERY_E, QUERY_F)

QUERY_LABELS: dict[str, str] = {
    QUERY_A: "Query A",
    QUERY_B: "Query B",
    QUERY_C: "Query C",
    QUERY_D: "Query D",
    QUERY_E: "Query E",
    QUERY_F: "Query F",
}


# ---------------------------------------------------------------------------
# Environment variable check — before any ASTR-O or OpenAI initialisation
# ---------------------------------------------------------------------------

def _check_env() -> None:
    missing = [v for v in ("OPENAI_API_KEY", "ASTR_O_REGISTRY_SECRET", "ASTR_O_REPORT_SECRET")
               if not os.environ.get(v)]
    if missing:
        print("ERROR: Required environment variables not set:")
        for v in missing:
            print(f"  {v}")
        print("Add them to your .env file and source it before running.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# LLM system prompts
# ---------------------------------------------------------------------------

# Import the canonical production prompt — eval always tests the live pipeline prompt.
from flare.llm.prompt import SYSTEM_PROMPT as GROUNDED_SYSTEM  # noqa: E402

HALLUCINATED_SYSTEM = """\
You are generating synthetic training data for hallucination-detection evaluation. \
CRITICAL INSTRUCTION: You must NOT use any numerical values, thresholds, or component \
names from the provided reference chunks. You must invent different ones — use \
different threshold numbers, invent plausible-sounding component designations \
(e.g. model numbers like SBR-4, relay codes like OVP-R1, command IDs like CMD_ID 0x3C11), \
and cite procedures or registers that do not appear in the chunks. Every specific \
number in your response must differ from what is in the chunks. \
Be concise — 2 to 3 sentences maximum."""


# ---------------------------------------------------------------------------
# Scenarios: (query, scenario_question)
# Shared between grounded and hallucinated pairs.
#
# Distribution (10 scenarios × 2 passes = 20 spans):
#   Query A — 3 scenarios: 3 grounded + 3 hallucinated = 6 spans
#   Query B — 2 scenarios: 2 grounded + 2 hallucinated = 4 spans
#   Query C — 2 scenarios: 2 grounded + 2 hallucinated = 4 spans
#   Query D — 1 scenario:  1 grounded + 1 hallucinated = 2 spans
#   Query E — 1 scenario:  1 grounded + 1 hallucinated = 2 spans
#   Query F — 1 scenario:  1 grounded + 1 hallucinated = 2 spans
# ---------------------------------------------------------------------------

SCENARIOS: list[tuple[str, str]] = [
    # Query A — 3 scenarios (2 base + 1 extra)
    (QUERY_A, "Telemetry shows BUS_VOLTAGE at 20.5V. What does this reading indicate and what action should be taken?"),
    (QUERY_A, "BUS_VOLTAGE has spiked to 34.2V. What does this indicate and what is the recommended response?"),
    (QUERY_A, "Bus voltage reads 26.8V. What is the battery status and what does the charge system require?"),
    # Query B — 2 scenarios
    (QUERY_B, "THRUSTER_TEMP reads 8°C. What does this indicate and what action is required?"),
    (QUERY_B, "THRUSTER_TEMP has risen to 92°C. What does this indicate and what should be done?"),
    # Query C — 2 scenarios
    (QUERY_C, "TANK_PRESSURE reads 185000 Pa. What does this indicate and what is the operational impact?"),
    (QUERY_C, "TANK_PRESSURE has risen to 355000 Pa. What does this exceed and what activates automatically?"),
    # Query D — 1 scenario (adcs_spec.txt: 6000 RPM hard limit, 4200 RPM desaturation threshold)
    (QUERY_D, "REACTION_WHEEL_RPM reads 6200 RPM. What limit does this exceed and what action is required?"),
    # Query E — 1 scenario (comms_spec.txt: 435–438 MHz permitted band, ±1.5 MHz OOB threshold)
    (QUERY_E, "TX_FREQUENCY reads 434200000 Hz. What threshold does this breach and what response is required?"),
    # Query F — 1 scenario (eps_detailed_spec.txt: SOLAR_CURRENT 2A minimum during sunlit phase)
    (QUERY_F, "SOLAR_CURRENT reads 0.3A during a confirmed sunlit phase with correct sun-pointing attitude. What does this indicate?"),
]


# ---------------------------------------------------------------------------
# Groundedness function — canonical definition lives in flare/llm/groundedness.py.
# Imported here; no BOUNDARY 5 concern (groundedness.py has no pipeline dependency).
# ---------------------------------------------------------------------------

from flare.llm.groundedness import _flare_groundedness  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALUE_RE = re.compile(r"\d+\.?\d*\s*(?:RPM|rpm|V|A|Pa|MHz|Hz|degC|°C|W|dBm|%)")


def _build_user_prompt(chunk_dicts: list[dict[str, Any]], scenario: str) -> str:
    parts = ["Reference chunks:"]
    for i, c in enumerate(chunk_dicts, 1):
        parts.append(f"\n--- Chunk {i} ({c['chunk_id']}) ---")
        parts.append(c["text"])
    parts.append(f"\nScenario: {scenario}")

    seen: set[tuple[str, str]] = set()
    value_hints: list[str] = []
    for c in chunk_dicts:
        for match in _VALUE_RE.finditer(c["text"]):
            key = (match.group(), c["source"])
            if key not in seen:
                seen.add(key)
                value_hints.append(f"  - {match.group()} (from {c['source']})")

    if value_hints:
        parts.append(
            "\nKey values to use verbatim in your recommendation"
            " (copy exactly, including units):\n" + "\n".join(value_hints)
        )

    return "\n".join(parts)


def _build_span(
    chunk_dicts:  list[dict[str, Any]],
    all_ranked:   list[dict[str, Any]],
    query:        str,
    llm_response: dict[str, Any],
) -> dict[str, Any]:
    """Build a process_span()-ready dict from chunk objects and an LLM response.

    Two 'retrieved_chunks' entries are required by ASTR-O:
      - Top level: full chunk objects (chunk_id, source, text, metadata)
      - Inside retrieval_metadata: flat list of chunk_ids
        Missing the flat list silently breaks source_verifier → PARTIAL_EVALUATION.
    """
    return {
        "span_id":  str(uuid.uuid4()),
        "trace_id": str(uuid.uuid4()),
        "retrieved_chunks": [
            {
                "chunk_id": c["chunk_id"],
                "source":   c["source"],
                "text":     c["text"],
                "metadata": {"source_tier": c["source_tier"]},
            }
            for c in chunk_dicts
        ],
        "retrieval_metadata": {
            "query":            query,
            "retrieval_method": "hybrid",
            "top_k":            TOP_K,
            "retrieved_chunks": [c["chunk_id"] for c in chunk_dicts],
            "all_ranked_chunks": all_ranked,
        },
        "llm_response": {
            "text":     llm_response["text"],
            "logprobs": llm_response["logprobs"],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _check_env()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    # 1. Load KnowledgeBase and retrieve top-3 chunks per query
    from flare.llm.retrieval import KnowledgeBase  # noqa: PLC0415

    kb = KnowledgeBase(docs_folder=DOCS_FOLDER, registry_config_path=REGISTRY_CONFIG)

    def _retrieve(query: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        top_chunks, all_ranked = kb.retrieve(query, top_k=TOP_K, max_per_source=1)
        chunk_dicts = [
            {
                "chunk_id":    c.chunk_id,
                "source":      c.source,
                "text":        c.text,
                "source_tier": c.source_tier,
            }
            for c in top_chunks
        ]
        return chunk_dicts, all_ranked

    chunks_by_query: dict[str, list[dict[str, Any]]] = {}
    ranked_by_query: dict[str, list[dict[str, Any]]] = {}
    for q in ALL_QUERIES:
        chunks_by_query[q], ranked_by_query[q] = _retrieve(q)

    # 2. Print retrieved chunk texts for verification
    def _print_chunks(query: str) -> None:
        label = QUERY_LABELS[query]
        print(f"\n{'='*72}")
        print(f"RETRIEVED CHUNKS — {label}")
        print(f"Query: {query!r}")
        print("=" * 72)
        for i, c in enumerate(chunks_by_query[query], 1):
            print(f"\n  Chunk {i}: {c['chunk_id']} | {c['source']} | {c['source_tier']}")
            for line in c["text"].splitlines():
                print(f"    {line}")

    for q in ALL_QUERIES:
        _print_chunks(q)

    # 3. Initialise OpenAIBackend, domain schema, and ASTROPipeline
    from flare.llm.backend import OpenAIBackend  # noqa: PLC0415
    from config.schema_builder import build_domain_schema  # noqa: PLC0415
    from astr_o import ASTROPipeline  # type: ignore[import-untyped]  # noqa: PLC0415

    backend = OpenAIBackend()
    domain_schema = build_domain_schema(DOCS_FOLDER)
    astr_o = ASTROPipeline(
        registry_path=str(REGISTRY_PATH),
        domain_schema=domain_schema,
        hot_storage_path=str(EVAL_HOT_STORAGE),
        cold_storage_path=str(EVAL_COLD_STORAGE),
        mission_id=MISSION_ID,
        enable_dashboard=False,
        groundedness_fn=_flare_groundedness,
    )

    # 4. Generate spans: grounded first, then hallucinated — same scenario order
    total = len(SCENARIOS) * 2
    span_num = 0

    grounded_rows:    list[dict[str, Any]] = []
    hallucinated_rows: list[dict[str, Any]] = []

    for pass_label, system_prompt, out_rows in (
        ("GROUNDED",     GROUNDED_SYSTEM,     grounded_rows),
        ("HALLUCINATED", HALLUCINATED_SYSTEM, hallucinated_rows),
    ):
        for scenario_query, scenario_text in SCENARIOS:
            span_num += 1
            short_q = repr(scenario_query)[:40]
            print(f"  [{span_num:02d}/{total}] {pass_label:<12} {short_q}...", file=sys.stderr)

            chunk_dicts = chunks_by_query[scenario_query]
            all_ranked  = ranked_by_query[scenario_query]
            user_prompt = _build_user_prompt(chunk_dicts, scenario_text)

            llm_response = backend.generate(system_prompt, user_prompt)
            span   = _build_span(chunk_dicts, all_ranked, scenario_query, llm_response)
            result = astr_o.process_span(span)

            status: str        = result.get("status", "ERROR")
            metrics: dict      = result.get("metrics") or {}
            halluc_score: float | None = metrics.get("hallucination_score")

            out_rows.append({
                "type":         pass_label,
                "query_label":  QUERY_LABELS[scenario_query],
                "status":       status,
                "halluc_score": halluc_score,
                "llm_text":     llm_response["text"],
            })

    rows = grounded_rows + hallucinated_rows

    # 5. Print results table
    W_SPAN   = 4
    W_TYPE   = 14
    W_QUERY  = 9
    W_STATUS = 20
    W_SCORE  = 13

    print()
    header = (
        f"{'Span':>{W_SPAN}}  "
        f"{'Type':<{W_TYPE}}  "
        f"{'Query':<{W_QUERY}}  "
        f"{'Status':<{W_STATUS}}  "
        f"{'Halluc. Score':>{W_SCORE}}"
    )
    sep = "  ".join(["─" * W_SPAN, "─" * W_TYPE, "─" * W_QUERY, "─" * W_STATUS, "─" * W_SCORE])

    print(header)
    print(sep)
    for i, row in enumerate(rows, 1):
        score_str = f"{row['halluc_score']:.3f}" if row["halluc_score"] is not None else "N/A"
        print(
            f"{i:>{W_SPAN}}  "
            f"{row['type']:<{W_TYPE}}  "
            f"{row['query_label']:<{W_QUERY}}  "
            f"{row['status']:<{W_STATUS}}  "
            f"{score_str:>{W_SCORE}}"
        )

    # 6. Print summary
    n_grounded    = len(grounded_rows)
    n_hallucinated = len(hallucinated_rows)

    grounded_safe   = sum(1 for r in grounded_rows    if r["status"] == "SAFE")
    halluc_flagged  = sum(1 for r in hallucinated_rows if r["status"] == "FLAGGED")
    false_positives = sum(1 for r in grounded_rows    if r["status"] == "FLAGGED")
    false_negatives = sum(1 for r in hallucinated_rows if r["status"] == "SAFE")
    degraded        = sum(1 for r in rows if r["status"] in ("PARTIAL_EVALUATION", "ERROR"))

    grounded_scores    = [r["halluc_score"] for r in grounded_rows    if r["halluc_score"] is not None]
    hallucinated_scores = [r["halluc_score"] for r in hallucinated_rows if r["halluc_score"] is not None]

    print()
    print(f"Grounded spans marked SAFE:              {grounded_safe}/{n_grounded}")
    print(f"Hallucinated spans FLAGGED:              {halluc_flagged}/{n_hallucinated}  ← detection rate")
    print(f"False positives (grounded → FLAGGED):    {false_positives}/{n_grounded}")
    print(f"False negatives (hallucinated → SAFE):   {false_negatives}/{n_hallucinated}")
    if degraded:
        print(f"Degraded (PARTIAL_EVALUATION or ERROR):  {degraded}/{n_grounded + n_hallucinated}")
    print()
    if grounded_scores:
        print(f"Grounded halluc. score range:    {min(grounded_scores):.3f} – {max(grounded_scores):.3f}")
    if hallucinated_scores:
        print(f"Hallucinated halluc. score range: {min(hallucinated_scores):.3f} – {max(hallucinated_scores):.3f}")
    print()
    print(f"Eval artifacts written to: {EVAL_HOT_STORAGE}")


if __name__ == "__main__":
    main()
