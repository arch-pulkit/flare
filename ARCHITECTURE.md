# FLARE — Architecture

Streaming telemetry anomaly detection with an independent audit layer.
Telemetry in → anomaly → grounded recommendation → signed report.

**FLARE detects. ASTR-O verifies. Neither trusts the other.**

---

## Flow

```
CSV / CCSDS
    → TelemetryFrame        typed, immutable
    → IsolationForest       + 3-stage pruning  → AnomalyDetected
    → Incident              DETECTED → INVESTIGATING → RESOLVED (JSONL, replayable)
    → Retrieval + LLM       hybrid RRF over 6 spec docs → RecommendationGenerated
    → ASTR-O                9 verification layers → AuditCompleted (HMAC-SHA256)
    → Streamlit HUD         reads hot_storage JSON only
```

Lineage: `TelemetryFrame.row_index` → `AnomalyDetected.event_id` → `RecommendationGenerated.parent_event_id` → `AuditCompleted.parent_event_id`. Any report replays to the frame that caused it.

---

## Patterns

| Question | Pattern |
|---|---|
| How is it structured? | Hexagonal — domain core, swappable adapters |
| How does computation flow? | Event-driven pipeline, fixed handler sequence |
| How does state survive a crash? | Event sourcing — append-only log + replay |

**Ports / adapters:** `BaseDetector` (sklearn ‖ ONNX) · `LLMBackend` (OpenAI ‖ llama.cpp) · `IncidentRepository` (JSONL ‖ SQLite).

**Boundaries** — dependencies point inward only:

```
frame.py          imports nothing from flare.*
detection/events  imports flare.ingestion
llm/events        imports flare.detection.events
audit/span_builder imports flare.llm.events + astr_o
pipeline.py       the only file allowed to import everything
```

Needing to break one means you're in the wrong layer.

---

## Decisions

**Isolation Forest, not LSTM.** Unsupervised — SMAP has 69 labeled sequences across 25 channels, too few to train supervised. A new satellite has no anomaly history at all. IF also exports to ONNX and needs no ML runtime.

**StandardScaler is mandatory.** TX_FREQUENCY at 437 MHz dominates tree splitting otherwise; every other channel goes invisible. Wrapped in a sklearn `Pipeline`, baked into the ONNX graph.

**Three pruning checks before escalation.** Nominal range → score magnitude → 3 consecutive frames. Costs ~2s of detection latency, buys the FPR reduction.

**Invariants live on the entity.** `Incident.transition_to()` raises `InvalidTransitionError`. The handler orchestrates; the entity enforces. An illegal state is unconstructable anywhere in the codebase.

**Replay bypasses `transition_to()`.** It would mint new UUIDs and destroy `transition_id` continuity. The log only ever contains transitions that already passed validation — replaying known-good history needs no re-checking.

**JSONL, not a database.** Stdlib-only, human-readable, greppable, replays directly into memory on restart. The log is the database; the in-memory dict is a materialised view.

**Direct dispatch, not a message bus.** All four handler relationships were fixed at design time. A bus was built, then removed — every caller needing intermediate events was already reaching past it into private attributes.

**Synchronous, not async.** Retrieval → LLM → audit is inherently sequential. The seam is preserved: make each handler call an `await`.

**Value objects and events are `frozen=True`.** Validation lives in the service layer, never in `__post_init__`.

---

## Detection

6 features per frame: `[value, mean, std, delta_from_nominal, min, max]` over a 60-frame per-channel rolling window.

Three anomaly classes: **point** (one bad value) · **collective** (sequence wrong, no single value out of range) · **contextual** (normal alone, wrong given history). Threshold systems catch only the first.

`score()` always returns `AnomalyDetected` — check `.is_anomaly`, never check for `None`.

---

## ASTR-O

Nine independent layers: registry lookup · contradiction detector · token confidence · source verifier · groundedness · criteria gate · causal chain mapper · integrity signer · storage.

Four statuses, all four must be handled: `SAFE` · `FLAGGED` · `PARTIAL_EVALUATION` · `ERROR`. `metrics` is always a dict, never `None`. `signed_report` is `None` on ERROR.

Groundedness is word-level overlap against retrieved chunks, so generation is constrained to spec vocabulary *before* the check runs — the LLM arranges spec phrases rather than composing prose.

`CrossSpanCorrelator` runs after every audit over the last 20 spans: retrieval drift, hallucination clusters, repeated failure categories.

---

## Measurement

Detection is unsupervised, so ground truth is manufactured: `inject_anomalies.py` places known anomalies at known frames, `LegacyBaseline` (pure threshold, no context, no persistence) runs on the identical sequence.

**FPR** — escalations vs threshold alarms on the same 84,176 SMAP frames. The metric that holds up.
**TTD** — −2.0s. FLARE is structurally slower on threshold-crossing anomalies; that's the persistence cost, stated plainly.
**Latency** — ~3.1s from `detected_at` to `audited_at`. This is pipeline processing time, not a human resolution time. Don't compare the two.

---

## Concepts used

Hexagonal architecture · dependency inversion · event sourcing · CQRS-style materialised view · DDD aggregate + invariant enforcement · value objects · domain events with explicit lineage · deep modules (small interface, large implementation) · append-only log as source of truth · unsupervised anomaly detection · hybrid dense+sparse retrieval with reciprocal rank fusion · retrieval-constrained generation · HMAC-SHA256 integrity signing.
