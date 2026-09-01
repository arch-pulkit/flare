# FLARE Sprint Log
Design and rationale: `ARCHITECTURE.md`

---

| Date | Entry | Tests |
|------|-------|-------|
| 2026-09-01 | Doc consolidation (5,853 → 929 md lines); dead code removed (audit.py, website/, stray data + storage dirs); run_pipeline escalation counter fixed; train.py _CHANNEL_MAP consolidated | 70/70 |
| 2026-07-02 | Eval prompt sync — GROUNDED_SYSTEM imports SYSTEM_PROMPT; Issue #9 RESOLVED (6/10 SAFE) | 70/70 |
| 2026-07-02 | Prompt Rule 7 — context-constrained RECOMMENDED_ACTION generation | 70/70 |
| 2026-07-02 | top_k restored to 5 + dashboard import fix | 70/70 |
| 2026-06-27 | Over-engineering audit G1–G7 | 70/70 |
| 2026-06-27 | Over-engineering audit F1–F4 | 70/70 |
| 2026-06-27 | MessageBus removed — direct dispatch | 70/70 |
| 2026-06-26 | Dead-code cleanup C1–C8 | 70/70 |
| 2026-06-26 | Architecture fixes B1–B3 (groundedness, channel map, score() type) | 70/70 |
| 2026-06-26 | TTD invisible claim — honest documentation | 70/70 |
| 2026-06-26 | Audit fix — top_k mismatch + ISRO_M1 stub removal | 70/70 |
| 2026-06-11 | Issue #9 — Stopword iteration 2 + decision to stop | 70/70 |
| 2026-06-11 | Post-S9 — Issues #8 and #9: contradiction + groundedness fix | 70/70 |
| 2026-06-11 | Post-S9 — Improvement 4: dual-track MTTR baseline | 70/70 |
| 2026-06-11 | Improvement 2 — SCENARIO_TTD_INVISIBLE | 70/70 |
| 2026-06-11 | Post-S9 — Connection audit + audit script | 69/69 |
| 2026-06-10 | Post-S9 — domain_schema + meta.json standardisation | 69/69 |
| 2026-06-09 | Post-S9 — Real NASA SMAP .npy swap | 69/69 |
| 2026-06-07 | Post-S9 — Real SMAP data integration (Improvement 1) | 69/69 |
| 2026-06-06 | Post-S9 — SchemaRegistry wiring | 69/69 |
| 2026-06-06 | Post-S9 — ASTR-O integration fixes: Issues #3, #6, #7 | 69/69 |
| 2026-06-06 | S9 — Mission HUD (Streamlit dashboard) | 69/69 |
| 2026-06-06 | Post-S8 — SCENARIO_TTD_INVISIBLE nominal range mismatch | 69/69 |
| 2026-06-06 | S8 — Simulation Environment | 69/69 |
| 2026-06-06 | S7 — Validation Matrix | 62/62 |
| 2026-06-05 | S6 — _flare_groundedness injection | 51/51 |
| 2026-06-05 | S6 — Groundedness diagnosis | 51/51 |
| 2026-06-02 | S6 — Prompt hardening attempt 1 | 51/51 |
| 2026-06-02 | Pre-S6 — LLM output parsing fix | 51/51 |
| 2026-06-02 | Pre-S6 — Incident resolution on SAFE verdict | 48/48 |
| 2026-06-02 | Pre-S6 — Hallucination eval expanded to 6 queries | 48/48 |
| 2026-06-02 | Pre-S6 — Knowledge base expansion | 46/46 |
| 2026-06-01 | Pre-S6 eval — Hallucination detection | 45/45 |
| 2026-06-01 | Pre-S6 fix — anomaly_threshold calibration | 46/46 |
| 2026-06-01 | Pre-S6 fix — decision_threshold calibration | 46/46 |
| 2026-06-01 | Pre-S6 fix — Pruning filter | 46/46 |
| 2026-06-01 | Pre-S6 fix — StandardScaler | 45/45 |
| — | S5 — CLI + Storage | 45/45 |
| — | S4 — LLM + Span Builder | 39/39 |
| — | S3 — Retrieval + Knowledge Base | 25/25 |
| — | S2 — State Machine | 17/17 |
| — | S1 — Ingestion + Detection | 9/9 |
