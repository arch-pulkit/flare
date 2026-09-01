# FLARE — GitHub Upload Readiness Audit

**Audited:** 2026-09-01 · Full file-by-file review of `/Users/dr_bolty/flare`
**Test status at audit time:** 70/70 passing in 53s (`python -m pytest tests/ -q`)

---

## Repo state

**There is no git repository.** `git rev-parse` finds no `.git` in `/Users/dr_bolty/flare`
or any parent. An earlier session snapshot reported a repository rooted at the *home*
directory (`/Users/dr_bolty`), which would have staged the entire home folder; that is
gone now. First step before anything else:

```bash
git init
```

Run `git init` **inside `flare/`**, never from the home directory.

---

## Blockers

### 1. 261 MB of regenerable data would be committed

`.gitignore` covers `data/smap_raw/` — a directory that does not exist — but misses the
ones that do.

| Path | Size | Why it should not ship |
|---|---|---|
| `data/test/` | 115 MB (82 `.npy`) | Kaggle download; README documents how to fetch |
| `data/2018-05-19_15.00.10/` | 95 MB (247 files) | Orphaned telemanom output — **zero references in FLARE code** |
| `data/train/` | 51 MB (82 `.npy`) | Same as `data/test/` |
| `data/schema_registry.db` (+ `-shm`, `-wal`) | 2.1 MB | Auto-created at pipeline init |

Total to exclude across all categories: ~480 files / 472 MB.

### 2. ASTR-O cannot be installed by anyone else

Hard import in `flare/audit/handlers.py`, `flare/pipeline.py`, `config/schema_builder.py`,
and `build_registry.py`. Installed editable from `/Users/dr_bolty/astr-o`, which **is not a
git repository** and has never been pushed. `requirements.txt` does not list it. README
links to `github.com/arch-pulkit/ASTR-O` — nothing can install FLARE until that repo exists.

### 3. `run_pipeline.py:208` — `logger` is never defined

The module imports `logging` but never calls `getLogger`. Any pipeline exception reaches
`logger.error(...)` in the `except` handler and raises `NameError`, destroying the original
traceback.

`TelemetryFrame` at line 166 is also undefined, but harmless — `from __future__ import
annotations` means the local variable annotation is never evaluated. Lint error only.

### 4. `simulation_harness.py` reproduces the bug CLAUDE.md #35 says was fixed

`run_flare_on_sequence` counts escalations by `open_incidents()` delta — the pattern its
docstring still calls "same pattern as run_pipeline.py", which is stale since
`run_pipeline.py` was fixed on 2026-09-01. A SAFE verdict resolves the incident *inside*
`process_frame()`, so the incident never appears in either sample point. **Every SAFE
escalation is invisible to the counter.**

This corrupts the headline number:

- Persisted matrix reports `fpr_flare = 0.000677` (≈57 escalations / 84,176 frames) → **3.6× FPR reduction**
- `hot_storage_sim/SKYROOT_M1/incidents.jsonl` holds 971 unique incidents and 248 RESOLVED transitions
- Not one RESOLVED incident could have been counted

MTTR survives only because a separate audit-fallback path captures timings. FPR — described
in `CLAUDE.md` as "the metric that holds up" — is inflated by an unknown but large factor.

**Fix:** switch the detection condition to watch `pipeline.last_audit.event_id` change,
matching `run_pipeline.py`. Then re-run the benchmark. The honest FPR number may be
materially worse than 3.6×.

### 5. `dashboard/app.py` hardcodes stale figures

The "About these metrics" expander states `Measured FLARE MTTR: ~5.8s` and
`Reductions: 52× (simple) / 155× (novel)`. The actual persisted value is 3.09s, and those
ratios derive from the broken FPR path. Reviewers reading the HUD get numbers that do not
match the project's own artifacts.

### 6. Missing files

`LICENSE` and `.env.example` are absent. Both needed.

### Not a problem

No secrets are hardcoded anywhere. `.env` is correctly ignored.
`data/knowledge_base/reference_registry.json` contains only SHA-256 hashes — no secret
material (`grep -ci secret` → 0).

---

## File-by-file verdict

Legend: ✅ ready to commit · ⚠️ needed but not ready · ⛔ do not commit (gitignore)

| File / Path | Verdict | Note |
|---|---|---|
| **Root** | | |
| `README.md` | ⚠️ Not ready | Install steps unfollowable until ASTR-O is published; no license section |
| `ARCHITECTURE.md` | ✅ Ready | |
| `SPRINT_LOG.md` | ✅ Ready | |
| `CLAUDE.md` | ✅ Ready | Agent-facing, but harmless and useful to publish |
| `GITHUB_READINESS.md` | ✅ Ready | This file |
| `run_pipeline.py` | ⚠️ Not ready | Undefined `logger` (L208); undefined `TelemetryFrame` annotation (L166) |
| `build_registry.py` | ✅ Ready | |
| `requirements.txt` | ⚠️ Not ready | Missing `astr_o` |
| `requirements-dev.txt` | ✅ Ready | |
| `.gitignore` | ⚠️ Not ready | Misses 261 MB; see blocker 1 |
| `LICENSE` | ⚠️ Absent | Needed |
| `.env.example` | ⚠️ Absent | Needed — README describes three vars |
| `.env` | ⛔ Ignore | Already ignored ✓ |
| `.DS_Store` | ⛔ Ignore | Not currently ignored |
| `.vscode/settings.json` | ⛔ Ignore | Local editor config |
| `.claude/settings.local.json` | ⛔ Ignore | Contains absolute local paths |
| `__pycache__/`, `.pytest_cache/` | ⛔ Ignore | Already ignored ✓ |
| **flare/** | | |
| `ingestion/frame.py`, `ingestion/reader.py` | ✅ Ready | |
| `detection/events.py`, `detector.py`, `train.py` | ✅ Ready | |
| `state/incident.py`, `state/handlers.py` | ✅ Ready | |
| `llm/events.py`, `backend.py`, `groundedness.py`, `prompt.py`, `retrieval.py`, `handlers.py` | ✅ Ready | |
| `audit/events.py`, `span_builder.py`, `handlers.py` | ✅ Ready | |
| `metrics/__init__.py`, `validation.py` | ✅ Ready | |
| `pipeline.py` | ✅ Ready | |
| all `__init__.py` | ✅ Ready | |
| **config/** | | |
| `mission_profiles.yaml`, `registry_config.json`, `schema_builder.py`, `__init__.py` | ✅ Ready | |
| **dashboard/** | | |
| `readers.py`, `__init__.py` | ✅ Ready | |
| `app.py` | ⚠️ Not ready | Stale hardcoded metrics (5.8s MTTR, 52×/155×) |
| **scripts/** | | |
| `__init__.py`, `build_smap_csv.py`, `inject_anomalies.py`, `eval_hallucination_detection.py` | ✅ Ready | |
| `simulation_harness.py` | ⚠️ Not ready | Escalation-count bug inflates FPR; stale docstring |
| **tests/** | | |
| all 11 `test_*.py` | ✅ Ready | 70/70 pass |
| **data/** | | |
| `knowledge_base/*.txt` (6 files) | ✅ Ready | |
| `knowledge_base/reference_registry.json` | ✅ Ready | Hashes only, no secret material |
| `telemetry.csv` (3.8 MB) | ✅ Ready | Committed intentionally per README |
| `schema_registry.db` + `-shm`/`-wal` | ⛔ Ignore | Auto-created at init |
| `train/` (51 MB), `test/` (115 MB) | ⛔ Ignore | Kaggle-sourced; README documents fetch |
| `2018-05-19_15.00.10/` (95 MB) | ⛔ Ignore *or delete* | Orphaned telemanom output, zero code references |
| **models/** | | |
| `isolation_forest.joblib` / `.onnx` / `.meta.json` (1.9 MB) | ✅ Ready | Committed intentionally |
| **storage** | | |
| `hot_storage/` (73 MB), `hot_storage_sim/` (130 MB) | ⛔ Ignore | Already ignored ✓ |

**Tally:** 44 source files ready · 6 need fixes · 2 need creating · ~480 files / 472 MB to exclude.

---

## Suggested `.gitignore` additions

```gitignore
# Raw SMAP dataset — fetch from Kaggle, see README
data/train/
data/test/
data/smap_raw/

# Orphaned telemanom output — not used by FLARE
data/2018-05-19_15.00.10/

# Runtime-generated schema registry
data/schema_registry.db
data/schema_registry.db-shm
data/schema_registry.db-wal

# OS / editor
.DS_Store
.vscode/
.claude/
```

---

## Pre-upload checklist

- [ ] `git init` inside `flare/`
- [ ] Extend `.gitignore` (above)
- [ ] Fix `run_pipeline.py` undefined `logger`
- [ ] Fix `simulation_harness.py` escalation counter → re-run benchmark → update reported FPR
- [ ] Update `dashboard/app.py` stale metric text
- [ ] Add `astr_o` to `requirements.txt` (or document the install path)
- [ ] Publish ASTR-O, or vendor it / mark FLARE as not-yet-runnable standalone
- [ ] Add `LICENSE`
- [ ] Add `.env.example`
- [ ] `git status --porcelain -uall | wc -l` → confirm well under 480 before first commit
