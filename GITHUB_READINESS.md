# FLARE — GitHub Upload Readiness Audit

**Audited:** 2026-09-01 · Full file-by-file review
**Repo path:** `/Users/dr_bolty/Projects/flare` — moved since the audit. ASTR-O moved too
and now lives at `/Users/dr_bolty/Projects/astr-o`.
**Test status:** 71/71 passing (`python -m pytest tests/ -q`)

> **Remediation status — updated 2026-09-01.** Git is initialised and the tree is
> committed. Blockers 1, 3, 5, and 6 are resolved; blocker 4 is fixed in code but the
> benchmark has not been re-run, so **no FPR figure should be quoted yet**. Blocker 2
> (ASTR-O availability) is not a defect to fix — ASTR-O is in its deployment phase and
> not yet public — but it does gate third-party installability until release. See the
> checklist at the bottom for current state.

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

### 2. ASTR-O is not yet public — FLARE is not third-party installable until it is

**Status: expected, not a defect.** ASTR-O is in its deployment phase and has not been
released. This is a sequencing constraint on FLARE's publication, not a bug to fix here.

Hard import in `flare/audit/handlers.py`, `flare/pipeline.py`, `config/schema_builder.py`,
and `build_registry.py`, so FLARE will not start without it. It is installed editable from
a local checkout at `/Users/dr_bolty/Projects/astr-o` and is not on PyPI, so it cannot be
a resolvable line in `requirements.txt`.

Resolved since the audit:

- The editable install still pointed at the pre-move path `/Users/dr_bolty/astr-o`, so
  `import astr_o` failed with `ModuleNotFoundError` and the test suite could not run at
  all. Reinstalled from the new location.
- README no longer links to `github.com/arch-pulkit/ASTR-O`. That repo is not live, so
  the link 404'd for any reader who followed it. README and `requirements.txt` now state
  the deployment-phase status and document the editable-install route instead.

Remaining: FLARE can be published before ASTR-O, but the README must keep saying plainly
that the audit layer is not yet available, or reviewers will clone it and hit an import
error on the first run.

### 3. `run_pipeline.py:208` — `logger` is never defined

The module imports `logging` but never calls `getLogger`. Any pipeline exception reaches
`logger.error(...)` in the `except` handler and raises `NameError`, destroying the original
traceback.

`TelemetryFrame` at line 166 is also undefined, but harmless — `from __future__ import
annotations` means the local variable annotation is never evaluated. Lint error only.

### 4. `simulation_harness.py` reproduces a known, already-documented bug

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
in the engineering notes as "the metric that holds up" — is inflated by an unknown but
large factor.

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
| `README.md` | ✅ Ready | ASTR-O deployment-phase status + editable install documented; license section added |
| `ARCHITECTURE.md` | ✅ Ready | |
| `SPRINT_LOG.md` | ✅ Ready | |
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
| `*.local.json` | ⛔ Ignore | Local tool config, absolute paths |
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

# OS / editor / local tool config
.DS_Store
.vscode/
*.local.json
```

---

## Pre-upload checklist

- [x] `git init` inside `flare/` — initialised on `main`
- [x] Extend `.gitignore` — 69 files / 6.0 MB tracked, down from ~480 files / 472 MB
- [x] Fix `run_pipeline.py` undefined `logger` — commit `a98ae90`
- [x] Fix `simulation_harness.py` escalation counter — commit `6f8fe11`, regression test added
- [ ] **Re-run the benchmark and update the reported FPR** — still outstanding; the
      persisted validation matrix predates the counter fix and is stale
- [x] Update `dashboard/app.py` stale metric text — commit `efb321d`
- [x] Document the ASTR-O install path in `requirements.txt` — commit `bbfb48a`
- [x] Mark FLARE as not-yet-runnable standalone — README and `requirements.txt` state
      ASTR-O's deployment-phase status; publishing ASTR-O itself is a separate release
- [x] Add `LICENSE` — Apache 2.0, commit `b0a9a51`
- [x] Add `.env.example` — commit `bbfb48a`
- [x] Confirm tracked file count before first commit — 69 files

### Still open

1. **Re-run `python -m scripts.simulation_harness --scenario fpr`.** The counter fix
   changes the measured escalation count, so every FPR figure in the persisted matrix
   is stale. The honest number may be materially worse than the 3.6× previously claimed.
   Nothing should quote an FPR reduction until this runs.
2. **ASTR-O release gates third-party installability.** It is in its deployment phase
   and not yet public, which is expected — but until it ships, nobody outside this
   machine can run FLARE end-to-end. It is also not under version control at its new
   path (`/Users/dr_bolty/Projects/astr-o`); worth doing before release.
3. **Fill in the LICENSE copyright holder.** The Apache appendix keeps the
   `[yyyy] [name of copyright owner]` template.
4. **Rotate `ASTR_O_REGISTRY_SECRET`.** The local `.env` sets it to a short dictionary
   word. It is the HMAC key over the reference registry. `.env` is correctly ignored,
   so nothing leaked, but the value is weak for a demo that presents cryptographic
   grounding as a feature.
