# FLARE

**Forensic Lifecycle Anomaly Recognition Engine** — streaming telemetry anomaly detection with an independent cryptographic audit layer.

Telemetry in → anomaly detected → incident tracked → spec-grounded recommendation → HMAC-signed audit report.

Runs end-to-end on real NASA SMAP satellite data. See [ARCHITECTURE.md](ARCHITECTURE.md) for design and [SPRINT_LOG.md](SPRINT_LOG.md) for history.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Requires **ASTR-O**, which provides the audit layer — the grounding checks, criteria
gate, and HMAC-SHA256 report signing. FLARE imports it directly and will not start
without it.

ASTR-O is currently in its deployment phase and **not yet public**, so there is no
clone URL to point at and it is not on PyPI. Until it is released, FLARE runs only
where ASTR-O is already available locally; install it editable from your checkout
before installing the requirements above:

```bash
pip install -e /path/to/astr-o
```

If the checkout moves after installation, the editable install keeps pointing at the
old absolute path and `import astr_o` fails with `ModuleNotFoundError`. Re-run
`pip install -e .` from the new location.

Create `.env` (never commit):

```bash
ASTR_O_REGISTRY_SECRET=<any-secret>
ASTR_O_REPORT_SECRET=<any-secret>
OPENAI_API_KEY=<your-key>
```

```bash
set -a && source .env && set +a
```

## Data and model

`data/telemetry.csv` and `models/` are committed. Rebuild only if missing.

```bash
# Telemetry CSV from real SMAP .npy files.
# Needs data/train/ and data/test/ from:
#   kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
python -m scripts.build_smap_csv --smap-dir data --output data/telemetry.csv

# Train the detector
python -m flare.detection.train \
  --csv data/telemetry.csv \
  --mission config/mission_profiles.yaml \
  --mission-id SKYROOT_M1 \
  --output models/

# Signed knowledge-base registry (re-run when mission config or KB docs change)
python build_registry.py
```

## Run

```bash
pytest tests/ -v

python run_pipeline.py \
  --telemetry data/telemetry.csv \
  --mission SKYROOT_M1 \
  --backend openai \
  --log-level WARNING
```

Mission HUD, in a second terminal while the pipeline runs:

```bash
streamlit run dashboard/app.py     # http://localhost:8501
```

Benchmarks against the legacy threshold baseline:

```bash
python -m scripts.simulation_harness --scenario fpr
```

## Layout

```
flare/          pipeline — ingestion, detection, state, llm, audit
dashboard/      Streamlit HUD (reads hot_storage JSON only, no flare.* imports)
scripts/        training data prep, benchmarks, evaluation
config/         mission profiles, schema builder
data/           telemetry.csv, knowledge base, signed registry
tests/          pytest
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The copyright line in the license appendix is the unmodified Apache
template. Replace `[yyyy] [name of copyright owner]` with the year and the
holder you want on record before publishing.
