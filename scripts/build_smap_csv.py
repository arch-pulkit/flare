# BOUNDARY: scripts/ — stdlib + numpy + flare.ingestion only
# One-time data prep script — wraps flare.ingestion.reader.preprocess_smap_npy().
# Run once to regenerate data/telemetry.csv from raw NASA SMAP .npy files.
# Not part of the live pipeline; historical after data/telemetry.csv is committed.
"""
Build data/telemetry.csv from SMAP-structured .npy files.

Primary intent: use real NASA SMAP .npy files from data/train/ and data/test/.
Real files have shape (N, 25); column 0 is the telemetry channel value.
Columns 1-24 are contextual features used by Telemanom's LSTM — not needed by
FLARE's IsolationForest. Column 0 is normalized to [-1, 1] by the Hundman et al.
pipeline; this script re-scales it to physical units using
  physical = nominal_center + normalized * nominal_half
from config/mission_profiles.yaml SKYROOT_M1 nominal ranges.

Fallback: T-1 (THRUSTER_TEMP) and T-2 (TANK_PRESSURE) have no T-prefix counterpart
in the SMAP dataset and are always generated with AR(1) autocorrelated noise scaled
to FLARE physical unit ranges.

Run: python -m scripts.build_smap_csv [--smap-dir data] [--output data/telemetry.csv]
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Channel map: SMAP file stem → FLARE channel_id
CHANNEL_MAP: dict[str, str] = {
    "P-1": "BUS_VOLTAGE",
    "P-2": "SOLAR_CURRENT",
    "T-1": "THRUSTER_TEMP",
    "T-2": "TANK_PRESSURE",
    "A-1": "REACTION_WHEEL_RPM",
    "S-1": "TX_FREQUENCY",
}

# Real NASA SMAP channels present in data/train/ and data/test/
# T-1 and T-2 have no T-prefix counterpart in SMAP — AR(1) fallback only
_REAL_SMAP_CHANNELS: frozenset[str] = frozenset({"A-1", "P-1", "P-2", "S-1"})

# Per-channel generation spec.
# Nominal ranges from config/mission_profiles.yaml SKYROOT_M1.
# Anomaly positions for P-1, P-2, A-1, S-1 are the exact sequences from
# labeled_anomalies.csv (Hundman et al. 2018). T-1 and T-2 are synthetic
# (no T-prefix channels in SMAP dataset) with similar anomaly density.
#
# n_train: nominal training frames (no anomalies — IsolationForest learns normal)
# n_test:  test frames including labeled anomaly windows
# nominal_center / nominal_half: center and half-width of FLARE nominal range
# noise_sigma: AR(1) innovation std (stationary std ≈ noise_sigma / sqrt(1 - rho^2))
# rho: AR(1) autocorrelation coefficient
# anomaly_windows: list of [start, end] indices in the TEST array
# anomaly_type: "point" (spike, values 30% outside nominal) |
#               "contextual" (sustained drift, 8% outside nominal)
_CHANNEL_SPECS: dict[str, dict] = {
    "P-1": dict(
        n_train=12000, n_test=8505,
        nominal_center=30.0, nominal_half=2.5, noise_sigma=0.07, rho=0.97,
        anomaly_windows=[[2149, 2349], [3539, 3779], [4536, 4844]],
        anomaly_type="contextual",
    ),
    "P-2": dict(
        n_train=11500, n_test=8209,
        nominal_center=5.0, nominal_half=3.0, noise_sigma=0.08, rho=0.96,
        anomaly_windows=[[5350, 6575]],
        anomaly_type="point",
    ),
    # T-1 / T-2 have no SMAP counterpart; generated with realistic anomaly density
    "T-1": dict(
        n_train=11800, n_test=8400,
        nominal_center=330.0, nominal_half=50.0, noise_sigma=1.2, rho=0.97,
        anomaly_windows=[[3500, 3620]],
        anomaly_type="point",
    ),
    "T-2": dict(
        n_train=11600, n_test=8300,
        nominal_center=275000.0, nominal_half=75000.0, noise_sigma=1800.0, rho=0.97,
        anomaly_windows=[[4000, 4080]],
        anomaly_type="point",
    ),
    "A-1": dict(
        n_train=12000, n_test=8640,
        nominal_center=3500.0, nominal_half=2500.0, noise_sigma=45.0, rho=0.95,
        anomaly_windows=[[4690, 4774]],
        anomaly_type="point",
    ),
    "S-1": dict(
        n_train=10500, n_test=7331,
        nominal_center=437050000.0, nominal_half=50000.0, noise_sigma=1100.0, rho=0.96,
        anomaly_windows=[[5300, 5747]],
        anomaly_type="point",
    ),
}


def _ar1_sequence(n: int, center: float, noise_sigma: float, rho: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Generate an AR(1) stationary sequence around `center`."""
    # Innovation std so stationary variance equals noise_sigma^2
    innov_std = noise_sigma * np.sqrt(1 - rho ** 2)
    x = np.zeros(n, dtype=np.float64)
    x[0] = center
    for t in range(1, n):
        x[t] = center + rho * (x[t - 1] - center) + rng.normal(0, innov_std)
    return x


def _generate_channel_npy(stem: str, spec: dict, out_train: Path, out_test: Path,
                           rng: np.random.Generator) -> None:
    """Generate AR(1) train and test .npy files for channels with no SMAP counterpart."""
    c = spec["nominal_center"]
    h = spec["nominal_half"]
    sig = spec["noise_sigma"]
    rho = spec["rho"]

    # Training array: purely nominal, shape (n_train, 1)
    train_vals = _ar1_sequence(spec["n_train"], c, sig, rho, rng)
    train_arr = train_vals.reshape(-1, 1).astype(np.float32)
    np.save(out_train / f"{stem}.npy", train_arr)
    logger.info("  train/%s.npy  shape=%s  mean=%.3f  std=%.4f",
                stem, train_arr.shape, float(train_arr.mean()), float(train_arr.std()))

    # Test array: nominal baseline + anomaly injection, shape (n_test, 1)
    test_vals = _ar1_sequence(spec["n_test"], c, sig, rho, rng)

    for window in spec["anomaly_windows"]:
        lo, hi = window[0], window[1]
        hi = min(hi, spec["n_test"] - 1)
        if spec["anomaly_type"] == "point":
            # Spike values 30–40% outside the nominal upper bound
            anomaly_center = c + h * 1.35
            anomaly_sig = sig * 1.5
            test_vals[lo:hi] = _ar1_sequence(hi - lo, anomaly_center, anomaly_sig, rho, rng)
        else:
            # Contextual: sustained drift 8–12% outside nominal upper bound
            anomaly_center = c + h * 1.10
            anomaly_sig = sig * 1.2
            test_vals[lo:hi] = _ar1_sequence(hi - lo, anomaly_center, anomaly_sig, rho, rng)

    test_arr = test_vals.reshape(-1, 1).astype(np.float32)
    anomaly_frames = sum(w[1] - w[0] for w in spec["anomaly_windows"])
    anomaly_pct = 100.0 * anomaly_frames / spec["n_test"]
    np.save(out_test / f"{stem}.npy", test_arr)
    logger.info("  test/%s.npy   shape=%s  anomaly_frames=%d (%.1f%%)",
                stem, test_arr.shape, anomaly_frames, anomaly_pct)


def _process_real_channel(
    stem: str,
    spec: dict,
    train_npy: Path,
    test_npy: Path,
    out_train: Path,
    out_test: Path,
    rng: np.random.Generator,
) -> None:
    """Load real NASA SMAP .npy, extract column 0, scale to FLARE physical units.

    Real SMAP files have shape (N, 25). Column 0 is the telemetry channel value
    normalized to [-1, 1] by the Hundman et al. pipeline. We re-scale using:
        physical = nominal_center + normalized * nominal_half
    so that the training distribution maps to the FLARE nominal range and
    anomalous values (outside training distribution) fall outside that range.
    Falls back to AR(1) if the real file is missing.
    """
    center = spec["nominal_center"]
    half = spec["nominal_half"]

    for src, out, split_name, n_fallback in [
        (train_npy, out_train, "train", spec["n_train"]),
        (test_npy, out_test, "test", spec["n_test"]),
    ]:
        if src.exists():
            raw = np.load(src)
            # Real SMAP: extract channel value (column 0 only)
            # Columns 1-24 are contextual features used by
            # Telemanom's LSTM but not needed by FLARE's IF
            if raw.ndim == 2 and raw.shape[1] > 1:
                raw = raw[:, 0]
            # Scale normalized [-1, 1] → physical units
            scaled = (center + raw.astype(np.float64) * half).astype(np.float32).reshape(-1, 1)
            np.save(out / f"{stem}.npy", scaled)
            logger.info(
                "  %s/%s.npy  shape=%s  mean=%.3f  std=%.4f  [real SMAP, col0 scaled]",
                split_name, stem, scaled.shape,
                float(scaled.mean()), float(scaled.std()),
            )
        else:
            logger.warning(
                "Real file %s not found — AR(1) fallback for %s/%s", src, split_name, stem
            )
            vals = _ar1_sequence(n_fallback, center, spec["noise_sigma"], spec["rho"], rng)
            np.save(out / f"{stem}.npy", vals.reshape(-1, 1).astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FLARE telemetry CSV from SMAP data")
    parser.add_argument("--smap-dir", default="data",
                        help="Directory containing train/ and test/ .npy subdirs")
    parser.add_argument("--output", default="data/telemetry.csv",
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for AR(1) fallback generation (T-1, T-2)")
    args = parser.parse_args()

    smap_path = Path(args.smap_dir)
    train_dir = smap_path / "train"
    test_dir = smap_path / "test"

    # Use a temp directory so preprocess_smap_npy sees only the 6 FLARE channels.
    # data/train/ contains 54+ SMAP channels; we extract and scale the 6 we need.
    tmp_dir = Path("data/smap_tmp")
    tmp_train = tmp_dir / "train"
    tmp_test = tmp_dir / "test"
    tmp_train.mkdir(parents=True, exist_ok=True)
    tmp_test.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Real SMAP channels: extract col 0, scale normalized [-1,1] → physical units
    for stem in sorted(_REAL_SMAP_CHANNELS):
        spec = _CHANNEL_SPECS[stem]
        logger.info("Channel %s → %s [real SMAP]", stem, CHANNEL_MAP[stem])
        _process_real_channel(
            stem, spec,
            train_dir / f"{stem}.npy",
            test_dir / f"{stem}.npy",
            tmp_train, tmp_test,
            rng,
        )

    # T-1 / T-2: no SMAP counterpart — AR(1) fallback
    for stem in ("T-1", "T-2"):
        spec = _CHANNEL_SPECS[stem]
        logger.info("Channel %s → %s [AR(1) fallback — no SMAP counterpart]",
                    stem, CHANNEL_MAP[stem])
        _generate_channel_npy(stem, spec, tmp_train, tmp_test, rng)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from flare.ingestion.reader import preprocess_smap_npy

    logger.info("Running preprocess_smap_npy: %s → %s", tmp_dir, args.output)
    preprocess_smap_npy(
        smap_dir=str(tmp_dir),
        output_csv=args.output,
        channel_map=CHANNEL_MAP,
    )

    shutil.rmtree(tmp_dir)
    logger.info("Cleaned up temp dir: %s", tmp_dir)

    import pandas as pd
    df = pd.read_csv(args.output)
    logger.info("Output CSV: %d rows, channels=%s", len(df), sorted(df["channel_id"].unique()))
    for ch in sorted(df["channel_id"].unique()):
        sub = df[df["channel_id"] == ch]["value"]
        logger.info("  %-8s  n=%6d  mean=%14.3f  std=%12.3f  min=%14.3f  max=%14.3f",
                    ch, len(sub), sub.mean(), sub.std(), sub.min(), sub.max())


if __name__ == "__main__":
    main()
