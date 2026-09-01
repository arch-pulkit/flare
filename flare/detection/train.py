# BOUNDARY: flare/detection/train.py → imports from flare.ingestion and flare.detection only
#no log, no event, one time setup script.
#train once, deploy everywhere pattern.
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _build_feature_matrix(
    csv_path: str,
    mission_profile: dict[str, Any],
    window_size: int = 60,
) -> np.ndarray:
    from flare.detection.detector import ChannelWindowManager
    from flare.ingestion.reader import _CHANNEL_MAP, SMAPReader

    # Must be the same map the pipeline serves with. A local copy that drifts would
    # train the model on differently-labeled channels than it scores at inference,
    # silently corrupting the delta_from_nominal feature (looked up by channel_id).
    reader = SMAPReader(csv_path, mission_profile, _CHANNEL_MAP)
    manager = ChannelWindowManager(window_size=window_size)

    rows: list[list[float]] = []
    for frame in reader.stream():
        _, stats = manager.update_with_profile(frame, mission_profile)
        nominal_center = (stats.nominal_low + stats.nominal_high) / 2.0
        delta = frame.value - nominal_center
        rows.append([frame.value, stats.mean, stats.std, delta, stats.min, stats.max])

    return np.array(rows, dtype=np.float32)

#run once before the pipeline starts, reads all frames from SMAPReader
#Builds the
def train_and_export(
    csv_path: str,
    mission_profile: dict[str, Any],
    output_dir: str,
    contamination: float = 0.1,
    n_estimators: int = 100,
) -> dict[str, str]:
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Building feature matrix from %s", csv_path)
    X = _build_feature_matrix(csv_path, mission_profile)
    logger.info("Feature matrix shape: %s", X.shape)


#fits IsolationForest
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        ))
    ])
    model.fit(X)
    logger.info("Trained IsolationForest: %d estimators", n_estimators)

    #saves model with joblib
    joblib_path = output_path / "isolation_forest.joblib"
    joblib.dump(model, joblib_path)
    logger.info("Saved joblib model: %s", joblib_path)

    # ONNX export — ai.onnx.ml capped at v3 by skl2onnx
    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset={"": 15, "ai.onnx.ml": 3},
    )

    #exports to onnx
    onnx_path = output_path / "isolation_forest.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())
    logger.info("Saved ONNX model: %s", onnx_path)

    # skl2onnx ≥1.17 outputs decision_function; store offset_ so OnnxDetector
    # can recover score_samples = onnx_raw + offset_
    import json
    meta_path = output_path / "isolation_forest.meta.json"
    meta_path.write_text(json.dumps({"clf_offset": float(model.named_steps["detector"].offset_)}))

    #runs one forward pass through InferenceSession to verify
    # Verify ONNX export — mandatory forward pass before declaring done
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    logger.info("ONNX input: %s | outputs: %s", input_name, output_names)

    sample = X[:1]
    onnx_out = session.run(None, {input_name: sample})
    assert len(onnx_out) >= 2, f"Expected >=2 outputs, got {len(onnx_out)}"
    # skl2onnx ≥1.17 outputs decision_function; add offset_ to recover score_samples
    raw_onnx_score = float(onnx_out[1].flat[0]) + model.named_steps["detector"].offset_
    sklearn_score = float(model.score_samples(sample)[0])
    diff = abs(raw_onnx_score - sklearn_score)
    logger.info(
        "ONNX verification — sklearn: %.6f  onnx(adj): %.6f  diff: %.2e",
        sklearn_score, raw_onnx_score, diff,
    )
    if diff > 1e-3:
        logger.warning("ONNX/sklearn score diff %.2e exceeds 1e-3 — check export", diff)

    model_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:12]
    return {
        "joblib_path": str(joblib_path),
        "onnx_path": str(onnx_path),
        "model_hash": model_hash,
    }


def _load_profile(yaml_path: str, mission_id: str) -> dict[str, Any]:
    with open(yaml_path) as f:
        all_profiles = yaml.safe_load(f)
    if mission_id not in all_profiles:
        raise ValueError(f"Mission ID {mission_id!r} not in {yaml_path}")
    return all_profiles[mission_id]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train FLARE anomaly detector")
    parser.add_argument("--csv", required=True, help="Path to telemetry CSV")
    parser.add_argument("--mission", required=True, help="Path to mission_profiles.yaml")
    parser.add_argument("--mission-id", required=True, help="Mission ID key")
    parser.add_argument("--output", required=True, help="Output directory for models")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    profile = _load_profile(args.mission, args.mission_id)
    result = train_and_export(
        csv_path=args.csv,
        mission_profile=profile,
        output_dir=args.output,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
    )
    print("joblib:", result["joblib_path"])
    print("onnx:  ", result["onnx_path"])
    print("hash:  ", result["model_hash"])
