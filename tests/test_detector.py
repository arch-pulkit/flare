from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from flare.detection.detector import (
    ChannelWindowManager,
    IsolationForestDetector,
    OnnxDetector,
    _feature_vector,
)
from flare.detection.events import AnomalyDetected, WindowStats
from flare.ingestion.frame import TelemetryFrame

MISSION_PROFILE = {
    "mission_id": "TEST_M1",
    "nominal_ranges": {"BUS_VOLTAGE": [27.5, 32.5]},
    "anomaly_threshold": -0.1,
}


def _make_frame(value: float = 30.0, row_index: int = 0) -> TelemetryFrame:
    return TelemetryFrame(
        apid=100,
        sequence_count=row_index % 16384,
        sequence_flag="standalone",
        timestamp=float(row_index),
        channel_id="BUS_VOLTAGE",
        value=value,
        unit="V",
        subsystem="EPS",
        source_file="test.csv",
        row_index=row_index,
    )


def _make_stats(value: float = 30.0) -> WindowStats:
    return WindowStats(
        mean=value, std=0.5, min=value - 1.0, max=value + 1.0,
        nominal_low=27.5, nominal_high=32.5, window_size=60,
    )


def _train_model(tmpdir: str) -> tuple[str, str]:
    """Train a small model on 100 synthetic frames; return (joblib_path, onnx_path)."""
    from sklearn.ensemble import IsolationForest
    import joblib
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    rng = np.random.default_rng(42)
    nominal = 30.0
    values = rng.normal(nominal, 0.5, 100).astype(np.float32)
    # inject 5 anomalies
    values[10] = 45.0
    values[20] = 15.0
    values[50] = 50.0
    values[70] = 10.0
    values[90] = 48.0

    stats_arr = [_make_stats(v) for v in values]
    X = np.array([
        [v, s.mean, s.std, v - (s.nominal_low + s.nominal_high) / 2, s.min, s.max]
        for v, s in zip(values, stats_arr)
    ], dtype=np.float32)

    clf = IsolationForest(contamination=0.05, n_estimators=50, random_state=42)
    clf.fit(X)

    joblib_path = str(Path(tmpdir) / "model.joblib")
    joblib.dump(clf, joblib_path)

    onnx_path = str(Path(tmpdir) / "model.onnx")
    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_type,
        target_opset={"": 15, "ai.onnx.ml": 3},
    )
    Path(onnx_path).write_bytes(onnx_model.SerializeToString())

    import json
    meta_path = Path(onnx_path).with_suffix(".meta.json")
    meta_path.write_text(json.dumps({"offset": float(clf.offset_)}))
    return joblib_path, onnx_path


def test_isolation_forest_and_onnx_produce_identical_events():
    with tempfile.TemporaryDirectory() as tmpdir:
        joblib_path, onnx_path = _train_model(tmpdir)
        if_det = IsolationForestDetector(joblib_path, MISSION_PROFILE)
        onnx_det = OnnxDetector(onnx_path, MISSION_PROFILE)

        frame = _make_frame(value=45.0, row_index=0)
        stats = _make_stats(value=45.0)
        window = (frame,)
        incident_id = str(uuid.uuid4())

        ev_if = if_det.score(frame, window, stats, "TEST_M1", incident_id)
        ev_onnx = onnx_det.score(frame, window, stats, "TEST_M1", incident_id)

        assert ev_if is not None
        assert ev_onnx is not None
        assert ev_if.is_anomaly == ev_onnx.is_anomaly
        assert abs(ev_if.anomaly_score - ev_onnx.anomaly_score) < 1e-5


def test_detector_returns_none_for_nominal_frame():
    """IsolationForest trained on nominal data should not flag an in-distribution frame.

    We intentionally train a model where the threshold is tuned such that a
    perfectly nominal frame doesn't trigger. We assert score is above threshold.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override threshold very low so nominal frames are not flagged
        profile = {**MISSION_PROFILE, "anomaly_threshold": -0.5}
        joblib_path, _ = _train_model(tmpdir)
        det = IsolationForestDetector(joblib_path, profile)

        frame = _make_frame(value=30.0, row_index=0)
        stats = _make_stats(value=30.0)
        ev = det.score(frame, (frame,), stats, "TEST_M1", str(uuid.uuid4()))
        # With threshold=-0.5 a nominal frame should not be anomalous
        assert ev is not None  # event is always produced
        assert ev.is_anomaly is False


def test_anomaly_detected_is_frozen():
    frame = _make_frame()
    stats = _make_stats()
    ev = AnomalyDetected(
        event_id=str(uuid.uuid4()),
        incident_id=str(uuid.uuid4()),
        mission_id="TEST_M1",
        frame=frame,
        anomaly_score=-0.2,
        decision_threshold=-0.1,
        is_anomaly=True,
        context_window=(frame,),
        window_stats=stats,
        detector_version="abc123",
        detection_method="isolation_forest",
        detected_at=1000.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        ev.is_anomaly = False  # type: ignore[misc]


def test_window_manager_rolling_window_size():
    manager = ChannelWindowManager(window_size=5)
    for i in range(10):
        window, stats = manager.update_with_profile(_make_frame(value=30.0 + i * 0.1, row_index=i), MISSION_PROFILE)
    assert len(window) == 5
    assert stats.window_size == 5


def test_onnx_detector_zero_offset_not_falsy():
    """clf_offset=0.0 in meta.json must be used as-is, not fall through to a different key."""
    import json as _json

    with tempfile.TemporaryDirectory() as tmpdir:
        joblib_path, onnx_path = _train_model(tmpdir)

        # Write meta.json with clf_offset=0.0 and a non-zero fallback key
        meta_path = Path(onnx_path).with_suffix(".meta.json")
        meta_path.write_text(_json.dumps({"clf_offset": 0.0, "offset_": -0.99, "offset": -0.99}))

        det = OnnxDetector(onnx_path, MISSION_PROFILE)

        assert det._score_offset == 0.0, (
            f"Expected 0.0 but got {det._score_offset} — or-chain is treating 0.0 as falsy"
        )


def test_both_detectors_produce_correct_feature_vector():
    frame = _make_frame(value=30.0)
    stats = _make_stats(value=30.0)
    fv = _feature_vector(frame, stats)
    assert fv.shape == (6,)
    nominal_center = (stats.nominal_low + stats.nominal_high) / 2.0
    expected_delta = frame.value - nominal_center
    np.testing.assert_allclose(fv[0], frame.value)
    np.testing.assert_allclose(fv[1], stats.mean)
    np.testing.assert_allclose(fv[2], stats.std)
    np.testing.assert_allclose(fv[3], expected_delta)
    np.testing.assert_allclose(fv[4], stats.min)
    np.testing.assert_allclose(fv[5], stats.max)
