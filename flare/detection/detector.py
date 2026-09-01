# BOUNDARY: flare/detection/detector.py → imports ONLY from flare.ingestion and flare.detection
#asks is this weird
#This is where AnomalyDetected is first created, rasied for the first time.
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from flare.detection.events import AnomalyDetected, WindowStats
from flare.ingestion.frame import TelemetryFrame

logger = logging.getLogger(__name__)


def _model_hash(model_path: str) -> str:
    return hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:12]


def _feature_vector(
    frame: TelemetryFrame,
    stats: WindowStats,
) -> np.ndarray:
    nominal_center = (stats.nominal_low + stats.nominal_high) / 2.0
    delta = frame.value - nominal_center
    return np.array(
        [frame.value, stats.mean, stats.std, delta, stats.min, stats.max],
        dtype=np.float32,
    )


#Abstract base. define one methods: score(). Both implementations muse produce identical output shapes.
class BaseDetector(ABC):
    @abstractmethod
    def score(
        self,
        frame: TelemetryFrame,
        context_window: tuple[TelemetryFrame, ...],
        window_stats: WindowStats,
        mission_id: str,
        incident_id: str,
    ) -> AnomalyDetected:
        raise NotImplementedError


#loads a joblib model. acts on score().
#receives frame + context_window + window_stats
class IsolationForestDetector(BaseDetector):
    def __init__(self, model_path: str, mission_profile: dict[str, Any]) -> None:
        import joblib

        self._model_path = model_path
        self._mission_profile = mission_profile
        self._model = joblib.load(model_path)
        self._version = _model_hash(model_path)
        self._threshold: float = mission_profile.get("anomaly_threshold", -0.1)

    def score(
        self,
        frame: TelemetryFrame,
        context_window: tuple[TelemetryFrame, ...],
        window_stats: WindowStats,
        mission_id: str,
        incident_id: str,
    ) -> AnomalyDetected:
        fv = _feature_vector(frame, window_stats).reshape(1, -1)    #builds feature vector
        raw_score: float = float(self._model.score_samples(fv)[0])
        is_anomaly = raw_score < self._threshold
        #if score < threshold creates and returns AnomalyDetected, else returns None
        return AnomalyDetected(
            event_id=str(uuid.uuid4()),
            incident_id=incident_id,
            mission_id=mission_id,
            frame=frame,
            anomaly_score=raw_score,
            decision_threshold=self._threshold,
            is_anomaly=is_anomaly,
            context_window=context_window,
            window_stats=window_stats,
            detector_version=self._version,
            detection_method="isolation_forest",
            detected_at=time.time(),
        )

#loads an ONNX model via InferenceSession, same feature vector, same logic
#returns identical AnomalyDetected shape
class OnnxDetector(BaseDetector):
    def __init__(self, model_path: str, mission_profile: dict[str, Any]) -> None:
        import json
        import onnxruntime as ort

        self._model_path = model_path
        self._mission_profile = mission_profile
        self._version = _model_hash(model_path)
        self._threshold: float = mission_profile.get("anomaly_threshold", -0.1)

        # skl2onnx ≥1.17 exports decision_function, not score_samples.
        # score_samples = onnx_raw + clf.offset_  (stored by train_and_export)
        # Fall back to -0.5 (typical contamination=0.1 offset) if file absent.
        meta_path = Path(model_path).with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            for key in ("clf_offset", "offset_", "offset"):
                value = meta.get(key)
                if value is not None:
                    self._score_offset: float = value
                    break
            else:
                self._score_offset = 0.0
                logger.warning(
                    "No recognized offset key in meta.json — using 0.0"
                )
        else:
            logger.warning("No .meta.json found for %s — using offset=-0.5", model_path)
            self._score_offset = -0.5

        providers = []
        available = ort.get_available_providers()
        for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"):
            if p in available:
                providers.append(p)
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name: str = self._session.get_inputs()[0].name
        self._score_output_name: str = self._session.get_outputs()[1].name

    def score(
        self,
        frame: TelemetryFrame,
        context_window: tuple[TelemetryFrame, ...],
        window_stats: WindowStats,
        mission_id: str,
        incident_id: str,
    ) -> AnomalyDetected:
        fv = _feature_vector(frame, window_stats).reshape(1, -1)
        outputs = self._session.run([self._score_output_name], {self._input_name: fv})
        # skl2onnx outputs decision_function; add stored offset_ to recover score_samples
        raw_score: float = float(outputs[0].flat[0]) + self._score_offset
        is_anomaly = raw_score < self._threshold
        return AnomalyDetected(
            event_id=str(uuid.uuid4()),
            incident_id=incident_id,
            mission_id=mission_id,
            frame=frame,
            anomaly_score=raw_score,
            decision_threshold=self._threshold,
            is_anomaly=is_anomaly,
            context_window=context_window,
            window_stats=window_stats,
            detector_version=self._version,
            detection_method="onnx_isolation_forest",
            detected_at=time.time(),
        )


#maintains a rolling window of the last 60 frames per channel
#everytime a new frame arrives, it updates the window for that channel and computers fresh WindowStats.
class ChannelWindowManager:
    def __init__(self, window_size: int = 60) -> None:
        self._window_size = window_size
        self._windows: dict[str, deque[TelemetryFrame]] = {}

    def update_with_profile(
        self,
        frame: TelemetryFrame,
        mission_profile: dict[str, Any],
    ) -> tuple[tuple[TelemetryFrame, ...], WindowStats]:
        """Appends frame to the rolling window and returns (window, stats) with nominal ranges."""
        channel = frame.channel_id
        if channel not in self._windows:
            self._windows[channel] = deque(maxlen=self._window_size)
        self._windows[channel].append(frame)
        window = tuple(self._windows[channel])
        values = np.array([f.value for f in window], dtype=np.float64)
        ranges = mission_profile.get("nominal_ranges", {}).get(channel, [0.0, 0.0])
        return window, WindowStats(
            mean=float(np.mean(values)),
            std=float(np.std(values)) if len(values) > 1 else 0.0,
            min=float(np.min(values)),
            max=float(np.max(values)),
            nominal_low=float(ranges[0]),
            nominal_high=float(ranges[1]),
            window_size=len(window),
        )
