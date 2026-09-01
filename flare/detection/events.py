# BOUNDARY: flare/detection/events.py → imports ONLY from flare.ingestion
#Everythng downsteram starts from thsis one event. 
from __future__ import annotations

from dataclasses import dataclass

from flare.ingestion.frame import TelemetryFrame

#two forzen dataclasses live here.
#stats over a rolling window(value object)

#pre-computer stats over the last N frames for one channel
#computer by CHannelWindowManager, sealed, passed into Anomaly Detector
@dataclass(frozen=True)
class WindowStats:
    mean: float
    std: float
    min: float
    max: float
    nominal_low: float
    nominal_high: float
    window_size: int


#first domain event in the system, created by the detector when a frame scores below the threslhold
#it is immutable, carries everything downstream handlers need to know about the anomaly. 
@dataclass(frozen=True)
class AnomalyDetected:
    event_id: str       #uuid4 - unique ID for this event
    incident_id: str    #uuid4 - stable across the incident lifetime
    mission_id: str
    frame: TelemetryFrame       #the TelemetryFrame that trigerred this
    anomaly_score: float        #raw IF score
    decision_threshold: float
    is_anomaly: bool
    context_window: tuple[TelemetryFrame, ...]  #tuple of last N TelemetryFrames
    window_stats: WindowStats       #WindowStats
    detector_version: str   #sha256[:12] of model file
    detection_method: str
    detected_at: float
