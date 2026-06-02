"""
Multi-Modal Sensor Fusion
Combines radar, PIR, and CSI into a unified scene state.
Applies confidence weighting and cross-modal validation.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

log = logging.getLogger(__name__)


@dataclass
class SceneTarget:
    """A fused, confirmed target in the scene."""
    target_id: str
    x_m: float
    y_m: float
    range_m: float
    az_deg: float
    velocity_mps: float = 0.0

    target_class: str = "unknown"    # person | vehicle | drone | unknown
    activity: str = "unknown"
    activity_display: str = ""
    activity_confidence: float = 0.0
    activity_is_alert: bool = False
    person_id: str = "unknown"
    person_similarity: float = 0.0
    class_confidence: float = 0.0

    # Vital signs
    breath_rate_bpm: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    vitals_status: str = "unknown"
    vitals_confidence: float = 0.0

    # Track metadata
    confirmed_by: List[str] = field(default_factory=list)  # which sensors
    dwell_time_s: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    history: List[tuple] = field(default_factory=list)

    # Alerts
    is_alert: bool = False
    alert_reason: str = ""


@dataclass
class SceneState:
    """Complete fused picture of the monitored space at a moment in time."""
    timestamp: float
    targets: List[SceneTarget]
    environment_map: dict
    anomaly_events: list
    pir_detected: bool
    total_count: int
    person_count: int
    vehicle_count: int
    drone_count: int
    alerts: List[str]
    radar_image: Optional[np.ndarray] = None     # 2D spatial image for display
    x_axis_m: Optional[np.ndarray] = None
    y_axis_m: Optional[np.ndarray] = None


class SensorFusion:
    """
    Fuses outputs from all sensors into a single coherent SceneState.
    Applies confidence weighting: radar primary, PIR confirmation, CSI context.
    """

    def __init__(self):
        self._last_pir_time = 0.0
        self._alert_log: List[dict] = []

    def fuse(self,
             radar_tracks: list,
             pir_detected: bool,
             fingerprint_anomalies: list,
             vitals_map: dict,
             radar_image=None) -> SceneState:
        """
        radar_tracks: confirmed Track objects from KalmanTracker
        pir_detected: bool — PIR currently triggered
        fingerprint_anomalies: list of AnomalyEvent from RFFingerprinter
        vitals_map: {track_id: VitalSigns} from VitalSignsExtractor
        radar_image: SpatialImage object
        """
        now = time.time()
        targets = []
        alerts = []

        for track in radar_tracks:
            x, y = float(track.state[0]), float(track.state[1])
            vx, vy = float(track.state[2]), float(track.state[3])
            velocity = float(np.sqrt(vx**2 + vy**2))
            range_m = float(np.sqrt(x**2 + y**2))
            az_deg = float(np.degrees(np.arctan2(x, y)))

            confirmed_by = ["radar"]

            # PIR confirmation — if PIR triggered and target is within 10m
            if pir_detected and range_m < 10.0:
                confirmed_by.append("pir")

            # Vitals from extractor
            vitals = vitals_map.get(track.track_id)
            breath_rate = vitals.breath_rate_bpm if vitals else None
            heart_rate = vitals.heart_rate_bpm if vitals else None
            vitals_status = vitals.status if vitals else "unknown"
            vitals_conf = vitals.breath_confidence if vitals else 0.0

            # Compute dwell time
            dwell = now - track.first_seen

            # Build fused target
            st = SceneTarget(
                target_id=track.track_id,
                x_m=round(x, 2),
                y_m=round(y, 2),
                range_m=round(range_m, 2),
                az_deg=round(az_deg, 1),
                velocity_mps=round(velocity, 2),
                target_class=track.target_class,
                class_confidence=round(track.class_confidence, 2),
                breath_rate_bpm=breath_rate,
                heart_rate_bpm=heart_rate,
                vitals_status=vitals_status,
                vitals_confidence=round(vitals_conf, 2),
                confirmed_by=confirmed_by,
                dwell_time_s=round(dwell, 1),
                first_seen=track.first_seen,
                last_seen=now,
                history=track.history[-20:]
            )

            # Alert logic
            alert_reason = self._check_alerts(st, fingerprint_anomalies)
            if alert_reason:
                st.is_alert = True
                st.alert_reason = alert_reason
                alerts.append(f"{track.track_id}: {alert_reason}")

            targets.append(st)

        # Fingerprint-only anomalies (no radar track — e.g. distant vehicle)
        for evt in fingerprint_anomalies:
            if not any(t.target_id == f"fp_{evt.zone_id}" for t in targets):
                desc = evt.description
                if "vehicle" in evt.anomaly_type:
                    alerts.append(f"RF fingerprint: {desc} in zone {evt.zone_id}")

        # Counts
        person_count  = sum(1 for t in targets if t.target_class == "person")
        vehicle_count = sum(1 for t in targets if t.target_class == "vehicle")
        drone_count   = sum(1 for t in targets if t.target_class == "drone")

        # Build environment map summary
        env_map = {}
        for evt in fingerprint_anomalies:
            env_map[evt.zone_id] = {
                "type": evt.anomaly_type,
                "score": evt.anomaly_score,
                "description": evt.description
            }

        state = SceneState(
            timestamp=now,
            targets=targets,
            environment_map=env_map,
            anomaly_events=[{
                "zone": e.zone_id, "type": e.anomaly_type,
                "score": round(e.anomaly_score, 3), "desc": e.description
            } for e in fingerprint_anomalies],
            pir_detected=pir_detected,
            total_count=len(targets),
            person_count=person_count,
            vehicle_count=vehicle_count,
            drone_count=drone_count,
            alerts=alerts,
            radar_image=radar_image.power_db if radar_image else None,
            x_axis_m=radar_image.x_axis_m if radar_image else None,
            y_axis_m=radar_image.y_axis_m if radar_image else None,
        )

        return state

    def _check_alerts(self, target: SceneTarget, anomalies: list) -> str:
        """Return alert string if this target triggers any alert condition."""

        # Possible casualty — present but not breathing
        if target.vitals_status == "possible_casualty" and target.vitals_confidence > 0.3:
            return "CASUALTY — no breathing detected"

        # Unknown/unclassified target at close range
        if target.target_class == "unknown" and target.range_m < 5.0:
            return "Unidentified target at close range"

        # Drone
        if target.target_class == "drone" and target.class_confidence > 0.6:
            return f"Drone detected at {target.range_m}m"

        # Fast-moving target
        if target.velocity_mps > 8.0:
            return f"Fast-moving target {target.velocity_mps:.1f} m/s"

        return ""

    def serialize_state(self, state: SceneState) -> dict:
        """Convert SceneState to JSON-serialisable dict for WebSocket."""
        return {
            "timestamp": state.timestamp,
            "counts": {
                "total": state.total_count,
                "person": state.person_count,
                "vehicle": state.vehicle_count,
                "drone": state.drone_count
            },
            "pir": state.pir_detected,
            "alerts": state.alerts,
            "targets": [
                {
                    "id": t.target_id,
                    "x_m": t.x_m,
                    "y_m": t.y_m,
                    "range_m": t.range_m,
                    "az_deg": t.az_deg,
                    "velocity_mps": t.velocity_mps,
                    "class": t.target_class,
                    "activity": t.activity,
                    "activity_display": t.activity_display,
                    "activity_conf": t.activity_confidence,
                    "activity_alert": t.activity_is_alert,
                    "person_id": t.person_id,
                    "class_conf": t.class_confidence,
                    "breath_bpm": t.breath_rate_bpm,
                    "hr_bpm": t.heart_rate_bpm,
                    "vitals_status": t.vitals_status,
                    "vitals_conf": t.vitals_confidence,
                    "confirmed_by": t.confirmed_by,
                    "dwell_s": t.dwell_time_s,
                    "is_alert": t.is_alert,
                    "alert": t.alert_reason,
                    "history": t.history[-10:]
                }
                for t in state.targets
            ],
            "environment": state.environment_map,
            "anomalies": state.anomaly_events,
            "radar_image": state.radar_image.tolist() if state.radar_image is not None else None,
            "image_axes": {
                "x_m": state.x_axis_m.tolist() if state.x_axis_m is not None else [],
                "y_m": state.y_axis_m.tolist() if state.y_axis_m is not None else []
            }
        }
