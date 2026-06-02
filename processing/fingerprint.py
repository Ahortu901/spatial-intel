"""
RF Fingerprinting + Environment Mapping
Detects new objects, vehicles, and changes to the environment
by comparing live CSI/radar channel state to a stored baseline.
"""

import numpy as np
import json
import time
import os
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import mahalanobis

log = logging.getLogger(__name__)

from config.settings import (
    FINGERPRINT_BASELINE_SEC, FINGERPRINT_ANOMALY_THRESH,
    FINGERPRINT_UPDATE_RATE, UPDATE_RATE_HZ
)


@dataclass
class ZoneFingerprint:
    zone_id: str
    mean: np.ndarray              # mean CSI amplitude per subcarrier
    covariance: np.ndarray        # covariance matrix
    variance: np.ndarray          # per-subcarrier variance
    sample_count: int = 0
    created_at: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class AnomalyEvent:
    timestamp: float
    zone_id: str
    anomaly_score: float
    anomaly_type: str             # "new_object" | "person" | "vehicle" | "drone" | "unknown"
    change_magnitude: float
    affected_subcarriers: List[int] = field(default_factory=list)
    estimated_range_m: Optional[float] = None
    description: str = ""


class RFFingerprinter:
    """
    Builds and maintains RF fingerprints for monitored zones.
    Detects environmental changes by comparing live CSI to baseline.
    Classifies the type of change (person, vehicle, new object).
    """

    def __init__(self, baseline_path: str = "data/fingerprints"):
        self.baseline_path = baseline_path
        os.makedirs(baseline_path, exist_ok=True)

        self._zones: Dict[str, ZoneFingerprint] = {}
        self._active_zone = "default"
        self._calibrating = False
        self._cal_buffer: List[np.ndarray] = []
        self._cal_target_samples = int(FINGERPRINT_BASELINE_SEC * UPDATE_RATE_HZ)

        # Live anomaly state
        self._anomaly_history: List[AnomalyEvent] = []
        self._current_anomaly_score: Dict[str, float] = {}
        self._environment_map: Dict[str, dict] = {}
        self._lock = threading.Lock()

        # Load any saved fingerprints
        self._load_saved_fingerprints()

    # ── Calibration ──────────────────────────────────────────────────────────

    def start_calibration(self, zone_id: str = "default", description: str = ""):
        """Begin collecting baseline for a zone. Run in empty/normal environment."""
        log.info(f"Starting calibration for zone '{zone_id}' — collecting {FINGERPRINT_BASELINE_SEC}s of baseline")
        self._active_zone = zone_id
        self._cal_buffer = []
        self._calibrating = True
        self._cal_description = description

    @property
    def calibration_progress(self) -> float:
        """0.0 – 1.0 progress through calibration."""
        return min(1.0, len(self._cal_buffer) / self._cal_target_samples)

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    def _finish_calibration(self):
        data = np.array(self._cal_buffer)   # [N x subcarriers]
        mean = np.mean(data, axis=0)
        variance = np.var(data, axis=0)

        # Regularised covariance matrix (add small diagonal for invertibility)
        if data.shape[0] > data.shape[1]:
            cov = np.cov(data.T)
            cov += np.eye(cov.shape[0]) * 1e-6
        else:
            # Too few samples for full covariance — use diagonal
            cov = np.diag(variance + 1e-6)

        fp = ZoneFingerprint(
            zone_id=self._active_zone,
            mean=mean,
            covariance=cov,
            variance=variance,
            sample_count=len(data),
            description=self._cal_description
        )
        with self._lock:
            self._zones[self._active_zone] = fp

        self._save_fingerprint(fp)
        self._calibrating = False
        log.info(f"Calibration complete for zone '{self._active_zone}' — {len(data)} samples")

    # ── Runtime comparison ────────────────────────────────────────────────────

    def process_sample(self, csi_amplitude: np.ndarray,
                       zone_id: Optional[str] = None) -> Optional[AnomalyEvent]:
        """
        Feed a live CSI amplitude vector.
        Returns AnomalyEvent if a significant change is detected, else None.

        csi_amplitude: 1D array of per-subcarrier amplitudes (e.g. 52 or 256 values)
        """
        zone_id = zone_id or self._active_zone

        # During calibration — accumulate
        if self._calibrating and zone_id == self._active_zone:
            self._cal_buffer.append(csi_amplitude.copy())
            if len(self._cal_buffer) >= self._cal_target_samples:
                self._finish_calibration()
            return None

        # No baseline yet
        if zone_id not in self._zones:
            return None

        fp = self._zones[zone_id]

        # Compute anomaly score
        score = self._compute_anomaly_score(csi_amplitude, fp)
        self._current_anomaly_score[zone_id] = score

        # Slow drift correction — update baseline very slowly
        fp.mean += FINGERPRINT_UPDATE_RATE * (csi_amplitude - fp.mean)

        if score < FINGERPRINT_ANOMALY_THRESH:
            # Normal — update environment map
            self._environment_map[zone_id] = {
                "status": "clear",
                "baseline_match_pct": round((1 - score) * 100, 1),
                "updated_at": time.time()
            }
            return None

        # Anomaly detected — classify it
        anomaly_type, magnitude, affected = self._classify_change(csi_amplitude, fp)

        event = AnomalyEvent(
            timestamp=time.time(),
            zone_id=zone_id,
            anomaly_score=float(score),
            anomaly_type=anomaly_type,
            change_magnitude=float(magnitude),
            affected_subcarriers=affected,
            description=self._describe_anomaly(anomaly_type, score, magnitude)
        )

        self._anomaly_history.append(event)
        if len(self._anomaly_history) > 500:
            self._anomaly_history.pop(0)

        # Update environment map
        self._environment_map[zone_id] = {
            "status": "anomaly",
            "anomaly_type": anomaly_type,
            "anomaly_score": round(score, 3),
            "baseline_match_pct": round(max(0, (1 - score)) * 100, 1),
            "description": event.description,
            "updated_at": time.time()
        }

        return event

    def _compute_anomaly_score(self, live: np.ndarray, fp: ZoneFingerprint) -> float:
        """
        Compute normalised Mahalanobis distance between live vector and baseline.
        Score 0 = identical to baseline, 1+ = significant anomaly.
        """
        diff = live - fp.mean

        # Use diagonal approximation for speed (full Mahal is O(n³))
        normalised = diff / (np.sqrt(fp.variance) + 1e-10)
        score = float(np.sqrt(np.mean(normalised**2)))

        return np.clip(score, 0, 5) / 5.0   # normalise to 0–1

    def _classify_change(self, live: np.ndarray,
                         fp: ZoneFingerprint) -> Tuple[str, float, List[int]]:
        """
        Classify the type of environmental change from the CSI delta pattern.
        Returns (type_string, magnitude, affected_subcarrier_indices).
        """
        diff = np.abs(live - fp.mean)
        magnitude = float(np.max(diff))

        # Which subcarriers changed most
        affected = list(np.argsort(diff)[-10:][::-1].astype(int))

        # Pattern analysis
        n = len(diff)

        # Contiguous large-change bands → static object / vehicle
        # (large objects affect clusters of adjacent subcarriers)
        large_mask = diff > (np.mean(diff) + 2 * np.std(diff))
        contiguous_runs = self._count_contiguous_runs(large_mask)

        # Distributed change across many subcarriers → person / motion
        spread = np.sum(large_mask) / n

        if spread > 0.4 and contiguous_runs < 3:
            anomaly_type = "person_motion"
        elif spread < 0.2 and contiguous_runs >= 2 and magnitude > 0.3:
            anomaly_type = "static_large_object"
        elif magnitude > 0.5 and spread > 0.3:
            anomaly_type = "vehicle"
        elif spread > 0.6:
            anomaly_type = "drone_or_moving"
        else:
            anomaly_type = "unknown_change"

        return anomaly_type, magnitude, affected

    def _count_contiguous_runs(self, mask: np.ndarray) -> int:
        runs = 0
        in_run = False
        for v in mask:
            if v and not in_run:
                runs += 1
                in_run = True
            elif not v:
                in_run = False
        return runs

    def _describe_anomaly(self, atype: str, score: float, magnitude: float) -> str:
        confidence = "high" if score > 0.6 else "medium" if score > 0.4 else "low"
        descriptions = {
            "person_motion":     f"Human motion detected ({confidence} confidence)",
            "static_large_object": f"New large static object in area ({confidence} confidence)",
            "vehicle":           f"Vehicle-class object detected ({confidence} confidence)",
            "drone_or_moving":   f"Fast-moving or airborne target detected ({confidence} confidence)",
            "unknown_change":    f"Environmental change detected — type uncertain"
        }
        return descriptions.get(atype, "Anomaly detected")

    # ── Environment map ───────────────────────────────────────────────────────

    def get_environment_map(self) -> dict:
        """Return current state of all monitored zones."""
        return dict(self._environment_map)

    def get_anomaly_history(self, last_n: int = 50) -> List[dict]:
        return [
            {
                "timestamp": e.timestamp,
                "zone": e.zone_id,
                "type": e.anomaly_type,
                "score": round(e.anomaly_score, 3),
                "description": e.description
            }
            for e in self._anomaly_history[-last_n:]
        ]

    def get_zone_score(self, zone_id: str = None) -> float:
        zone_id = zone_id or self._active_zone
        return self._current_anomaly_score.get(zone_id, 0.0)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_fingerprint(self, fp: ZoneFingerprint):
        path = os.path.join(self.baseline_path, f"{fp.zone_id}.npz")
        np.savez(path,
                 mean=fp.mean,
                 covariance=fp.covariance,
                 variance=fp.variance,
                 sample_count=np.array([fp.sample_count]),
                 created_at=np.array([fp.created_at]))
        log.info(f"Fingerprint saved: {path}")

    def _load_saved_fingerprints(self):
        if not os.path.exists(self.baseline_path):
            return
        for fname in os.listdir(self.baseline_path):
            if fname.endswith(".npz"):
                zone_id = fname[:-4]
                try:
                    data = np.load(os.path.join(self.baseline_path, fname), allow_pickle=True)
                    fp = ZoneFingerprint(
                        zone_id=zone_id,
                        mean=data["mean"],
                        covariance=data["covariance"],
                        variance=data["variance"],
                        sample_count=int(data["sample_count"][0]),
                        created_at=float(data["created_at"][0])
                    )
                    self._zones[zone_id] = fp
                    log.info(f"Loaded fingerprint: {zone_id} ({fp.sample_count} samples)")
                except Exception as e:
                    log.warning(f"Failed to load fingerprint {fname}: {e}")
