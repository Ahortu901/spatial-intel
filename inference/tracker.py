"""
Multi-Target Kalman Tracker + Threat Classifier
Maintains persistent tracks across frames, classifies target type,
and estimates trajectory, velocity, and dwell time.
"""

import numpy as np
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
import logging

log = logging.getLogger(__name__)

from config.settings import (
    TRACKER_MAX_TARGETS, TRACKER_INIT_THRESHOLD,
    TRACKER_DELETE_THRESHOLD, TRACKER_GATE_DISTANCE_M,
    DRONE_BLADE_HZ_MIN, DRONE_BLADE_HZ_MAX,
    PERSON_CONF_THRESHOLD, VEHICLE_CONF_THRESHOLD, DRONE_CONF_THRESHOLD
)


@dataclass
class Track:
    track_id: str
    state: np.ndarray        # [x, y, vx, vy] Kalman state
    covariance: np.ndarray   # 4x4 covariance matrix
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    target_class: str = "unknown"     # "person" | "vehicle" | "drone" | "unknown"
    class_confidence: float = 0.0
    micro_doppler_profile: Optional[np.ndarray] = None
    entry_point: Optional[Tuple[float, float]] = None
    history: List[Tuple[float, float]] = field(default_factory=list)  # (x, y) trail

    # Vital signs (populated by VitalSignsExtractor)
    breath_rate_bpm: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    vitals_status: str = "unknown"


class KalmanTracker:
    """
    Multi-target tracker using Kalman filter + Hungarian assignment.
    Constant-velocity motion model.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self._tracks: Dict[str, Track] = {}
        self._next_id = 0

        # State transition matrix [x, y, vx, vy]
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

        # Observation matrix (we observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Process noise
        q = 0.5
        self.Q = np.eye(4) * q
        self.Q[2, 2] = q * 4
        self.Q[3, 3] = q * 4

        # Measurement noise
        self.R = np.eye(2) * 0.5

    # ── Main update step ─────────────────────────────────────────────────────

    def update(self, detections: List[dict]) -> List[Track]:
        """
        detections: list of {x_m, y_m, range_m, az_deg, power, ...}
        Returns list of confirmed Track objects.
        """
        # 1. Predict all existing tracks
        for track in self._tracks.values():
            track.state, track.covariance = self._predict(track.state, track.covariance)

        # 2. Associate detections to tracks (Hungarian algorithm)
        if detections and self._tracks:
            cost_matrix = self._build_cost_matrix(detections)
            if cost_matrix.size > 0:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                matched_det = set()
                matched_trk = set()

                track_ids = list(self._tracks.keys())

                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < TRACKER_GATE_DISTANCE_M:
                        tid = track_ids[c]
                        self._update_track(tid, detections[r])
                        matched_det.add(r)
                        matched_trk.add(tid)

                # Unmatched detections → new tentative tracks
                for i, det in enumerate(detections):
                    if i not in matched_det:
                        self._new_track(det)

                # Unmatched tracks → increment miss counter
                for tid in track_ids:
                    if tid not in matched_trk:
                        self._tracks[tid].misses += 1
        elif detections:
            for det in detections:
                self._new_track(det)
        else:
            for track in self._tracks.values():
                track.misses += 1

        # 3. Confirm and delete tracks
        to_delete = []
        for tid, track in self._tracks.items():
            if track.hits >= TRACKER_INIT_THRESHOLD:
                track.confirmed = True
            if track.misses > TRACKER_DELETE_THRESHOLD:
                to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]

        # 4. Enforce max targets
        if len(self._tracks) > TRACKER_MAX_TARGETS:
            sorted_tracks = sorted(self._tracks.items(),
                                   key=lambda x: x[1].last_seen)
            for tid, _ in sorted_tracks[:len(self._tracks) - TRACKER_MAX_TARGETS]:
                del self._tracks[tid]

        confirmed = [t for t in self._tracks.values() if t.confirmed]
        return confirmed

    # ── Kalman predict / correct ──────────────────────────────────────────────

    def _predict(self, state, P):
        state_pred = self.F @ state
        P_pred = self.F @ P @ self.F.T + self.Q
        return state_pred, P_pred

    def _correct(self, state, P, z):
        z = np.array(z)
        y = z - self.H @ state                          # innovation
        S = self.H @ P @ self.H.T + self.R             # innovation covariance
        K = P @ self.H.T @ np.linalg.inv(S)            # Kalman gain
        state_new = state + K @ y
        P_new = (np.eye(4) - K @ self.H) @ P
        return state_new, P_new

    # ── Track management ──────────────────────────────────────────────────────

    def _new_track(self, det: dict):
        x, y = det["x_m"], det["y_m"]
        state = np.array([x, y, 0.0, 0.0])
        P = np.eye(4) * 2.0
        tid = f"T{self._next_id:04d}"
        self._next_id += 1

        track = Track(
            track_id=tid,
            state=state,
            covariance=P,
            hits=1,
            entry_point=(x, y),
            history=[(x, y)]
        )
        self._tracks[tid] = track

    def _update_track(self, tid: str, det: dict):
        track = self._tracks[tid]
        z = [det["x_m"], det["y_m"]]
        track.state, track.covariance = self._correct(track.state, track.covariance, z)
        track.hits += 1
        track.misses = 0
        track.last_seen = time.time()

        # Append to history (max 50 points)
        track.history.append((float(track.state[0]), float(track.state[1])))
        if len(track.history) > 50:
            track.history.pop(0)

    def _build_cost_matrix(self, detections: List[dict]) -> np.ndarray:
        track_ids = list(self._tracks.keys())
        n_det = len(detections)
        n_trk = len(track_ids)
        cost = np.full((n_det, n_trk), fill_value=999.0)

        for d_idx, det in enumerate(detections):
            for t_idx, tid in enumerate(track_ids):
                trk = self._tracks[tid]
                dx = det["x_m"] - trk.state[0]
                dy = det["y_m"] - trk.state[1]
                cost[d_idx, t_idx] = np.sqrt(dx**2 + dy**2)

        return cost

    # ── Public accessors ──────────────────────────────────────────────────────

    def get_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.confirmed]

    def get_track(self, tid: str) -> Optional[Track]:
        return self._tracks.get(tid)

    def count_by_class(self) -> dict:
        counts = {"person": 0, "vehicle": 0, "drone": 0, "unknown": 0}
        for t in self.get_tracks():
            counts[t.target_class] = counts.get(t.target_class, 0) + 1
        return counts


class TargetClassifier:
    """
    Classifies detected targets as person / vehicle / drone / unknown
    using micro-Doppler spectral features.
    """

    def classify(self, track: Track, doppler_spectrum: Optional[np.ndarray],
                 radar_cross_section: float) -> Tuple[str, float]:
        """
        Returns (class_label, confidence).
        Uses heuristic feature rules — replace with TFLite model for production.
        """
        if doppler_spectrum is None:
            return self._classify_by_rcs(radar_cross_section)

        freqs = np.linspace(0, 500, len(doppler_spectrum))
        power = np.abs(doppler_spectrum)

        # Feature: energy in blade-rotation band (drone)
        blade_mask = (freqs >= DRONE_BLADE_HZ_MIN) & (freqs <= DRONE_BLADE_HZ_MAX)
        blade_energy = np.sum(power[blade_mask]) / (np.sum(power) + 1e-10)

        # Feature: energy in gait band (person walking ~1–3 Hz)
        gait_mask = (freqs >= 1.0) & (freqs <= 3.5)
        gait_energy = np.sum(power[gait_mask]) / (np.sum(power) + 1e-10)

        # Feature: broadband spread (vehicle engine vibration)
        spread = np.std(power) / (np.mean(power) + 1e-10)

        if blade_energy > 0.35:
            return "drone", float(np.clip(blade_energy * 2.5, 0, 1))
        elif gait_energy > 0.25 and radar_cross_section < 2.0:
            return "person", float(np.clip(gait_energy * 3.0, 0, 1))
        elif radar_cross_section > 4.0:
            return "vehicle", float(np.clip(radar_cross_section / 20.0, 0, 1))
        else:
            return "unknown", 0.4

    def _classify_by_rcs(self, rcs: float) -> Tuple[str, float]:
        if rcs > 10.0:
            return "vehicle", 0.65
        elif rcs > 0.5:
            return "person", 0.55
        else:
            return "unknown", 0.3
