"""
Training Data Collector
Records labelled radar frames directly from your DFRobot hardware.
Run this before training any model — generates the dataset for all five models.

Usage:
    python3 training/data_collector/collect.py --label person_walking --duration 60
    python3 training/data_collector/collect.py --label drone --duration 60
    python3 training/data_collector/collect.py --label empty --duration 120
    python3 training/data_collector/collect.py --list-labels
"""

import numpy as np
import os
import sys
import time
import argparse
import threading
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

log = logging.getLogger(__name__)

# ── All supported activity and target labels ──────────────────────────────────

ACTIVITY_LABELS = [
    "walking",
    "running",
    "sitting",
    "standing",
    "lying_down",
    "falling",
    "crawling",
    "gesturing",
    "person_still_breathing",
    "person_hiding",
]

TARGET_LABELS = [
    "empty",
    "person",
    "vehicle_car",
    "vehicle_truck",
    "drone_quad",
    "drone_fixedwing",
    "animal",
]

ALL_LABELS = ACTIVITY_LABELS + TARGET_LABELS

LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(ALL_LABELS)}
IDX_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_IDX.items()}

# Separate index maps for each model's task
ACTIVITY_LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(ACTIVITY_LABELS)}
TARGET_LABEL_TO_IDX   = {lbl: i for i, lbl in enumerate(TARGET_LABELS)}


@dataclass
class LabelledFrame:
    """One labelled radar observation."""
    timestamp: float
    label: str
    label_idx: int

    # Raw I/Q [chirps x samples] — stored as float32 real+imag
    iq_real: np.ndarray
    iq_imag: np.ndarray

    # Processed features (computed on collection)
    range_profile: np.ndarray        # magnitude [range_bins]
    spectrogram: np.ndarray          # STFT [freq_bins x time_steps] — for CNN
    phase_sequence: np.ndarray       # unwrapped phase [time_steps] — for LSTM
    doppler_map: np.ndarray          # range-Doppler [range x doppler] — for detection
    csi_proxy: np.ndarray            # per-subcarrier amplitude [52] — for autoencoder

    # Metadata
    range_m: float = 0.0             # estimated target range
    velocity_mps: float = 0.0
    snr_db: float = 0.0


@dataclass
class CollectionSession:
    """Metadata for one collection run."""
    label: str
    duration_s: float
    frame_count: int
    started_at: float
    hardware: str = "dfrobot_24ghz"
    notes: str = ""
    frames: List[LabelledFrame] = field(default_factory=list)


class FeatureExtractor:
    """Extracts all required features from raw I/Q for training."""

    def __init__(self):
        from config.settings import (
            RADAR_NUM_SAMPLES, RADAR_NUM_CHIRPS,
            RADAR_FREQ_START_GHZ, RADAR_BANDWIDTH_GHZ,
            UPDATE_RATE_HZ
        )
        self.N = RADAR_NUM_SAMPLES
        self.M = RADAR_NUM_CHIRPS
        self.c = 3e8
        self.B = RADAR_BANDWIDTH_GHZ * 1e9
        self.range_per_bin = self.c / (2 * self.B)
        self.fs = UPDATE_RATE_HZ

        # Rolling phase buffer for sequence features
        self._phase_buffer = []
        self._spec_buffer  = []
        self._max_seq      = 50    # 5 seconds at 10 Hz

    def extract(self, iq: np.ndarray, label: str) -> LabelledFrame:
        """Full feature extraction from one I/Q frame."""
        # Range FFT
        window = np.hanning(iq.shape[1])
        range_fft = np.fft.fft(iq * window[np.newaxis, :], axis=1)
        range_profile = np.abs(range_fft[:, :self.N//2]).mean(axis=0)

        # Static clutter removal
        clutter_free = range_fft - range_fft.mean(axis=0)[np.newaxis, :]

        # Range-Doppler map (2D FFT)
        doppler_window = np.hanning(self.M)
        rd_map = np.abs(np.fft.fft2(
            clutter_free[:, :self.N//2] * doppler_window[:, np.newaxis]))
        rd_map = np.fft.fftshift(rd_map, axes=0)

        # Phase at peak range bin
        peak_bin = int(np.argmax(range_profile))
        phase = np.angle(clutter_free[:, peak_bin]).mean()
        self._phase_buffer.append(phase)
        if len(self._phase_buffer) > self._max_seq:
            self._phase_buffer.pop(0)
        phase_seq = np.unwrap(np.array(self._phase_buffer))
        # Pad / trim to fixed length
        phase_seq = self._pad_or_trim(phase_seq, self._max_seq)

        # STFT spectrogram on phase sequence
        from scipy.signal import spectrogram as sp_spectrogram
        if len(self._phase_buffer) >= 16:
            f, t, Sxx = sp_spectrogram(
                np.array(self._phase_buffer),
                fs=self.fs, nperseg=16, noverlap=8,
                nfft=64, scaling='spectrum')
            spec = np.log1p(Sxx)
        else:
            spec = np.zeros((33, 4))
        self._spec_buffer.append(spec)

        # Fixed-size spectrogram for CNN [33 x 20 x 1]
        spec_fixed = self._make_fixed_spec(self._spec_buffer)

        # CSI proxy: first 52 bins of range FFT magnitude
        csi_proxy = range_profile[:52]
        csi_proxy /= (np.max(csi_proxy) + 1e-10)

        # Estimated range and velocity
        range_m = peak_bin * self.range_per_bin
        doppler_peak = np.unravel_index(np.argmax(rd_map), rd_map.shape)
        prf = 1.0 / (40e-6 * self.M)
        vel = (doppler_peak[0] - self.M // 2) * (self.c * prf) / (4 * 24e9)

        # SNR estimate
        peak_power = range_profile[peak_bin]
        noise = np.median(range_profile)
        snr = 20 * np.log10(peak_power / (noise + 1e-10) + 1e-10)

        return LabelledFrame(
            timestamp=time.time(),
            label=label,
            label_idx=LABEL_TO_IDX.get(label, -1),
            iq_real=iq.real.astype(np.float32),
            iq_imag=iq.imag.astype(np.float32),
            range_profile=range_profile.astype(np.float32),
            spectrogram=spec_fixed.astype(np.float32),
            phase_sequence=phase_seq.astype(np.float32),
            doppler_map=rd_map[:64, :64].astype(np.float32),
            csi_proxy=csi_proxy.astype(np.float32),
            range_m=float(range_m),
            velocity_mps=float(vel),
            snr_db=float(snr)
        )

    def _pad_or_trim(self, arr: np.ndarray, length: int) -> np.ndarray:
        if len(arr) >= length:
            return arr[-length:]
        return np.pad(arr, (length - len(arr), 0), mode='constant')

    def _make_fixed_spec(self, spec_buffer, rows=33, cols=20) -> np.ndarray:
        """Concatenate recent spectrograms into a fixed [rows x cols] image."""
        if not spec_buffer:
            return np.zeros((rows, cols))
        combined = np.concatenate([s for s in spec_buffer[-4:]], axis=1)
        # Resize to fixed shape
        from scipy.ndimage import zoom
        if combined.shape[0] != rows or combined.shape[1] != cols:
            scale = (rows / combined.shape[0], cols / combined.shape[1])
            combined = zoom(combined, scale, order=1)
        return combined[:rows, :cols]


class DataCollector:
    """
    Collects labelled training data from the radar.
    Saves to .npz files organised by label.
    """

    def __init__(self, output_dir: str = "training/datasets"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.extractor = FeatureExtractor()
        self._collecting = False
        self._session: Optional[CollectionSession] = None

    def collect(self, label: str, duration_s: float,
                radar=None, simulate: bool = False) -> str:
        """
        Collect labelled frames for `duration_s` seconds.
        Returns path to saved .npz file.
        """
        if label not in ALL_LABELS:
            raise ValueError(f"Unknown label '{label}'. Valid: {ALL_LABELS}")

        print(f"\n{'='*50}")
        print(f"COLLECTING: {label.upper()}")
        print(f"Duration:   {duration_s}s")
        print(f"{'='*50}")
        if not simulate:
            input("Position subject, then press ENTER to start...")

        session = CollectionSession(
            label=label,
            duration_s=duration_s,
            frame_count=0,
            started_at=time.time()
        )

        from config.settings import UPDATE_RATE_HZ
        dt = 1.0 / UPDATE_RATE_HZ
        t_end = time.time() + duration_s
        frame_n = 0

        while time.time() < t_end:
            t0 = time.time()

            if simulate or radar is None:
                iq = self._simulate_iq(label)
            else:
                frame = radar.get_latest_frame()
                if frame and frame.raw_iq is not None:
                    iq = frame.raw_iq
                else:
                    time.sleep(dt)
                    continue

            lf = self.extractor.extract(iq, label)
            session.frames.append(lf)
            frame_n += 1

            elapsed = time.time() - session.started_at
            remaining = duration_s - elapsed
            if frame_n % 10 == 0:
                print(f"  {elapsed:.1f}s / {duration_s}s  |  "
                      f"{frame_n} frames  |  "
                      f"range={lf.range_m:.1f}m  snr={lf.snr_db:.1f}dB")

            sleep_t = max(0, dt - (time.time() - t0))
            time.sleep(sleep_t)

        session.frame_count = len(session.frames)
        path = self._save_session(session)
        print(f"\nSaved {session.frame_count} frames → {path}")
        return path

    def _save_session(self, session: CollectionSession) -> str:
        ts = int(session.started_at)
        fname = f"{session.label}_{ts}_{session.frame_count}frames.npz"
        path = os.path.join(self.output_dir, session.label, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        frames = session.frames
        np.savez_compressed(
            path,
            label=np.array([session.label] * len(frames)),
            label_idx=np.array([f.label_idx for f in frames], dtype=np.int32),
            spectrograms=np.array([f.spectrogram for f in frames]),
            phase_sequences=np.array([f.phase_sequence for f in frames]),
            doppler_maps=np.array([f.doppler_map for f in frames]),
            csi_proxies=np.array([f.csi_proxy for f in frames]),
            range_profiles=np.array([f.range_profile for f in frames]),
            range_m=np.array([f.range_m for f in frames]),
            velocity_mps=np.array([f.velocity_mps for f in frames]),
            snr_db=np.array([f.snr_db for f in frames]),
        )
        return path

    def _simulate_iq(self, label: str) -> np.ndarray:
        """Generate synthetic I/Q for testing without hardware."""
        from config.settings import RADAR_NUM_CHIRPS, RADAR_NUM_SAMPLES
        from config.settings import RADAR_FREQ_START_GHZ, RADAR_BANDWIDTH_GHZ
        N, M = RADAR_NUM_SAMPLES, RADAR_NUM_CHIRPS
        c = 3e8
        B = RADAR_BANDWIDTH_GHZ * 1e9
        f0 = RADAR_FREQ_START_GHZ * 1e9
        t_now = time.time()

        sim_params = {
            "walking":              {"r": 4.0, "v": 0.8, "br": 0.3,  "bv": 0.4},
            "running":              {"r": 5.0, "v": 2.5, "br": 0.5,  "bv": 0.7},
            "sitting":              {"r": 3.0, "v": 0.0, "br": 0.25, "bv": 0.05},
            "standing":             {"r": 3.5, "v": 0.0, "br": 0.28, "bv": 0.04},
            "lying_down":           {"r": 2.0, "v": 0.0, "br": 0.18, "bv": 0.03},
            "falling":              {"r": 3.0, "v": 1.8, "br": 0.0,  "bv": 1.2},
            "crawling":             {"r": 4.0, "v": 0.3, "br": 0.25, "bv": 0.15},
            "gesturing":            {"r": 2.0, "v": 0.5, "br": 0.28, "bv": 0.2},
            "person_still_breathing":{"r":3.0, "v": 0.0, "br": 0.22, "bv": 0.02},
            "person_hiding":        {"r": 5.0, "v": 0.0, "br": 0.20, "bv": 0.01},
            "empty":                {"r": 0.0, "v": 0.0, "br": 0.0,  "bv": 0.0},
            "person":               {"r": 4.0, "v": 0.5, "br": 0.28, "bv": 0.2},
            "vehicle_car":          {"r": 8.0, "v": 3.0, "br": 0.0,  "bv": 0.0},
            "vehicle_truck":        {"r":10.0, "v": 2.0, "br": 0.0,  "bv": 0.0},
            "drone_quad":           {"r": 6.0, "v": 1.0, "br": 0.0,  "bv": 0.0},
            "drone_fixedwing":      {"r":12.0, "v": 8.0, "br": 0.0,  "bv": 0.0},
            "animal":               {"r": 4.0, "v": 0.6, "br": 0.0,  "bv": 0.3},
        }
        p = sim_params.get(label, sim_params["empty"])

        iq = np.zeros((M, N), dtype=complex)
        noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * 0.03

        if p["r"] > 0:
            # Breathing displacement
            disp = p["bv"] * np.sin(2 * np.pi * p["br"] * t_now)
            r = p["r"] + disp + p["v"] * 0.01 * np.random.randn()

            # Drone blade modulation
            if "drone" in label:
                blade_hz = 120 if "quad" in label else 60
                blade_mod = 0.1 * np.sin(2 * np.pi * blade_hz * t_now)
                r += blade_mod

            fb = 2 * B * r / (c * 40e-6)
            t_fast = np.linspace(0, 40e-6, N)

            for chirp in range(M):
                phase_offset = 4 * np.pi * f0 * r / c
                iq[chirp] += 0.8 * np.exp(1j * (2 * np.pi * fb * t_fast + phase_offset))

        return iq + noise


def load_dataset(dataset_dir: str = "training/datasets",
                 labels: List[str] = None,
                 feature: str = "spectrogram"):
    """
    Load all collected .npz files into arrays for training.

    feature: "spectrogram" | "phase_sequence" | "doppler_map" | "csi_proxy"
    labels:  subset of ALL_LABELS to include (None = all)
    """
    if labels is None:
        labels = ALL_LABELS

    X_list, y_list = [], []
    found = 0

    for lbl in labels:
        lbl_dir = os.path.join(dataset_dir, lbl)
        if not os.path.exists(lbl_dir):
            continue
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".npz"):
                continue
            path = os.path.join(lbl_dir, fname)
            data = np.load(path, allow_pickle=True)
            X = data[feature + "s"] if feature + "s" in data else data[feature]
            y = data["label_idx"]
            # Remap to subset indices
            mask = np.array([str(data["label"][i]) in labels for i in range(len(y))])
            if mask.any():
                X_list.append(X[mask])
                y_list.append(y[mask])
                found += len(X[mask])

    if not X_list:
        raise ValueError(f"No data found in {dataset_dir} for labels {labels}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"Loaded {found} samples from {dataset_dir}")
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, help="Label to collect")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--list-labels", action="store_true")
    parser.add_argument("--output-dir", default="training/datasets")
    args = parser.parse_args()

    if args.list_labels:
        print("\nActivity labels:")
        for l in ACTIVITY_LABELS: print(f"  {l}")
        print("\nTarget labels:")
        for l in TARGET_LABELS: print(f"  {l}")
        sys.exit(0)

    if not args.label:
        parser.print_help()
        sys.exit(1)

    collector = DataCollector(args.output_dir)
    collector.collect(args.label, args.duration, simulate=args.simulate)
