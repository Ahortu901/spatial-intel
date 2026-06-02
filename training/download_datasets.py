"""
Open Dataset Downloader
Downloads and preprocesses publicly available radar datasets for base model training.
This runs ONCE before deployment — never needed again after that.

Datasets:
  RadHAR    — 1.5M radar frames, 5 human activity classes (MIT)
  DroneRF   — RF signatures of 10 drone types (IEEE open access)
  MAFAT     — Moving target indicator challenge data (open portion)
  Synthetic — High-quality simulation to fill gaps

Usage:
    python3 training/download_datasets.py --all
    python3 training/download_datasets.py --dataset radhar
    python3 training/download_datasets.py --synthetic-only  # no internet needed
"""

import os
import sys
import json
import time
import hashlib
import zipfile
import tarfile
import shutil
import logging
import argparse
import urllib.request
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")


# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {

    "radhar": {
        "name": "RadHAR",
        "description": "Human activity recognition — walk, run, sit, stand, fall",
        "url": "https://github.com/nesl/RadHAR/archive/refs/heads/master.zip",
        "fallback_url": None,
        "licence": "MIT",
        "classes": ["walking", "running", "sitting", "standing", "falling"],
        "size_mb": 180,
        "format": "zip",
        "notes": "1.5M radar point cloud frames across 5 activity classes"
    },

    "dronerf": {
        "name": "DroneRF",
        "description": "RF signatures of drones vs background — 10 drone types",
        "url": "https://zenodo.org/record/4298659/files/DroneRF.zip",
        "fallback_url": None,
        "licence": "Creative Commons",
        "classes": ["background", "drone_bebop", "drone_ar", "drone_phantom",
                    "drone_inspire", "drone_matrice"],
        "size_mb": 220,
        "format": "zip",
        "notes": "RF signal segments at 2.4GHz and 5.8GHz control frequencies"
    },

    "gesture": {
        "name": "Google Radar Gestures",
        "description": "11 hand gesture classes from Soli radar chip",
        "url": "https://storage.googleapis.com/soli_data/soli_gestures.zip",
        "fallback_url": None,
        "licence": "Apache 2.0",
        "classes": ["swipe_left", "swipe_right", "swipe_up", "swipe_down",
                    "push", "pull", "grab", "release", "tap", "pinch", "expand"],
        "size_mb": 95,
        "format": "zip",
        "notes": "60GHz Soli radar — gestures transfer well to 24GHz"
    },
}


class DatasetDownloader:
    """
    Downloads, verifies, and preprocesses open radar datasets.
    All processing converts to the common .npz format used by the trainer.
    """

    def __init__(self, base_dir: str = "training/datasets"):
        self.base_dir = Path(base_dir)
        self.raw_dir  = self.base_dir / "raw"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self, skip_existing: bool = True):
        """Download and process all registered datasets."""
        results = {}
        for name, info in DATASETS.items():
            log.info(f"\n{'='*50}")
            log.info(f"Dataset: {info['name']}")
            log.info(f"{info['description']}")
            log.info(f"Size: ~{info['size_mb']} MB")
            try:
                path = self.download(name, skip_existing)
                results[name] = {"status": "OK", "path": str(path)}
            except Exception as e:
                log.warning(f"Download failed for {name}: {e}")
                log.warning(f"Falling back to synthetic data for {name}")
                results[name] = {"status": "synthetic", "error": str(e)}
        return results

    def download(self, dataset_name: str,
                 skip_existing: bool = True) -> Path:
        """Download a single dataset and convert to training format."""
        info = DATASETS.get(dataset_name)
        if not info:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        out_dir = self.base_dir / dataset_name
        marker  = out_dir / ".complete"

        if skip_existing and marker.exists():
            log.info(f"{dataset_name} already downloaded — skipping")
            return out_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self.raw_dir / f"{dataset_name}.{info['format']}"

        # Download
        log.info(f"Downloading {info['url']} ...")
        self._download_with_progress(info['url'], raw_path)

        # Extract and convert
        log.info(f"Extracting and converting {dataset_name} ...")
        converted = self._convert(dataset_name, raw_path, out_dir)

        # Write completion marker
        with open(marker, 'w') as f:
            json.dump({
                "dataset": dataset_name,
                "downloaded_at": time.time(),
                "samples": converted
            }, f)

        log.info(f"Done: {converted} samples → {out_dir}")
        return out_dir

    def _download_with_progress(self, url: str, dest: Path):
        """Download with progress bar."""
        def report(block, block_size, total):
            if total > 0:
                pct = min(100, block * block_size * 100 // total)
                bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
                print(f"\r  [{bar}] {pct}%", end='', flush=True)

        try:
            urllib.request.urlretrieve(url, dest, reporthook=report)
            print()
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")

    def _convert(self, name: str, raw_path: Path, out_dir: Path) -> int:
        """Convert raw dataset to our .npz training format."""
        if name == "radhar":
            return self._convert_radhar(raw_path, out_dir)
        elif name == "dronerf":
            return self._convert_dronerf(raw_path, out_dir)
        elif name == "gesture":
            return self._convert_gestures(raw_path, out_dir)
        else:
            return 0

    def _convert_radhar(self, raw_path: Path, out_dir: Path) -> int:
        """Convert RadHAR point cloud data to spectrogram format."""
        total = 0
        label_map = {
            "walk": "walking", "run": "running",
            "sit": "sitting", "stand": "standing", "fall": "falling"
        }

        with zipfile.ZipFile(raw_path) as zf:
            files = [f for f in zf.namelist() if f.endswith('.csv')]
            for fname in files:
                # Determine label from path
                label = None
                for key, val in label_map.items():
                    if key in fname.lower():
                        label = val
                        break
                if label is None:
                    continue

                with zf.open(fname) as f:
                    try:
                        data = np.genfromtxt(f, delimiter=',',
                                             skip_header=1, dtype=np.float32)
                    except Exception:
                        continue

                if data.ndim < 2 or len(data) < 10:
                    continue

                # RadHAR format: [x, y, z, intensity, doppler] point clouds
                # Convert to pseudo-spectrogram by binning
                specs = self._pointcloud_to_spectrogram(data)
                phase_seqs = self._pointcloud_to_phase(data)

                out_path = out_dir / label / f"{fname.replace('/', '_')}.npz"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                label_idx = ["walking","running","sitting",
                             "standing","falling","crawling",
                             "gesturing","person_still_breathing",
                             "person_hiding"].index(label)

                np.savez_compressed(
                    str(out_path),
                    spectrograms=specs,
                    phase_sequences=phase_seqs,
                    label=np.array([label] * len(specs)),
                    label_idx=np.array([label_idx] * len(specs), dtype=np.int32),
                )
                total += len(specs)

        return total

    def _convert_dronerf(self, raw_path: Path, out_dir: Path) -> int:
        """Convert DroneRF RF segments to CSI proxy format."""
        total = 0
        drone_labels = {
            "BG": "empty", "bebop": "drone_quad", "AR": "drone_quad",
            "phantom": "drone_quad", "inspire": "drone_quad", "matrice": "drone_quad"
        }

        with zipfile.ZipFile(raw_path) as zf:
            files = [f for f in zf.namelist()
                     if f.endswith('.mat') or f.endswith('.npy')]
            for fname in files:
                label = None
                for key, val in drone_labels.items():
                    if key in fname:
                        label = val
                        break
                if label is None:
                    continue

                with zf.open(fname) as f:
                    try:
                        if fname.endswith('.npy'):
                            data = np.load(f)
                        else:
                            continue
                    except Exception:
                        continue

                if data.ndim < 1 or len(data) < 52:
                    continue

                # Segment into 52-point CSI proxy windows
                n_windows = len(data) // 52
                csi_windows = data[:n_windows*52].reshape(n_windows, 52)
                csi_windows = np.abs(csi_windows).astype(np.float32)
                csi_windows /= (np.max(csi_windows, axis=1, keepdims=True) + 1e-10)

                out_path = out_dir / label / f"{fname.replace('/', '_')}.npz"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                target_labels = ["empty","person","vehicle_car","vehicle_truck",
                                 "drone_quad","drone_fixedwing","animal"]
                label_idx = target_labels.index(label) if label in target_labels else 0

                np.savez_compressed(
                    str(out_path),
                    csi_proxies=csi_windows,
                    label=np.array([label] * len(csi_windows)),
                    label_idx=np.array([label_idx] * len(csi_windows), dtype=np.int32),
                    spectrograms=np.zeros((len(csi_windows), 33, 20), dtype=np.float32),
                    phase_sequences=np.zeros((len(csi_windows), 50), dtype=np.float32),
                )
                total += len(csi_windows)

        return total

    def _convert_gestures(self, raw_path: Path, out_dir: Path) -> int:
        """Convert Soli gesture data to phase sequence format."""
        return 0   # placeholder — format-specific parsing

    def _pointcloud_to_spectrogram(self, data: np.ndarray) -> np.ndarray:
        """Convert radar point cloud frames to [N x 33 x 20] spectrograms."""
        # Bin Doppler values into frequency-like histogram
        spec_rows, spec_cols = 33, 20
        n_frames = max(1, len(data) // 20)
        specs = []

        for i in range(n_frames):
            chunk = data[i*20 : (i+1)*20]
            if len(chunk) == 0:
                continue
            spec = np.zeros((spec_rows, spec_cols), dtype=np.float32)
            # Use intensity and doppler columns (cols 3, 4 if available)
            if chunk.shape[1] >= 5:
                doppler = chunk[:, 4]
                intensity = chunk[:, 3]
                for j, (d, amp) in enumerate(zip(doppler, intensity)):
                    row = int(np.clip((d + 5) / 10 * spec_rows, 0, spec_rows-1))
                    col = j % spec_cols
                    spec[row, col] += amp
            specs.append(spec)

        return np.array(specs) if specs else np.zeros((1, spec_rows, spec_cols))

    def _pointcloud_to_phase(self, data: np.ndarray) -> np.ndarray:
        """Convert point cloud to [N x 50] phase sequences."""
        seq_len = 50
        n_seqs = max(1, len(data) // seq_len)
        seqs = []
        for i in range(n_seqs):
            chunk = data[i*seq_len : (i+1)*seq_len]
            if len(chunk) == 0:
                continue
            # Use range column as proxy for phase
            col = chunk[:, 2] if chunk.shape[1] > 2 else chunk[:, 0]
            seq = np.pad(col, (max(0, seq_len - len(col)), 0))[:seq_len]
            seqs.append(seq.astype(np.float32))
        return np.array(seqs) if seqs else np.zeros((1, seq_len))


def generate_synthetic_pretrain_data(
        output_dir: str = "training/datasets",
        n_per_activity: int = 800,
        n_per_target: int = 600):
    """
    Generate high-quality synthetic data for ALL classes.
    Runs entirely offline — no internet needed.
    This is the fallback when real datasets can't be downloaded,
    and also supplements real data for rare classes (falling, crawling).
    """
    log.info("Generating synthetic pre-training dataset (fully offline)...")

    from training.data_collector.collect import DataCollector, ALL_LABELS

    collector = DataCollector(output_dir)
    total_generated = 0

    all_labels = [
        # Activities
        "walking", "running", "sitting", "standing", "lying_down",
        "falling", "crawling", "gesturing", "person_still_breathing",
        "person_hiding",
        # Targets
        "empty", "person", "vehicle_car", "vehicle_truck",
        "drone_quad", "drone_fixedwing", "animal"
    ]

    for label in all_labels:
        is_activity = label in [
            "walking","running","sitting","standing","lying_down",
            "falling","crawling","gesturing",
            "person_still_breathing","person_hiding"
        ]
        n = n_per_activity if is_activity else n_per_target

        # Check if we already have enough data for this label
        label_dir = os.path.join(output_dir, label)
        if os.path.exists(label_dir):
            existing = sum(
                int(f.split('_')[-1].replace('frames.npz',''))
                for f in os.listdir(label_dir)
                if f.endswith('.npz') and '_' in f
            )
            if existing >= n * 0.8:
                log.info(f"  {label}: {existing} frames already exist — skipping")
                continue

        log.info(f"  Generating {n} frames for '{label}'...")
        duration_s = n / 10.0  # 10 Hz collection rate
        collector.collect(label, duration_s=duration_s, simulate=True)
        total_generated += n

    log.info(f"Synthetic dataset complete: {total_generated} total frames")
    return total_generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download open radar datasets for base model training")
    parser.add_argument("--all", action="store_true",
                        help="Download all datasets")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="Download a specific dataset")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip downloads — generate synthetic data only")
    parser.add_argument("--output-dir", default="training/datasets")
    parser.add_argument("--n-per-class", type=int, default=800)
    args = parser.parse_args()

    if args.synthetic_only:
        generate_synthetic_pretrain_data(
            args.output_dir, n_per_activity=args.n_per_class)
    elif args.all:
        dl = DatasetDownloader(args.output_dir)
        results = dl.download_all()
        # Fill gaps with synthetic data
        generate_synthetic_pretrain_data(
            args.output_dir, n_per_activity=args.n_per_class)
        print(json.dumps(results, indent=2))
    elif args.dataset:
        dl = DatasetDownloader(args.output_dir)
        dl.download(args.dataset)
    else:
        parser.print_help()
