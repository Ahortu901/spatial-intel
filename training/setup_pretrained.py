"""
Pre-Deployment Setup
Run this ONCE before going to the field — requires internet.
After this, the device runs completely offline forever.

Steps:
  1. Download open radar datasets (RadHAR, DroneRF)
  2. Generate additional synthetic data to fill gaps
  3. Train all five base models
  4. Compute EWC Fisher matrix (for continual learning)
  5. Export all models to TFLite INT8 (fast on CM5)
  6. Flash summary — what's on the device

Usage:
    python3 training/setup_pretrained.py
    python3 training/setup_pretrained.py --offline  # skip downloads, use synthetic only
    python3 training/setup_pretrained.py --fast     # quick test with fewer samples
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("setup")


def run_setup(offline: bool = False,
              fast: bool = False,
              dataset_dir: str = "training/datasets",
              model_dir: str = "models"):

    t_total = time.time()
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs("data/continual_buffers", exist_ok=True)
    os.makedirs("data/checkpoints", exist_ok=True)

    n_per_class = 200 if fast else 800

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   SPATIAL INTELLIGENCE — PRE-DEPLOYMENT SETUP   ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Mode:     {'OFFLINE (synthetic only)' if offline else 'ONLINE (real datasets)'}")
    print(f"  Fast:     {fast}")
    print(f"  Samples:  {n_per_class} per class")
    print()

    results = {}

    # ── Step 1: Datasets ──────────────────────────────────────────────────
    print("STEP 1/5 — Building training dataset")
    print("-" * 50)

    if not offline:
        print("  Attempting to download open datasets...")
        try:
            from training.download_datasets import DatasetDownloader
            dl = DatasetDownloader(dataset_dir)
            dl_results = dl.download_all()
            results["downloads"] = dl_results
            print(f"  Downloads: {dl_results}")
        except Exception as e:
            print(f"  Download failed ({e}) — using synthetic data")

    print("  Generating synthetic data for all classes...")
    from training.download_datasets import generate_synthetic_pretrain_data
    total = generate_synthetic_pretrain_data(
        dataset_dir, n_per_activity=n_per_class, n_per_target=n_per_class)
    print(f"  Generated {total} synthetic frames")
    results["synthetic_frames"] = total

    # ── Step 2: Train all models ──────────────────────────────────────────
    print()
    print("STEP 2/5 — Training base models (offline from here)")
    print("-" * 50)
    print("  Internet no longer needed from this point.")
    print()

    epochs = 5 if fast else 30

    # Target classifier
    print("[2a] Target classifier...")
    try:
        from training.trainers.train_target_classifier import train_target_classifier
        _, tc_path = train_target_classifier(
            dataset_dir=dataset_dir, output_dir=model_dir,
            epochs=epochs, simulate_data=True)
        results["target_classifier"] = "OK"
        print(f"     ✓ {tc_path}")
    except Exception as e:
        results["target_classifier"] = f"FAILED: {e}"
        print(f"     ✗ {e}")

    # Activity recogniser
    print("[2b] Activity recogniser (CNN-LSTM)...")
    try:
        from training.trainers.train_activity_recogniser import train_activity_recogniser
        _, ar_path = train_activity_recogniser(
            dataset_dir=dataset_dir, output_dir=model_dir,
            epochs=epochs, simulate_data=True)
        results["activity_recogniser"] = "OK"
        print(f"     ✓ {ar_path}")
    except Exception as e:
        results["activity_recogniser"] = f"FAILED: {e}"
        print(f"     ✗ {e}")

    # RF autoencoder
    print("[2c] RF environment autoencoder...")
    try:
        from training.trainers.train_other_models import train_autoencoder
        _, ae_path = train_autoencoder(
            output_dir=model_dir, epochs=epochs, simulate_data=True)
        results["autoencoder"] = "OK"
        print(f"     ✓ {ae_path}")
    except Exception as e:
        results["autoencoder"] = f"FAILED: {e}"
        print(f"     ✗ {e}")

    # Vital signs CNN
    print("[2d] Vital signs CNN...")
    try:
        from training.trainers.train_other_models import train_vitals_cnn
        _, vc_path = train_vitals_cnn(output_dir=model_dir, epochs=min(epochs, 20))
        results["vitals_cnn"] = "OK"
        print(f"     ✓ {vc_path}")
    except Exception as e:
        results["vitals_cnn"] = f"FAILED: {e}"
        print(f"     ✗ {e}")

    # Gait re-ID
    print("[2e] Gait re-identification...")
    try:
        from training.trainers.train_other_models import train_siamese_reid
        _, gr = train_siamese_reid(output_dir=model_dir, epochs=min(epochs, 20))
        results["gait_reid"] = "OK"
        print(f"     ✓ gait_embedder.h5")
    except Exception as e:
        results["gait_reid"] = f"FAILED: {e}"
        print(f"     ✗ {e}")

    # ── Step 3: EWC Fisher matrix ─────────────────────────────────────────
    print()
    print("STEP 3/5 — Computing EWC importance matrix")
    print("-" * 50)
    print("  This prevents forgetting during field adaptation.")
    try:
        _compute_ewc(model_dir, dataset_dir, fast)
        results["ewc"] = "OK"
        print("  ✓ Fisher matrix computed")
    except Exception as e:
        results["ewc"] = f"FAILED: {e}"
        print(f"  ✗ EWC failed: {e}")

    # ── Step 4: Verify all TFLite models ──────────────────────────────────
    print()
    print("STEP 4/5 — Verifying model files")
    print("-" * 50)
    model_files = {
        "target_classifier.tflite":     "Target classification",
        "activity_recogniser.tflite":   "Activity recognition",
        "rf_autoencoder.tflite":        "RF environment anomaly",
        "vitals_cnn.tflite":            "Vital signs estimation",
        "gait_embedder.h5":             "Person re-identification",
    }
    all_ok = True
    for fname, desc in model_files.items():
        fpath = os.path.join(model_dir, fname)
        if os.path.exists(fpath):
            kb = os.path.getsize(fpath) // 1024
            print(f"  ✓ {desc:35s} {kb:5d} KB")
        else:
            print(f"  ✗ {desc:35s} MISSING")
            all_ok = False

    # ── Step 5: Write deployment manifest ────────────────────────────────
    print()
    print("STEP 5/5 — Writing deployment manifest")
    print("-" * 50)

    manifest = {
        "deployed_at": time.time(),
        "setup_mode": "offline" if offline else "online",
        "fast_mode": fast,
        "models": {},
        "results": results,
        "internet_required_again": False,
        "continual_learning": {
            "enabled": True,
            "retrain_interval_hours": 1,
            "min_samples_to_retrain": 50,
            "ewc_protection": results.get("ewc") == "OK",
            "mesh_sharing": True,
        }
    }

    for fname in os.listdir(model_dir):
        fpath = os.path.join(model_dir, fname)
        manifest["models"][fname] = {
            "size_kb": os.path.getsize(fpath) // 1024,
            "modified": os.path.getmtime(fpath)
        }

    manifest_path = os.path.join(model_dir, "deployment_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t_total

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║              SETUP COMPLETE                      ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Time:          {elapsed/60:5.1f} minutes                    ║")
    print(f"║  Models ready:  {sum(1 for v in results.values() if v == 'OK'):5d}                         ║")
    print(f"║  Internet:      NOT NEEDED AGAIN                ║")
    print(f"║  Adapts itself: YES — every hour from live data  ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Next step:                                      ║")
    print("║    python3 main.py                               ║")
    print("║    Open http://[CM5-IP]:8000                     ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    return manifest


def _compute_ewc(model_dir: str, dataset_dir: str, fast: bool):
    """Compute EWC Fisher matrix on training data."""
    import tensorflow as tf
    from training.continual.continual_learner import EWC

    model_path = os.path.join(model_dir, "activity_best.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Activity model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    # Load small sample of training data
    from training.trainers.train_activity_recogniser import (
        _generate_synthetic_activity_data, ACTIVITY_LABEL_NAMES
    )
    n = 50 if fast else 200
    specs, seqs, labels = _generate_synthetic_activity_data(n_per_class=n // 10)

    if specs.ndim == 3:
        specs = specs[..., np.newaxis]

    ewc = EWC(os.path.join("data/checkpoints", "fisher_importance.npz"))
    ewc.compute(model, specs, seqs, labels, n_samples=min(n, len(specs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-deployment setup — run once, then offline forever")
    parser.add_argument("--offline", action="store_true",
                        help="Skip downloads, use synthetic data only")
    parser.add_argument("--fast", action="store_true",
                        help="Quick setup with fewer samples (for testing)")
    parser.add_argument("--dataset-dir", default="training/datasets")
    parser.add_argument("--model-dir",   default="models")
    args = parser.parse_args()

    run_setup(
        offline=args.offline,
        fast=args.fast,
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir
    )
