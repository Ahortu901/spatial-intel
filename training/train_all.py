"""
Master Training Pipeline
Trains all five models in sequence, or individually.

Usage:
    # Train everything from scratch using simulated data (no hardware needed)
    python3 training/train_all.py --simulate

    # Train everything from collected hardware data
    python3 training/train_all.py

    # Train one model only
    python3 training/train_all.py --model activity --simulate

    # Collect data first, then train
    python3 training/data_collector/collect.py --label walking --duration 60 --simulate
    python3 training/train_all.py
"""

import os
import sys
import json
import time
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("train_all")


def train_all(simulate: bool = False,
              dataset_dir: str = "training/datasets",
              output_dir: str = "models",
              epochs_fast: int = 20,
              epochs_full: int = 40):

    os.makedirs(output_dir, exist_ok=True)
    results = {}
    t_start = time.time()

    print("\n" + "=" * 60)
    print("SPATIAL INTELLIGENCE — FULL ML TRAINING PIPELINE")
    print("=" * 60)
    print(f"Mode:      {'SIMULATED DATA' if simulate else 'COLLECTED DATA'}")
    print(f"Output:    {output_dir}/")
    print()

    # ── Model 1: Target Classifier ────────────────────────────────────────
    print("\n[1/5] Target Classifier (Micro-Doppler CNN)")
    try:
        from training.trainers.train_target_classifier import train_target_classifier
        _, path = train_target_classifier(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            epochs=epochs_full,
            simulate_data=simulate
        )
        results["target_classifier"] = {"status": "OK", "path": path}
        print(f"  ✓ target_classifier.tflite")
    except Exception as e:
        log.error(f"Target classifier failed: {e}", exc_info=True)
        results["target_classifier"] = {"status": "FAILED", "error": str(e)}

    # ── Model 2: Activity Recogniser ──────────────────────────────────────
    print("\n[2/5] Activity Recogniser (CNN-LSTM hybrid)")
    try:
        from training.trainers.train_activity_recogniser import train_activity_recogniser
        _, path = train_activity_recogniser(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            epochs=epochs_full,
            simulate_data=simulate
        )
        results["activity_recogniser"] = {"status": "OK", "path": path}
        print(f"  ✓ activity_recogniser.tflite")
    except Exception as e:
        log.error(f"Activity recogniser failed: {e}", exc_info=True)
        results["activity_recogniser"] = {"status": "FAILED", "error": str(e)}

    # ── Model 3: RF Autoencoder ───────────────────────────────────────────
    print("\n[3/5] RF Environment Autoencoder (unsupervised)")
    try:
        from training.trainers.train_other_models import train_autoencoder
        _, path = train_autoencoder(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            epochs=epochs_full,
            simulate_data=simulate
        )
        results["rf_autoencoder"] = {"status": "OK", "path": path}
        print(f"  ✓ rf_autoencoder.tflite")
    except Exception as e:
        log.error(f"RF autoencoder failed: {e}", exc_info=True)
        results["rf_autoencoder"] = {"status": "FAILED", "error": str(e)}

    # ── Model 4: Vital Signs CNN ──────────────────────────────────────────
    print("\n[4/5] Vital Signs CNN (1D)")
    try:
        from training.trainers.train_other_models import train_vitals_cnn
        _, path = train_vitals_cnn(
            output_dir=output_dir,
            epochs=epochs_fast,
            simulate_data=True   # always simulate — needs labelled vitals
        )
        results["vitals_cnn"] = {"status": "OK", "path": path}
        print(f"  ✓ vitals_cnn.tflite")
    except Exception as e:
        log.error(f"Vitals CNN failed: {e}", exc_info=True)
        results["vitals_cnn"] = {"status": "FAILED", "error": str(e)}

    # ── Model 5: Gait Re-ID ───────────────────────────────────────────────
    print("\n[5/5] Siamese Gait Re-Identification")
    try:
        from training.trainers.train_other_models import train_siamese_reid
        _, _ = train_siamese_reid(
            output_dir=output_dir,
            epochs=epochs_fast,
            simulate_data=True
        )
        results["gait_reid"] = {"status": "OK",
                                "path": os.path.join(output_dir, "gait_embedder.h5")}
        print(f"  ✓ gait_embedder.h5")
    except Exception as e:
        log.error(f"Gait re-ID failed: {e}", exc_info=True)
        results["gait_reid"] = {"status": "FAILED", "error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — {elapsed/60:.1f} minutes")
    print("=" * 60)
    for name, res in results.items():
        status = "✓" if res["status"] == "OK" else "✗"
        print(f"  {status} {name:30s}  {res['status']}")

    # List all model files
    print(f"\nModel files in {output_dir}/:")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:40s}  {size/1024:.1f} KB")

    summary = {
        "trained_at": time.time(),
        "simulated": simulate,
        "elapsed_s": elapsed,
        "results": results
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return results


def train_one(model_name: str, simulate: bool, dataset_dir: str, output_dir: str):
    """Train a single model by name."""
    dispatch = {
        "target": lambda: __import__(
            "training.trainers.train_target_classifier",
            fromlist=["train_target_classifier"]
        ).train_target_classifier(dataset_dir, output_dir, simulate_data=simulate),

        "activity": lambda: __import__(
            "training.trainers.train_activity_recogniser",
            fromlist=["train_activity_recogniser"]
        ).train_activity_recogniser(dataset_dir, output_dir, simulate_data=simulate),

        "autoencoder": lambda: __import__(
            "training.trainers.train_other_models",
            fromlist=["train_autoencoder"]
        ).train_autoencoder(dataset_dir, output_dir, simulate_data=simulate),

        "vitals": lambda: __import__(
            "training.trainers.train_other_models",
            fromlist=["train_vitals_cnn"]
        ).train_vitals_cnn(output_dir=output_dir),

        "reid": lambda: __import__(
            "training.trainers.train_other_models",
            fromlist=["train_siamese_reid"]
        ).train_siamese_reid(output_dir=output_dir),
    }
    fn = dispatch.get(model_name)
    if fn:
        fn()
    else:
        print(f"Unknown model '{model_name}'. Choose from: {list(dispatch.keys())}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train all spatial intelligence models")
    p.add_argument("--simulate", action="store_true",
                   help="Use synthetic data (no hardware needed)")
    p.add_argument("--model", default="all",
                   choices=["all", "target", "activity", "autoencoder", "vitals", "reid"],
                   help="Which model to train")
    p.add_argument("--dataset-dir", default="training/datasets")
    p.add_argument("--output-dir",  default="models")
    p.add_argument("--epochs-fast", type=int, default=20)
    p.add_argument("--epochs-full", type=int, default=40)
    args = p.parse_args()

    if args.model == "all":
        train_all(simulate=args.simulate,
                  dataset_dir=args.dataset_dir,
                  output_dir=args.output_dir,
                  epochs_fast=args.epochs_fast,
                  epochs_full=args.epochs_full)
    else:
        train_one(args.model, args.simulate, args.dataset_dir, args.output_dir)
