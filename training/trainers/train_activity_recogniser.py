"""
Model 2 — Activity Recognition Engine
LSTM + CNN hybrid that classifies human activities from temporal radar patterns.

Activities: walking, running, sitting, standing, lying_down,
            falling, crawling, gesturing, person_still_breathing, person_hiding

Architecture:
  - Spatial branch:  CNN on per-frame spectrogram [33 x 20 x 1]
  - Temporal branch: Bidirectional LSTM on phase time-series [50 x 1]
  - Fusion:          Concatenate + Dense → softmax
  - Advantage:       CNN captures spectral shape, LSTM captures motion rhythm

Real-time inference on CM5: ~12ms per frame
"""

import numpy as np
import os
import sys
import json
import time
import collections
from typing import Optional, Dict, List
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

log = logging.getLogger(__name__)

ACTIVITY_LABEL_NAMES = [
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
ACTIVITY_LABEL_TO_IDX = {l: i for i, l in enumerate(ACTIVITY_LABEL_NAMES)}
IDX_TO_ACTIVITY       = {i: l for l, i in ACTIVITY_LABEL_TO_IDX.items()}
NUM_ACTIVITY_CLASSES  = len(ACTIVITY_LABEL_NAMES)

# Human-readable display names
ACTIVITY_DISPLAY = {
    "walking":                  "Walking",
    "running":                  "Running",
    "sitting":                  "Sitting",
    "standing":                 "Standing",
    "lying_down":               "Lying down",
    "falling":                  "FALLING",
    "crawling":                 "Crawling",
    "gesturing":                "Gesturing",
    "person_still_breathing":   "Still — breathing",
    "person_hiding":            "Hiding — very still",
}

# Alert-worthy activities
ALERT_ACTIVITIES = {"falling", "person_hiding", "crawling"}


# ── Model architecture ────────────────────────────────────────────────────────

def build_activity_model(spec_shape=(33, 20, 1),
                         seq_length=50,
                         num_classes=NUM_ACTIVITY_CLASSES):
    """
    Hybrid CNN-LSTM activity recognition model.

    spec_shape : shape of one STFT spectrogram frame
    seq_length : number of time steps in the phase sequence
    num_classes: number of activity classes
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    # ── Branch 1: CNN on spectrogram ──────────────────────────────────────
    spec_input = tf.keras.Input(shape=spec_shape, name="spectrogram")

    sx = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(spec_input)
    sx = layers.BatchNormalization()(sx)
    sx = layers.MaxPooling2D((2, 2))(sx)

    sx = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(sx)
    sx = layers.BatchNormalization()(sx)
    sx = layers.MaxPooling2D((2, 2))(sx)

    sx = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(sx)
    sx = layers.BatchNormalization()(sx)
    sx = layers.GlobalAveragePooling2D()(sx)

    sx = layers.Dense(64, activation='relu')(sx)
    sx = layers.Dropout(0.3)(sx)

    # ── Branch 2: Bidirectional LSTM on phase sequence ────────────────────
    seq_input = tf.keras.Input(shape=(seq_length, 1), name="phase_sequence")

    lx = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True))(seq_input)
    lx = layers.Dropout(0.3)(lx)
    lx = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False))(lx)
    lx = layers.Dense(32, activation='relu')(lx)
    lx = layers.Dropout(0.2)(lx)

    # ── Fusion ────────────────────────────────────────────────────────────
    fused = layers.Concatenate()([sx, lx])
    fused = layers.Dense(64, activation='relu')(fused)
    fused = layers.Dropout(0.3)(fused)
    fused = layers.Dense(32, activation='relu')(fused)
    output = layers.Dense(num_classes, activation='softmax', name="activity")(fused)

    model = Model(
        inputs=[spec_input, seq_input],
        outputs=output,
        name="activity_recogniser"
    )
    return model


# ── Trainer ───────────────────────────────────────────────────────────────────

def train_activity_recogniser(dataset_dir: str = "training/datasets",
                               output_dir: str = "models",
                               epochs: int = 40,
                               simulate_data: bool = False):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    print("=" * 55)
    print("TRAINING: Activity Recogniser (CNN-LSTM hybrid)")
    print("=" * 55)

    if simulate_data:
        print("Generating synthetic activity data...")
        specs, seqs, y = _generate_synthetic_activity_data(n_per_class=400)
    else:
        from training.data_collector.collect import load_dataset
        specs, y = load_dataset(dataset_dir, labels=ACTIVITY_LABEL_NAMES,
                                feature="spectrogram")
        seqs_raw, _ = load_dataset(dataset_dir, labels=ACTIVITY_LABEL_NAMES,
                                   feature="phase_sequence")
        seqs = seqs_raw[..., np.newaxis]   # [N x 50 x 1]

    # Remap y to local activity indices
    y_local = y % NUM_ACTIVITY_CLASSES

    if specs.ndim == 3:
        specs = specs[..., np.newaxis]
    specs = specs.astype(np.float32)
    specs = (specs - specs.mean(axis=(1, 2, 3), keepdims=True)) / \
            (specs.std(axis=(1, 2, 3), keepdims=True) + 1e-8)

    seqs = seqs.astype(np.float32)
    seqs = (seqs - seqs.mean(axis=1, keepdims=True)) / \
           (seqs.std(axis=1, keepdims=True) + 1e-8)

    (specs_tr, specs_vl,
     seqs_tr,  seqs_vl,
     y_tr,     y_vl) = train_test_split(
        specs, seqs, y_local, test_size=0.2,
        stratify=y_local, random_state=42)

    print(f"Train: {len(specs_tr)}  Val: {len(specs_vl)}")
    print(f"Classes: {NUM_ACTIVITY_CLASSES}  {ACTIVITY_LABEL_NAMES}")

    cw = compute_class_weight('balanced',
                              classes=np.unique(y_tr), y=y_tr)
    class_weight = dict(enumerate(cw))

    model = build_activity_model(
        spec_shape=specs.shape[1:],
        seq_length=seqs.shape[1],
        num_classes=NUM_ACTIVITY_CLASSES
    )
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs(output_dir, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=7, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=4, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "activity_best.h5"),
            save_best_only=True, verbose=0)
    ]

    history = model.fit(
        {"spectrogram": specs_tr, "phase_sequence": seqs_tr},
        y_tr,
        validation_data=(
            {"spectrogram": specs_vl, "phase_sequence": seqs_vl}, y_vl),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"\nBest validation accuracy: {val_acc:.4f}")

    # Confusion matrix
    _print_confusion_matrix(model, specs_vl, seqs_vl, y_vl)

    # Convert to TFLite
    tflite_path = _convert_activity_to_tflite(
        model, specs_tr[:100], seqs_tr[:100],
        os.path.join(output_dir, "activity_recogniser.tflite"))

    meta = {
        "model": "activity_recogniser",
        "labels": ACTIVITY_LABEL_NAMES,
        "display_names": ACTIVITY_DISPLAY,
        "alert_activities": list(ALERT_ACTIVITIES),
        "spec_shape": list(specs.shape[1:]),
        "seq_length": int(seqs.shape[1]),
        "val_accuracy": float(val_acc),
        "tflite_path": tflite_path
    }
    with open(os.path.join(output_dir, "activity_recogniser_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved → {tflite_path}")
    return model, tflite_path


def _print_confusion_matrix(model, specs, seqs, y_true):
    from sklearn.metrics import classification_report
    y_pred = np.argmax(
        model.predict({"spectrogram": specs, "phase_sequence": seqs}, verbose=0),
        axis=1)
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        target_names=ACTIVITY_LABEL_NAMES,
        zero_division=0))


def _generate_synthetic_activity_data(n_per_class=400):
    """Generate synthetic training data for all activity classes."""
    from training.data_collector.collect import DataCollector, FeatureExtractor

    extractor = FeatureExtractor()
    specs, seqs, labels = [], [], []

    # Activity-specific simulation parameters
    activity_params = {
        "walking": {
            "range_m": 4.0, "breath_hz": 0.3, "breath_amp": 0.006,
            "motion_hz": 1.8, "motion_amp": 0.08, "velocity": 0.8
        },
        "running": {
            "range_m": 5.0, "breath_hz": 0.5, "breath_amp": 0.01,
            "motion_hz": 3.0, "motion_amp": 0.15, "velocity": 2.5
        },
        "sitting": {
            "range_m": 3.0, "breath_hz": 0.25, "breath_amp": 0.005,
            "motion_hz": 0.0, "motion_amp": 0.001, "velocity": 0.0
        },
        "standing": {
            "range_m": 3.5, "breath_hz": 0.28, "breath_amp": 0.005,
            "motion_hz": 0.0, "motion_amp": 0.002, "velocity": 0.0
        },
        "lying_down": {
            "range_m": 2.0, "breath_hz": 0.18, "breath_amp": 0.004,
            "motion_hz": 0.0, "motion_amp": 0.001, "velocity": 0.0
        },
        "falling": {
            "range_m": 3.0, "breath_hz": 0.0, "breath_amp": 0.0,
            "motion_hz": 2.0, "motion_amp": 0.8, "velocity": 1.8,
            "transient": True    # brief burst then stillness
        },
        "crawling": {
            "range_m": 4.0, "breath_hz": 0.25, "breath_amp": 0.005,
            "motion_hz": 0.6, "motion_amp": 0.04, "velocity": 0.3
        },
        "gesturing": {
            "range_m": 2.0, "breath_hz": 0.28, "breath_amp": 0.005,
            "motion_hz": 4.0, "motion_amp": 0.05, "velocity": 0.5
        },
        "person_still_breathing": {
            "range_m": 3.0, "breath_hz": 0.22, "breath_amp": 0.006,
            "motion_hz": 0.0, "motion_amp": 0.0005, "velocity": 0.0
        },
        "person_hiding": {
            "range_m": 5.0, "breath_hz": 0.20, "breath_amp": 0.003,
            "motion_hz": 0.0, "motion_amp": 0.0002, "velocity": 0.0
        },
    }

    N, M = 256, 128
    c = 3e8
    B = 0.25e9
    f0 = 24e9

    for label, params in activity_params.items():
        idx = ACTIVITY_LABEL_TO_IDX[label]
        for sample_i in range(n_per_class):
            # Build a sequence of 50 I/Q frames for this sample
            phase_seq = []
            last_spec = None

            for frame_i in range(50):
                t = frame_i * 0.1 + sample_i * 0.01  # time offset

                r = params["range_m"] + np.random.randn() * 0.3
                r += params["breath_amp"] * np.sin(2 * np.pi * params["breath_hz"] * t)

                if params.get("transient") and frame_i < 5:
                    # Fall: sharp transient at start
                    r += params["motion_amp"] * np.exp(-frame_i * 0.5)
                elif params["motion_hz"] > 0:
                    r += params["motion_amp"] * np.sin(2 * np.pi * params["motion_hz"] * t)

                r = max(r, 0.3)  # physical minimum range

                # Build I/Q
                iq = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * 0.03
                fb = 2 * B * r / (c * 40e-6)
                t_fast = np.linspace(0, 40e-6, N)
                for chirp in range(M):
                    iq[chirp] += 0.8 * np.exp(
                        1j * (2 * np.pi * fb * t_fast + 4 * np.pi * f0 * r / c))

                frame = extractor.extract(iq, label)
                phase_seq.append(float(frame.phase_sequence[-1]))
                last_spec = frame.spectrogram

            phase_arr = np.unwrap(np.array(phase_seq))
            phase_arr = (phase_arr - phase_arr.mean()) / (phase_arr.std() + 1e-8)

            if last_spec is not None:
                spec_norm = (last_spec - last_spec.mean()) / (last_spec.std() + 1e-8)
                specs.append(spec_norm[..., np.newaxis])
                seqs.append(phase_arr[:, np.newaxis])
                labels.append(idx)

    specs  = np.array(specs,  dtype=np.float32)
    seqs   = np.array(seqs,   dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Shuffle
    idx_arr = np.random.permutation(len(labels))
    return specs[idx_arr], seqs[idx_arr], labels[idx_arr]


def _convert_activity_to_tflite(model, rep_specs, rep_seqs, output_path):
    import tensorflow as tf
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(min(50, len(rep_specs))):
            yield [rep_specs[i:i+1], rep_seqs[i:i+1]]

    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Activity TFLite: {len(tflite_model)/1024:.1f} KB → {output_path}")
    return output_path


# ── Real-time inference engine ────────────────────────────────────────────────

class ActivityRecogniser:
    """
    Real-time activity recogniser that plugs into the main pipeline.
    Maintains per-target rolling buffers and runs TFLite inference.
    """

    def __init__(self, model_path: str = "models/activity_recogniser.tflite",
                 meta_path:  str = "models/activity_recogniser_meta.json"):
        self._interpreter = None
        self._keras_model = None
        self._meta = None
        self.model_path = model_path
        self.meta_path  = meta_path
        self.loaded = False

        # Per-target rolling state buffers
        self._spec_buffers  = {}   # target_id → deque of spectrograms
        self._phase_buffers = {}   # target_id → deque of phase values
        self._result_buffers= {}   # target_id → deque of recent predictions
        self._seq_len = 50
        self._smoothing_window = 5  # majority vote over last N predictions

    def load(self) -> bool:
        """Load TFLite model. Falls back to Keras .h5 if TFLite not found."""
        # Load metadata
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                self._meta = json.load(f)
            self._seq_len = self._meta.get("seq_length", 50)

        if os.path.exists(self.model_path):
            try:
                import tflite_runtime.interpreter as tflite
                self._interpreter = tflite.Interpreter(model_path=self.model_path)
            except ImportError:
                import tensorflow as tf
                self._interpreter = tf.lite.Interpreter(model_path=self.model_path)

            self._interpreter.allocate_tensors()
            self._input_details  = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self.loaded = True
            log.info(f"Activity model loaded from {self.model_path}")
            return True

        # Try Keras .h5
        h5_path = self.model_path.replace(".tflite", ".h5").replace(
            "activity_recogniser", "activity_best")
        if os.path.exists(h5_path):
            import tensorflow as tf
            self._keras_model = tf.keras.models.load_model(h5_path)
            self.loaded = True
            log.info(f"Activity Keras model loaded from {h5_path}")
            return True

        log.warning(f"No activity model found at {self.model_path} — using heuristics")
        return False

    def update(self, target_id: str, spectrogram: np.ndarray,
               phase_value: float) -> Optional['ActivityResult']:
        """
        Feed one frame for a target. Returns ActivityResult when enough
        data accumulated, else None.

        spectrogram: [33 x 20] float array
        phase_value: scalar phase reading for this frame
        """
        # Initialise buffers
        if target_id not in self._spec_buffers:
            self._spec_buffers[target_id]   = collections.deque(maxlen=self._seq_len)
            self._phase_buffers[target_id]  = collections.deque(maxlen=self._seq_len)
            self._result_buffers[target_id] = collections.deque(maxlen=self._smoothing_window)

        self._spec_buffers[target_id].append(spectrogram)
        self._phase_buffers[target_id].append(phase_value)

        # Need full sequence before inference
        min_frames = max(16, self._seq_len // 3)
        if len(self._phase_buffers[target_id]) < min_frames:
            return None

        # Run inference
        if self.loaded:
            result = self._run_inference(target_id)
        else:
            result = self._heuristic_fallback(target_id)

        if result:
            self._result_buffers[target_id].append(result.label)
            result.smoothed_label = self._smooth_prediction(
                self._result_buffers[target_id])
            result.is_alert = result.smoothed_label in ALERT_ACTIVITIES

        return result

    def _run_inference(self, target_id: str) -> 'ActivityResult':
        """Run TFLite or Keras inference."""
        # Build spectrogram input [1 x 33 x 20 x 1]
        spec_buf = list(self._spec_buffers[target_id])
        spec = spec_buf[-1].astype(np.float32)
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        spec_in = spec[np.newaxis, ..., np.newaxis]

        # Build phase sequence [1 x seq_len x 1]
        phase_buf = np.array(list(self._phase_buffers[target_id]))
        phase_unwrapped = np.unwrap(phase_buf)
        phase_norm = (phase_unwrapped - phase_unwrapped.mean()) / \
                     (phase_unwrapped.std() + 1e-8)
        # Pad to seq_len
        if len(phase_norm) < self._seq_len:
            pad = np.zeros(self._seq_len - len(phase_norm))
            phase_norm = np.concatenate([pad, phase_norm])
        else:
            phase_norm = phase_norm[-self._seq_len:]
        seq_in = phase_norm[np.newaxis, :, np.newaxis].astype(np.float32)

        if self._interpreter is not None:
            # TFLite inference
            # Find which input is spec vs seq by shape
            probs = self._tflite_infer(spec_in, seq_in)
        else:
            # Keras inference
            probs = self._keras_model.predict(
                {"spectrogram": spec_in, "phase_sequence": seq_in},
                verbose=0)[0]

        label_idx = int(np.argmax(probs))
        confidence = float(probs[label_idx])
        label = IDX_TO_ACTIVITY.get(label_idx, "unknown")

        return ActivityResult(
            target_id=target_id,
            label=label,
            label_idx=label_idx,
            confidence=confidence,
            probabilities={IDX_TO_ACTIVITY[i]: float(p)
                           for i, p in enumerate(probs)},
            timestamp=time.time()
        )

    def _tflite_infer(self, spec_in, seq_in):
        """Run TFLite inference with two inputs."""
        for detail in self._input_details:
            shape = detail['shape']
            if shape[-1] == 1 and len(shape) == 4:
                self._interpreter.set_tensor(detail['index'], spec_in)
            elif len(shape) == 3:
                self._interpreter.set_tensor(detail['index'], seq_in)

        self._interpreter.invoke()
        probs = self._interpreter.get_tensor(
            self._output_details[0]['index'])[0]
        return probs

    def _heuristic_fallback(self, target_id: str) -> 'ActivityResult':
        """Simple heuristic when no model is loaded — uses phase variance."""
        phase_buf = np.array(list(self._phase_buffers[target_id]))
        phase_var = float(np.var(np.diff(np.unwrap(phase_buf))))

        if phase_var > 0.5:
            label, conf = "walking", min(0.7, phase_var)
        elif phase_var > 0.1:
            label, conf = "gesturing", 0.5
        elif phase_var > 0.005:
            label, conf = "sitting", 0.55
        else:
            label, conf = "person_still_breathing", 0.6

        idx = ACTIVITY_LABEL_TO_IDX.get(label, 0)
        probs = np.zeros(NUM_ACTIVITY_CLASSES)
        probs[idx] = conf
        return ActivityResult(
            target_id=target_id,
            label=label,
            label_idx=idx,
            confidence=conf,
            probabilities={IDX_TO_ACTIVITY[i]: float(p)
                           for i, p in enumerate(probs)},
            timestamp=time.time(),
            from_heuristic=True
        )

    def _smooth_prediction(self, result_deque) -> str:
        """Majority vote over recent predictions for stability."""
        if not result_deque:
            return "unknown"
        from collections import Counter
        counts = Counter(result_deque)
        return counts.most_common(1)[0][0]

    def clear_target(self, target_id: str):
        self._spec_buffers.pop(target_id, None)
        self._phase_buffers.pop(target_id, None)
        self._result_buffers.pop(target_id, None)

    def get_all_activities(self) -> dict:
        """Return latest activity for every tracked target."""
        return {}  # populated by main pipeline


from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ActivityResult:
    target_id: str
    label: str
    label_idx: int
    confidence: float
    probabilities: Dict[str, float]
    timestamp: float
    smoothed_label: str = ""
    is_alert: bool = False
    from_heuristic: bool = False
    display_name: str = ""

    def __post_init__(self):
        if not self.smoothed_label:
            self.smoothed_label = self.label
        self.display_name = ACTIVITY_DISPLAY.get(self.label, self.label.replace("_", " ").title())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--dataset-dir", default="training/datasets")
    p.add_argument("--output-dir", default="models")
    args = p.parse_args()
    train_activity_recogniser(args.dataset_dir, args.output_dir,
                              args.epochs, args.simulate)
