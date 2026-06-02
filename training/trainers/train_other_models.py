"""
Model 3 — Autoencoder RF Environment Fingerprint (unsupervised)
Model 4 — 1D CNN Vital Signs Refiner
Model 5 — Siamese Network Gait Re-identification
"""

import numpy as np
import os
import sys
import json
import time
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — AUTOENCODER RF FINGERPRINT
# Unsupervised — learns "normal" environment, high reconstruction error = anomaly
# ═══════════════════════════════════════════════════════════════════════════════

def build_autoencoder(input_dim: int = 52, latent_dim: int = 8):
    """
    Fully connected autoencoder for CSI anomaly detection.
    Trains on normal environment data only — no labels needed.
    Anomaly score = mean squared reconstruction error.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = tf.keras.Input(shape=(input_dim,), name="csi_vector")

    # Encoder
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    latent = layers.Dense(latent_dim, activation='relu', name="latent")(x)

    # Decoder
    x = layers.Dense(32, activation='relu')(latent)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    recon = layers.Dense(input_dim, activation='linear', name="reconstruction")(x)

    autoencoder = Model(inp, recon, name="rf_autoencoder")
    encoder     = Model(inp, latent, name="rf_encoder")
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def train_autoencoder(dataset_dir: str = "training/datasets",
                      output_dir: str = "models",
                      epochs: int = 80,
                      simulate_data: bool = False):
    import tensorflow as tf

    print("=" * 50)
    print("TRAINING: RF Autoencoder (unsupervised)")
    print("=" * 50)

    if simulate_data:
        # Normal environment — empty + low activity CSI proxy
        n = 5000
        X = np.random.randn(n, 52) * 0.1 + 0.5
        X += np.sin(np.linspace(0, 4*np.pi, 52))[np.newaxis] * 0.2
    else:
        from training.data_collector.collect import load_dataset
        X, _ = load_dataset(dataset_dir, labels=["empty"],
                            feature="csi_proxy")

    X = X.astype(np.float32)
    # Normalise
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    input_dim = X.shape[1]
    ae, encoder = build_autoencoder(input_dim=input_dim, latent_dim=8)
    ae.summary()

    # Threshold: 99th percentile of training reconstruction error
    # (computed after training to set the anomaly threshold)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    ae.fit(X_norm, X_norm, epochs=epochs, batch_size=64,
           validation_split=0.1, callbacks=callbacks, verbose=1)

    # Compute training reconstruction error for threshold
    recon = ae.predict(X_norm, verbose=0)
    errors = np.mean((X_norm - recon) ** 2, axis=1)
    threshold = float(np.percentile(errors, 99))
    print(f"Anomaly threshold (99th pct): {threshold:.6f}")

    os.makedirs(output_dir, exist_ok=True)

    # Save normalisation stats + threshold
    meta = {
        "model": "rf_autoencoder",
        "input_dim": input_dim,
        "latent_dim": 8,
        "anomaly_threshold": threshold,
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
    }
    with open(os.path.join(output_dir, "autoencoder_meta.json"), "w") as f:
        json.dump(meta, f)

    # Convert to TFLite
    tflite_path = os.path.join(output_dir, "rf_autoencoder.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(ae)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"Autoencoder TFLite: {len(tflite_bytes)/1024:.1f} KB → {tflite_path}")

    return ae, tflite_path


class AutoencoderAnomalyDetector:
    """Runtime anomaly detection using the trained autoencoder."""

    def __init__(self, model_path="models/rf_autoencoder.tflite",
                 meta_path="models/autoencoder_meta.json"):
        self._interpreter = None
        self._meta = None
        self._threshold = 0.05
        self._X_mean = None
        self._X_std  = None
        self.loaded = False
        self.model_path = model_path
        self.meta_path  = meta_path

    def load(self) -> bool:
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                self._meta = json.load(f)
            self._threshold = self._meta["anomaly_threshold"]
            self._X_mean = np.array(self._meta["X_mean"])
            self._X_std  = np.array(self._meta["X_std"])

        if os.path.exists(self.model_path):
            try:
                import tflite_runtime.interpreter as tflite
                self._interpreter = tflite.Interpreter(model_path=self.model_path)
            except ImportError:
                import tensorflow as tf
                self._interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self._interpreter.allocate_tensors()
            self._in_detail  = self._interpreter.get_input_details()[0]
            self._out_detail = self._interpreter.get_output_details()[0]
            self.loaded = True
            log.info("RF autoencoder loaded")
            return True
        return False

    def score(self, csi_vector: np.ndarray) -> float:
        """Return anomaly score 0–1. >0.5 is anomalous."""
        if not self.loaded or self._X_mean is None:
            return 0.0
        x = (csi_vector - self._X_mean) / self._X_std
        x_in = x[np.newaxis].astype(np.float32)
        self._interpreter.set_tensor(self._in_detail['index'], x_in)
        self._interpreter.invoke()
        recon = self._interpreter.get_tensor(self._out_detail['index'])[0]
        mse = float(np.mean((x - recon) ** 2))
        return float(np.clip(mse / (self._threshold * 2), 0, 1))

    def is_anomalous(self, csi_vector: np.ndarray) -> bool:
        return self.score(csi_vector) > 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — 1D CNN VITAL SIGNS REFINER
# More robust than FFT — handles motion artefacts better
# ═══════════════════════════════════════════════════════════════════════════════

def build_vitals_cnn(seq_length: int = 100, num_outputs: int = 2):
    """
    1D CNN that regresses breathing rate and heart rate directly from
    a phase time-series. More robust than FFT for noisy / moving subjects.

    Input:  [seq_length x 1] normalised phase sequence
    Output: [breath_rate_bpm, heart_rate_bpm]
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = tf.keras.Input(shape=(seq_length, 1), name="phase_signal")

    x = layers.Conv1D(32, 7, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    breath_out = layers.Dense(1, activation='linear', name="breath_rate")(x)
    hr_out     = layers.Dense(1, activation='linear', name="heart_rate")(x)

    model = Model(inp, [breath_out, hr_out], name="vitals_cnn")
    model.compile(
        optimizer='adam',
        loss={'breath_rate': 'mse', 'heart_rate': 'mse'},
        loss_weights={'breath_rate': 1.0, 'heart_rate': 1.0},
        metrics={'breath_rate': 'mae', 'heart_rate': 'mae'}
    )
    return model


def train_vitals_cnn(output_dir="models", epochs=60, simulate_data=True):
    import tensorflow as tf

    print("=" * 50)
    print("TRAINING: Vital Signs CNN")
    print("=" * 50)

    seq_len = 100
    # Synthetic training: generate phase sequences with known breath/HR
    n = 3000
    X, y_breath, y_hr = [], [], []

    for _ in range(n):
        br = np.random.uniform(10, 30)     # breaths/min
        hr = np.random.uniform(50, 100)    # BPM
        t = np.linspace(0, 10, seq_len)   # 10 seconds

        # Ground truth signals
        breath_sig = 0.4 * np.sin(2 * np.pi * (br/60) * t)
        heart_sig  = 0.02 * np.sin(2 * np.pi * (hr/60) * t)

        # Motion artefact
        motion = 0.1 * np.random.randn() * np.sin(2 * np.pi * 0.2 * t + np.random.rand())

        # Noise
        noise = np.random.randn(seq_len) * 0.05

        phase = breath_sig + heart_sig + motion + noise
        X.append(phase)
        y_breath.append(br)
        y_hr.append(hr)

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y_breath = np.array(y_breath, dtype=np.float32)
    y_hr     = np.array(y_hr, dtype=np.float32)

    split = int(0.8 * n)
    model = build_vitals_cnn(seq_len)
    model.fit(
        X[:split],
        {'breath_rate': y_breath[:split], 'heart_rate': y_hr[:split]},
        validation_data=(
            X[split:],
            {'breath_rate': y_breath[split:], 'heart_rate': y_hr[split:]}),
        epochs=epochs, batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
        verbose=1
    )

    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "vitals_cnn.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    with open(tflite_path, "wb") as f: f.write(tflite_bytes)
    print(f"Vitals CNN TFLite: {len(tflite_bytes)/1024:.1f} KB → {tflite_path}")
    return model, tflite_path


class VitalsCNNEstimator:
    """Runtime vital signs estimation using 1D CNN."""

    def __init__(self, model_path="models/vitals_cnn.tflite"):
        self._interpreter = None
        self.loaded = False
        self.model_path = model_path
        self._phase_buffers = {}
        self._seq_len = 100

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        try:
            import tflite_runtime.interpreter as tflite
            self._interpreter = tflite.Interpreter(model_path=self.model_path)
        except ImportError:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self._interpreter.allocate_tensors()
        self._in_d  = self._interpreter.get_input_details()[0]
        self._out_d = self._interpreter.get_output_details()
        self.loaded = True
        log.info("Vitals CNN loaded")
        return True

    def update(self, target_id: str, phase_value: float):
        if target_id not in self._phase_buffers:
            self._phase_buffers[target_id] = []
        self._phase_buffers[target_id].append(phase_value)
        if len(self._phase_buffers[target_id]) > self._seq_len:
            self._phase_buffers[target_id].pop(0)

        if len(self._phase_buffers[target_id]) < self._seq_len // 2:
            return None

        buf = np.array(self._phase_buffers[target_id])
        buf = np.unwrap(buf)
        buf = (buf - buf.mean()) / (buf.std() + 1e-8)
        if len(buf) < self._seq_len:
            buf = np.pad(buf, (self._seq_len - len(buf), 0))
        buf = buf[-self._seq_len:]

        x = buf[np.newaxis, :, np.newaxis].astype(np.float32)
        self._interpreter.set_tensor(self._in_d['index'], x)
        self._interpreter.invoke()

        breath = float(self._interpreter.get_tensor(self._out_d[0]['index'])[0][0])
        hr     = float(self._interpreter.get_tensor(self._out_d[1]['index'])[0][0])
        return {
            "breath_rate_bpm": round(np.clip(breath, 6, 40), 1),
            "heart_rate_bpm":  round(np.clip(hr, 40, 120), 1)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 5 — SIAMESE NETWORK PERSON RE-IDENTIFICATION
# Learns whether two gait signatures came from the same individual
# ═══════════════════════════════════════════════════════════════════════════════

def build_siamese_network(input_shape=(50, 1)):
    """
    Siamese network for gait-based person re-identification.
    Two identical sub-networks share weights.
    Output: similarity score 0 (different) – 1 (same person).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    def build_embedding_net(input_shape):
        inp = tf.keras.Input(shape=input_shape)
        x = layers.Conv1D(32, 5, activation='relu', padding='same')(inp)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1),
                          name="embedding")(x)
        return Model(inp, x, name="gait_embedder")

    embedder = build_embedding_net(input_shape)

    inp_a = tf.keras.Input(shape=input_shape, name="gait_a")
    inp_b = tf.keras.Input(shape=input_shape, name="gait_b")

    emb_a = embedder(inp_a)
    emb_b = embedder(inp_b)

    # Cosine similarity
    similarity = layers.Dot(axes=1, normalize=True,
                            name="similarity")([emb_a, emb_b])

    siamese = Model([inp_a, inp_b], similarity, name="siamese_reid")
    siamese.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Also return embedder for runtime use (embed → compare)
    return siamese, embedder


def train_siamese_reid(output_dir="models", epochs=40, simulate_data=True):
    import tensorflow as tf

    print("=" * 50)
    print("TRAINING: Siamese Gait Re-ID")
    print("=" * 50)

    seq_len = 50
    n_persons = 20
    n_seqs_per_person = 100

    # Generate synthetic gait signatures
    # Each person has a unique gait frequency + amplitude
    person_params = [(np.random.uniform(0.8, 2.5),   # step freq
                      np.random.uniform(0.03, 0.12))  # step amplitude
                     for _ in range(n_persons)]

    def make_gait(person_id, n=seq_len):
        freq, amp = person_params[person_id]
        t = np.linspace(0, 5, n)
        sig  = amp * np.sin(2 * np.pi * freq * t)
        sig += (amp * 0.3) * np.sin(2 * np.pi * freq * 2 * t)
        sig += np.random.randn(n) * amp * 0.2
        return sig.astype(np.float32)

    # Build pairs
    seqs_a, seqs_b, labels = [], [], []
    for _ in range(4000):
        if np.random.random() > 0.5:
            # Same person
            p = np.random.randint(n_persons)
            seqs_a.append(make_gait(p))
            seqs_b.append(make_gait(p))
            labels.append(1.0)
        else:
            # Different people
            p1, p2 = np.random.choice(n_persons, 2, replace=False)
            seqs_a.append(make_gait(p1))
            seqs_b.append(make_gait(p2))
            labels.append(0.0)

    A = np.array(seqs_a)[..., np.newaxis]
    B = np.array(seqs_b)[..., np.newaxis]
    Y = np.array(labels, dtype=np.float32)

    siamese, embedder = build_siamese_network((seq_len, 1))
    siamese.fit(
        {"gait_a": A, "gait_b": B}, Y,
        validation_split=0.2, epochs=epochs, batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
        verbose=1
    )

    os.makedirs(output_dir, exist_ok=True)
    embedder.save(os.path.join(output_dir, "gait_embedder.h5"))
    print(f"Gait embedder saved → {output_dir}/gait_embedder.h5")
    return siamese, embedder


class GaitReIdentifier:
    """
    Runtime person re-identification using gait embeddings.
    Compares new gait signatures against a gallery of known individuals.
    """

    def __init__(self, model_path="models/gait_embedder.h5"):
        self._model = None
        self.loaded = False
        self.model_path = model_path
        self._gallery = {}    # name → embedding vector
        self._threshold = 0.85   # cosine similarity threshold

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        import tensorflow as tf
        self._model = tf.keras.models.load_model(self.model_path)
        self.loaded = True
        log.info("Gait re-ID model loaded")
        return True

    def enroll(self, name: str, phase_sequences: list):
        """Enrol a known person by averaging their gait embeddings."""
        if not self.loaded: return
        embeddings = []
        for seq in phase_sequences:
            seq = np.array(seq)
            seq = (seq - seq.mean()) / (seq.std() + 1e-8)
            seq_in = seq[np.newaxis, :, np.newaxis].astype(np.float32)
            emb = self._model.predict(seq_in, verbose=0)[0]
            embeddings.append(emb)
        self._gallery[name] = np.mean(embeddings, axis=0)
        log.info(f"Enrolled '{name}' in gait gallery")

    def identify(self, phase_sequence: np.ndarray) -> tuple:
        """
        Returns (name, similarity) for best match, or ("unknown", score).
        """
        if not self.loaded or not self._gallery:
            return "unknown", 0.0

        seq = np.unwrap(phase_sequence)
        seq = (seq - seq.mean()) / (seq.std() + 1e-8)
        seq_in = seq[np.newaxis, :, np.newaxis].astype(np.float32)
        emb = self._model.predict(seq_in, verbose=0)[0]

        best_name, best_sim = "unknown", 0.0
        for name, gallery_emb in self._gallery.items():
            sim = float(np.dot(emb, gallery_emb) /
                        (np.linalg.norm(emb) * np.linalg.norm(gallery_emb) + 1e-10))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim < self._threshold:
            return "unknown", best_sim
        return best_name, best_sim


# ── Train all models ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["autoencoder","vitals","reid","all"],
                   default="all")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()

    if args.model in ("autoencoder", "all"):
        train_autoencoder(output_dir=args.output_dir,
                          epochs=args.epochs, simulate_data=True)
    if args.model in ("vitals", "all"):
        train_vitals_cnn(output_dir=args.output_dir,
                         epochs=args.epochs, simulate_data=True)
    if args.model in ("reid", "all"):
        train_siamese_reid(output_dir=args.output_dir,
                           epochs=args.epochs, simulate_data=True)
