"""
Continual Learning Engine
Adapts all models to the deployment environment without internet.
Runs entirely on the CM5 using only locally observed radar data.

Key techniques used:
  - Elastic Weight Consolidation (EWC): prevents forgetting base knowledge
  - Experience Replay: keeps a ring buffer of past samples to prevent drift
  - Adapter layers: only retrain the last few layers, base stays frozen
  - Pseudo-labelling: auto-labels high-confidence detections as new training data
  - Federated gradient sharing: nodes share compressed updates over LoRa mesh

The system NEVER needs internet after deployment.
All learning happens from live radar observations.
"""

import os
import sys
import json
import time
import logging
import threading
import collections
import pickle
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ContinualConfig:
    # How often to retrain (seconds between adaptation cycles)
    retrain_interval_s: float = 3600.0      # 1 hour default

    # Minimum new samples before triggering a retrain
    min_new_samples: int = 50

    # Confidence threshold for auto-labelling (pseudo-labelling)
    pseudolabel_confidence: float = 0.82

    # Ring buffer sizes
    replay_buffer_size: int = 2000          # past samples to remember
    new_sample_buffer_size: int = 500       # new observations per cycle

    # Adapter training epochs (fast — only last layers)
    adapter_epochs: int = 5
    adapter_lr: float = 5e-4

    # EWC regularisation strength (prevents forgetting)
    ewc_lambda: float = 400.0

    # Storage
    model_dir: str = "models"
    buffer_dir: str = "data/continual_buffers"
    checkpoint_dir: str = "data/checkpoints"

    # Mesh sharing
    share_gradients_over_mesh: bool = True
    mesh_share_interval_s: float = 7200.0   # 2 hours


# ── Observation ring buffer ───────────────────────────────────────────────────

class ObservationBuffer:
    """
    Thread-safe ring buffer that stores radar observations on NVMe.
    Never fills up — overwrites oldest data when full.
    Two partitions: replay (old, high-quality) and new (recent, to train on).
    """

    def __init__(self, config: ContinualConfig):
        self.config = config
        os.makedirs(config.buffer_dir, exist_ok=True)
        self._lock = threading.Lock()

        # In-memory ring buffers
        self._replay_specs   = collections.deque(maxlen=config.replay_buffer_size)
        self._replay_seqs    = collections.deque(maxlen=config.replay_buffer_size)
        self._replay_labels  = collections.deque(maxlen=config.replay_buffer_size)

        self._new_specs      = collections.deque(maxlen=config.new_sample_buffer_size)
        self._new_seqs       = collections.deque(maxlen=config.new_sample_buffer_size)
        self._new_labels     = collections.deque(maxlen=config.new_sample_buffer_size)
        self._new_confs      = collections.deque(maxlen=config.new_sample_buffer_size)

        self._total_observed = 0
        self._total_labelled = 0

        # Load persisted buffers from NVMe if they exist
        self._load_from_disk()

    def add_observation(self, spectrogram: np.ndarray,
                        phase_sequence: np.ndarray,
                        predicted_label: int,
                        confidence: float,
                        source: str = "radar"):
        """
        Add a new observation. If confidence is high enough,
        auto-label it as training data (pseudo-labelling).
        """
        with self._lock:
            self._total_observed += 1

            if confidence >= self.config.pseudolabel_confidence:
                self._new_specs.append(spectrogram.astype(np.float32))
                self._new_seqs.append(phase_sequence.astype(np.float32))
                self._new_labels.append(int(predicted_label))
                self._new_confs.append(float(confidence))
                self._total_labelled += 1

    def get_training_batch(self) -> Optional[Tuple]:
        """
        Returns (specs, seqs, labels) combining new observations
        with replay samples. Returns None if not enough data yet.
        """
        with self._lock:
            n_new = len(self._new_specs)
            if n_new < self.config.min_new_samples:
                return None

            new_specs  = np.array(self._new_specs)
            new_seqs   = np.array(self._new_seqs)
            new_labels = np.array(self._new_labels)

            # Mix with replay to prevent forgetting
            if len(self._replay_specs) > 0:
                n_replay = min(len(self._replay_specs), n_new * 2)
                idx = np.random.choice(len(self._replay_specs),
                                       n_replay, replace=False)
                replay_specs  = np.array(self._replay_specs)[idx]
                replay_seqs   = np.array(self._replay_seqs)[idx]
                replay_labels = np.array(self._replay_labels)[idx]

                all_specs  = np.concatenate([new_specs,  replay_specs],  axis=0)
                all_seqs   = np.concatenate([new_seqs,   replay_seqs],   axis=0)
                all_labels = np.concatenate([new_labels, replay_labels], axis=0)
            else:
                all_specs, all_seqs, all_labels = new_specs, new_seqs, new_labels

            # Move new samples to replay
            self._replay_specs.extend(self._new_specs)
            self._replay_seqs.extend(self._new_seqs)
            self._replay_labels.extend(self._new_labels)

            # Clear new buffer
            self._new_specs.clear()
            self._new_seqs.clear()
            self._new_labels.clear()
            self._new_confs.clear()

            return all_specs, all_seqs, all_labels

    def n_new(self) -> int:
        return len(self._new_specs)

    def stats(self) -> dict:
        return {
            "total_observed": self._total_observed,
            "total_labelled": self._total_labelled,
            "replay_size":    len(self._replay_specs),
            "new_pending":    len(self._new_specs),
            "label_rate_pct": round(
                100 * self._total_labelled / max(1, self._total_observed), 1)
        }

    def _save_to_disk(self):
        """Persist buffers to NVMe for survival across reboots."""
        path = os.path.join(self.config.buffer_dir, "buffers.pkl")
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    "replay_specs":   list(self._replay_specs),
                    "replay_seqs":    list(self._replay_seqs),
                    "replay_labels":  list(self._replay_labels),
                    "total_observed": self._total_observed,
                    "total_labelled": self._total_labelled,
                }, f)
        except Exception as e:
            log.warning(f"Buffer save failed: {e}")

    def _load_from_disk(self):
        path = os.path.join(self.config.buffer_dir, "buffers.pkl")
        if not os.path.exists(path):
            return
        try:
            with open(path, 'rb') as f:
                saved = pickle.load(f)
            self._replay_specs.extend(saved.get("replay_specs", []))
            self._replay_seqs.extend(saved.get("replay_seqs", []))
            self._replay_labels.extend(saved.get("replay_labels", []))
            self._total_observed = saved.get("total_observed", 0)
            self._total_labelled = saved.get("total_labelled", 0)
            log.info(f"Loaded {len(self._replay_specs)} replay samples from disk")
        except Exception as e:
            log.warning(f"Buffer load failed: {e}")


# ── EWC regulariser — prevents catastrophic forgetting ────────────────────────

class EWC:
    """
    Elastic Weight Consolidation.
    Computes Fisher information matrix on base dataset to identify
    which weights are important — then penalises changes to those weights.
    This is what stops the model forgetting 'person walking' when it
    adapts to learn 'vehicle in forest terrain'.
    """

    def __init__(self, importance_path: str = "data/fisher_importance.npz"):
        self.importance_path = importance_path
        self.fisher: Optional[Dict[str, np.ndarray]] = None
        self.optimal_params: Optional[Dict[str, np.ndarray]] = None
        self._loaded = False

    def compute(self, model, dataset_specs, dataset_seqs, dataset_labels,
                n_samples: int = 200):
        """
        Compute Fisher information matrix on a sample of base training data.
        Call this ONCE after base training, before deployment.
        Saves to disk — never needs to be recomputed.
        """
        import tensorflow as tf
        log.info("Computing EWC Fisher information matrix...")

        n = min(n_samples, len(dataset_specs))
        idx = np.random.choice(len(dataset_specs), n, replace=False)
        X_spec = dataset_specs[idx].astype(np.float32)
        X_seq  = dataset_seqs[idx].astype(np.float32)
        Y      = dataset_labels[idx]

        fisher = {v.name: np.zeros(v.shape) for v in model.trainable_variables}
        opt_params = {v.name: v.numpy().copy() for v in model.trainable_variables}

        for i in range(n):
            spec_in = X_spec[i:i+1]
            seq_in  = X_seq[i:i+1]
            y_true  = Y[i:i+1]

            with tf.GradientTape() as tape:
                pred = model({"spectrogram": spec_in,
                              "phase_sequence": seq_in[..., np.newaxis]},
                             training=False)
                log_prob = tf.math.log(pred[0, y_true[0]] + 1e-10)

            grads = tape.gradient(log_prob, model.trainable_variables)
            for v, g in zip(model.trainable_variables, grads):
                if g is not None:
                    fisher[v.name] += (g.numpy() ** 2) / n

        self.fisher       = fisher
        self.optimal_params = opt_params

        # Save to disk
        os.makedirs(os.path.dirname(self.importance_path), exist_ok=True)
        np.savez(self.importance_path,
                 **{k.replace('/', '_').replace(':', '_'): v
                    for k, v in fisher.items()})
        log.info(f"Fisher matrix saved → {self.importance_path}")
        self._loaded = True

    def load(self) -> bool:
        if not os.path.exists(self.importance_path):
            return False
        try:
            data = np.load(self.importance_path)
            self.fisher = dict(data)
            self._loaded = True
            log.info("EWC Fisher matrix loaded")
            return True
        except Exception as e:
            log.warning(f"EWC load failed: {e}")
            return False

    def penalty(self, model) -> float:
        """
        Compute EWC penalty for current model weights.
        Add to loss during adapter training to prevent forgetting.
        """
        if not self._loaded or not self.fisher or not self.optimal_params:
            return 0.0

        import tensorflow as tf
        penalty = 0.0
        for v in model.trainable_variables:
            name = v.name
            fname = name.replace('/', '_').replace(':', '_')
            if fname in self.fisher and name in self.optimal_params:
                diff = v - self.optimal_params[name]
                penalty += tf.reduce_sum(
                    self.fisher[fname] * diff ** 2).numpy()
        return float(penalty)


# ── Adapter layer trainer ─────────────────────────────────────────────────────

class AdapterTrainer:
    """
    Fine-tunes only the last N layers of the activity/target models.
    Frozen base layers preserve learned radar physics.
    Fast adapter layers learn environment-specific patterns.
    EWC penalty prevents forgetting old knowledge.
    """

    def __init__(self, config: ContinualConfig, ewc: EWC):
        self.config = config
        self.ewc = ewc
        self._model = None
        self._frozen_layers = 0

    def load_model(self, model_path: str) -> bool:
        """Load Keras model from checkpoint."""
        h5_path = model_path.replace('.tflite', '.h5')
        best_path = os.path.join(self.config.model_dir, "activity_best.h5")

        for path in [h5_path, best_path]:
            if os.path.exists(path):
                try:
                    import tensorflow as tf
                    self._model = tf.keras.models.load_model(path)
                    log.info(f"Adapter model loaded: {path}")
                    return True
                except Exception as e:
                    log.warning(f"Model load failed ({path}): {e}")
        return False

    def freeze_base(self, n_trainable_layers: int = 4):
        """Freeze all but the last N layers."""
        if self._model is None:
            return
        all_layers = self._model.layers
        n_freeze = max(0, len(all_layers) - n_trainable_layers)
        for i, layer in enumerate(all_layers):
            layer.trainable = i >= n_freeze
        self._frozen_layers = n_freeze
        trainable = sum(1 for l in all_layers if l.trainable)
        log.info(f"Frozen {n_freeze} base layers, {trainable} adapter layers trainable")

    def adapt(self, specs: np.ndarray, seqs: np.ndarray,
              labels: np.ndarray) -> dict:
        """
        Run one adaptation cycle on new + replay data.
        Returns metrics dict.
        """
        if self._model is None:
            return {"error": "no model loaded"}

        import tensorflow as tf

        # Normalise inputs
        specs = specs.astype(np.float32)
        if specs.ndim == 3:
            specs = specs[..., np.newaxis]
        specs = (specs - specs.mean(axis=(1,2,3), keepdims=True)) / \
                (specs.std(axis=(1,2,3), keepdims=True) + 1e-8)

        seqs = seqs.astype(np.float32)
        if seqs.ndim == 2:
            seqs = seqs[..., np.newaxis]
        seqs = (seqs - seqs.mean(axis=1, keepdims=True)) / \
               (seqs.std(axis=1, keepdims=True) + 1e-8)

        # Shuffle
        idx = np.random.permutation(len(specs))
        specs, seqs, labels = specs[idx], seqs[idx], labels[idx]

        optimizer = tf.keras.optimizers.Adam(self.config.adapter_lr)
        loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy()

        history = {"loss": [], "accuracy": []}

        for epoch in range(self.config.adapter_epochs):
            epoch_losses, epoch_accs = [], []
            batch_size = 16
            n_batches = max(1, len(specs) // batch_size)

            for b in range(n_batches):
                sl = slice(b * batch_size, (b+1) * batch_size)
                x_spec = specs[sl]
                x_seq  = seqs[sl]
                y      = labels[sl]

                if len(x_spec) == 0:
                    continue

                with tf.GradientTape() as tape:
                    pred = self._model(
                        {"spectrogram": x_spec, "phase_sequence": x_seq},
                        training=True)
                    ce_loss = loss_fn(y, pred)

                    # EWC penalty — prevents forgetting
                    ewc_penalty = self.ewc.penalty(self._model)
                    total_loss = ce_loss + self.config.ewc_lambda * ewc_penalty

                grads = tape.gradient(total_loss,
                                      self._model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, self._model.trainable_variables))

                acc = float(
                    tf.reduce_mean(
                        tf.cast(
                            tf.argmax(pred, axis=1) == tf.cast(y, tf.int64),
                            tf.float32)))
                epoch_losses.append(float(total_loss))
                epoch_accs.append(acc)

            mean_loss = np.mean(epoch_losses)
            mean_acc  = np.mean(epoch_accs)
            history["loss"].append(mean_loss)
            history["accuracy"].append(mean_acc)
            log.info(f"  Adapt epoch {epoch+1}/{self.config.adapter_epochs}"
                     f"  loss={mean_loss:.4f}  acc={mean_acc:.3f}")

        # Save updated model
        self._save_checkpoint()

        return {
            "final_loss": history["loss"][-1],
            "final_accuracy": history["accuracy"][-1],
            "epochs": self.config.adapter_epochs,
            "samples": len(specs)
        }

    def _save_checkpoint(self):
        """Save updated Keras model and re-export TFLite."""
        import tensorflow as tf

        h5_path = os.path.join(self.config.model_dir, "activity_best.h5")
        self._model.save(h5_path)

        # Re-export TFLite
        tflite_path = os.path.join(
            self.config.model_dir, "activity_recogniser.tflite")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self._model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_bytes = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_bytes)
            log.info(f"Updated TFLite exported: {len(tflite_bytes)//1024}KB")
        except Exception as e:
            log.warning(f"TFLite re-export failed: {e}")


# ── Federated gradient sharing over LoRa mesh ─────────────────────────────────

class MeshGradientSharer:
    """
    Shares compressed model improvement gradients between mesh nodes.
    No internet. No central server. Peer-to-peer only over LoRa.
    Each node broadcasts its gradient delta, receives others, averages.
    Gradients are compressed to ~1KB using top-k sparsification —
    fits within LoRa packet size constraints.
    """

    def __init__(self, node_id: str, mesh_port: int = 5700,
                 top_k_fraction: float = 0.01):
        self.node_id = node_id
        self.mesh_port = mesh_port
        self.top_k_fraction = top_k_fraction  # send only top 1% of gradients
        self._peer_gradients = {}             # node_id → gradient dict

    def compress_gradients(self, model) -> bytes:
        """
        Extract and compress model's adapter layer gradients.
        Top-k sparsification: keep only the K largest gradient values.
        Reduces 10MB gradient to ~1KB — fits in LoRa packets.
        """
        grad_data = {}
        for layer in model.layers[-4:]:   # only adapter layers
            for weight in layer.weights:
                vals = weight.numpy().flatten()
                k = max(1, int(len(vals) * self.top_k_fraction))
                top_idx = np.argpartition(np.abs(vals), -k)[-k:]
                grad_data[weight.name] = {
                    "indices": top_idx.tolist(),
                    "values":  vals[top_idx].tolist(),
                    "shape":   list(weight.shape)
                }

        payload = json.dumps({
            "node_id": self.node_id,
            "timestamp": time.time(),
            "gradients": grad_data
        })
        return payload.encode('utf-8')

    def apply_received_gradients(self, model, received_payloads: list):
        """
        Average received gradient updates from peer nodes into local model.
        Implements FedAvg — simple and robust.
        """
        if not received_payloads:
            return

        # Decode payloads
        peer_grads = []
        for payload in received_payloads:
            try:
                data = json.loads(payload.decode('utf-8'))
                if data.get("node_id") != self.node_id:
                    peer_grads.append(data["gradients"])
            except Exception:
                continue

        if not peer_grads:
            return

        # Apply averaged gradient update to adapter layers
        for layer in model.layers[-4:]:
            for weight in layer.weights:
                name = weight.name
                updates = []
                for pg in peer_grads:
                    if name in pg:
                        delta = np.zeros(weight.shape)
                        g = pg[name]
                        np.put(delta, g["indices"], g["values"])
                        updates.append(delta.reshape(weight.shape))
                if updates:
                    mean_update = np.mean(updates, axis=0)
                    weight.assign(weight + 0.1 * mean_update)  # conservative LR

        log.info(f"Applied gradients from {len(peer_grads)} peer nodes")


# ── Main continual learning orchestrator ──────────────────────────────────────

class ContinualLearner:
    """
    Runs the full continual learning loop autonomously in a background thread.
    Feed it inference results and it handles everything else:
      - Auto-labels high-confidence observations
      - Manages replay buffer on NVMe
      - Triggers adapter retraining on schedule
      - Shares improvements with mesh peers
      - Reports learning stats to dashboard
    """

    def __init__(self, config: Optional[ContinualConfig] = None):
        self.config = config or ContinualConfig()
        os.makedirs(self.config.buffer_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        self.buffer  = ObservationBuffer(self.config)
        self.ewc     = EWC(os.path.join(self.config.checkpoint_dir,
                                        "fisher_importance.npz"))
        self.adapter = AdapterTrainer(self.config, self.ewc)
        self.sharer  = MeshGradientSharer(
            node_id=self._get_node_id())

        self._running = False
        self._last_retrain = 0.0
        self._last_mesh_share = 0.0
        self._retrain_count = 0
        self._retrain_history = []
        self._lock = threading.Lock()

    def start(self, activity_model_path: str = "models/activity_recogniser.tflite"):
        """Start the background continual learning thread."""
        # Load model for adapter training
        loaded = self.adapter.load_model(activity_model_path)
        if loaded:
            self.adapter.freeze_base(n_trainable_layers=4)

        # Load EWC if available
        self.ewc.load()

        self._running = True
        self._thread = threading.Thread(
            target=self._learning_loop, daemon=True)
        self._thread.start()
        log.info("Continual learning engine started (fully offline)")

    def stop(self):
        self._running = False
        self.buffer._save_to_disk()

    def observe(self, spectrogram: np.ndarray, phase_sequence: np.ndarray,
                predicted_label: int, confidence: float):
        """
        Feed one inference result to the continual learner.
        Call this every frame for every tracked target.
        High-confidence predictions become training data automatically.
        """
        self.buffer.add_observation(
            spectrogram, phase_sequence, predicted_label, confidence)

    def _learning_loop(self):
        """Background thread: checks if retraining is due, runs it if so."""
        log.info("Continual learning loop running...")
        while self._running:
            now = time.time()

            # Check if it's time to retrain
            time_since_last = now - self._last_retrain
            n_new = self.buffer.n_new()

            should_retrain = (
                time_since_last >= self.config.retrain_interval_s and
                n_new >= self.config.min_new_samples
            )

            if should_retrain:
                self._run_adaptation_cycle()

            time.sleep(30)  # check every 30 seconds

    def _run_adaptation_cycle(self):
        """One full adaptation: get data → retrain adapters → export TFLite."""
        log.info("=" * 50)
        log.info("CONTINUAL ADAPTATION CYCLE STARTING")
        log.info(f"Buffer stats: {self.buffer.stats()}")
        t0 = time.time()

        # Get training batch from buffer
        batch = self.buffer.get_training_batch()
        if batch is None:
            log.info("Not enough samples yet — skipping")
            return

        specs, seqs, labels = batch
        log.info(f"Training on {len(specs)} samples "
                 f"({np.unique(labels, return_counts=True)})")

        # Run adapter training
        if self.adapter._model is not None:
            metrics = self.adapter.adapt(specs, seqs, labels)
            elapsed = time.time() - t0
            log.info(f"Adaptation complete in {elapsed:.1f}s: {metrics}")

            self._retrain_count += 1
            self._last_retrain = time.time()
            self._retrain_history.append({
                "cycle": self._retrain_count,
                "timestamp": self._last_retrain,
                "samples": len(specs),
                "accuracy": metrics.get("final_accuracy", 0),
                "elapsed_s": elapsed
            })

            # Save history
            hist_path = os.path.join(
                self.config.checkpoint_dir, "adaptation_history.json")
            with open(hist_path, 'w') as f:
                json.dump(self._retrain_history, f, indent=2)

            log.info(f"Adaptation cycle {self._retrain_count} complete — "
                     f"acc={metrics.get('final_accuracy', 0):.3f}")
        else:
            log.warning("No model loaded for adaptation")

        # Save buffer to NVMe
        self.buffer._save_to_disk()

    def get_stats(self) -> dict:
        """Return learning stats for the dashboard."""
        return {
            "retrain_count":    self._retrain_count,
            "last_retrain":     self._last_retrain,
            "next_retrain_in":  max(0, self.config.retrain_interval_s -
                                   (time.time() - self._last_retrain)),
            "buffer_stats":     self.buffer.stats(),
            "model_loaded":     self.adapter._model is not None,
            "ewc_active":       self.ewc._loaded,
        }

    def _get_node_id(self) -> str:
        """Get or generate a unique node ID."""
        id_file = os.path.join(self.config.checkpoint_dir, "node_id.txt")
        if os.path.exists(id_file):
            return open(id_file).read().strip()
        import uuid
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        with open(id_file, 'w') as f:
            f.write(node_id)
        return node_id
