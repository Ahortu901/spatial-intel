"""
ML Inference Engine
Drop-in replacement for the rule-based classifier.
Loads all trained TFLite models and routes inference requests.
Falls back to heuristics if models are not yet trained.
"""

import numpy as np
import os
import json
import time
import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    target_class: str          # person | vehicle | drone | empty | animal
    class_confidence: float
    activity: str              # walking | sitting | falling | ...
    activity_confidence: float
    activity_display: str
    activity_is_alert: bool
    breath_rate_bpm: Optional[float]
    heart_rate_bpm:  Optional[float]
    anomaly_score: float
    person_id: str             # re-ID result ("unknown" if not enrolled)
    person_similarity: float
    from_model: bool           # True = ML model, False = heuristic


class MLInferenceEngine:
    """
    Unified inference engine for all five models.
    Call .load() once at startup, then .infer() per target per frame.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self._loaded = {}

        # Lazy-load model instances
        self._target_clf   = None
        self._activity_rec = None
        self._ae_detector  = None
        self._vitals_cnn   = None
        self._gait_reid    = None

    def load_all(self):
        """Load all available models. Safe to call even if models are missing."""
        self._load_target_classifier()
        self._load_activity_recogniser()
        self._load_autoencoder()
        self._load_vitals_cnn()
        self._load_gait_reid()

        loaded = [k for k, v in self._loaded.items() if v]
        missing = [k for k, v in self._loaded.items() if not v]
        log.info(f"Models loaded:  {loaded}")
        if missing:
            log.warning(f"Models missing (using heuristics): {missing}")

    def _load_target_classifier(self):
        path = os.path.join(self.model_dir, "target_classifier.tflite")
        try:
            try:
                import tflite_runtime.interpreter as tflite
                interp = tflite.Interpreter(model_path=path)
            except ImportError:
                import tensorflow as tf
                interp = tf.lite.Interpreter(model_path=path)
            interp.allocate_tensors()
            self._tc_interp = interp
            self._tc_in  = interp.get_input_details()[0]
            self._tc_out = interp.get_output_details()[0]

            meta_path = path.replace(".tflite", "_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self._tc_meta = json.load(f)
                self._tc_labels = self._tc_meta["labels"]
            else:
                self._tc_labels = ["empty","person","vehicle_car",
                                   "vehicle_truck","drone_quad","drone_fixedwing","animal"]
            self._loaded["target_classifier"] = True
        except Exception as e:
            log.debug(f"Target classifier not loaded: {e}")
            self._loaded["target_classifier"] = False

    def _load_activity_recogniser(self):
        try:
            from training.trainers.train_activity_recogniser import ActivityRecogniser
            ar = ActivityRecogniser(
                model_path=os.path.join(self.model_dir, "activity_recogniser.tflite"),
                meta_path =os.path.join(self.model_dir, "activity_recogniser_meta.json")
            )
            loaded = ar.load()
            self._activity_rec = ar
            self._loaded["activity_recogniser"] = loaded
        except Exception as e:
            log.debug(f"Activity recogniser not loaded: {e}")
            self._loaded["activity_recogniser"] = False

    def _load_autoencoder(self):
        try:
            from training.trainers.train_other_models import AutoencoderAnomalyDetector
            ae = AutoencoderAnomalyDetector(
                model_path=os.path.join(self.model_dir, "rf_autoencoder.tflite"),
                meta_path =os.path.join(self.model_dir, "autoencoder_meta.json")
            )
            loaded = ae.load()
            self._ae_detector = ae
            self._loaded["autoencoder"] = loaded
        except Exception as e:
            log.debug(f"Autoencoder not loaded: {e}")
            self._loaded["autoencoder"] = False

    def _load_vitals_cnn(self):
        try:
            from training.trainers.train_other_models import VitalsCNNEstimator
            vc = VitalsCNNEstimator(
                model_path=os.path.join(self.model_dir, "vitals_cnn.tflite"))
            loaded = vc.load()
            self._vitals_cnn = vc
            self._loaded["vitals_cnn"] = loaded
        except Exception as e:
            log.debug(f"Vitals CNN not loaded: {e}")
            self._loaded["vitals_cnn"] = False

    def _load_gait_reid(self):
        try:
            from training.trainers.train_other_models import GaitReIdentifier
            gr = GaitReIdentifier(
                model_path=os.path.join(self.model_dir, "gait_embedder.h5"))
            loaded = gr.load()
            self._gait_reid = gr
            self._loaded["gait_reid"] = loaded
        except Exception as e:
            log.debug(f"Gait re-ID not loaded: {e}")
            self._loaded["gait_reid"] = False

    # ── Main inference call ───────────────────────────────────────────────────

    def infer(self,
              target_id: str,
              spectrogram: np.ndarray,
              phase_value: float,
              csi_vector: Optional[np.ndarray] = None,
              doppler_spectrum: Optional[np.ndarray] = None,
              radar_cross_section: float = 1.0) -> InferenceResult:
        """
        Run all applicable models for one target frame.
        All inputs are optional — gracefully falls back when unavailable.
        """

        # ── 1. Target classification ──────────────────────────────────────
        if self._loaded.get("target_classifier") and spectrogram is not None:
            target_class, class_conf, from_model = self._classify_target(spectrogram)
        else:
            target_class, class_conf = self._heuristic_target(radar_cross_section)
            from_model = False

        # ── 2. Activity recognition ───────────────────────────────────────
        act_label, act_conf, act_display, act_alert = "unknown", 0.0, "Unknown", False
        if self._loaded.get("activity_recogniser") and self._activity_rec:
            result = self._activity_rec.update(target_id, spectrogram, phase_value)
            if result:
                act_label   = result.smoothed_label
                act_conf    = result.confidence
                act_display = result.display_name
                act_alert   = result.is_alert
                from_model  = True

        # ── 3. Vital signs (CNN) ──────────────────────────────────────────
        breath_bpm, hr_bpm = None, None
        if self._loaded.get("vitals_cnn") and self._vitals_cnn:
            vitals = self._vitals_cnn.update(target_id, phase_value)
            if vitals:
                breath_bpm = vitals["breath_rate_bpm"]
                hr_bpm     = vitals["heart_rate_bpm"]

        # ── 4. RF anomaly score ───────────────────────────────────────────
        anomaly_score = 0.0
        if self._loaded.get("autoencoder") and self._ae_detector and csi_vector is not None:
            anomaly_score = self._ae_detector.score(csi_vector)

        # ── 5. Gait re-identification ─────────────────────────────────────
        person_id, person_sim = "unknown", 0.0
        if (self._loaded.get("gait_reid") and self._gait_reid and
                self._gait_reid.loaded and target_class == "person"):
            # Use phase buffer as gait signature
            if (self._activity_rec and
                    target_id in self._activity_rec._phase_buffers):
                phase_seq = np.array(list(
                    self._activity_rec._phase_buffers[target_id]))
                if len(phase_seq) >= 30:
                    person_id, person_sim = self._gait_reid.identify(phase_seq)

        return InferenceResult(
            target_class=target_class,
            class_confidence=round(class_conf, 3),
            activity=act_label,
            activity_confidence=round(act_conf, 3),
            activity_display=act_display,
            activity_is_alert=act_alert,
            breath_rate_bpm=breath_bpm,
            heart_rate_bpm=hr_bpm,
            anomaly_score=round(anomaly_score, 3),
            person_id=person_id,
            person_similarity=round(person_sim, 3),
            from_model=from_model
        )

    def _classify_target(self, spectrogram: np.ndarray) -> Tuple[str, float, bool]:
        spec = spectrogram.astype(np.float32)
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        spec_in = spec[np.newaxis, ..., np.newaxis]
        self._tc_interp.set_tensor(self._tc_in['index'], spec_in)
        self._tc_interp.invoke()
        probs = self._tc_interp.get_tensor(self._tc_out['index'])[0]
        idx = int(np.argmax(probs))
        # Map to canonical class names
        raw_label = self._tc_labels[idx] if idx < len(self._tc_labels) else "unknown"
        canonical = _canonicalise_target(raw_label)
        return canonical, float(probs[idx]), True

    def _heuristic_target(self, rcs: float) -> Tuple[str, float]:
        if rcs > 8.0:   return "vehicle", 0.6
        elif rcs > 0.5: return "person",  0.5
        else:           return "unknown", 0.3

    def enroll_person(self, name: str, phase_sequences: list):
        """Add a person to the gait re-ID gallery."""
        if self._gait_reid and self._gait_reid.loaded:
            self._gait_reid.enroll(name, phase_sequences)

    @property
    def model_status(self) -> Dict[str, bool]:
        return dict(self._loaded)


def _canonicalise_target(label: str) -> str:
    """Map fine-grained labels to system canonical classes."""
    if label in ("person",):                          return "person"
    if label in ("vehicle_car", "vehicle_truck"):     return "vehicle"
    if label in ("drone_quad", "drone_fixedwing"):    return "drone"
    if label == "empty":                              return "empty"
    if label == "animal":                             return "animal"
    return "unknown"
