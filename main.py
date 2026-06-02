"""
Spatial Intelligence System — Main Orchestrator
Starts all sensor threads, runs the processing pipeline,
and pushes fused state to the dashboard at 10 Hz.
"""

import asyncio
import threading
import time
import logging
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
log = logging.getLogger("main")

from config.settings import (
    MODE, UPDATE_RATE_HZ, RADAR_UART_PORT, RADAR_BAUD,
    PIR_GPIO_PIN, DASHBOARD_HOST, DASHBOARD_PORT
)
from drivers.dfrobot_radar import DFRobotRadar, SimulatedRadar
from drivers.pir_sensor import PIRSensor
from processing.backprojection import BackProjectionImager
from processing.cfar import cfar_1d, remove_static_clutter
from processing.vital_signs import VitalSignsExtractor
from processing.fingerprint import RFFingerprinter
from inference.tracker import KalmanTracker, TargetClassifier
from inference.ml_engine import MLInferenceEngine
from training.continual.continual_learner import ContinualLearner, ContinualConfig
from fusion.sensor_fusion import SensorFusion
from outputs.dashboard import app, set_system, broadcast, run_server


class SpatialIntelligenceSystem:

    def __init__(self, simulate: bool = False):
        self.mode = MODE
        self.simulate = simulate
        self.start_time = time.time()
        self.latest_state = None
        self._running = False

        # Initialise hardware
        log.info("Initialising hardware...")
        if simulate:
            self.radar = SimulatedRadar()
        else:
            self.radar = DFRobotRadar(RADAR_UART_PORT, RADAR_BAUD)

        self.pir = PIRSensor(PIR_GPIO_PIN)

        # Processing pipeline
        self.imager       = BackProjectionImager()
        self.vitals       = VitalSignsExtractor()
        self.fingerprinter= RFFingerprinter()
        self.tracker      = KalmanTracker(dt=1.0/UPDATE_RATE_HZ)
        self.classifier   = TargetClassifier()
        self.ml_engine    = MLInferenceEngine(model_dir="models")
        self.continual     = ContinualLearner(ContinualConfig(model_dir="models"))
        self.fusion       = SensorFusion()

        # State
        self.radar_connected = False
        self.pir_connected   = False
        self._vitals_map     = {}
        self._pir_state      = False
        self._recent_anomalies = []
        self._frame_count    = 0
        self._raw_frame_buffer = []   # rolling I/Q buffer for clutter removal

    def set_mode(self, mode: str):
        self.mode = mode
        log.info(f"Mode changed to: {mode}")

    # ── Startup ──────────────────────────────────────────────────────────────

    def start(self):
        log.info("Starting Spatial Intelligence System...")
        self._running = True

        # Connect hardware
        self.radar_connected = self.radar.connect()
        self.pir_connected   = self.pir.connect()

        # Register PIR callback
        self.pir.register_callback(self._on_pir_event)

        # Start radar reading
        self.radar.start()
        self.ml_engine.load_all()
        self.continual.start("models/activity_recogniser.tflite")

        # Start processing loop in background thread
        self._proc_thread = threading.Thread(
            target=self._processing_loop, daemon=True)
        self._proc_thread.start()

        log.info(f"System running — radar={'OK' if self.radar_connected else 'SIM'}, "
                 f"PIR={'OK' if self.pir_connected else 'SIM'}")

    def stop(self):
        self._running = False
        self.radar.disconnect()
        self.pir.disconnect()
        log.info("System stopped")

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_pir_event(self, event):
        self._pir_state = event.detected
        if event.detected:
            log.debug(f"PIR triggered — thermal presence confirmed")

    # ── Main processing loop ──────────────────────────────────────────────────

    def _processing_loop(self):
        dt = 1.0 / UPDATE_RATE_HZ
        log.info(f"Processing loop started at {UPDATE_RATE_HZ} Hz")

        while self._running:
            t_start = time.time()
            try:
                self._process_frame()
            except Exception as e:
                log.error(f"Processing error: {e}", exc_info=True)
            elapsed = time.time() - t_start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)

    def _process_frame(self):
        self._frame_count += 1

        # ── 1. Get radar frame ────────────────────────────────────────────
        radar_frame = self.radar.get_latest_frame()
        spatial_image = None
        detections = []

        if radar_frame and radar_frame.raw_iq is not None:
            iq = radar_frame.raw_iq   # [chirps x samples]

            # Buffer raw frames for clutter removal
            self._raw_frame_buffer.append(iq)
            if len(self._raw_frame_buffer) > 20:
                self._raw_frame_buffer.pop(0)

            if len(self._raw_frame_buffer) >= 5:
                # ── 2. Build spatial image ────────────────────────────────
                spatial_image = self.imager.process_frame(iq)

                # ── 3. Detect targets in spatial image ────────────────────
                raw_detections = self.imager.detect_peaks(spatial_image, threshold=0.55)

                # ── 4. CFAR on mean range profile for additional targets ───
                range_time = self.imager.get_range_time_matrix()
                if range_time is not None and len(range_time) >= 5:
                    mean_profile = np.mean(np.abs(range_time[-5:]), axis=0)
                    cfar_mask = cfar_1d(mean_profile)
                    cfar_ranges = np.where(cfar_mask)[0]

                    from config.settings import RADAR_FREQ_START_GHZ, RADAR_BANDWIDTH_GHZ
                    c = 3e8
                    B = RADAR_BANDWIDTH_GHZ * 1e9
                    range_per_bin = c / (2 * B)

                    for bin_idx in cfar_ranges:
                        r = bin_idx * range_per_bin
                        if r < 0.5 or r > 20.0:
                            continue
                        # Check if already in detections
                        already = any(abs(d["range_m"] - r) < 0.8 for d in raw_detections)
                        if not already:
                            raw_detections.append({
                                "x_m": 0.0, "y_m": round(r, 2),
                                "range_m": round(r, 2), "az_deg": 0.0,
                                "power": 0.5, "size_px": 3
                            })

                detections = raw_detections

                # ── 5. Update vitals for each detection ───────────────────
                range_profile = spatial_image.power_db.mean(axis=1)   # simplified
                for det in detections:
                    # Feed latest range profile to vital signs extractor
                    # Use a synthetic complex profile for phase extraction
                    if range_time is not None:
                        complex_profile = range_time[-1].astype(complex)
                        vitals = self.vitals.update(
                            target_id=hash(round(det["range_m"], 1)) & 0xFFFF,
                            range_m=det["range_m"],
                            range_profile=complex_profile
                        )
                        if vitals:
                            tid_key = f"T{hash(round(det['range_m'], 1)) & 0xFFFF:04d}"
                            self._vitals_map[tid_key] = vitals

        # ── 6. Track targets ──────────────────────────────────────────────
        confirmed_tracks = self.tracker.update(detections)

        # ── 7. Classify targets ───────────────────────────────────────────
        for track in confirmed_tracks:
            if track.target_class == "unknown":
                cls, conf = self.classifier.classify(
                    track,
                    doppler_spectrum=None,
                    radar_cross_section=track.covariance[0, 0]
                )
                track.target_class = cls
                track.class_confidence = conf

        # ── 8. RF fingerprinting (every 5th frame for efficiency) ─────────
        self._recent_anomalies = []
        if self._frame_count % 5 == 0:
            # Simulate CSI from radar amplitude as proxy
            if radar_frame and radar_frame.raw_iq is not None:
                csi_proxy = np.abs(np.fft.fft(
                    radar_frame.raw_iq.mean(axis=0)))[:52]
                csi_proxy /= (np.max(csi_proxy) + 1e-10)
                evt = self.fingerprinter.process_sample(csi_proxy)
                if evt:
                    self._recent_anomalies.append(evt)

        # ── 9. Fuse everything ────────────────────────────────────────────
        scene = self.fusion.fuse(
            radar_tracks=confirmed_tracks,
            pir_detected=self._pir_state,
            fingerprint_anomalies=self._recent_anomalies,
            vitals_map=self._vitals_map,
            radar_image=spatial_image
        )
        self.latest_state = scene

        # Log alerts
        for alert in scene.alerts:
            log.warning(f"ALERT: {alert}")

        # ── 10. Push to dashboard ─────────────────────────────────────────
        if self._frame_count % 1 == 0:   # every frame
            payload = self.fusion.serialize_state(scene)
            payload["type"] = "state_update"
            asyncio.run_coroutine_threadsafe(
                broadcast(payload), self._event_loop)

    # ── Async runner ──────────────────────────────────────────────────────────

    async def _run_async(self):
        self._event_loop = asyncio.get_event_loop()
        self.start()

        import uvicorn
        set_system(self)
        config = uvicorn.Config(
            app,
            host=DASHBOARD_HOST,
            port=DASHBOARD_PORT,
            log_level="warning"
        )
        server = uvicorn.Server(config)
        log.info(f"Dashboard at http://localhost:{DASHBOARD_PORT}")
        await server.serve()

    def run(self):
        """Blocking run — starts everything and serves the dashboard."""
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            log.info("Shutting down...")
            self.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spatial Intelligence System")
    parser.add_argument("--simulate", action="store_true",
                        help="Run with simulated radar (no hardware required)")
    parser.add_argument("--calibrate", type=str, default=None,
                        help="Calibrate fingerprint for zone name")
    args = parser.parse_args()

    system = SpatialIntelligenceSystem(simulate=args.simulate)

    if args.calibrate:
        system.fingerprinter.start_calibration(args.calibrate)

    system.run()
