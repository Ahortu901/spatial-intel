"""
DFRobot 24GHz Millimetre-Wave Radar Driver
Reads raw I/Q / presence / Doppler data over UART.
Supports both high-level parsed output and raw frame capture.
"""

import serial
import struct
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import logging

log = logging.getLogger(__name__)


@dataclass
class RadarFrame:
    timestamp: float
    raw_iq: Optional[np.ndarray] = None       # complex I/Q [num_chirps x num_samples]
    presence: bool = False
    target_range_m: float = 0.0
    target_velocity_mps: float = 0.0
    signal_strength: float = 0.0
    # Parsed high-level fields from DFRobot protocol
    motion_detected: bool = False
    stationary_detected: bool = False
    movement_speed: float = 0.0               # m/s
    breath_distance_m: float = 0.0


@dataclass
class RadarConfig:
    max_range_m: float = 6.0
    sensitivity: int = 7                      # 0-9
    unmanned_delay_s: int = 5


class DFRobotRadar:
    """
    Driver for DFRobot SEN0395 / SEN0609 24GHz FMCW radar.
    Handles UART framing, checksum validation, and raw I/Q extraction.
    """

    # DFRobot frame constants
    FRAME_HEADER = 0xFD
    FRAME_FOOTER = 0xFC
    CMD_HEADER   = bytes([0xFD, 0xFC, 0xFB, 0xFA])
    CMD_FOOTER   = bytes([0x04, 0x03, 0x02, 0x01])

    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame: Optional[RadarFrame] = None
        self._frame_callbacks: List = []
        self._raw_buffer = bytearray()
        self._frame_count = 0

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            time.sleep(0.2)
            self._configure_radar()
            log.info(f"DFRobot radar connected on {self.port}")
            return True
        except serial.SerialException as e:
            log.error(f"Radar connection failed: {e}")
            return False

    def disconnect(self):
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()

    # ── Configuration ────────────────────────────────────────────────────────

    def _configure_radar(self):
        """Send DFRobot configuration commands."""
        # Enable engineering mode for raw data
        self._send_command(bytes([0xFF, 0x00, 0x01, 0x00]))
        time.sleep(0.1)
        # Set max detection range: 6m (value * 75cm)
        self._send_command(bytes([0x60, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00]))
        time.sleep(0.1)

    def _send_command(self, cmd_data: bytes):
        if self._serial and self._serial.is_open:
            frame = self.CMD_HEADER + len(cmd_data).to_bytes(2, 'little') + cmd_data + self.CMD_FOOTER
            with self._lock:
                self._serial.write(frame)

    # ── Reading loop ─────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while self._running:
            try:
                if self._serial and self._serial.in_waiting:
                    data = self._serial.read(self._serial.in_waiting)
                    self._raw_buffer.extend(data)
                    self._parse_buffer()
                else:
                    time.sleep(0.005)
            except Exception as e:
                log.warning(f"Radar read error: {e}")
                time.sleep(0.1)

    def _parse_buffer(self):
        """Parse DFRobot binary protocol frames from buffer."""
        while len(self._raw_buffer) >= 4:
            # Find frame header
            start = -1
            for i in range(len(self._raw_buffer) - 3):
                if (self._raw_buffer[i] == 0xF4 and
                    self._raw_buffer[i+1] == 0xF3 and
                    self._raw_buffer[i+2] == 0xF2 and
                    self._raw_buffer[i+3] == 0xF1):
                    start = i
                    break

            if start == -1:
                self._raw_buffer = self._raw_buffer[-3:]
                return

            if start > 0:
                self._raw_buffer = self._raw_buffer[start:]

            # Need at least 8 bytes for header + length
            if len(self._raw_buffer) < 8:
                return

            data_len = struct.unpack_from('<H', self._raw_buffer, 4)[0]
            frame_len = 4 + 2 + data_len + 2 + 4  # header+len+data+checksum+footer

            if len(self._raw_buffer) < frame_len:
                return

            frame_bytes = bytes(self._raw_buffer[:frame_len])
            self._raw_buffer = self._raw_buffer[frame_len:]

            # Validate footer
            if frame_bytes[-4:] != bytes([0x01, 0x02, 0x03, 0x04]):
                continue

            payload = frame_bytes[6:6+data_len]
            self._dispatch_frame(payload)

    def _dispatch_frame(self, payload: bytes):
        """Parse payload and build RadarFrame."""
        if len(payload) < 4:
            return

        frame = RadarFrame(timestamp=time.time())
        frame_type = payload[0]

        if frame_type == 0x01:
            # Basic presence frame
            frame.presence = bool(payload[1])
            frame.stationary_detected = bool(payload[2] & 0x01)
            frame.motion_detected = bool(payload[2] & 0x02)

            if len(payload) >= 8:
                frame.target_range_m = struct.unpack_from('<H', payload, 4)[0] * 0.01
                frame.movement_speed = struct.unpack_from('<H', payload, 6)[0] * 0.01
                frame.breath_distance_m = struct.unpack_from('<H', payload, 8)[0] * 0.01 if len(payload) > 9 else 0.0

        elif frame_type == 0x02:
            # Engineering / raw I/Q frame
            # Reconstruct complex I/Q samples
            num_samples = (len(payload) - 2) // 4
            if num_samples > 0:
                iq_data = np.frombuffer(payload[2:2 + num_samples*4], dtype=np.int16)
                i_samples = iq_data[0::2].astype(np.float32) / 32768.0
                q_samples = iq_data[1::2].astype(np.float32) / 32768.0
                frame.raw_iq = i_samples + 1j * q_samples
                frame.presence = True

        self._latest_frame = frame
        self._frame_count += 1

        for cb in self._frame_callbacks:
            try:
                cb(frame)
            except Exception as e:
                log.warning(f"Frame callback error: {e}")

    # ── Public API ───────────────────────────────────────────────────────────

    def get_latest_frame(self) -> Optional[RadarFrame]:
        return self._latest_frame

    def register_callback(self, fn):
        self._frame_callbacks.append(fn)

    def get_frame_rate(self) -> float:
        """Approximate frames per second."""
        return self._frame_count / max(1, time.time() - getattr(self, '_start_time', time.time()))


class SimulatedRadar(DFRobotRadar):
    """
    Software simulation of the DFRobot radar for development/testing
    without physical hardware. Generates synthetic I/Q data with
    configurable targets, breathing, and motion.
    """

    def __init__(self):
        super().__init__(port="SIM", baud=0)
        self._targets = [
            {"range_m": 4.2, "angle_deg": -15, "breath_hz": 0.27, "hr_hz": 1.1, "moving": False},
            {"range_m": 6.8, "angle_deg": 20,  "breath_hz": 0.0,  "hr_hz": 0.0,  "moving": True},
        ]
        self._t = 0.0

    def connect(self) -> bool:
        log.info("Simulated radar initialised")
        return True

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._thread.start()

    def _sim_loop(self):
        from config.settings import RADAR_NUM_SAMPLES, RADAR_NUM_CHIRPS, UPDATE_RATE_HZ
        dt = 1.0 / UPDATE_RATE_HZ
        c = 3e8
        while self._running:
            frame = self._generate_frame()
            self._latest_frame = frame
            self._frame_count += 1
            for cb in self._frame_callbacks:
                try:
                    cb(frame)
                except Exception:
                    pass
            time.sleep(dt)
            self._t += dt

    def _generate_frame(self) -> RadarFrame:
        from config.settings import (RADAR_NUM_SAMPLES, RADAR_NUM_CHIRPS,
                                      RADAR_FREQ_START_GHZ, RADAR_BANDWIDTH_GHZ)
        N = RADAR_NUM_SAMPLES
        M = RADAR_NUM_CHIRPS
        c = 3e8
        B = RADAR_BANDWIDTH_GHZ * 1e9
        f0 = RADAR_FREQ_START_GHZ * 1e9

        iq = np.zeros((M, N), dtype=complex)
        noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * 0.02

        for tgt in self._targets:
            # Add breathing / heartbeat displacement
            displacement = 0.0
            if tgt["breath_hz"] > 0:
                displacement += 0.006 * np.sin(2 * np.pi * tgt["breath_hz"] * self._t)
            if tgt["hr_hz"] > 0:
                displacement += 0.0003 * np.sin(2 * np.pi * tgt["hr_hz"] * self._t)

            r = tgt["range_m"] + displacement
            if tgt["moving"]:
                r += 0.3 * np.sin(2 * np.pi * 0.5 * self._t)

            # Beat frequency for this range
            fb = 2 * B * r / (c * 40e-6)
            t_fast = np.linspace(0, 40e-6, N)

            for chirp in range(M):
                phase_offset = 2 * np.pi * 2 * f0 * r / c
                iq[chirp] += np.exp(1j * (2 * np.pi * fb * t_fast + phase_offset))

        iq += noise

        frame = RadarFrame(timestamp=time.time())
        frame.raw_iq = iq
        frame.presence = True
        frame.target_range_m = self._targets[0]["range_m"]
        return frame
