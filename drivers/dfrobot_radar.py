"""
DFRobot SEN0395 24GHz mmWave Radar Driver
Parses ASCII output: $JYBSS,1, , , * (presence) or $JYBSS,0, , , * (clear)
"""

import serial
import re
import threading
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Callable
import logging

log = logging.getLogger(__name__)


@dataclass
class RadarFrame:
    timestamp: float
    presence: bool = False
    motion_detected: bool = False
    stationary_detected: bool = False
    target_range_m: float = 0.0
    move_energy: int = 0
    static_energy: int = 0
    detect_distance_cm: int = 0
    target_velocity_mps: float = 0.0
    breath_distance_m: float = 0.0
    signal_strength: float = 0.0
    raw_iq: Optional[np.ndarray] = None
    raw_line: str = ""


class SEN0395Radar:

    def __init__(self, port: str = "/dev/ttyAMA0", baud: int = 115200):
        self.port = port
        self.baud = baud
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[RadarFrame] = None
        self._callbacks: List[Callable] = []
        self._frame_count = 0
        self._start_time = 0.0
        self._range_history: List[float] = []
        self._max_history = 500

    def connect(self) -> bool:
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            log.info(f"Opened {self.port} @ {self.baud}")
        except serial.SerialException as e:
            log.error(f"Cannot open {self.port}: {e}")
            return False
        time.sleep(0.3)
        self._configure()
        return True

    def _configure(self):
        log.info("Starting SEN0395 (config already saved to flash)...")
        self._serial.write(b'sensorStart\r\n')
        time.sleep(0.3)
        log.info("SEN0395 running")

    def _cmd(self, command: str):
        if self._serial and self._serial.is_open:
            with self._lock:
                self._serial.write((command + '\r\n').encode())

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._serial and self._serial.is_open:
            self._serial.close()
        log.info("Radar disconnected")

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="sen0395")
        self._thread.start()
        log.info("SEN0395 reader started")

    def _read_loop(self):
        while self._running:
            try:
                if not (self._serial and self._serial.is_open):
                    time.sleep(0.1)
                    continue
                line = self._serial.readline()
                if line:
                    decoded = line.decode('ascii', errors='ignore').strip()
                    if decoded:
                        self._parse_line(decoded)
            except serial.SerialException as e:
                log.error(f"Serial error: {e}")
                time.sleep(1.0)
            except Exception as e:
                log.warning(f"Read error: {e}")
                time.sleep(0.05)

    def _parse_line(self, line: str):
        # Strip CLI prompt prefix e.g. "leapMMW:/>"
        if '>' in line:
            line = line[line.rfind('>') + 1:].strip()
        if not line:
            return

        # Only care about $JYBSS lines — grab the first digit after $JYBSS,
        if 'JYBSS' not in line:
            return

        m = re.search(r'JYBSS,([01])', line)
        if not m:
            return

        presence = m.group(1) == '1'

        frame = RadarFrame(timestamp=time.time())
        frame.presence = presence
        frame.motion_detected = presence
        frame.stationary_detected = presence
        frame.raw_line = line

        # Carry forward last known range
        if presence and self._latest_frame and self._latest_frame.target_range_m > 0:
            frame.target_range_m = self._latest_frame.target_range_m

        self._latest_frame = frame
        self._frame_count += 1

        for cb in self._callbacks:
            try:
                cb(frame)
            except Exception as e:
                log.warning(f"Callback error: {e}")

    def get_latest_frame(self) -> Optional[RadarFrame]:
        return self._latest_frame

    def register_callback(self, fn: Callable):
        self._callbacks.append(fn)

    def get_range_history(self) -> List[float]:
        return list(self._range_history)

    def is_connected(self) -> bool:
        return bool(self._serial and self._serial.is_open)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def frame_rate(self) -> float:
        elapsed = time.time() - self._start_time
        return self._frame_count / max(1, elapsed)

    def stats(self) -> dict:
        f = self._latest_frame
        return {
            "connected": self.is_connected(),
            "frame_count": self._frame_count,
            "frame_rate": round(self.frame_rate, 1),
            "presence": f.presence if f else False,
            "range_m": f.target_range_m if f else 0.0,
        }

    def configure_range(self, max_m: float = 6.0):
        self._cmd('sensorStop')
        time.sleep(0.2)
        idx = int(max_m / 0.15)
        self._cmd(f'detRangeCfg -1 0 {idx}')
        time.sleep(0.1)
        self._cmd('saveConfig')
        time.sleep(0.2)
        self._cmd('sensorStart')

    def set_sensitivity(self, level: int):
        level = max(0, min(9, level))
        self._cmd('sensorStop')
        time.sleep(0.1)
        self._cmd(f'setSensitivity {level}')
        time.sleep(0.1)
        self._cmd('saveConfig')
        time.sleep(0.1)
        self._cmd('sensorStart')


class SimulatedRadar(SEN0395Radar):

    def __init__(self):
        super().__init__(port="SIM")
        self._t = 0.0
        self._targets = [
            {"range_m": 4.2, "breath_hz": 0.27, "moving": False},
        ]

    def connect(self) -> bool:
        log.info("Simulated radar initialised")
        return True

    def _configure(self):
        pass

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._sim_loop, daemon=True)
        self._thread.start()

    def _sim_loop(self):
        dt = 0.1
        while self._running:
            self._t += dt
            frame = self._generate_frame()
            self._latest_frame = frame
            self._frame_count += 1
            if frame.target_range_m > 0:
                self._range_history.append(frame.target_range_m)
                if len(self._range_history) > self._max_history:
                    self._range_history.pop(0)
            for cb in self._callbacks:
                try:
                    cb(frame)
                except Exception:
                    pass
            time.sleep(dt)

    def _generate_frame(self) -> RadarFrame:
        frame = RadarFrame(timestamp=time.time())
        if not self._targets:
            return frame
        tgt = self._targets[0]
        t = self._t
        breath = 0.006 * np.sin(2 * np.pi * tgt["breath_hz"] * t)
        r = tgt["range_m"] + breath + np.random.randn() * 0.001
        if tgt["moving"]:
            r += 0.4 * np.sin(2 * np.pi * 0.4 * t)
        r = max(0.3, r)
        frame.presence = True
        frame.motion_detected = tgt["moving"]
        frame.stationary_detected = not tgt["moving"]
        frame.target_range_m = round(r, 3)
        frame.detect_distance_cm = int(r * 100)
        frame.static_energy = 70
        frame.signal_strength = 0.7
        return frame

    def set_scenario(self, scenario: str):
        scenarios = {
            "empty":          [],
            "person_still":   [{"range_m": 3.5, "breath_hz": 0.25, "moving": False}],
            "person_walking": [{"range_m": 5.0, "breath_hz": 0.0,  "moving": True}],
            "person_hiding":  [{"range_m": 5.0, "breath_hz": 0.18, "moving": False}],
        }
        self._targets = scenarios.get(scenario, [])


# Backwards compatibility alias
DFRobotRadar = SEN0395Radar
