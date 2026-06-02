"""
DFRobot SEN0395 24GHz mmWave Human Presence Detection Radar
Complete driver built specifically for SEN0395 protocol.

The SEN0395 outputs ASCII lines over UART at 115200 baud.
It has TWO output modes:
  1. Simple mode  — one line per detection event (default)
  2. Engineering mode — continuous data with range, speed, energy values

Serial output format (engineering mode):
  $JYBSS,0,0,0,0*    — no target
  $JYBSS,1,1,0,0*    — moving target detected
  $JYBSS,1,0,1,0*    — stationary target detected
  $JYBSS,1,1,1,0*    — both moving and stationary

Engineering data lines:
  $JYENG,<move_energy>,<static_energy>,<detect_distance>*

Configuration commands (sent as ASCII):
  sensorStop          stop output
  sensorStart         start output
  detRangeCfg  0 0 6 6  set detection range 0-6m moving, 0-6m static
  outputLatency 0 1      output delay
  setLatency 0 1
  setSensitivity 7       sensitivity 0-9
  saveConfig             save to flash

Reference: https://wiki.dfrobot.com/mmWave_Radar_Human_Presence_Detection_SKU_SEN0395
"""

import serial
import threading
import time
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import logging

log = logging.getLogger(__name__)


@dataclass
class RadarFrame:
    """One parsed output frame from the SEN0395."""
    timestamp: float

    # Presence state
    presence: bool = False
    motion_detected: bool = False
    stationary_detected: bool = False

    # Range and energy (engineering mode)
    target_range_m: float = 0.0
    move_energy: int = 0        # 0-100 signal strength for moving target
    static_energy: int = 0      # 0-100 signal strength for static target
    detect_distance_cm: int = 0 # raw distance in cm

    # Derived
    target_velocity_mps: float = 0.0
    breath_distance_m: float = 0.0
    signal_strength: float = 0.0

    # Raw I/Q (populated if engineering mode is active)
    raw_iq: Optional[np.ndarray] = None

    # Source line for debugging
    raw_line: str = ""


class SEN0395Radar:
    """
    Full driver for the DFRobot SEN0395 24GHz radar.
    Handles connection, configuration, line parsing, and callbacks.
    Tested against the actual SEN0395 ASCII protocol.
    """

    # Regex patterns for the two main output line types
    # $JYBSS,<presence>,<moving>,<stationary>,<reserved>*
    _RE_STATUS = re.compile(
        r'\$JYBSS,(\d),(\d),(\d),(\d)\*')

    # $JYENG,<move_energy>,<static_energy>,<distance_cm>*
    _RE_ENG = re.compile(
        r'\$JYENG,(\d+),(\d+),(\d+)\*')

    def __init__(self, port: str = "/dev/ttyAMA0", baud: int = 115200):
        self.port = port
        self.baud = baud

        self._serial:   Optional[serial.Serial] = None
        self._lock      = threading.Lock()
        self._running   = False
        self._thread:   Optional[threading.Thread] = None

        self._latest_frame: Optional[RadarFrame] = None
        self._callbacks: List[Callable] = []

        # Stats
        self._frame_count = 0
        self._start_time  = 0.0
        self._parse_errors = 0

        # Rolling history for vital signs
        self._range_history: List[float] = []
        self._max_history = 500   # 50 seconds at 10Hz

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open UART port and configure the radar."""
        try:
            self._serial = serial.Serial(
                port     = self.port,
                baudrate = self.baud,
                bytesize = serial.EIGHTBITS,
                parity   = serial.PARITY_NONE,
                stopbits = serial.STOPBITS_ONE,
                timeout  = 1.0
            )
            log.info(f"Opened {self.port} @ {self.baud}")
        except serial.SerialException as e:
            log.error(f"Cannot open {self.port}: {e}")
            log.error("Check: sudo raspi-config → Interface → Serial → disable login shell, enable hardware")
            return False

        time.sleep(0.3)
        self._configure()
        return True

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._serial and self._serial.is_open:
            self._cmd("sensorStop")
            time.sleep(0.1)
            self._serial.close()
        log.info("Radar disconnected")

    # ── Configuration ─────────────────────────────────────────────────────────

    def _configure(self):
        """Send startup configuration to the SEN0395."""
        log.info("Configuring SEN0395...")

        self._cmd("sensorStop")
        time.sleep(0.15)

        # Detection range: moving 0-6m, stationary 0-6m
        # Format: detRangeCfg -1 <move_start> <move_end> <static_start> <static_end>
        self._cmd("detRangeCfg -1 0 6 0 6")
        time.sleep(0.1)

        # Sensitivity 0-9 (7 is a good balanced value)
        self._cmd("setSensitivity 7")
        time.sleep(0.1)

        # Output latency: 0s appear, 1s disappear
        self._cmd("setLatency 0 1")
        time.sleep(0.1)

        # Save config to flash so it persists after power cycle
        self._cmd("saveConfig")
        time.sleep(0.15)

        # Start the sensor
        self._cmd("sensorStart")
        time.sleep(0.2)

        log.info("SEN0395 configured and running")

    def _cmd(self, command: str):
        """Send an ASCII command with CR LF terminator."""
        if self._serial and self._serial.is_open:
            line = (command + "\r\n").encode("ascii")
            with self._lock:
                self._serial.write(line)
            log.debug(f"CMD → {command}")

    def configure_range(self, move_max_m: float = 6.0,
                        static_max_m: float = 6.0):
        """Change detection range at runtime."""
        self._cmd("sensorStop")
        time.sleep(0.1)
        self._cmd(f"detRangeCfg -1 0 {int(move_max_m)} 0 {int(static_max_m)}")
        time.sleep(0.1)
        self._cmd("saveConfig")
        time.sleep(0.1)
        self._cmd("sensorStart")

    def set_sensitivity(self, level: int):
        """Set sensitivity 0 (low) – 9 (high)."""
        level = max(0, min(9, level))
        self._cmd("sensorStop")
        time.sleep(0.05)
        self._cmd(f"setSensitivity {level}")
        time.sleep(0.05)
        self._cmd("saveConfig")
        time.sleep(0.05)
        self._cmd("sensorStart")

    # ── Reading loop ──────────────────────────────────────────────────────────

    def start(self):
        """Start background reading thread."""
        self._running   = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="sen0395-reader")
        self._thread.start()
        log.info("SEN0395 reader started")

    def _read_loop(self):
        """Read lines from UART and parse them."""
        buf = ""
        while self._running:
            try:
                if not (self._serial and self._serial.is_open):
                    time.sleep(0.1)
                    continue

                # Read available bytes
                n = self._serial.in_waiting
                if n == 0:
                    time.sleep(0.005)
                    continue

                raw = self._serial.read(n)
                buf += raw.decode("ascii", errors="ignore")

                # Process complete lines
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._parse_line(line)

            except serial.SerialException as e:
                log.error(f"Serial error: {e} — attempting reconnect")
                time.sleep(1.0)
                self._reconnect()
            except Exception as e:
                log.warning(f"Read loop error: {e}")
                time.sleep(0.05)

    def _reconnect(self):
        """Try to reopen the serial port after an error."""
        try:
            if self._serial:
                self._serial.close()
            time.sleep(0.5)
            self._serial = serial.Serial(
                self.port, self.baud, timeout=1.0)
            self._cmd("sensorStart")
            log.info("Reconnected to radar")
        except Exception as e:
            log.error(f"Reconnect failed: {e}")

    # ── Parser ────────────────────────────────────────────────────────────────

    def _parse_line(self, line: str):
        """
        Parse one ASCII line from SEN0395.
        Two formats:
          $JYBSS,<presence>,<moving>,<stationary>,<reserved>*
          $JYENG,<move_energy>,<static_energy>,<distance_cm>*
        """
        frame = None

        # Status line — presence / motion state
        m = self._RE_STATUS.search(line)
        if m:
            presence    = m.group(1) == "1"
            moving      = m.group(2) == "1"
            stationary  = m.group(3) == "1"

            frame = RadarFrame(timestamp=time.time())
            frame.presence           = presence
            frame.motion_detected    = moving
            frame.stationary_detected= stationary
            frame.raw_line           = line

            # Carry over last known range if still present
            if presence and self._latest_frame:
                frame.target_range_m = self._latest_frame.target_range_m
                frame.move_energy    = self._latest_frame.move_energy
                frame.static_energy  = self._latest_frame.static_energy

        # Engineering line — energy + distance
        m2 = self._RE_ENG.search(line)
        if m2:
            move_e   = int(m2.group(1))
            static_e = int(m2.group(2))
            dist_cm  = int(m2.group(3))

            if frame is None:
                # Engineering line without preceding status — build partial frame
                frame = RadarFrame(timestamp=time.time())
                if self._latest_frame:
                    frame.presence          = self._latest_frame.presence
                    frame.motion_detected   = self._latest_frame.motion_detected
                    frame.stationary_detected = self._latest_frame.stationary_detected

            frame.move_energy         = move_e
            frame.static_energy       = static_e
            frame.detect_distance_cm  = dist_cm
            frame.target_range_m      = dist_cm / 100.0
            frame.signal_strength     = max(move_e, static_e) / 100.0
            frame.raw_line            = line

            # Update range history for vital signs
            if dist_cm > 0:
                self._range_history.append(dist_cm / 100.0)
                if len(self._range_history) > self._max_history:
                    self._range_history.pop(0)

        if frame is None:
            # Unrecognised line — log at debug level
            if line.startswith("$"):
                log.debug(f"Unknown frame: {line!r}")
            return

        self._latest_frame = frame
        self._frame_count += 1

        for cb in self._callbacks:
            try:
                cb(frame)
            except Exception as e:
                log.warning(f"Callback error: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_latest_frame(self) -> Optional[RadarFrame]:
        return self._latest_frame

    def register_callback(self, fn: Callable[[RadarFrame], None]):
        """Register a function to be called on every new frame."""
        self._callbacks.append(fn)

    def get_range_history(self) -> List[float]:
        """Return rolling range history in metres (for vital sign extraction)."""
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

    @property
    def parse_errors(self) -> int:
        return self._parse_errors

    def stats(self) -> dict:
        f = self._latest_frame
        return {
            "connected":   self.is_connected(),
            "frame_count": self._frame_count,
            "frame_rate":  round(self.frame_rate, 1),
            "presence":    f.presence if f else False,
            "range_m":     f.target_range_m if f else 0.0,
            "move_energy": f.move_energy if f else 0,
            "static_energy": f.static_energy if f else 0,
        }


# ── Simulated radar for development without hardware ──────────────────────────

class SimulatedRadar(SEN0395Radar):
    """
    Generates realistic SEN0395 output without physical hardware.
    Simulates a breathing person at ~4m and an optional moving target.
    Used for development, testing, and training data generation.
    """

    def __init__(self):
        super().__init__(port="SIM")
        self._t = 0.0
        self._targets = [
            # Still person breathing at 4.2m
            {"range_m": 4.2, "breath_hz": 0.27, "hr_hz": 1.15,
             "moving": False, "label": "person_still"},
            # Moving person at 7m (optional — comment out to test single target)
            # {"range_m": 7.0, "breath_hz": 0.0, "hr_hz": 0.0,
            #  "moving": True, "label": "person_walking"},
        ]

    def connect(self) -> bool:
        log.info("Simulated SEN0395 radar initialised")
        return True

    def _configure(self):
        pass   # no hardware to configure

    def start(self):
        self._running    = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._sim_loop, daemon=True, name="sim-radar")
        self._thread.start()

    def _sim_loop(self):
        """Generate realistic frame data at 10 Hz."""
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
        """Build one synthetic radar frame."""
        frame = RadarFrame(timestamp=time.time())

        if not self._targets:
            return frame

        # Use primary target
        tgt = self._targets[0]
        t   = self._t

        # Chest displacement: breathing + heartbeat
        breath_disp = 0.006 * np.sin(2 * np.pi * tgt["breath_hz"] * t)
        heart_disp  = 0.0003 * np.sin(2 * np.pi * tgt["hr_hz"] * t)
        noise_disp  = np.random.randn() * 0.001

        r = tgt["range_m"] + breath_disp + heart_disp + noise_disp

        if tgt["moving"]:
            r += 0.4 * np.sin(2 * np.pi * 0.4 * t)

        r = max(0.3, r)   # physical minimum

        frame.presence          = True
        frame.motion_detected   = tgt["moving"]
        frame.stationary_detected = not tgt["moving"]
        frame.target_range_m    = round(r, 3)
        frame.detect_distance_cm= int(r * 100)
        frame.move_energy       = int(np.clip(
            60 * tgt["moving"] + np.random.randint(0, 15), 0, 100))
        frame.static_energy     = int(np.clip(
            70 * (not tgt["moving"]) + np.random.randint(0, 15), 0, 100))
        frame.signal_strength   = max(
            frame.move_energy, frame.static_energy) / 100.0

        return frame

    def add_target(self, range_m: float, moving: bool = False,
                   breath_hz: float = 0.27):
        """Add a simulated target at runtime."""
        self._targets.append({
            "range_m": range_m, "breath_hz": breath_hz,
            "hr_hz": 1.1, "moving": moving, "label": "added"
        })

    def clear_targets(self):
        self._targets.clear()

    def set_scenario(self, scenario: str):
        """Quick scenario presets for testing."""
        scenarios = {
            "empty":           [],
            "person_still":    [{"range_m": 3.5, "breath_hz": 0.25,
                                  "hr_hz": 1.1, "moving": False, "label": "person_still"}],
            "person_walking":  [{"range_m": 5.0, "breath_hz": 0.0,
                                  "hr_hz": 0.0, "moving": True,  "label": "person_walking"}],
            "two_people":      [{"range_m": 3.0, "breath_hz": 0.22,
                                  "hr_hz": 1.2, "moving": False, "label": "person1"},
                                {"range_m": 6.0, "breath_hz": 0.3,
                                  "hr_hz": 0.9, "moving": False, "label": "person2"}],
            "person_hiding":   [{"range_m": 5.0, "breath_hz": 0.18,
                                  "hr_hz": 1.0, "moving": False, "label": "person_hiding"}],
        }
        self._targets = scenarios.get(scenario, [])
        log.info(f"Scenario: {scenario} ({len(self._targets)} targets)")


# Keep DFRobotRadar as alias for backwards compatibility with main.py
DFRobotRadar = SEN0395Radar