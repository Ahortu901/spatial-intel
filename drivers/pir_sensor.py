"""
PIR Sensor Driver — SimplyTronics Wide Angle ST-00001
GPIO interrupt-based presence detection with confidence scoring.
"""

import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable

log = logging.getLogger(__name__)


@dataclass
class PIREvent:
    timestamp: float
    detected: bool
    duration_s: float = 0.0
    confidence: float = 0.0


class PIRSensor:
    def __init__(self, gpio_pin: int = 17):
        self.pin = gpio_pin
        self._detected = False
        self._last_trigger = 0.0
        self._callbacks = []
        self._event_log = []
        self._running = False

    def connect(self) -> bool:
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            GPIO.add_event_detect(self.pin, GPIO.BOTH, callback=self._gpio_callback, bouncetime=200)
            log.info(f"PIR sensor initialised on GPIO {self.pin}")
            return True
        except ImportError:
            log.warning("RPi.GPIO not available — running PIR in simulation mode")
            self._start_simulation()
            return True
        except Exception as e:
            log.error(f"PIR init failed: {e}")
            return False

    def _gpio_callback(self, channel):
        try:
            import RPi.GPIO as GPIO
            state = GPIO.input(self.pin)
        except Exception:
            state = 1

        now = time.time()
        self._detected = bool(state)

        duration = now - self._last_trigger if self._detected else 0.0
        if self._detected:
            self._last_trigger = now

        event = PIREvent(
            timestamp=now,
            detected=self._detected,
            duration_s=duration,
            confidence=0.9 if self._detected else 0.0
        )
        self._event_log.append(event)
        if len(self._event_log) > 200:
            self._event_log.pop(0)

        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                log.warning(f"PIR callback error: {e}")

    def _start_simulation(self):
        """Simulate PIR detections for development."""
        def _sim():
            import random
            while True:
                time.sleep(random.uniform(5, 15))
                self._gpio_callback(self.pin)
                time.sleep(random.uniform(1, 4))
                self._gpio_callback(self.pin)
        t = threading.Thread(target=_sim, daemon=True)
        t.start()

    def is_detected(self) -> bool:
        return self._detected

    def time_since_last(self) -> float:
        return time.time() - self._last_trigger if self._last_trigger > 0 else float('inf')

    def register_callback(self, fn: Callable):
        self._callbacks.append(fn)

    def disconnect(self):
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup(self.pin)
        except Exception:
            pass
