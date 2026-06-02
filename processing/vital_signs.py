"""
Vital Signs Extraction
Contactless breathing rate and heart rate from 24GHz FMCW radar.

Method:
  1. Lock to occupied range bin (target location)
  2. Extract complex phase over time — phase ∝ displacement
  3. Bandpass 0.1–0.6 Hz → breathing
  4. Bandpass 0.8–2.5 Hz → heart rate harmonic
  5. FFT peak detection → rate in BPM / breaths-per-min
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import time
import logging

log = logging.getLogger(__name__)

from config.settings import (
    BREATH_FREQ_MIN_HZ, BREATH_FREQ_MAX_HZ,
    HEART_FREQ_MIN_HZ, HEART_FREQ_MAX_HZ,
    VITALS_WINDOW_SEC, VITALS_SAMPLE_RATE
)


@dataclass
class VitalSigns:
    target_id: int
    range_m: float
    timestamp: float

    breath_rate_bpm: Optional[float] = None      # breaths per minute
    heart_rate_bpm: Optional[float] = None       # beats per minute (estimate)
    breath_confidence: float = 0.0               # 0–1
    heart_confidence: float = 0.0               # 0–1

    breath_waveform: List[float] = field(default_factory=list)   # last N samples
    heart_waveform: List[float] = field(default_factory=list)

    status: str = "unknown"   # "breathing", "still", "no_signal", "possible_casualty"


class VitalSignsExtractor:
    """
    Per-target vital signs extraction from phase history.
    Maintains a rolling phase buffer for each tracked target.
    """

    def __init__(self):
        self._phase_buffers: Dict[int, np.ndarray] = {}     # target_id → phase array
        self._timestamps: Dict[int, List[float]] = {}
        self._window_size = int(VITALS_WINDOW_SEC * VITALS_SAMPLE_RATE)

        # Butterworth bandpass filters
        fs = VITALS_SAMPLE_RATE
        nyq = fs / 2.0

        b_breath, a_breath = butter(4,
            [BREATH_FREQ_MIN_HZ / nyq, BREATH_FREQ_MAX_HZ / nyq],
            btype='band')
        b_heart, a_heart = butter(4,
            [HEART_FREQ_MIN_HZ / nyq, HEART_FREQ_MAX_HZ / nyq],
            btype='band')

        self._breath_filter = (b_breath, a_breath)
        self._heart_filter  = (b_heart, a_heart)

    def update(self, target_id: int, range_m: float,
               range_profile: np.ndarray) -> Optional[VitalSigns]:
        """
        Feed a new range profile snapshot for a given target.
        Returns VitalSigns if enough data has accumulated, else None.

        range_profile: complex 1D array (range bins)
        range_m: target range in metres (to select the right bin)
        """
        # Convert range to bin index
        from config.settings import RADAR_NUM_SAMPLES, RADAR_FREQ_START_GHZ, RADAR_BANDWIDTH_GHZ
        c = 3e8
        B = RADAR_BANDWIDTH_GHZ * 1e9
        range_per_bin = c / (2 * B)
        bin_idx = int(range_m / range_per_bin)
        bin_idx = min(bin_idx, len(range_profile) - 1)

        # Extract phase at target range bin
        complex_val = range_profile[bin_idx]
        phase = np.angle(complex_val)

        # Maintain rolling buffer
        if target_id not in self._phase_buffers:
            self._phase_buffers[target_id] = np.array([phase])
            self._timestamps[target_id] = [time.time()]
        else:
            self._phase_buffers[target_id] = np.append(
                self._phase_buffers[target_id], phase)
            self._timestamps[target_id].append(time.time())

            # Trim to window
            if len(self._phase_buffers[target_id]) > self._window_size:
                self._phase_buffers[target_id] = self._phase_buffers[target_id][-self._window_size:]
                self._timestamps[target_id] = self._timestamps[target_id][-self._window_size:]

        buf = self._phase_buffers[target_id]

        # Need enough data
        min_samples = int(VITALS_SAMPLE_RATE * 4)    # 4 seconds minimum
        if len(buf) < min_samples:
            return None

        # Unwrap phase to remove discontinuities
        phase_unwrapped = np.unwrap(buf)

        # Remove DC drift
        phase_detrended = phase_unwrapped - np.polyval(
            np.polyfit(np.arange(len(phase_unwrapped)), phase_unwrapped, 1),
            np.arange(len(phase_unwrapped)))

        return self._extract_rates(target_id, range_m, phase_detrended)

    def _extract_rates(self, target_id: int, range_m: float,
                       phase: np.ndarray) -> VitalSigns:
        fs = VITALS_SAMPLE_RATE
        vitals = VitalSigns(
            target_id=target_id,
            range_m=range_m,
            timestamp=time.time()
        )

        # ── Breathing ──────────────────────────────────────────────────────
        try:
            breath_signal = filtfilt(*self._breath_filter, phase)
            vitals.breath_waveform = breath_signal[-50:].tolist()   # last 5s @ 10Hz

            # FFT peak detection
            N = len(breath_signal)
            freqs = rfftfreq(N, 1.0 / fs)
            spectrum = np.abs(rfft(breath_signal))

            mask = (freqs >= BREATH_FREQ_MIN_HZ) & (freqs <= BREATH_FREQ_MAX_HZ)
            if mask.any():
                sub_freqs  = freqs[mask]
                sub_spec   = spectrum[mask]
                peak_idx   = np.argmax(sub_spec)
                peak_freq  = sub_freqs[peak_idx]
                peak_power = sub_spec[peak_idx]
                noise_floor = np.median(sub_spec)

                snr = peak_power / (noise_floor + 1e-10)
                vitals.breath_rate_bpm   = round(peak_freq * 60, 1)
                vitals.breath_confidence = float(np.clip((snr - 1) / 9.0, 0, 1))
        except Exception as e:
            log.debug(f"Breath extraction error: {e}")

        # ── Heart rate ─────────────────────────────────────────────────────
        try:
            heart_signal = filtfilt(*self._heart_filter, phase)
            vitals.heart_waveform = heart_signal[-50:].tolist()

            N = len(heart_signal)
            freqs = rfftfreq(N, 1.0 / fs)
            spectrum = np.abs(rfft(heart_signal))

            mask = (freqs >= HEART_FREQ_MIN_HZ) & (freqs <= HEART_FREQ_MAX_HZ)
            if mask.any():
                sub_freqs  = freqs[mask]
                sub_spec   = spectrum[mask]
                peak_idx   = np.argmax(sub_spec)
                peak_freq  = sub_freqs[peak_idx]
                peak_power = sub_spec[peak_idx]
                noise_floor = np.median(sub_spec)

                snr = peak_power / (noise_floor + 1e-10)
                vitals.heart_rate_bpm   = round(peak_freq * 60, 1)
                vitals.heart_confidence = float(np.clip((snr - 2) / 8.0, 0, 1))
        except Exception as e:
            log.debug(f"Heart rate extraction error: {e}")

        # ── Status classification ──────────────────────────────────────────
        if vitals.breath_confidence > 0.4:
            vitals.status = "breathing"
        elif vitals.breath_confidence < 0.15:
            # Check if we have enough data — might be a casualty
            buf_len = len(self._phase_buffers.get(target_id, []))
            if buf_len >= self._window_size * 0.8:
                vitals.status = "possible_casualty"
            else:
                vitals.status = "insufficient_data"
        else:
            vitals.status = "still"

        return vitals

    def clear_target(self, target_id: int):
        """Remove a target's buffer when its track is deleted."""
        self._phase_buffers.pop(target_id, None)
        self._timestamps.pop(target_id, None)

    def get_all_vitals_summary(self) -> List[dict]:
        """Return latest vitals for all tracked targets."""
        # Returns cached last-known values
        return []
