"""
Back-Projection Spatial Imaging Engine
Reconstructs a 2D through-wall image from FMCW radar I/Q data.
This is the core algorithm that produces the RuView-style spatial picture.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from config.settings import (
    IMAGE_RANGE_M, IMAGE_WIDTH_M, IMAGE_GRID_STEPS,
    RADAR_FREQ_START_GHZ, RADAR_FREQ_END_GHZ,
    RADAR_NUM_SAMPLES, RADAR_NUM_CHIRPS
)


@dataclass
class SpatialImage:
    """Output of the back-projection: a 2D power map with metadata."""
    power_db: np.ndarray           # [grid x grid] image in dB
    x_axis_m: np.ndarray           # lateral axis (metres)
    y_axis_m: np.ndarray           # range axis (metres from radar)
    timestamp: float = 0.0
    detections: List[dict] = field(default_factory=list)  # detected targets


class BackProjectionImager:
    """
    Converts FMCW radar I/Q frames into a focused 2D spatial image.

    Algorithm:
      1. Range FFT on each chirp → range-compressed data
      2. For each pixel (x,y) in the output grid:
           - Compute expected round-trip delay to that pixel
           - Extract and phase-correct the range-compressed return
           - Coherently accumulate across all chirps and frequencies
      3. Magnitude → power in dB → 2D image

    With the patch antenna as a second element, we get a 2-element
    synthetic aperture that improves azimuth (lateral) resolution.
    """

    def __init__(self):
        self.c = 3e8
        self.f_start = RADAR_FREQ_START_GHZ * 1e9
        self.f_end   = RADAR_FREQ_END_GHZ   * 1e9
        self.B       = self.f_end - self.f_start

        # Build output grid
        self.x_m = np.linspace(-IMAGE_WIDTH_M / 2, IMAGE_WIDTH_M / 2, IMAGE_GRID_STEPS)
        self.y_m = np.linspace(0.5, IMAGE_RANGE_M, IMAGE_GRID_STEPS)
        self.grid_x, self.grid_y = np.meshgrid(self.x_m, self.y_m)

        # Precompute frequency array
        self.freq_array = np.linspace(self.f_start, self.f_end, RADAR_NUM_SAMPLES)

        # Precompute range array (metres per range bin)
        self.range_per_bin = self.c / (2 * self.B)

        # Precompute pixel ranges (shape: [grid x grid])
        self._pixel_ranges = np.sqrt(self.grid_x**2 + self.grid_y**2)

        # Rolling buffer for slow-time Doppler and vital signs
        self._range_profiles = []      # list of 1D range profiles over time
        self._max_buffer = 200         # ~20 seconds at 10 Hz

    def process_frame(self, iq_data: np.ndarray) -> SpatialImage:
        """
        Main entry point. Takes raw I/Q [num_chirps x num_samples] and
        returns a focused SpatialImage.
        """
        import time

        # Step 1: Range compression (FFT along fast-time axis)
        range_profiles = self._range_fft(iq_data)      # [chirps x range_bins]

        # Step 2: Static clutter removal (subtract mean over chirps)
        clutter_removed = self._remove_clutter(range_profiles)

        # Step 3: Store range profile for temporal analysis
        mean_profile = np.mean(np.abs(clutter_removed), axis=0)
        self._range_profiles.append(mean_profile)
        if len(self._range_profiles) > self._max_buffer:
            self._range_profiles.pop(0)

        # Step 4: Coherent back-projection → 2D image
        image = self._back_project(clutter_removed)

        # Step 5: Convert to dB
        power_db = 20 * np.log10(np.abs(image) + 1e-10)

        # Normalise to [0, 1] range for display
        power_db = np.clip(power_db, np.max(power_db) - 40, np.max(power_db))
        power_norm = (power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-10)

        img = SpatialImage(
            power_db=power_norm,
            x_axis_m=self.x_m,
            y_axis_m=self.y_m,
            timestamp=time.time()
        )

        return img

    def _range_fft(self, iq: np.ndarray) -> np.ndarray:
        """Apply Hanning window and FFT along fast-time (range) axis."""
        window = np.hanning(iq.shape[1])
        windowed = iq * window[np.newaxis, :]
        return np.fft.fft(windowed, axis=1)[:, :iq.shape[1]//2]

    def _remove_clutter(self, range_profiles: np.ndarray) -> np.ndarray:
        """Subtract temporal mean to remove static reflections."""
        mean = np.mean(range_profiles, axis=0)
        return range_profiles - mean[np.newaxis, :]

    def _back_project(self, range_profiles: np.ndarray) -> np.ndarray:
        """
        Coherent back-projection.
        For each output pixel, phase-correct and accumulate contributions
        from all chirps. Produces a focused 2D image.
        """
        image = np.zeros_like(self.grid_x, dtype=complex)
        num_chirps, num_bins = range_profiles.shape

        # Range bin index for each pixel
        range_bin_indices = (self._pixel_ranges / self.range_per_bin).astype(int)
        valid_mask = range_bin_indices < num_bins

        for chirp_idx in range(num_chirps):
            profile = range_profiles[chirp_idx]

            # Phase correction for each pixel's round-trip distance
            # phi = 4*pi*f0*r/c (two-way phase at carrier frequency)
            phase_correction = np.exp(-1j * 4 * np.pi * self.f_start * self._pixel_ranges / self.c)

            # Extract range-compressed values at each pixel's range bin
            safe_indices = np.clip(range_bin_indices, 0, num_bins - 1)
            pixel_values = profile[safe_indices]
            pixel_values[~valid_mask] = 0

            image += pixel_values * phase_correction

        # Normalise by number of chirps
        image /= num_chirps
        return image

    def get_range_time_matrix(self) -> Optional[np.ndarray]:
        """
        Return the accumulated range-time matrix for Doppler / vital signs.
        Shape: [time_frames x range_bins]
        """
        if len(self._range_profiles) < 2:
            return None
        return np.array(self._range_profiles)

    def detect_peaks(self, image: SpatialImage, threshold: float = 0.6) -> List[dict]:
        """
        Find target peaks in the spatial image above threshold.
        Returns list of dicts with x_m, y_m, range_m, power.
        """
        from scipy.ndimage import label, center_of_mass

        binary = image.power_db > threshold
        labeled, num_features = label(binary)

        targets = []
        for region_id in range(1, num_features + 1):
            mask = labeled == region_id
            if mask.sum() < 3:   # ignore tiny blobs
                continue

            # Centre of mass in grid coordinates
            cy, cx = center_of_mass(image.power_db * mask)
            cy_i, cx_i = int(cy), int(cx)

            if 0 <= cy_i < len(image.y_axis_m) and 0 <= cx_i < len(image.x_axis_m):
                x_m = image.x_axis_m[cx_i]
                y_m = image.y_axis_m[cy_i]
                r_m = np.sqrt(x_m**2 + y_m**2)
                az_deg = np.degrees(np.arctan2(x_m, y_m))
                peak_power = float(np.max(image.power_db[mask]))

                targets.append({
                    "x_m":     round(x_m, 2),
                    "y_m":     round(y_m, 2),
                    "range_m": round(r_m, 2),
                    "az_deg":  round(az_deg, 1),
                    "power":   round(peak_power, 3),
                    "size_px": int(mask.sum())
                })

        # Sort by power descending
        targets.sort(key=lambda t: t["power"], reverse=True)
        return targets[:8]   # max 8 targets
