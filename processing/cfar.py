"""
CFAR (Constant False Alarm Rate) clutter removal.
Isolates targets from static reflections (walls, furniture, ground).
"""

import numpy as np
from config.settings import CFAR_GUARD_CELLS, CFAR_TRAINING_CELLS, CFAR_PFA


def cfar_1d(signal: np.ndarray, guard: int = None, train: int = None, pfa: float = None) -> np.ndarray:
    """
    1D Cell-Averaging CFAR detector.
    Returns boolean mask: True = target detected in that cell.

    signal  : 1D array of power values (range profile)
    guard   : guard cells each side (protect target from noise estimate)
    train   : training cells each side (estimate local noise floor)
    pfa     : desired probability of false alarm
    """
    guard = guard or CFAR_GUARD_CELLS
    train = train or CFAR_TRAINING_CELLS
    pfa   = pfa   or CFAR_PFA

    # Threshold multiplier alpha from PFA for CA-CFAR
    alpha = train * (pfa ** (-1.0 / train) - 1.0)
    N = len(signal)
    detections = np.zeros(N, dtype=bool)
    half = guard + train

    for i in range(half, N - half):
        left  = signal[i - half : i - guard]
        right = signal[i + guard + 1 : i + half + 1]
        noise_est = np.mean(np.concatenate([left, right]))
        threshold = alpha * noise_est
        if signal[i] > threshold:
            detections[i] = True

    return detections


def cfar_2d(range_doppler: np.ndarray, guard: int = 2, train: int = 4, pfa: float = 1e-4) -> np.ndarray:
    """
    2D Cell-Averaging CFAR on a range-Doppler map.
    Returns boolean mask of detected targets.
    """
    power = np.abs(range_doppler) ** 2
    alpha = (guard + train) ** 2 * (pfa ** (-1.0 / ((guard + train) ** 2 - guard ** 2)) - 1.0)
    rows, cols = power.shape
    detections = np.zeros((rows, cols), dtype=bool)
    pad = guard + train

    for r in range(pad, rows - pad):
        for c in range(pad, cols - pad):
            cell = power[r, c]
            # Training region excluding guard cells
            region = power[r-pad:r+pad+1, c-pad:c+pad+1].copy()
            region[r-pad+train:r-pad+train+2*guard+1,
                   c-pad+train:c-pad+train+2*guard+1] = 0
            noise_cells = region[region > 0]
            if len(noise_cells) == 0:
                continue
            noise_est = np.mean(noise_cells)
            if cell > alpha * noise_est:
                detections[r, c] = True

    return detections


def remove_static_clutter(frames: np.ndarray) -> np.ndarray:
    """
    Remove static background by subtracting temporal mean.
    frames: [num_frames x num_range_bins] complex array
    Returns clutter-free frames.
    """
    mean_frame = np.mean(frames, axis=0)
    return frames - mean_frame[np.newaxis, :]
