"""Climate downscaling metrics."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spectral_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative high-frequency energy deficit (downscaling sharpness)."""
    fft_true = np.fft.fft2(y_true)
    fft_pred = np.fft.fft2(y_pred)
    n = y_true.shape[0]
    cy, cx = n // 2, n // 2
    yy, xx = np.ogrid[:n, :n]
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) > (n // 4) ** 2
    hf_true = np.sum(np.abs(fft_true[mask]) ** 2)
    hf_pred = np.sum(np.abs(fft_pred[mask]) ** 2)
    if hf_true < 1e-12:
        return 0.0
    return float(1.0 - hf_pred / hf_true)


def physics_residual_rms(laplacian: np.ndarray) -> float:
    return float(np.sqrt(np.mean(laplacian**2)))
