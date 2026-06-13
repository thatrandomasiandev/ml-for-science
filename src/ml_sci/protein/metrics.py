"""Protein property prediction metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else 0.0


def rotation_consistency(
    y_pred_original: np.ndarray,
    y_pred_rotated: np.ndarray,
) -> float:
    """Fraction of predictions unchanged under rotation (equivariance score)."""
    rel_err = np.abs(y_pred_original - y_pred_rotated) / (np.abs(y_pred_original) + 1e-6)
    return float(np.mean(rel_err < 0.05))
