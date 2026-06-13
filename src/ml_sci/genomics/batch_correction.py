"""Batch correction via linear adjustment and optimal transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def linear_batch_correction(expression: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
    """Subtract per-batch mean shift (ComBat-style simplified)."""
    corrected = expression.copy()
    global_mean = expression.mean(axis=0)
    for batch in np.unique(batch_labels):
        mask = batch_labels == batch
        batch_mean = expression[mask].mean(axis=0)
        corrected[mask] -= batch_mean - global_mean
    return corrected


def sinkhorn_transport(
    source: np.ndarray,
    target: np.ndarray,
    reg: float = 0.1,
    n_iters: int = 100,
) -> np.ndarray:
    """Entropic OT plan via Sinkhorn between batch centroids."""
    n_src, n_tgt = source.shape[0], target.shape[0]
    cost = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    k = np.exp(-cost / reg)
    u = np.ones(n_src) / n_src
    v = np.ones(n_tgt) / n_tgt
    for _ in range(n_iters):
        u = 1.0 / (k @ v + 1e-12)
        v = 1.0 / (k.T @ u + 1e-12)
    return (u[:, None] * k) * v[None, :]


def sinkhorn_batch_correction(expression: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
    """OT-based batch alignment via cell-level transport in PCA space."""
    batches = np.unique(batch_labels)
    if len(batches) < 2:
        return expression.copy()

    from sklearn.decomposition import PCA

    corrected = expression.copy()
    pca = PCA(n_components=min(10, expression.shape[1], expression.shape[0] - 1))
    low_dim = pca.fit_transform(expression)

    ref_batch = batches[0]
    ref_mask = batch_labels == ref_batch
    ref_cells = low_dim[ref_mask]

    for batch in batches[1:]:
        mask = batch_labels == batch
        batch_cells = low_dim[mask]
        n_sample = min(80, ref_cells.shape[0], batch_cells.shape[0])
        rng = np.random.default_rng(0)
        ref_idx = rng.choice(ref_cells.shape[0], size=n_sample, replace=False)
        batch_idx = rng.choice(batch_cells.shape[0], size=n_sample, replace=False)
        plan = sinkhorn_transport(batch_cells[batch_idx], ref_cells[ref_idx], reg=0.1)
        mapped = plan @ ref_cells[ref_idx]
        shift = (mapped - batch_cells[batch_idx]).mean(axis=0)
        corrected[mask] = expression[mask] + pca.inverse_transform(
            np.tile(shift, (mask.sum(), 1))
        )

    return corrected


@dataclass
class BatchCorrectionResult:
    """Batch correction outcome."""

    corrected: np.ndarray
    method: str


def apply_batch_correction(
    expression: np.ndarray,
    batch_labels: np.ndarray,
    method: Literal["none", "linear", "sinkhorn"] = "linear",
) -> BatchCorrectionResult:
    """Apply batch correction method."""
    if method == "none":
        return BatchCorrectionResult(corrected=expression.copy(), method=method)
    if method == "linear":
        return BatchCorrectionResult(
            corrected=linear_batch_correction(expression, batch_labels),
            method=method,
        )
    return BatchCorrectionResult(
        corrected=sinkhorn_batch_correction(expression, batch_labels),
        method=method,
    )


def mnn_alignment(expression: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
    """Mutual nearest neighbor batch alignment (simplified)."""
    batches = np.unique(batch_labels)
    if len(batches) < 2:
        return expression.copy()

    corrected = expression.copy()
    b0, b1 = batches[0], batches[1]
    idx0 = np.where(batch_labels == b0)[0]
    idx1 = np.where(batch_labels == b1)[0]
    x0, x1 = expression[idx0], expression[idx1]

    dists = np.linalg.norm(x0[:, None, :] - x1[None, :, :], axis=2)
    pairs0 = dists.argmin(axis=1)
    pairs1 = dists.argmin(axis=0)

    mutual = set()
    for i, j in enumerate(pairs0):
        if pairs1[j] == i:
            mutual.add((i, j))

    if not mutual:
        return linear_batch_correction(expression, batch_labels)

    shifts = []
    for i, j in mutual:
        shifts.append(x0[i] - x1[j])
    mean_shift = np.mean(shifts, axis=0)
    corrected[idx1] = x1 + mean_shift
    return corrected
