"""Batch correction methods for single-cell expression data.

Includes classical approaches (linear mean-shift, Sinkhorn OT, MNN)
and two structured correctors:

* :class:`CombatCorrector` — empirical Bayes location/scale adjustment
  (Johnson et al., 2007).
* :class:`HarmonyCorrector` — simplified iterative soft-clustering EM
  (Korsunsky et al., 2019).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple functional correctors (legacy)
# ---------------------------------------------------------------------------


def linear_batch_correction(
    expression: np.ndarray, batch_labels: np.ndarray
) -> np.ndarray:
    """Subtract per-batch mean shift (ComBat-style simplified).

    Args:
        expression: Expression matrix ``(N, G)``.
        batch_labels: Batch assignment per cell ``(N,)``.

    Returns:
        Corrected expression matrix ``(N, G)``.
    """
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
    """Entropic OT plan via Sinkhorn between two point clouds.

    Args:
        source: Source points ``(n_src, D)``.
        target: Target points ``(n_tgt, D)``.
        reg: Entropic regularisation coefficient.
        n_iters: Number of Sinkhorn iterations.

    Returns:
        Transport plan ``(n_src, n_tgt)``.
    """
    n_src, n_tgt = source.shape[0], target.shape[0]
    cost = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    k = np.exp(-cost / reg)
    u = np.ones(n_src) / n_src
    v = np.ones(n_tgt) / n_tgt
    for _ in range(n_iters):
        u = 1.0 / (k @ v + 1e-12)
        v = 1.0 / (k.T @ u + 1e-12)
    return (u[:, None] * k) * v[None, :]


def sinkhorn_batch_correction(
    expression: np.ndarray, batch_labels: np.ndarray
) -> np.ndarray:
    """OT-based batch alignment via cell-level transport in PCA space.

    Args:
        expression: Expression matrix ``(N, G)``.
        batch_labels: Batch assignment per cell ``(N,)``.

    Returns:
        Corrected expression matrix ``(N, G)``.
    """
    batches = np.unique(batch_labels)
    if len(batches) < 2:
        return expression.copy()

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
        plan = sinkhorn_transport(
            batch_cells[batch_idx], ref_cells[ref_idx], reg=0.1
        )
        mapped = plan @ ref_cells[ref_idx]
        shift = (mapped - batch_cells[batch_idx]).mean(axis=0)
        corrected[mask] = expression[mask] + pca.inverse_transform(
            np.tile(shift, (mask.sum(), 1))
        )

    return corrected


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class BatchCorrectionResult:
    """Batch correction outcome.

    Args:
        corrected: Corrected expression matrix ``(N, G)``.
        method: Name of the correction method used.
    """

    corrected: np.ndarray
    method: str


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def apply_batch_correction(
    expression: np.ndarray,
    batch_labels: np.ndarray,
    method: Literal["none", "linear", "sinkhorn"] = "linear",
) -> BatchCorrectionResult:
    """Apply a batch correction method.

    Args:
        expression: Expression matrix ``(N, G)``.
        batch_labels: Batch assignment ``(N,)``.
        method: ``"none"`` | ``"linear"`` | ``"sinkhorn"``.

    Returns:
        :class:`BatchCorrectionResult`.
    """
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


# ---------------------------------------------------------------------------
# MNN alignment (legacy)
# ---------------------------------------------------------------------------


def mnn_alignment(
    expression: np.ndarray, batch_labels: np.ndarray
) -> np.ndarray:
    """Mutual nearest neighbor batch alignment (simplified).

    Args:
        expression: Expression matrix ``(N, G)``.
        batch_labels: Batch assignment ``(N,)``.

    Returns:
        Corrected expression matrix ``(N, G)``.
    """
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


# ---------------------------------------------------------------------------
# CombatCorrector — empirical Bayes batch correction
# ---------------------------------------------------------------------------


class CombatCorrector:
    r"""Empirical Bayes batch correction (ComBat).

    Implements the location-scale model of Johnson et al. (2007):

    .. math::

        Y_{ijg} = \alpha_g + X\beta_g + \gamma_{ig} + \delta_{ig}\epsilon_{ijg}

    where :math:`\gamma_{ig}` is the additive batch effect and
    :math:`\delta_{ig}` is the multiplicative batch effect, both
    shrunken toward their prior via empirical Bayes.

    Args:
        seed: Random seed (currently unused; deterministic algorithm).
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._grand_mean: np.ndarray | None = None
        self._gamma_star: dict[int, np.ndarray] = {}
        self._delta_star: dict[int, np.ndarray] = {}
        self._batches: np.ndarray | None = None
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray, batches: np.ndarray) -> CombatCorrector:
        """Estimate batch-effect parameters with empirical Bayes shrinkage.

        Args:
            X: Expression matrix ``(N, G)``.
            batches: Batch labels ``(N,)``.

        Returns:
            ``self`` for method chaining.
        """
        self._batches = np.unique(batches)
        n_genes = X.shape[1]
        self._grand_mean = X.mean(axis=0)

        gamma_hat: dict[int, np.ndarray] = {}
        delta_hat: dict[int, np.ndarray] = {}

        for b in self._batches:
            mask = batches == b
            batch_data = X[mask]
            gamma_hat[b] = batch_data.mean(axis=0) - self._grand_mean
            residuals = batch_data - self._grand_mean - gamma_hat[b]
            delta_hat[b] = residuals.var(axis=0) + 1e-8

        gamma_vals = np.stack(list(gamma_hat.values()), axis=0)
        gamma_prior_mean = gamma_vals.mean(axis=0)
        gamma_prior_var = gamma_vals.var(axis=0) + 1e-8

        delta_vals = np.stack(list(delta_hat.values()), axis=0)
        delta_prior_mean = delta_vals.mean(axis=0)
        delta_prior_var = delta_vals.var(axis=0) + 1e-8

        for b in self._batches:
            mask = batches == b
            n_b = mask.sum()

            shrink_w = gamma_prior_var / (gamma_prior_var + delta_hat[b] / n_b + 1e-12)
            self._gamma_star[b] = shrink_w * gamma_hat[b] + (1 - shrink_w) * gamma_prior_mean

            alpha = delta_prior_mean ** 2 / (delta_prior_var + 1e-12) + 2.0
            beta_param = delta_prior_mean * (alpha - 1.0)
            ss = ((X[mask] - self._grand_mean - gamma_hat[b]) ** 2).sum(axis=0)
            self._delta_star[b] = (ss + 2.0 * beta_param) / (n_b + 2.0 * alpha - 2.0 + 1e-12)

        self._is_fitted = True
        logger.info("CombatCorrector fitted on %d batches", len(self._batches))
        return self

    def transform(self, X: np.ndarray, batches: np.ndarray) -> np.ndarray:
        """Apply the fitted correction to expression data.

        Args:
            X: Expression matrix ``(N, G)``.
            batches: Batch labels ``(N,)``.

        Returns:
            Corrected expression matrix ``(N, G)``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("CombatCorrector has not been fitted.")
        corrected = X.copy()
        for b in self._batches:
            mask = batches == b
            if not mask.any():
                continue
            corrected[mask] = (
                (X[mask] - self._gamma_star[b])
                / np.sqrt(self._delta_star[b] + 1e-12)
                * np.sqrt(np.ones(X.shape[1]))  # unit scale
                + self._grand_mean
            )
        return corrected

    def fit_transform(self, X: np.ndarray, batches: np.ndarray) -> np.ndarray:
        """Fit and transform in one call.

        Args:
            X: Expression matrix ``(N, G)``.
            batches: Batch labels ``(N,)``.

        Returns:
            Corrected expression matrix ``(N, G)``.
        """
        return self.fit(X, batches).transform(X, batches)


# ---------------------------------------------------------------------------
# HarmonyCorrector — soft-clustering EM batch integration
# ---------------------------------------------------------------------------


class HarmonyCorrector:
    r"""Simplified Harmony-style batch integration via iterative EM.

    Projects data into PCA space, then alternates between:

    1. **E-step**: soft-assign cells to :math:`K` clusters using a
       batch-diversity-regularised objective.
    2. **M-step**: compute per-cluster, per-batch centroids and correct
       each cell toward its cluster's global centroid.

    The algorithm converges when cluster assignments stabilise.

    Args:
        n_components: Number of PCA dimensions.
        n_clusters: Number of soft clusters :math:`K`.
        max_iters: Maximum EM iterations.
        sigma: Bandwidth for soft assignment kernel.
        theta: Diversity penalty strength.
        seed: Random seed.
    """

    def __init__(
        self,
        n_components: int = 20,
        n_clusters: int = 5,
        max_iters: int = 20,
        sigma: float = 0.1,
        theta: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.sigma = sigma
        self.theta = theta
        self.seed = seed

    def fit_transform(
        self, X: np.ndarray, batches: np.ndarray
    ) -> np.ndarray:
        """Correct batch effects in-place in PCA space and project back.

        Args:
            X: Expression matrix ``(N, G)``.
            batches: Batch labels ``(N,)``.

        Returns:
            Corrected expression matrix ``(N, G)``.
        """
        rng = np.random.default_rng(self.seed)
        n_cells, n_genes = X.shape
        n_comp = min(self.n_components, n_genes, n_cells - 1)

        pca = PCA(n_components=n_comp)
        Z = pca.fit_transform(X)

        unique_batches = np.unique(batches)
        n_batches = len(unique_batches)
        batch_idx = np.zeros(n_cells, dtype=np.int64)
        for i, b in enumerate(unique_batches):
            batch_idx[batches == b] = i

        batch_freq = np.array(
            [np.mean(batch_idx == i) for i in range(n_batches)]
        )

        K = min(self.n_clusters, n_cells)
        init_idx = rng.choice(n_cells, size=K, replace=False)
        centroids = Z[init_idx].copy()

        for iteration in range(self.max_iters):
            dists = np.sum(
                (Z[:, None, :] - centroids[None, :, :]) ** 2, axis=2
            )
            logits = -dists / (2.0 * self.sigma ** 2 + 1e-12)
            logits -= logits.max(axis=1, keepdims=True)
            R = np.exp(logits)

            for k in range(K):
                for b in range(n_batches):
                    mask_b = batch_idx == b
                    penalty = (R[mask_b, k].sum() / (R[:, k].sum() + 1e-12)) / (
                        batch_freq[b] + 1e-12
                    )
                    R[mask_b, k] *= np.maximum(1.0 - self.theta * penalty, 0.1)

            R /= R.sum(axis=1, keepdims=True) + 1e-12

            new_centroids = (R.T @ Z) / (R.sum(axis=0)[:, None] + 1e-12)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < 1e-6:
                logger.info("HarmonyCorrector converged at iteration %d", iteration)
                break

            for k in range(K):
                global_centroid = centroids[k]
                for b in range(n_batches):
                    mask = batch_idx == b
                    weights = R[mask, k]
                    if weights.sum() < 1e-12:
                        continue
                    batch_centroid = (weights[:, None] * Z[mask]).sum(axis=0) / (
                        weights.sum() + 1e-12
                    )
                    correction = global_centroid - batch_centroid
                    Z[mask] += weights[:, None] * correction[None, :]

        corrected = pca.inverse_transform(Z)
        return corrected
