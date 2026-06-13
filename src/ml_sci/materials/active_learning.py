"""Active learning for expensive materials property oracle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ml_sci.data.materials_dgp import property_oracle
from ml_sci.materials.metrics import normalized_score, regret, top_k_hit_rate


@dataclass
class ActiveLearningResult:
    """Active learning run outcome."""

    best_property: float
    best_composition: np.ndarray
    all_properties: list[float] = field(default_factory=list)
    n_evals: int = 0


def _gp_surrogate(seed: int) -> GaussianProcessRegressor:
    kernel = RBF(length_scale=0.5) + WhiteKernel(noise_level=1e-4)
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)


def random_search(
    pool: np.ndarray,
    budget: int,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    length_scale: float,
    seed: int,
) -> ActiveLearningResult:
    """Uniform random sampling from composition pool."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool.shape[0], size=budget, replace=False)
    props = [float(property_oracle(pool[i : i + 1], optimum, active_dims, length_scale)[0]) for i in idx]
    best_i = int(np.argmax(props))
    return ActiveLearningResult(
        best_property=props[best_i],
        best_composition=pool[idx[best_i]].copy(),
        all_properties=props,
        n_evals=budget,
    )


def uncertainty_search(
    pool: np.ndarray,
    budget: int,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    length_scale: float,
    seed: int,
    init_size: int = 5,
) -> ActiveLearningResult:
    """GP uncertainty sampling (max posterior variance)."""
    rng = np.random.default_rng(seed)
    n_pool = pool.shape[0]
    init_idx = rng.choice(n_pool, size=init_size, replace=False)
    observed_idx = list(init_idx)
    props: list[float] = []

    for i in observed_idx:
        props.append(float(property_oracle(pool[i : i + 1], optimum, active_dims, length_scale)[0]))

    remaining = budget - init_size
    for _ in range(remaining):
        gp = _gp_surrogate(seed)
        x_obs = pool[observed_idx]
        gp.fit(x_obs, props)
        candidates = np.setdiff1d(np.arange(n_pool), observed_idx)
        _, std = gp.predict(pool[candidates], return_std=True)
        pick = candidates[int(np.argmax(std))]
        observed_idx.append(pick)
        props.append(float(property_oracle(pool[pick : pick + 1], optimum, active_dims, length_scale)[0]))

    best_i = int(np.argmax(props))
    return ActiveLearningResult(
        best_property=props[best_i],
        best_composition=pool[observed_idx[best_i]].copy(),
        all_properties=props,
        n_evals=budget,
    )


def expected_improvement_search(
    pool: np.ndarray,
    budget: int,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    length_scale: float,
    seed: int,
    init_size: int = 5,
) -> ActiveLearningResult:
    """GP expected improvement acquisition."""
    rng = np.random.default_rng(seed)
    n_pool = pool.shape[0]
    init_idx = rng.choice(n_pool, size=init_size, replace=False)
    observed_idx = list(init_idx)
    props: list[float] = []

    for i in observed_idx:
        props.append(float(property_oracle(pool[i : i + 1], optimum, active_dims, length_scale)[0]))

    remaining = budget - init_size
    for _ in range(remaining):
        gp = _gp_surrogate(seed)
        x_obs = pool[observed_idx]
        gp.fit(x_obs, props)
        best_so_far = max(props)
        candidates = np.setdiff1d(np.arange(n_pool), observed_idx)
        mu, std = gp.predict(pool[candidates], return_std=True)
        std = np.maximum(std, 1e-9)
        z = (mu - best_so_far) / std
        from scipy.stats import norm

        ei = (mu - best_so_far) * norm.cdf(z) + std * norm.pdf(z)
        pick = candidates[int(np.argmax(ei))]
        observed_idx.append(pick)
        props.append(float(property_oracle(pool[pick : pick + 1], optimum, active_dims, length_scale)[0]))

    best_i = int(np.argmax(props))
    return ActiveLearningResult(
        best_property=props[best_i],
        best_composition=pool[observed_idx[best_i]].copy(),
        all_properties=props,
        n_evals=budget,
    )


def run_active_learning(
    pool: np.ndarray,
    budget: int,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    length_scale: float,
    method: Literal["random", "uncertainty", "expected_improvement"],
    seed: int,
) -> ActiveLearningResult:
    """Dispatch active learning strategy."""
    if method == "random":
        return random_search(pool, budget, optimum, active_dims, length_scale, seed)
    if method == "uncertainty":
        return uncertainty_search(pool, budget, optimum, active_dims, length_scale, seed)
    return expected_improvement_search(pool, budget, optimum, active_dims, length_scale, seed)
