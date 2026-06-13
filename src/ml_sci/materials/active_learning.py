"""Active learning and Bayesian optimisation for materials discovery.

Provides both the original pool-based search functions and a
composable object-oriented stack:

* :func:`expected_improvement`, :func:`upper_confidence_bound`,
  :func:`probability_of_improvement` — standalone acquisition functions.
* :class:`BayesianOptimizer` — surrogate GP + acquisition loop.
* :class:`ActiveLearningLoop` — high-level driver wrapping the
  optimizer with budget tracking and diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ml_sci.data.materials_dgp import property_oracle
from ml_sci.materials.metrics import normalized_score, regret, top_k_hit_rate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ActiveLearningResult:
    """Active learning run outcome.

    Args:
        best_property: Best observed property value.
        best_composition: Composition vector yielding the best value.
        all_properties: All observed property values in order.
        n_evals: Total number of oracle evaluations.
    """

    best_property: float
    best_composition: np.ndarray
    all_properties: list[float] = field(default_factory=list)
    n_evals: int = 0


# ---------------------------------------------------------------------------
# GP factory
# ---------------------------------------------------------------------------


def _gp_surrogate(seed: int) -> GaussianProcessRegressor:
    kernel = RBF(length_scale=0.5) + WhiteKernel(noise_level=1e-4)
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=seed
    )


# ---------------------------------------------------------------------------
# Pool-based search functions (legacy)
# ---------------------------------------------------------------------------


def random_search(
    pool: np.ndarray,
    budget: int,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    length_scale: float,
    seed: int,
) -> ActiveLearningResult:
    """Uniform random sampling from composition pool.

    Args:
        pool: Candidate compositions ``(M, D)``.
        budget: Number of oracle evaluations.
        optimum: True optimum composition ``(D,)``.
        active_dims: Active feature dimensions.
        length_scale: Oracle length scale.
        seed: Random seed.

    Returns:
        :class:`ActiveLearningResult`.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool.shape[0], size=budget, replace=False)
    props = [
        float(
            property_oracle(pool[i : i + 1], optimum, active_dims, length_scale)[0]
        )
        for i in idx
    ]
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
    """GP uncertainty sampling (max posterior variance).

    Args:
        pool: Candidate compositions ``(M, D)``.
        budget: Total oracle evaluations (including initial).
        optimum: True optimum composition.
        active_dims: Active feature dimensions.
        length_scale: Oracle length scale.
        seed: Random seed.
        init_size: Number of initial random samples.

    Returns:
        :class:`ActiveLearningResult`.
    """
    rng = np.random.default_rng(seed)
    n_pool = pool.shape[0]
    init_idx = rng.choice(n_pool, size=init_size, replace=False)
    observed_idx = list(init_idx)
    props: list[float] = []

    for i in observed_idx:
        props.append(
            float(
                property_oracle(pool[i : i + 1], optimum, active_dims, length_scale)[0]
            )
        )

    remaining = budget - init_size
    for _ in range(remaining):
        gp = _gp_surrogate(seed)
        x_obs = pool[observed_idx]
        gp.fit(x_obs, props)
        candidates = np.setdiff1d(np.arange(n_pool), observed_idx)
        _, std = gp.predict(pool[candidates], return_std=True)
        pick = candidates[int(np.argmax(std))]
        observed_idx.append(pick)
        props.append(
            float(
                property_oracle(pool[pick : pick + 1], optimum, active_dims, length_scale)[0]
            )
        )

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
    """GP expected improvement acquisition.

    Args:
        pool: Candidate compositions ``(M, D)``.
        budget: Total oracle evaluations.
        optimum: True optimum composition.
        active_dims: Active feature dimensions.
        length_scale: Oracle length scale.
        seed: Random seed.
        init_size: Number of initial random samples.

    Returns:
        :class:`ActiveLearningResult`.
    """
    rng = np.random.default_rng(seed)
    n_pool = pool.shape[0]
    init_idx = rng.choice(n_pool, size=init_size, replace=False)
    observed_idx = list(init_idx)
    props: list[float] = []

    for i in observed_idx:
        props.append(
            float(
                property_oracle(pool[i : i + 1], optimum, active_dims, length_scale)[0]
            )
        )

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
        ei = (mu - best_so_far) * norm.cdf(z) + std * norm.pdf(z)
        pick = candidates[int(np.argmax(ei))]
        observed_idx.append(pick)
        props.append(
            float(
                property_oracle(pool[pick : pick + 1], optimum, active_dims, length_scale)[0]
            )
        )

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
    """Dispatch active learning strategy.

    Args:
        pool: Candidate compositions.
        budget: Oracle evaluation budget.
        optimum: True optimum.
        active_dims: Active feature dimensions.
        length_scale: Oracle length scale.
        method: Strategy name.
        seed: Random seed.

    Returns:
        :class:`ActiveLearningResult`.
    """
    if method == "random":
        return random_search(pool, budget, optimum, active_dims, length_scale, seed)
    if method == "uncertainty":
        return uncertainty_search(
            pool, budget, optimum, active_dims, length_scale, seed
        )
    return expected_improvement_search(
        pool, budget, optimum, active_dims, length_scale, seed
    )


# ---------------------------------------------------------------------------
# Standalone acquisition functions
# ---------------------------------------------------------------------------


def expected_improvement(
    mu: np.ndarray,
    std: np.ndarray,
    best_f: float,
    xi: float = 0.01,
) -> np.ndarray:
    r"""Expected improvement acquisition.

    .. math::

        \text{EI}(x) = (\mu(x) - f^* - \xi)\,\Phi(Z)
                       + \sigma(x)\,\phi(Z),
        \quad Z = \frac{\mu(x) - f^* - \xi}{\sigma(x)}

    Args:
        mu: Posterior mean ``(M,)``.
        std: Posterior standard deviation ``(M,)``.
        best_f: Best observed function value.
        xi: Exploration-exploitation trade-off parameter.

    Returns:
        EI values ``(M,)``.
    """
    std = np.maximum(std, 1e-9)
    z = (mu - best_f - xi) / std
    return (mu - best_f - xi) * norm.cdf(z) + std * norm.pdf(z)


def upper_confidence_bound(
    mu: np.ndarray,
    std: np.ndarray,
    kappa: float = 2.0,
) -> np.ndarray:
    r"""Upper confidence bound acquisition.

    .. math::

        \text{UCB}(x) = \mu(x) + \kappa\,\sigma(x)

    Args:
        mu: Posterior mean ``(M,)``.
        std: Posterior standard deviation ``(M,)``.
        kappa: Exploration weight.

    Returns:
        UCB values ``(M,)``.
    """
    return mu + kappa * std


def probability_of_improvement(
    mu: np.ndarray,
    std: np.ndarray,
    best_f: float,
    xi: float = 0.01,
) -> np.ndarray:
    r"""Probability of improvement acquisition.

    .. math::

        \text{PI}(x) = \Phi\!\left(\frac{\mu(x) - f^* - \xi}{\sigma(x)}\right)

    Args:
        mu: Posterior mean ``(M,)``.
        std: Posterior standard deviation ``(M,)``.
        best_f: Best observed function value.
        xi: Exploration margin.

    Returns:
        PI values ``(M,)``.
    """
    std = np.maximum(std, 1e-9)
    z = (mu - best_f - xi) / std
    return norm.cdf(z)


# ---------------------------------------------------------------------------
# BayesianOptimizer
# ---------------------------------------------------------------------------

_ACQUISITION_FNS: dict[str, Callable[..., np.ndarray]] = {
    "EI": expected_improvement,
    "UCB": upper_confidence_bound,
    "PI": probability_of_improvement,
}


class BayesianOptimizer:
    """Gaussian-process Bayesian optimiser with pluggable acquisition.

    Args:
        pool: Fixed candidate pool ``(M, D)``.
        acquisition: Acquisition function name (``"EI"`` | ``"UCB"`` |
            ``"PI"``).
        kappa: UCB exploration weight (only used when ``acquisition="UCB"``).
        xi: EI/PI exploration margin.
        seed: Random seed.
    """

    def __init__(
        self,
        pool: np.ndarray,
        acquisition: Literal["EI", "UCB", "PI"] = "EI",
        kappa: float = 2.0,
        xi: float = 0.01,
        seed: int = 42,
    ) -> None:
        if acquisition not in _ACQUISITION_FNS:
            raise ValueError(
                f"Unknown acquisition {acquisition!r}; choose from {list(_ACQUISITION_FNS)}"
            )
        self.pool = pool
        self.acquisition = acquisition
        self.kappa = kappa
        self.xi = xi
        self.seed = seed

        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._gp = _gp_surrogate(seed)
        self._observed_idx: set[int] = set()

    @property
    def best_f(self) -> float:
        """Best observed function value so far.

        Returns:
            Maximum of all observed values, or ``-inf`` if empty.
        """
        return max(self._y) if self._y else float("-inf")

    @property
    def best_x(self) -> np.ndarray | None:
        """Composition with the best observed value.

        Returns:
            Best composition vector, or ``None`` if no observations.
        """
        if not self._y:
            return None
        return self._X[int(np.argmax(self._y))]

    def update(self, x: np.ndarray, y: float) -> None:
        """Record a new observation.

        Args:
            x: Composition vector ``(D,)``.
            y: Observed property value.
        """
        self._X.append(x.copy())
        self._y.append(y)

    def suggest(self) -> int:
        """Suggest the next candidate index from the pool.

        Fits the GP surrogate to all observations, evaluates the
        acquisition function over un-observed candidates, and returns
        the index of the maximiser.

        Returns:
            Pool index of the suggested candidate.

        Raises:
            RuntimeError: If all pool members have been observed.
        """
        if len(self._X) == 0:
            rng = np.random.default_rng(self.seed)
            idx = rng.integers(0, self.pool.shape[0])
            self._observed_idx.add(int(idx))
            return int(idx)

        X_obs = np.stack(self._X, axis=0)
        y_obs = np.array(self._y)
        self._gp = _gp_surrogate(self.seed)
        self._gp.fit(X_obs, y_obs)

        candidates = np.array(
            sorted(set(range(self.pool.shape[0])) - self._observed_idx)
        )
        if len(candidates) == 0:
            raise RuntimeError("All pool members have been observed.")

        mu, std = self._gp.predict(self.pool[candidates], return_std=True)

        acq_fn = _ACQUISITION_FNS[self.acquisition]
        if self.acquisition == "UCB":
            acq_vals = acq_fn(mu, std, kappa=self.kappa)
        else:
            acq_vals = acq_fn(mu, std, best_f=self.best_f, xi=self.xi)

        best_local = int(np.argmax(acq_vals))
        pool_idx = int(candidates[best_local])
        self._observed_idx.add(pool_idx)
        return pool_idx


# ---------------------------------------------------------------------------
# ActiveLearningLoop
# ---------------------------------------------------------------------------


@dataclass
class ActiveLearningLoopResult:
    """Results from a full :class:`ActiveLearningLoop` run.

    Args:
        best_property: Best observed value.
        best_composition: Best composition vector.
        all_properties: Full observation history.
        n_evals: Total oracle evaluations.
        regret_history: Simple regret at each step.
    """

    best_property: float
    best_composition: np.ndarray
    all_properties: list[float] = field(default_factory=list)
    n_evals: int = 0
    regret_history: list[float] = field(default_factory=list)


class ActiveLearningLoop:
    """High-level active learning driver.

    Wraps a :class:`BayesianOptimizer` with an oracle function, budget
    management, and per-step diagnostics.

    Args:
        pool: Candidate pool ``(M, D)``.
        oracle_fn: Callable ``(x) -> float`` evaluating a single candidate.
        budget: Maximum number of oracle calls.
        init_size: Number of random initial evaluations.
        acquisition: Acquisition function name.
        kappa: UCB exploration weight.
        xi: EI/PI margin.
        seed: Random seed.
    """

    def __init__(
        self,
        pool: np.ndarray,
        oracle_fn: Callable[[np.ndarray], float],
        budget: int = 50,
        init_size: int = 5,
        acquisition: Literal["EI", "UCB", "PI"] = "EI",
        kappa: float = 2.0,
        xi: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.pool = pool
        self.oracle_fn = oracle_fn
        self.budget = budget
        self.init_size = init_size
        self.seed = seed
        self.optimizer = BayesianOptimizer(
            pool=pool,
            acquisition=acquisition,
            kappa=kappa,
            xi=xi,
            seed=seed,
        )

    def run(self, oracle_max: float | None = None) -> ActiveLearningLoopResult:
        """Execute the active learning loop.

        Args:
            oracle_max: True optimum value for regret tracking.  If
                ``None``, regret is not computed.

        Returns:
            :class:`ActiveLearningLoopResult` with full diagnostics.
        """
        rng = np.random.default_rng(self.seed)
        all_props: list[float] = []
        regret_hist: list[float] = []

        init_idx = rng.choice(
            self.pool.shape[0], size=min(self.init_size, self.budget), replace=False
        )
        for idx in init_idx:
            y = self.oracle_fn(self.pool[idx])
            self.optimizer.update(self.pool[idx], y)
            self.optimizer._observed_idx.add(int(idx))
            all_props.append(y)
            if oracle_max is not None:
                regret_hist.append(oracle_max - max(all_props))

        remaining = self.budget - len(init_idx)
        for step in range(remaining):
            idx = self.optimizer.suggest()
            y = self.oracle_fn(self.pool[idx])
            self.optimizer.update(self.pool[idx], y)
            all_props.append(y)
            if oracle_max is not None:
                regret_hist.append(oracle_max - max(all_props))
            if step % 10 == 0:
                logger.info(
                    "Step %d/%d: best_f=%.4f", step, remaining, max(all_props)
                )

        best_i = int(np.argmax(all_props))
        return ActiveLearningLoopResult(
            best_property=all_props[best_i],
            best_composition=self.optimizer._X[best_i].copy(),
            all_properties=all_props,
            n_evals=len(all_props),
            regret_history=regret_hist,
        )
