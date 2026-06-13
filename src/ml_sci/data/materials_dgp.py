"""Synthetic materials property landscape for active learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_sci.data.base import MaterialsDataset
from ml_sci.utils.seed import set_seed


@dataclass
class MaterialsDGPConfig:
    """Configuration for materials discovery active-learning benchmark."""

    n_pool: int = 2000
    composition_dim: int = 6
    n_active_dims: int = 3
    length_scale: float = 0.5
    seed: int = 42


def property_oracle(
    compositions: np.ndarray,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    length_scale: float,
) -> np.ndarray:
    """Smooth GP-like materials property with known global optimum."""
    z = compositions[:, active_dims]
    z_star = optimum[active_dims]
    sq_dist = np.sum((z - z_star) ** 2, axis=1)
    return np.exp(-sq_dist / (2.0 * length_scale**2)).astype(np.float64)


def generate_materials_data(config: MaterialsDGPConfig) -> MaterialsDataset:
    """Generate composition pool with oracle property scores."""
    rng = set_seed(config.seed)
    compositions = rng.uniform(-1.0, 1.0, size=(config.n_pool, config.composition_dim))
    active_dims = np.sort(
        rng.choice(config.composition_dim, size=config.n_active_dims, replace=False)
    )
    optimum = np.zeros(config.composition_dim, dtype=np.float64)
    optimum[active_dims] = rng.uniform(-0.5, 0.5, size=config.n_active_dims)
    properties = property_oracle(compositions, optimum, active_dims, config.length_scale)

    return MaterialsDataset(
        compositions=compositions.astype(np.float64),
        properties=properties.astype(np.float64),
        metadata={
            "dgp": "gp_materials_landscape",
            "composition_dim": config.composition_dim,
            "n_active_dims": config.n_active_dims,
            "length_scale": config.length_scale,
            "seed": config.seed,
        },
        ground_truth={
            "optimum": optimum,
            "active_dims": active_dims,
            "oracle_value": float(
                property_oracle(optimum[None], optimum, active_dims, config.length_scale)[0]
            ),
            "top_1_percent_threshold": float(np.quantile(properties, 0.99)),
        },
    )
