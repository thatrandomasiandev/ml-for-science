"""Synthetic climate downscaling from a known Poisson field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_sci.data.base import ClimateDataset
from ml_sci.utils.seed import set_seed


@dataclass
class ClimateDGPConfig:
    """Configuration for climate downscaling benchmark."""

    fine_size: int = 64
    downscale_factor: int = 4
    n_modes: int = 6
    noise_std: float = 0.05
    seed: int = 42


def _poisson_field(
    x: np.ndarray,
    y: np.ndarray,
    modes: list[tuple[float, float, float]],
) -> np.ndarray:
    """Steady-state temperature field as sum of Fourier modes."""
    field = np.zeros_like(x, dtype=np.float64)
    for kx, ky, amp in modes:
        field += amp * np.sin(kx * x) * np.sin(ky * y)
    return field


def _laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """5-point Laplacian on a 2D grid."""
    lap = (
        -4.0 * field
        + np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
    ) / (dx**2)
    return lap


def generate_climate_data(config: ClimateDGPConfig) -> ClimateDataset:
    """Generate coarse/fine temperature grids from a known spectral field."""
    rng = set_seed(config.seed)
    n_fine = config.fine_size
    factor = config.downscale_factor
    n_coarse = n_fine // factor

    x = np.linspace(0.0, 2.0 * np.pi, n_fine, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, n_fine, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    modes = [
        (float(rng.integers(1, 4)), float(rng.integers(1, 4)), float(rng.uniform(0.5, 1.5)))
        for _ in range(config.n_modes)
    ]
    fine = _poisson_field(xx, yy, modes)
    fine += rng.normal(0.0, config.noise_std, size=fine.shape)

    coarse = fine.reshape(n_coarse, factor, n_coarse, factor).mean(axis=(1, 3))
    dx_fine = 2.0 * np.pi / n_fine
    pde_residual = _laplacian(fine, dx_fine)

    return ClimateDataset(
        coarse=coarse.astype(np.float64),
        fine=fine.astype(np.float64),
        x_fine=xx.astype(np.float64),
        y_fine=yy.astype(np.float64),
        metadata={
            "dgp": "spectral_poisson_field",
            "fine_size": n_fine,
            "downscale_factor": factor,
            "noise_std": config.noise_std,
            "seed": config.seed,
        },
        ground_truth={
            "modes": modes,
            "dx_fine": dx_fine,
            "pde_residual_rms": float(np.sqrt(np.mean(pde_residual**2))),
            "fine_mean": float(fine.mean()),
            "fine_std": float(fine.std()),
        },
    )
