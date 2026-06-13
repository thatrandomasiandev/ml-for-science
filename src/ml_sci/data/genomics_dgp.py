"""Synthetic single-cell expression with batch effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_sci.data.base import GenomicsDataset
from ml_sci.utils.seed import set_seed


@dataclass
class GenomicsDGPConfig:
    """Configuration for scRNA-seq batch correction benchmark."""

    n_cells_per_type: int = 200
    n_cell_types: int = 4
    n_genes: int = 50
    n_batches: int = 2
    batch_shift_scale: float = 2.0
    noise_std: float = 0.3
    seed: int = 42


def generate_genomics_data(config: GenomicsDGPConfig) -> GenomicsDataset:
    """Generate expression with shared biology and batch-specific shifts."""
    rng = set_seed(config.seed)
    n_cells = config.n_cells_per_type * config.n_cell_types
    n_genes = config.n_genes

    type_profiles = rng.standard_normal((config.n_cell_types, n_genes))
    type_profiles /= np.linalg.norm(type_profiles, axis=1, keepdims=True)

    cell_types = np.repeat(np.arange(config.n_cell_types), config.n_cells_per_type)
    batch_labels = rng.integers(0, config.n_batches, size=n_cells)

    expression = type_profiles[cell_types].copy()
    batch_shifts = rng.standard_normal((config.n_batches, n_genes)) * config.batch_shift_scale
    expression += batch_shifts[batch_labels]
    expression += rng.normal(0.0, config.noise_std, size=expression.shape)

    return GenomicsDataset(
        expression=expression.astype(np.float64),
        batch_labels=batch_labels.astype(np.int64),
        cell_types=cell_types.astype(np.int64),
        metadata={
            "dgp": "scrna_batch_effect",
            "n_cell_types": config.n_cell_types,
            "n_batches": config.n_batches,
            "batch_shift_scale": config.batch_shift_scale,
            "noise_std": config.noise_std,
            "seed": config.seed,
        },
        ground_truth={
            "type_profiles": type_profiles,
            "batch_shifts": batch_shifts,
            "type_centroids": type_profiles.copy(),
        },
    )
