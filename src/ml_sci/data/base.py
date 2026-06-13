"""Dataset protocols for scientific ML benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ProteinDataset:
    """3D residue coordinates with SE(3)-invariant property labels."""

    coords: np.ndarray
    properties: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_proteins(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_residues(self) -> int:
        return int(self.coords.shape[1])


@dataclass
class ClimateDataset:
    """Coarse/fine gridded fields from a known PDE solution."""

    coarse: np.ndarray
    fine: np.ndarray
    x_fine: np.ndarray
    y_fine: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def fine_shape(self) -> tuple[int, int]:
        return int(self.fine.shape[0]), int(self.fine.shape[1])


@dataclass
class GenomicsDataset:
    """Single-cell expression with batch labels and cell-type structure."""

    expression: np.ndarray
    batch_labels: np.ndarray
    cell_types: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.expression.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.expression.shape[1])


@dataclass
class MaterialsDataset:
    """Material compositions with expensive property oracle."""

    compositions: np.ndarray
    properties: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.compositions.shape[0])

    @property
    def composition_dim(self) -> int:
        return int(self.compositions.shape[1])
