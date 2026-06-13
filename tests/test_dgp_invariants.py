"""Tests for synthetic DGP invariants."""

import numpy as np

from ml_sci.data.climate_dgp import ClimateDGPConfig, generate_climate_data
from ml_sci.data.genomics_dgp import GenomicsDGPConfig, generate_genomics_data
from ml_sci.data.materials_dgp import MaterialsDGPConfig, generate_materials_data, property_oracle
from ml_sci.data.protein_dgp import ProteinDGPConfig, generate_protein_data, stability_oracle
from ml_sci.models.egnn import rotate_coords


def test_protein_shapes():
    data = generate_protein_data(ProteinDGPConfig(n_proteins=50, n_residues=16, seed=0))
    assert data.coords.shape == (50, 16, 3)
    assert data.properties.shape == (50,)


def test_protein_stability_rotation_invariant():
    data = generate_protein_data(ProteinDGPConfig(n_proteins=10, seed=1))
    rng = np.random.default_rng(0)
    gt = data.ground_truth
    for i in range(3):
        rotated = rotate_coords(data.coords[i], rng)
        p_orig = stability_oracle(data.coords[i], gt["template"], gt["w_rg"], gt["w_contact"])
        p_rot = stability_oracle(rotated, gt["template"], gt["w_rg"], gt["w_contact"])
        assert abs(p_orig - p_rot) < 1e-6


def test_climate_coarse_fine_consistency():
    data = generate_climate_data(ClimateDGPConfig(downscale_factor=4, seed=2))
    factor = data.metadata["downscale_factor"]
    pooled = data.fine.reshape(16, factor, 16, factor).mean(axis=(1, 3))
    np.testing.assert_allclose(pooled, data.coarse, rtol=0.15)


def test_genomics_batch_structure():
    data = generate_genomics_data(GenomicsDGPConfig(n_batches=2, seed=3))
    assert data.expression.shape[1] == data.n_genes
    assert len(np.unique(data.batch_labels)) == 2
    assert len(np.unique(data.cell_types)) == data.metadata["n_cell_types"]


def test_materials_oracle_optimum_is_peak():
    data = generate_materials_data(MaterialsDGPConfig(n_pool=500, seed=4))
    gt = data.ground_truth
    oracle = property_oracle(
        gt["optimum"][None],
        gt["optimum"],
        gt["active_dims"],
        data.metadata["length_scale"],
    )[0]
    assert oracle >= np.max(data.properties) - 1e-6
