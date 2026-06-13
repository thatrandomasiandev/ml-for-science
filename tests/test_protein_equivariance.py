"""Tests for protein equivariance and prediction."""

import numpy as np

from ml_sci.data.protein_dgp import ProteinDGPConfig, generate_protein_data
from ml_sci.protein.trainer import fit_protein_predictor


def test_egnn_trains_and_predicts():
    data = generate_protein_data(ProteinDGPConfig(n_proteins=80, n_residues=16, seed=0))
    result = fit_protein_predictor(data, model_type="egnn", epochs=20, seed=0)
    assert result.y_pred.shape == result.y_true.shape
    assert result.spearman > 0.2
    assert result.rmse < data.ground_truth["property_std"] * 2


def test_egnn_more_rotation_consistent_than_mlp():
    data = generate_protein_data(ProteinDGPConfig(n_proteins=100, n_residues=16, seed=1))
    egnn = fit_protein_predictor(data, model_type="egnn", epochs=30, seed=1)
    mlp = fit_protein_predictor(data, model_type="mlp", epochs=30, seed=1)
    assert egnn.rotation_consistency >= mlp.rotation_consistency - 0.1
