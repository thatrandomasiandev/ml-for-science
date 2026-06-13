"""Tests for materials active learning."""

import numpy as np

from ml_sci.data.materials_dgp import MaterialsDGPConfig, generate_materials_data
from ml_sci.materials.active_learning import run_active_learning
from ml_sci.materials.metrics import regret


def test_regret_non_negative():
    data = generate_materials_data(MaterialsDGPConfig(seed=0))
    gt = data.ground_truth
    result = run_active_learning(
        data.compositions,
        20,
        gt["optimum"],
        gt["active_dims"],
        data.metadata["length_scale"],
        method="random",
        seed=0,
    )
    assert regret(result.best_property, gt["oracle_value"]) >= 0.0


def test_uncertainty_beats_random():
    data = generate_materials_data(MaterialsDGPConfig(seed=1))
    gt = data.ground_truth
    budget = 30
    random_r = run_active_learning(
        data.compositions,
        budget,
        gt["optimum"],
        gt["active_dims"],
        data.metadata["length_scale"],
        method="random",
        seed=1,
    )
    unc_r = run_active_learning(
        data.compositions,
        budget,
        gt["optimum"],
        gt["active_dims"],
        data.metadata["length_scale"],
        method="uncertainty",
        seed=1,
    )
    assert unc_r.best_property >= random_r.best_property - 0.05


def test_ei_finds_high_scoring_material():
    data = generate_materials_data(MaterialsDGPConfig(seed=2))
    gt = data.ground_truth
    result = run_active_learning(
        data.compositions,
        40,
        gt["optimum"],
        gt["active_dims"],
        data.metadata["length_scale"],
        method="expected_improvement",
        seed=2,
    )
    assert result.best_property > np.quantile(data.properties, 0.9)
