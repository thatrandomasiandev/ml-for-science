"""Tests for climate PINN downscaling."""

from ml_sci.climate.downscale import fit_bicubic_downscaler, fit_pinn_downscaler
from ml_sci.data.climate_dgp import ClimateDGPConfig, generate_climate_data


def test_bicubic_produces_finite_output():
    data = generate_climate_data(ClimateDGPConfig(downscale_factor=4, seed=0))
    result = fit_bicubic_downscaler(data)
    assert result.prediction.shape == data.fine.shape
    assert result.rmse < 2.0


def test_pinn_beats_bicubic_on_rmse():
    data = generate_climate_data(ClimateDGPConfig(downscale_factor=4, noise_std=0.05, seed=1))
    bicubic = fit_bicubic_downscaler(data)
    pinn = fit_pinn_downscaler(data, epochs=150, seed=1)
    assert pinn.rmse < bicubic.rmse
