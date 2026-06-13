"""Climate downscaling with PINNs and classical baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ml_sci.climate.metrics import physics_residual_rms, rmse, spectral_bias
from ml_sci.data.base import ClimateDataset
from ml_sci.models.pinn import BicubicDownscaler, ClimatePINN
from ml_sci.utils.device import get_device
from ml_sci.utils.seed import set_torch_seed


@dataclass
class DownscaleResult:
    """Downscaling outcome with reconstruction metrics."""

    prediction: np.ndarray
    rmse: float
    spectral_bias: float
    physics_residual: float
    train_loss: float


def _grid_coords(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return xx, yy


def fit_bicubic_downscaler(data: ClimateDataset) -> DownscaleResult:
    """Classical bicubic upsampling baseline."""
    factor = data.metadata["downscale_factor"]
    scaler = BicubicDownscaler(factor)
    pred = scaler.predict(data.coarse)
    dx = data.ground_truth["dx_fine"]
    lap = (
        -4.0 * pred
        + np.roll(pred, 1, axis=0)
        + np.roll(pred, -1, axis=0)
        + np.roll(pred, 1, axis=1)
        + np.roll(pred, -1, axis=1)
    ) / (dx**2)
    return DownscaleResult(
        prediction=pred,
        rmse=rmse(data.fine, pred),
        spectral_bias=spectral_bias(data.fine, pred),
        physics_residual=physics_residual_rms(lap),
        train_loss=0.0,
    )


def fit_pinn_downscaler(
    data: ClimateDataset,
    epochs: int = 150,
    lr: float = 0.005,
    hidden_dim: int = 64,
    physics_weight: float = 0.0,
    seed: int = 42,
    device: str = "cpu",
) -> DownscaleResult:
    """Train physics-regularized residual PINN on top of bicubic upsampling."""
    set_torch_seed(seed)
    dev = get_device(device)
    n_fine = data.fine.shape[0]
    factor = data.metadata["downscale_factor"]
    dx = data.ground_truth["dx_fine"]

    xx, yy = _grid_coords(n_fine)
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    fine_flat = data.fine.ravel().astype(np.float32)
    bicubic = BicubicDownscaler(factor).predict(data.coarse)
    base_flat = bicubic.ravel().astype(np.float32)
    residual = fine_flat - base_flat

    model = ClimatePINN(hidden_dim=hidden_dim, n_layers=5).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    xy_t = torch.tensor(xy, device=dev)
    res_t = torch.tensor(residual, device=dev)
    base_t = torch.tensor(base_flat, device=dev)

    train_loss = 0.0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        res_pred = model(xy_t)
        full_pred = base_t + res_pred
        data_loss = mse(res_pred, res_t)
        lap = model.laplacian_residual(xy_t, full_pred, dx)
        phys_loss = torch.mean(lap**2)
        loss = data_loss + physics_weight * phys_loss
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        pred = (base_t + model(xy_t)).cpu().numpy().reshape(n_fine, n_fine)

    lap_np = (
        -4.0 * pred
        + np.roll(pred, 1, axis=0)
        + np.roll(pred, -1, axis=0)
        + np.roll(pred, 1, axis=1)
        + np.roll(pred, -1, axis=1)
    ) / (dx**2)

    return DownscaleResult(
        prediction=pred,
        rmse=rmse(data.fine, pred),
        spectral_bias=spectral_bias(data.fine, pred),
        physics_residual=physics_residual_rms(lap_np),
        train_loss=train_loss,
    )
