"""Physics-informed neural network for climate downscaling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ClimatePINN(nn.Module):
    """MLP mapping (x, y) -> temperature with Laplacian physics penalty."""

    def __init__(self, hidden_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy).squeeze(-1)

    def laplacian_residual(self, xy: torch.Tensor, field: torch.Tensor, dx: float) -> torch.Tensor:
        """Compute discrete Laplacian residual on a regular grid."""
        n = int(np.sqrt(xy.shape[0]))
        u = field.reshape(n, n)
        lap = (
            -4.0 * u
            + torch.roll(u, 1, dims=0)
            + torch.roll(u, -1, dims=0)
            + torch.roll(u, 1, dims=1)
            + torch.roll(u, -1, dims=1)
        ) / (dx**2)
        return lap.reshape(-1)


class BicubicDownscaler:
    """Classical baseline: bicubic upsampling of coarse grid."""

    def __init__(self, factor: int) -> None:
        self.factor = factor

    def predict(self, coarse: np.ndarray) -> np.ndarray:
        from scipy.ndimage import zoom

        return zoom(coarse, self.factor, order=3)
