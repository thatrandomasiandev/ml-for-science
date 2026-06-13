"""E(n)-equivariant graph neural network for protein property prediction."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances for (B, N, 3) coordinates."""
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    return torch.sqrt((diff**2).sum(-1) + 1e-8)


class EGNNLayer(nn.Module):
    """Simplified EGNN message-passing layer with invariant edge features."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dists = _pairwise_distances(coords)
        n = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        edge_in = torch.cat([h_i, h_j, dists.unsqueeze(-1)], dim=-1)
        edge_msg = self.edge_mlp(edge_in)

        agg = edge_msg.mean(dim=2)
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1)) + h

        rel = coords.unsqueeze(2) - coords.unsqueeze(1)
        rel = rel / (torch.norm(rel, dim=-1, keepdim=True) + 1e-8)
        coord_weights = self.coord_mlp(edge_msg).squeeze(-1)
        coord_delta = (coord_weights.unsqueeze(-1) * rel).mean(dim=2)
        coords_new = coords + 0.01 * coord_delta

        return h_new, coords_new


class EGNNPropertyPredictor(nn.Module):
    """EGNN encoder with invariant global pooling readout."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_embed = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Predict property from (B, N, 3) coordinates."""
        dists = _pairwise_distances(coords)
        h = self.node_embed(dists.mean(dim=-1, keepdim=True))
        x = coords
        for layer in self.layers:
            h, x = layer(h, x)
            h = self.dropout(h)
        pooled = h.mean(dim=1)
        return self.readout(pooled).squeeze(-1)


class MLPPropertyPredictor(nn.Module):
    """Non-equivariant baseline on flattened coordinates."""

    def __init__(
        self,
        n_residues: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = n_residues * 3
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        flat = coords.reshape(coords.shape[0], -1)
        return self.net(flat).squeeze(-1)


def build_protein_model(
    model_type: Literal["egnn", "mlp"],
    n_residues: int,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory for protein property predictors."""
    if model_type == "egnn":
        return EGNNPropertyPredictor(hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
    return MLPPropertyPredictor(
        n_residues=n_residues,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )


def rotate_coords(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random 3D rotation to protein coordinates."""
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return coords @ q.T
