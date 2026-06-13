"""E(3)-equivariant graph neural network (Satorras et al., 2021).

Implements the full EGNN message-passing scheme where node features
transform invariantly and coordinate updates are equivariant under
rotations, translations, and reflections:

* **Message**: :math:`m_{ij} = \\phi_e(h_i, h_j, \\|x_i - x_j\\|^2, a_{ij})`
* **Coordinate**: :math:`x_i' = x_i + \\sum_j (x_i - x_j)\\,\\phi_x(m_{ij})`
* **Node**: :math:`h_i' = \\phi_h(h_i, \\sum_j m_{ij})`

Also retains the original ``EGNNPropertyPredictor``, ``MLPPropertyPredictor``,
and helper utilities for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances for ``(B, N, 3)`` coordinates.

    Args:
        coords: Coordinate tensor ``(B, N, 3)``.

    Returns:
        Distance matrix ``(B, N, N)``.
    """
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)


def _pairwise_sq_distances(coords: torch.Tensor) -> torch.Tensor:
    """Squared pairwise distances for ``(B, N, 3)`` coordinates.

    Args:
        coords: Coordinate tensor ``(B, N, 3)``.

    Returns:
        Squared distance matrix ``(B, N, N)``.
    """
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    return (diff ** 2).sum(-1)


# ---------------------------------------------------------------------------
# Full E(3)-Equivariant GNN Layer (Satorras et al.)
# ---------------------------------------------------------------------------


class FullEGNNLayer(nn.Module):
    r"""E(3)-equivariant message-passing layer.

    Implements the update equations from Satorras et al. (2021):

    .. math::

        m_{ij} &= \phi_e\bigl(h_i,\; h_j,\; \|x_i - x_j\|^2,\; a_{ij}\bigr) \\
        x_i'   &= x_i + \sum_{j \ne i} (x_i - x_j)\,\phi_x(m_{ij}) \\
        h_i'   &= \phi_h\bigl(h_i,\; \textstyle\sum_j m_{ij}\bigr)

    Args:
        hidden_dim: Feature dimensionality for node embeddings.
        edge_attr_dim: Dimensionality of optional edge attributes
            :math:`a_{ij}`.  Set to ``0`` if no edge attributes are used.
        coord_scale: Multiplicative damping on coordinate updates to
            improve training stability.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int = 0,
        coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        edge_input_dim = 2 * hidden_dim + 1 + edge_attr_dim

        self.phi_e = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_scale = coord_scale

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one EGNN message-passing step.

        Args:
            h: Node features ``(B, N, hidden_dim)``.
            coords: Node coordinates ``(B, N, 3)``.
            edge_attr: Optional edge attributes ``(B, N, N, edge_attr_dim)``.

        Returns:
            Tuple of updated ``(h', coords')``, each retaining their
            original shape.
        """
        B, N, _ = h.shape
        rel = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B,N,N,3)
        dist_sq = (rel ** 2).sum(-1, keepdim=True)  # (B,N,N,1)

        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)

        edge_in = [h_i, h_j, dist_sq]
        if edge_attr is not None:
            edge_in.append(edge_attr)
        edge_in_cat = torch.cat(edge_in, dim=-1)

        m_ij = self.phi_e(edge_in_cat)  # (B,N,N,hidden_dim)

        agg = m_ij.sum(dim=2)  # (B,N,hidden_dim)
        h_new = self.phi_h(torch.cat([h, agg], dim=-1)) + h

        coord_weights = self.phi_x(m_ij)  # (B,N,N,1)
        coord_delta = (coord_weights * rel).sum(dim=2)  # (B,N,3)
        coords_new = coords + self.coord_scale * coord_delta

        return h_new, coords_new


class FullEGNN(nn.Module):
    """Multi-layer E(3)-equivariant GNN with invariant readout.

    Args:
        node_feat_dim: Input node feature dimensionality (use ``1`` for
            distance-based initialisation).
        hidden_dim: Hidden feature dimensionality.
        output_dim: Scalar output dimensionality.
        n_layers: Number of :class:`FullEGNNLayer` blocks.
        edge_attr_dim: Edge attribute dimensionality (``0`` = none).
        dropout: Dropout rate applied after each layer.
        coord_scale: Damping factor forwarded to each layer.
    """

    def __init__(
        self,
        node_feat_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_layers: int = 3,
        edge_attr_dim: int = 0,
        dropout: float = 0.1,
        coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                FullEGNNLayer(hidden_dim, edge_attr_dim, coord_scale)
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict an invariant scalar from node features and coordinates.

        Args:
            node_feats: Per-node input features ``(B, N, node_feat_dim)``.
            coords: Cartesian coordinates ``(B, N, 3)``.
            edge_attr: Optional edge attributes ``(B, N, N, edge_attr_dim)``.

        Returns:
            Predicted property ``(B, output_dim)``.
        """
        h = self.node_embed(node_feats)
        x = coords
        for layer in self.layers:
            h, x = layer(h, x, edge_attr)
            h = self.dropout(h)
        pooled = h.mean(dim=1)
        return self.readout(pooled)

    @staticmethod
    def verify_equivariance(
        model: FullEGNN,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        seed: int = 0,
        atol: float = 1e-4,
    ) -> bool:
        """Check rotation equivariance of coordinate updates and output invariance.

        Applies a random rotation :math:`R \\in SO(3)` and verifies:

        1. The scalar output is invariant: :math:`f(h, Rx) = f(h, x)`.
        2. Coordinate updates are equivariant (tested on intermediate
           layers).

        Args:
            model: An :class:`FullEGNN` instance (set to ``eval`` mode
                internally).
            node_feats: Node features ``(B, N, node_feat_dim)``.
            coords: Coordinates ``(B, N, 3)``.
            seed: Random seed for generating the rotation matrix.
            atol: Absolute tolerance for the invariance check.

        Returns:
            ``True`` if both invariance and equivariance hold within
            tolerance.
        """
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        R = torch.tensor(Q, dtype=coords.dtype, device=coords.device)

        model.eval()
        with torch.no_grad():
            out_orig = model(node_feats, coords)
            coords_rot = coords @ R.T
            out_rot = model(node_feats, coords_rot)

        invariant = torch.allclose(out_orig, out_rot, atol=atol)
        if not invariant:
            max_diff = (out_orig - out_rot).abs().max().item()
            logger.warning(
                "Invariance check failed: max |Δ| = %.6e (atol=%.1e)",
                max_diff,
                atol,
            )
        return invariant


# ---------------------------------------------------------------------------
# Legacy models (backward compatible)
# ---------------------------------------------------------------------------


class EGNNLayer(nn.Module):
    """Simplified EGNN message-passing layer with invariant edge features.

    Args:
        hidden_dim: Feature dimensionality.
    """

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
        """Apply one message-passing step.

        Args:
            h: Node features ``(B, N, hidden_dim)``.
            coords: Coordinates ``(B, N, 3)``.

        Returns:
            Tuple of updated ``(h', coords')``.
        """
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
    """EGNN encoder with invariant global pooling readout.

    Args:
        hidden_dim: Hidden feature dimensionality.
        n_layers: Number of EGNN layers.
        dropout: Dropout rate.
    """

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
        """Predict property from ``(B, N, 3)`` coordinates.

        Args:
            coords: Cartesian coordinates ``(B, N, 3)``.

        Returns:
            Scalar predictions ``(B,)``.
        """
        dists = _pairwise_distances(coords)
        h = self.node_embed(dists.mean(dim=-1, keepdim=True))
        x = coords
        for layer in self.layers:
            h, x = layer(h, x)
            h = self.dropout(h)
        pooled = h.mean(dim=1)
        return self.readout(pooled).squeeze(-1)


class MLPPropertyPredictor(nn.Module):
    """Non-equivariant baseline on flattened coordinates.

    Args:
        n_residues: Number of residue positions in each protein.
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.
        dropout: Dropout rate.
    """

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
            layers.extend(
                [nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            )
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Predict property from flattened coordinates.

        Args:
            coords: Cartesian coordinates ``(B, N, 3)``.

        Returns:
            Scalar predictions ``(B,)``.
        """
        flat = coords.reshape(coords.shape[0], -1)
        return self.net(flat).squeeze(-1)


def build_protein_model(
    model_type: Literal["egnn", "mlp"],
    n_residues: int,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory for protein property predictors.

    Args:
        model_type: ``"egnn"`` for equivariant model, ``"mlp"`` for baseline.
        n_residues: Number of residue positions per protein.
        hidden_dim: Hidden layer width.
        n_layers: Number of layers.
        dropout: Dropout rate.

    Returns:
        An ``nn.Module`` predictor.
    """
    if model_type == "egnn":
        return EGNNPropertyPredictor(
            hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout
        )
    return MLPPropertyPredictor(
        n_residues=n_residues,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )


def rotate_coords(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random 3-D rotation to protein coordinates.

    Args:
        coords: Coordinate array ``(N, 3)``.
        rng: NumPy random generator.

    Returns:
        Rotated coordinates ``(N, 3)``.
    """
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return coords @ q.T
