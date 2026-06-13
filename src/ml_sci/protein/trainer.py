"""Protein property prediction training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml_sci.data.base import ProteinDataset
from ml_sci.models.egnn import build_protein_model, rotate_coords
from ml_sci.protein.metrics import rotation_consistency, rmse, spearman_correlation
from ml_sci.utils.device import get_device
from ml_sci.utils.seed import set_torch_seed


@dataclass
class ProteinTrainResult:
    """Training outcome with test metrics."""

    y_pred: np.ndarray
    y_true: np.ndarray
    train_loss: float
    rmse: float
    spearman: float
    rotation_consistency: float


def _split_indices(n: int, seed: int, train_frac: float = 0.7, val_frac: float = 0.15):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def fit_protein_predictor(
    data: ProteinDataset,
    model_type: Literal["egnn", "mlp"] = "egnn",
    epochs: int = 80,
    lr: float = 0.001,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
) -> ProteinTrainResult:
    """Train protein property predictor and evaluate on held-out set."""
    set_torch_seed(seed)
    dev = get_device(device)
    train_idx, val_idx, test_idx = _split_indices(data.n_proteins, seed)

    coords = torch.tensor(data.coords, dtype=torch.float32)
    props = torch.tensor(data.properties, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(coords[train_idx], props[train_idx]),
        batch_size=min(32, len(train_idx)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(coords[val_idx], props[val_idx]),
        batch_size=min(32, len(val_idx)),
    )

    model = build_protein_model(
        model_type,
        n_residues=data.n_residues,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    train_loss = 0.0

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                val_loss += loss_fn(model(xb), yb).item()
        val_loss /= max(len(val_loader), 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_coords = coords[test_idx].to(dev)
    with torch.no_grad():
        y_pred = model(test_coords).cpu().numpy()
    y_true = props[test_idx].numpy()

    rng = np.random.default_rng(seed)
    rot_coords = np.stack(
        [rotate_coords(data.coords[i], rng) for i in test_idx],
        axis=0,
    )
    with torch.no_grad():
        y_pred_rot = model(torch.tensor(rot_coords, dtype=torch.float32).to(dev)).cpu().numpy()

    return ProteinTrainResult(
        y_pred=y_pred,
        y_true=y_true,
        train_loss=float(train_loss),
        rmse=rmse(y_true, y_pred),
        spearman=spearman_correlation(y_true, y_pred),
        rotation_consistency=rotation_consistency(y_pred, y_pred_rot),
    )
