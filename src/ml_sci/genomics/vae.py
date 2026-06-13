"""Variational autoencoder for single-cell expression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml_sci.utils.device import get_device
from ml_sci.utils.seed import set_torch_seed


class ExpressionVAE(nn.Module):
    """Simple VAE for gene expression."""

    def __init__(self, n_genes: int, latent_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


@dataclass
class VAEResult:
    """VAE training outcome."""

    latent: np.ndarray
    reconstructed: np.ndarray
    train_loss: float


def _vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.01 * kl


def train_vae(
    expression: np.ndarray,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    epochs: int = 80,
    lr: float = 0.001,
    seed: int = 42,
    device: str = "cpu",
) -> VAEResult:
    """Train VAE on expression matrix."""
    set_torch_seed(seed)
    dev = get_device(device)
    x = torch.tensor(expression, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=128, shuffle=True)

    model = ExpressionVAE(expression.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = 0.0
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(dev)
            optimizer.zero_grad()
            recon, mu, logvar = model(xb)
            loss = _vae_loss(recon, xb, mu, logvar) / xb.shape[0]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / max(len(loader), 1)

    model.eval()
    with torch.no_grad():
        recon, mu, _ = model(x.to(dev))
        latent = mu.cpu().numpy()
        reconstructed = recon.cpu().numpy()

    return VAEResult(latent=latent, reconstructed=reconstructed, train_loss=float(train_loss))
