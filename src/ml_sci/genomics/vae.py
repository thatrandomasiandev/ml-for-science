"""Variational autoencoders for single-cell gene expression.

Provides three VAE variants:

* :class:`ExpressionVAE` — vanilla VAE with MSE reconstruction (legacy).
* :class:`scVAE` — single-cell VAE with BatchNorm encoder and negative
  binomial likelihood for UMI count data, trained with β-KL annealing.
* :class:`LDVAE` — linear-decoder VAE for interpretable latent dimensions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ml_sci.utils.device import get_device
from ml_sci.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Negative-binomial helpers
# ---------------------------------------------------------------------------


def _log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Log probability of the negative binomial parameterised by mean and inverse dispersion.

    .. math::

        p(x \mid \mu, \theta) = \frac{\Gamma(x + \theta)}{\Gamma(\theta)\,x!}
        \left(\frac{\theta}{\theta + \mu}\right)^\theta
        \left(\frac{\mu}{\theta + \mu}\right)^x

    Args:
        x: Observed counts ``(N, G)``.
        mu: Predicted mean ``(N, G)``, positive.
        theta: Inverse dispersion ``(G,)`` or ``(1,)``, positive.
        eps: Numerical stability constant.

    Returns:
        Log-likelihood tensor ``(N, G)``.
    """
    log_theta_mu = torch.log(theta + mu + eps)
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
    )


# ---------------------------------------------------------------------------
# scVAE — negative-binomial VAE for count data
# ---------------------------------------------------------------------------


class scVAE(nn.Module):
    r"""Single-cell VAE with negative binomial reconstruction.

    Encoder maps gene counts to a latent Gaussian via an MLP with
    :class:`~torch.nn.BatchNorm1d`.  The decoder outputs the mean
    :math:`\mu` of a negative binomial distribution whose inverse
    dispersion :math:`\theta` is a learned gene-level parameter.

    The training objective is the β-VAE ELBO:

    .. math::

        \mathcal{L} = -\mathbb{E}_{q(z|x)}[\log p(x|z)]
                      + \beta\,\mathrm{KL}[q(z|x) \| p(z)]

    Args:
        n_genes: Number of input genes (features).
        latent_dim: Dimensionality of the latent space.
        hidden_dim: Width of hidden layers.
        n_hidden: Number of hidden layers in encoder and decoder.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        enc_layers: list[nn.Module] = []
        in_dim = n_genes
        for _ in range(n_hidden):
            enc_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        dec_layers: list[nn.Module] = []
        in_dim = latent_dim
        for _ in range(n_hidden):
            dec_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        dec_layers.append(nn.Linear(hidden_dim, n_genes))
        self.decoder = nn.Sequential(*dec_layers)

        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode counts into latent Gaussian parameters.

        Args:
            x: Gene expression counts ``(N, n_genes)``.

        Returns:
            Tuple ``(mu, log_var)`` each of shape ``(N, latent_dim)``.
        """
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample :math:`z \\sim \\mathcal{N}(\\mu, \\sigma^2)` via reparameterization.

        Args:
            mu: Mean ``(N, latent_dim)``.
            logvar: Log variance ``(N, latent_dim)``.

        Returns:
            Latent sample ``(N, latent_dim)``.
        """
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to NB mean parameter.

        Args:
            z: Latent code ``(N, latent_dim)``.

        Returns:
            Predicted mean :math:`\\mu` ``(N, n_genes)`` (softmax-scaled).
        """
        return F.softmax(self.decoder(z), dim=-1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: Input counts ``(N, n_genes)``.

        Returns:
            ``(mu_x, mu_z, logvar_z)`` where ``mu_x`` is the decoded
            NB mean scaled by library size.
        """
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        lib_size = x.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mu_x = self.decode(z) * lib_size
        return mu_x, mu_z, logvar_z

    def loss(
        self,
        x: torch.Tensor,
        mu_x: torch.Tensor,
        mu_z: torch.Tensor,
        logvar_z: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        r"""β-ELBO loss with negative binomial reconstruction.

        Args:
            x: Input counts ``(N, n_genes)``.
            mu_x: Decoded mean ``(N, n_genes)``.
            mu_z: Encoder mean ``(N, latent_dim)``.
            logvar_z: Encoder log-variance ``(N, latent_dim)``.
            beta: KL divergence weight.

        Returns:
            Scalar loss tensor.
        """
        theta = torch.exp(self.log_theta).clamp(min=1e-4)
        recon_loss = -_log_nb_positive(x, mu_x, theta).sum(dim=-1).mean()
        kl = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=-1).mean()
        return recon_loss + beta * kl


# ---------------------------------------------------------------------------
# LDVAE — linear decoder for interpretability
# ---------------------------------------------------------------------------


class LDVAE(nn.Module):
    r"""Linear-decoder VAE for interpretable latent factors.

    The encoder is a BatchNorm MLP identical to :class:`scVAE`, but the
    decoder is a single linear layer :math:`W z + b` so that each latent
    dimension maps directly to a gene loading vector.

    Args:
        n_genes: Number of input genes.
        latent_dim: Latent dimensionality.
        hidden_dim: Encoder hidden width.
        n_hidden: Number of encoder hidden layers.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 10,
        hidden_dim: int = 128,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        enc_layers: list[nn.Module] = []
        in_dim = n_genes
        for _ in range(n_hidden):
            enc_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Linear(latent_dim, n_genes)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent Gaussian parameters.

        Args:
            x: Gene counts ``(N, n_genes)``.

        Returns:
            ``(mu, log_var)`` each ``(N, latent_dim)``.
        """
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick.

        Args:
            mu: Mean ``(N, latent_dim)``.
            logvar: Log-variance ``(N, latent_dim)``.

        Returns:
            Sampled latent ``(N, latent_dim)``.
        """
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Linear decode.

        Args:
            z: Latent code ``(N, latent_dim)``.

        Returns:
            Reconstructed expression ``(N, n_genes)``.
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: Input ``(N, n_genes)``.

        Returns:
            ``(x_hat, mu, logvar)``.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        r"""β-VAE MSE loss.

        Args:
            x: Input ``(N, n_genes)``.
            x_hat: Reconstruction ``(N, n_genes)``.
            mu: Encoder mean.
            logvar: Encoder log-variance.
            beta: KL weight.

        Returns:
            Scalar loss.
        """
        recon = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return recon + beta * kl

    @property
    def loadings(self) -> np.ndarray:
        """Gene loading matrix ``(latent_dim, n_genes)``.

        Returns:
            Weight matrix of the linear decoder as a NumPy array.
        """
        return self.decoder.weight.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Legacy ExpressionVAE
# ---------------------------------------------------------------------------


class ExpressionVAE(nn.Module):
    """Simple VAE for gene expression with MSE reconstruction.

    Args:
        n_genes: Number of input genes.
        latent_dim: Latent dimensionality.
        hidden_dim: Hidden layer width.
    """

    def __init__(
        self, n_genes: int, latent_dim: int = 16, hidden_dim: int = 64
    ) -> None:
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
        """Encode input to latent parameters.

        Args:
            x: Input ``(N, n_genes)``.

        Returns:
            ``(mu, logvar)`` each ``(N, latent_dim)``.
        """
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick.

        Args:
            mu: Mean ``(N, latent_dim)``.
            logvar: Log-variance ``(N, latent_dim)``.

        Returns:
            Sampled latent ``(N, latent_dim)``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to gene space.

        Args:
            z: Latent ``(N, latent_dim)``.

        Returns:
            Reconstruction ``(N, n_genes)``.
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: Input ``(N, n_genes)``.

        Returns:
            ``(reconstruction, mu, logvar)``.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class VAEResult:
    """VAE training outcome.

    Args:
        latent: Latent embeddings ``(N, latent_dim)``.
        reconstructed: Reconstructed expressions ``(N, n_genes)``.
        train_loss: Final epoch training loss.
    """

    latent: np.ndarray
    reconstructed: np.ndarray
    train_loss: float


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
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
    """Train the legacy :class:`ExpressionVAE` on an expression matrix.

    Args:
        expression: Expression matrix ``(N, G)`` as NumPy array.
        latent_dim: Latent space dimensionality.
        hidden_dim: Hidden layer width.
        epochs: Number of training epochs.
        lr: Learning rate.
        seed: Random seed.
        device: Compute device string.

    Returns:
        :class:`VAEResult` with latent embeddings and reconstruction.
    """
    set_torch_seed(seed)
    dev = get_device(device)
    x = torch.tensor(expression, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=128, shuffle=True)

    model = ExpressionVAE(
        expression.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim
    ).to(dev)
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

    return VAEResult(
        latent=latent, reconstructed=reconstructed, train_loss=float(train_loss)
    )


def train_scvae(
    expression: np.ndarray,
    latent_dim: int = 16,
    hidden_dim: int = 128,
    n_hidden: int = 2,
    epochs: int = 100,
    lr: float = 1e-3,
    beta: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
) -> VAEResult:
    """Train :class:`scVAE` on a count expression matrix.

    Args:
        expression: Count matrix ``(N, G)``.
        latent_dim: Latent dimensionality.
        hidden_dim: Hidden width.
        n_hidden: Number of encoder/decoder hidden layers.
        epochs: Training epochs.
        lr: Learning rate.
        beta: KL weight for β-VAE.
        seed: Random seed.
        device: Compute device.

    Returns:
        :class:`VAEResult` with latent embeddings and reconstructed means.
    """
    set_torch_seed(seed)
    dev = get_device(device)
    x = torch.tensor(expression, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=128, shuffle=True)

    model = scVAE(
        n_genes=expression.shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(dev)
            optimizer.zero_grad()
            mu_x, mu_z, logvar_z = model(xb)
            loss = model.loss(xb, mu_x, mu_z, logvar_z, beta=beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / max(len(loader), 1)

    model.eval()
    with torch.no_grad():
        mu_x, mu_z, _ = model(x.to(dev))
        latent = mu_z.cpu().numpy()
        reconstructed = mu_x.cpu().numpy()

    return VAEResult(
        latent=latent, reconstructed=reconstructed, train_loss=float(train_loss)
    )
