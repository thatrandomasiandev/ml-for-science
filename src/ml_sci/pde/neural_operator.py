"""Fourier Neural Operator for learning PDE solution operators.

Implements the 1-D FNO architecture of Li et al. (2021) where each
layer applies:

.. math::

    v_{l+1}(x) = \\sigma\\bigl(W_l\\,v_l(x)
                 + \\mathcal{F}^{-1}\\bigl(R_l \\cdot \\mathcal{F}(v_l)\\bigr)(x)
                 + b_l\\bigr)

The spectral convolution kernel :math:`R_l` is truncated to the first
``k_max`` Fourier modes, making it resolution-invariant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_sci.utils.device import get_device
from ml_sci.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spectral convolution layer
# ---------------------------------------------------------------------------


class SpectralConv1d(nn.Module):
    r"""Fourier-space convolution truncated to ``k_max`` modes.

    Multiplies the first ``k_max`` Fourier coefficients of the input
    by a learned complex weight matrix :math:`R \in \mathbb{C}^{
    d_{\text{out}} \times d_{\text{in}} \times k_{\max}}`.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        k_max: Number of Fourier modes to retain.
    """

    def __init__(self, in_channels: int, out_channels: int, k_max: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_max = k_max

        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, k_max, 2)
        )

    def _complex_mul(
        self, inp: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Batched complex multiplication in (real, imag) representation.

        Args:
            inp: ``(B, C_in, k_max, 2)``.
            weight: ``(C_in, C_out, k_max, 2)``.

        Returns:
            ``(B, C_out, k_max, 2)``.
        """
        return torch.stack(
            [
                torch.einsum("bik,iok->bok", inp[..., 0], weight[..., 0])
                - torch.einsum("bik,iok->bok", inp[..., 1], weight[..., 1]),
                torch.einsum("bik,iok->bok", inp[..., 0], weight[..., 1])
                + torch.einsum("bik,iok->bok", inp[..., 1], weight[..., 0]),
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Args:
            x: Input tensor ``(B, C_in, N)`` where ``N`` is the
                spatial resolution.

        Returns:
            Output tensor ``(B, C_out, N)``.
        """
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)

        x_ft_trunc = torch.stack(
            [x_ft[..., : self.k_max].real, x_ft[..., : self.k_max].imag], dim=-1
        )

        out_ft = self._complex_mul(x_ft_trunc, self.weight)

        n_freq = N // 2 + 1
        out_full_real = torch.zeros(B, self.out_channels, n_freq, device=x.device)
        out_full_imag = torch.zeros(B, self.out_channels, n_freq, device=x.device)
        out_full_real[..., : self.k_max] = out_ft[..., 0]
        out_full_imag[..., : self.k_max] = out_ft[..., 1]

        out_complex = torch.complex(out_full_real, out_full_imag)
        return torch.fft.irfft(out_complex, n=N, dim=-1)


# ---------------------------------------------------------------------------
# FNO Layer
# ---------------------------------------------------------------------------


class FNOBlock(nn.Module):
    r"""Single Fourier Neural Operator layer.

    .. math::

        v_{l+1} = \sigma\bigl(W\,v_l + \mathcal{K}(v_l) + b\bigr)

    where :math:`\mathcal{K}` is the spectral convolution.

    Args:
        channels: Number of channels (in == out for interior layers).
        k_max: Number of retained Fourier modes.
        activation: Activation function applied element-wise.
    """

    def __init__(
        self,
        channels: int,
        k_max: int,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, k_max)
        self.linear = nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = activation or nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FNO block.

        Args:
            x: Input ``(B, channels, N)``.

        Returns:
            Output ``(B, channels, N)``.
        """
        return self.activation(self.spectral(x) + self.linear(x))


# ---------------------------------------------------------------------------
# Full Fourier Neural Operator
# ---------------------------------------------------------------------------


class FourierNeuralOperator(nn.Module):
    r"""1-D Fourier Neural Operator (Li et al., 2021).

    Maps an input function :math:`u(x)` discretised on ``N`` grid
    points to an output function :math:`\mathcal{G}_\theta(u)(x)` on
    the same grid.  The architecture is:

    1. **Lifting**: pointwise linear map from ``in_channels`` to
       ``width``.
    2. **FNO blocks**: ``n_layers`` spectral convolution layers.
    3. **Projection**: pointwise linear map from ``width`` to
       ``out_channels``.

    Args:
        in_channels: Number of input channels (e.g. 1 for scalar fields,
            2 if the spatial coordinate is appended).
        out_channels: Number of output channels.
        width: Hidden channel width.
        k_max: Number of Fourier modes retained in each layer.
        n_layers: Number of FNO blocks.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        width: int = 64,
        k_max: int = 16,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock(width, k_max) for _ in range(n_layers)]
        )
        self.project = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the neural operator.

        Args:
            x: Input function values ``(B, in_channels, N)``.

        Returns:
            Output function values ``(B, out_channels, N)``.
        """
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        return self.project(h)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


@dataclass
class FNOTrainResult:
    """Training diagnostics for :class:`FourierNeuralOperator`.

    Args:
        train_losses: Per-epoch training losses.
        val_loss: Final validation loss (``None`` if no validation set).
    """

    train_losses: list[float] = field(default_factory=list)
    val_loss: float | None = None


def train_fno(
    model: FourierNeuralOperator,
    train_a: torch.Tensor,
    train_u: torch.Tensor,
    val_a: torch.Tensor | None = None,
    val_u: torch.Tensor | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
    device: str = "cpu",
) -> FNOTrainResult:
    """Train an FNO on paired input/output function data.

    Args:
        model: A :class:`FourierNeuralOperator` instance.
        train_a: Training inputs ``(N_train, in_channels, resolution)``.
        train_u: Training outputs ``(N_train, out_channels, resolution)``.
        val_a: Optional validation inputs.
        val_u: Optional validation outputs.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        seed: Random seed.
        device: Compute device.

    Returns:
        :class:`FNOTrainResult` with loss history.
    """
    set_torch_seed(seed)
    dev = get_device(device)
    model = model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    train_a = train_a.to(dev)
    train_u = train_u.to(dev)
    n_train = train_a.shape[0]

    result = FNOTrainResult()
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            a_batch = train_a[idx]
            u_batch = train_u[idx]
            optimizer.zero_grad()
            pred = model(a_batch)
            loss = loss_fn(pred, u_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        result.train_losses.append(epoch_loss / max(n_batches, 1))

    if val_a is not None and val_u is not None:
        model.eval()
        with torch.no_grad():
            pred_val = model(val_a.to(dev))
            result.val_loss = float(loss_fn(pred_val, val_u.to(dev)).item())

    return result
