"""Physics-informed neural networks for PDE-constrained learning.

Provides a general-purpose PINN backbone, canonical PDE residuals
(heat equation, Burgers' equation), and an adaptive trainer with
collocation point refinement.  The original ``ClimatePINN`` and
``BicubicDownscaler`` are retained for backward compatibility with the
climate downscaling pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn

from ml_sci.utils.device import get_device
from ml_sci.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activation look-up
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def _get_activation(name: str) -> nn.Module:
    """Return an activation module by string key.

    Args:
        name: One of ``"tanh"``, ``"relu"``, ``"silu"``, ``"gelu"``.

    Returns:
        An instantiated ``nn.Module`` activation layer.

    Raises:
        ValueError: If *name* is not a recognised activation.
    """
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}; choose from {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


# ---------------------------------------------------------------------------
# General-purpose PINN
# ---------------------------------------------------------------------------


class PINN(nn.Module):
    r"""General-purpose physics-informed neural network.

    A fully-connected MLP :math:`f_\theta: \mathbb{R}^d \to \mathbb{R}^{d_{\text{out}}}`
    trained with a combined data-fitting + PDE-residual loss:

    .. math::

        \mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_r \mathcal{L}_{\text{res}}

    Args:
        input_dim: Dimension of the input (e.g. space-time coordinates).
        hidden_dim: Width of hidden layers.
        output_dim: Dimension of the network output.
        n_hidden: Number of hidden layers.
        activation: Activation function name (``"tanh"`` | ``"relu"`` |
            ``"silu"`` | ``"gelu"``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_hidden: int = 4,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), _get_activation(activation)]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), _get_activation(activation)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape ``(N, input_dim)``.

        Returns:
            Output tensor of shape ``(N, output_dim)``.
        """
        return self.net(x)

    def residual(
        self,
        x: torch.Tensor,
        pde_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        r"""Evaluate the PDE residual via automatic differentiation.

        Computes :math:`u = f_\theta(x)`, the Jacobian
        :math:`\nabla_x u`, and passes ``(x, u, grad_u)`` to *pde_fn*
        which should return a tensor whose squared mean becomes the
        physics loss.

        Args:
            x: Collocation points ``(N, input_dim)`` — **must** have
                ``requires_grad=True``.
            pde_fn: Callable ``(x, u, grad_u) -> residual``.

        Returns:
            Per-point residual tensor of shape ``(N,)`` or ``(N, output_dim)``.
        """
        x = x.detach().requires_grad_(True)
        u = self.forward(x)
        grad_u = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        return pde_fn(x, u, grad_u)

    def total_loss(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        x_colloc: torch.Tensor,
        pde_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        lambda_r: float = 0.1,
    ) -> torch.Tensor:
        r"""Combined data + physics loss.

        .. math::

            \mathcal{L} = \frac{1}{N_d}\sum\|f_\theta(x_i)-y_i\|^2
                          + \lambda_r \frac{1}{N_c}\sum r(x_j)^2

        Args:
            x_data: Labelled inputs ``(N_d, input_dim)``.
            y_data: Labelled targets ``(N_d, output_dim)``.
            x_colloc: Collocation inputs ``(N_c, input_dim)``.
            pde_fn: PDE residual callable (see :meth:`residual`).
            lambda_r: Relative weight of the physics residual.

        Returns:
            Scalar loss tensor.
        """
        pred = self.forward(x_data)
        data_loss = nn.functional.mse_loss(pred, y_data)
        res = self.residual(x_colloc, pde_fn)
        physics_loss = torch.mean(res ** 2)
        return data_loss + lambda_r * physics_loss


# ---------------------------------------------------------------------------
# Canonical PDE residuals
# ---------------------------------------------------------------------------


def heat_equation_residual(
    x_xt: torch.Tensor,
    u: torch.Tensor,
    u_grad: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    r"""Residual of the 1-D heat equation via autograd.

    .. math::

        \frac{\partial u}{\partial t}
        - \alpha \frac{\partial^2 u}{\partial x^2} = 0

    The input ``x_xt`` is expected to be ``(N, 2)`` with columns
    :math:`(x, t)`.  Second spatial derivatives are computed with a
    second ``autograd`` call.

    Args:
        x_xt: Collocation coordinates ``(N, 2)`` with ``requires_grad``.
        u: Network output ``(N, 1)``.
        u_grad: Jacobian ``(N, 2)`` — columns are
            :math:`[\partial u/\partial x,\; \partial u/\partial t]`.
        alpha: Thermal diffusivity.

    Returns:
        Per-point residual ``(N, 1)``.
    """
    du_dx = u_grad[:, 0:1]
    du_dt = u_grad[:, 1:2]
    d2u_dx2 = torch.autograd.grad(
        du_dx,
        x_xt,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    return du_dt - alpha * d2u_dx2


def burgers_residual(
    x_xt: torch.Tensor,
    u: torch.Tensor,
    u_grad: torch.Tensor,
    nu: float = 0.01,
) -> torch.Tensor:
    r"""Residual of the 1-D viscous Burgers' equation.

    .. math::

        \frac{\partial u}{\partial t}
        + u\,\frac{\partial u}{\partial x}
        - \nu\,\frac{\partial^2 u}{\partial x^2} = 0

    Args:
        x_xt: Collocation coordinates ``(N, 2)`` with ``requires_grad``.
        u: Network output ``(N, 1)``.
        u_grad: Jacobian ``(N, 2)``.
        nu: Viscosity coefficient.

    Returns:
        Per-point residual ``(N, 1)``.
    """
    du_dx = u_grad[:, 0:1]
    du_dt = u_grad[:, 1:2]
    d2u_dx2 = torch.autograd.grad(
        du_dx,
        x_xt,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    return du_dt + u * du_dx - nu * d2u_dx2


# ---------------------------------------------------------------------------
# Trainer with adaptive collocation refinement
# ---------------------------------------------------------------------------


@dataclass
class PINNTrainResult:
    """Container for PINN training diagnostics.

    Args:
        data_losses: Per-epoch data loss history.
        physics_losses: Per-epoch physics loss history.
        total_losses: Per-epoch total loss history.
        colloc_sizes: Number of collocation points per refinement cycle.
    """

    data_losses: list[float] = field(default_factory=list)
    physics_losses: list[float] = field(default_factory=list)
    total_losses: list[float] = field(default_factory=list)
    colloc_sizes: list[int] = field(default_factory=list)


class PINNTrainer:
    """Trainer for :class:`PINN` with adaptive collocation point refinement.

    Every *refine_every* epochs the residual is evaluated on the current
    collocation set and the top-*refine_fraction* worst-performing points
    are duplicated (with small perturbation) to focus capacity where the
    PDE is most violated.

    Args:
        model: A :class:`PINN` instance.
        pde_fn: PDE residual callable compatible with :meth:`PINN.residual`.
        lr: Adam learning rate.
        lambda_r: Physics loss weight.
        refine_every: Epochs between adaptive refinement steps.
        refine_fraction: Fraction of worst-residual points to duplicate.
        seed: Random seed for reproducibility.
        device: Compute device string.
    """

    def __init__(
        self,
        model: PINN,
        pde_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        lr: float = 1e-3,
        lambda_r: float = 0.1,
        refine_every: int = 50,
        refine_fraction: float = 0.2,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.pde_fn = pde_fn
        self.lr = lr
        self.lambda_r = lambda_r
        self.refine_every = refine_every
        self.refine_fraction = refine_fraction
        self.seed = seed
        self.device = get_device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _refine_collocation(
        self,
        x_colloc: torch.Tensor,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Double the worst-residual collocation points.

        Args:
            x_colloc: Current collocation points on *self.device*.
            rng: NumPy random generator for perturbation noise.

        Returns:
            Augmented collocation tensor on *self.device*.
        """
        self.model.eval()
        x_c = x_colloc.detach().requires_grad_(True)
        with torch.enable_grad():
            res = self.model.residual(x_c, self.pde_fn)
        res_mag = (res ** 2).sum(dim=-1).detach()
        n_refine = max(1, int(self.refine_fraction * x_colloc.shape[0]))
        _, top_idx = torch.topk(res_mag, n_refine)
        new_pts = x_colloc[top_idx].detach().clone()
        noise = torch.tensor(
            rng.normal(0, 1e-3, size=new_pts.shape).astype(np.float32),
            device=self.device,
        )
        new_pts = new_pts + noise
        return torch.cat([x_colloc, new_pts], dim=0)

    def train(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        x_colloc: torch.Tensor,
        epochs: int = 200,
    ) -> PINNTrainResult:
        """Run the training loop with periodic collocation refinement.

        Args:
            x_data: Labelled inputs ``(N_d, input_dim)``.
            y_data: Labelled targets ``(N_d, output_dim)``.
            x_colloc: Initial collocation points ``(N_c, input_dim)``.
            epochs: Number of optimisation epochs.

        Returns:
            A :class:`PINNTrainResult` with loss histories.
        """
        set_torch_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        result = PINNTrainResult()

        x_d = x_data.to(self.device)
        y_d = y_data.to(self.device)
        x_c = x_colloc.to(self.device)

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            pred = self.model(x_d)
            data_loss = nn.functional.mse_loss(pred, y_d)

            x_c_grad = x_c.detach().requires_grad_(True)
            res = self.model.residual(x_c_grad, self.pde_fn)
            physics_loss = torch.mean(res ** 2)

            loss = data_loss + self.lambda_r * physics_loss
            loss.backward()
            self.optimizer.step()

            result.data_losses.append(float(data_loss.item()))
            result.physics_losses.append(float(physics_loss.item()))
            result.total_losses.append(float(loss.item()))

            if epoch % self.refine_every == 0:
                x_c = self._refine_collocation(x_c, rng)
                result.colloc_sizes.append(x_c.shape[0])
                logger.info(
                    "Epoch %d: refined collocation to %d points", epoch, x_c.shape[0]
                )

        return result


# ---------------------------------------------------------------------------
# Legacy climate models (kept for backward compatibility)
# ---------------------------------------------------------------------------


class ClimatePINN(nn.Module):
    """MLP mapping (x, y) -> temperature with Laplacian physics penalty.

    Args:
        hidden_dim: Width of hidden layers.
        n_layers: Number of hidden layers.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Predict scalar field from 2-D coordinates.

        Args:
            xy: Input coordinates ``(N, 2)``.

        Returns:
            Predicted field values ``(N,)``.
        """
        return self.net(xy).squeeze(-1)

    def laplacian_residual(
        self, xy: torch.Tensor, field: torch.Tensor, dx: float
    ) -> torch.Tensor:
        """Compute discrete Laplacian residual on a regular grid.

        Args:
            xy: Grid coordinates ``(N, 2)``.
            field: Predicted field values ``(N,)``.
            dx: Grid spacing.

        Returns:
            Laplacian residual ``(N,)``.
        """
        n = int(math.isqrt(xy.shape[0]))
        u = field.reshape(n, n)
        lap = (
            -4.0 * u
            + torch.roll(u, 1, dims=0)
            + torch.roll(u, -1, dims=0)
            + torch.roll(u, 1, dims=1)
            + torch.roll(u, -1, dims=1)
        ) / (dx ** 2)
        return lap.reshape(-1)


class BicubicDownscaler:
    """Classical baseline: bicubic upsampling of coarse grid.

    Args:
        factor: Integer upsampling factor.
    """

    def __init__(self, factor: int) -> None:
        self.factor = factor

    def predict(self, coarse: np.ndarray) -> np.ndarray:
        """Upsample a coarse 2-D field via bicubic interpolation.

        Args:
            coarse: Low-resolution field ``(H, W)``.

        Returns:
            High-resolution field ``(H*factor, W*factor)``.
        """
        from scipy.ndimage import zoom

        return zoom(coarse, self.factor, order=3)
