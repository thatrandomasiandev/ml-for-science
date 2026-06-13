"""Tests for PINN, PDE residuals, and EGNN rotation invariance."""

from __future__ import annotations

import math

import numpy as np
import torch

from ml_sci.models.egnn import FullEGNN
from ml_sci.models.pinn import (
    PINN,
    PINNTrainer,
    burgers_residual,
    heat_equation_residual,
)


# ------------------------------------------------------------------ #
# Heat equation: exact solution has zero residual
# ------------------------------------------------------------------ #


def test_heat_residual_zero_for_exact_solution():
    """The exact solution u(x,t) = sin(pi*x) * exp(-pi^2*t) satisfies
    the heat equation du/dt - d^2u/dx^2 = 0 with alpha=1.  A PINN
    that reproduces this solution should yield a near-zero residual.
    """
    N = 64
    x_vals = torch.linspace(0.0, 1.0, N)
    t_vals = torch.linspace(0.0, 0.5, N)
    x_grid, t_grid = torch.meshgrid(x_vals, t_vals, indexing="ij")
    x_xt = torch.stack([x_grid.reshape(-1), t_grid.reshape(-1)], dim=-1)
    x_xt.requires_grad_(True)

    u_exact = (
        torch.sin(math.pi * x_xt[:, 0:1])
        * torch.exp(-(math.pi ** 2) * x_xt[:, 1:2])
    )

    grad_u = torch.autograd.grad(
        u_exact,
        x_xt,
        grad_outputs=torch.ones_like(u_exact),
        create_graph=True,
    )[0]

    residual = heat_equation_residual(x_xt, u_exact, grad_u, alpha=1.0)
    assert residual.abs().max().item() < 1e-4, (
        f"Heat residual too large: {residual.abs().max().item():.6e}"
    )


# ------------------------------------------------------------------ #
# PINN converges on a simple regression target
# ------------------------------------------------------------------ #


def test_pinn_converges_on_sinusoid():
    """Train a PINN (data-only, no physics) to fit sin(x) and verify
    the final loss is small.
    """
    torch.manual_seed(0)
    x = torch.linspace(0, 2 * math.pi, 100).unsqueeze(-1)
    y = torch.sin(x)

    model = PINN(input_dim=1, hidden_dim=32, output_dim=1, n_hidden=3, activation="tanh")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(500):
        optimizer.zero_grad()
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

    final_loss = torch.nn.functional.mse_loss(model(x), y).item()
    assert final_loss < 0.05, f"PINN did not converge: loss={final_loss:.4f}"


# ------------------------------------------------------------------ #
# PINNTrainer with heat equation
# ------------------------------------------------------------------ #


def test_pinn_trainer_reduces_loss():
    """PINNTrainer should reduce total loss over training epochs."""
    torch.manual_seed(1)

    x_data = torch.rand(50, 2)
    u_exact = torch.sin(math.pi * x_data[:, 0:1]) * torch.exp(
        -(math.pi ** 2) * x_data[:, 1:2]
    )
    x_colloc = torch.rand(100, 2)

    model = PINN(input_dim=2, hidden_dim=32, output_dim=1, n_hidden=3)

    def pde_fn(
        x: torch.Tensor, u: torch.Tensor, grad_u: torch.Tensor
    ) -> torch.Tensor:
        return heat_equation_residual(x, u, grad_u, alpha=1.0)

    trainer = PINNTrainer(
        model, pde_fn, lr=1e-3, lambda_r=0.1, refine_every=25, seed=1
    )
    result = trainer.train(x_data, u_exact, x_colloc, epochs=50)

    assert result.total_losses[-1] < result.total_losses[0], (
        "Training loss did not decrease"
    )


# ------------------------------------------------------------------ #
# Burgers residual shape check
# ------------------------------------------------------------------ #


def test_burgers_residual_shape():
    """burgers_residual should return the same shape as u."""
    x_xt = torch.rand(32, 2, requires_grad=True)
    u = torch.sin(x_xt[:, 0:1]) * torch.exp(-x_xt[:, 1:2])
    grad_u = torch.autograd.grad(
        u, x_xt, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    res = burgers_residual(x_xt, u, grad_u, nu=0.01)
    assert res.shape == u.shape


# ------------------------------------------------------------------ #
# EGNN rotation invariance
# ------------------------------------------------------------------ #


def test_full_egnn_rotation_invariance():
    """FullEGNN scalar output should be invariant under SO(3) rotation."""
    torch.manual_seed(42)
    B, N, D = 4, 10, 1
    coords = torch.randn(B, N, 3)
    feats = torch.randn(B, N, D)

    model = FullEGNN(
        node_feat_dim=D,
        hidden_dim=32,
        output_dim=1,
        n_layers=2,
        dropout=0.0,
        coord_scale=0.01,
    )

    assert FullEGNN.verify_equivariance(model, feats, coords, seed=7, atol=1e-3)


def test_full_egnn_translation_invariance():
    """FullEGNN output should be invariant under uniform translation."""
    torch.manual_seed(99)
    B, N, D = 3, 8, 1
    coords = torch.randn(B, N, 3)
    feats = torch.randn(B, N, D)

    model = FullEGNN(
        node_feat_dim=D,
        hidden_dim=32,
        output_dim=1,
        n_layers=2,
        dropout=0.0,
        coord_scale=0.01,
    )
    model.eval()

    with torch.no_grad():
        out1 = model(feats, coords)
        shift = torch.tensor([10.0, -5.0, 3.0])
        out2 = model(feats, coords + shift)

    assert torch.allclose(out1, out2, atol=1e-4), (
        f"Translation invariance failed: max |Δ|={( out1 - out2).abs().max():.6e}"
    )
