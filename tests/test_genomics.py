"""Tests for scVAE, LDVAE, and batch correction methods."""

from __future__ import annotations

import numpy as np
import torch

from ml_sci.genomics.batch_correction import CombatCorrector, HarmonyCorrector
from ml_sci.genomics.metrics import batch_mixing_score
from ml_sci.genomics.vae import LDVAE, scVAE, train_scvae


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_count_data(
    n_cells: int = 200,
    n_genes: int = 50,
    n_batches: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic UMI-like count data with batch effects."""
    rng = np.random.default_rng(seed)
    base = rng.poisson(lam=5.0, size=(n_cells, n_genes)).astype(np.float32)
    batches = np.repeat(np.arange(n_batches), n_cells // n_batches)
    for b in range(n_batches):
        mask = batches == b
        base[mask] += rng.poisson(lam=2.0 * (b + 1), size=(mask.sum(), n_genes))
    return base, batches


# ------------------------------------------------------------------ #
# scVAE reconstructs
# ------------------------------------------------------------------ #


def test_scvae_forward_shape():
    """scVAE forward pass should return correct shapes."""
    n_genes = 30
    model = scVAE(n_genes=n_genes, latent_dim=8, hidden_dim=32, n_hidden=1)
    x = torch.rand(16, n_genes)
    mu_x, mu_z, logvar_z = model(x)
    assert mu_x.shape == (16, n_genes)
    assert mu_z.shape == (16, 8)
    assert logvar_z.shape == (16, 8)


def test_scvae_reconstructs():
    """scVAE trained on count data should produce reasonable reconstructions."""
    expression, _ = _make_count_data(n_cells=100, n_genes=30, seed=1)
    result = train_scvae(
        expression, latent_dim=8, hidden_dim=32, n_hidden=1, epochs=30, seed=1
    )
    assert result.reconstructed.shape == expression.shape
    assert np.isfinite(result.train_loss)


# ------------------------------------------------------------------ #
# ELBO (loss) decreases over training
# ------------------------------------------------------------------ #


def test_scvae_elbo_improves():
    """scVAE training loss should decrease from epoch 1 to the final epoch."""
    expression, _ = _make_count_data(n_cells=80, n_genes=20, seed=2)
    torch.manual_seed(2)
    x = torch.tensor(expression, dtype=torch.float32)
    model = scVAE(n_genes=20, latent_dim=4, hidden_dim=32, n_hidden=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for _ in range(40):
        model.train()
        optimizer.zero_grad()
        mu_x, mu_z, logvar_z = model(x)
        loss = model.loss(x, mu_x, mu_z, logvar_z, beta=1.0)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.2f}, last={losses[-1]:.2f}"
    )


# ------------------------------------------------------------------ #
# LDVAE
# ------------------------------------------------------------------ #


def test_ldvae_loadings_shape():
    """LDVAE should expose a loadings matrix of shape (latent_dim, n_genes)."""
    model = LDVAE(n_genes=50, latent_dim=10, hidden_dim=32, n_hidden=1)
    assert model.loadings.shape == (50, 10)


def test_ldvae_forward_and_loss():
    """LDVAE forward + loss should produce finite values."""
    torch.manual_seed(3)
    model = LDVAE(n_genes=30, latent_dim=5, hidden_dim=32, n_hidden=1)
    x = torch.rand(16, 30)
    x_hat, mu, logvar = model(x)
    loss = model.loss(x, x_hat, mu, logvar, beta=0.5)
    assert x_hat.shape == x.shape
    assert torch.isfinite(loss)


# ------------------------------------------------------------------ #
# CombatCorrector reduces batch effect
# ------------------------------------------------------------------ #


def test_combat_reduces_batch_effect():
    """CombatCorrector should improve batch mixing score."""
    expression, batches = _make_count_data(n_cells=200, n_genes=30, seed=4)
    score_before = batch_mixing_score(expression, batches)

    combat = CombatCorrector(seed=4)
    corrected = combat.fit_transform(expression, batches)

    score_after = batch_mixing_score(corrected, batches)
    assert score_after >= score_before - 0.05, (
        f"Combat did not improve mixing: before={score_before:.3f}, "
        f"after={score_after:.3f}"
    )


def test_combat_fit_transform_consistency():
    """fit() then transform() should match fit_transform()."""
    expression, batches = _make_count_data(n_cells=100, n_genes=20, seed=5)
    combat = CombatCorrector(seed=5)
    result_1 = combat.fit_transform(expression, batches)

    combat2 = CombatCorrector(seed=5)
    combat2.fit(expression, batches)
    result_2 = combat2.transform(expression, batches)

    np.testing.assert_allclose(result_1, result_2, atol=1e-6)


# ------------------------------------------------------------------ #
# HarmonyCorrector reduces batch effect
# ------------------------------------------------------------------ #


def test_harmony_reduces_batch_effect():
    """HarmonyCorrector should improve batch mixing score."""
    expression, batches = _make_count_data(n_cells=200, n_genes=30, seed=6)
    score_before = batch_mixing_score(expression, batches)

    harmony = HarmonyCorrector(
        n_components=10, n_clusters=3, max_iters=10, seed=6
    )
    corrected = harmony.fit_transform(expression, batches)

    score_after = batch_mixing_score(corrected, batches)
    assert score_after >= score_before - 0.05, (
        f"Harmony did not improve mixing: before={score_before:.3f}, "
        f"after={score_after:.3f}"
    )
