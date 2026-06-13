"""Tests for genomics batch correction."""

from ml_sci.data.genomics_dgp import GenomicsDGPConfig, generate_genomics_data
from ml_sci.genomics.batch_correction import apply_batch_correction
from ml_sci.genomics.metrics import batch_mixing_score, biological_preservation
from ml_sci.genomics.vae import train_vae


def test_linear_correction_improves_batch_mixing():
    data = generate_genomics_data(GenomicsDGPConfig(seed=0))
    none = apply_batch_correction(data.expression, data.batch_labels, method="none")
    linear = apply_batch_correction(data.expression, data.batch_labels, method="linear")
    score_none = batch_mixing_score(none.corrected, data.batch_labels)
    score_linear = batch_mixing_score(linear.corrected, data.batch_labels)
    assert score_linear >= score_none


def test_sinkhorn_preserves_biology():
    data = generate_genomics_data(GenomicsDGPConfig(seed=1))
    corrected = apply_batch_correction(data.expression, data.batch_labels, method="sinkhorn")
    bio = biological_preservation(corrected.corrected, data.expression, data.cell_types)
    assert bio > 0.7


def test_vae_reconstruction():
    data = generate_genomics_data(GenomicsDGPConfig(n_cells_per_type=50, seed=2))
    vae = train_vae(data.expression, epochs=20, seed=2)
    assert vae.reconstructed.shape == data.expression.shape
    assert vae.train_loss < 1e6
