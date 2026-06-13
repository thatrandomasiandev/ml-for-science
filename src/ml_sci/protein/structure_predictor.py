"""Contact-map–based protein structure prediction.

Provides:

* :class:`ProteinStructurePredictor` — a neural network that encodes
  amino-acid sequences and predicts pairwise residue contacts as an
  :math:`L \\times L` symmetric probability matrix.
* :class:`ProteinContactDataset` — a ``torch.utils.data.Dataset`` that
  synthesises random sequences with deterministic contact maps for
  training / testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ml_sci.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)

_NUM_AMINO_ACIDS: int = 20


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ProteinContactDataset(Dataset):
    """Synthetic dataset of protein sequences with contact maps.

    Each sample is a randomly generated integer sequence (amino-acid
    indices) paired with a deterministic binary contact map derived from
    a latent 3-D structure.

    Args:
        n_proteins: Number of proteins in the dataset.
        seq_len: Sequence length :math:`L`.
        contact_threshold: Distance threshold (Å) for defining contacts
            in the latent structure.
        seed: Random seed for reproducible data generation.
    """

    def __init__(
        self,
        n_proteins: int = 200,
        seq_len: int = 50,
        contact_threshold: float = 8.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_proteins = n_proteins
        self.seq_len = seq_len
        self.contact_threshold = contact_threshold

        rng = np.random.default_rng(seed)
        self.sequences = rng.integers(0, _NUM_AMINO_ACIDS, size=(n_proteins, seq_len))

        self.contact_maps = np.zeros(
            (n_proteins, seq_len, seq_len), dtype=np.float32
        )
        for i in range(n_proteins):
            coords = np.cumsum(rng.normal(0, 3.8, size=(seq_len, 3)), axis=0)
            dists = np.sqrt(
                ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)
            )
            contacts = (dists < contact_threshold).astype(np.float32)
            np.fill_diagonal(contacts, 0.0)
            self.contact_maps[i] = contacts

    def __len__(self) -> int:
        return self.n_proteins

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(sequence, contact_map)`` for index *idx*.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(sequence_tensor, contact_map_tensor)`` where
            ``sequence_tensor`` is ``(L,)`` long and
            ``contact_map_tensor`` is ``(L, L)`` float.
        """
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        cmap = torch.tensor(self.contact_maps[idx], dtype=torch.float32)
        return seq, cmap


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ProteinStructurePredictor(nn.Module):
    r"""Contact-map prediction network.

    Architecture:

    1. **Embedding**: learnable amino-acid embedding
       :math:`e_i \in \mathbb{R}^{d}`.
    2. **1-D convolution encoder**: extracts local motifs along the
       sequence.
    3. **Outer product + MLP**: produces pairwise features
       :math:`z_{ij} = \text{MLP}([e_i \| e_j \| |i-j|])` and predicts
       contact probability via sigmoid.

    The output is a symmetric :math:`L \times L` matrix of contact
    probabilities.

    Args:
        vocab_size: Number of amino-acid types (default 20).
        embed_dim: Embedding dimensionality.
        hidden_dim: Width of the pairwise MLP.
        n_filters: Number of 1-D convolution filters.
        kernel_size: Convolution kernel size (odd recommended).
    """

    def __init__(
        self,
        vocab_size: int = _NUM_AMINO_ACIDS,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_filters: int = 64,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.conv_encoder = nn.Sequential(
            nn.Conv1d(embed_dim, n_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )

        pair_input_dim = 2 * n_filters + 1
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode amino-acid sequence into per-residue features.

        Args:
            seq: Integer sequence tensor ``(B, L)``.

        Returns:
            Residue features ``(B, L, n_filters)``.
        """
        emb = self.embed(seq)  # (B, L, embed_dim)
        h = self.conv_encoder(emb.transpose(1, 2))  # (B, n_filters, L)
        return h.transpose(1, 2)  # (B, L, n_filters)

    def predict_contacts(self, seq: torch.Tensor) -> torch.Tensor:
        r"""Predict the :math:`L \times L` contact probability matrix.

        Args:
            seq: Integer sequence ``(B, L)``.

        Returns:
            Symmetric contact probabilities ``(B, L, L)``.
        """
        h = self.encode_sequence(seq)  # (B, L, D)
        B, L, D = h.shape

        h_i = h.unsqueeze(2).expand(-1, -1, L, -1)  # (B, L, L, D)
        h_j = h.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, L, D)

        pos = torch.arange(L, device=h.device, dtype=h.dtype)
        sep = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().unsqueeze(0)  # (1, L, L)
        sep = sep.unsqueeze(-1).expand(B, -1, -1, -1) / L  # (B, L, L, 1)

        pair_feats = torch.cat([h_i, h_j, sep], dim=-1)  # (B, L, L, 2D+1)
        logits = self.pair_mlp(pair_feats).squeeze(-1)  # (B, L, L)

        logits = (logits + logits.transpose(1, 2)) / 2.0
        return torch.sigmoid(logits)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Forward pass (alias for :meth:`predict_contacts`).

        Args:
            seq: Integer sequence ``(B, L)``.

        Returns:
            Contact probability matrix ``(B, L, L)``.
        """
        return self.predict_contacts(seq)

    @staticmethod
    def contact_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        pos_weight: float = 5.0,
    ) -> torch.Tensor:
        """Binary cross-entropy loss for contact prediction.

        Uses a positive class weight to handle the sparsity of true
        contacts.

        Args:
            pred: Predicted contact probabilities ``(B, L, L)`` — values
                in :math:`[0, 1]` (already sigmoided).
            target: Ground-truth binary contact map ``(B, L, L)``.
            pos_weight: Weight applied to positive (contact) entries.

        Returns:
            Scalar loss tensor.
        """
        weight = torch.where(target > 0.5, pos_weight, 1.0)
        eps = 1e-7
        pred = pred.clamp(eps, 1.0 - eps)
        bce = -(
            weight * target * torch.log(pred)
            + (1.0 - target) * torch.log(1.0 - pred)
        )
        return bce.mean()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class StructurePredictionResult:
    """Contact-map prediction evaluation result.

    Args:
        precision_at_L: Precision of top-L predicted contacts.
        mean_loss: Mean loss over the evaluation set.
    """

    precision_at_L: float
    mean_loss: float
