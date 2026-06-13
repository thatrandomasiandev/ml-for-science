"""Synthetic protein structures with SE(3)-invariant stability labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_sci.data.base import ProteinDataset
from ml_sci.utils.seed import set_seed


@dataclass
class ProteinDGPConfig:
    """Configuration for protein property prediction benchmark."""

    n_proteins: int = 400
    n_residues: int = 32
    bond_length: float = 1.5
    noise_std: float = 0.05
    seed: int = 42


def _random_walk_chain(rng: np.random.Generator, n_residues: int, bond_length: float) -> np.ndarray:
    """Generate a 3D polymer chain via random walk with fixed bond length."""
    directions = rng.standard_normal((n_residues - 1, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    coords = np.zeros((n_residues, 3), dtype=np.float64)
    for i in range(1, n_residues):
        coords[i] = coords[i - 1] + bond_length * directions[i - 1]
    return coords


def _radius_of_gyration(coords: np.ndarray) -> float:
    center = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))


def _contact_density(coords: np.ndarray, cutoff: float = 4.0) -> float:
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    n = coords.shape[0]
    contacts = (dists < cutoff) & (dists > 0.5)
    return float(contacts.sum() / (n * (n - 1)))


def stability_oracle(coords: np.ndarray, template: np.ndarray, w_rg: float, w_contact: float) -> float:
    """SE(3)-invariant folding stability proxy from geometry invariants."""
    centered = coords - coords.mean(axis=0)
    template_c = template - template.mean(axis=0)
    rg = _radius_of_gyration(centered)
    template_rg = _radius_of_gyration(template_c)
    rg_term = -((rg - template_rg) / max(template_rg, 1e-6)) ** 2
    contact_term = _contact_density(centered)
    return float(w_rg * rg_term + w_contact * contact_term)


def generate_protein_data(config: ProteinDGPConfig) -> ProteinDataset:
    """Generate protein chains with known invariant stability landscape."""
    rng = set_seed(config.seed)
    template = _random_walk_chain(rng, config.n_residues, config.bond_length)
    w_rg = 2.0
    w_contact = 3.0

    coords = np.zeros((config.n_proteins, config.n_residues, 3), dtype=np.float64)
    properties = np.zeros(config.n_proteins, dtype=np.float64)

    for i in range(config.n_proteins):
        chain = _random_walk_chain(rng, config.n_residues, config.bond_length)
        if rng.random() < 0.4:
            # Perturb toward compact or extended conformations
            scale = rng.uniform(0.6, 1.4)
            chain = (chain - chain.mean(axis=0)) * scale + chain.mean(axis=0)
        coords[i] = chain
        prop = stability_oracle(chain, template, w_rg, w_contact)
        properties[i] = prop + rng.normal(0.0, config.noise_std)

    return ProteinDataset(
        coords=coords,
        properties=properties,
        metadata={
            "dgp": "se3_invariant_stability",
            "n_residues": config.n_residues,
            "bond_length": config.bond_length,
            "noise_std": config.noise_std,
            "seed": config.seed,
        },
        ground_truth={
            "template": template,
            "w_rg": w_rg,
            "w_contact": w_contact,
            "property_mean": float(properties.mean()),
            "property_std": float(properties.std()),
        },
    )
