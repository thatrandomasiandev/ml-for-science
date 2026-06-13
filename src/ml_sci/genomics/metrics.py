"""Genomics batch correction metrics."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def batch_mixing_score(expression: np.ndarray, batch_labels: np.ndarray, k: int = 15) -> float:
    """kNN batch entropy: higher means better batch mixing."""
    scaler = StandardScaler()
    x = scaler.fit_transform(expression)
    n_batches = len(np.unique(batch_labels))
    if n_batches < 2:
        return 1.0
    k = min(k, len(x) - 1)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x, batch_labels)
    probs = knn.predict_proba(x)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    max_ent = np.log(n_batches)
    return float(np.mean(entropy / max_ent))


def biological_preservation(
    corrected: np.ndarray,
    original: np.ndarray,
    cell_types: np.ndarray,
) -> float:
    """Correlation of cell-type centroids before vs after correction."""
    types = np.unique(cell_types)
    orig_centroids = np.stack([original[cell_types == t].mean(axis=0) for t in types])
    corr_centroids = np.stack([corrected[cell_types == t].mean(axis=0) for t in types])
    corrs = [
        np.corrcoef(orig_centroids[i], corr_centroids[i])[0, 1]
        for i in range(len(types))
    ]
    return float(np.nanmean(corrs))


def reconstruction_rmse(original: np.ndarray, corrected: np.ndarray) -> float:
    return float(np.sqrt(np.mean((original - corrected) ** 2)))
