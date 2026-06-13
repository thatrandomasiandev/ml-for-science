"""Materials discovery metrics."""

from __future__ import annotations

import numpy as np


def regret(best_found: float, oracle_value: float) -> float:
    return float(max(0.0, oracle_value - best_found))


def normalized_score(best_found: float, random_baseline: float, oracle_value: float) -> float:
    denom = oracle_value - random_baseline
    if abs(denom) < 1e-12:
        return 0.0
    return float((best_found - random_baseline) / denom)


def top_k_hit_rate(properties: list[float], threshold: float, k: int = 10) -> float:
    if not properties:
        return 0.0
    top_k = sorted(properties, reverse=True)[:k]
    return float(np.mean([p >= threshold for p in top_k]))
