"""Materials module exports."""

from ml_sci.materials.active_learning import (
    ActiveLearningLoop,
    ActiveLearningLoopResult,
    ActiveLearningResult,
    BayesianOptimizer,
    expected_improvement,
    expected_improvement_search,
    probability_of_improvement,
    random_search,
    run_active_learning,
    uncertainty_search,
    upper_confidence_bound,
)
from ml_sci.materials.metrics import normalized_score, regret, top_k_hit_rate

__all__ = [
    "ActiveLearningLoop",
    "ActiveLearningLoopResult",
    "ActiveLearningResult",
    "BayesianOptimizer",
    "expected_improvement",
    "expected_improvement_search",
    "normalized_score",
    "probability_of_improvement",
    "random_search",
    "regret",
    "run_active_learning",
    "top_k_hit_rate",
    "uncertainty_search",
    "upper_confidence_bound",
]
