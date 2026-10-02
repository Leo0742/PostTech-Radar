from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def weighted_soft_vote(probabilities: Sequence[np.ndarray], *, weights: Sequence[float] | None = None) -> np.ndarray:
    if not probabilities:
        raise ValueError("at least one probability matrix is required")
    shape = probabilities[0].shape
    if any(matrix.shape != shape for matrix in probabilities):
        raise ValueError("all probability matrices must have identical shape")
    actual_weights = np.asarray(weights if weights is not None else np.ones(len(probabilities)), dtype=float)
    if len(actual_weights) != len(probabilities) or np.any(actual_weights < 0) or actual_weights.sum() <= 0:
        raise ValueError("weights must be non-negative and match probability matrices")
    output = np.average(np.stack(probabilities), axis=0, weights=actual_weights)
    return output / output.sum(axis=1, keepdims=True)
