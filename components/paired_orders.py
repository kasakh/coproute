"""Ordering helpers shared by the canonical paired experiment suite.

The paired suite compares ordering rules while holding query-specific factorized
prefix sizing fixed.  An order is converted to a nested score family by assigning
model at position ``k`` the all-miss probability of the preceding prefix,
expressed as ``1 - residual``.  Thresholding these scores therefore selects an
exact prefix, while one global threshold can produce different prefix lengths for
different queries.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def per_query_orders_factorized_scores(
    probabilities: np.ndarray,
    orders: np.ndarray | Sequence[Sequence[int]],
) -> np.ndarray:
    """Convert one full model order per query to nested factorized scores.

    Args:
        probabilities: Estimated correctness probabilities with shape
            ``(n_queries, n_models)``.
        orders: A full permutation of model indices for every query, with shape
            ``(n_queries, n_models)``.

    Returns:
        A finite score matrix with the same shape as ``probabilities``.  The
        first ordered model has score zero.  Scores are nondecreasing along each
        order and selecting scores at or below a threshold yields a prefix.

    Full permutations are required deliberately.  A partial learned path would
    give different methods different endpoint sets, making certification and
    structural feasibility non-paired.
    """

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("probabilities must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    order_matrix = np.asarray(orders)
    if order_matrix.shape != values.shape:
        raise ValueError("orders must contain one full model permutation per query")
    if not np.issubdtype(order_matrix.dtype, np.integer):
        if not np.all(np.isfinite(order_matrix)) or not np.all(order_matrix == np.floor(order_matrix)):
            raise ValueError("orders must contain integer model indices")
    order_matrix = order_matrix.astype(int, copy=False)

    n_queries, n_models = values.shape
    expected = np.arange(n_models)
    if np.any((order_matrix < 0) | (order_matrix >= n_models)):
        raise ValueError("orders contain an out-of-range model index")
    if any(not np.array_equal(np.sort(row), expected) for row in order_matrix):
        raise ValueError("each order row must be a full permutation of model indices")

    scores = np.empty_like(values, dtype=float)
    for row in range(n_queries):
        residual = 1.0
        for model_index in order_matrix[row]:
            scores[row, model_index] = 1.0 - residual
            residual *= 1.0 - values[row, model_index]

    ordered_scores = np.take_along_axis(scores, order_matrix, axis=1)
    if not np.all(ordered_scores[:, 0] == 0.0):
        raise RuntimeError("factorized score construction lost its zero first score")
    if np.any(np.diff(ordered_scores, axis=1) < -1e-15):
        raise RuntimeError("factorized scores are not nested along an order")
    return scores


__all__ = ["per_query_orders_factorized_scores"]
