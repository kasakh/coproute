"""Dependence discounted stopping scores for one frozen model order.

The design split supplies one chain conditional per model position.  Their
cumulative all miss rate is compared with the product of design marginal miss
rates.  The resulting prefix adjustment is applied to each query's router
product.  A minimum survivor count controls shrinkage toward the pure product.
The value zero is reserved for the exact pure product endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ChainDiscountProfile:
    """Design statistics for one complete fixed order."""

    order: tuple[int, ...]
    survivor_counts: np.ndarray
    chain_conditional_miss: np.ndarray
    empirical_prefix_miss: np.ndarray
    independent_prefix_miss: np.ndarray
    raw_discount: np.ndarray


def _binary_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError(f"{name} must contain only binary values")
    return matrix.astype(np.int8, copy=False)


def _probability_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("probabilities must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("probabilities must be finite")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return matrix


def _full_order(order: Sequence[int], n_models: int) -> np.ndarray:
    values = np.asarray(order)
    if values.ndim != 1 or values.size != n_models:
        raise ValueError("fixed_order must be a full model permutation")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError("fixed_order must contain integer model indices")
    values = values.astype(int, copy=False)
    if not np.array_equal(np.sort(values), np.arange(n_models)):
        raise ValueError("fixed_order must be a full model permutation")
    return values


def _minimum_survivor_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("minimum_survivor_count must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError("minimum_survivor_count must be a non-negative integer")
    return value


def fit_chain_discount(
    design_labels: np.ndarray,
    fixed_order: Sequence[int],
) -> ChainDiscountProfile:
    """Count chain conditionals and prefix product adjustments on design rows.

    Correctness labels use one for correct and zero for missed.  Prefix index
    zero is the empty prefix.  If no design row survives to a chain position,
    its conditional miss estimate is recorded as zero and the survivor count
    records that the estimate has no support.
    """

    labels = _binary_matrix(design_labels, "design_labels")
    n_rows, n_models = labels.shape
    order = _full_order(fixed_order, n_models)
    misses = 1 - labels[:, order]

    survivor_counts = np.empty(n_models + 1, dtype=np.int64)
    empirical_prefix_miss = np.empty(n_models + 1, dtype=float)
    chain_conditional_miss = np.empty(n_models, dtype=float)
    surviving = np.ones(n_rows, dtype=bool)
    survivor_counts[0] = n_rows
    empirical_prefix_miss[0] = 1.0
    for position in range(n_models):
        before = int(np.count_nonzero(surviving))
        surviving &= misses[:, position].astype(bool)
        after = int(np.count_nonzero(surviving))
        survivor_counts[position + 1] = after
        chain_conditional_miss[position] = after / before if before else 0.0
        empirical_prefix_miss[position + 1] = after / n_rows

    marginal_miss = np.mean(1 - labels, axis=0)
    ordered_marginal_miss = marginal_miss[order]
    independent_prefix_miss = np.concatenate(
        (np.ones(1, dtype=float), np.cumprod(ordered_marginal_miss, dtype=float))
    )
    raw_discount = np.ones(n_models + 1, dtype=float)
    supported = independent_prefix_miss > 0.0
    raw_discount[supported] = (
        empirical_prefix_miss[supported] / independent_prefix_miss[supported]
    )

    return ChainDiscountProfile(
        order=tuple(int(index) for index in order),
        survivor_counts=survivor_counts,
        chain_conditional_miss=chain_conditional_miss,
        empirical_prefix_miss=empirical_prefix_miss,
        independent_prefix_miss=independent_prefix_miss,
        raw_discount=raw_discount,
    )


def effective_chain_discount(
    profile: ChainDiscountProfile,
    *,
    minimum_survivor_count: int,
) -> np.ndarray:
    """Shrink a chain adjustment toward one when design support is thin.

    Positive settings use linear count shrinkage.  A prefix with at least the
    requested number of surviving design rows receives the full adjustment.
    A prefix with fewer rows receives that count divided by the requested count.
    Zero disables the adjustment exactly and is the required independence
    product endpoint.
    """

    minimum = _minimum_survivor_count(minimum_survivor_count)
    if minimum == 0:
        return np.ones_like(profile.raw_discount, dtype=float)
    reliability = np.minimum(
        1.0, np.asarray(profile.survivor_counts, dtype=float) / float(minimum)
    )
    return 1.0 + reliability * (np.asarray(profile.raw_discount) - 1.0)


def chain_discounted_scores(
    probabilities: np.ndarray,
    profile: ChainDiscountProfile,
    *,
    minimum_survivor_count: int,
) -> np.ndarray:
    """Return nested scores for one frozen order and chain adjustment.

    The score assigned to the model at position k is one minus the adjusted
    router all miss probability of the preceding prefix.  Adjusted residuals
    are clipped to [0, 1] and replaced by their cumulative minimum.  Therefore
    scores are nondecreasing along the order and every score sublevel set is an
    exact prefix.
    """

    values = _probability_matrix(probabilities)
    n_queries, n_models = values.shape
    order = _full_order(profile.order, n_models)
    discount = effective_chain_discount(
        profile, minimum_survivor_count=minimum_survivor_count
    )
    if discount.shape != (n_models + 1,) or not np.all(np.isfinite(discount)):
        raise ValueError("chain discount profile has invalid dimensions or values")

    ordered_miss = 1.0 - values[:, order]
    router_prefix_miss = np.concatenate(
        (
            np.ones((n_queries, 1), dtype=float),
            np.cumprod(ordered_miss, axis=1, dtype=float),
        ),
        axis=1,
    )
    adjusted_residual = np.clip(router_prefix_miss * discount[None, :], 0.0, 1.0)
    adjusted_residual = np.minimum.accumulate(adjusted_residual, axis=1)

    scores = np.empty_like(values, dtype=float)
    for position, model_index in enumerate(order):
        scores[:, model_index] = 1.0 - adjusted_residual[:, position]

    ordered_scores = scores[:, order]
    if not np.all(ordered_scores[:, 0] == 0.0):
        raise RuntimeError("chain discounted score lost its zero first score")
    if np.any(np.diff(ordered_scores, axis=1) < -1e-15):
        raise RuntimeError("chain discounted scores are not nested along the order")
    return scores


def chain_discount_diagnostics(
    profile: ChainDiscountProfile,
    *,
    minimum_survivor_count: int,
) -> dict[str, object]:
    """Return JSON-safe design and shrinkage diagnostics for one candidate."""

    minimum = _minimum_survivor_count(minimum_survivor_count)
    effective = effective_chain_discount(
        profile, minimum_survivor_count=minimum
    )
    if minimum == 0:
        engaged = np.zeros_like(profile.survivor_counts, dtype=bool)
    else:
        engaged = profile.survivor_counts < minimum
    return {
        "minimum_survivor_count": minimum,
        "discount_fully_off": minimum == 0,
        "order": list(profile.order),
        "survivor_counts": [int(value) for value in profile.survivor_counts],
        "chain_conditional_miss": [
            float(value) for value in profile.chain_conditional_miss
        ],
        "empirical_prefix_miss": [
            float(value) for value in profile.empirical_prefix_miss
        ],
        "independent_prefix_miss": [
            float(value) for value in profile.independent_prefix_miss
        ],
        "raw_discount": [float(value) for value in profile.raw_discount],
        "effective_discount": [float(value) for value in effective],
        "shrinkage_engaged_by_prefix": [bool(value) for value in engaged],
        "shrinkage_engaged_prefix_count": int(np.count_nonzero(engaged)),
    }


__all__ = [
    "ChainDiscountProfile",
    "chain_discount_diagnostics",
    "chain_discounted_scores",
    "effective_chain_discount",
    "fit_chain_discount",
]
