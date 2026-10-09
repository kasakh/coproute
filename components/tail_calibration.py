"""Rare-tail calibration diagnostics for model co-failure probabilities.

Rows in a miss matrix are queries and columns are models. A value of one means
that the model missed the query. The routines report empirical all-miss rates,
the product-of-marginals independence estimate, supplied Gaussian and Student t
predictions, and exact binomial uncertainty.

Clopper-Pearson intervals invert beta distributions. SciPy is imported lazily so
other diagnostics remain usable without it. No approximate interval is silently
substituted when SciPy is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import inf, isfinite, prod
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


Subset = Tuple[int, ...]


def _as_binary_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("misses must be a non-empty two-dimensional array")
    try:
        finite = np.isfinite(matrix.astype(float))
    except (TypeError, ValueError) as exc:
        raise ValueError("misses must contain only zero and one") from exc
    if not np.all(finite) or not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("misses must contain only zero and one")
    return matrix.astype(bool, copy=False)


def _normalize_subset(subset: Iterable[int], n_models: int) -> Subset:
    normalized = tuple(sorted(int(index) for index in subset))
    if not normalized:
        raise ValueError("subsets must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("subsets must not contain duplicate model indices")
    if any(index < 0 or index >= n_models for index in normalized):
        raise IndexError("subset contains an out-of-range model index")
    return normalized


def _normalize_subsets(
    subsets: Iterable[Iterable[int]], n_models: int
) -> Tuple[Subset, ...]:
    normalized = tuple(_normalize_subset(subset, n_models) for subset in subsets)
    if not normalized:
        raise ValueError("at least one subset is required")
    if len(normalized) != len(set(normalized)):
        raise ValueError("subsets must be unique")
    return normalized


def _validate_probability(value: Any, name: str) -> float:
    probability = float(value)
    if not isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return probability


def _beta_quantile(probability: float, shape_a: float, shape_b: float) -> float:
    """Evaluate a beta quantile or fail instead of using an approximation."""

    try:
        from scipy.stats import beta as beta_distribution
    except ImportError as exc:
        raise ImportError(
            "Exact Clopper-Pearson intervals require scipy.stats.beta.ppf; "
            "install SciPy or omit interval-producing diagnostics. No normal "
            "or Wilson approximation is used."
        ) from exc
    value = float(beta_distribution.ppf(probability, shape_a, shape_b))
    if not isfinite(value):
        raise RuntimeError("scipy.stats.beta.ppf returned a non-finite quantile")
    return value


def clopper_pearson_interval(
    event_count: int,
    n_trials: int,
    *,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Return the exact equal-tailed Clopper-Pearson binomial interval."""

    if isinstance(event_count, bool) or int(event_count) != event_count:
        raise ValueError("event_count must be an integer")
    if isinstance(n_trials, bool) or int(n_trials) != n_trials:
        raise ValueError("n_trials must be an integer")
    event_count = int(event_count)
    n_trials = int(n_trials)
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if event_count < 0 or event_count > n_trials:
        raise ValueError("event_count must lie between zero and n_trials")
    if not isfinite(float(confidence_level)) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    alpha_over_two = (1.0 - float(confidence_level)) / 2.0
    lower = (
        0.0
        if event_count == 0
        else _beta_quantile(
            alpha_over_two,
            event_count,
            n_trials - event_count + 1,
        )
    )
    upper = (
        1.0
        if event_count == n_trials
        else _beta_quantile(
            1.0 - alpha_over_two,
            event_count + 1,
            n_trials - event_count,
        )
    )
    return lower, upper


def simultaneous_clopper_pearson_intervals(
    event_counts: Sequence[int],
    n_trials: int,
    *,
    family_confidence_level: float = 0.95,
) -> Tuple[Tuple[float, float], ...]:
    """Return Bonferroni simultaneous exact intervals for one family.

    The intervals have family-wise coverage of at least
    ``family_confidence_level`` under arbitrary dependence among the reported
    subsets.
    """

    counts = tuple(event_counts)
    if not counts:
        raise ValueError("event_counts must be non-empty")
    if not isfinite(float(family_confidence_level)) or not (
        0.0 < family_confidence_level < 1.0
    ):
        raise ValueError(
            "family_confidence_level must lie strictly between zero and one"
        )
    family_alpha = 1.0 - float(family_confidence_level)
    marginal_confidence = 1.0 - family_alpha / len(counts)
    return tuple(
        clopper_pearson_interval(
            count,
            n_trials,
            confidence_level=marginal_confidence,
        )
        for count in counts
    )


@dataclass(frozen=True)
class EmpiricalTailEstimate:
    """Empirical all-miss estimate for one model subset."""

    subset: Subset
    all_miss_count: int
    n_queries: int
    probability: float


@dataclass(frozen=True)
class RareTailCalibrationRow:
    """One row of a rare-tail calibration table."""

    subset: Subset
    subset_size: int
    n_queries: int
    all_miss_count: int
    empirical_probability: float
    interval_lower: float
    interval_upper: float
    marginal_confidence_level: float
    independence_probability: float
    gaussian_probability: float
    student_t_probability: float
    independence_underpricing_ratio: float
    gaussian_underpricing_ratio: float
    student_t_underpricing_ratio: float


def empirical_all_miss_by_subset(
    misses: Any,
    subsets: Iterable[Iterable[int]],
) -> Dict[Subset, EmpiricalTailEstimate]:
    """Estimate held-out all-miss probabilities for each requested subset."""

    matrix = _as_binary_matrix(misses)
    normalized = _normalize_subsets(subsets, matrix.shape[1])
    estimates: Dict[Subset, EmpiricalTailEstimate] = {}
    for subset in normalized:
        count = int(np.sum(np.all(matrix[:, subset], axis=1)))
        estimates[subset] = EmpiricalTailEstimate(
            subset=subset,
            all_miss_count=count,
            n_queries=int(matrix.shape[0]),
            probability=count / matrix.shape[0],
        )
    return estimates


def independence_all_miss_by_subset(
    misses: Any,
    subsets: Iterable[Iterable[int]],
) -> Dict[Subset, float]:
    """Return product-of-empirical-marginals all-miss estimates by subset."""

    matrix = _as_binary_matrix(misses)
    normalized = _normalize_subsets(subsets, matrix.shape[1])
    marginal_miss_rates = np.mean(matrix, axis=0)
    return {
        subset: float(prod(float(marginal_miss_rates[index]) for index in subset))
        for subset in normalized
    }


def underpricing_ratio(empirical: float, predicted: float) -> float:
    """Return empirical probability divided by predicted probability.

    A ratio above one means that a model underprices co-failure. If both values
    are zero, the ratio is one by calibration convention. Positive empirical
    mass against a zero prediction has infinite underpricing.
    """

    empirical = _validate_probability(empirical, "empirical")
    predicted = _validate_probability(predicted, "predicted")
    if predicted == 0.0:
        return 1.0 if empirical == 0.0 else inf
    return empirical / predicted


def _prediction_for_subset(
    predictions: Mapping[Subset, float],
    subset: Subset,
    name: str,
) -> float:
    if subset not in predictions:
        raise KeyError(f"{name} is missing subset {subset}")
    return _validate_probability(predictions[subset], name)


def _table_subsets(
    n_models: int,
    subset_sizes: Iterable[int],
    selected_subsets: Iterable[Iterable[int]],
) -> Tuple[Subset, ...]:
    generated = []
    for raw_size in subset_sizes:
        if isinstance(raw_size, bool) or int(raw_size) != raw_size:
            raise ValueError("subset sizes must be integers")
        size = int(raw_size)
        if size <= 0 or size > n_models:
            raise ValueError("subset sizes must lie between one and n_models")
        generated.extend(combinations(range(n_models), size))
    generated.extend(
        _normalize_subset(subset, n_models) for subset in selected_subsets
    )
    unique = tuple(sorted(set(generated), key=lambda subset: (len(subset), subset)))
    if not unique:
        raise ValueError("provide at least one subset size or selected subset")
    return unique


def rare_tail_calibration_table(
    misses: Any,
    *,
    gaussian_predictions: Mapping[Subset, float],
    student_t_predictions: Mapping[Subset, float],
    subset_sizes: Iterable[int] = (),
    selected_subsets: Iterable[Iterable[int]] = (),
    family_confidence_level: float = 0.95,
) -> Tuple[RareTailCalibrationRow, ...]:
    """Build a simultaneous rare-tail calibration table.

    ``subset_sizes`` includes every combination of each requested size.
    ``selected_subsets`` adds named combinations of interest. Duplicate subsets
    across those two sources are reported once. Gaussian and Student t inputs
    are frozen predicted all-miss probabilities keyed by normalized index tuple.
    Bonferroni-adjusted Clopper-Pearson intervals protect the complete table as
    one family under arbitrary dependence between subset statistics.
    """

    matrix = _as_binary_matrix(misses)
    subsets = _table_subsets(
        matrix.shape[1], subset_sizes, selected_subsets
    )
    empirical = empirical_all_miss_by_subset(matrix, subsets)
    independence = independence_all_miss_by_subset(matrix, subsets)
    intervals = simultaneous_clopper_pearson_intervals(
        [empirical[subset].all_miss_count for subset in subsets],
        matrix.shape[0],
        family_confidence_level=family_confidence_level,
    )
    marginal_confidence = 1.0 - (
        1.0 - float(family_confidence_level)
    ) / len(subsets)

    rows = []
    for subset, (lower, upper) in zip(subsets, intervals):
        estimate = empirical[subset]
        gaussian = _prediction_for_subset(
            gaussian_predictions, subset, "gaussian_predictions"
        )
        student_t = _prediction_for_subset(
            student_t_predictions, subset, "student_t_predictions"
        )
        independent = independence[subset]
        rows.append(
            RareTailCalibrationRow(
                subset=subset,
                subset_size=len(subset),
                n_queries=estimate.n_queries,
                all_miss_count=estimate.all_miss_count,
                empirical_probability=estimate.probability,
                interval_lower=lower,
                interval_upper=upper,
                marginal_confidence_level=marginal_confidence,
                independence_probability=independent,
                gaussian_probability=gaussian,
                student_t_probability=student_t,
                independence_underpricing_ratio=underpricing_ratio(
                    estimate.probability, independent
                ),
                gaussian_underpricing_ratio=underpricing_ratio(
                    estimate.probability, gaussian
                ),
                student_t_underpricing_ratio=underpricing_ratio(
                    estimate.probability, student_t
                ),
            )
        )
    return tuple(rows)
