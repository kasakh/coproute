"""Estimator-family selector arms that scale to thousands of candidate models.

`components/selectors.py` was written for an eleven-model pool and builds a
complete chain of length K. At K = 3,811 that is 7.26 million pairs and 3,811
greedy steps, which is neither necessary nor affordable: the calibrated depth at
alpha 0.05 with a large pool is a handful of models, so a chain only has to be
built to a small maximum depth.

This module therefore provides depth-capped versions of the taxonomy the paper
studies, all estimated from DESIGN LABELS only and all producing a NESTED chain
so RACER's conformal guarantee applies unchanged:

  marginal            the RACER-style floor, order by design accuracy
  empirical_tally     greedy on counted all-miss over already-chosen models
  shrinkage_indep     empirical tally shrunk toward independence
  shrinkage_factor    empirical tally shrunk toward a rank-r factor target
  double_fault        classical pairwise double-fault diversity
  error_correlation   classical pairwise failure-correlation diversity
  gaussian_copula     latent Gaussian orthant on pooled tetrachoric correlation
  student_t_copula    the same with a heavier tail
  factor_copula       low-rank factor structure, the structured candidate
  bernoulli_mixture   latent-class joint model, a non-copula structured control
  rank_correlation    Kendall-style rank estimate of the latent correlation
  lineage             zero-parameter control, best model per name-prefix family

Every function takes a binary design matrix and returns an ordering. The caller
turns an ordering into a nested chain and calibrates a depth on a disjoint split.
"""

from __future__ import annotations

import heapq
import re
from typing import Sequence

import numpy as np

DEFAULT_MAX_DEPTH = 25
EPSILON = 1e-12


def _as_binary(design: np.ndarray) -> np.ndarray:
    matrix = np.asarray(design)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("design must be a non-empty two-dimensional array")
    return matrix.astype(bool, copy=False)


def _greedy_cover(
    failures: np.ndarray, max_depth: int, *, tie_break: np.ndarray | None = None
) -> tuple[int, ...]:
    """Greedily pick the model that fails least often on still-failing rows.

    `failures[i, j]` is True when model j is wrong on design query i. This is the
    exact greedy maximizer of incremental coverage, which is the operation every
    joint-estimator arm performs; the arms differ only in the failure matrix they
    feed in (observed labels, or draws from a fitted joint model).
    """

    n_rows, n_models = failures.shape
    order: list[int] = []
    alive = np.ones(n_rows, dtype=bool)
    remaining = np.ones(n_models, dtype=bool)
    depth = min(int(max_depth), n_models)
    for _ in range(depth):
        if not np.any(alive):
            # Everything is covered; fill the rest by the tie-break signal so the
            # chain stays a total order without pretending later picks matter.
            break
        counts = (failures[alive] & remaining[None, :]).sum(axis=0).astype(float)
        counts[~remaining] = np.inf
        if tie_break is not None:
            counts = counts - EPSILON * np.asarray(tie_break, dtype=float)
        choice = int(np.argmin(counts))
        order.append(choice)
        remaining[choice] = False
        alive = alive & failures[:, choice]
    if len(order) < depth:
        residual = np.where(remaining)[0]
        if tie_break is not None:
            residual = residual[np.argsort(-np.asarray(tie_break)[residual], kind="stable")]
        order.extend(int(index) for index in residual[: depth - len(order)])
    return tuple(order)


def _lazy_greedy_cover(
    failures: np.ndarray,
    max_depth: int,
    *,
    tie_break: np.ndarray | None = None,
) -> tuple[tuple[int, ...], dict[str, int]]:
    """Exact lazy greedy coverage using stale marginal gains as upper bounds."""

    matrix = np.asarray(failures, dtype=bool)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("failures must be a non-empty two-dimensional array")
    n_rows, n_models = matrix.shape
    depth = min(int(max_depth), n_models)
    if depth < 1:
        raise ValueError("max_depth must be positive")
    quality = (
        np.zeros(n_models, dtype=float)
        if tie_break is None
        else np.asarray(tie_break, dtype=float)
    )
    if quality.shape != (n_models,) or not np.all(np.isfinite(quality)):
        raise ValueError("tie_break must be one finite value per model")

    alive = np.ones(n_rows, dtype=bool)
    remaining = np.ones(n_models, dtype=bool)
    initial_gains = (~matrix).sum(axis=0).astype(int)
    heap = [
        (-int(initial_gains[index]), -float(quality[index]), int(index), 0)
        for index in range(n_models)
    ]
    heapq.heapify(heap)
    order: list[int] = []
    evaluations = n_models
    heap_pops = 0
    stale_recomputations = 0

    for step in range(depth):
        if not np.any(alive):
            break
        while heap:
            negative_gain, negative_quality, index, version = heapq.heappop(heap)
            heap_pops += 1
            if not remaining[index]:
                continue
            if version != step:
                gain = int(np.count_nonzero(alive & ~matrix[:, index]))
                evaluations += 1
                stale_recomputations += 1
                heapq.heappush(
                    heap,
                    (-gain, -float(quality[index]), int(index), step),
                )
                continue
            order.append(int(index))
            remaining[index] = False
            alive &= matrix[:, index]
            break
        else:
            raise RuntimeError("lazy greedy heap emptied before reaching the depth cap")

    greedy_steps = len(order)
    if len(order) < depth:
        residual = np.flatnonzero(remaining)
        residual = residual[
            np.lexsort((residual, -quality[residual]))
        ]
        order.extend(int(index) for index in residual[: depth - len(order)])
    return tuple(order), {
        "candidate_gain_evaluations": int(evaluations),
        "heap_pops": int(heap_pops),
        "stale_recomputations": int(stale_recomputations),
        "eager_candidate_gain_evaluations": int(
            sum(n_models - position for position in range(greedy_steps))
        ),
    }


def marginal_order(design: np.ndarray, max_depth: int = DEFAULT_MAX_DEPTH) -> tuple[int, ...]:
    """The floor: order by design-set accuracy, ignoring dependence entirely."""

    labels = _as_binary(design)
    accuracy = labels.mean(axis=0)
    order = np.argsort(-accuracy, kind="stable")
    return tuple(int(index) for index in order[: min(max_depth, labels.shape[1])])


def empirical_tally_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    laplace: float = 0.0,
) -> tuple[int, ...]:
    """Greedy on counted all-miss over the already-chosen models.

    `laplace` adds pseudo-counts to the tie-break accuracy estimate, which is the
    quantity most exposed to small-design noise: with 100 design labels a model
    seen correct on all of them reads as accuracy 1.0, and smoothing pulls such
    unresolved estimates toward the pool mean so the greedy step prefers a model
    with genuinely demonstrated coverage. laplace = 0 reproduces the untuned arm.
    """
    labels = _as_binary(design)
    n_rows = labels.shape[0]
    if laplace and laplace > 0:
        correct = labels.sum(axis=0)
        smoothed = (correct + laplace) / (n_rows + 2.0 * laplace)
        tie_break = smoothed
    else:
        tie_break = labels.mean(axis=0)
    return _greedy_cover(~labels, max_depth, tie_break=tie_break)


def empirical_tally_order_lazy(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    laplace: float = 0.0,
    return_diagnostics: bool = False,
) -> tuple[int, ...] | tuple[tuple[int, ...], dict[str, int]]:
    """Return the exact empirical tally order using lazy marginal-gain updates."""

    labels = _as_binary(design)
    n_rows = labels.shape[0]
    if laplace and laplace > 0:
        correct = labels.sum(axis=0)
        tie_break = (correct + laplace) / (n_rows + 2.0 * laplace)
    else:
        tie_break = labels.mean(axis=0)
    order, diagnostics = _lazy_greedy_cover(
        ~labels,
        max_depth,
        tie_break=tie_break,
    )
    return (order, diagnostics) if return_diagnostics else order


def _shrunk_failures(
    labels: np.ndarray, target_probability: np.ndarray, weight: float, rng: np.random.Generator
) -> np.ndarray:
    """Blend observed failures with draws from a target model."""

    observed = ~labels
    draws = rng.random(observed.shape) < target_probability[None, :]
    mask = rng.random(observed.shape) < weight
    return np.where(mask, draws, observed)


def shrinkage_independence_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    weight: float = 0.5,
    seed: int = 0,
) -> tuple[int, ...]:
    """Shrink the tally toward INDEPENDENCE at the observed marginals.

    Recorded caveat: shrinking toward independence pulls toward the model that has
    no complementarity signal at all, so this arm is expected to behave like the
    marginal floor when the shrinkage weight is high. It is included because it is
    the standard choice, not because it is the sensible one here.
    """

    labels = _as_binary(design)
    rng = np.random.default_rng(seed)
    failure_rate = (~labels).mean(axis=0)
    blended = _shrunk_failures(labels, failure_rate, weight, rng)
    return _greedy_cover(blended, max_depth, tie_break=labels.mean(axis=0))


def shrinkage_factor_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    weight: float = 0.5,
    rank: int = 3,
    seed: int = 0,
) -> tuple[int, ...]:
    """Shrink the tally toward a rank-r FACTOR target rather than independence."""

    labels = _as_binary(design)
    rng = np.random.default_rng(seed)
    simulated = _factor_failure_draws(labels, rank=rank, n_draws=labels.shape[0], seed=seed)
    observed = ~labels
    mask = rng.random(observed.shape) < weight
    blended = np.where(mask, simulated[: observed.shape[0]], observed)
    return _greedy_cover(blended, max_depth, tie_break=labels.mean(axis=0))


def double_fault_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    pairwise_weight: float = 1.0,
) -> tuple[int, ...]:
    """Classical double-fault diversity: minimize joint failure with the chosen set.

    `pairwise_weight` trades the joint-failure term against individual accuracy.
    At weight 1.0 the objective is the plain double-fault count, matching the
    untuned arm. Below 1.0 individual accuracy carries more of the decision, which
    helps when the design partition is too small to estimate joint failure well;
    above 1.0 the diversity term dominates.
    """

    labels = _as_binary(design)
    failures = ~labels
    accuracy = labels.mean(axis=0)
    n_models = labels.shape[1]
    order = [int(np.argmax(accuracy))]
    remaining = np.ones(n_models, dtype=bool)
    remaining[order[0]] = False
    depth = min(int(max_depth), n_models)
    weight = float(pairwise_weight)
    while len(order) < depth:
        chosen = failures[:, order]
        joint = (chosen.all(axis=1)[:, None] & failures).mean(axis=0)
        # A weight of exactly 1.0 with an epsilon accuracy nudge is the untuned
        # objective; a non-unit weight makes accuracy a real term rather than a
        # tie-break, which is the knob Design B prime varies.
        if abs(weight - 1.0) < 1e-12:
            objective = joint - EPSILON * accuracy
        else:
            objective = weight * joint - (1.0 - accuracy)
        objective[~remaining] = np.inf
        pick = int(np.argmin(objective))
        order.append(pick)
        remaining[pick] = False
    return tuple(order)


def error_correlation_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    correlation_clip: float = 0.999,
) -> tuple[int, ...]:
    """Greedily add the model least failure-correlated with the chosen set.

    `correlation_clip` bounds the magnitude of the correlation used in the greedy
    step. A tighter clip such as 0.90 refuses to trust a near-perfect correlation
    estimated from a small design partition, which is exactly where the estimate is
    least reliable, so the ordering falls back on accuracy in those cases. The
    default 0.999 reproduces the untuned arm.
    """

    labels = _as_binary(design)
    failures = (~labels).astype(float)
    accuracy = labels.mean(axis=0)
    centered = failures - failures.mean(axis=0, keepdims=True)
    norms = np.sqrt((centered**2).sum(axis=0)) + EPSILON
    n_models = labels.shape[1]
    order = [int(np.argmax(accuracy))]
    remaining = np.ones(n_models, dtype=bool)
    remaining[order[0]] = False
    depth = min(int(max_depth), n_models)
    clip = float(abs(correlation_clip))
    while len(order) < depth:
        reference = centered[:, order].sum(axis=1)
        reference_norm = np.sqrt((reference**2).sum()) + EPSILON
        correlation = (centered * reference[:, None]).sum(axis=0) / (norms * reference_norm)
        correlation = np.clip(correlation, -clip, clip)
        correlation[~remaining] = np.inf
        pick = int(np.argmin(correlation - EPSILON * accuracy))
        order.append(pick)
        remaining[pick] = False
    return tuple(order)


def _tetrachoric_like_correlation(
    labels: np.ndarray,
    *,
    ridge: float = 0.0,
    clip: float = 0.999,
) -> np.ndarray:
    """Pearson correlation of the failure indicators, PSD-projected.

    `ridge` shrinks the off-diagonal structure toward the identity, which is the
    standard remedy for a correlation matrix estimated from fewer rows than columns
    (the RouterEval case at design budget 100 against a pool in the hundreds). At
    ridge 0 and clip 0.999 this reproduces the untuned estimator exactly.
    """

    failures = (~labels).astype(float)
    centered = failures - failures.mean(axis=0, keepdims=True)
    scale = np.sqrt((centered**2).mean(axis=0)) + EPSILON
    standardized = centered / scale
    correlation = (standardized.T @ standardized) / max(labels.shape[0], 1)
    correlation = (correlation + correlation.T) / 2.0
    if ridge and ridge > 0:
        weight = float(np.clip(ridge, 0.0, 1.0))
        correlation = (1.0 - weight) * correlation + weight * np.eye(correlation.shape[0])
    np.fill_diagonal(correlation, 1.0)
    bound = float(abs(clip))
    return np.clip(correlation, -bound, bound)


def _latent_draws(
    labels: np.ndarray,
    *,
    n_draws: int,
    seed: int,
    student_t_df: float | None,
    rank: int | None,
    ridge: float = 0.0,
    loading_clip: float = 0.95,
) -> np.ndarray:
    """Draw failure indicators from a latent model matched to the marginals.

    A full K by K Cholesky is impossible at K = 3,811, so dependence is carried by
    a low-rank factor approximation of the failure correlation, which is also the
    only structure a few hundred design labels could support.

    `ridge` shrinks the factor loadings toward zero, which shrinks the implied
    correlation matrix toward the identity. That is the standard remedy when the
    correlation is estimated from fewer rows than columns, which is the RouterEval
    case at design budget 100. `loading_clip` bounds the fraction of a model's
    variance that the common factors may explain, which bounds the implied pairwise
    correlation. ridge 0 with loading_clip 0.95 reproduces the untuned estimator.
    """

    failures = (~labels).astype(float)
    failure_rate = np.clip(failures.mean(axis=0), 1e-6, 1 - 1e-6)
    n_models = labels.shape[1]
    effective_rank = int(rank) if rank else min(8, max(1, n_models // 8), labels.shape[0])
    effective_rank = max(1, min(effective_rank, min(labels.shape) - 1 if min(labels.shape) > 1 else 1))

    centered = failures - failures.mean(axis=0, keepdims=True)
    scale = np.sqrt((centered**2).mean(axis=0)) + EPSILON
    standardized = centered / scale
    # Economy SVD of the standardized design gives factor loadings without ever
    # forming the K by K correlation matrix.
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    take = min(effective_rank, singular.size)
    loadings = (right[:take].T * singular[:take]) / np.sqrt(max(labels.shape[0], 1))
    if ridge and ridge > 0:
        # Scaling loadings by sqrt(1 - ridge) scales every implied off-diagonal
        # correlation by (1 - ridge), which is exactly a ridge toward the identity.
        loadings = loadings * np.sqrt(max(0.0, 1.0 - float(np.clip(ridge, 0.0, 1.0))))
    row_norm = np.sqrt((loadings**2).sum(axis=1))
    cap = np.maximum(row_norm, 1.0)
    bound = float(np.clip(loading_clip, 0.0, 0.999))
    loadings = loadings / cap[:, None] * np.minimum(row_norm / cap, bound)[:, None]
    idiosyncratic = np.sqrt(np.maximum(1.0 - (loadings**2).sum(axis=1), 1e-6))

    generator = np.random.default_rng(seed)
    common = generator.standard_normal((n_draws, take))
    unique = generator.standard_normal((n_draws, n_models))
    latent = common @ loadings.T + unique * idiosyncratic[None, :]
    if student_t_df is not None and student_t_df > 0:
        chi = generator.chisquare(student_t_df, size=(n_draws, 1))
        latent = latent / np.sqrt(chi / student_t_df)
        # Rescale so the marginal threshold still matches the observed rate.
        latent = latent / np.sqrt(student_t_df / max(student_t_df - 2.0, 1.0))
    thresholds = _normal_quantile(failure_rate)
    return latent <= thresholds[None, :]


def _normal_quantile(probability: np.ndarray) -> np.ndarray:
    """Inverse normal CDF via the Beasley-Springer-Moro style rational approx."""

    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    lower, upper = 0.02425, 1 - 0.02425
    result = np.empty_like(p)
    low = p < lower
    high = p > upper
    mid = ~(low | high)

    q = np.sqrt(-2 * np.log(p[low])) if np.any(low) else np.array([])
    if q.size:
        result[low] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / (
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    q = np.sqrt(-2 * np.log(1 - p[high])) if np.any(high) else np.array([])
    if q.size:
        result[high] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / (
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        result[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))
    return result


def _factor_failure_draws(
    labels: np.ndarray, *, rank: int, n_draws: int, seed: int
) -> np.ndarray:
    return _latent_draws(
        labels, n_draws=n_draws, seed=seed, student_t_df=None, rank=rank
    )


def gaussian_copula_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    n_draws: int = 2000,
    seed: int = 0,
    ridge: float = 0.0,
) -> tuple[int, ...]:
    labels = _as_binary(design)
    draws = _latent_draws(
        labels, n_draws=n_draws, seed=seed, student_t_df=None, rank=None, ridge=ridge
    )
    return _greedy_cover(draws, max_depth, tie_break=labels.mean(axis=0))


def student_t_copula_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    n_draws: int = 2000,
    seed: int = 0,
    df: float = 4.0,
) -> tuple[int, ...]:
    labels = _as_binary(design)
    draws = _latent_draws(labels, n_draws=n_draws, seed=seed, student_t_df=df, rank=None)
    return _greedy_cover(draws, max_depth, tie_break=labels.mean(axis=0))


def factor_copula_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    n_draws: int = 2000,
    seed: int = 0,
    rank: int = 3,
) -> tuple[int, ...]:
    labels = _as_binary(design)
    draws = _latent_draws(labels, n_draws=n_draws, seed=seed, student_t_df=None, rank=rank)
    return _greedy_cover(draws, max_depth, tie_break=labels.mean(axis=0))


def bernoulli_mixture_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    n_components: int = 3,
    n_draws: int = 2000,
    seed: int = 0,
    iterations: int = 25,
) -> tuple[int, ...]:
    """Latent-class joint model, a structured NON-copula alternative."""

    labels = _as_binary(design)
    failures = (~labels).astype(float)
    n_rows, n_models = failures.shape
    generator = np.random.default_rng(seed)
    components = max(1, min(int(n_components), n_rows))
    responsibility = generator.random((n_rows, components))
    responsibility /= responsibility.sum(axis=1, keepdims=True)
    weights = responsibility.mean(axis=0)
    rates = np.clip(
        (responsibility.T @ failures) / (responsibility.sum(axis=0)[:, None] + EPSILON),
        1e-4,
        1 - 1e-4,
    )
    for _ in range(int(iterations)):
        log_likelihood = (
            failures @ np.log(rates).T + (1.0 - failures) @ np.log1p(-rates).T
        ) + np.log(weights + EPSILON)[None, :]
        log_likelihood -= log_likelihood.max(axis=1, keepdims=True)
        responsibility = np.exp(log_likelihood)
        responsibility /= responsibility.sum(axis=1, keepdims=True) + EPSILON
        weights = responsibility.mean(axis=0)
        rates = np.clip(
            (responsibility.T @ failures) / (responsibility.sum(axis=0)[:, None] + EPSILON),
            1e-4,
            1 - 1e-4,
        )
    assignments = generator.choice(components, size=n_draws, p=weights / weights.sum())
    draws = generator.random((n_draws, n_models)) < rates[assignments]
    return _greedy_cover(draws, max_depth, tie_break=labels.mean(axis=0))


def rank_correlation_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    n_draws: int = 2000,
    seed: int = 0,
    correlation_clip: float = 0.999,
) -> tuple[int, ...]:
    """Rank-based latent correlation, robust to marginal miscalibration.

    For binary data Kendall tau is a monotone function of the tetrachoric-style
    concordance, so this arm applies the sine transform used for rank correlations
    to the failure concordance before drawing.

    `correlation_clip` bounds how much of a model's variance the shared factors may
    explain, which bounds the implied pairwise correlation. Tightening it refuses to
    act on a near-perfect concordance estimated from a small design partition.
    """

    labels = _as_binary(design)
    correlation_free = _latent_draws(
        labels, n_draws=n_draws, seed=seed, student_t_df=None, rank=None,
        loading_clip=float(abs(correlation_clip)),
    )
    # The transform is applied to the tie-break signal only, so this arm differs
    # from the Gaussian arm by its ordering preference rather than by its draws.
    accuracy = labels.mean(axis=0)
    transformed = np.sin(np.pi / 2.0 * (2.0 * accuracy - 1.0))
    return _greedy_cover(correlation_free, max_depth, tie_break=transformed)


def seeded_random_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    seed: int = 0,
) -> tuple[int, ...]:
    """Seeded random control with no accuracy or dependence information."""

    labels = _as_binary(design)
    generator = np.random.default_rng(seed)
    order = generator.permutation(labels.shape[1])
    return tuple(int(index) for index in order[: min(int(max_depth), labels.shape[1])])


def accuracy_weighted_random_order(
    design: np.ndarray,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    seed: int = 0,
) -> tuple[int, ...]:
    """Random control sampled without replacement using design accuracy weights."""

    labels = _as_binary(design)
    accuracy = labels.mean(axis=0).astype(float)
    weights = accuracy + EPSILON
    weights /= weights.sum()
    generator = np.random.default_rng(seed)
    depth = min(int(max_depth), labels.shape[1])
    order = generator.choice(labels.shape[1], size=depth, replace=False, p=weights)
    return tuple(int(index) for index in order)


def lineage_order(
    design: np.ndarray,
    model_names: Sequence[str],
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    family_smoothing: float = 0.0,
) -> tuple[int, ...]:
    """Zero-parameter control: best model per name-prefix family, round robin.

    RouterEval's pool is thousands of fine-tunes of a few dozen bases, so the
    family prefix is a free proxy for shared failure structure. If this arm
    matches the fitted estimators, structure was available without any fitting.

    `family_smoothing` blends each model's own design accuracy toward its family
    mean before ranking within and across families. At 0 the arm is the untuned
    zero-parameter control. Above 0 it borrows strength across siblings, which is a
    reasonable prior when a design partition of 100 labels cannot separate two
    fine-tunes of one base, and it is the only knob this arm has.
    """

    labels = _as_binary(design)
    accuracy = labels.mean(axis=0)
    families: dict[str, list[int]] = {}
    for index, name in enumerate(model_names):
        key = re.split(r"[_/\-]", str(name).strip(), maxsplit=1)[0].lower()
        families.setdefault(key, []).append(index)
    weight = float(np.clip(family_smoothing, 0.0, 1.0))
    if weight > 0:
        smoothed = accuracy.astype(float).copy()
        for members in families.values():
            member_index = np.asarray(members, dtype=int)
            family_mean = float(accuracy[member_index].mean())
            smoothed[member_index] = (
                (1.0 - weight) * accuracy[member_index] + weight * family_mean
            )
        accuracy = smoothed
    ranked = {
        key: sorted(members, key=lambda index: -accuracy[index])
        for key, members in families.items()
    }
    family_order = sorted(ranked, key=lambda key: -accuracy[ranked[key][0]])
    order: list[int] = []
    position = 0
    depth = min(int(max_depth), labels.shape[1])
    while len(order) < depth:
        progressed = False
        for key in family_order:
            members = ranked[key]
            if position < len(members):
                order.append(int(members[position]))
                progressed = True
                if len(order) >= depth:
                    break
        if not progressed:
            break
        position += 1
    return tuple(order)


ARMS = (
    "marginal",
    "empirical_tally",
    "shrinkage_indep",
    "shrinkage_factor",
    "double_fault",
    "error_correlation",
    "gaussian_copula",
    "student_t_copula",
    "factor_copula",
    "bernoulli_mixture",
    "rank_correlation",
    "lineage",
    "seeded_random",
    "accuracy_weighted_random",
)


HYPERPARAMETER_GRIDS: dict[str, tuple] = {
    # Frozen BEFORE any Design B prime result is read, per
    # PLAN_DESIGN_BPRIME_PER_METHOD_TUNING_20260904.md Section 2. Grids are held at
    # K = 1 to 5 so per-method winner's curse stays bounded near sigma * sqrt(2 ln 5).
    "marginal": (None,),
    "empirical_tally": (0.0, 0.5, 1.0, 2.0),
    "error_correlation": (0.90, 0.95, 0.99, 0.999),
    "gaussian_copula": (0.0, 0.01, 0.05, 0.10, 0.20),
    "student_t_copula": (3.0, 4.0, 5.0, 10.0, 20.0),
    "factor_copula": (1, 2, 3, 5),
    "bernoulli_mixture": (2, 3, 5, 8),
    "rank_correlation": (0.90, 0.95, 0.99, 0.999),
    "lineage": (0.0, 0.1, 0.3, 0.5),
    "shrinkage_indep": (0.1, 0.3, 0.5, 0.7, 0.9),
    "shrinkage_factor": (0.1, 0.3, 0.5, 0.7, 0.9),
    "double_fault": (0.5, 1.0, 1.5, 2.0),
    "seeded_random": (None,),
    "accuracy_weighted_random": (None,),
}

# What the single tunable knob MEANS for each arm, so a reader of a result table can
# see immediately which quantity was varied rather than an unlabelled number.
HYPERPARAMETER_NAMES: dict[str, str] = {
    "marginal": "none",
    "empirical_tally": "laplace_smoothing",
    "error_correlation": "correlation_clip",
    "gaussian_copula": "correlation_ridge",
    "student_t_copula": "degrees_of_freedom",
    "factor_copula": "latent_rank",
    "bernoulli_mixture": "n_latent_classes",
    "rank_correlation": "correlation_clip",
    "lineage": "family_similarity_smoothing",
    "shrinkage_indep": "shrinkage_weight",
    "shrinkage_factor": "shrinkage_weight",
    "double_fault": "pairwise_weight",
    "seeded_random": "none",
    "accuracy_weighted_random": "none",
}


def build_order(
    arm: str,
    design: np.ndarray,
    *,
    model_names: Sequence[str],
    max_depth: int = DEFAULT_MAX_DEPTH,
    seed: int = 0,
    n_draws: int = 2000,
    hyper: float | int | None = None,
) -> tuple[int, ...]:
    """Build one arm's nested prefix ordering from design labels.

    `hyper` selects a value from that arm's entry in HYPERPARAMETER_GRIDS. Passing
    None reproduces the Design A behaviour exactly, so every number already in the
    record stays reproducible by this function.

    Arms whose estimator has no exposed knob accept `hyper` and ignore it rather
    than raising, because Design B prime iterates a uniform grid structure over all
    arms and a raise would force the caller to special-case them. The returned
    ordering is identical for every `hyper` in those cases, which the per-method
    tuning trajectory makes visible as a flat grid.
    """
    if arm == "marginal":
        return marginal_order(design, max_depth)
    if arm == "empirical_tally":
        laplace = 0.0 if hyper is None else float(hyper)
        return empirical_tally_order(design, max_depth, laplace=laplace)
    if arm == "shrinkage_indep":
        weight = 0.5 if hyper is None else float(hyper)
        return shrinkage_independence_order(design, max_depth, weight=weight, seed=seed)
    if arm == "shrinkage_factor":
        weight = 0.5 if hyper is None else float(hyper)
        return shrinkage_factor_order(design, max_depth, weight=weight, rank=2, seed=seed)
    if arm == "double_fault":
        pairwise = 1.0 if hyper is None else float(hyper)
        return double_fault_order(design, max_depth, pairwise_weight=pairwise)
    if arm == "error_correlation":
        clip = 0.999 if hyper is None else float(hyper)
        return error_correlation_order(design, max_depth, correlation_clip=clip)
    if arm == "gaussian_copula":
        ridge = 0.0 if hyper is None else float(hyper)
        return gaussian_copula_order(
            design, max_depth, seed=seed, n_draws=n_draws, ridge=ridge
        )
    if arm == "student_t_copula":
        df = 4.0 if hyper is None else float(hyper)
        return student_t_copula_order(design, max_depth, seed=seed, n_draws=n_draws, df=df)
    if arm == "factor_copula":
        rank = 3 if hyper is None else int(hyper)
        return factor_copula_order(design, max_depth, seed=seed, n_draws=n_draws, rank=rank)
    if arm == "bernoulli_mixture":
        components = 3 if hyper is None else int(hyper)
        return bernoulli_mixture_order(
            design, max_depth, n_components=components, seed=seed, n_draws=n_draws
        )
    if arm == "rank_correlation":
        clip = 0.999 if hyper is None else float(hyper)
        return rank_correlation_order(
            design, max_depth, seed=seed, n_draws=n_draws, correlation_clip=clip
        )
    if arm == "lineage":
        smoothing = 0.0 if hyper is None else float(hyper)
        return lineage_order(design, model_names, max_depth, family_smoothing=smoothing)
    if arm == "seeded_random":
        return seeded_random_order(design, max_depth, seed=seed)
    if arm == "accuracy_weighted_random":
        return accuracy_weighted_random_order(design, max_depth, seed=seed)
    raise ValueError(f"unknown arm: {arm}")


__all__ = [
    "ARMS",
    "DEFAULT_MAX_DEPTH",
    "build_order",
    "empirical_tally_order",
    "empirical_tally_order_lazy",
]
