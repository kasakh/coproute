"""E2: a dependence-aware score that lives INSIDE RACER's policy class.

Every arm in this program so far committed one integer depth per task, which is
not RACER's policy family: RACER thresholds per query and can abstain, and that
is how it reached 1.657 calls on HellaSwag where our fixed-depth arms needed 2.0.
Until a dependence-aware ordering is expressed as a per-model SCORE that RACER's
own machinery can threshold, no result here can be called an improvement to
RACER.

The construction, fixed in ISSUE-31 at 2026-09-03T14:04Z before any run:

    for query x, greedily order the models using the frozen dependence fit and
    the query's router probabilities, giving m_1, m_2, ...
    let q_j(x) be the working residual score after the first j picks, q_0 = 1
    score s(x, m_k) = 1 - q_{k-1}(x)

Under exact factorization q_j is an all-miss probability. For the pairwise and
correlation arms it is only a bounded ranking heuristic and need not be the
probability of any coherent joint distribution.

Three properties make this admissible under RACER Theorem 4.3, all asserted in
tests rather than argued in prose:

  1. s(x, m_1) = 0 always, so the top pick is in every non-empty set, exactly as
     RACER-G's argmax has gap zero.
  2. q_j is non-increasing in j, so s is non-decreasing along the order and the
     sublevel set {m : s <= lambda} is exactly a PREFIX. The family is nested in
     lambda, which is the only structural requirement of the theorem.
  3. q_j depends on the query's own marginals, so at ONE global lambda the prefix
     length varies by query. That is the property fixed-depth arms lack.

Under exact factorization q_j is the product of individual miss probabilities and the
greedy order is descending marginal accuracy, so the construction degenerates to
a marginal-ordering policy. That degeneracy is a sanity check, not a claim of
strict containment of RACER.

RACER's own null column, nonconformity map, smoothing and CRC scan are untouched,
so any change in abstention is attributable to the real-model scores alone.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

EPSILON = 1e-12
FACTORIZED_INDEPENDENCE = "factorized_independence"
PAIRWISE_HEURISTIC = "pairwise_heuristic"
CORRELATION_HEURISTIC = "correlation_heuristic"
REGULARIZED_CORRELATION = "regularized_correlation"
QUERY_LOCAL_EMPIRICAL_COVERAGE = "query_local_empirical_coverage"
FIXED_TALLY_FACTORIZED = "fixed_tally_factorized"
FIXED_DESIGN_MARGINAL_FACTORIZED = "fixed_design_marginal_factorized"
LOCAL_ORDER_FACTORIZED = "local_order_factorized"

DependenceModel = Literal[
    "factorized_independence",
    "pairwise_heuristic",
    "correlation_heuristic",
    "regularized_correlation",
    "query_local_empirical_coverage",
    "fixed_tally_factorized",
    "fixed_design_marginal_factorized",
    "local_order_factorized",
]

_READ_ALIASES = {
    "independence": FACTORIZED_INDEPENDENCE,
    "pairwise_count": PAIRWISE_HEURISTIC,
    "error_correlation": CORRELATION_HEURISTIC,
    "gaussian_copula": REGULARIZED_CORRELATION,
}


def canonical_dependence_name(name: str) -> str:
    """Return the public query-level arm name, accepting old result aliases."""

    value = str(name)
    return _READ_ALIASES.get(value, value)


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("probabilities must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return values


def _pairwise_joint_miss(design: np.ndarray, *, laplace_smoothing: float = 0.0) -> np.ndarray:
    """Observed pairwise co-failure rate used by a ranking heuristic.

    The downstream averaging rule is not asserted to define a coherent joint
    Bernoulli distribution for sets larger than two.
    """

    failures = (~np.asarray(design, dtype=bool)).astype(float)
    n_rows = max(failures.shape[0], 1)
    counts = failures.T @ failures
    if laplace_smoothing > 0.0:
        # Additive smoothing: pretend we saw `laplace_smoothing` prior co-failures per pair.
        counts = counts + float(laplace_smoothing)
        n_rows = n_rows + float(laplace_smoothing)
    return counts / n_rows


def _failure_correlation(design: np.ndarray, *, clip_magnitude: float = 0.999, ridge: float = 0.0) -> np.ndarray:
    """Pooled failure correlation used as a heuristic adjustment matrix."""
    failures = (~np.asarray(design, dtype=bool)).astype(float)
    centered = failures - failures.mean(axis=0, keepdims=True)
    scale = np.sqrt((centered**2).mean(axis=0)) + EPSILON
    standardized = centered / scale
    correlation = (standardized.T @ standardized) / max(failures.shape[0], 1)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    if ridge > 0.0:
        # Convex mix toward identity: (1-r) * corr + r * I. Pulls off-diagonals toward 0.
        r = float(np.clip(ridge, 0.0, 1.0))
        eye = np.eye(correlation.shape[0])
        correlation = (1.0 - r) * correlation + r * eye
    clip = float(np.clip(clip_magnitude, 0.0, 0.999999))
    return np.clip(correlation, -clip, clip)


def _conditional_miss_factor(
    model_index: int,
    selected: list[int],
    *,
    miss_probability: np.ndarray,
    joint: np.ndarray | None,
    correlation: np.ndarray | None,
) -> float:
    """Multiplicative factor for adding `model_index` to `selected`.

    Exact factorization gives the model's own miss probability. The other arms
    apply bounded pairwise heuristics. They are ranking scores, not claims that a
    coherent multivariate copula or joint Bernoulli law has been fitted.
    """

    own = float(np.clip(miss_probability[model_index], EPSILON, 1.0 - EPSILON))
    if not selected:
        return own
    if joint is not None:
        # Heuristic average of pairwise conditional miss rates. It is not a
        # higher-order conditional probability for the whole selected set.
        conditionals = []
        for other in selected:
            marginal_other = float(np.clip(miss_probability[other], EPSILON, 1.0))
            conditionals.append(float(joint[model_index, other]) / marginal_other)
        return float(np.clip(np.mean(conditionals), EPSILON, 1.0))
    if correlation is not None:
        # Bounded correlation adjustment used only to rank the next model.
        rho = float(np.mean([correlation[model_index, other] for other in selected]))
        adjusted = own + rho * (1.0 - own) if rho > 0 else own * (1.0 + rho)
        return float(np.clip(adjusted, EPSILON, 1.0))
    return own


def _conditional_factors_vectorized(
    candidates: np.ndarray,
    selected: list[int],
    *,
    miss_probability: np.ndarray,
    joint: np.ndarray | None,
    correlation: np.ndarray | None,
) -> np.ndarray:
    """Conditional miss factors for ALL remaining candidates at once.

    Mathematically identical to calling `_conditional_miss_factor` once per
    candidate; the loop over candidates becomes an array operation, which is what
    makes a pool of a hundred models tractable. `test_racer_wrapper.py` asserts the
    two agree to floating-point tolerance on random inputs, so this is an
    optimisation and not a second implementation of the model.
    """
    own = np.clip(miss_probability[candidates], EPSILON, 1.0 - EPSILON)
    if not selected:
        return own
    selected_index = np.asarray(selected, dtype=int)
    if joint is not None:
        marginal_other = np.clip(miss_probability[selected_index], EPSILON, 1.0)
        # joint[candidates][:, selected] / marginal_other, averaged over selected
        conditionals = joint[np.ix_(candidates, selected_index)] / marginal_other[None, :]
        return np.clip(conditionals.mean(axis=1), EPSILON, 1.0)
    if correlation is not None:
        rho = correlation[np.ix_(candidates, selected_index)].mean(axis=1)
        adjusted = np.where(rho > 0, own + rho * (1.0 - own), own * (1.0 + rho))
        return np.clip(adjusted, EPSILON, 1.0)
    return own


def _validate_neighbor_correctness(
    neighbor_correctness: np.ndarray,
    *,
    n_queries: int,
    n_models: int,
) -> np.ndarray:
    """Validate one frozen KNN correctness support per query."""

    values = np.asarray(neighbor_correctness)
    if values.ndim != 3 or values.shape[0] != n_queries or values.shape[2] != n_models:
        raise ValueError(
            "neighbor_correctness must have shape "
            "(n_queries, n_neighbors, n_models)"
        )
    if values.shape[1] < 1:
        raise ValueError("neighbor_correctness must contain at least one neighbor")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("neighbor_correctness must be binary")
    return values.astype(bool, copy=False)


def query_local_empirical_scores(
    neighbor_correctness: np.ndarray,
    *,
    neighbor_weights: np.ndarray | None = None,
    max_depth: int | None = None,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Score prefixes from each query's coherent empirical KNN distribution.

    The same frozen neighbors used to estimate the marginal router probabilities
    are treated as an empirical joint correctness distribution. At each step the
    next model minimizes the weighted frequency with which every selected model
    misses. Exact coverage ties use weighted marginal accuracy, then model index.
    A model's score is the predecessor prefix coverage, so thresholding the scores
    produces the greedy prefix family.
    """

    values = np.asarray(neighbor_correctness)
    if values.ndim != 3:
        raise ValueError(
            "neighbor_correctness must have shape "
            "(n_queries, n_neighbors, n_models)"
        )
    n_queries, n_neighbors, n_models = values.shape
    labels = _validate_neighbor_correctness(
        values, n_queries=n_queries, n_models=n_models
    )
    if neighbor_weights is None:
        weights = np.ones((n_queries, n_neighbors), dtype=float)
    else:
        weights = np.asarray(neighbor_weights, dtype=float)
        if weights.ndim == 1 and weights.shape == (n_neighbors,):
            weights = np.broadcast_to(weights, (n_queries, n_neighbors))
        if weights.shape != (n_queries, n_neighbors):
            raise ValueError(
                "neighbor_weights must have shape (n_neighbors,) or "
                "(n_queries, n_neighbors)"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("neighbor_weights must be finite and non-negative")
    totals = weights.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("each query must have positive total neighbor weight")

    depth = n_models if max_depth is None else min(int(max_depth), n_models)
    if depth < 1:
        raise ValueError("max_depth must be positive")
    scores = np.ones((n_queries, n_models), dtype=float)
    orders = np.full((n_queries, depth), -1, dtype=int)
    residuals = np.ones((n_queries, depth + 1), dtype=float)
    for row in range(n_queries):
        remaining = np.ones(n_models, dtype=bool)
        all_miss = np.ones(n_neighbors, dtype=bool)
        predecessor_all_miss = 1.0
        row_weights = weights[row]
        row_failures = ~labels[row]
        marginal_accuracy = (
            row_weights[:, None] * labels[row].astype(float)
        ).sum(axis=0) / totals[row]
        for position in range(depth):
            candidates = np.flatnonzero(remaining)
            weighted_all_miss = row_weights * all_miss
            candidate_all_miss = (
                weighted_all_miss @ row_failures[:, candidates]
            ) / totals[row]

            minimum_miss = float(candidate_all_miss.min())
            miss_tied = candidate_all_miss <= minimum_miss + EPSILON
            tied_positions = np.flatnonzero(miss_tied)
            tied_marginals = marginal_accuracy[candidates[tied_positions]]
            maximum_marginal = float(tied_marginals.max())
            marginal_tied = tied_marginals >= maximum_marginal - EPSILON
            # candidates and tied_positions are ascending, so the first survivor
            # is the smallest model index after both tolerance-based comparisons.
            best_position = int(tied_positions[np.flatnonzero(marginal_tied)[0]])
            best_miss = float(candidate_all_miss[best_position])
            best_index = int(candidates[best_position])
            scores[row, best_index] = 1.0 - predecessor_all_miss
            orders[row, position] = best_index
            all_miss &= ~labels[row, :, best_index]
            predecessor_all_miss = best_miss
            residuals[row, position + 1] = predecessor_all_miss
            remaining[best_index] = False
    if return_trace:
        log_residuals = np.full_like(residuals, -np.inf)
        positive = residuals > 0.0
        log_residuals[positive] = np.log(residuals[positive])
        return scores, {
            "order": orders,
            "residuals": residuals,
            "log_residuals": log_residuals,
            "factorized_underflow": np.zeros_like(residuals, dtype=bool),
            "factorized_saturation": np.zeros_like(residuals, dtype=bool),
        }
    return scores


def query_local_empirical_order(
    neighbor_correctness: np.ndarray,
    *,
    neighbor_weights: np.ndarray | None = None,
    max_depth: int | None = None,
) -> np.ndarray:
    """Return each query's deterministic greedy local empirical model order."""

    _, trace = query_local_empirical_scores(
        neighbor_correctness,
        neighbor_weights=neighbor_weights,
        max_depth=max_depth,
        return_trace=True,
    )
    return trace["order"]


def local_order_factorized_scores(
    probabilities: np.ndarray,
    neighbor_correctness: np.ndarray,
    *,
    neighbor_weights: np.ndarray | None = None,
    max_depth: int | None = None,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Use local empirical ordering with query-marginal factorized sizing.

    The frozen KNN support determines only the greedy order. Prefix scores are
    then recomputed as the exact product of the query's marginal miss scores,
    cleanly separating ordering from sizing.
    """

    correctness = _validate_probabilities(probabilities)
    n_queries, n_models = correctness.shape
    labels = _validate_neighbor_correctness(
        neighbor_correctness, n_queries=n_queries, n_models=n_models
    )
    orders = query_local_empirical_order(
        labels,
        neighbor_weights=neighbor_weights,
        max_depth=max_depth,
    )
    depth = orders.shape[1]
    scores = np.ones_like(correctness, dtype=float)
    residuals = np.ones((n_queries, depth + 1), dtype=float)
    log_residuals = np.zeros((n_queries, depth + 1), dtype=float)
    underflow = np.zeros((n_queries, depth + 1), dtype=bool)
    saturation = np.zeros((n_queries, depth + 1), dtype=bool)

    for row in range(n_queries):
        for position, model_index in enumerate(orders[row]):
            predecessor = residuals[row, position]
            score = 1.0 - predecessor
            scores[row, model_index] = score
            saturation[row, position] = score == 1.0 and predecessor > 0.0

            miss = 1.0 - correctness[row, model_index]
            with np.errstate(divide="ignore", invalid="ignore"):
                next_log = log_residuals[row, position] + np.log(miss)
            next_residual = predecessor * miss
            residuals[row, position + 1] = next_residual
            log_residuals[row, position + 1] = next_log
            underflow[row, position + 1] = (
                next_residual == 0.0 and np.isfinite(next_log)
            )

    if return_trace:
        return scores, {
            "order": orders,
            "residuals": residuals,
            "log_residuals": log_residuals,
            "factorized_underflow": underflow,
            "factorized_saturation": saturation,
        }
    return scores


def fixed_design_marginal_order(
    design: np.ndarray,
    *,
    max_depth: int | None = None,
) -> tuple[int, ...]:
    """Sort models by design-split marginal accuracy, then by model index."""

    labels = np.asarray(design)
    if labels.ndim != 2 or labels.size == 0:
        raise ValueError("design must be a non-empty two-dimensional array")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("design must be binary")
    n_models = labels.shape[1]
    depth = n_models if max_depth is None else min(int(max_depth), n_models)
    if depth < 1:
        raise ValueError("max_depth must be positive")
    accuracy = labels.astype(float).mean(axis=0)
    order = np.lexsort((np.arange(n_models), -accuracy))[:depth]
    return tuple(int(index) for index in order)


def fixed_design_marginal_factorized_scores(
    probabilities: np.ndarray,
    design: np.ndarray,
    *,
    max_depth: int | None = None,
) -> np.ndarray:
    """Use design marginal accuracy order with query-factorized sizing."""

    correctness = _validate_probabilities(probabilities)
    labels = np.asarray(design)
    if labels.ndim != 2 or labels.shape[1] != correctness.shape[1]:
        raise ValueError("design and probabilities must share the model dimension")
    order = fixed_design_marginal_order(labels, max_depth=max_depth)
    return fixed_order_factorized_scores(correctness, order)


def per_query_score_diagnostics(
    raw_scores: np.ndarray,
    trace: dict[str, np.ndarray],
    *,
    threshold: float | None = None,
    selection_scores: np.ndarray | None = None,
) -> list[dict]:
    """Return JSON-safe score-resolution diagnostics for every query.

    Ties and plateaus are measured on raw scores. Selected depth and full-pool
    calls use the actual possibly smoothed scores calibrated by the runner.
    """

    raw = _validate_probabilities(raw_scores)
    selected = raw if selection_scores is None else np.asarray(selection_scores, dtype=float)
    if selected.shape != raw.shape:
        raise ValueError("selection_scores and raw_scores must share a shape")
    order = np.asarray(trace["order"], dtype=int)
    residuals = np.asarray(trace["residuals"], dtype=float)
    logs = np.asarray(trace["log_residuals"], dtype=float)
    underflow = np.asarray(trace["factorized_underflow"], dtype=bool)
    saturation = np.asarray(trace["factorized_saturation"], dtype=bool)
    if order.shape[0] != raw.shape[0] or residuals.shape != logs.shape:
        raise ValueError("trace arrays do not match the score rows")

    records = []
    for row in range(raw.shape[0]):
        finite = raw[row][np.isfinite(raw[row])]
        values, counts = np.unique(finite, return_counts=True)
        tied = [
            {"score": float(value), "size": int(count)}
            for value, count in zip(values, counts)
            if count > 1
        ]
        row_residuals = residuals[row]
        zero_positions = np.flatnonzero(row_residuals == 0.0)
        first_zero = int(zero_positions[0]) if zero_positions.size else None
        plateau_depth = None
        plateau_value = None
        for depth in range(max(len(row_residuals) - 1, 0)):
            if np.all(row_residuals[depth:] == row_residuals[depth]):
                plateau_depth = int(depth)
                plateau_value = float(row_residuals[depth])
                break

        if threshold is None:
            selected_depth = None
            full_pool = None
        else:
            selected_depth = int(np.count_nonzero(selected[row] <= threshold))
            full_pool = bool(selected_depth == raw.shape[1])
        log_values = [
            float(value) if np.isfinite(value) else None for value in logs[row]
        ]
        records.append({
            "query_offset": int(row),
            "order": [int(index) for index in order[row]],
            "distinct_finite_score_count": int(len(values)),
            "first_zero_miss_depth": first_zero,
            "plateau_depth": plateau_depth,
            "plateau_value": plateau_value,
            "tied_score_blocks": tied,
            "tied_block_sizes": [block["size"] for block in tied],
            "score_one_tail_size": int(np.count_nonzero(raw[row] == 1.0)),
            "selected_depth": selected_depth,
            "full_pool_call_indicator": full_pool,
            "log_residual": log_values,
            "factorized_underflow_count": int(np.count_nonzero(underflow[row])),
            "factorized_saturation_count": int(np.count_nonzero(saturation[row])),
        })
    return records


def fixed_order_factorized_scores(
    probabilities: np.ndarray,
    fixed_order: list[int] | tuple[int, ...] | np.ndarray,
    *,
    max_depth: int | None = None,
) -> np.ndarray:
    """Use one fixed ordering with query-adaptive factorized prefix sizing."""

    correctness = _validate_probabilities(probabilities)
    _, n_models = correctness.shape
    order = np.asarray(fixed_order, dtype=int)
    if order.ndim != 1 or order.size == 0:
        raise ValueError("fixed_order must be a non-empty one-dimensional sequence")
    if np.any((order < 0) | (order >= n_models)) or len(np.unique(order)) != len(order):
        raise ValueError("fixed_order must contain unique valid model indices")
    depth = len(order) if max_depth is None else min(int(max_depth), len(order))
    if depth < 1:
        raise ValueError("max_depth must be positive")

    scores = np.ones_like(correctness, dtype=float)
    residual = np.ones(correctness.shape[0], dtype=float)
    for model_index in order[:depth]:
        scores[:, model_index] = 1.0 - residual
        residual *= 1.0 - correctness[:, model_index]
    return scores


def sequential_scores(
    probabilities: np.ndarray,
    *,
    dependence: DependenceModel,
    design: np.ndarray | None = None,
    max_depth: int | None = None,
    hyper: dict | None = None,
    neighbor_correctness: np.ndarray | None = None,
    neighbor_weights: np.ndarray | None = None,
    fixed_order: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> np.ndarray:
    """Per-query, per-model RACER-compatible dependence-aware scores.

    Returns an array shaped like ``probabilities`` where entry (i, m) is
    ``1 - q_{k-1}`` for the position k at which model m enters query i's greedy
    order. Larger means the query was already better covered before that model,
    so a sublevel set is a prefix of the order.
    """

    correctness = _validate_probabilities(probabilities)
    n_queries, n_models = correctness.shape
    dependence = canonical_dependence_name(dependence)
    if dependence == LOCAL_ORDER_FACTORIZED:
        if neighbor_correctness is None:
            raise ValueError(
                "local_order_factorized requires frozen neighbor correctness"
            )
        local = _validate_neighbor_correctness(
            neighbor_correctness, n_queries=n_queries, n_models=n_models
        )
        return local_order_factorized_scores(
            correctness,
            local,
            neighbor_weights=neighbor_weights,
            max_depth=max_depth,
        )
    if dependence == FIXED_DESIGN_MARGINAL_FACTORIZED:
        if design is None:
            raise ValueError(
                "fixed_design_marginal_factorized requires design labels"
            )
        return fixed_design_marginal_factorized_scores(
            correctness, design, max_depth=max_depth
        )
    if dependence == QUERY_LOCAL_EMPIRICAL_COVERAGE:
        if neighbor_correctness is None:
            raise ValueError(
                "query_local_empirical_coverage requires frozen neighbor correctness"
            )
        local = _validate_neighbor_correctness(
            neighbor_correctness, n_queries=n_queries, n_models=n_models
        )
        return query_local_empirical_scores(
            local, neighbor_weights=neighbor_weights, max_depth=max_depth
        )
    if dependence == FIXED_TALLY_FACTORIZED:
        if fixed_order is None:
            raise ValueError("fixed_tally_factorized requires a fixed empirical tally order")
        return fixed_order_factorized_scores(
            correctness, fixed_order, max_depth=max_depth
        )
    miss = 1.0 - correctness
    depth = int(max_depth) if max_depth else n_models

    joint = None
    correlation = None
    h = hyper or {}
    if dependence == PAIRWISE_HEURISTIC:
        if design is None:
            raise ValueError("pairwise_heuristic requires design labels")
        joint = _pairwise_joint_miss(design, laplace_smoothing=float(h.get("laplace_smoothing", 0.0)))
    elif dependence == CORRELATION_HEURISTIC:
        if design is None:
            raise ValueError("correlation_heuristic requires design labels")
        correlation = _failure_correlation(design, clip_magnitude=float(h.get("clip_magnitude", 0.999)))
    elif dependence == REGULARIZED_CORRELATION:
        if design is None:
            raise ValueError("regularized_correlation requires design labels")
        correlation = _failure_correlation(design, ridge=float(h.get("ridge", 0.0)))
    elif dependence != FACTORIZED_INDEPENDENCE:
        raise ValueError(f"unknown dependence model: {dependence}")

    scores = np.ones((n_queries, n_models), dtype=float)
    for row in range(n_queries):
        remaining = np.ones(n_models, dtype=bool)
        selected: list[int] = []
        residual = 1.0
        for _ in range(min(depth, n_models)):
            candidates = np.flatnonzero(remaining)
            if candidates.size == 0:
                break
            factors = _conditional_factors_vectorized(
                candidates,
                selected,
                miss_probability=miss[row],
                joint=joint,
                correlation=correlation,
            )
            candidate_residuals = residual * factors
            # The scalar implementation this replaces switched its running best only
            # on a STRICT improvement exceeding EPSILON, scanning candidates in
            # ascending index order. That is a tie-break policy, not an accident: with
            # a pool of a hundred models, near-ties within 1e-12 are common, and plain
            # argmin would silently pick a different model. The scan below reproduces
            # the original rule exactly. It runs over one small float array per step,
            # so it costs nothing next to the factor computation that was vectorized.
            best_position = 0
            best_residual = float(candidate_residuals[0])
            for position in range(1, candidate_residuals.size):
                value = float(candidate_residuals[position])
                if value < best_residual - EPSILON:
                    best_residual = value
                    best_position = position
            best_index = int(candidates[best_position])
            scores[row, best_index] = 1.0 - residual
            residual = best_residual
            selected.append(best_index)
            remaining[best_index] = False
        # Models never selected keep score 1.0, so they enter only at lambda >= 1.
    return scores


def augment_with_racer_null(
    probabilities: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    """Append RACER's own null column so abstention behaviour is unchanged.

    RACER's null score under the one-minus-probability map is ``max_m p(x, m)``.
    Reusing it verbatim means any difference in abstention rate between the
    marginal and dependence-aware arms comes from the real-model scores only.
    """

    correctness = _validate_probabilities(probabilities)
    values = np.asarray(scores, dtype=float)
    if values.shape != correctness.shape:
        raise ValueError("scores and probabilities must share a shape")
    null_column = np.max(correctness, axis=1, keepdims=True)
    return np.concatenate((values, null_column), axis=1)


def is_prefix_family(scores_row: np.ndarray, order: list[int]) -> bool:
    """Check that thresholding a score row yields prefixes of `order`."""

    values = np.asarray(scores_row, dtype=float)
    ordered = values[order]
    return bool(np.all(np.diff(ordered) >= -EPSILON))


__all__ = [
    "FACTORIZED_INDEPENDENCE",
    "PAIRWISE_HEURISTIC",
    "CORRELATION_HEURISTIC",
    "REGULARIZED_CORRELATION",
    "QUERY_LOCAL_EMPIRICAL_COVERAGE",
    "FIXED_TALLY_FACTORIZED",
    "FIXED_DESIGN_MARGINAL_FACTORIZED",
    "LOCAL_ORDER_FACTORIZED",
    "DependenceModel",
    "augment_with_racer_null",
    "canonical_dependence_name",
    "fixed_design_marginal_factorized_scores",
    "fixed_design_marginal_order",
    "fixed_order_factorized_scores",
    "is_prefix_family",
    "local_order_factorized_scores",
    "per_query_score_diagnostics",
    "query_local_empirical_order",
    "query_local_empirical_scores",
    "sequential_scores",
    "_conditional_factors_vectorized",
    "_conditional_miss_factor",
]
