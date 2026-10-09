"""Design-time model selector baselines for dependence-aware routing.

Every deployable selector in this module returns a :class:`SelectionFamily`.
Its members include depth zero and are prefixes of one deterministic ordering,
so a caller can safely verify nestedness before independent CRC calibration.
Selector construction alone is not a risk certificate.

The exact cardinality and exact cost routines are deliberately different.
They are labeled ``noncertified_reference`` and do not claim nestedness. Their
independently optimized subsets measure design-split headroom only.

All correctness matrices use rows for queries and columns for models. A value
of one means that the model answered the query correctly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import log2, sqrt
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


CERTIFIABLE_AFTER_CRC = "eligible_after_independent_crc"
NONCERTIFIED_REFERENCE = "noncertified_reference"
_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SelectorMetadata:
    """Machine-readable scope declaration for one selector or reference."""

    name: str
    nested: bool
    certification: str
    reference_only: bool
    objective: str
    notes: str


@dataclass(frozen=True)
class SelectionFamily:
    """A selector family indexed by depth.

    ``members[0]`` is the empty set. For a nested chain, ``order[d - 1]`` is
    the model added to obtain ``members[d]``. Members are stored as sorted
    tuples so equality does not depend on insertion order.
    """

    members: Tuple[Tuple[int, ...], ...]
    order: Tuple[int, ...]
    metadata: SelectorMetadata
    objective_values: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.members or self.members[0] != ():
            raise ValueError("a selection family must begin with the empty set")
        for member in self.members:
            if len(member) != len(set(member)) or any(index < 0 for index in member):
                raise ValueError("family members must contain unique non-negative indices")
        if self.objective_values and len(self.objective_values) != len(self.members):
            raise ValueError("objective_values must align with members")
        if self.metadata.nested and not self.is_nested:
            raise ValueError("selector declared a nested family but returned a nonnested one")
        if self.metadata.nested:
            if len(self.members) != len(self.order) + 1:
                raise ValueError("a nested chain must have one member per order prefix")
            if tuple(sorted(self.order)) != tuple(range(len(self.order))):
                raise ValueError("a complete chain order must be a permutation of models")

    @property
    def is_nested(self) -> bool:
        """Whether every family member is contained in its successor."""

        return all(
            set(left).issubset(right)
            for left, right in zip(self.members, self.members[1:])
        )

    @property
    def depths(self) -> Tuple[int, ...]:
        return tuple(range(len(self.members)))


@dataclass(frozen=True)
class OracleSelection:
    """One independently optimized exact cost reference at a target risk."""

    subset: Optional[Tuple[int, ...]]
    all_miss_rate: float
    total_cost: float
    alpha: float
    feasible: bool
    metadata: SelectorMetadata


SELECTOR_METADATA: Mapping[str, SelectorMetadata] = {
    "marginal_depth": SelectorMetadata(
        "marginal_depth",
        True,
        CERTIFIABLE_AFTER_CRC,
        False,
        "descending marginal correctness",
        "Fixed-cardinality optimum only under the matching independence assumption.",
    ),
    "cost_aware_independence": SelectorMetadata(
        "cost_aware_independence",
        True,
        CERTIFIABLE_AFTER_CRC,
        False,
        "independence miss reduction per unit cost",
        "Orders by p_j / cost_j because the current independent miss factor is common.",
    ),
    "cheapest_first": SelectorMetadata(
        "cheapest_first",
        True,
        CERTIFIABLE_AFTER_CRC,
        False,
        "ascending model cost",
        "Cost-only control with deterministic index tie breaking.",
    ),
    "empirical_all_miss_greedy": SelectorMetadata(
        "empirical_all_miss_greedy",
        True,
        CERTIFIABLE_AFTER_CRC,
        False,
        "largest empirical reduction in all-miss probability",
        "Equivalent to greedy empirical incremental coverage.",
    ),
    "shrinkage_empirical_greedy": SelectorMetadata(
        "shrinkage_empirical_greedy",
        True,
        CERTIFIABLE_AFTER_CRC,
        False,
        "largest shrinkage all-miss reduction",
        "Shrinks subset all-miss frequency toward the independence plug-in target.",
    ),
    "pairwise_double_fault": SelectorMetadata(
        "pairwise_double_fault",
        True,
        CERTIFIABLE_AFTER_CRC,
        False,
        "lowest mean pairwise probability that both models fail",
        "A pairwise diversity heuristic, not a joint all-miss estimator.",
    ),
    "pairwise_error_correlation": SelectorMetadata(
        "pairwise_error_correlation",
        True,
        CERTIFIABLE_AFTER_CRC,
        False,
        "lowest mean pairwise error correlation",
        "A pairwise diversity heuristic with zero correlation for degenerate errors.",
    ),
    "mi_style_greedy": SelectorMetadata(
        "mi_style_greedy",
        True,
        CERTIFIABLE_AFTER_CRC,
        False,
        "empirical conditional mutual information with the true label",
        "A direct plug-in approximation to the greedy objective of arXiv:2602.08003.",
    ),
    "complementary_incremental_coverage": SelectorMetadata(
        "complementary_incremental_coverage",
        True,
        CERTIFIABLE_AFTER_CRC,
        False,
        "largest correctness coverage on queries missed by the current set",
        "The explicit complementary-agent form of empirical coverage greedy.",
    ),
    "gaussian_copula_greedy": SelectorMetadata(
        "gaussian_copula_greedy",
        True,
        CERTIFIABLE_AFTER_CRC,
        False,
        "largest frozen Monte Carlo Gaussian-copula miss reduction",
        "Uses one design-seeded latent draw bank for every later query.",
    ),
    "exact_cardinality_oracle": SelectorMetadata(
        "exact_cardinality_oracle",
        False,
        NONCERTIFIED_REFERENCE,
        True,
        "minimum empirical all-miss rate independently at each cardinality",
        "The optimal subsets may be nonnested and are not a calibratable chain.",
    ),
    "exact_cost_oracle": SelectorMetadata(
        "exact_cost_oracle",
        False,
        NONCERTIFIED_REFERENCE,
        True,
        "minimum exact cost subject to empirical all-miss risk at one target",
        "Independent target-specific enumeration, not a certified router.",
    ),
}


def _as_correctness_matrix(correctness: Any) -> np.ndarray:
    values = np.asarray(correctness)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("correctness must be a non-empty two-dimensional array")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("correctness must contain only binary values")
    return values.astype(bool, copy=False)


def _as_marginals(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2:
        array = _as_correctness_matrix(array).mean(axis=0)
    elif array.ndim != 1 or array.size == 0:
        raise ValueError("marginals must be a non-empty vector or correctness matrix")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("marginal correctness probabilities must lie in [0, 1]")
    return array


def _as_costs(costs: Any, n_models: int) -> np.ndarray:
    values = np.asarray(costs, dtype=float)
    if values.shape != (n_models,):
        raise ValueError("costs must have one entry per model")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("costs must be finite and strictly positive")
    return values


def _validate_model_count(n_models: int) -> None:
    if isinstance(n_models, bool) or not isinstance(n_models, (int, np.integer)):
        raise TypeError("n_models must be a positive integer")
    if int(n_models) <= 0:
        raise ValueError("n_models must be a positive integer")


def _prefix_members(order: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    current = set()
    members = [()]
    for index in order:
        current.add(int(index))
        members.append(tuple(sorted(current)))
    return tuple(members)


def _make_chain(
    order: Sequence[int],
    metadata_name: str,
    objective_values: Sequence[float] = (),
) -> SelectionFamily:
    normalized_order = tuple(int(index) for index in order)
    return SelectionFamily(
        members=_prefix_members(normalized_order),
        order=normalized_order,
        metadata=SELECTOR_METADATA[metadata_name],
        objective_values=tuple(float(value) for value in objective_values),
    )


def is_nested_family(members: Sequence[Iterable[int]]) -> bool:
    """Return true when the supplied sequence is nested by set inclusion."""

    normalized = [set(member) for member in members]
    return all(left.issubset(right) for left, right in zip(normalized, normalized[1:]))


def empirical_all_miss_rate(correctness: Any, subset: Iterable[int]) -> float:
    """Return the empirical probability that every selected model is wrong."""

    labels = _as_correctness_matrix(correctness)
    indices = tuple(int(index) for index in subset)
    if len(indices) != len(set(indices)):
        raise ValueError("subset must not contain duplicate indices")
    if any(index < 0 or index >= labels.shape[1] for index in indices):
        raise IndexError("subset contains an out-of-range model index")
    if not indices:
        return 1.0
    return float(np.mean(~np.any(labels[:, indices], axis=1)))


def marginal_depth_chain(marginals_or_correctness: Any) -> SelectionFamily:
    """Build the nested baseline ordered by descending marginal correctness."""

    marginals = _as_marginals(marginals_or_correctness)
    order = sorted(range(marginals.size), key=lambda j: (-marginals[j], j))
    values = [1.0]
    current_miss = 1.0
    for index in order:
        current_miss *= 1.0 - marginals[index]
        values.append(current_miss)
    return _make_chain(order, "marginal_depth", values)


def cost_aware_independence_chain(
    marginals_or_correctness: Any, costs: Any
) -> SelectionFamily:
    """Build the independent miss-reduction-per-cost chain.

    Under independence, adding model ``j`` reduces the current all-miss
    probability by ``q(S) * p_j``. Since ``q(S)`` is common to every candidate
    at one depth, the deterministic greedy order is descending ``p_j / cost_j``.
    """

    marginals = _as_marginals(marginals_or_correctness)
    cost_values = _as_costs(costs, marginals.size)
    ratios = marginals / cost_values
    order = sorted(range(marginals.size), key=lambda j: (-ratios[j], j))
    values = [1.0]
    current_miss = 1.0
    for index in order:
        current_miss *= 1.0 - marginals[index]
        values.append(current_miss)
    return _make_chain(order, "cost_aware_independence", values)


def cheapest_first_chain(costs: Any) -> SelectionFamily:
    """Build a nested cost-only control ordered from cheapest to costliest."""

    values = np.asarray(costs, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("costs must be a non-empty one-dimensional array")
    values = _as_costs(values, values.size)
    order = sorted(range(values.size), key=lambda j: (values[j], j))
    cumulative = [0.0]
    for index in order:
        cumulative.append(cumulative[-1] + float(values[index]))
    return _make_chain(order, "cheapest_first", cumulative)


def _empirical_coverage_order(correctness: np.ndarray) -> Tuple[int, ...]:
    remaining = set(range(correctness.shape[1]))
    selected = []
    uncovered = np.ones(correctness.shape[0], dtype=bool)
    while remaining:
        gains = {
            index: int(np.count_nonzero(uncovered & correctness[:, index]))
            for index in remaining
        }
        best = min(remaining, key=lambda j: (-gains[j], j))
        selected.append(best)
        uncovered &= ~correctness[:, best]
        remaining.remove(best)
    return tuple(selected)


def empirical_all_miss_greedy_chain(correctness: Any) -> SelectionFamily:
    """Greedily maximize empirical coverage, equivalently reduce all miss."""

    labels = _as_correctness_matrix(correctness)
    order = _empirical_coverage_order(labels)
    members = _prefix_members(order)
    values = [empirical_all_miss_rate(labels, member) for member in members]
    return _make_chain(order, "empirical_all_miss_greedy", values)


def shrinkage_empirical_greedy_chain(
    correctness: Any, *, prior_strength: float = 10.0
) -> SelectionFamily:
    """Greedily reduce a shrinkage estimate of subset all-miss probability.

    For each subset, the empirical joint all-miss frequency receives weight
    ``n`` and the independence plug-in receives ``prior_strength``. Marginals
    in the plug-in use Jeffreys smoothing. The full chain is fitted once on the
    design matrix and is deterministic.
    """

    labels = _as_correctness_matrix(correctness)
    if not np.isfinite(prior_strength) or prior_strength < 0.0:
        raise ValueError("prior_strength must be finite and non-negative")
    n_rows, n_models = labels.shape
    smoothed_p = (labels.sum(axis=0) + 0.5) / (n_rows + 1.0)

    def shrunk_miss(subset: Tuple[int, ...]) -> float:
        empirical = empirical_all_miss_rate(labels, subset)
        independent = float(np.prod(1.0 - smoothed_p[list(subset)])) if subset else 1.0
        return float(
            (n_rows * empirical + prior_strength * independent)
            / (n_rows + prior_strength)
        )

    remaining = set(range(n_models))
    selected = []
    values = [1.0]
    while remaining:
        current = tuple(selected)
        current_miss = shrunk_miss(current)
        gains = {
            index: current_miss - shrunk_miss(tuple(selected + [index]))
            for index in remaining
        }
        best = min(remaining, key=lambda j: (-gains[j], j))
        selected.append(best)
        remaining.remove(best)
        values.append(shrunk_miss(tuple(selected)))
    return _make_chain(selected, "shrinkage_empirical_greedy", values)


def _error_correlation(left: np.ndarray, right: np.ndarray) -> float:
    centered_left = left.astype(float) - float(np.mean(left))
    centered_right = right.astype(float) - float(np.mean(right))
    scale = sqrt(float(np.mean(centered_left**2) * np.mean(centered_right**2)))
    if scale <= _TIE_TOLERANCE:
        return 0.0
    return float(np.mean(centered_left * centered_right) / scale)


def pairwise_diversity_greedy_chain(
    correctness: Any, *, metric: str = "double_fault"
) -> SelectionFamily:
    """Build a nested pairwise double-fault or error-correlation chain.

    The first model has highest marginal correctness. Each later model has the
    smallest mean pairwise dependence statistic with the selected models.
    Marginal correctness and then model index break ties deterministically.
    Pairwise summaries do not recover higher-order all-miss probabilities.
    """

    labels = _as_correctness_matrix(correctness)
    normalized_metric = str(metric).strip().lower().replace("-", "_")
    if normalized_metric not in {"double_fault", "correlation", "error_correlation"}:
        raise ValueError("metric must be 'double_fault' or 'correlation'")
    errors = ~labels
    marginals = labels.mean(axis=0)
    n_models = labels.shape[1]
    matrix = np.zeros((n_models, n_models), dtype=float)
    for left in range(n_models):
        for right in range(left + 1, n_models):
            if normalized_metric == "double_fault":
                value = float(np.mean(errors[:, left] & errors[:, right]))
            else:
                value = _error_correlation(errors[:, left], errors[:, right])
            matrix[left, right] = matrix[right, left] = value

    first = min(range(n_models), key=lambda j: (-marginals[j], j))
    selected = [first]
    remaining = set(range(n_models)) - {first}
    while remaining:
        mean_dependence = {
            index: float(np.mean(matrix[index, selected])) for index in remaining
        }
        best = min(
            remaining,
            key=lambda j: (mean_dependence[j], -marginals[j], j),
        )
        selected.append(best)
        remaining.remove(best)

    metadata_name = (
        "pairwise_double_fault"
        if normalized_metric == "double_fault"
        else "pairwise_error_correlation"
    )
    return _make_chain(selected, metadata_name)


def pairwise_double_fault_chain(correctness: Any) -> SelectionFamily:
    return pairwise_diversity_greedy_chain(correctness, metric="double_fault")


def pairwise_correlation_chain(correctness: Any) -> SelectionFamily:
    return pairwise_diversity_greedy_chain(correctness, metric="correlation")


def _hashable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_hashable(item) for item in value.tolist())
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    return value


def _conditional_entropy(labels: np.ndarray, contexts: Sequence[Tuple[Any, ...]]) -> float:
    grouped: defaultdict[Tuple[Any, ...], Counter[Any]] = defaultdict(Counter)
    for label, context in zip(labels, contexts):
        grouped[context][_hashable(label)] += 1
    total = float(labels.size)
    entropy = 0.0
    for counts in grouped.values():
        group_size = float(sum(counts.values()))
        group_entropy = 0.0
        for count in counts.values():
            probability = count / group_size
            group_entropy -= probability * log2(probability)
        entropy += (group_size / total) * group_entropy
    return entropy


def mi_style_greedy_chain(
    predictions: Any,
    true_labels: Any,
    *,
    costs: Optional[Any] = None,
) -> SelectionFamily:
    """Greedily maximize plug-in information about labels from predictions.

    Türkmen, Büyükateş, and Baştöpçü (arXiv:2602.08003) formulate budgeted LLM
    ensemble selection as maximizing mutual information between the true label
    and predictions of selected models. This implementation is the transparent
    empirical approximation to that objective. At each step it maximizes
    ``I(label; candidate prediction | selected predictions)``. With costs it
    uses conditional information gain per unit cost. It uses discrete plug-in
    frequencies and is therefore a baseline approximation, not a reproduction
    of every estimator or experiment in that paper.
    """

    prediction_values = np.asarray(predictions)
    labels = np.asarray(true_labels)
    if prediction_values.ndim != 2 or prediction_values.shape[0] == 0:
        raise ValueError("predictions must be a non-empty two-dimensional array")
    if labels.ndim != 1 or labels.shape[0] != prediction_values.shape[0]:
        raise ValueError("true_labels must have one entry per prediction row")
    n_rows, n_models = prediction_values.shape
    cost_values = (
        np.ones(n_models, dtype=float)
        if costs is None
        else _as_costs(costs, n_models)
    )

    selected = []
    remaining = set(range(n_models))
    contexts = [tuple() for _ in range(n_rows)]
    current_entropy = _conditional_entropy(labels, contexts)
    cumulative_information = [0.0]
    while remaining:
        gains = {}
        next_contexts = {}
        for index in remaining:
            candidate_contexts = [
                context + (_hashable(prediction_values[row, index]),)
                for row, context in enumerate(contexts)
            ]
            next_contexts[index] = candidate_contexts
            gain = current_entropy - _conditional_entropy(labels, candidate_contexts)
            gains[index] = max(0.0, float(gain)) / cost_values[index]
        best = min(remaining, key=lambda j: (-gains[j], j))
        unscaled_gain = current_entropy - _conditional_entropy(
            labels, next_contexts[best]
        )
        cumulative_information.append(
            cumulative_information[-1] + max(0.0, float(unscaled_gain))
        )
        contexts = next_contexts[best]
        current_entropy = _conditional_entropy(labels, contexts)
        selected.append(best)
        remaining.remove(best)
    return _make_chain(selected, "mi_style_greedy", cumulative_information)


def complementary_incremental_coverage_chain(correctness: Any) -> SelectionFamily:
    """Select the model covering most rows missed by all selected models.

    At each depth the score is the number of design queries for which the
    candidate is correct and every selected model is wrong. This explicit
    complementary-agent criterion is algebraically the empirical all-miss
    reduction criterion, but it is retained as a named comparator.
    """

    labels = _as_correctness_matrix(correctness)
    order = _empirical_coverage_order(labels)
    members = _prefix_members(order)
    coverage = [1.0 - empirical_all_miss_rate(labels, member) for member in members]
    return _make_chain(order, "complementary_incremental_coverage", coverage)


class FrozenGaussianCopulaMap:
    """A design-frozen Monte Carlo Gaussian copula mapping.

    One iid standard-normal bank is sampled in ``__init__`` from ``seed``.
    Every calibration and test query reuses that same bank. Query-specific
    marginal probabilities and correlation matrices transform common random
    numbers, so the randomized query-to-family map is fixed once at design
    time. No method mutates or resamples the bank.
    """

    def __init__(
        self,
        n_models: int,
        *,
        n_draws: int = 8192,
        seed: int = 0,
    ) -> None:
        _validate_model_count(n_models)
        if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)):
            raise TypeError("n_draws must be a positive integer")
        if int(n_draws) <= 0:
            raise ValueError("n_draws must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be a non-negative integer")
        if int(seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        self.n_models = int(n_models)
        self.n_draws = int(n_draws)
        self.seed = int(seed)
        rng = np.random.default_rng(np.random.SeedSequence(self.seed))
        latent = rng.standard_normal((self.n_draws, self.n_models))
        latent.setflags(write=False)
        self._design_latent_sample = latent

    @property
    def design_latent_sample(self) -> np.ndarray:
        """Return a copy of the immutable design-time latent bank."""

        return self._design_latent_sample.copy()

    def _correlation_matrix(self, correlation: Any) -> np.ndarray:
        matrix = np.asarray(correlation, dtype=float)
        if matrix.shape != (self.n_models, self.n_models):
            raise ValueError("correlation must have shape (n_models, n_models)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("correlation must contain only finite values")
        if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0.0):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(matrix), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("correlation must have a unit diagonal")
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError("correlation must be positive definite") from error
        return matrix

    def failure_sample(
        self, marginal_correctness: Any, correlation: Any
    ) -> np.ndarray:
        """Map one query to frozen Monte Carlo model-failure indicators."""

        marginals = _as_marginals(marginal_correctness)
        if marginals.shape != (self.n_models,):
            raise ValueError("marginal_correctness must have one entry per model")
        matrix = self._correlation_matrix(correlation)
        cholesky = np.linalg.cholesky(matrix)
        correlated = self._design_latent_sample @ cholesky.T
        normal = NormalDist()
        thresholds = np.array(
            [
                float("inf")
                if probability <= 0.0
                else float("-inf")
                if probability >= 1.0
                else normal.inv_cdf(1.0 - float(probability))
                for probability in marginals
            ]
        )
        failures = correlated <= thresholds[None, :]
        failures.setflags(write=False)
        return failures

    def all_miss_probability(
        self,
        marginal_correctness: Any,
        correlation: Any,
        subset: Iterable[int],
    ) -> float:
        indices = tuple(int(index) for index in subset)
        if len(indices) != len(set(indices)):
            raise ValueError("subset must not contain duplicate indices")
        if any(index < 0 or index >= self.n_models for index in indices):
            raise IndexError("subset contains an out-of-range model index")
        if not indices:
            return 1.0
        failures = self.failure_sample(marginal_correctness, correlation)
        return float(np.mean(np.all(failures[:, indices], axis=1)))


def gaussian_copula_greedy_chain(
    marginal_correctness: Any,
    correlation: Any,
    frozen_map: FrozenGaussianCopulaMap,
) -> SelectionFamily:
    """Build a nested coverage chain from one frozen Gaussian copula map."""

    marginals = _as_marginals(marginal_correctness)
    if not isinstance(frozen_map, FrozenGaussianCopulaMap):
        raise TypeError("frozen_map must be a FrozenGaussianCopulaMap")
    if marginals.size != frozen_map.n_models:
        raise ValueError("frozen_map and marginals must use the same model count")
    failures = frozen_map.failure_sample(marginals, correlation)
    order = _empirical_coverage_order(~failures)
    members = _prefix_members(order)
    values = [
        1.0 if not member else float(np.mean(np.all(failures[:, member], axis=1)))
        for member in members
    ]
    return _make_chain(order, "gaussian_copula_greedy", values)


def _validate_enumeration_size(n_models: int, max_models: int) -> None:
    if isinstance(max_models, bool) or not isinstance(max_models, (int, np.integer)):
        raise TypeError("max_models must be a positive integer")
    if int(max_models) <= 0:
        raise ValueError("max_models must be a positive integer")
    if n_models > int(max_models):
        raise ValueError(
            f"exact enumeration is limited to {int(max_models)} models; got {n_models}"
        )


def exact_cardinality_oracle(
    correctness: Any, *, max_models: int = 20
) -> SelectionFamily:
    """Enumerate the best empirical all-miss subset at every cardinality.

    Cardinalities are optimized independently with lexicographic tie breaking.
    The returned family is explicitly a noncertified, potentially nonnested
    reference. Monotone objective values do not turn its subsets into a chain.
    """

    labels = _as_correctness_matrix(correctness)
    n_models = labels.shape[1]
    _validate_enumeration_size(n_models, max_models)
    members = [()]
    values = [1.0]
    for cardinality in range(1, n_models + 1):
        best_subset = None
        best_rate = float("inf")
        for subset in combinations(range(n_models), cardinality):
            rate = empirical_all_miss_rate(labels, subset)
            if rate < best_rate - _TIE_TOLERANCE:
                best_subset = subset
                best_rate = rate
        assert best_subset is not None
        members.append(best_subset)
        values.append(best_rate)
    return SelectionFamily(
        members=tuple(members),
        order=(),
        metadata=SELECTOR_METADATA["exact_cardinality_oracle"],
        objective_values=tuple(values),
    )


def exact_cost_oracle(
    correctness: Any,
    costs: Any,
    alpha: float,
    *,
    max_models: int = 20,
) -> OracleSelection:
    """Enumerate the cheapest subset meeting one empirical all-miss target.

    Separate calls at separate targets may return nonnested subsets. The result
    is a noncertified reference learned from the supplied matrix, not a policy
    with distribution-free risk control.
    """

    labels = _as_correctness_matrix(correctness)
    n_models = labels.shape[1]
    _validate_enumeration_size(n_models, max_models)
    cost_values = _as_costs(costs, n_models)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")

    best_subset = None
    best_key = None
    best_rate = float("inf")
    for cardinality in range(n_models + 1):
        for subset in combinations(range(n_models), cardinality):
            rate = empirical_all_miss_rate(labels, subset)
            if rate > alpha + _TIE_TOLERANCE:
                continue
            total_cost = float(np.sum(cost_values[list(subset)])) if subset else 0.0
            key = (total_cost, rate, cardinality, subset)
            if best_key is None or key < best_key:
                best_key = key
                best_subset = subset
                best_rate = rate

    metadata = SELECTOR_METADATA["exact_cost_oracle"]
    if best_subset is None:
        return OracleSelection(
            subset=None,
            all_miss_rate=float("inf"),
            total_cost=float("inf"),
            alpha=float(alpha),
            feasible=False,
            metadata=metadata,
        )
    return OracleSelection(
        subset=best_subset,
        all_miss_rate=float(best_rate),
        total_cost=float(best_key[0]),
        alpha=float(alpha),
        feasible=True,
        metadata=metadata,
    )


# Short aliases keep experiment tables readable while preserving one implementation.
marginal_chain = marginal_depth_chain
empirical_greedy_chain = empirical_all_miss_greedy_chain
shrinkage_greedy_chain = shrinkage_empirical_greedy_chain
mi_greedy_chain = mi_style_greedy_chain
complementary_agent_chain = complementary_incremental_coverage_chain


__all__ = [
    "CERTIFIABLE_AFTER_CRC",
    "NONCERTIFIED_REFERENCE",
    "OracleSelection",
    "SELECTOR_METADATA",
    "SelectionFamily",
    "SelectorMetadata",
    "FrozenGaussianCopulaMap",
    "cheapest_first_chain",
    "complementary_agent_chain",
    "complementary_incremental_coverage_chain",
    "cost_aware_independence_chain",
    "empirical_all_miss_greedy_chain",
    "empirical_all_miss_rate",
    "empirical_greedy_chain",
    "exact_cardinality_oracle",
    "exact_cost_oracle",
    "gaussian_copula_greedy_chain",
    "is_nested_family",
    "marginal_chain",
    "marginal_depth_chain",
    "mi_greedy_chain",
    "mi_style_greedy_chain",
    "pairwise_correlation_chain",
    "pairwise_diversity_greedy_chain",
    "pairwise_double_fault_chain",
    "shrinkage_empirical_greedy_chain",
    "shrinkage_greedy_chain",
]
