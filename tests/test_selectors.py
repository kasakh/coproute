"""Invariant tests for design-time selector baselines."""

from pathlib import Path
import inspect
import sys

import numpy as np
import pytest


CORRECTED_ROOT = Path(__file__).resolve().parents[1]
if str(CORRECTED_ROOT) not in sys.path:
    sys.path.insert(0, str(CORRECTED_ROOT))

from components.selectors import (  # noqa: E402
    CERTIFIABLE_AFTER_CRC,
    NONCERTIFIED_REFERENCE,
    FrozenGaussianCopulaMap,
    SelectionFamily,
    cheapest_first_chain,
    complementary_incremental_coverage_chain,
    cost_aware_independence_chain,
    empirical_all_miss_greedy_chain,
    empirical_all_miss_rate,
    exact_cardinality_oracle,
    exact_cost_oracle,
    gaussian_copula_greedy_chain,
    is_nested_family,
    marginal_depth_chain,
    mi_style_greedy_chain,
    pairwise_correlation_chain,
    pairwise_double_fault_chain,
    shrinkage_empirical_greedy_chain,
)


CORRECTNESS = np.array(
    [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
    ],
    dtype=int,
)
COSTS = np.array([4.0, 1.0, 2.0, 1.5])
TRUE_LABELS = np.array([0, 0, 1, 1, 0, 1, 0, 1])
PREDICTIONS = np.column_stack(
    [
        TRUE_LABELS,
        np.array([0, 1, 0, 1, 1, 0, 1, 0]),
        np.array([0, 0, 0, 1, 0, 1, 1, 1]),
        np.array([1, 1, 0, 0, 1, 0, 1, 0]),
    ]
)


def assert_complete_nested_chain(family: SelectionFamily, n_models: int) -> None:
    assert family.metadata.nested
    assert family.metadata.certification == CERTIFIABLE_AFTER_CRC
    assert not family.metadata.reference_only
    assert family.is_nested
    assert is_nested_family(family.members)
    assert family.members[0] == ()
    assert len(family.members) == n_models + 1
    assert len(family.order) == n_models
    assert sorted(family.order) == list(range(n_models))
    for depth, member in enumerate(family.members):
        assert len(member) == depth
        assert member == tuple(sorted(member))
        assert set(member) == set(family.order[:depth])


def build_all_nested_families() -> list[SelectionFamily]:
    frozen = FrozenGaussianCopulaMap(4, n_draws=2048, seed=91)
    return [
        marginal_depth_chain(CORRECTNESS),
        cost_aware_independence_chain(CORRECTNESS, COSTS),
        cheapest_first_chain(COSTS),
        empirical_all_miss_greedy_chain(CORRECTNESS),
        shrinkage_empirical_greedy_chain(CORRECTNESS, prior_strength=6.0),
        pairwise_double_fault_chain(CORRECTNESS),
        pairwise_correlation_chain(CORRECTNESS),
        mi_style_greedy_chain(PREDICTIONS, TRUE_LABELS),
        complementary_incremental_coverage_chain(CORRECTNESS),
        gaussian_copula_greedy_chain(
            CORRECTNESS.mean(axis=0), np.eye(4), frozen
        ),
    ]


def test_every_deployable_selector_returns_a_complete_nested_chain() -> None:
    families = build_all_nested_families()

    assert len({family.metadata.name for family in families}) == len(families)
    for family in families:
        assert_complete_nested_chain(family, CORRECTNESS.shape[1])


def test_every_selector_is_deterministic() -> None:
    first = build_all_nested_families()
    second = build_all_nested_families()

    for left, right in zip(first, second):
        assert left.metadata == right.metadata
        assert left.order == right.order
        assert left.members == right.members
        np.testing.assert_allclose(left.objective_values, right.objective_values)


def test_marginal_depth_uses_stable_descending_correctness() -> None:
    family = marginal_depth_chain([0.5, 0.9, 0.9, 0.1])

    assert family.order == (1, 2, 0, 3)
    assert family.objective_values[0] == 1.0
    assert np.all(np.diff(family.objective_values) <= 0.0)


def test_cost_aware_independence_uses_gain_per_cost() -> None:
    family = cost_aware_independence_chain(
        [0.9, 0.6, 0.4], [9.0, 2.0, 1.0]
    )

    assert family.order == (2, 1, 0)
    assert family.metadata.objective == "independence miss reduction per unit cost"


def test_cheapest_first_uses_index_as_the_tie_rule() -> None:
    family = cheapest_first_chain([3.0, 1.0, 1.0, 2.0])

    assert family.order == (1, 2, 3, 0)
    assert family.objective_values == (0.0, 1.0, 2.0, 4.0, 7.0)


def test_empirical_greedy_reduces_all_miss_monotonically() -> None:
    family = empirical_all_miss_greedy_chain(CORRECTNESS)

    expected = tuple(
        empirical_all_miss_rate(CORRECTNESS, member)
        for member in family.members
    )
    assert family.objective_values == expected
    assert np.all(np.diff(family.objective_values) <= 1e-12)
    assert family.objective_values[-1] == 0.0


def test_complementary_agent_is_the_explicit_incremental_coverage_form() -> None:
    empirical = empirical_all_miss_greedy_chain(CORRECTNESS)
    complementary = complementary_incremental_coverage_chain(CORRECTNESS)

    assert complementary.order == empirical.order
    np.testing.assert_allclose(
        complementary.objective_values,
        1.0 - np.asarray(empirical.objective_values),
    )


def test_zero_strength_shrinkage_matches_empirical_greedy() -> None:
    empirical = empirical_all_miss_greedy_chain(CORRECTNESS)
    shrinkage = shrinkage_empirical_greedy_chain(
        CORRECTNESS, prior_strength=0.0
    )

    assert shrinkage.order == empirical.order
    np.testing.assert_allclose(
        shrinkage.objective_values, empirical.objective_values
    )


def test_pairwise_baselines_choose_a_complement_after_the_best_model() -> None:
    correctness = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )

    double_fault = pairwise_double_fault_chain(correctness)
    correlation = pairwise_correlation_chain(correctness)

    assert double_fault.order[:2] == (0, 2)
    assert correlation.order[:2] == (0, 2)
    assert "pairwise" in double_fault.metadata.notes.lower()
    assert "pairwise" in correlation.metadata.notes.lower()


def test_mi_style_greedy_uses_label_information_and_documents_source() -> None:
    family = mi_style_greedy_chain(PREDICTIONS, TRUE_LABELS)

    assert family.order[0] == 0
    assert family.objective_values[1] == pytest.approx(1.0)
    assert np.all(np.diff(family.objective_values) >= -1e-12)
    source = inspect.getdoc(mi_style_greedy_chain)
    assert source is not None
    assert "arXiv:2602.08003" in source
    assert "mutual information" in source
    assert "approximation" in source


def test_mi_costs_change_the_greedy_information_ratio() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    predictions = np.column_stack(
        [
            labels,
            np.array([0, 0, 0, 1, 0, 1, 1, 1]),
        ]
    )

    no_cost = mi_style_greedy_chain(predictions, labels)
    with_cost = mi_style_greedy_chain(predictions, labels, costs=[100.0, 1.0])

    assert no_cost.order[0] == 0
    assert with_cost.order[0] == 1


def cardinality_counterexample() -> np.ndarray:
    """Return the weighted four-atom nonnested oracle example."""

    return np.array(
        [[0, 0, 1]] * 2
        + [[1, 0, 1]] * 3
        + [[0, 1, 0]] * 2
        + [[1, 1, 0]] * 3,
        dtype=int,
    )


def test_exact_cardinality_oracle_is_explicitly_nonnested_and_noncertified() -> None:
    oracle = exact_cardinality_oracle(cardinality_counterexample())

    assert oracle.members[1] == (0,)
    assert oracle.members[2] == (1, 2)
    assert not oracle.is_nested
    assert not oracle.metadata.nested
    assert oracle.metadata.reference_only
    assert oracle.metadata.certification == NONCERTIFIED_REFERENCE
    np.testing.assert_allclose(oracle.objective_values, [1.0, 0.4, 0.0, 0.0])


def test_exact_cost_oracle_finds_target_specific_cheapest_subset() -> None:
    oracle = exact_cost_oracle(
        cardinality_counterexample(), [5.0, 1.0, 1.0], alpha=0.0
    )

    assert oracle.feasible
    assert oracle.subset == (1, 2)
    assert oracle.total_cost == 2.0
    assert oracle.all_miss_rate == 0.0
    assert not oracle.metadata.nested
    assert oracle.metadata.reference_only
    assert oracle.metadata.certification == NONCERTIFIED_REFERENCE


def test_exact_cost_oracle_reports_infeasibility_instead_of_certifying_pool() -> None:
    oracle = exact_cost_oracle(
        np.zeros((5, 3), dtype=int), [1.0, 2.0, 3.0], alpha=0.5
    )

    assert not oracle.feasible
    assert oracle.subset is None
    assert np.isinf(oracle.total_cost)
    assert np.isinf(oracle.all_miss_rate)


def test_frozen_gaussian_map_reuses_one_design_seeded_latent_bank() -> None:
    mapper = FrozenGaussianCopulaMap(3, n_draws=4096, seed=123)
    initial_bank = mapper.design_latent_sample
    marginals = np.array([0.35, 0.55, 0.75])
    correlation = np.array(
        [
            [1.0, 0.25, -0.10],
            [0.25, 1.0, 0.20],
            [-0.10, 0.20, 1.0],
        ]
    )

    first = mapper.failure_sample(marginals, correlation)
    repeated = mapper.failure_sample(marginals, correlation)
    mapper.failure_sample([0.2, 0.4, 0.8], np.eye(3))
    after_other_query = mapper.failure_sample(marginals, correlation)

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first, after_other_query)
    np.testing.assert_array_equal(initial_bank, mapper.design_latent_sample)
    assert not first.flags.writeable
    assert not np.shares_memory(initial_bank, mapper.design_latent_sample)


def test_frozen_gaussian_map_is_seed_deterministic() -> None:
    first = FrozenGaussianCopulaMap(3, n_draws=512, seed=7)
    repeated = FrozenGaussianCopulaMap(3, n_draws=512, seed=7)
    other = FrozenGaussianCopulaMap(3, n_draws=512, seed=8)

    np.testing.assert_array_equal(
        first.design_latent_sample, repeated.design_latent_sample
    )
    assert not np.array_equal(
        first.design_latent_sample, other.design_latent_sample
    )


def test_frozen_gaussian_all_miss_values_are_nested_on_the_same_draws() -> None:
    mapper = FrozenGaussianCopulaMap(3, n_draws=8192, seed=19)
    marginals = [0.35, 0.55, 0.75]
    correlation = np.eye(3)

    empty = mapper.all_miss_probability(marginals, correlation, [])
    one = mapper.all_miss_probability(marginals, correlation, [0])
    two = mapper.all_miss_probability(marginals, correlation, [0, 1])
    three = mapper.all_miss_probability(marginals, correlation, [0, 1, 2])

    assert empty == 1.0
    assert empty >= one >= two >= three >= 0.0


def test_gaussian_copula_selector_reuses_mapper_and_returns_monotone_chain() -> None:
    mapper = FrozenGaussianCopulaMap(4, n_draws=4096, seed=41)
    first = gaussian_copula_greedy_chain(
        [0.2, 0.4, 0.6, 0.8], np.eye(4), mapper
    )
    second = gaussian_copula_greedy_chain(
        [0.2, 0.4, 0.6, 0.8], np.eye(4), mapper
    )

    assert_complete_nested_chain(first, 4)
    assert first == second
    assert np.all(np.diff(first.objective_values) <= 1e-12)


def test_selection_family_rejects_a_false_nestedness_claim() -> None:
    metadata = marginal_depth_chain([0.7, 0.5, 0.2]).metadata

    with pytest.raises(ValueError, match="nonnested"):
        SelectionFamily(
            members=((), (0,), (1, 2), (0, 1, 2)),
            order=(0, 1, 2),
            metadata=metadata,
        )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: marginal_depth_chain([0.5, np.nan]), "probabilities"),
        (
            lambda: cost_aware_independence_chain([0.5, 0.4], [1.0, 0.0]),
            "positive",
        ),
        (
            lambda: empirical_all_miss_greedy_chain([[1, 2], [0, 1]]),
            "binary",
        ),
        (
            lambda: shrinkage_empirical_greedy_chain(
                CORRECTNESS, prior_strength=-1.0
            ),
            "non-negative",
        ),
        (
            lambda: pairwise_double_fault_chain(np.ones((3, 0))),
            "non-empty",
        ),
    ],
)
def test_invalid_inputs_fail_loudly(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_gaussian_map_rejects_invalid_correlation_and_subset() -> None:
    mapper = FrozenGaussianCopulaMap(2, n_draws=128, seed=0)

    with pytest.raises(ValueError, match="symmetric"):
        mapper.failure_sample([0.5, 0.5], [[1.0, 0.2], [0.1, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        mapper.failure_sample([0.5, 0.5], [[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(IndexError, match="out-of-range"):
        mapper.all_miss_probability([0.5, 0.5], np.eye(2), [2])
