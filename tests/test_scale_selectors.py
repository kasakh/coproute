"""Tests for the scale-aware estimator-family arms used by the E1 grid."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.scale_selectors import (  # noqa: E402
    ARMS,
    build_order,
    empirical_tally_order,
    empirical_tally_order_lazy,
)

NAMES = tuple(f"family{i % 5}__variant{i}" for i in range(40))


def _design(seed: int = 0, rows: int = 60, models: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.random((rows, models)) > 0.45
    return base.astype(int)


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_returns_a_valid_capped_order(arm: str) -> None:
    design = _design()
    order = build_order(arm, design, model_names=NAMES, max_depth=8, seed=1)
    assert len(order) == 8, f"{arm} returned {len(order)} picks"
    assert len(set(order)) == len(order), f"{arm} repeated a model"
    assert all(0 <= index < design.shape[1] for index in order)


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_is_deterministic_for_a_seed(arm: str) -> None:
    design = _design()
    first = build_order(arm, design, model_names=NAMES, max_depth=6, seed=7)
    second = build_order(arm, design, model_names=NAMES, max_depth=6, seed=7)
    assert first == second, f"{arm} is not reproducible at a fixed seed"


def test_marginal_order_is_exactly_accuracy_ranking() -> None:
    design = np.array([[1, 1, 0], [1, 0, 0], [1, 0, 1]])
    order = build_order("marginal", design, model_names=("a", "b", "c"), max_depth=3)
    assert order[0] == 0


def test_empirical_tally_prefers_a_complementary_second_pick() -> None:
    """A weaker but complementary model must beat a strong duplicate."""

    # Model 0 is strong. Model 1 duplicates model 0. Model 2 is weaker but covers
    # exactly the rows model 0 misses.
    design = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    order = build_order("empirical_tally", design, model_names=("a", "b", "c"), max_depth=2)
    assert order[0] == 0
    assert order[1] == 2, "greedy incremental coverage must pick the complement"


def test_marginal_order_picks_the_duplicate_instead() -> None:
    """The floor cannot see complementarity, which is why it is the floor."""

    design = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    order = build_order("marginal", design, model_names=("a", "b", "c"), max_depth=2)
    assert set(order) == {0, 1}


def test_lineage_spreads_across_families_before_repeating_one() -> None:
    design = _design(seed=3, rows=40, models=10)
    names = ("alpha_1", "alpha_2", "beta_1", "beta_2", "gamma_1",
             "gamma_2", "delta_1", "delta_2", "eps_1", "eps_2")
    order = build_order("lineage", design, model_names=names, max_depth=5)
    prefixes = [names[i].split("_")[0] for i in order]
    assert len(set(prefixes)) == 5, "first five picks must come from five families"


def test_depth_cap_is_respected_on_a_wide_pool() -> None:
    design = _design(seed=5, rows=40, models=500)
    names = tuple(f"m{i}" for i in range(500))
    for arm in ("empirical_tally", "gaussian_copula", "factor_copula", "lineage"):
        order = build_order(arm, design, model_names=names, max_depth=4, seed=2, n_draws=200)
        assert len(order) == 4, arm


def test_random_controls_have_distinct_information_contracts() -> None:
    design = np.zeros((100, 4), dtype=int)
    design[:, 0] = 1
    design[:70, 1] = 1
    design[:30, 2] = 1
    names = ("a", "b", "c", "d")

    seeded = build_order(
        "seeded_random", design, model_names=names, max_depth=4, seed=19
    )
    weighted = build_order(
        "accuracy_weighted_random", design, model_names=names, max_depth=4, seed=19
    )
    assert sorted(seeded) == [0, 1, 2, 3]
    assert sorted(weighted) == [0, 1, 2, 3]
    assert build_order(
        "seeded_random", 1 - design, model_names=names, max_depth=4, seed=19
    ) == seeded
    assert weighted == build_order(
        "accuracy_weighted_random", design, model_names=names, max_depth=4, seed=19
    )


def test_unknown_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        build_order("not-an-arm", _design(), model_names=NAMES)


def test_lazy_empirical_tally_is_exact_across_random_designs() -> None:
    for seed in range(10):
        design = _design(seed=seed, rows=80, models=60)
        for depth in (1, 5, 15, 60):
            eager = empirical_tally_order(design, depth, laplace=0.5)
            lazy = empirical_tally_order_lazy(design, depth, laplace=0.5)
            assert lazy == eager, (seed, depth)


def test_lazy_empirical_tally_preserves_ties_and_early_fill() -> None:
    design = np.asarray(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    assert empirical_tally_order_lazy(design, 5) == empirical_tally_order(design, 5)


def test_lazy_empirical_tally_reports_fewer_gain_evaluations_on_redundant_pool() -> None:
    rng = np.random.default_rng(83)
    base = (rng.random((200, 20)) < 0.5).astype(np.int8)
    design = np.repeat(base, 10, axis=1)
    eager = empirical_tally_order(design, 20)
    lazy, diagnostics = empirical_tally_order_lazy(
        design, 20, return_diagnostics=True
    )
    assert lazy == eager
    assert diagnostics["candidate_gain_evaluations"] < diagnostics[
        "eager_candidate_gain_evaluations"
    ]
