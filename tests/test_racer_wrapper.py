"""Tests for the E2 wrapper: the three admissibility properties, asserted.

The construction is only usable if RACER Theorem 4.3 still applies, which needs
nestedness in lambda. These tests pin exactly that, plus the independence
degeneracy that stops us overclaiming containment of RACER.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.racer_wrapper import (  # noqa: E402
    augment_with_racer_null,
    sequential_scores,
)

DESIGN = np.array(
    [
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [1, 0, 1, 1],
    ]
)


def _probabilities(seed: int = 0, rows: int = 12, models: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.95, size=(rows, models))


@pytest.mark.parametrize(
    "dependence", ["independence", "pairwise_count", "error_correlation", "gaussian_copula"]
)
def test_property_one_top_pick_always_scores_zero(dependence: str) -> None:
    """PROPERTY 1: every query's first pick has score exactly zero."""

    scores = sequential_scores(
        _probabilities(), dependence=dependence, design=DESIGN
    )
    assert np.allclose(scores.min(axis=1), 0.0), dependence


@pytest.mark.parametrize(
    "dependence", ["independence", "pairwise_count", "error_correlation", "gaussian_copula"]
)
def test_property_two_sublevel_sets_are_prefixes(dependence: str) -> None:
    """PROPERTY 2: scores are non-decreasing along the greedy order.

    That is exactly the nestedness Theorem 4.3 requires. Verified by checking that
    every distinct threshold produces a set that is a prefix under the score's own
    ordering, with no gaps.
    """

    scores = sequential_scores(
        _probabilities(seed=3), dependence=dependence, design=DESIGN
    )
    for row in scores:
        order = np.argsort(row, kind="stable")
        ordered = row[order]
        assert np.all(np.diff(ordered) >= -1e-12)
        for threshold in np.unique(ordered):
            selected = set(np.flatnonzero(row <= threshold + 1e-12).tolist())
            size = len(selected)
            assert selected == set(order[:size].tolist()), (
                f"{dependence}: threshold {threshold} did not select a prefix"
            )


@pytest.mark.parametrize(
    "dependence", ["independence", "pairwise_count", "error_correlation", "gaussian_copula"]
)
def test_property_three_set_size_varies_by_query(dependence: str) -> None:
    """PROPERTY 3: at ONE global threshold, prefix length differs across queries.

    This is the property every fixed-depth arm in this program lacks.
    """

    probabilities = np.array(
        [
            [0.95, 0.90, 0.10, 0.05],
            [0.30, 0.28, 0.26, 0.24],
        ]
    )
    scores = sequential_scores(probabilities, dependence=dependence, design=DESIGN)
    threshold = 0.6
    sizes = [(row <= threshold).sum() for row in scores]
    assert len(set(sizes)) > 1, f"{dependence}: sizes {sizes} did not vary by query"


def test_independence_degenerates_to_marginal_ordering() -> None:
    """Sanity check, NOT a containment claim.

    Under independence the greedy order must be descending marginal accuracy.
    """

    probabilities = np.array([[0.2, 0.9, 0.5, 0.7]])
    scores = sequential_scores(probabilities, dependence="independence")
    order = np.argsort(scores[0], kind="stable")
    assert list(order) == [1, 3, 2, 0]


def test_independence_residual_is_the_product_of_miss_probabilities() -> None:
    probabilities = np.array([[0.5, 0.8]])
    scores = sequential_scores(probabilities, dependence="independence")
    # First pick is model 1 (p=0.8), score 0. Second pick score is 1 - 0.2 = 0.8.
    assert scores[0, 1] == pytest.approx(0.0)
    assert scores[0, 0] == pytest.approx(1.0 - 0.2, abs=1e-9)


def test_dependence_can_reorder_away_from_marginal() -> None:
    """A complementary weaker model must be able to outrank a strong duplicate."""

    # Models 0 and 1 always fail together in the design; model 2 is complementary.
    design = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    probabilities = np.array([[0.80, 0.78, 0.40]])
    marginal = sequential_scores(probabilities, dependence="independence")
    counted = sequential_scores(probabilities, dependence="pairwise_count", design=design)
    assert list(np.argsort(marginal[0], kind="stable"))[:2] == [0, 1]
    assert list(np.argsort(counted[0], kind="stable"))[1] == 2, (
        "pairwise counting must promote the complement to second place"
    )


def test_null_column_is_racer_own_and_appended_last() -> None:
    probabilities = np.array([[0.2, 0.9, 0.5]])
    scores = sequential_scores(probabilities, dependence="independence")
    augmented = augment_with_racer_null(probabilities, scores)
    assert augmented.shape == (1, 4)
    assert augmented[0, -1] == pytest.approx(0.9)
    assert np.allclose(augmented[0, :-1], scores[0])


def test_missing_design_is_refused_for_dependence_models() -> None:
    for dependence in ("pairwise_count", "error_correlation", "gaussian_copula"):
        with pytest.raises(ValueError, match="requires design labels"):
            sequential_scores(_probabilities(), dependence=dependence)


def test_out_of_range_probabilities_are_refused() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sequential_scores(np.array([[1.5, 0.2]]), dependence="independence")
