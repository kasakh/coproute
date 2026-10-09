"""Tests for the ISSUE-77 chain discounted stopping score."""
from __future__ import annotations

import numpy as np
import pytest

from components.chain_discount import (
    chain_discount_diagnostics,
    chain_discounted_scores,
    effective_chain_discount,
    fit_chain_discount,
)
from components.racer_wrapper import fixed_order_factorized_scores


def _design() -> np.ndarray:
    return np.asarray(
        [
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=np.int8,
    )


def test_chain_counts_and_count_shrinkage_are_exact() -> None:
    profile = fit_chain_discount(_design(), [0, 1, 2])
    assert profile.order == (0, 1, 2)
    assert profile.survivor_counts.tolist() == [4, 3, 1, 0]
    np.testing.assert_allclose(profile.chain_conditional_miss, [0.75, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(profile.empirical_prefix_miss, [1.0, 0.75, 0.25, 0.0])
    np.testing.assert_allclose(
        profile.independent_prefix_miss, [1.0, 0.75, 0.375, 0.1875]
    )
    np.testing.assert_allclose(profile.raw_discount, [1.0, 1.0, 2.0 / 3.0, 0.0])
    np.testing.assert_allclose(
        effective_chain_discount(profile, minimum_survivor_count=2),
        [1.0, 1.0, 5.0 / 6.0, 1.0],
    )


def test_discount_fully_off_reproduces_fixed_order_product_exactly() -> None:
    rng = np.random.default_rng(77)
    design = rng.integers(0, 2, size=(31, 7), dtype=np.int8)
    probabilities = rng.uniform(0.01, 0.99, size=(19, 7))
    order = [3, 0, 6, 2, 5, 1, 4]
    profile = fit_chain_discount(design, order)
    endpoint = chain_discounted_scores(
        probabilities, profile, minimum_survivor_count=0
    )
    expected = fixed_order_factorized_scores(
        probabilities, order, max_depth=len(order)
    )
    np.testing.assert_array_equal(endpoint, expected)


def test_scores_are_nested_even_when_raw_adjustment_would_invert() -> None:
    design = np.asarray(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.int8,
    )
    order = [0, 1, 2]
    profile = fit_chain_discount(design, order)
    assert profile.raw_discount[2] == pytest.approx(2.0)
    probabilities = np.asarray([[0.10, 0.10, 0.40], [0.80, 0.05, 0.30]])
    scores = chain_discounted_scores(
        probabilities, profile, minimum_survivor_count=1
    )
    ordered = scores[:, order]
    assert np.all(np.diff(ordered, axis=1) >= 0.0)
    assert ordered[0, 2] == pytest.approx(ordered[0, 1])


def test_diagnostics_mark_off_endpoint_and_thin_prefixes() -> None:
    profile = fit_chain_discount(_design(), [0, 1, 2])
    off = chain_discount_diagnostics(profile, minimum_survivor_count=0)
    assert off["discount_fully_off"] is True
    assert off["effective_discount"] == [1.0, 1.0, 1.0, 1.0]
    assert off["shrinkage_engaged_prefix_count"] == 0

    active = chain_discount_diagnostics(profile, minimum_survivor_count=2)
    assert active["discount_fully_off"] is False
    assert active["shrinkage_engaged_by_prefix"] == [False, False, True, True]
    assert active["shrinkage_engaged_prefix_count"] == 2


@pytest.mark.parametrize(
    ("design", "order", "message"),
    [
        (np.ones((2, 3)), [0, 1], "full model permutation"),
        (np.ones((2, 3)), [0, 0, 2], "full model permutation"),
        (np.asarray([[0, 2, 1]]), [0, 1, 2], "binary"),
    ],
)
def test_invalid_profiles_fail_loudly(design, order, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit_chain_discount(design, order)


def test_invalid_shrinkage_schedule_fails_loudly() -> None:
    profile = fit_chain_discount(_design(), [0, 1, 2])
    with pytest.raises(ValueError, match="non-negative"):
        chain_discounted_scores(
            np.full((2, 3), 0.5), profile, minimum_survivor_count=-1
        )
    with pytest.raises(TypeError, match="integer"):
        effective_chain_discount(profile, minimum_survivor_count=1.5)
