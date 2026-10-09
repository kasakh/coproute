"""Regression tests for prefix-preserving smoothing and E2 grid floors."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from components.racer_protocol import smooth_nonconformity_scores  # noqa: E402
from components.racer_wrapper import (  # noqa: E402
    CORRELATION_HEURISTIC,
    FACTORIZED_INDEPENDENCE,
    PAIRWISE_HEURISTIC,
    REGULARIZED_CORRELATION,
    sequential_scores,
)
from scripts import design_bprime_e2_run as e2  # noqa: E402


def test_independent_smoothing_can_cut_a_tied_plateau_out_of_prefix_order() -> None:
    raw = np.array([[0.2, 0.2, 0.2, 0.8]])

    legacy = smooth_nonconformity_scores(raw, seed=0, split_seed=0)
    threshold = float(np.sort(legacy[0, :3])[1])
    selected = np.flatnonzero(legacy[0] <= threshold).tolist()

    assert selected != [0, 1]
    assert np.argsort(legacy[0], kind="stable").tolist() == [2, 1, 0, 3]


def test_prefix_preserving_smoothing_is_monotone_and_deterministic() -> None:
    raw = np.array(
        [
            [0.2, 0.2, 0.2, 0.8],
            [0.0, 5e-8, 0.4, 0.4],
        ]
    )

    first = smooth_nonconformity_scores(
        raw, seed=1, split_seed=2, preserve_prefixes=True
    )
    repeated = smooth_nonconformity_scores(
        raw, seed=99, split_seed=100, preserve_prefixes=True
    )

    np.testing.assert_array_equal(first, repeated)
    for raw_row, smoothed_row in zip(raw, first):
        fitted_sequence = np.argsort(raw_row, kind="stable")
        assert np.argsort(smoothed_row, kind="stable").tolist() == fitted_sequence.tolist()
        assert np.all(np.diff(smoothed_row[fitted_sequence]) > 0.0)


def test_legacy_independent_smoothing_remains_available_by_default() -> None:
    raw = np.zeros((4, 3))

    implicit = smooth_nonconformity_scores(raw, seed=17, split_seed=3)
    explicit = smooth_nonconformity_scores(
        raw, seed=17, split_seed=3, preserve_prefixes=False
    )

    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.parametrize(
    "arm,off_hyper",
    [
        (PAIRWISE_HEURISTIC, {"adjustment_strength": 0.0}),
        (CORRELATION_HEURISTIC, {"adjustment_strength": 0.0}),
        (REGULARIZED_CORRELATION, {"ridge": 1.0}),
    ],
)
def test_adjustment_off_grid_endpoint_equals_exact_independence(
    arm: str, off_hyper: dict,
) -> None:
    probabilities = np.array(
        [
            [0.85, 0.60, 0.35, 0.15],
            [0.40, 0.90, 0.55, 0.20],
            [0.51, 0.49, 0.70, 0.30],
        ]
    )
    design = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    )
    exact = sequential_scores(
        probabilities,
        dependence=FACTORIZED_INDEPENDENCE,
        max_depth=4,
    )
    candidate = e2._sequential_scores_for_grid_candidate(
        probabilities,
        arm=arm,
        design_labels=design,
        max_depth=4,
        hyper=off_hyper,
        fixed_tally_order=(0, 1, 2, 3),
    )

    np.testing.assert_allclose(candidate, exact, rtol=0.0, atol=1e-15)
    assert off_hyper in e2.WRAPPER_HYPER_GRIDS[arm]
