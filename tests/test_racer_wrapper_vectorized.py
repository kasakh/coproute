"""Equivalence tests for the vectorized wrapper score path.

The greedy inner loop over candidate models was replaced with an array operation so
a pool of a hundred models is tractable. These tests assert the optimisation did not
change the model: the per-candidate factors agree with the scalar function, and the
score matrix agrees with a reference scalar implementation kept here for the purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.racer_wrapper import (
    EPSILON,
    _conditional_factors_vectorized,
    _conditional_miss_factor,
    _failure_correlation,
    _pairwise_joint_miss,
    is_prefix_family,
    sequential_scores,
)


def _reference_sequential_scores(probabilities, *, dependence, design, max_depth=None):
    """The pre-optimisation scalar implementation, preserved as the reference."""
    correctness = np.asarray(probabilities, dtype=float)
    n_queries, n_models = correctness.shape
    miss = 1.0 - correctness
    depth = int(max_depth) if max_depth else n_models
    joint = None
    correlation = None
    if dependence == "pairwise_count":
        joint = _pairwise_joint_miss(design)
    elif dependence in ("error_correlation", "gaussian_copula"):
        correlation = _failure_correlation(design)
    scores = np.ones((n_queries, n_models), dtype=float)
    for row in range(n_queries):
        remaining = set(range(n_models))
        selected: list[int] = []
        residual = 1.0
        for _ in range(min(depth, n_models)):
            best_index = None
            best_residual = None
            for candidate in sorted(remaining):
                factor = _conditional_miss_factor(
                    candidate, selected,
                    miss_probability=miss[row], joint=joint, correlation=correlation,
                )
                candidate_residual = residual * factor
                if best_residual is None or candidate_residual < best_residual - EPSILON:
                    best_residual = candidate_residual
                    best_index = candidate
            if best_index is None:
                break
            scores[row, best_index] = 1.0 - residual
            residual = float(best_residual)
            selected.append(best_index)
            remaining.discard(best_index)
    return scores


def _fixture(seed=0, n_queries=12, n_models=9):
    rng = np.random.default_rng(seed)
    probabilities = rng.uniform(0.05, 0.95, size=(n_queries, n_models))
    design = (rng.uniform(size=(60, n_models)) > 0.4).astype(int)
    return probabilities, design


@pytest.mark.parametrize("dependence", ["independence", "pairwise_count", "error_correlation", "gaussian_copula"])
def test_vectorized_factors_match_scalar(dependence):
    probabilities, design = _fixture(seed=1)
    miss = 1.0 - probabilities[0]
    joint = _pairwise_joint_miss(design) if dependence == "pairwise_count" else None
    correlation = (
        _failure_correlation(design)
        if dependence in ("error_correlation", "gaussian_copula") else None
    )
    for selected in ([], [0], [0, 3], [2, 5, 7]):
        candidates = np.asarray(
            [m for m in range(probabilities.shape[1]) if m not in selected], dtype=int
        )
        vectorized = _conditional_factors_vectorized(
            candidates, selected, miss_probability=miss, joint=joint, correlation=correlation
        )
        scalar = np.asarray([
            _conditional_miss_factor(
                int(c), selected, miss_probability=miss, joint=joint, correlation=correlation
            )
            for c in candidates
        ])
        assert np.allclose(vectorized, scalar, rtol=0, atol=1e-12)


@pytest.mark.parametrize("dependence", ["independence", "pairwise_count", "error_correlation", "gaussian_copula"])
def test_scores_match_reference_implementation(dependence):
    probabilities, design = _fixture(seed=2)
    fast = sequential_scores(probabilities, dependence=dependence, design=design)
    reference = _reference_sequential_scores(
        probabilities, dependence=dependence, design=design
    )
    assert np.allclose(fast, reference, rtol=0, atol=1e-12)


@pytest.mark.parametrize("dependence", ["pairwise_count", "error_correlation"])
def test_scores_match_reference_with_depth_cap(dependence):
    probabilities, design = _fixture(seed=3)
    fast = sequential_scores(probabilities, dependence=dependence, design=design, max_depth=4)
    reference = _reference_sequential_scores(
        probabilities, dependence=dependence, design=design, max_depth=4
    )
    assert np.allclose(fast, reference, rtol=0, atol=1e-12)


def test_admissibility_properties_survive_vectorization():
    probabilities, design = _fixture(seed=4)
    scores = sequential_scores(probabilities, dependence="pairwise_count", design=design)
    # property 1: every row has a zero-score top pick
    assert np.allclose(scores.min(axis=1), 0.0, atol=1e-12)
    # property 2: thresholding yields prefixes of the induced order
    for row in range(scores.shape[0]):
        order = list(np.argsort(scores[row], kind="stable"))
        assert is_prefix_family(scores[row], order)
    # property 3: the prefix length varies by query at one global threshold
    threshold = float(np.median(scores))
    sizes = (scores <= threshold).sum(axis=1)
    assert len(set(sizes.tolist())) > 1


def test_vectorized_is_faster_on_a_larger_pool():
    import time
    rng = np.random.default_rng(5)
    probabilities = rng.uniform(0.05, 0.95, size=(40, 60))
    design = (rng.uniform(size=(80, 60)) > 0.4).astype(int)
    start = time.perf_counter()
    fast = sequential_scores(probabilities, dependence="pairwise_count", design=design)
    fast_seconds = time.perf_counter() - start
    start = time.perf_counter()
    reference = _reference_sequential_scores(
        probabilities, dependence="pairwise_count", design=design
    )
    reference_seconds = time.perf_counter() - start
    assert np.allclose(fast, reference, rtol=0, atol=1e-12)
    # A correctness-preserving speedup is the point; assert it is at least not slower.
    assert fast_seconds <= reference_seconds
