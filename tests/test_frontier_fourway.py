"""Tests for the four-way frontier driver primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.frontier_fourway import (
    _calibrated_depth,
    _cumulative_miss,
    _split_indices,
)


def test_split_indices_disjoint_and_covers():
    parts = _split_indices(
        n_queries=1000,
        design_budget=200,
        cal_ratio=0.4,
        val_ratio=0.3,
        test_ratio=0.3,
        seed=42,
    )
    all_idx = np.concatenate([parts["design"], parts["cal"], parts["val"], parts["test"]])
    assert len(np.unique(all_idx)) == 1000
    assert len(parts["design"]) == 200
    assert len(parts["cal"]) + len(parts["val"]) + len(parts["test"]) == 800
    # cal ratio 0.4 of 800 = 320, val and test each 240
    assert len(parts["cal"]) == 320
    assert abs(len(parts["val"]) - 240) <= 1
    assert abs(len(parts["test"]) - 240) <= 1


def test_split_indices_deterministic():
    a = _split_indices(1000, 100, 0.5, 0.25, 0.25, seed=17)
    b = _split_indices(1000, 100, 0.5, 0.25, 0.25, seed=17)
    for key in ("design", "cal", "val", "test"):
        assert np.array_equal(a[key], b[key])


def test_split_indices_rejects_oversize_design():
    with pytest.raises(ValueError):
        _split_indices(100, 100, 0.5, 0.25, 0.25, seed=0)


def test_cumulative_miss_shape_and_values():
    binary = np.array([
        [1, 0, 1, 0],  # query 0: model 0 correct
        [0, 0, 0, 1],  # query 1: models 0..2 miss, model 3 correct
        [0, 0, 0, 0],  # query 2: all miss
    ])
    order = [0, 1, 2, 3]
    cum = _cumulative_miss(binary, order)
    assert cum.shape == (3, 4)
    # query 0 already covered at depth 1
    assert not cum[0, 0]
    assert not cum[0, 1]
    # query 1 misses at depths 1..3, covered at depth 4
    assert cum[1, 0] and cum[1, 1] and cum[1, 2] and not cum[1, 3]
    # query 2 misses everywhere
    assert cum[2, 0] and cum[2, 1] and cum[2, 2] and cum[2, 3]


def test_calibrated_depth_returns_smallest_valid():
    # All-miss matrix over 100 cal queries, 5 depths
    # depth 1: 80 miss (0.8), depth 2: 30 miss (0.30), depth 3: 5 miss (0.05), depth 4: 2, depth 5: 0
    cal = np.zeros((100, 5), dtype=bool)
    cal[:80, 0] = True
    cal[:30, 1] = True
    cal[:5, 2] = True
    cal[:2, 3] = True
    # alpha=0.10 with n_cal=100: target = (0.10 * 101 - 1)/100 = 0.091.
    # depth 3 has 0.05 <= 0.091, depth 2 has 0.30 > 0.091. So k=3.
    assert _calibrated_depth(cal, 0.10) == 3
    # alpha=0.03 with n_cal=100: target = (0.03 * 101 - 1)/100 = 0.0203.
    # depth 3 has 0.05 > 0.0203, depth 4 has 0.02 <= 0.0203. So k=4.
    assert _calibrated_depth(cal, 0.03) == 4
    # alpha=0.5 always feasible at depth 1 (0.8? no); at 0.5 target is (0.5*101-1)/100 = 0.495.
    # depth 1 has 0.8 > 0.495. depth 2 has 0.30 <= 0.495. So k=2.
    assert _calibrated_depth(cal, 0.5) == 2


def test_calibrated_depth_infeasible_returns_none():
    cal = np.ones((50, 3), dtype=bool)  # 100 pct miss at every depth
    assert _calibrated_depth(cal, 0.05) is None


def test_calibrated_depth_handles_negative_target():
    cal = np.ones((5, 2), dtype=bool)
    # alpha=0.01 with n_cal=5: target = (0.01*6 - 1)/5 = -0.188 -> None
    assert _calibrated_depth(cal, 0.01) is None
