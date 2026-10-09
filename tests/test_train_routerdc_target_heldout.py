from __future__ import annotations

import numpy as np
import pytest

from scripts.train_routerdc_target_heldout import deterministic_kmeans, repeat_rows


def test_deterministic_kmeans_is_seed_stable_and_complete() -> None:
    features = np.asarray([
        [1.0, 0.0], [0.9, 0.1],
        [0.0, 1.0], [0.1, 0.9],
        [-1.0, 0.0], [-0.9, -0.1],
    ])
    left = deterministic_kmeans(features, 3, 17)
    right = deterministic_kmeans(features, 3, 17)
    assert np.array_equal(left, right)
    assert set(left.tolist()) == {0, 1, 2}


def test_deterministic_kmeans_rejects_zero_rows() -> None:
    with pytest.raises(ValueError, match="zero row"):
        deterministic_kmeans(np.asarray([[0.0, 0.0], [1.0, 0.0]]), 2, 1)


def test_repeat_rows_is_deterministic_and_exact_length() -> None:
    rows = np.asarray([4, 7, 9])
    assert repeat_rows(rows, 8).tolist() == [4, 7, 9, 4, 7, 9, 4, 7]
