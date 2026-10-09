"""Synthetic tests for rare-tail and co-failure calibration tools."""

from pathlib import Path
import math
import sys

import numpy as np
import pytest


CORRECTED_ROOT = Path(__file__).resolve().parents[1]
if str(CORRECTED_ROOT) not in sys.path:
    sys.path.insert(0, str(CORRECTED_ROOT))

from components.tail_calibration import (  # noqa: E402
    clopper_pearson_interval,
    empirical_all_miss_by_subset,
    independence_all_miss_by_subset,
    rare_tail_calibration_table,
    simultaneous_clopper_pearson_intervals,
    underpricing_ratio,
)


def test_clopper_pearson_matches_known_exact_values_and_boundaries() -> None:
    lower, upper = clopper_pearson_interval(5, 10)
    assert lower == pytest.approx(0.18708602844739852)
    assert upper == pytest.approx(0.8129139715526015)

    zero_lower, zero_upper = clopper_pearson_interval(0, 10)
    full_lower, full_upper = clopper_pearson_interval(10, 10)
    assert zero_lower == 0.0
    assert zero_upper == pytest.approx(0.3084971078187608)
    assert full_lower == pytest.approx(1.0 - zero_upper)
    assert full_upper == 1.0


def test_clopper_pearson_rejects_invalid_binomial_inputs() -> None:
    with pytest.raises(ValueError, match="n_trials must be positive"):
        clopper_pearson_interval(0, 0)
    with pytest.raises(ValueError, match="between zero and n_trials"):
        clopper_pearson_interval(11, 10)
    with pytest.raises(ValueError, match="event_count must be an integer"):
        clopper_pearson_interval(1.5, 10)
    with pytest.raises(ValueError, match="strictly between zero and one"):
        clopper_pearson_interval(1, 10, confidence_level=1.0)


def test_empirical_and_independence_estimates_expose_cofailure_dependence() -> None:
    misses = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    subsets = [(0, 1), (0, 2), (2,)]

    empirical = empirical_all_miss_by_subset(misses, subsets)
    independent = independence_all_miss_by_subset(misses, subsets)

    assert empirical[(0, 1)].all_miss_count == 2
    assert empirical[(0, 1)].probability == 0.5
    assert independent[(0, 1)] == 0.25
    assert empirical[(0, 2)].probability == 0.0
    assert independent[(0, 2)] == 0.25
    assert empirical[(2,)].probability == independent[(2,)] == 0.5


def test_underpricing_ratio_handles_calibration_and_zero_predictions() -> None:
    assert underpricing_ratio(0.12, 0.03) == pytest.approx(4.0)
    assert underpricing_ratio(0.0, 0.0) == 1.0
    assert math.isinf(underpricing_ratio(0.01, 0.0))
    with pytest.raises(ValueError, match="predicted"):
        underpricing_ratio(0.1, -0.01)


def test_simultaneous_intervals_are_bonferroni_wider_than_pointwise() -> None:
    counts = [0, 5, 20]
    pointwise = [clopper_pearson_interval(count, 40) for count in counts]
    simultaneous = simultaneous_clopper_pearson_intervals(counts, 40)

    for point, family in zip(pointwise, simultaneous):
        assert family[0] <= point[0]
        assert family[1] >= point[1]
    assert simultaneous[1][0] < pointwise[1][0]
    assert simultaneous[1][1] > pointwise[1][1]


def test_rare_tail_table_combines_sizes_selected_subsets_and_predictions() -> None:
    misses = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    subsets = ((0,), (1,), (2,), (0, 1), (0, 2))
    gaussian = {subset: 0.10 + 0.01 * index for index, subset in enumerate(subsets)}
    student_t = {subset: 0.12 + 0.01 * index for index, subset in enumerate(subsets)}

    rows = rare_tail_calibration_table(
        misses,
        subset_sizes=[1],
        selected_subsets=[(1, 0), (0, 2), (0, 1)],
        gaussian_predictions=gaussian,
        student_t_predictions=student_t,
        family_confidence_level=0.95,
    )

    assert tuple(row.subset for row in rows) == subsets
    assert all(row.marginal_confidence_level == pytest.approx(0.99) for row in rows)
    pair = next(row for row in rows if row.subset == (0, 1))
    assert pair.subset_size == 2
    assert pair.n_queries == 8
    assert pair.all_miss_count == 2
    assert pair.empirical_probability == 0.25
    assert pair.independence_probability == pytest.approx(3 / 8 * 2 / 8)
    assert pair.gaussian_probability == gaussian[(0, 1)]
    assert pair.student_t_probability == student_t[(0, 1)]
    assert pair.independence_underpricing_ratio == pytest.approx(8 / 3)
    assert pair.gaussian_underpricing_ratio == pytest.approx(
        0.25 / gaussian[(0, 1)]
    )
    assert pair.interval_lower <= pair.empirical_probability <= pair.interval_upper


def test_rare_tail_table_reports_infinite_underpricing_for_zero_prediction() -> None:
    misses = np.array([[1, 1], [1, 1], [0, 0], [0, 0]])
    rows = rare_tail_calibration_table(
        misses,
        selected_subsets=[(0, 1)],
        gaussian_predictions={(0, 1): 0.0},
        student_t_predictions={(0, 1): 0.25},
    )

    assert len(rows) == 1
    assert math.isinf(rows[0].gaussian_underpricing_ratio)
    assert rows[0].student_t_underpricing_ratio == pytest.approx(2.0)


def test_table_validates_subsets_prediction_coverage_and_probabilities() -> None:
    misses = np.array([[1, 0], [0, 1]])
    with pytest.raises(KeyError, match="student_t_predictions is missing"):
        rare_tail_calibration_table(
            misses,
            selected_subsets=[(0, 1)],
            gaussian_predictions={(0, 1): 0.1},
            student_t_predictions={},
        )
    with pytest.raises(ValueError, match="finite probabilities"):
        rare_tail_calibration_table(
            misses,
            selected_subsets=[(0, 1)],
            gaussian_predictions={(0, 1): 1.1},
            student_t_predictions={(0, 1): 0.1},
        )
    with pytest.raises(IndexError, match="out-of-range"):
        empirical_all_miss_by_subset(misses, [(0, 2)])
    with pytest.raises(ValueError, match="duplicate"):
        independence_all_miss_by_subset(misses, [(0, 0)])
    with pytest.raises(ValueError, match="zero and one"):
        empirical_all_miss_by_subset([[0, 2]], [(0,)])


def test_missing_scipy_fails_loudly_without_approximate_interval(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.delitem(sys.modules, "scipy.stats", raising=False)

    with pytest.raises(ImportError, match="Exact Clopper-Pearson.*SciPy"):
        clopper_pearson_interval(1, 2)
