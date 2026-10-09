"""Synthetic-array tests for the exact RACER protocol."""

from pathlib import Path
import sys

import numpy as np


def approx(expected: float, *, rtol: float = 1e-7, atol: float = 1e-12):
    """Return a dependency-free scalar approximate matcher."""

    class ApproximateValue:
        def __eq__(self, actual: object) -> bool:
            return bool(np.isclose(actual, expected, rtol=rtol, atol=atol))

    return ApproximateValue()


CORRECTED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORRECTED_ROOT))

from components.racer_protocol import (  # noqa: E402
    RACERProtocol,
    augmented_misrouting_loss,
    augment_ground_truth,
    augment_router_scores,
    deterministic_split_indices,
    ground_truth_critical_scores,
    misrouting_loss,
    prediction_sets,
    raw_nonconformity_scores,
    safe_action_loss,
    select_crc_threshold,
    select_standard_crc_threshold,
    smooth_nonconformity_scores,
    standard_all_miss_loss,
    standard_ground_truth_critical_scores,
)


def test_augmented_null_score_is_one_minus_standard_maximum() -> None:
    base = np.array([[0.8, 0.3, 0.1], [0.2, 0.4, 0.35]])

    augmented = augment_router_scores(base)

    np.testing.assert_allclose(
        augmented,
        np.array([[0.8, 0.3, 0.1, 0.2], [0.2, 0.4, 0.35, 0.6]]),
    )


def test_augmented_ground_truth_uses_null_only_for_all_model_failure() -> None:
    labels = np.array([[1, 0, 1], [0, 0, 0], [0, 1, 0]])

    augmented = augment_ground_truth(labels)

    np.testing.assert_array_equal(
        augmented,
        np.array(
            [
                [True, False, True, False],
                [False, False, False, True],
                [False, True, False, False],
            ]
        ),
    )


def test_racer_g_uses_maximum_over_the_augmented_pool() -> None:
    augmented = np.array(
        [
            [0.8, 0.3, 0.2],
            [0.2, 0.4, 0.6],
        ]
    )

    scores = raw_nonconformity_scores(augmented, "G")

    np.testing.assert_allclose(scores, [[0.0, 0.5, 0.6], [0.4, 0.2, 0.0]])


def test_racer_p_is_one_minus_each_augmented_score() -> None:
    augmented = np.array([[0.8, 0.3, 0.2], [0.2, 0.4, 0.6]])

    scores = raw_nonconformity_scores(augmented, "P")

    np.testing.assert_allclose(scores, [[0.2, 0.7, 0.8], [0.8, 0.6, 0.4]])


def test_randomized_smoothing_is_bounded_independent_and_split_seeded() -> None:
    raw = np.zeros((100, 4))

    first = smooth_nonconformity_scores(raw, seed=17, split_seed=3)
    repeated = smooth_nonconformity_scores(raw, seed=17, split_seed=3)
    other_split = smooth_nonconformity_scores(raw, seed=17, split_seed=4)

    np.testing.assert_array_equal(first, repeated)
    assert np.all(first >= 0.0)
    assert np.all(first <= 1e-6)
    assert np.unique(first).size == first.size
    assert not np.array_equal(first, other_split)


def test_zero_smoothing_is_an_exact_noop_copy() -> None:
    raw = np.array([[0.0, 0.2], [0.4, 0.6]])

    result = smooth_nonconformity_scores(raw, smoothing=0.0, seed=1, split_seed=2)

    np.testing.assert_array_equal(result, raw)
    assert result is not raw


def test_crc_selects_smallest_feasible_threshold() -> None:
    critical = np.array([0.1, 0.2, 0.3, 0.4])

    result = select_crc_threshold(critical, alpha=0.4)

    assert result.feasible
    assert result.max_misses == 1
    assert result.threshold == approx(0.3)
    assert result.calibration_misses == 1
    assert result.crc_bound == approx(0.4)
    assert (1 + np.count_nonzero(critical > result.threshold)) / 5 <= 0.4
    assert (1 + np.count_nonzero(critical > 0.2)) / 5 > 0.4


def test_crc_threshold_handles_ties_using_strict_miss_inequality() -> None:
    critical = np.array([0.1, 0.3, 0.3, 0.9])

    result = select_crc_threshold(critical, alpha=0.6)

    assert result.feasible
    assert result.threshold == approx(0.3)
    assert result.max_misses == 2
    assert result.calibration_misses == 1
    assert result.crc_bound == approx(0.4)


def test_crc_below_sample_resolution_is_non_certification() -> None:
    critical = np.array([0.1, 0.2, 0.3, 0.4])

    result = select_crc_threshold(critical, alpha=0.1)

    assert not result.feasible
    assert np.isneginf(result.threshold)
    assert result.max_misses == -1
    assert result.calibration_misses == 0
    assert result.crc_bound == approx(0.2)
    assert result.crc_bound > result.alpha


def test_prediction_sets_are_variable_size_and_size_excludes_null() -> None:
    base = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.35],
            [0.1, 0.1],
            [0.6, 0.55],
        ]
    )
    augmented = augment_router_scores(base)
    scores = raw_nonconformity_scores(augmented, "P")

    routed = prediction_sets(augmented, scores, threshold=0.5)

    np.testing.assert_array_equal(routed.size, [1, 0, 0, 2])
    np.testing.assert_array_equal(routed.abstain, [False, True, True, False])
    np.testing.assert_array_equal(
        routed.null_selected, [False, True, True, False]
    )
    np.testing.assert_array_equal(
        routed.standard_mask,
        np.array(
            [
                [True, False],
                [False, False],
                [False, False],
                [True, True],
            ]
        ),
    )


def test_abstention_depends_on_standard_models_not_null_membership() -> None:
    augmented = np.array([[0.8, 0.2, 0.2]])
    scores = np.array([[0.9, 0.8, 0.7]])

    empty = prediction_sets(augmented, scores, threshold=0.6)
    null_only = prediction_sets(augmented, scores, threshold=0.75)

    assert empty.abstain.item()
    assert not empty.null_selected.item()
    assert null_only.abstain.item()
    assert null_only.null_selected.item()
    assert empty.size.item() == null_only.size.item() == 0


def test_misrouting_loss_recognizes_correct_null_abstention() -> None:
    labels = augment_ground_truth(np.array([[1, 0], [0, 0], [0, 1]]))
    selected = np.array(
        [
            [True, False, False],
            [False, False, True],
            [True, False, False],
        ]
    )

    loss = misrouting_loss(selected, labels)

    np.testing.assert_array_equal(loss, [False, False, True])



def test_null_selection_never_suppresses_selected_real_calls() -> None:
    augmented = np.array([[0.8, 0.7, 0.2]])
    scores = np.array([[0.1, 0.2, 0.05]])

    routed = prediction_sets(augmented, scores, threshold=0.15)

    assert routed.null_selected.item()
    assert not routed.safe_action.item()
    assert not routed.abstain.item()
    assert routed.size.item() == 1
    np.testing.assert_array_equal(routed.standard_mask, [[True, False]])
    np.testing.assert_array_equal(routed.augmented_mask, [[True, False, True]])


def test_augmented_standard_and_safe_action_losses_are_distinct() -> None:
    standard_labels = np.array([[1, 0], [0, 0], [0, 1]], dtype=bool)
    augmented_labels = augment_ground_truth(standard_labels)
    augmented = np.array(
        [
            [True, False, True],
            [False, False, True],
            [True, False, False],
        ]
    )
    router_scores = np.ones_like(augmented, dtype=float)
    score_values = np.where(augmented, 0.0, 1.0)
    routed = prediction_sets(router_scores, score_values, threshold=0.0)

    np.testing.assert_array_equal(
        augmented_misrouting_loss(routed.augmented_mask, augmented_labels),
        [False, False, True],
    )
    np.testing.assert_array_equal(
        standard_all_miss_loss(routed.standard_mask, standard_labels),
        [False, True, True],
    )
    np.testing.assert_array_equal(
        safe_action_loss(routed, standard_labels),
        [False, False, True],
    )

def test_ground_truth_critical_score_takes_best_correct_augmented_entry() -> None:
    scores = np.array([[0.7, 0.2, 0.8], [0.9, 0.6, 0.1]])
    labels = np.array([[True, True, False], [False, False, True]])

    critical = ground_truth_critical_scores(scores, labels)

    np.testing.assert_allclose(critical, [0.2, 0.1])


def test_protocol_calibration_counts_unanswerable_rows_as_standard_misses() -> None:
    base = np.array(
        [
            [0.9, 0.2],
            [0.2, 0.3],
            [0.6, 0.5],
            [0.1, 0.1],
        ]
    )
    labels = np.array(
        [
            [1, 0],
            [0, 0],
            [0, 1],
            [0, 0],
        ]
    )
    protocol = RACERProtocol("P", seed=9, smoothing=0.0)

    calibration = protocol.calibrate(base, labels, alpha=0.4, split_seed=12)

    np.testing.assert_allclose(
        calibration.critical_scores, [0.1, np.inf, 0.5, np.inf]
    )
    assert not calibration.feasible
    assert calibration.unavoidable_misses == 2
    assert calibration.calibration_misses == 2
    assert calibration.crc_bound == approx(0.6)
    assert calibration.non_certification_reason == "standard_risk_structurally_infeasible"


def test_standard_threshold_scan_finds_smallest_exact_operating_point() -> None:
    scores = np.array(
        [[0.1, 0.9], [0.2, 0.3], [0.8, 0.4], [0.7, 0.6]]
    )
    labels = np.array([[1, 0], [0, 0], [0, 1], [1, 0]])

    critical = standard_ground_truth_critical_scores(scores, labels)
    calibration = select_standard_crc_threshold(scores, labels, alpha=0.6)

    np.testing.assert_allclose(critical, [0.1, np.inf, 0.4, 0.7])
    assert calibration.feasible
    assert calibration.threshold == approx(0.4)
    assert calibration.unavoidable_misses == 1
    assert calibration.calibration_misses == 2
    assert calibration.crc_bound == approx(0.6)


def test_protocol_non_certification_cannot_be_routed() -> None:
    base = np.array([[0.8, 0.1], [0.2, 0.3]])
    labels = np.array([[1, 0], [0, 0]])
    protocol = RACERProtocol("G", seed=4, smoothing=0.0)

    calibration = protocol.calibrate(base, labels, alpha=0.2, split_seed=1)

    assert not calibration.feasible
    with np.testing.assert_raises_regex(ValueError, "cannot route a non-certified"):
        protocol.route(base, calibration, split_seed=2)


def test_deterministic_split_is_complete_disjoint_and_seeded() -> None:
    split = deterministic_split_indices(11, seed=123)
    repeated = deterministic_split_indices(11, seed=123)
    other = deterministic_split_indices(11, seed=124)

    np.testing.assert_array_equal(split.calibration, repeated.calibration)
    np.testing.assert_array_equal(split.validation, repeated.validation)
    np.testing.assert_array_equal(split.test, repeated.test)
    assert [len(split.calibration), len(split.validation), len(split.test)] == [6, 1, 4]

    combined = np.concatenate((split.calibration, split.validation, split.test))
    np.testing.assert_array_equal(np.sort(combined), np.arange(11))
    assert np.unique(combined).size == 11
    assert not np.array_equal(split.calibration, other.calibration)


def test_arc_split_fractions_are_supported_exactly() -> None:
    split = deterministic_split_indices(10, seed=7, fractions=(0.4, 0.2, 0.4))

    assert [len(split.calibration), len(split.validation), len(split.test)] == [4, 2, 4]


def test_invalid_router_scores_are_rejected() -> None:
    bad_inputs = (
        np.array([0.2, 0.3]),
        np.array([[0.2, np.nan]]),
        np.array([[0.2, 1.1]]),
    )
    for bad_scores in bad_inputs:
        with np.testing.assert_raises(ValueError):
            augment_router_scores(bad_scores)


def test_invalid_variant_is_rejected() -> None:
    for variant in ("unknown", "", "gap"):
        with np.testing.assert_raises(ValueError):
            RACERProtocol(variant)


def _run_tests() -> None:
    """Run every synthetic test without requiring a test framework."""

    import traceback

    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception:
            failures += 1
            print(f"FAIL: {test.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS: {test.__name__}")
    print(f"Ran {len(tests)} tests: {len(tests) - failures} passed, {failures} failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_tests()
