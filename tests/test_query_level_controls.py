"""Exact tests for the decisive query-level control baselines."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from components.racer_protocol import (  # noqa: E402
    augment_ground_truth,
    augment_router_scores,
)
from components.racer_wrapper import (  # noqa: E402
    EPSILON,
    FIXED_DESIGN_MARGINAL_FACTORIZED,
    FIXED_TALLY_FACTORIZED,
    LOCAL_ORDER_FACTORIZED,
    QUERY_LOCAL_EMPIRICAL_COVERAGE,
    augment_with_racer_null,
    fixed_design_marginal_factorized_scores,
    fixed_design_marginal_order,
    fixed_order_factorized_scores,
    local_order_factorized_scores,
    per_query_score_diagnostics,
    query_local_empirical_order,
    query_local_empirical_scores,
)
from components.scale_selectors import build_order  # noqa: E402
from scripts import design_bprime_e2_run as e2  # noqa: E402
from scripts import design_e3_unified as e3  # noqa: E402


def _slow_query_local_empirical_scores(
    neighbor_correctness: np.ndarray,
    *,
    neighbor_weights: np.ndarray | None = None,
    max_depth: int | None = None,
) -> np.ndarray:
    """Straightforward candidate loop used only as an equivalence oracle."""

    labels = np.asarray(neighbor_correctness, dtype=bool)
    n_queries, n_neighbors, n_models = labels.shape
    if neighbor_weights is None:
        weights = np.ones((n_queries, n_neighbors), dtype=float)
    else:
        weights = np.asarray(neighbor_weights, dtype=float)
        if weights.ndim == 1:
            weights = np.broadcast_to(weights, (n_queries, n_neighbors))
    totals = weights.sum(axis=1)
    depth = n_models if max_depth is None else min(int(max_depth), n_models)

    scores = np.ones((n_queries, n_models), dtype=float)
    for row in range(n_queries):
        remaining = np.ones(n_models, dtype=bool)
        all_miss = np.ones(n_neighbors, dtype=bool)
        predecessor_all_miss = 1.0
        marginal_accuracy = np.array([
            weights[row, labels[row, :, model]].sum() / totals[row]
            for model in range(n_models)
        ])
        for _ in range(depth):
            candidates = np.flatnonzero(remaining)
            candidate_all_miss = np.array([
                weights[row, all_miss & ~labels[row, :, candidate]].sum()
                / totals[row]
                for candidate in candidates
            ])

            minimum_miss = float(candidate_all_miss.min())
            miss_tied_positions = [
                position
                for position, value in enumerate(candidate_all_miss)
                if value <= minimum_miss + EPSILON
            ]
            maximum_marginal = max(
                float(marginal_accuracy[candidates[position]])
                for position in miss_tied_positions
            )
            best_position = next(
                position
                for position in miss_tied_positions
                if marginal_accuracy[candidates[position]]
                >= maximum_marginal - EPSILON
            )
            best_index = int(candidates[best_position])
            scores[row, best_index] = 1.0 - predecessor_all_miss
            all_miss &= ~labels[row, :, best_index]
            predecessor_all_miss = float(candidate_all_miss[best_position])
            remaining[best_index] = False
    return scores


def _complementary_neighbors() -> np.ndarray:
    return np.array(
        [[
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]],
        dtype=int,
    )


@pytest.mark.parametrize(
    ("shape", "max_depth"),
    [
        ((1, 3, 4), None),
        ((2, 7, 5), 1),
        ((3, 11, 9), 4),
        ((4, 40, 17), 10),
        ((2, 5, 6), 20),
    ],
)
@pytest.mark.parametrize("weight_mode", ["unweighted", "shared", "per_query"])
def test_query_local_empirical_matches_slow_reference_random_cases(
    shape: tuple[int, int, int],
    max_depth: int | None,
    weight_mode: str,
) -> None:
    mode_seed = {"unweighted": 0, "shared": 1, "per_query": 2}[weight_mode]
    rng = np.random.default_rng(9000 + sum(shape) + mode_seed)
    neighbors = rng.integers(0, 2, size=shape, dtype=np.int8)
    weights = None
    if weight_mode == "shared":
        weights = rng.random(shape[1])
        weights[rng.random(shape[1]) < 0.2] = 0.0
    elif weight_mode == "per_query":
        weights = rng.random(shape[:2])
        weights[rng.random(shape[:2]) < 0.2] = 0.0

    expected = _slow_query_local_empirical_scores(
        neighbors,
        neighbor_weights=weights,
        max_depth=max_depth,
    )
    actual = query_local_empirical_scores(
        neighbors,
        neighbor_weights=weights,
        max_depth=max_depth,
    )

    if weight_mode == "unweighted":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)


def test_query_local_empirical_degenerate_ties_use_smallest_index() -> None:
    shared_column = np.array([1, 0, 1, 0], dtype=np.int8)
    neighbors = np.broadcast_to(shared_column[:, None], (4, 4))[None, :, :]

    scores = query_local_empirical_scores(neighbors)

    np.testing.assert_array_equal(
        np.argsort(scores[0], kind="stable"),
        np.array([0, 1, 2, 3]),
    )
    np.testing.assert_array_equal(
        scores,
        _slow_query_local_empirical_scores(neighbors),
    )


def test_query_local_empirical_tolerance_tie_prefers_larger_marginal() -> None:
    neighbors = np.array(
        [[
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
        ]],
        dtype=np.int8,
    )
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1e-13, 1e-13])

    scores = query_local_empirical_scores(neighbors, neighbor_weights=weights)

    np.testing.assert_array_equal(
        np.argsort(scores[0], kind="stable"),
        np.array([0, 2, 1]),
    )
    np.testing.assert_allclose(
        scores,
        _slow_query_local_empirical_scores(
            neighbors,
            neighbor_weights=weights,
        ),
        rtol=0.0,
        atol=2e-15,
    )


def test_query_local_empirical_pool500_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(9182026)
    neighbors = rng.integers(0, 2, size=(270, 40, 500), dtype=np.int8)
    weights = rng.random((270, 40)) + 0.01

    query_local_empirical_scores(
        neighbors[:2], neighbor_weights=weights[:2], max_depth=100
    )
    fast_start = perf_counter()
    actual = query_local_empirical_scores(
        neighbors, neighbor_weights=weights, max_depth=100
    )
    fast_seconds = perf_counter() - fast_start

    slow_start = perf_counter()
    expected = _slow_query_local_empirical_scores(
        neighbors, neighbor_weights=weights, max_depth=100
    )
    slow_seconds = perf_counter() - slow_start

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)
    speedup = slow_seconds / fast_seconds
    with capsys.disabled():
        print(
            "\nquery_local_empirical_scores pool500 benchmark: "
            f"slow={slow_seconds:.3f}s, vectorized={fast_seconds:.3f}s, "
            f"speedup={speedup:.2f}x"
        )


def test_query_local_empirical_all_miss_equals_direct_neighbor_frequency() -> None:
    neighbors = _complementary_neighbors()
    scores = query_local_empirical_scores(neighbors)
    order = np.argsort(scores[0], kind="stable")

    for position in range(1, len(order)):
        predecessor = order[:position]
        direct_all_miss = np.all(neighbors[0][:, predecessor] == 0, axis=1).mean()
        score_implied_all_miss = 1.0 - scores[0, order[position]]
        assert score_implied_all_miss == pytest.approx(direct_all_miss)


def test_query_local_empirical_uses_complementary_second_pick() -> None:
    scores = query_local_empirical_scores(_complementary_neighbors())
    order = np.argsort(scores[0], kind="stable")
    assert list(order[:2]) == [0, 2]


def test_knn_marginals_and_local_joint_share_one_frozen_support() -> None:
    train_embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    train_correctness = np.array([[1, 0], [0, 1], [1, 1]])
    query_embeddings = np.array([[1.0, 0.0]])

    probabilities, support = e2._knn_probabilities(
        train_embeddings,
        train_correctness,
        query_embeddings,
        2,
        return_support=True,
    )

    np.testing.assert_allclose(probabilities, support.mean(axis=1))
    assert support.shape == (1, 2, 2)


def test_fixed_tally_order_has_query_factorized_scores() -> None:
    design = _complementary_neighbors()[0]
    order = build_order(
        "empirical_tally",
        design,
        model_names=("a", "b", "c"),
        max_depth=3,
    )
    assert order == (0, 2, 1)

    probabilities = np.array([[0.8, 0.4, 0.7]])
    scores = fixed_order_factorized_scores(probabilities, order)
    assert scores[0, 0] == pytest.approx(0.0)
    assert scores[0, 2] == pytest.approx(0.8)
    assert scores[0, 1] == pytest.approx(0.94)


def test_both_controls_are_named_e2_and_e3_arms_with_metadata() -> None:
    for arm in (QUERY_LOCAL_EMPIRICAL_COVERAGE, FIXED_TALLY_FACTORIZED):
        assert arm in e2.WRAPPER_ARMS
        assert arm in e3.WRAPPER_ARMS
        assert e2.WRAPPER_HYPER_GRIDS[arm] == [{}]
        assert arm in e2.WRAPPER_ARM_METADATA


def test_final_calibration_perturbation_cannot_change_new_arm_identity() -> None:
    probabilities = np.array([
        [0.9, 0.4], [0.8, 0.3], [0.7, 0.6], [0.6, 0.5],
        [0.9, 0.2], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
        [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8],
    ])
    standard = np.array([
        [1, 0], [1, 0], [1, 1], [0, 1],
        [1, 0], [1, 0], [0, 1], [0, 1],
        [1, 0], [0, 1], [0, 1], [0, 1],
    ], dtype=bool)
    raw = fixed_order_factorized_scores(probabilities, (0, 1))
    scores = e2.smooth_nonconformity_scores(
        augment_with_racer_null(probabilities, raw), smoothing=0.0
    )
    candidates = [
        {
            "candidate": f"wrapper:{FIXED_TALLY_FACTORIZED}",
            "policy_class": "per_query_wrapper",
            "variant": FIXED_TALLY_FACTORIZED,
            "val_calls": 1.0,
            "val_risk": 0.0,
            "_scores": scores,
        },
        {
            "candidate": f"wrapper:{QUERY_LOCAL_EMPIRICAL_COVERAGE}",
            "policy_class": "per_query_wrapper",
            "variant": QUERY_LOCAL_EMPIRICAL_COVERAGE,
            "val_calls": 2.0,
            "val_risk": 0.0,
            "_scores": scores[:, ::-1],
        },
    ]
    frozen = e3._select(candidates, alpha=0.5, class_penalty=False)["selected"]
    calibration_idx = np.arange(4, 8)
    test_idx = np.arange(8, 12)
    augmented_probabilities = augment_router_scores(probabilities)

    original = e3._finalize_wrapper(
        candidate=frozen,
        augmented_probabilities=augmented_probabilities,
        standard_labels=standard,
        augmented_labels=augment_ground_truth(standard),
        cal_idx=calibration_idx,
        test_idx=test_idx,
        alpha=0.5,
    )
    perturbed_standard = standard.copy()
    perturbed_standard[calibration_idx] = ~perturbed_standard[calibration_idx]
    perturbed = e3._finalize_wrapper(
        candidate=frozen,
        augmented_probabilities=augmented_probabilities,
        standard_labels=perturbed_standard,
        augmented_labels=augment_ground_truth(perturbed_standard),
        cal_idx=calibration_idx,
        test_idx=test_idx,
        alpha=0.5,
    )

    expected = f"wrapper:{FIXED_TALLY_FACTORIZED}"
    assert frozen["candidate"] == expected
    assert original["candidate"] == expected
    assert perturbed["candidate"] == expected
    assert original["final_calibration_count"] == 1
    assert perturbed["final_calibration_count"] == 1


def test_design_permutation_preserves_each_model_marginal() -> None:
    design = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=int)
    original = design.copy()

    shuffled = e2._shuffle_design_columns(design, seed=9182)

    np.testing.assert_array_equal(shuffled.sum(axis=0), design.sum(axis=0))
    np.testing.assert_array_equal(design, original)


def test_design_permutation_is_deterministic_and_seeded_per_record() -> None:
    design = np.arange(48).reshape(12, 4)
    seed = e2._design_permutation_seed(7001, redraw=3, permutation_index=2)

    first = e2._shuffle_design_columns(design, seed)
    second = e2._shuffle_design_columns(design, seed)

    assert seed == e2._design_permutation_seed(7001, redraw=3, permutation_index=2)
    assert seed != e2._design_permutation_seed(7001, redraw=3, permutation_index=3)
    assert seed != e2._design_permutation_seed(7001, redraw=4, permutation_index=2)
    np.testing.assert_array_equal(first, second)


def test_design_permutation_omits_design_independent_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_arms = []

    def fake_score_grid(**kwargs):
        assert kwargs["arms"] == e2.DESIGN_DEPENDENT_ARMS
        return {arm: (arm, {}, np.empty((0, 0))) for arm in kwargs["arms"]}

    def fake_select_and_finalize(**kwargs):
        observed_arms.append(kwargs["arm"])
        return {
            "calls_test": 2.0,
            "final_calibration_count": 1,
            "chosen_hyper": {},
            "validation_trajectory": [],
        }

    monkeypatch.setattr(e2, "_score_wrapper_grid", fake_score_grid)
    monkeypatch.setattr(e2, "_select_and_finalize_wrapper_arm", fake_select_and_finalize)

    alpha_key = "0.1"
    real_results = {
        alpha_key: {
            arm: {"calls_test": 1.5} for arm in e2.DESIGN_DEPENDENT_ARMS
        }
    }
    result = e2._evaluate_design_permutation(
        design_labels=np.array([[1, 0], [0, 1], [1, 1], [0, 0]]),
        permutation_seed=12345,
        permutation_index=0,
        probabilities=np.full((6, 2), 0.5),
        neighbor_correctness=np.zeros((6, 2, 2), dtype=int),
        augmented_probabilities=np.full((6, 3), 0.5),
        standard_labels=np.zeros((6, 2), dtype=bool),
        augmented_labels=np.zeros((6, 3), dtype=bool),
        val_idx=np.array([0, 1]),
        cal_idx=np.array([2, 3]),
        test_idx=np.array([4, 5]),
        alphas=[0.1],
        real_results_by_alpha=real_results,
        max_depth=2,
        seed=7001,
        split_seed=1,
    )

    alpha_record = result["per_alpha"][alpha_key]
    assert tuple(alpha_record["arms"]) == e2.DESIGN_DEPENDENT_ARMS
    assert observed_arms == list(e2.DESIGN_DEPENDENT_ARMS)
    assert alpha_record["omitted_arms"] == e2.DESIGN_PERMUTATION_OMISSIONS
    assert e2.FACTORIZED_INDEPENDENCE not in alpha_record["arms"]
    assert e2.QUERY_LOCAL_EMPIRICAL_COVERAGE not in alpha_record["arms"]
    for record in alpha_record["arms"].values():
        assert record["permutation_seed"] == 12345
        assert record["real_minus_shuffled_calls"] == pytest.approx(-0.5)


def test_wrapper_selection_does_not_read_final_calibration_labels() -> None:
    probabilities = np.array([
        [0.9, 0.4], [0.8, 0.3], [0.7, 0.6], [0.6, 0.5],
        [0.9, 0.2], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
        [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8],
    ])
    standard = np.array([
        [1, 0], [1, 0], [1, 1], [0, 1],
        [1, 0], [1, 0], [0, 1], [0, 1],
        [1, 0], [0, 1], [0, 1], [0, 1],
    ], dtype=bool)
    raw = fixed_order_factorized_scores(probabilities, (0, 1))
    scores = e2.smooth_nonconformity_scores(
        augment_with_racer_null(probabilities, raw), smoothing=0.0
    )
    arm = e2.PAIRWISE_HEURISTIC
    fitted = {
        f"wrapper_{arm}::0": (arm, {"laplace_smoothing": 0.0}, scores),
        f"wrapper_{arm}::1": (arm, {"laplace_smoothing": 1.0}, scores.copy()),
    }
    val_idx = np.arange(4)
    cal_idx = np.arange(4, 8)
    test_idx = np.arange(8, 12)
    augmented_probabilities = augment_router_scores(probabilities)

    original = e2._select_and_finalize_wrapper_arm(
        arm=arm,
        fitted=fitted,
        augmented_probabilities=augmented_probabilities,
        standard_labels=standard,
        augmented_labels=augment_ground_truth(standard),
        val_idx=val_idx,
        cal_idx=cal_idx,
        test_idx=test_idx,
        alpha=0.5,
        neighbor_support_size=2,
    )
    perturbed_standard = standard.copy()
    perturbed_standard[cal_idx] = ~perturbed_standard[cal_idx]
    perturbed = e2._select_and_finalize_wrapper_arm(
        arm=arm,
        fitted=fitted,
        augmented_probabilities=augmented_probabilities,
        standard_labels=perturbed_standard,
        augmented_labels=augment_ground_truth(perturbed_standard),
        val_idx=val_idx,
        cal_idx=cal_idx,
        test_idx=test_idx,
        alpha=0.5,
        neighbor_support_size=2,
    )

    assert original["chosen_hyper"] == perturbed["chosen_hyper"]
    assert original["validation_trajectory"] == perturbed["validation_trajectory"]
    assert original["selection_source"] == "validation_only"
    assert perturbed["selection_source"] == "validation_only"
    assert original["final_calibration_count"] == 1
    assert perturbed["final_calibration_count"] == 1


def test_e2_validation_infeasible_fallback_can_be_final_certified() -> None:
    probabilities = np.array([[0.9, 0.1]] * 12)
    scores = np.array([[0.1, 0.9, 0.0]] * 12)
    standard = np.array(
        [[0, 0], [1, 0], [1, 0], [1, 0]]
        + [[1, 0]] * 8,
        dtype=bool,
    )
    augmented_probabilities = augment_router_scores(probabilities)
    augmented = augment_ground_truth(standard)
    val_idx = np.arange(4)
    cal_idx = np.arange(4, 8)
    test_idx = np.arange(8, 12)

    records = [
        e2._temporary_validation_record(
            arm_key=arm_key,
            hyper={"stable_id": arm_key},
            scores=scores,
            augmented_probabilities=augmented_probabilities,
            standard_labels=standard,
            augmented_labels=augmented,
            val_idx=val_idx,
            alpha=0.30,
        )
        for arm_key in ("candidate_z", "candidate_a")
    ]
    choice = e2._select_validation_candidate(records, alpha=0.30)
    frozen_identity = choice["chosen"]["arm_key"]
    original_final = e2._final_calibrate_and_test(
        scores=scores,
        augmented_probabilities=augmented_probabilities,
        standard_labels=standard,
        augmented_labels=augmented,
        cal_idx=cal_idx,
        test_idx=test_idx,
        alpha=0.30,
    )

    perturbed_standard = standard.copy()
    perturbed_standard[cal_idx] = False
    perturbed_augmented = augment_ground_truth(perturbed_standard)
    perturbed_choice = e2._select_validation_candidate(records, alpha=0.30)
    perturbed_final = e2._final_calibrate_and_test(
        scores=scores,
        augmented_probabilities=augmented_probabilities,
        standard_labels=perturbed_standard,
        augmented_labels=perturbed_augmented,
        cal_idx=cal_idx,
        test_idx=test_idx,
        alpha=0.30,
    )

    assert all(record["validation_certified"] is False for record in records)
    assert all(record["validation_fallback"] is True for record in records)
    assert all(record["fallback"] is True for record in records)
    assert all(record["validation_threshold"] == pytest.approx(0.1) for record in records)
    assert all(record["standard_all_miss_risk"] == pytest.approx(0.25) for record in records)
    assert all(record["calls"] == pytest.approx(1.0) for record in records)
    assert choice["fallback_used"] is True
    assert frozen_identity == "candidate_a"
    assert perturbed_choice["chosen"]["arm_key"] == frozen_identity

    certified_record = {
        **records[0],
        "arm_key": "certified_candidate",
        "validation_certified": True,
        "validation_fallback": False,
        "fallback": False,
        "calls": 2.0,
        "standard_all_miss_risk": 0.25,
    }
    certified_choice = e2._select_validation_candidate(
        [*records, certified_record], alpha=0.30
    )
    assert certified_choice["chosen"]["arm_key"] == "certified_candidate"
    assert certified_choice["fallback_used"] is False

    assert original_final["certified"] is True
    assert original_final["risk_validity_claimed"] is True
    assert original_final["final_calibration_count"] == 1
    assert perturbed_final["certified"] is False
    assert perturbed_final["risk_validity_claimed"] is False
    assert perturbed_final["final_calibration_count"] == 1

    arm = e2.PAIRWISE_HEURISTIC
    fitted = {
        "candidate_z": (arm, {"stable_id": "z"}, scores),
        "candidate_a": (arm, {"stable_id": "a"}, scores.copy()),
    }
    integrated = e2._select_and_finalize_wrapper_arm(
        arm=arm,
        fitted=fitted,
        augmented_probabilities=augmented_probabilities,
        standard_labels=standard,
        augmented_labels=augmented,
        val_idx=val_idx,
        cal_idx=cal_idx,
        test_idx=test_idx,
        alpha=0.30,
        neighbor_support_size=2,
    )
    integrated_perturbed = e2._select_and_finalize_wrapper_arm(
        arm=arm,
        fitted=fitted,
        augmented_probabilities=augmented_probabilities,
        standard_labels=perturbed_standard,
        augmented_labels=perturbed_augmented,
        val_idx=val_idx,
        cal_idx=cal_idx,
        test_idx=test_idx,
        alpha=0.30,
        neighbor_support_size=2,
    )

    assert integrated["chosen_hyper"] == {"stable_id": "a"}
    assert integrated_perturbed["chosen_hyper"] == integrated["chosen_hyper"]
    assert integrated["selection_fallback_used"] is True
    assert integrated["selection_non_certification"] is True
    assert integrated["certified"] is True
    assert integrated["risk_validity_claimed"] is True
    assert integrated_perturbed["selection_fallback_used"] is True
    assert integrated_perturbed["certified"] is False
    assert integrated_perturbed["risk_validity_claimed"] is False


def test_local_arms_share_order_but_use_different_sizing() -> None:
    neighbors = _complementary_neighbors()
    probabilities = neighbors.mean(axis=1)

    empirical, empirical_trace = query_local_empirical_scores(
        neighbors, return_trace=True
    )
    factorized, factorized_trace = local_order_factorized_scores(
        probabilities, neighbors, return_trace=True
    )

    expected_order = np.array([[0, 2, 1]])
    np.testing.assert_array_equal(empirical_trace["order"], expected_order)
    np.testing.assert_array_equal(factorized_trace["order"], expected_order)
    np.testing.assert_array_equal(query_local_empirical_order(neighbors), expected_order)
    assert empirical[0, 1] == pytest.approx(1.0)
    assert factorized[0, 1] == pytest.approx(7.0 / 9.0)
    assert not np.array_equal(empirical, factorized)


def test_local_order_factorized_scores_are_exact_products() -> None:
    neighbors = _complementary_neighbors()
    probabilities = np.array([[0.8, 0.4, 0.7]])

    scores, trace = local_order_factorized_scores(
        probabilities, neighbors, return_trace=True
    )

    np.testing.assert_array_equal(trace["order"], np.array([[0, 2, 1]]))
    np.testing.assert_allclose(
        scores,
        np.array([[0.0, 0.94, 0.8]]),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        trace["residuals"],
        np.array([[1.0, 0.2, 0.06, 0.036]]),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        trace["log_residuals"],
        np.log(np.array([[1.0, 0.2, 0.06, 0.036]])),
        rtol=0.0,
        atol=1e-15,
    )


def test_fixed_design_marginal_control_uses_accuracy_then_index() -> None:
    design = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
    ])
    probabilities = np.array([[0.2, 0.8, 0.5, 0.4]])

    order = fixed_design_marginal_order(design)
    scores = fixed_design_marginal_factorized_scores(probabilities, design)

    assert order == (1, 0, 2, 3)
    np.testing.assert_allclose(
        scores,
        np.array([[0.8, 0.0, 0.84, 0.92]]),
        rtol=0.0,
        atol=1e-15,
    )


def test_local_empirical_plateau_and_tie_diagnostics_are_auditable() -> None:
    neighbors = np.array(
        [[
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ]],
        dtype=int,
    )
    scores, trace = query_local_empirical_scores(neighbors, return_trace=True)

    diagnostics = per_query_score_diagnostics(
        scores,
        trace,
        threshold=1.0,
        selection_scores=scores,
    )[0]

    assert diagnostics["distinct_finite_score_count"] == 3
    assert diagnostics["first_zero_miss_depth"] == 2
    assert diagnostics["plateau_depth"] == 2
    assert diagnostics["plateau_value"] == pytest.approx(0.0)
    assert diagnostics["tied_score_blocks"] == [{"score": 1.0, "size": 2}]
    assert diagnostics["tied_block_sizes"] == [2]
    assert diagnostics["score_one_tail_size"] == 2
    assert diagnostics["selected_depth"] == 4
    assert diagnostics["full_pool_call_indicator"] is True
    assert diagnostics["log_residual"][-1] is None
    assert diagnostics["factorized_underflow_count"] == 0
    assert diagnostics["factorized_saturation_count"] == 0


def test_new_arms_are_registered_without_replacing_local_empirical() -> None:
    expected = {
        QUERY_LOCAL_EMPIRICAL_COVERAGE,
        LOCAL_ORDER_FACTORIZED,
        FIXED_DESIGN_MARGINAL_FACTORIZED,
    }
    assert expected.issubset(e2.WRAPPER_ARMS)
    for arm in expected:
        assert e2.WRAPPER_HYPER_GRIDS[arm] == [{}]
        assert arm in e2.WRAPPER_ARM_METADATA


def test_neighbor_k_grid_preserves_40_as_primary() -> None:
    assert e2._parse_neighbor_k_grid("10,20,40,80,20", 40) == (40, 10, 20, 80)
    assert e2._parse_neighbor_k_grid(None, 40) == (40,)
    with pytest.raises(ValueError, match="positive"):
        e2._parse_neighbor_k_grid("10,0,40", 40)


def test_new_control_scores_do_not_read_target_labels() -> None:
    design = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
    ])
    neighbors = np.array([
        [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
        [[0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1]],
    ])
    probabilities = neighbors.mean(axis=1)
    target_labels = np.array([[1, 0, 0], [0, 1, 0]])

    original_local = local_order_factorized_scores(probabilities, neighbors)
    original_fixed = fixed_design_marginal_factorized_scores(probabilities, design)
    target_labels[:] = 1 - target_labels
    perturbed_local = local_order_factorized_scores(probabilities, neighbors)
    perturbed_fixed = fixed_design_marginal_factorized_scores(probabilities, design)

    np.testing.assert_array_equal(original_local, perturbed_local)
    np.testing.assert_array_equal(original_fixed, perturbed_fixed)
