"""Regression tests for the September 2026 RACER protocol repair."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from components.racer_protocol import (  # noqa: E402
    augment_ground_truth,
    augment_router_scores,
    ground_truth_critical_scores,
    prediction_sets,
    select_crc_threshold,
)
from components.racer_wrapper import (  # noqa: E402
    CORRELATION_HEURISTIC,
    FACTORIZED_INDEPENDENCE,
    PAIRWISE_HEURISTIC,
    REGULARIZED_CORRELATION,
    canonical_dependence_name,
    sequential_scores,
)
from scripts import design_bprime_e2_run as e2  # noqa: E402
from scripts import design_bprime_run as e1  # noqa: E402
from scripts import design_e3_unified as e3  # noqa: E402


def test_query_level_public_names_and_read_aliases() -> None:
    assert canonical_dependence_name("independence") == FACTORIZED_INDEPENDENCE
    assert canonical_dependence_name("pairwise_count") == PAIRWISE_HEURISTIC
    assert canonical_dependence_name("error_correlation") == CORRELATION_HEURISTIC
    assert canonical_dependence_name("gaussian_copula") == REGULARIZED_CORRELATION

    probabilities = np.array([[0.8, 0.5, 0.2]])
    design = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(
        sequential_scores(
            probabilities, dependence="gaussian_copula", design=design,
            hyper={"ridge": 0.1},
        ),
        sequential_scores(
            probabilities, dependence=REGULARIZED_CORRELATION, design=design,
            hyper={"ridge": 0.1},
        ),
    )


def test_exact_factorized_independence_is_named_in_e2_and_e3() -> None:
    assert FACTORIZED_INDEPENDENCE in e2.WRAPPER_ARMS
    assert FACTORIZED_INDEPENDENCE in e3.WRAPPER_ARMS
    assert e2.WRAPPER_HYPER_GRIDS[FACTORIZED_INDEPENDENCE] == [{}]


def test_racer_g_and_p_are_separate_controls() -> None:
    assert e2.RACER_CONTROLS == ("racer_G", "racer_P")
    assert e3.RACER_CONTROLS == ("racer_G", "racer_P")


def test_e1_hyperparameter_selection_ignores_final_calibration_labels(monkeypatch) -> None:
    monkeypatch.setitem(e1.HYPERPARAMETER_GRIDS, "unit_arm", (0, 1))
    monkeypatch.setitem(e1.HYPERPARAMETER_NAMES, "unit_arm", "unit")

    def build_order(_arm, _design, **kwargs):
        return [0, 1] if kwargs["hyper"] == 0 else [1, 0]

    monkeypatch.setattr(e1, "build_order", build_order)
    design = np.ones((4, 2), dtype=int)
    validation = np.array(
        [[1, 0], [1, 0], [1, 1], [1, 0], [1, 1],
         [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]],
        dtype=int,
    )
    calibration_a = validation.copy()
    calibration_b = np.array(
        [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
         [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
        dtype=int,
    )
    test = validation.copy()
    common = dict(
        arm="unit_arm",
        design=design,
        val=validation,
        test=test,
        names=["a", "b"],
        max_depth=2,
        seed=1,
        alpha=0.3,
    )

    first = e1._tune_one_arm(cal=calibration_a, **common)
    perturbed = e1._tune_one_arm(cal=calibration_b, **common)

    assert first["chosen_hyper"] == perturbed["chosen_hyper"] == 0
    assert first["final_calibration_count"] == perturbed["final_calibration_count"] == 1
    assert first["final_calibrated_depth"] != perturbed["final_calibrated_depth"]


def _make_query_level_fixture():
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
    raw_a = sequential_scores(probabilities, dependence=FACTORIZED_INDEPENDENCE)
    raw_b = raw_a[:, ::-1]
    scores_a = e2.smooth_nonconformity_scores(
        e2.augment_with_racer_null(probabilities, raw_a), smoothing=0.0
    )
    scores_b = e2.smooth_nonconformity_scores(
        e2.augment_with_racer_null(probabilities, raw_b), smoothing=0.0
    )
    return probabilities, standard, scores_a, scores_b


def test_e2_final_calibration_perturbation_cannot_change_selected_hyperparameter() -> None:
    probabilities, standard, scores_a, scores_b = _make_query_level_fixture()
    augmented_probabilities = augment_router_scores(probabilities)
    validation_idx = np.arange(4)
    calibration_idx = np.arange(4, 8)
    test_idx = np.arange(8, 12)
    augmented = augment_ground_truth(standard)
    records = [
        e2._temporary_validation_record(
            arm_key="candidate_a", hyper={"ridge": 0.01}, scores=scores_a,
            augmented_probabilities=augmented_probabilities,
            standard_labels=standard, augmented_labels=augmented,
            val_idx=validation_idx, alpha=0.5,
        ),
        e2._temporary_validation_record(
            arm_key="candidate_b", hyper={"ridge": 0.10}, scores=scores_b,
            augmented_probabilities=augmented_probabilities,
            standard_labels=standard, augmented_labels=augmented,
            val_idx=validation_idx, alpha=0.5,
        ),
    ]
    selected_before = e2._select_validation_candidate(records, 0.5)["chosen"]["arm_key"]

    perturbed_standard = standard.copy()
    perturbed_standard[calibration_idx] = ~perturbed_standard[calibration_idx]
    perturbed_augmented = augment_ground_truth(perturbed_standard)
    selected_after = e2._select_validation_candidate(records, 0.5)["chosen"]["arm_key"]

    chosen_scores = scores_a if selected_before == "candidate_a" else scores_b
    original_final = e2._final_calibrate_and_test(
        scores=chosen_scores,
        augmented_probabilities=augmented_probabilities,
        standard_labels=standard,
        augmented_labels=augmented,
        cal_idx=calibration_idx,
        test_idx=test_idx,
        alpha=0.5,
    )
    perturbed_final = e2._final_calibrate_and_test(
        scores=chosen_scores,
        augmented_probabilities=augmented_probabilities,
        standard_labels=perturbed_standard,
        augmented_labels=perturbed_augmented,
        cal_idx=calibration_idx,
        test_idx=test_idx,
        alpha=0.5,
    )

    assert selected_before == selected_after
    assert original_final["final_calibration_count"] == 1
    assert perturbed_final["final_calibration_count"] == 1


def test_e3_selector_has_no_final_calibration_inputs() -> None:
    parameters = set(inspect.signature(e3._select).parameters)
    assert parameters == {"candidates", "alpha", "class_penalty"}
    candidates = [
        {
            "candidate": "fixed:a", "policy_class": "fixed_prefix",
            "variant": "a", "validation_certified": True,
            "val_calls": 2.0, "val_risk": 0.04,
        },
        {
            "candidate": "wrapper:b", "policy_class": "per_query_wrapper",
            "variant": "b", "validation_certified": True,
            "val_calls": 3.0, "val_risk": 0.03,
        },
    ]
    first = e3._select(candidates, alpha=0.05, class_penalty=False)
    second = e3._select(candidates, alpha=0.05, class_penalty=False)
    assert first["selected"]["candidate"] == second["selected"]["candidate"] == "fixed:a"


def test_e3_missing_marginal_candidate_emits_explicit_noncertified_baseline(
    monkeypatch,
) -> None:
    monkeypatch.setattr(e3, "build_order", lambda *args, **kwargs: [0])
    validation_record = e3._prepare_fixed_prefix_candidate(
        arm="marginal",
        design=np.ones((4, 1), dtype=int),
        validation=np.zeros((20, 1), dtype=int),
        names=["m0"],
        max_depth=1,
        seed=1,
        alpha=0.10,
    )
    assert validation_record["candidate"] == "fixed:marginal"
    assert not validation_record["validation_certified"]
    assert validation_record["validation_fallback"] is True
    assert validation_record["fallback"] is True
    assert validation_record["val_calls"] == pytest.approx(1.0)
    assert validation_record["val_standard_all_miss_risk"] == pytest.approx(1.0)

    baseline = e3._finalize_predeclared_fixed_baseline(
        candidates=[],
        candidate_id="fixed:marginal",
        variant="marginal",
        calibration_labels=np.ones((20, 1), dtype=int),
        test_labels=np.ones((4, 1), dtype=int),
        alpha=0.10,
    )
    assert baseline["candidate"] == "fixed:marginal"
    assert baseline["non_certification_reason"] == "predeclared_candidate_record_missing"
    assert baseline["validation_certified"] is False
    assert baseline["final_certified"] is False
    assert baseline["test_calls"] is None
    assert baseline["test_risk"] is None
    assert baseline["standard_all_miss_risk"] is None
    assert baseline["risk_validity_claimed"] is False


def test_e3_all_validation_infeasible_selects_fallback_then_final_calibrates() -> None:
    candidates = [
        {
            "candidate": "fixed:z",
            "policy_class": "fixed_prefix",
            "variant": "z",
            "validation_certified": False,
            "validation_fallback": True,
            "fallback": True,
            "val_calls": 1.0,
            "val_risk": 0.25,
            "_order": [0],
        },
        {
            "candidate": "fixed:a",
            "policy_class": "fixed_prefix",
            "variant": "a",
            "validation_certified": False,
            "validation_fallback": True,
            "fallback": True,
            "val_calls": 1.0,
            "val_risk": 0.25,
            "_order": [0, 1],
        },
        {
            "candidate": "wrapper:lower_priority",
            "policy_class": "per_query_wrapper",
            "variant": "lower_priority",
            "validation_certified": False,
            "validation_fallback": True,
            "fallback": True,
            "val_calls": 0.0,
            "val_risk": 0.50,
        },
    ]

    choice = e3._select(candidates, alpha=0.10, class_penalty=True)
    frozen = choice["selected"]
    certified = e3._finalize_fixed_prefix(
        candidate=frozen,
        calibration_labels=np.ones((20, 2), dtype=int),
        test_labels=np.ones((4, 2), dtype=int),
        alpha=0.10,
    )
    failed_after_perturbation = e3._finalize_fixed_prefix(
        candidate=frozen,
        calibration_labels=np.zeros((20, 2), dtype=int),
        test_labels=np.ones((4, 2), dtype=int),
        alpha=0.10,
    )

    assert choice["fallback_used"] is True
    assert choice["selection_non_certification"] is True
    assert choice["fallback_reason"] == "no_validation_certified_candidate"
    assert frozen["candidate"] == "fixed:a"
    assert frozen["validation_certified"] is False
    assert frozen["validation_fallback"] is True
    assert certified["candidate"] == "fixed:a"
    assert certified["validation_certified"] is False
    assert certified["final_certified"] is True
    assert certified["risk_validity_claimed"] is True
    assert certified["final_calibration_count"] == 1
    assert failed_after_perturbation["candidate"] == "fixed:a"
    assert failed_after_perturbation["final_certified"] is False
    assert failed_after_perturbation["final_calibration_count"] == 1


def test_e3_final_noncertification_has_no_operating_point_or_risk_claim() -> None:
    candidate = {
        "candidate": "fixed:marginal",
        "policy_class": "fixed_prefix",
        "variant": "marginal",
        "validation_operating_point": 1.0,
        "validation_certified": True,
        "validation_non_certification": False,
        "val_calls": 1.0,
        "val_risk": 0.0,
        "_order": [0],
    }
    result = e3._finalize_fixed_prefix(
        candidate=candidate,
        calibration_labels=np.zeros((20, 1), dtype=int),
        test_labels=np.ones((4, 1), dtype=int),
        alpha=0.10,
    )

    assert result["validation_certified"] is True
    assert result["final_certified"] is False
    assert result["non_certification"] is True
    assert result["non_certification_reason"] == "final_calibration_non_certification"
    assert result["final_calibration_count"] == 1
    assert result["test_calls"] is None
    assert result["test_risk"] is None
    assert result["standard_all_miss_risk"] is None
    assert result["risk_validity_claimed"] is False


def test_alpha_below_finite_sample_resolution_is_non_certification_only() -> None:
    critical = np.array([0.1, 0.2, 0.3, 0.4])
    calibration = select_crc_threshold(critical, alpha=0.1)

    assert not calibration.feasible
    assert np.isneginf(calibration.threshold)
    assert calibration.non_certification_reason == "below_finite_sample_resolution"
    assert calibration.crc_bound > calibration.alpha


def test_result_writers_refuse_existing_output_paths() -> None:
    e1_source = Path(e1.__file__).read_text()
    e2_source = Path(e2.__file__).read_text()
    e3_source = Path(e3.__file__).read_text()
    assert "refusing to overwrite immutable result" in e1_source
    assert "refusing to overwrite immutable result" in e2_source
    assert "refusing to overwrite immutable result" in e3_source


def test_e2_standard_calibration_keeps_real_calls_when_null_is_selected() -> None:
    scores = np.array(
        [
            [0.1, 0.9, 0.0],
            [0.2, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [0.2, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [0.2, 0.9, 0.0],
        ]
    )
    standard = np.array([[1, 0]] * 6, dtype=bool)
    probabilities = np.array([[0.9, 0.1]] * 6)

    result = e2._final_calibrate_and_test(
        scores=scores,
        augmented_probabilities=augment_router_scores(probabilities),
        standard_labels=standard,
        augmented_labels=augment_ground_truth(standard),
        cal_idx=np.arange(4),
        test_idx=np.arange(4, 6),
        alpha=0.4,
    )

    assert result["certified"] is True
    assert result["calls_test"] == pytest.approx(1.0)
    assert result["standard_all_miss_risk"] == pytest.approx(0.0)
    assert result["null_selection_rate"] == pytest.approx(1.0)
    assert result["empty_set_rate"] == pytest.approx(0.0)
    assert result["risk_test"] == result["standard_all_miss_risk"]


def test_e2_structurally_infeasible_standard_target_has_no_zero_call_record() -> None:
    scores = np.array(
        [
            [0.1, 0.9, 0.0],
            [0.2, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [0.2, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [0.2, 0.9, 0.0],
        ]
    )
    standard = np.array(
        [[0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0]], dtype=bool
    )
    probabilities = np.array([[0.9, 0.1]] * 6)

    result = e2._final_calibrate_and_test(
        scores=scores,
        augmented_probabilities=augment_router_scores(probabilities),
        standard_labels=standard,
        augmented_labels=augment_ground_truth(standard),
        cal_idx=np.arange(4),
        test_idx=np.arange(4, 6),
        alpha=0.4,
    )

    assert result["certified"] is False
    assert result["non_certification"] is True
    assert result["non_certification_reason"] == "standard_risk_structurally_infeasible"
    assert result["unavoidable_calibration_misses"] == 2
    assert result["calls_test"] is None
    assert result["risk_test"] is None
    assert result["risk_validity_claimed"] is False


def test_e3_zero_call_noncertified_candidate_cannot_beat_certified_policy() -> None:
    candidates = [
        {
            "candidate": "noncertified:all_abstain",
            "policy_class": "per_query_wrapper",
            "variant": "all_abstain",
            "validation_certified": False,
            "val_calls": 0.0,
            "val_risk": 0.0,
        },
        {
            "candidate": "fixed:marginal",
            "policy_class": "fixed_prefix",
            "variant": "marginal",
            "validation_certified": True,
            "val_calls": 2.0,
            "val_risk": 0.04,
        },
    ]

    choice = e3._select(candidates, alpha=0.05, class_penalty=False)

    assert choice["selected"]["candidate"] == "fixed:marginal"
    assert choice["selection_non_certification"] is False
