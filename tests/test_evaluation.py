from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.artifacts import validate_raw_artifacts  # noqa: E402
from components.evaluation import (  # noqa: E402
    FrozenAggregator,
    compare_selector_evaluations,
    evaluate_routing_result,
    evaluate_selector,
    evaluate_selectors,
    select_common_aggregator,
)
from components.racer_protocol import RACERProtocol  # noqa: E402


def _artifact_metadata(n_queries: int) -> dict:
    return {
        "split_ids": {
            "design": ["design-0"],
            "calibration": ["calibration-0"],
            "test": [f"test-{index}" for index in range(n_queries)],
        },
        "seeds": {"design": 11, "calibration": 12, "test": 13},
        "correlations": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def _evaluation_inputs() -> dict:
    return {
        "tasks": ["mmlu", "mmlu"],
        "responses": [
            ["A", "A", "B"],
            ["B", "B", "A"],
        ],
        "router_scores": [
            [0.4, 0.3, 0.9],
            [0.4, 0.3, 0.9],
        ],
        "reference_answers": ["A", "B"],
        "correctness": [
            [1, 1, 0],
            [1, 1, 0],
        ],
        "thresholds": [0.2, 0.2],
        "depths": [3, 3],
        "model_costs": [1.0, 2.0, 4.0],
        "latencies": [0.05, 0.07],
        "query_ids": ["test-0", "test-1"],
        **_artifact_metadata(2),
    }


def test_validation_selects_canonical_majority_and_freezes_it() -> None:
    inputs = _evaluation_inputs()
    frozen = select_common_aggregator(
        tasks=inputs["tasks"],
        responses=inputs["responses"],
        router_scores=inputs["router_scores"],
        reference_answers=inputs["reference_answers"],
        selected_sets={
            "selector-a": [(0, 1, 2), (0, 1, 2)],
            "selector-b": [(0, 1, 2), (0, 1, 2)],
        },
    )

    assert isinstance(frozen, FrozenAggregator)
    assert frozen.name == "canonical_majority"
    assert frozen.scoring_method == "majority"
    assert tuple(score.name for score in frozen.validation_scores) == (
        "top_router",
        "canonical_majority",
        "score_weighted",
        "full_ensemble",
    )
    assert frozen.validation_scores[1].accuracy == 1.0


def test_every_selector_reuses_the_same_frozen_aggregator() -> None:
    inputs = _evaluation_inputs()
    frozen = select_common_aggregator(
        tasks=inputs["tasks"],
        responses=inputs["responses"],
        router_scores=inputs["router_scores"],
        reference_answers=inputs["reference_answers"],
        selected_sets=[(0, 1, 2), (0, 1, 2)],
    )

    multi_selector_inputs = dict(inputs)
    multi_selector_inputs["depths"] = {
        "marginal": [3, 3],
        "copula": [2, 2],
    }
    reports = evaluate_selectors(
        {
            "marginal": [(0, 1, 2), (0, 1, 2)],
            "copula": [(0, 1), (0, 1)],
        },
        frozen_aggregator=frozen,
        **multi_selector_inputs,
    )

    assert set(reports) == {"marginal", "copula"}
    for report in reports.values():
        assert report["frozen_aggregator"]["name"] == frozen.name
        assert {row["aggregator"] for row in report["predictions"]} == {frozen.name}
        validate_raw_artifacts(report)


def test_positive_model_label_cannot_make_wrong_aggregate_correct() -> None:
    frozen = select_common_aggregator(
        tasks=["mmlu"],
        responses=[["B", "A", "C"]],
        router_scores=[[0.9, 0.2, 0.1]],
        reference_answers=["A"],
        selected_sets=[(0, 1)],
        candidates=("top_router",),
    )

    # The selected top-router model is deliberately marked positive even
    # though its canonical answer disagrees with the supplied reference. The
    # correctness bit may establish selection coverage, but it must not score
    # the aggregate as correct.
    report = evaluate_selector(
        "label-isolation",
        frozen_aggregator=frozen,
        tasks=["mmlu"],
        responses=[["B", "A", "C"]],
        router_scores=[[0.9, 0.2, 0.1]],
        reference_answers=["A"],
        correctness=[[1, 0, 0]],
        selected_sets=[(0, 1)],
        thresholds=[0.3],
        depths=[2],
        model_costs=[1.0, 2.0, 4.0],
        latencies=[0.08],
        query_ids=["test-0"],
        **_artifact_metadata(1),
    )

    prediction = report["predictions"][0]
    assert prediction["selection_covered"] is True
    assert prediction["coverage_loss"] is False
    assert prediction["canonical_answer"] == "B"
    assert prediction["final_correct"] is False
    assert report["summary"]["selection_coverage"] == 1.0
    assert report["summary"]["aggregation_accuracy"] == 0.0


def test_evaluation_never_passes_model_labels_to_aggregation() -> None:
    inputs = _evaluation_inputs()
    frozen = select_common_aggregator(
        tasks=inputs["tasks"],
        responses=inputs["responses"],
        router_scores=inputs["router_scores"],
        reference_answers=inputs["reference_answers"],
        selected_sets=[(0, 1, 2), (0, 1, 2)],
        candidates=("canonical_majority",),
    )

    from components import evaluation

    real_aggregate = evaluation.aggregate_responses
    with mock.patch(
        "components.evaluation.aggregate_responses", wraps=real_aggregate
    ) as aggregate_spy:
        report = evaluate_selector(
            "no-labels",
            frozen_aggregator=frozen,
            selected_sets=[(0, 1, 2), (0, 1, 2)],
            **inputs,
        )

    assert report["summary"]["aggregation_accuracy"] == 1.0
    assert aggregate_spy.call_count == 2
    for call in aggregate_spy.call_args_list:
        assert "labels" not in call.kwargs
        assert "reference_answer" not in call.kwargs


def test_full_ensemble_costs_all_models_but_keeps_selection_coverage_separate() -> None:
    frozen = select_common_aggregator(
        tasks=["mmlu"],
        responses=[["A", "A", "B"]],
        router_scores=[[0.5, 0.4, 0.3]],
        reference_answers=["A"],
        selected_sets=[(2,)],
        candidates=("full_ensemble",),
    )
    report = evaluate_selector(
        "single-wrong-model",
        frozen_aggregator=frozen,
        tasks=["mmlu"],
        responses=[["A", "A", "B"]],
        router_scores=[[0.5, 0.4, 0.3]],
        reference_answers=["A"],
        correctness=[[1, 1, 0]],
        selected_sets=[(2,)],
        thresholds=[0.1],
        depths=[1],
        model_costs=[1.0, 2.0, 4.0],
        latencies=[0.1],
        query_ids=["test-0"],
        **_artifact_metadata(1),
    )

    prediction = report["predictions"][0]
    assert prediction["selected_set"] == [2]
    assert prediction["aggregation_used_set"] == [0, 1, 2]
    assert prediction["coverage_loss"] is True
    assert prediction["final_correct"] is True
    assert prediction["cost"] == pytest.approx(7.0)
    assert report["summary"]["selection_coverage"] == 0.0
    assert report["summary"]["aggregation_accuracy"] == 1.0


def test_depth_and_selected_set_size_must_agree() -> None:
    inputs = _evaluation_inputs()
    frozen = select_common_aggregator(
        tasks=inputs["tasks"],
        responses=inputs["responses"],
        router_scores=inputs["router_scores"],
        reference_answers=inputs["reference_answers"],
        selected_sets=[(0, 1, 2), (0, 1, 2)],
    )

    with pytest.raises(ValueError, match="depth"):
        evaluate_selector(
            "bad-depth",
            frozen_aggregator=frozen,
            selected_sets=[(0, 1), (0, 1)],
            **inputs,
        )


def test_racer_routing_result_flows_through_common_evaluation() -> None:
    frozen = select_common_aggregator(
        tasks=["mmlu", "mmlu"],
        responses=[["A", "C", "D"], ["C", "B", "D"]],
        router_scores=[[0.9, 0.2, 0.1], [0.2, 0.9, 0.1]],
        reference_answers=["A", "B"],
        selected_sets=[(0,), (1,)],
        candidates=("top_router",),
    )
    protocol = RACERProtocol("P", smoothing=0.0)
    routing = protocol.route(
        np.array([[0.9, 0.2, 0.1], [0.2, 0.9, 0.1]]),
        threshold=0.2,
    )

    report = evaluate_routing_result(
        "racer-p",
        routing=routing,
        calibration=0.2,
        frozen_aggregator=frozen,
        tasks=["mmlu", "mmlu"],
        responses=[["A", "C", "D"], ["C", "B", "D"]],
        router_scores=[[0.9, 0.2, 0.1], [0.2, 0.9, 0.1]],
        reference_answers=["A", "B"],
        correctness=[[1, 0, 0], [0, 1, 0]],
        model_costs=[1.0, 2.0, 4.0],
        latencies=[0.05, 0.06],
        query_ids=["test-0", "test-1"],
        **_artifact_metadata(2),
    )

    assert report["selected_sets"] == [[0], [1]]
    assert report["depths"] == [1, 1]
    assert report["thresholds"] == [0.2, 0.2]
    assert report["summary"]["selection_coverage"] == 1.0
    assert report["summary"]["aggregation_accuracy"] == 1.0


def test_selector_comparison_uses_paired_query_level_inference() -> None:
    inputs = _evaluation_inputs()
    frozen = select_common_aggregator(
        tasks=inputs["tasks"],
        responses=inputs["responses"],
        router_scores=inputs["router_scores"],
        reference_answers=inputs["reference_answers"],
        selected_sets=[(0, 1, 2), (0, 1, 2)],
        candidates=("canonical_majority",),
    )
    baseline = evaluate_selector(
        "baseline",
        frozen_aggregator=frozen,
        selected_sets=[(0, 1, 2), (0, 1, 2)],
        **inputs,
    )
    corrected_inputs = dict(inputs)
    corrected_inputs["depths"] = [2, 2]
    corrected = evaluate_selector(
        "corrected",
        frozen_aggregator=frozen,
        selected_sets=[(0, 1), (0, 1)],
        **corrected_inputs,
    )

    comparison = compare_selector_evaluations(
        baseline,
        corrected,
        n_bootstrap=100,
        seed=19,
    )

    assert comparison["resampling_unit"] == "query_pair"
    assert comparison["n_queries"] == 2
    assert comparison["metrics"]["calls"]["difference"]["estimate"] == -1.0
