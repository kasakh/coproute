"""Synthetic regression tests for the repaired RouterBench E2 runner."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.racer_wrapper import (  # noqa: E402
    FACTORIZED_INDEPENDENCE,
    PAIRWISE_HEURISTIC,
)
from scripts import design_bprime_e2_routerbench_run as runner  # noqa: E402
from scripts import routerbench_feasibility_report as feasibility  # noqa: E402


def _synthetic_task(n_queries: int = 120) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    generator = np.random.default_rng(991)
    embeddings = generator.normal(size=(n_queries, 8))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    correctness = np.zeros((n_queries, 4), dtype=np.int8)
    rows = np.arange(n_queries)
    correctness[rows, rows % 4] = 1
    correctness[rows[::5], (rows[::5] + 1) % 4] = 1
    query_ids = tuple(f"synthetic:{index}" for index in range(n_queries))
    return correctness, embeddings, query_ids


def test_partition_is_disjoint_complete_and_reproducible() -> None:
    first = runner.partition_task_rows(120, seed=7001, redraw=3)
    second = runner.partition_task_rows(120, seed=7001, redraw=3)
    changed = runner.partition_task_rows(120, seed=7001, redraw=4)

    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    combined = np.concatenate(tuple(first.values()))
    assert len(combined) == 120
    assert len(np.unique(combined)) == 120
    assert set(combined.tolist()) == set(range(120))
    assert any(not np.array_equal(first[name], changed[name]) for name in first)


def test_final_calibration_labels_cannot_change_selected_hyperparameter() -> None:
    correctness, embeddings, query_ids = _synthetic_task()
    split = runner.partition_task_rows(120, seed=7001, redraw=0)
    common = dict(
        embeddings=embeddings,
        query_ids=query_ids,
        alphas=(0.5,),
        redraw=0,
        seed=7001,
        knn_k=8,
        max_depth=4,
        wrapper_arms=(FACTORIZED_INDEPENDENCE, PAIRWISE_HEURISTIC),
        minimum_partition_size=1,
    )
    original, original_split = runner.evaluate_task_redraw(
        correctness=correctness, **common
    )

    perturbed = correctness.copy()
    perturbed[split["final_calibration"]] = 0
    after, after_split = runner.evaluate_task_redraw(correctness=perturbed, **common)

    assert original_split == after_split
    for arm in (FACTORIZED_INDEPENDENCE, PAIRWISE_HEURISTIC):
        name = f"wrapper_{arm}_tuned"
        before_record = original["0.5"][name]
        after_record = after["0.5"][name]
        assert before_record["chosen_hyper"] == after_record["chosen_hyper"]
        assert before_record["selection_source"] == "validation_only"
        assert after_record["selection_source"] == "validation_only"
        assert before_record["final_calibration_count"] == 1
        assert after_record["final_calibration_count"] == 1
        assert before_record["certified"] is True
        assert after_record["certified"] is False
        assert after_record["non_certification"] is True
        assert after_record["calls_test"] is None
        assert after_record["risk_test"] is None
        assert after_record["standard_all_miss_risk"] is None
        assert after_record["risk_validity_claimed"] is False


def test_controls_use_same_standard_noncertification_behavior() -> None:
    correctness, embeddings, query_ids = _synthetic_task()
    split = runner.partition_task_rows(120, seed=7001, redraw=1)
    correctness[split["final_calibration"]] = 0
    records, split_record = runner.evaluate_task_redraw(
        correctness=correctness,
        embeddings=embeddings,
        query_ids=query_ids,
        alphas=(0.5,),
        redraw=1,
        seed=7001,
        knn_k=8,
        max_depth=4,
        wrapper_arms=(FACTORIZED_INDEPENDENCE,),
        minimum_partition_size=1,
    )

    assert split_record["router_fit_scope"] == "same_task_dedicated_router_split_only"
    assert split_record["knn_k_effective"] == 8
    for control in runner.RACER_CONTROLS:
        record = records["0.5"][control]
        assert record["selection_source"] == "predeclared_control"
        assert record["final_calibration_count"] == 1
        assert record["certified"] is False
        assert record["calls_test"] is None
        assert record["risk_test"] is None


def test_runner_registers_only_issue50_routerbench_e2_tasks() -> None:
    assert runner.CLEAN_TASKS == ("mmlu", "hellaswag", "winogrande")
    assert runner._parse_tasks("mmlu,hellaswag,winogrande") == runner.CLEAN_TASKS
    with pytest.raises(ValueError, match="supports only"):
        runner._parse_tasks("arc-challenge")
    assert runner.ROUTER_PROTOCOL == "within_task_knn_on_mdeberta_dedicated_router_split"
    assert runner.SELECTION_PROTOCOL.endswith("one_final_calibration_then_test")


def test_feasibility_report_has_exact_depths_and_boundaries() -> None:
    labels = np.asarray(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.int8,
    )
    report = feasibility.task_feasibility(
        labels,
        ("a", "b", "c"),
        targets=(0.19, 0.20, 0.59, 0.60),
    )

    assert report["complete_pool_all_miss_count"] == 1
    assert report["complete_pool_all_miss_rate"] == pytest.approx(0.20)
    assert report["best_single_model"] == "a"
    assert report["best_single_model_miss_count"] == 3
    assert report["best_single_model_miss_rate"] == pytest.approx(0.60)
    assert [item["miss_count"] for item in report["marginal_depths"]] == [3, 2, 1]
    assert report["target_classification"] == {
        "0.19": "infeasible_even_with_complete_pool",
        "0.2": "feasible_with_multi_model_headroom",
        "0.59": "feasible_with_multi_model_headroom",
        "0.6": "target_already_met_with_one_call",
    }
