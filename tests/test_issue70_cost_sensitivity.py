"""Focused invariants for the ISSUE-70 public-price cost runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from components.scale_selectors import build_order
from scripts import run_issue70_cost_sensitivity as issue70


def _split() -> dict[str, np.ndarray]:
    return {
        "design": np.asarray([0, 1]),
        "router": np.asarray([2, 3]),
        "final_calibration": np.asarray([4, 5]),
        "validation": np.asarray([6, 7]),
        "test": np.asarray([8, 9]),
    }


def _task(rows: int = 500, models: int = 4) -> SimpleNamespace:
    generator = np.random.default_rng(20260920)
    correctness = np.zeros((rows, models), dtype=np.int8)
    for row in range(rows):
        correctness[row, row % models] = 1
        correctness[row, (row + 1) % models] = 1
    embeddings = generator.normal(size=(rows, 6))
    return SimpleNamespace(
        task="mmlu",
        query_ids=tuple(f"query-{row}" for row in range(rows)),
        embeddings=embeddings,
        correctness=correctness,
        model_ids=tuple(f"model-{model}" for model in range(models)),
        source_hashes={"routerbench": "a" * 64, "embeddings": "b" * 64},
        feature_audit={"degenerate": False, "duplicate_rate": 0.0},
    )


def _fake_knn(train_x, train_y, query_x, k):
    del train_x, train_y, k
    rows = len(query_x)
    base = np.asarray([0.82, 0.68, 0.54, 0.40])
    variation = (np.arange(rows)[:, None] % 7 - 3) * 0.005
    probabilities = np.clip(base[None, :] + variation, 0.01, 0.99)
    support = np.empty((rows, 0, len(base)))
    return probabilities, support


def test_cost_vector_uses_only_design_and_router_rows() -> None:
    split = _split()
    query_ids = [f"q-{row}" for row in range(10)]
    model_ids = ["a", "b", "c"]
    costs = np.arange(30, dtype=float).reshape(10, 3) + 1.0
    frozen, provenance = issue70.freeze_cost_vector(
        costs, split, query_ids, model_ids
    )
    expected = costs[[0, 1, 2, 3]].mean(axis=0)
    np.testing.assert_allclose(frozen, expected)
    assert provenance["fit_roles"] == ["design", "router"]
    assert provenance["fit_query_ids"] == ["q-0", "q-1", "q-2", "q-3"]
    assert provenance["held_out_rows_used"] is False

    changed = costs.copy()
    changed[[4, 5, 6, 7, 8, 9]] = 1_000_000.0
    changed_frozen, changed_provenance = issue70.freeze_cost_vector(
        changed, split, query_ids, model_ids
    )
    np.testing.assert_array_equal(changed_frozen, frozen)
    assert changed_provenance["cost_vector_sha256"] == provenance["cost_vector_sha256"]


def test_equal_costs_reproduce_count_tally_exactly() -> None:
    labels = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
        ],
        dtype=np.int8,
    )
    count_order = build_order(
        "empirical_tally",
        labels,
        model_names=["a", "b", "c", "d"],
        max_depth=4,
        seed=7001,
    )
    cost_order = issue70.cost_weighted_tally_order(
        labels, np.ones(4), max_depth=4
    )
    assert cost_order == count_order


def test_cost_weighted_greedy_prefers_cheaper_equal_benefit_model() -> None:
    labels = np.asarray(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.int8,
    )
    order = issue70.cost_weighted_tally_order(
        labels, [4.0, 1.0, 2.0], max_depth=3
    )
    assert order[0] == 1


def test_both_d2_discount_off_endpoints_are_exact() -> None:
    task = _task(rows=80)
    probabilities, _ = _fake_knn(None, None, task.embeddings, 8)
    labels = task.correctness
    design = labels[:30]
    count_order = build_order(
        "empirical_tally",
        design,
        model_names=list(task.model_ids),
        max_depth=4,
        seed=7001,
    )
    cost_order = issue70.cost_weighted_tally_order(
        design, [4.0, 1.0, 2.0, 3.0], max_depth=4
    )
    for order in (count_order, cost_order):
        d1 = issue70._fixed_order_scores(
            probabilities, order, base_seed=7001, split_seed=7014
        )
        candidates, _ = issue70._chain_candidates(
            probabilities,
            design,
            order,
            d1,
            base_seed=7001,
            split_seed=7014,
        )
        assert np.array_equal(candidates[0]["scores"], d1)
        assert candidates[0]["hyper"] == {"minimum_survivor_count": 0}


def test_event_and_archived_cost_arithmetic_are_exact() -> None:
    model_ids = ["a", "b", "c"]
    costs = np.asarray(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
        ]
    )
    result = {
        "certified": True,
        "selected_model_ids_by_query": [["a", "c"], ["b"]],
        "test_calls_by_query": [2, 1],
        "test_miss_by_query": [0, 1],
        "standard_all_miss_risk": 0.5,
    }
    enriched = issue70.attach_archived_test_costs(
        result, costs, np.asarray([0, 2]), model_ids
    )
    assert enriched["test_query_count"] == 2
    assert enriched["test_call_sum"] == 3
    assert enriched["test_miss_count"] == 1
    assert enriched["test_total_cost_by_query"] == [101.0, 30.0]
    assert enriched["test_total_cost_sum"] == 131.0
    assert enriched["test_mean_total_cost"] == 65.5
    assert sum(enriched["test_miss_by_query"]) == 1
    assert enriched["standard_all_miss_risk"] == pytest.approx(
        np.mean(enriched["test_miss_by_query"])
    )


def test_one_redraw_has_all_arms_vectors_endpoints_and_structural_floors() -> None:
    task = _task()
    row_scale = 1.0 + (np.arange(len(task.query_ids)) % 5)[:, None] * 0.02
    model_cost = np.asarray([4.0, 1.0, 2.0, 3.0])[None, :]
    costs = row_scale * model_cost
    payload = issue70.run_issue70_task(
        task,
        costs,
        host_id="test_host",
        run_stamp="20260920T000000Z",
        redraws=1,
        knn_k=8,
        knn_function=_fake_knn,
    )
    assert payload["method_identities"] == list(issue70.REGISTERED_METHODS)
    assert payload["preserve_prefixes"] is True
    assert payload["target_grid"] == list(issue70.ALPHA_GRID)
    assert payload["cost_fit_roles"] == ["design", "router"]
    assert len(payload["paired_bootstrap_summaries"]) == (
        len(issue70.ALPHA_GRID) * len(issue70.COMPARISON_PAIRS)
    )

    for cell in payload["cells"]:
        redraw = cell["redraws"][0]
        assert set(redraw["methods"]) == set(issue70.REGISTERED_METHODS)
        assert redraw["full_pool"]["test_miss_count"] == sum(
            redraw["full_pool"]["test_miss_by_query"]
        )
        structural = redraw["structural_feasibility"]
        assert structural["complete_pool_test_floor"] == pytest.approx(
            np.mean(redraw["full_pool"]["test_miss_by_query"])
        )
        for d2, d1 in (
            (issue70.COUNT_D2, issue70.COUNT_D1),
            (issue70.COST_D2, issue70.COST_D1),
        ):
            result = redraw["methods"][d2]
            assert result["endpoint_reference_method"] == d1
            assert result["endpoint_score_identity_verified"] is True
            assert result["endpoint_result_identity_verified"] is True
            assert result["chosen_hyper"]["minimum_survivor_count"] in issue70.CHAIN_GRID
        for result in redraw["methods"].values():
            if result["certified"]:
                assert len(result["selected_model_ids_by_query"]) == len(
                    redraw["split_ids"]["test"]
                )
                assert result["test_calls_by_query"] == [
                    len(ids) for ids in result["selected_model_ids_by_query"]
                ]
                assert result["test_total_cost_sum"] == pytest.approx(
                    sum(result["test_total_cost_by_query"])
                )
                assert result["standard_all_miss_risk"] == pytest.approx(
                    np.mean(result["test_miss_by_query"])
                )


def test_redraw_start_preserves_global_redraw_identity() -> None:
    task = _task()
    costs = np.ones_like(task.correctness, dtype=float)
    payload = issue70.run_issue70_task(
        task,
        costs,
        host_id="test_host",
        run_stamp="20260920T000000Z",
        redraws=1,
        redraw_start=7,
        knn_k=8,
        knn_function=_fake_knn,
    )
    assert payload["redraw_start"] == 7
    assert payload["redraw_end"] == 8
    assert {
        row["redraw_id"] for cell in payload["cells"] for row in cell["redraws"]
    } == {7}
    assert {row["redraw_id"] for row in payload["redraw_contexts"]} == {7}


def test_paired_bootstrap_is_deterministic() -> None:
    values = [1.5, -0.5, 2.0, 0.25, -1.0]
    first = issue70.deterministic_paired_bootstrap(
        values, seed=123, repetitions=10_000
    )
    second = issue70.deterministic_paired_bootstrap(
        values, seed=123, repetitions=10_000
    )
    assert first == second
    assert first["bootstrap_repetitions"] == 10_000
    assert first["difference_direction"] == "candidate_minus_reference"




def test_cost_fallback_event_scan_matches_brute_force() -> None:
    generator = np.random.default_rng(19)
    scores = generator.uniform(size=(12, 4))
    labels = (generator.uniform(size=(12, 4)) < 0.35).astype(np.int8)
    rows = np.arange(12)
    frozen_costs = np.asarray([4.0, 1.0, 3.0, 2.0])
    observed = np.unique(scores)
    brute = min(
        (
            float(np.mean(~np.any((scores <= threshold) & labels.astype(bool), axis=1))),
            float(np.mean((scores <= threshold) @ frozen_costs)),
            float(np.mean(np.sum(scores <= threshold, axis=1))),
            float(threshold),
        )
        for threshold in [
            float(np.nextafter(observed[0], -np.inf)),
            *map(float, observed),
        ]
    )
    assert issue70._fallback_with_cost(
        scores, labels, rows, frozen_costs
    ) == pytest.approx(brute)


def test_output_collision_is_refused_without_modification(tmp_path: Path) -> None:
    destination = tmp_path / "issue70_test_host_20260920T000000Z"
    payloads = {task: {"task": task} for task in issue70.APPROVED_TASKS}
    published = issue70.publish_output_directory(
        destination,
        payloads,
        host="test_host",
        stamp="20260920T000000Z",
    )
    before = {
        path.name: path.read_bytes()
        for path in published.iterdir()
        if path.is_file()
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        issue70.publish_output_directory(
            destination,
            payloads,
            host="test_host",
            stamp="20260920T000000Z",
        )
    after = {
        path.name: path.read_bytes()
        for path in published.iterdir()
        if path.is_file()
    }
    assert after == before
    assert not list(tmp_path.glob("*.tmp.*"))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        issue70.main(
            [
                "--routerbench-pickle",
                str(tmp_path / "missing.pkl"),
                "--embedding-archive",
                str(tmp_path / "missing.npz"),
                "--embedding-manifest",
                str(tmp_path / "missing.json"),
                "--host",
                "test_host",
                "--stamp",
                "20260920T000000Z",
                "--out-dir",
                str(destination),
            ]
        )
