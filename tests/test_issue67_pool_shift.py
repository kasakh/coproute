"""Focused synthetic invariants for the ISSUE-67 pool-shift runner."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from scripts import run_issue67_pool_shift as issue67


def _candidate_data(rows: int = 120) -> tuple[np.ndarray, list[str]]:
    """Build many quality families plus exact siblings of the competent core."""

    labels = np.zeros((rows, 44), dtype=np.int8)
    labels[:72, 0] = 1
    labels[60:, 1] = 1
    for column in range(2, 40):
        start = (column * 7) % rows
        labels[np.arange(start, start + 30) % rows, column] = 1
        labels[np.arange(column % 5, rows, 11), column] = 1
    labels[:, 40] = labels[:, 0]
    labels[:, 41] = labels[:, 1]
    labels[:, 42] = labels[:, 0]
    labels[:, 43] = labels[:, 1]
    model_ids = [f"org/family{column:02d}-instruct" for column in range(40)]
    model_ids.extend([
        "other/family00-chat",
        "other/family01-sft",
        "third/family00-dpo",
        "third/family01-rlhf",
    ])
    return labels, model_ids


def _split() -> dict[str, np.ndarray]:
    return {
        "design": np.asarray([0, 1]),
        "router": np.asarray([2, 3]),
        "final_calibration": np.asarray([4, 5]),
        "validation": np.asarray([6, 7]),
        "test": np.asarray([8, 9]),
    }


def test_duplicate_and_specialist_paths_separate_but_share_core() -> None:
    labels, model_ids = _candidate_data()
    result = issue67.construct_pool_paths(labels, model_ids)
    duplicate = result["paths"]["duplicate_growth"]["pools"]["32"]
    specialist = result["paths"]["specialist_growth"]["pools"]["32"]

    assert duplicate["model_ids"][:2] == specialist["model_ids"][:2]
    assert duplicate["model_ids"] != specialist["model_ids"]
    assert len(set(specialist["family_labels"])) == 32
    assert len(set(duplicate["family_labels"])) < 32
    assert result["construction_source"] == "design_labels_only"
    assert result["paths"]["duplicate_growth"]["held_out_roles_used"] is False
    assert result["paths"]["specialist_growth"]["held_out_roles_used"] is False


def test_pool_paths_are_nested_exact_sizes_and_hashes_are_immutable() -> None:
    labels, model_ids = _candidate_data()
    result = issue67.construct_pool_paths(labels, model_ids)
    issue67.validate_pool_paths(result)
    for path_name in issue67.PATH_NAMES:
        prior: list[str] = []
        for size in issue67.POOL_SIZES:
            pool = result["paths"][path_name]["pools"][str(size)]
            assert len(pool["model_ids"]) == size
            assert pool["model_ids"][: len(prior)] == prior
            assert pool["pool_sha256"] == issue67.hash_json(pool["model_ids"])
            prior = pool["model_ids"]

    broken = copy.deepcopy(result)
    broken["paths"]["duplicate_growth"]["pools"]["8"]["model_ids"][-1] = "drift"
    with pytest.raises(RuntimeError, match="pool hash drift"):
        issue67.validate_pool_paths(broken)


def test_pool_construction_cannot_see_held_out_labels() -> None:
    labels, model_ids = _candidate_data()
    design_rows = np.arange(40)
    first = issue67.construct_pool_paths(labels[design_rows], model_ids)
    changed = labels.copy()
    changed[40:] = 1 - changed[40:]
    second = issue67.construct_pool_paths(changed[design_rows], model_ids)
    assert first == second


def _certified_base() -> dict:
    return {
        "certified": True,
        "threshold": 0.5,
        "maximum_allowed_misses": 1,
        "test_calls": 1.5,
        "test_query_ids": ["q8", "q9"],
    }


def test_add_remove_are_explicit_and_keep_identical_test_rows() -> None:
    split = _split()
    query_ids = [f"q{row}" for row in range(10)]
    scores = np.asarray([[0.2, 0.8]] * 10)
    original_labels = np.asarray([[1, 0]] * 10, dtype=np.int8)

    removed = issue67._unchanged_intervention(
        _certified_base(), scores, original_labels, original_labels[:, 1:], split,
        query_ids, ["core", "peer"], ["peer"],
        kind="remove_core_model", changed_model_id="core",
    )
    added_labels = np.column_stack([original_labels, np.ones(10, dtype=np.int8)])
    added = issue67._unchanged_intervention(
        _certified_base(), scores, original_labels, added_labels, split,
        query_ids, ["core", "peer"], ["core", "peer", "specialist"],
        kind="add_held_out_specialist", changed_model_id="specialist",
    )

    assert removed["intervention"] == "remove_core_model"
    assert added["intervention"] == "add_held_out_specialist"
    assert removed["changed_pool_model_ids"] == ["peer"]
    assert added["changed_pool_model_ids"][-1] == "specialist"
    assert removed["test_query_ids"] == added["test_query_ids"] == ["q8", "q9"]
    assert removed["same_test_rows_as_declared_pool"] is True
    assert added["same_test_rows_as_declared_pool"] is True
    assert added["test_calls_by_query"] == [1, 1]
    assert added["test_miss_by_query"] == [0, 0]
    assert added["recalibration_performed"] is False
    assert removed["changed_complete_pool"]["test_miss_count"] == 2
    assert added["changed_complete_pool"]["test_miss_count"] == 0


def test_recalibrated_path_is_distinct_from_unchanged_certificate() -> None:
    split = _split()
    query_ids = [f"q{row}" for row in range(10)]
    labels = np.ones((10, 2), dtype=np.int8)
    scores = {method: np.asarray([[0.1, 0.9]] * 10) for method in issue67.REGISTERED_METHODS}
    metadata = {method: {} for method in issue67.REGISTERED_METHODS}
    result = issue67._recalibrated_intervention(
        "racer_g", scores, metadata, labels, split, query_ids, ["a", "b"], 0.20,
        kind="add_held_out_specialist", changed_model_id="b",
    )
    assert result["evaluation_path"] == "refit_and_recalibrated_changed_pool"
    assert result["refit_performed"] is True
    assert result["recalibration_performed"] is True
    assert result["changed_pool_calibration_labels_recomputed"] is True
    assert result["changed_pool_calibration_query_ids"] == ["q4", "q5"]
    assert result["test_query_ids"] == ["q8", "q9"]


def test_chain_discount_endpoint_is_exact() -> None:
    rng = np.random.default_rng(9)
    labels = (rng.uniform(size=(80, 4)) < np.asarray([0.8, 0.6, 0.4, 0.3])).astype(np.int8)
    probabilities = np.broadcast_to(np.asarray([0.8, 0.6, 0.4, 0.3]), labels.shape).copy()
    scores, metadata = issue67._fit_scores(
        probabilities, labels, np.arange(30), np.arange(30, 50),
        ["a", "b", "c", "d"], 0.20, split_seed=7001,
    )
    assert set(scores) == set(issue67.REGISTERED_METHODS)
    chain = metadata["fixed_tally_chain_discounted"]
    assert chain["endpoint_reference_method"] == "fixed_tally_factorized"
    assert chain["endpoint_score_identity_verified"] is True
    assert chain["endpoint_score_sha256"] == issue67.hash_json(
        scores["fixed_tally_factorized"].tolist()
    )
    assert chain["validation_trajectory"][0]["hyper"] == {
        "minimum_survivor_count": 0
    }


def _minimal_complete_cells() -> list[dict]:
    cells = []
    for alpha in issue67.ALPHA_GRID:
        paths = []
        for path_name in issue67.PATH_NAMES:
            for size in issue67.POOL_SIZES:
                model_ids = [f"{path_name}-{index}" for index in range(size)]
                stress = []
                if size >= 4:
                    for kind in issue67.STRESS_KINDS:
                        unchanged = {method: {} for method in issue67.REGISTERED_METHODS}
                        recalibrated = {method: {} for method in issue67.REGISTERED_METHODS}
                        stress.append({
                            "intervention": kind,
                            "unchanged_deployment": unchanged,
                            "refit_and_recalibrated": recalibrated,
                        })
                paths.append({
                    "path": path_name,
                    "pool_size": size,
                    "pool": {
                        "model_ids": model_ids,
                        "pool_sha256": issue67.hash_json(model_ids),
                    },
                    "methods": {
                        method: {"certified": True}
                        for method in issue67.REGISTERED_METHODS
                    },
                    "availability_stress": stress,
                })
        cells.append({"alpha": alpha, "redraws": [{"redraw_id": 0, "paths": paths}]})
    return cells




def test_subgroup_metadata_is_used_only_when_explicitly_exposed() -> None:
    class Task:
        query_ids = tuple(f"q{row}" for row in range(10))
        subgroup_labels = tuple("group-a" if row % 2 == 0 else "group-b" for row in range(10))

    available = issue67._subgroup_context(Task(), _split())
    assert available["available"] is True
    assert available["source_attribute"] == "subgroup_labels"
    assert available["test_group_by_query"] == ["group-a", "group-b"]

    class NoGroups:
        query_ids = Task.query_ids

    unavailable = issue67._subgroup_context(NoGroups(), _split())
    assert unavailable["available"] is False
    assert "no task or subgroup labels" in unavailable["reason"]


def test_no_hidden_aggregate_and_missing_intervention_fails() -> None:
    cells = _minimal_complete_cells()
    issue67.assert_complete_output(cells)
    broken = copy.deepcopy(cells)
    target = next(path for path in broken[0]["redraws"][0]["paths"] if path["pool_size"] == 4)
    target["availability_stress"].pop()
    with pytest.raises(RuntimeError, match="hidden or missing"):
        issue67.assert_complete_output(broken)


def test_bootstrap_is_deterministic() -> None:
    values = [1.0, -0.5, 0.25, 2.0, -1.0]
    first = issue67.deterministic_paired_bootstrap(values, seed=123, repetitions=10_000)
    second = issue67.deterministic_paired_bootstrap(values, seed=123, repetitions=10_000)
    assert first == second
    assert first["bootstrap_repetitions"] == 10_000
    assert first["difference_direction"] == "specialist_growth_minus_duplicate_growth"


def test_duplicate_real_model_ids_fail() -> None:
    labels, model_ids = _candidate_data()
    model_ids[-1] = model_ids[0]
    with pytest.raises(ValueError, match="duplicate model IDs"):
        issue67.construct_pool_paths(labels, model_ids)


def test_output_collision_is_atomic_and_refused(tmp_path: Path) -> None:
    destination = tmp_path / "issue67-host-stamp"
    payload = {"host": "host", "stamp": "stamp", "benchmark": "ifeval"}
    published = issue67.publish_output(destination, payload)
    before = {path.name: path.read_bytes() for path in published.iterdir()}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        issue67.publish_output(destination, payload)
    after = {path.name: path.read_bytes() for path in published.iterdir()}
    assert before == after
    assert not list(tmp_path.glob("*.tmp.*"))
