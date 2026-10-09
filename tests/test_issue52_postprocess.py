"""Focused tests for ISSUE-52 shard merging and transfer analysis."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures_src import analyze_issue52_transfer as analysis  # noqa: E402
from scripts import merge_issue52_shards as merger  # noqa: E402
from scripts import run_issue52_transfer as transfer  # noqa: E402
from scripts import run_paired_main_suite as paired  # noqa: E402


def _calibration(
    *,
    mode: str,
    alpha: float,
    redraw: int,
    method: str,
    query_ids: list[str],
    score_hash: str,
) -> dict:
    source_mode = mode == "source_calibrated"
    calls = [2, 2, 2, 2] if source_mode else [1, 1, 1, 1]
    misses = [int(source_mode and redraw % 2 == 0), 0, 0, 0]
    return {
        "fit_origin": "routereval:source_task",
        "calibration_origin": (
            "routereval:source_task:final_calibration"
            if source_mode
            else "routereval:target_task:final_calibration"
        ),
        "calibration_query_ids": [f"cal-{redraw}-{index}" for index in range(5)],
        "calibration_query_ids_sha256": "calibration-hash",
        "calibration_scores_sha256": f"calibration-score-{mode}-{redraw}-{method}",
        "frozen_target_scores_sha256": score_hash,
        "target_test_query_ids": query_ids,
        "target_test_query_ids_sha256": paired.hash_json(query_ids),
        "certified": True,
        "non_certification_reason": None,
        "threshold": 0.5,
        "corrected_calibration_bound": alpha,
        "calibration_event_count": redraw % 2,
        "calibration_trial_count": 5,
        "unavoidable_calibration_event_count": 0,
        "maximum_allowed_calibration_events": 1,
        "target_test_calls_by_query": calls,
        "target_test_miss_by_query": misses,
        "target_test_event_count": sum(misses),
        "target_test_trial_count": len(query_ids),
        "target_test_mean_calls": sum(calls) / len(calls),
        "target_test_all_miss_risk": sum(misses) / len(misses),
        "selected_model_ids_by_query": [["model-0"] for _ in query_ids],
    }


def _shard(redraw: int) -> dict:
    methods = list(transfer.registered_methods())
    target_grid = list(paired.ALPHA_GRID)
    cells = []
    for alpha in target_grid:
        query_ids = [f"query-{index}" for index in range(4)]
        method_records = {}
        for method in methods:
            score_hash = f"target-score-{redraw}-{method}"
            record = {
                "method_family": method.split("_seed_", 1)[0],
                "training_seed": int(method.rsplit("_", 1)[1]) if "_seed_" in method else None,
                "fit_origin": "routereval:source_task",
                "fit_mode": "source_fitted",
                "frozen_target_scores_sha256": score_hash,
                "frozen_target_score_shape": [100, 4],
                "preserve_prefixes": True,
                "endpoint": {
                    "model_ids": [f"model-{index}" for index in range(4)],
                    "model_ids_sha256": "model-hash",
                    "reaches_full_pool": True,
                },
                "calibrations": {
                    mode: _calibration(
                        mode=mode,
                        alpha=alpha,
                        redraw=redraw,
                        method=method,
                        query_ids=query_ids,
                        score_hash=score_hash,
                    )
                    for mode in transfer.CALIBRATION_MODES
                },
            }
            if method == paired.CHAIN_DISCOUNT_METHOD:
                record.update({
                    "chosen_hyperparameter": {
                        "minimum_survivor_count": 0 if redraw < 10 else 5
                    },
                    "hyperparameter_selection_origin": "routereval:source_task:validation",
                    "discount_off_endpoint_sha256": f"off-{redraw}",
                    "endpoint_reference_method": "fixed_tally_factorized",
                    "endpoint_reproduction_exact": True,
                })
            method_records[method] = record
        cells.append({
            "alpha": alpha,
            "redraws": [{
                "redraw_id": redraw,
                "split_seed": paired.BASE_SEED + paired.SPLIT_STEP * redraw,
                "source_split_ids": {
                    "design": [f"source-design-{redraw}"],
                    "router": [f"source-router-{redraw}"],
                    "validation": [f"source-validation-{redraw}"],
                    "final_calibration": [f"source-cal-{redraw}"],
                    "test": [f"source-test-{redraw}"],
                },
                "target_split_ids": {
                    "design": [f"target-design-{redraw}"],
                    "router": [f"target-router-{redraw}"],
                    "validation": [f"target-validation-{redraw}"],
                    "final_calibration": [f"target-cal-{redraw}"],
                    "test": query_ids,
                },
                "fit_label_access": {
                    "design_origin": "routereval:source_task",
                    "router_origin": "routereval:source_task",
                    "validation_origin": "routereval:source_task",
                    "target_fit_label_roles": [],
                    "target_label_roles": ["final_calibration", "test"],
                },
                "source_complete_pool": {},
                "target_complete_pool": {},
                "methods": method_records,
            }],
        })
    return {
        "schema_version": transfer.SCHEMA_VERSION,
        "design": "ISSUE_52_fit_and_calibration_transfer",
        "protocol": "routereval",
        "fit_mode": "source_fitted",
        "source_benchmark": "source_task",
        "target_benchmark": "target_task",
        "source_family": "source_family",
        "target_family": "target_family",
        "source_equals_target": False,
        "pool_size": 4,
        "pool_seed": paired.BASE_SEED,
        "model_ids": [f"model-{index}" for index in range(4)],
        "model_ids_sha256": "model-hash",
        "sampled_model_column_indices": [0, 1, 2, 3],
        "source_full_model_ids_sha256": "source-model-hash",
        "target_full_model_ids_sha256": "target-model-hash",
        "source_hashes": {"labels": "1" * 64, "embeddings": "2" * 64},
        "target_hashes": {"labels": "3" * 64, "embeddings": "4" * 64},
        "base_seed": paired.BASE_SEED,
        "knn_k": 40,
        "redraw_start": redraw,
        "redraw_end": redraw + 1,
        "redraw_range": {
            "start_inclusive": redraw,
            "end_exclusive": redraw + 1,
            "count": 1,
            "registered_total": transfer.REDRAW_COUNT,
        },
        "target_grid": target_grid,
        "calibration_modes": list(transfer.CALIBRATION_MODES),
        "method_identities": methods,
        "preserve_prefixes": True,
        "checkpoint_identities": {
            "flexrouter": [{"training_seed": seed} for seed in paired.TRAINING_SEEDS],
            "barp": [{"training_seed": seed} for seed in paired.TRAINING_SEEDS],
        },
        "cells": cells,
        "artifact_manifest": {
            "owning_host": "test",
            "output_path": f"shard-{redraw}.json",
        },
    }


def _write_shards(tmp_path: Path) -> list[Path]:
    paths = []
    for redraw in range(transfer.REDRAW_COUNT):
        path = tmp_path / f"shard-{redraw:02d}.json"
        paired.atomic_write_json(path, _shard(redraw))
        paths.append(path)
    return paths


def test_deterministic_merge_is_order_independent_and_contiguous(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path)
    forward = merger.load_and_merge(paths)
    reverse = merger.load_and_merge(list(reversed(paths)))

    assert forward == reverse
    assert paired.canonical_json_bytes(forward) == paired.canonical_json_bytes(reverse)
    assert forward["redraw_range"]["count"] == 20
    for cell in forward["cells"]:
        assert [row["redraw_id"] for row in cell["redraws"]] == list(range(20))

    with pytest.raises(ValueError, match="exactly 20"):
        merger.load_and_merge(paths[:-1])
    with pytest.raises(ValueError, match="contiguous"):
        merger.load_and_merge([*paths[:-1], paths[-2]])


def test_merge_refuses_any_immutable_identity_change(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path)
    changed = json.loads(paths[7].read_text(encoding="utf-8"))
    changed["source_benchmark"] = "different_source"
    changed_path = tmp_path / "changed.json"
    paired.atomic_write_json(changed_path, changed)
    inputs = [changed_path if index == 7 else path for index, path in enumerate(paths)]

    with pytest.raises(ValueError, match="immutable ISSUE-52 identity differs"):
        merger.load_and_merge(inputs)


def test_event_arithmetic_is_exact_and_corruption_is_refused(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path)
    merged = merger.load_and_merge(paths)
    result = analysis.analyze_payload(merged, repetitions=200, seed=19)
    first = next(
        row
        for row in result["rows"]
        if row["alpha"] == paired.ALPHA_GRID[0]
        and row["method"] == "fixed_tally_factorized"
    )

    assert first["source_calibrated_target_event_count"] == 10
    assert first["source_calibrated_target_trial_count"] == 80
    assert first["source_calibrated_target_risk"] == pytest.approx(0.125)
    assert first["target_recalibrated_target_event_count"] == 0
    assert first["target_recalibrated_target_trial_count"] == 80
    assert first["paired_risk_inflation_mean"] == pytest.approx(
        0.125 - paired.ALPHA_GRID[0]
    )
    assert first["paired_call_overhead_mean"] == pytest.approx(1.0)

    corrupt = copy.deepcopy(_shard(0))
    twin = corrupt["cells"][0]["redraws"][0]["methods"][
        "fixed_tally_factorized"
    ]["calibrations"]["source_calibrated"]
    twin["target_test_event_count"] += 1
    with pytest.raises(ValueError, match="target event arithmetic differs"):
        merger.validate_payload(corrupt, Path("corrupt.json"))


def test_intervals_are_reproducible_for_redraws_and_query_clusters() -> None:
    redraw_values = [float(index % 3) for index in range(20)]
    query_values = [
        (f"query-{index % 5}", float((index * 7) % 4)) for index in range(40)
    ]

    first_redraw = analysis.paired_redraw_interval(redraw_values, seed=71)
    second_redraw = analysis.paired_redraw_interval(redraw_values, seed=71)
    first_query = analysis.query_cluster_interval(query_values, seed=72)
    second_query = analysis.query_cluster_interval(query_values, seed=72)

    assert first_redraw == second_redraw
    assert first_query == second_query
    assert first_query[2] == 5


def test_calibration_twins_are_paired_and_o2xd2_frequencies_are_reported(
    tmp_path: Path,
) -> None:
    merged = merger.load_and_merge(_write_shards(tmp_path))
    result = analysis.analyze_payload(merged, repetitions=200, seed=23)
    chain = next(
        row
        for row in result["rows"]
        if row["alpha"] == paired.ALPHA_GRID[0]
        and row["method"] == paired.CHAIN_DISCOUNT_METHOD
    )
    frequencies = chain["o2xd2_hyperparameter_frequencies"]

    assert [item["count"] for item in frequencies] == [10, 10]
    assert sum(item["frequency"] for item in frequencies) == pytest.approx(1.0)
    assert chain["paired_call_overhead_redraw_count"] == 20
    assert chain["test_retuning_used"] is False
    assert result["test_retuning_used"] is False
    assert result["source_shard_hashes"] == merged["source_shards"]

    corrupt = copy.deepcopy(merged)
    target_twin = corrupt["cells"][0]["redraws"][0]["methods"][
        "fixed_tally_factorized"
    ]["calibrations"]["target_recalibrated"]
    target_twin["frozen_target_scores_sha256"] = "different-score-hash"
    with pytest.raises(analysis.TransferAnalysisError, match="score identity differs"):
        analysis.analyze_payload(corrupt, repetitions=200, seed=23)


def test_atomic_outputs_refuse_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "analysis.json"
    analysis.write_json({"ok": True}, destination)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        analysis.write_json({"ok": False}, destination)
