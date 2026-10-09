"""Synthetic coverage for the ISSUE-47 paired main-suite analyzer."""
from __future__ import annotations

import csv
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures_src"
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import analyze_paired_main_suite as analyzer  # noqa: E402


MODEL_IDS = ["model-a", "model-b", "model-c"]
QUERY_IDS = [f"query-{index}" for index in range(14)]


def _splits(redraw: int) -> dict[str, list[str]]:
    if redraw == 0:
        return {
            "design": QUERY_IDS[0:2],
            "router": QUERY_IDS[2:4],
            "validation": QUERY_IDS[4:6],
            "final_calibration": QUERY_IDS[6:10],
            "test": QUERY_IDS[10:12],
        }
    return {
        "design": QUERY_IDS[2:4],
        "router": QUERY_IDS[0:2],
        "validation": QUERY_IDS[6:8],
        "final_calibration": QUERY_IDS[8:12],
        "test": QUERY_IDS[12:14],
    }


def _full_pool(redraw: int) -> dict:
    return {
        "model_ids": MODEL_IDS,
        "calibration_misses": 0,
        "calibration_count": 4,
        "test_misses": redraw,
        "test_count": 2,
    }


def _certified_method(
    redraw: int,
    *,
    calls: list[float],
    misses: list[int],
) -> dict:
    test_ids = _splits(redraw)["test"]
    return {
        "certified": True,
        "calibration_misses": 0,
        "calibration_count": 4,
        "corrected_calibration_bound": 0.2,
        "endpoint_calibration_misses": 0,
        "endpoint_calibration_count": 4,
        "endpoint": {"model_ids": MODEL_IDS, "reaches_full_pool": True},
        "test_calls": sum(calls) / len(calls),
        "standard_all_miss_risk": sum(misses) / len(misses),
        "test_query_ids": test_ids,
        "test_calls_by_query": calls,
        "test_miss_by_query": misses,
    }


def _base_record(design: str, run_id: str, method_name: str) -> dict:
    redraws = []
    for redraw in range(2):
        if method_name == "racer_G":
            method = _certified_method(
                redraw, calls=[2.0, 2.0], misses=[0, 1]
            )
        elif method_name == "exact_independence":
            method = _certified_method(
                redraw, calls=[1.5, 2.0], misses=[0, 1]
            )
        else:
            method = _certified_method(
                redraw, calls=[1.0, 2.0], misses=[0, 0]
            )
        redraws.append(
            {
                "redraw_id": redraw,
                "split_ids": _splits(redraw),
                "full_pool": _full_pool(redraw),
                "methods": {method_name: method},
            }
        )
    return {
        "schema_version": "paired_main_suite_v1",
        "design": design,
        "run_id": run_id,
        "protocol": "routereval",
        "benchmark": "ifeval",
        "pool_id": "pool-abc",
        "pool_size": len(MODEL_IDS),
        "query_ids": QUERY_IDS,
        "model_ids": MODEL_IDS,
        "cells": [{"alpha": 0.2, "redraws": redraws}],
    }


def _three_records() -> list[tuple[str, dict]]:
    return [
        ("e2-racer.json", _base_record("E2", "e2-racer", "racer_G")),
        (
            "e2-independence.json",
            _base_record("E2", "e2-independence", "exact_independence"),
        ),
        ("e3-selected.json", _base_record("E3", "e3-selected", "e3_selected")),
    ]


def test_shared_endpoint_certification_and_paired_uncertainty() -> None:
    payload = analyzer.analyze_records(
        _three_records(),
        bootstrap_repetitions=300,
        comparison_pairs=[("e3_selected", "racer_g")],
    )
    assert payload["schema_version"] == "paired_main_suite_analysis_v1"
    assert payload["bootstrap"]["seed"] == 20260904
    assert payload["test_retuning_used"] is False
    assert payload["row_count"] == 1

    row = payload["rows"][0]
    assert row["paired_certified_redraw_count"] == 2
    assert row["paired_mean_calls_difference"] == pytest.approx(-0.5)
    assert row["paired_calls_difference_se"] == pytest.approx(0.0)
    assert row["paired_calls_redraw_ci95_low"] == pytest.approx(-0.5)
    assert row["paired_calls_redraw_ci95_high"] == pytest.approx(-0.5)
    assert row["paired_mean_standard_risk_difference"] == pytest.approx(-0.5)
    assert row["paired_standard_risk_difference_se"] == pytest.approx(0.0)
    assert row["paired_calls_query_cluster_ci95_low"] is not None
    assert row["paired_calls_query_cluster_ci95_high"] is not None
    assert row["query_cluster_count"] == 4

    assert row["method_test_miss_numerator"] == 0
    assert row["method_test_miss_denominator"] == 4
    assert row["reference_test_miss_numerator"] == 2
    assert row["reference_test_miss_denominator"] == 4
    assert row["method_empirical_joint_event_j_hat"] == pytest.approx(0.0)
    assert row["reference_empirical_joint_event_j_hat"] == pytest.approx(0.5)
    assert row["method_certification_rate"] == pytest.approx(1.0)
    assert 0.0 < row["method_certification_wilson_ci95_low"] < 1.0
    assert row["method_certification_wilson_ci95_high"] == pytest.approx(1.0)

    assert row["method_full_pool_calibration_misses"] == 0
    assert row["method_full_pool_calibration_denominator"] == 8
    assert row["method_full_pool_test_misses"] == 1
    assert row["method_full_pool_test_denominator"] == 4
    assert row["shared_endpoint_redraw_count"] == 2
    assert row["shared_endpoint_certification_match_count"] == 2
    assert row["shared_feasibility_equivalent"] is True
    assert row["endpoint_difference_count"] == 0
    assert row["certification_disagreement_count"] == 0
    assert row["certification_disagreement_reasons"] == {}
    assert row["method_per_redraw"][0]["corrected_calibration_bound"] == pytest.approx(0.2)
    assert row["method_per_redraw"][0]["test_misses"] == 0
    assert set(row["source_hashes"]) == {
        "e2-racer.json",
        "e2-independence.json",
        "e3-selected.json",
    }


def test_shallow_endpoint_certification_disagreement_is_explicit() -> None:
    records = _three_records()
    shallow = records[2][1]
    for redraw in shallow["cells"][0]["redraws"]:
        method = redraw["methods"]["e3_selected"]
        method.clear()
        method.update(
            {
                "certified": False,
                "calibration_misses": 1,
                "calibration_count": 4,
                "crc_bound": 0.4,
                "endpoint_calibration_misses": 1,
                "endpoint_calibration_count": 4,
                "endpoint": {
                    "model_ids": MODEL_IDS[:2],
                    "reaches_full_pool": False,
                },
                "non_certification_reason": "eligibility_cap_reached",
            }
        )
    payload = analyzer.analyze_records(
        records,
        bootstrap_repetitions=100,
        comparison_pairs=[("e3_selected", "racer_g")],
    )
    row = payload["rows"][0]
    assert row["method_certification_numerator"] == 0
    assert row["method_certification_denominator"] == 2
    assert row["method_empirical_joint_event_j_hat"] == 0.0
    assert row["paired_certified_redraw_count"] == 0
    assert row["paired_mean_calls_difference"] is None
    assert row["paired_calls_query_cluster_ci95_low"] is None
    assert row["shared_endpoint_redraw_count"] == 0
    assert row["shared_feasibility_equivalent"] is False
    assert row["endpoint_difference_count"] == 2
    assert row["certification_disagreement_count"] == 2
    assert row["certification_disagreement_reasons"] == {
        "method_shallow_eligibility_cap": 2
    }
    assert all(
        detail["non_certification_reason"] == "eligibility_cap_reached"
        for detail in row["method_per_redraw"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("query_ids", "nonpaired query_ids"),
        ("model_ids", "nonpaired model_ids"),
        ("split_ids", "nonpaired split IDs"),
    ],
)
def test_mismatched_records_are_rejected(mutation: str, message: str) -> None:
    records = _three_records()
    second = records[1][1]
    if mutation == "query_ids":
        second["query_ids"] = list(reversed(second["query_ids"]))
    elif mutation == "model_ids":
        second["model_ids"] = list(reversed(second["model_ids"]))
    else:
        second["cells"][0]["redraws"][0]["split_ids"]["test"] = [
            QUERY_IDS[9],
            QUERY_IDS[11],
        ]
        second["cells"][0]["redraws"][0]["split_ids"]["final_calibration"] = [
            QUERY_IDS[6],
            QUERY_IDS[7],
            QUERY_IDS[8],
            QUERY_IDS[10],
        ]
    with pytest.raises(analyzer.PairedAnalysisError, match=message):
        analyzer.analyze_records(
            records,
            bootstrap_repetitions=20,
            comparison_pairs=[("e3_selected", "racer_g")],
        )


def test_shared_endpoint_inconsistent_certification_is_rejected() -> None:
    records = _three_records()
    method = records[2][1]["cells"][0]["redraws"][0]["methods"]["e3_selected"]
    method["certified"] = False
    with pytest.raises(
        analyzer.PairedAnalysisError,
        match="certification indicator disagrees with corrected endpoint feasibility",
    ):
        analyzer.analyze_records(
            records,
            bootstrap_repetitions=20,
            comparison_pairs=[("e3_selected", "racer_g")],
        )


def test_query_cluster_bootstrap_is_reproducible_and_joint_across_redraws() -> None:
    records = _three_records()
    first = analyzer.analyze_records(
        records,
        bootstrap_repetitions=250,
        seed=17,
        comparison_pairs=[("e3_selected", "racer_g")],
    )
    second = analyzer.analyze_records(
        copy.deepcopy(records),
        bootstrap_repetitions=250,
        seed=17,
        comparison_pairs=[("e3_selected", "racer_g")],
    )
    assert first == second
    row = first["rows"][0]
    assert row["query_interval_method"] == "query_id_cluster_percentile_bootstrap"
    assert row["split_pool_variability_label"].startswith("paired redraw")
    assert row["query_population_uncertainty_label"].startswith("query-ID")


def test_cli_writes_deterministic_csv_json_and_source_hashes(tmp_path: Path) -> None:
    paths = []
    for name, record in _three_records():
        path = tmp_path / name
        path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        paths.append(path)
    csv_path = tmp_path / "paired.csv"
    json_path = tmp_path / "paired.json"
    status = analyzer.main(
        [
            *[str(path) for path in paths],
            "--output-csv",
            str(csv_path),
            "--output-json",
            str(json_path),
            "--bootstrap-repetitions",
            "50",
        ]
    )
    assert status == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["row_count"] == 3
    assert payload["bootstrap"]["repetitions"] == 50
    assert payload["source_hashes"] == {
        str(path): analyzer.sha256_file(path) for path in paths
    }
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert len(rows) == payload["row_count"]
    assert rows[0]["test_retuning_used"] == "False"
    assert json.loads(rows[0]["source_hashes"])


def test_aggregate_values_must_match_exact_per_query_counts() -> None:
    records = _three_records()
    method = records[0][1]["cells"][0]["redraws"][0]["methods"]["racer_G"]
    method["standard_all_miss_risk"] = 0.25
    with pytest.raises(analyzer.PairedAnalysisError, match="standard risk disagrees"):
        analyzer.analyze_records(
            records,
            bootstrap_repetitions=20,
            comparison_pairs=[("e3_selected", "racer_g")],
        )
