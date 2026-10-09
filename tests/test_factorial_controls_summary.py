"""Synthetic coverage for ISSUE-45 factorial-control reporting."""
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

import summarize_factorial_controls as summary  # noqa: E402


WRAPPER_ARMS = [
    "factorized_independence",
    "pairwise_heuristic",
    "correlation_heuristic",
    "regularized_correlation",
    "query_local_empirical_coverage",
    "local_order_factorized",
    "fixed_tally_factorized",
    "fixed_design_marginal_factorized",
]

RAW_ARMS = [
    "racer_G",
    "racer_P",
    *[f"wrapper_{name}_tuned" for name in WRAPPER_ARMS],
]

EXPECTED_METHODS = {
    "racer_g",
    "racer_p",
    "exact_independence",
    "pairwise_heuristic",
    "correlation_heuristic",
    "regularized_correlation",
    "local_empirical",
    "local_order_factorized",
    "fixed_tally_factorized",
    "fixed_design_marginal_factorized",
}

CALLS = {
    "racer_G": (12.0, 13.0),
    "racer_P": (11.0, 12.0),
    "wrapper_factorized_independence_tuned": (8.0, 9.0),
    "wrapper_pairwise_heuristic_tuned": (7.5, 8.5),
    "wrapper_correlation_heuristic_tuned": (7.25, 8.25),
    "wrapper_regularized_correlation_tuned": (7.0, 8.0),
    "wrapper_query_local_empirical_coverage_tuned": (9.0, 10.0),
    "wrapper_local_order_factorized_tuned": (5.0, 6.0),
    "wrapper_fixed_tally_factorized_tuned": (7.0, 8.0),
    "wrapper_fixed_design_marginal_factorized_tuned": (6.0, 7.0),
}

RISKS = {
    "wrapper_factorized_independence_tuned": (0.04, 0.05),
    "wrapper_local_order_factorized_tuned": (0.04, 0.06),
    "wrapper_fixed_tally_factorized_tuned": (0.03, 0.05),
    "wrapper_fixed_design_marginal_factorized_tuned": (0.035, 0.055),
}


def _diagnostics(redraw: int) -> list[dict]:
    return [
        {
            "query_offset": 0,
            "distinct_finite_score_count": 4 + redraw,
            "first_zero_miss_depth": 3 + redraw,
            "plateau_depth": 4 + redraw,
            "tied_block_sizes": [2, 3 + redraw],
            "score_one_tail_size": 6 - redraw,
            "selected_depth": 5 + redraw,
            "full_pool_call_indicator": redraw == 1,
            "factorized_underflow_count": redraw,
            "factorized_saturation_count": redraw + 1,
        },
        {
            "query_offset": 1,
            "distinct_finite_score_count": 3 + redraw,
            "first_zero_miss_depth": None,
            "plateau_depth": 5,
            "tied_block_sizes": [],
            "score_one_tail_size": 4,
            "selected_depth": 4,
            "full_pool_call_indicator": False,
            "factorized_underflow_count": 0,
            "factorized_saturation_count": 0,
        },
    ]


def synthetic_record() -> dict:
    per_redraw = {}
    for raw_name in RAW_ARMS:
        records = []
        for redraw in range(2):
            risks = RISKS.get(raw_name, (0.045, 0.055))
            record = {
                "redraw": redraw,
                "selected_arm": raw_name,
                "certified": not (raw_name == "racer_P" and redraw == 1),
                "calls_test": CALLS[raw_name][redraw],
                "standard_all_miss_risk": risks[redraw],
            }
            if raw_name in {
                "wrapper_query_local_empirical_coverage_tuned",
                "wrapper_local_order_factorized_tuned",
            }:
                record["score_diagnostics"] = _diagnostics(redraw)
                record["score_diagnostics_scope"] = "all_evaluation_pool_queries"
            records.append(record)
        per_redraw[raw_name] = records

    metadata = {
        arm: {
            "ordering_source": f"ordering:{arm}",
            "sizing_source": f"sizing:{arm}",
        }
        for arm in WRAPPER_ARMS
    }
    return {
        "design": summary.DESIGN,
        "protocol": "routereval",
        "benchmark": "ifeval",
        "pool_size": 100,
        "primary_neighbor_k": 40,
        "alphas": [0.05],
        "racer_controls": ["racer_G", "racer_P"],
        "wrapper_arms": WRAPPER_ARMS,
        "wrapper_arm_metadata": metadata,
        "per_redraw": {"0.05": per_redraw},
    }


def _rows_by_method(payload: dict) -> dict[str, dict]:
    return {row["method"]: row for row in payload["rows"]}


def test_factorial_summary_preserves_every_declared_arm_and_pairs_redraws() -> None:
    payload = summary.summarize_records([("synthetic.json", synthetic_record())])
    rows = _rows_by_method(payload)

    assert payload["row_count"] == len(RAW_ARMS)
    assert set(rows) == EXPECTED_METHODS
    local = rows["local_order_factorized"]
    assert local["mean_calls"] == pytest.approx(5.5)
    assert local["standard_all_miss_risk_mean"] == pytest.approx(0.05)
    assert local["certification_rate"] == pytest.approx(1.0)
    assert local["paired_calls_count_vs_fixed_tally"] == 2
    assert local["calls_delta_mean_vs_fixed_tally"] == pytest.approx(-2.0)
    assert local["calls_delta_mean_vs_exact_independence"] == pytest.approx(-3.0)
    assert local["calls_delta_mean_vs_fixed_design_marginal"] == pytest.approx(-1.0)
    assert local["standard_all_miss_risk_delta_mean_vs_fixed_tally"] == pytest.approx(0.01)

    racer_p = rows["racer_p"]
    assert racer_p["redraw_count"] == 2
    assert racer_p["certified_redraw_count"] == 1
    assert racer_p["certification_rate"] == pytest.approx(0.5)
    assert racer_p["mean_calls"] == pytest.approx(11.0)


def test_factorial_summary_aggregates_score_saturation_diagnostics() -> None:
    rows = _rows_by_method(
        summary.summarize_records([("synthetic.json", synthetic_record())])
    )
    local = rows["local_order_factorized"]

    assert local["diagnostic_redraw_count"] == 2
    assert local["diagnostic_query_count"] == 4
    assert local["distinct_finite_score_count_mean"] == pytest.approx(4.0)
    assert local["first_zero_miss_depth_mean"] == pytest.approx(3.5)
    assert local["first_zero_miss_depth_defined_rate"] == pytest.approx(0.5)
    assert local["plateau_depth_mean"] == pytest.approx(4.75)
    assert local["largest_tied_block_mean"] == pytest.approx(2.25)
    assert local["largest_tied_block_max"] == 4
    assert local["score_one_tail_size_mean"] == pytest.approx(4.75)
    assert local["selected_depth_mean"] == pytest.approx(4.75)
    assert local["full_pool_call_rate"] == pytest.approx(0.25)
    assert local["factorized_underflow_count"] == 1
    assert local["factorized_saturation_count"] == 3

    fixed = rows["fixed_tally_factorized"]
    assert fixed["diagnostic_query_count"] == 0
    assert fixed["full_pool_call_rate"] is None


def test_declared_arm_omission_is_rejected_instead_of_silently_reported() -> None:
    record = copy.deepcopy(synthetic_record())
    del record["per_redraw"]["0.05"]["wrapper_local_order_factorized_tuned"]
    with pytest.raises(ValueError, match="arm mismatch.*local_order_factorized"):
        summary.summarize_records([("missing.json", record)])


def test_factorial_outputs_csv_json_pdf_and_png(tmp_path: Path) -> None:
    record_path = tmp_path / "e2_pool100.json"
    record_path.write_text(json.dumps(synthetic_record()), encoding="utf-8")
    payload = summary.summarize_records(summary.discover_records([tmp_path]))
    csv_path = tmp_path / "factorial.csv"
    json_path = tmp_path / "factorial.json"
    figure_prefix = tmp_path / "factorial"

    summary.write_csv(payload["rows"], csv_path)
    summary.write_json(payload, json_path)
    pdf = summary.render_figure(payload["rows"], figure_prefix)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    restored = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(csv_rows) == len(RAW_ARMS)
    assert {row["method"] for row in csv_rows} == EXPECTED_METHODS
    assert restored["row_count"] == len(RAW_ARMS)
    assert pdf.exists() and pdf.stat().st_size > 0
    assert figure_prefix.with_suffix(".png").exists()
    assert figure_prefix.with_suffix(".png").stat().st_size > 0
