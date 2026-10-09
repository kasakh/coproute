"""Regression tests for repaired reporting and deterministic number checks."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures_src"
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import check_paper_numbers as checker  # noqa: E402
import plot_benchmark_level_sweep as benchmark_plot  # noqa: E402
import plot_calls_vs_risk as query_plot  # noqa: E402
import taxonomy_survival as taxonomy  # noqa: E402


def _fractions() -> dict[str, float]:
    return {
        "design": 0.30,
        "router": 0.20,
        "final_calibration": 0.20,
        "validation": 0.15,
        "test": 0.15,
    }


def old_query_record() -> dict:
    return {
        "pool_size": 100,
        "n_queries": 100,
        "redraws": 2,
        "alphas": [0.1],
        "partition_fractions": _fractions(),
        "summary": {
            "0.1": {
                "arms": {
                    "racer_marginal_G": {
                        "calls_mean": 12.0,
                        "risk_mean": 0.2,
                        "per_redraw_risk": [0.2, 0.2],
                    },
                    "racer_marginal_P": {
                        "calls_mean": 10.0,
                        "risk_mean": 0.1,
                        "per_redraw_risk": [1.0 / 15.0, 2.0 / 15.0],
                    },
                },
                "control_per_redraw_risk": [0.2, 0.2],
            }
        },
        "per_variant_tuned": {
            "0.1": {
                "wrapper_pairwise_count_tuned": {
                    "calls_test_mean": 6.0,
                    "risk_test_mean": 0.1,
                    "verdict": "OPTIMISTIC",
                },
                "wrapper_error_correlation_tuned": {
                    "calls_test_mean": 7.0,
                    "risk_test_mean": 0.1,
                    "verdict": "VALID",
                },
                "wrapper_gaussian_copula_tuned": {
                    "calls_test_mean": 8.0,
                    "risk_test_mean": 0.1,
                    "verdict": "VALID",
                },
            }
        },
    }


def new_query_record() -> dict:
    arm_names = [
        "racer_G",
        "racer_P",
        "wrapper_factorized_independence_tuned",
        "wrapper_query_local_empirical_coverage_tuned",
        "wrapper_local_order_factorized_tuned",
        "wrapper_fixed_tally_factorized_tuned",
        "wrapper_fixed_design_marginal_factorized_tuned",
        "wrapper_pairwise_heuristic_tuned",
        "wrapper_correlation_heuristic_tuned",
        "wrapper_regularized_correlation_tuned",
    ]
    arms = {}
    per_redraw = {}
    for index, name in enumerate(arm_names):
        risks = [0.2, 0.2] if name == "racer_G" else [1.0 / 15.0, 2.0 / 15.0]
        safe_mean = sum(risks) / len(risks)
        arms[name] = {
            "calls_mean": float(20 - index),
            "safe_action_risk_mean": safe_mean,
            "standard_all_miss_risk_mean": safe_mean + (1.0 / 30.0),
            "augmented_loss_mean": safe_mean + (2.0 / 30.0),
            "certification_rate": 0.5,
            "per_redraw_risk": risks,
            "per_redraw_calls": [float(20 - index), float(20 - index)],
        }
        per_redraw[name] = [
            {
                "calls_test": float(20 - index),
                "safe_action_risk": risks[0],
                "standard_all_miss_risk": risks[0] + (1.0 / 30.0),
                "augmented_loss": risks[0] + (2.0 / 30.0),
                "certified": True,
                "non_certification": False,
            },
            {
                "calls_test": float(20 - index),
                "safe_action_risk": risks[1],
                "standard_all_miss_risk": risks[1] + (1.0 / 30.0),
                "augmented_loss": risks[1] + (2.0 / 30.0),
                "certified": False,
                "non_certification": True,
            },
        ]
    return {
        "design": "B_prime_E2_frozen_selection",
        "pool_size": 100,
        "n_queries": 100,
        "redraws": 2,
        "alphas": [0.1],
        "partition_fractions": _fractions(),
        "summary": {"0.1": {"controls": {"racer_G": arms["racer_G"], "racer_P": arms["racer_P"]}, "arms": arms}},
        "per_redraw": {"0.1": per_redraw},
    }


def _by_method(rows: list[dict]) -> dict[str, dict]:
    return {row["method"]: row for row in rows}


def test_risk_classifier_is_strict_two_standard_errors() -> None:
    assert not query_plot.risk_above_target(0.12, 0.10, 0.01)
    assert query_plot.risk_above_target(0.1200001, 0.10, 0.01)
    with pytest.raises(ValueError, match="nonnegative"):
        query_plot.risk_above_target(0.1, 0.1, -0.1)


def test_old_schema_aliases_and_unavailable_semantics() -> None:
    rows = _by_method(query_plot.rows_for(old_query_record()))
    assert set(rows) == {
        "racer_g",
        "racer_p",
        "pairwise_heuristic",
        "correlation_heuristic",
        "regularized_correlation",
    }
    assert rows["racer_g"]["method_label"] == "RACER-G"
    assert rows["racer_p"]["method_label"] == "RACER-P"
    assert rows["racer_g"]["above_target_2se"] is True
    assert rows["pairwise_heuristic"]["reported_risk_name"] == "legacy_reported_risk"
    assert rows["pairwise_heuristic"]["reported_risk_se"] is None
    assert rows["pairwise_heuristic"]["risk_classification"] == "unavailable"
    assert rows["pairwise_heuristic"]["above_target_2se"] == ""
    assert rows["pairwise_heuristic"]["reported_event_count"] == 3
    assert rows["pairwise_heuristic"]["reported_trial_count"] == 30
    assert rows["pairwise_heuristic"]["safe_action_risk_mean"] is None
    assert rows["pairwise_heuristic"]["source_schema"] == "legacy"


def test_issue45_arm_names_have_canonical_public_labels() -> None:
    assert query_plot.canonical_method_name(
        "wrapper_fixed_design_marginal_factorized_tuned"
    ) == "fixed_design_marginal_factorized"
    assert query_plot.canonical_method_name(
        "wrapper_local_order_factorized_tuned"
    ) == "local_order_factorized"
    assert query_plot.METHOD_SPECS["fixed_design_marginal_factorized"]["label"] == (
        "Fixed design marginal + factorized sizing"
    )
    assert query_plot.METHOD_SPECS["local_order_factorized"]["label"] == (
        "Local order + factorized sizing"
    )


def test_repaired_schema_displays_all_required_arms_and_losses() -> None:
    rows = _by_method(query_plot.rows_for(new_query_record()))
    assert list(rows) == [
        "racer_g",
        "racer_p",
        "exact_independence",
        "local_empirical",
        "local_order_factorized",
        "fixed_tally_factorized",
        "fixed_design_marginal_factorized",
        "pairwise_heuristic",
        "correlation_heuristic",
        "regularized_correlation",
    ]
    exact = rows["exact_independence"]
    assert exact["reported_risk_name"] == "standard_all_miss_risk"
    assert exact["reported_risk_mean"] == pytest.approx(4.0 / 30.0)
    assert exact["reported_event_count"] == 4
    assert exact["safe_action_risk_mean"] == pytest.approx(0.1)
    assert exact["standard_all_miss_risk_mean"] == pytest.approx(4.0 / 30.0)
    assert exact["augmented_loss_mean"] == pytest.approx(5.0 / 30.0)
    assert exact["safe_action_event_count"] == 3
    assert exact["standard_all_miss_event_count"] == 4
    assert exact["augmented_loss_event_count"] == 5
    assert exact["certification_rate"] == pytest.approx(0.5)
    assert exact["non_certification_count"] == 1
    assert exact["source_schema"] == "repaired"


def test_figure_metadata_range_contains_every_racer_point(tmp_path: Path) -> None:
    rows = query_plot.rows_for(new_query_record())
    output = tmp_path / "calls"
    query_plot.render_figure(rows, output)
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    panel = metadata["panels"][0]
    racer_calls = [row["mean_calls"] for row in rows if row["method"].startswith("racer_")]
    assert sorted(panel["baseline_calls"]) == sorted(racer_calls)
    assert all(panel["axis_x_min"] <= value <= panel["axis_x_max"] for value in racer_calls)
    assert output.with_suffix(".pdf").exists()
    assert output.with_suffix(".png").exists()


def test_checker_passes_then_fails_on_value_flag_and_range_mismatches(tmp_path: Path) -> None:
    records = tmp_path / "records"
    records.mkdir()
    (records / "pool100.json").write_text(
        json.dumps(new_query_record()), encoding="utf-8"
    )
    rows = query_plot.rows_for(new_query_record())
    table = tmp_path / "table.csv"
    query_plot.write_table(rows, table)
    metadata_path = tmp_path / "figure.json"
    metadata_path.write_text(
        json.dumps(query_plot.figure_metadata(rows)), encoding="utf-8"
    )
    assert checker.run_checks(records, table, [metadata_path]) == []

    mutated = [dict(row) for row in rows]
    mutated[0]["mean_calls"] += 1.0
    mutated[0]["above_target_2se"] = False
    query_plot.write_table(mutated, table)
    metadata = query_plot.figure_metadata(rows)
    metadata["panels"][0]["axis_x_max"] = min(metadata["panels"][0]["baseline_calls"]) - 1.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    errors = checker.run_checks(records, table, [metadata_path])
    assert any("mean_calls mismatch" in error for error in errors)
    assert any("risk flag" in error or "above_target_2se mismatch" in error for error in errors)
    assert any("outside x range" in error for error in errors)


def test_benchmark_csv_normalizes_old_and_repaired_columns() -> None:
    old = benchmark_plot.normalize_csv_row(
        {
            "protocol": "routereval",
            "benchmark": "ifeval",
            "budget": "100",
            "alpha": "0.05",
            "method": "empirical_tally",
            "test_calls": "4.5",
            "test_risk": "0.04",
            "marginal_calls": "5.5",
            "marginal_risk": "0.05",
        }
    )
    repaired = benchmark_plot.normalize_csv_row(
        {
            "source_protocol": "routereval",
            "dataset": "ifeval",
            "design_budget": "100",
            "target_alpha": "0.05",
            "arm": "local_empirical_coverage",
            "mean_calls": "4.5",
            "standard_all_miss_risk_mean": "0.04",
            "baseline_mean_calls": "5.5",
            "baseline_reported_risk_mean": "0.05",
        }
    )
    assert old["design_budget"] == repaired["design_budget"] == "100"
    assert old["mean_calls"] == repaired["mean_calls"] == 4.5
    assert repaired["method"] == "local_empirical"


def _taxonomy_record(benchmark: str, *, sensitivity: bool) -> dict:
    seed_results = []
    for risk in (0.04, 0.06):
        seed_results.append(
            {
                "per_alpha": {
                    "0.05": {
                        "tuned": {"empirical_tally": {"test_risk": risk}}
                    }
                }
            }
        )
    return {
        "protocol": "routereval",
        "benchmark": benchmark,
        "aggregate": {
            "0.05": {
                "methods": {
                    "empirical_tally": {
                        "test_risk_mean": 0.05,
                        "dr_bp_4_pass": sensitivity,
                    }
                }
            }
        },
        "seed_results": seed_results,
    }


def test_taxonomy_uses_neutral_information_destruction_label_and_counts_benchmarks() -> None:
    records = {
        f"run{index}.json": _taxonomy_record(f"benchmark_{index}", sensitivity=index < 4)
        for index in range(6)
    }
    row = taxonomy.survival_rows(records)[0]
    assert row["risk_bar_pass_count"] == 1
    assert row["risk_bar_unknown_methods"] == "-"
    assert row["information_destruction_sensitivity_pass_count"] == 1
    assert row["information_destruction_sensitivity_defined_cells_min"] == 6
    assert row["information_destruction_sensitivity_defined_cells_max"] == 6
    assert all("artifact" not in column and "causal" not in column for column in row)


def test_checker_cli_exits_nonzero_on_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = tmp_path / "records"
    records.mkdir()
    (records / "pool100.json").write_text(json.dumps(new_query_record()), encoding="utf-8")
    rows = query_plot.rows_for(new_query_record())
    rows[0]["mean_calls"] += 2.0
    table = tmp_path / "table.csv"
    query_plot.write_table(rows, table)
    metadata = tmp_path / "figure.json"
    metadata.write_text(json.dumps(query_plot.figure_metadata(query_plot.rows_for(new_query_record()))), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_paper_numbers.py",
            "--records",
            str(records),
            "--table",
            str(table),
            "--figure-metadata",
            str(metadata),
        ],
    )
    assert checker.main() == 1
