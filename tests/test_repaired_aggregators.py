"""Synthetic tests for the repaired E1 and E3 run aggregators."""
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

import summarize_benchmark_runs as e1_summary  # noqa: E402
import summarize_e3 as e3_summary  # noqa: E402


def _e1_method(
    calls: float | None,
    risk: float | None,
    *,
    certified: bool,
    hyper: float,
    fallback: bool,
) -> dict:
    return {
        "arm": "gaussian_copula",
        "hyper_name": "correlation_ridge",
        "chosen_hyper": hyper,
        "test_calls": calls,
        "standard_all_miss_risk": risk,
        "certified": certified,
        "validation_fallback": fallback,
    }


def synthetic_e1_record() -> dict:
    return {
        "design": "B_prime_per_method_tuning",
        "protocol": "routereval",
        "benchmark": "ifeval",
        "pool_size": 100,
        "design_budget": 50,
        "alphas": [0.05],
        "seeds": [11, 12],
        "seed_results": [
            {
                "seed": 11,
                "per_alpha": {
                    "0.05": {
                        "marginal": {
                            "test_calls": 6.0,
                            "standard_all_miss_risk": 0.04,
                            "certified": True,
                        },
                        "tuned": {
                            "gaussian_copula": _e1_method(
                                3.0, 0.04, certified=True, hyper=0.1, fallback=False
                            )
                        },
                        "null_shuffled": {
                            "gaussian_copula": _e1_method(
                                4.0, 0.05, certified=True, hyper=0.1, fallback=False
                            )
                        },
                    }
                },
            },
            {
                "seed": 12,
                "per_alpha": {
                    "0.05": {
                        "marginal": {
                            "test_calls": 8.0,
                            "standard_all_miss_risk": 0.06,
                            "certified": True,
                        },
                        "tuned": {
                            "gaussian_copula": _e1_method(
                                5.0, 0.06, certified=True, hyper=0.2, fallback=True
                            )
                        },
                        "null_shuffled": {
                            "gaussian_copula": _e1_method(
                                None, None, certified=False, hyper=0.2, fallback=True
                            )
                        },
                    }
                },
            },
        ],
        "aggregate": {
            "0.05": {
                "marginal_test_calls_mean": 7.0,
                "marginal_standard_all_miss_risk_mean": 0.05,
                "methods": {
                    "gaussian_copula": {
                        "hyper_name": "correlation_ridge",
                        "test_calls_mean": 4.0,
                        "standard_all_miss_risk_mean": 0.05,
                    }
                },
            }
        },
    }


def _e3_policy(
    policy_class: str,
    variant: str,
    calls: float | None,
    risk: float | None,
    certified: bool,
) -> dict:
    return {
        "policy_class": policy_class,
        "variant": variant,
        "test_calls": calls,
        "standard_all_miss_risk": risk,
        "final_certified": certified,
        "non_certification": not certified,
    }


def synthetic_e3_record(*, include_exact: bool = True) -> dict:
    redraws = []
    selected_specs = [
        ("fixed_prefix", "empirical_tally", 2.0, 0.04),
        ("per_query_wrapper", "regularized_correlation", 4.0, 0.06),
    ]
    for index, (policy_class, variant, calls, risk) in enumerate(selected_specs):
        controls = {
            "racer_G": _e3_policy("racer_control", "racer_G", 5.0 + 2 * index, 0.03 + 0.02 * index, True),
            "racer_P": _e3_policy(
                "racer_control",
                "racer_P",
                6.0 if index == 0 else None,
                0.04 if index == 0 else None,
                index == 0,
            ),
        }
        entry = {
            "redraw": index,
            "selected": _e3_policy(policy_class, variant, calls, risk, True),
            "fallback_used": index == 1,
            "racer_controls": controls,
        }
        if include_exact:
            entry["controls"] = {
                "exact_independence": _e3_policy(
                    "control", "exact_independence", 3.0 + 2 * index, 0.04, True
                )
            }
        redraws.append(entry)
    return {
        "design": "E3_unified_frozen_selection",
        "protocol": "routereval",
        "benchmark": "ifeval",
        "pool_size": 100,
        "alphas": [0.05],
        "redraws": 2,
        "class_penalty_enabled": True,
        "class_penalty_margin": 0.05,
        "per_alpha_records": {"0.05": redraws},
        "aggregate": {"0.05": {"n_redraws": 2}},
    }


def test_e1_summary_uses_feasible_seeds_and_neutral_shuffle_difference() -> None:
    rows = e1_summary.rows_for_record(synthetic_e1_record(), "synthetic_e1.json")
    assert len(rows) == 1
    row = rows[0]
    assert row["mean_calls"] == pytest.approx(4.0)
    assert row["mean_calls_se"] == pytest.approx(1.0)
    assert row["standard_all_miss_risk_mean"] == pytest.approx(0.05)
    assert row["standard_all_miss_risk_se"] == pytest.approx(0.01)
    assert row["certification_count"] == 2
    assert row["certification_rate"] == pytest.approx(1.0)
    assert row["baseline_mean_calls"] == pytest.approx(7.0)
    assert row["baseline_standard_all_miss_risk_mean"] == pytest.approx(0.05)
    assert json.loads(row["selected_hyperparameters"]) == [0.1, 0.2]
    assert row["validation_fallback_count"] == 1
    assert row["information_destruction_permutation_count"] == 2
    assert row["information_destruction_feasible_count"] == 1
    assert row["information_destruction_unavailable_count"] == 1
    assert row["information_destruction_shuffled_calls_mean"] == pytest.approx(4.0)
    assert row["information_destruction_shuffled_calls_se"] is None
    assert row["real_minus_shuffled_calls"] == pytest.approx(0.0)
    distribution = json.loads(row["information_destruction_permutation_distribution"])
    assert distribution[1]["calls"] is None
    assert distribution[1]["certified"] is False


def test_e1_preserves_unavailable_metrics_and_csv_blanks(tmp_path: Path) -> None:
    record = synthetic_e1_record()
    for seed in record["seed_results"]:
        method = seed["per_alpha"]["0.05"]["tuned"]["gaussian_copula"]
        method.update({"test_calls": None, "standard_all_miss_risk": None, "certified": False})
    row = e1_summary.rows_for_record(record)[0]
    assert row["mean_calls"] is None
    assert row["standard_all_miss_risk_mean"] is None
    assert row["certification_count"] == 0
    assert row["certification_rate"] == pytest.approx(0.0)
    output = tmp_path / "summary.csv"
    e1_summary.write_csv([row], output)
    written = next(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert written["mean_calls"] == ""
    assert written["standard_all_miss_risk_mean"] == ""


def test_e3_summary_reports_selected_controls_wins_and_penalty() -> None:
    row = e3_summary.rows_for_record(synthetic_e3_record(), "pool100_penalty.json")[0]
    assert row["selected_mean_calls"] == pytest.approx(3.0)
    assert row["selected_mean_calls_se"] == pytest.approx(1.0)
    assert row["selected_standard_all_miss_risk_mean"] == pytest.approx(0.05)
    assert row["selected_standard_all_miss_risk_se"] == pytest.approx(0.01)
    assert row["selected_certification_count"] == 2
    assert row["selected_certification_rate"] == pytest.approx(1.0)
    assert json.loads(row["class_win_counts"]) == {
        "fixed_prefix": 1,
        "per_query_wrapper": 1,
    }
    assert json.loads(row["variant_win_counts"]) == {
        "empirical_tally": 1,
        "regularized_correlation": 1,
    }
    assert row["validation_fallback_count"] == 1
    assert row["racer_g_mean_calls"] == pytest.approx(6.0)
    assert row["racer_g_mean_calls_se"] == pytest.approx(1.0)
    assert row["racer_p_mean_calls"] == pytest.approx(6.0)
    assert row["racer_p_certification_count"] == 1
    assert row["racer_p_certification_rate"] == pytest.approx(0.5)
    assert row["exact_independence_mean_calls"] == pytest.approx(4.0)
    assert row["exact_independence_standard_all_miss_risk_mean"] == pytest.approx(0.04)
    assert row["class_penalty_enabled"] is True
    assert row["penalty_setting"] == "enabled:0.05"


def test_e3_does_not_invent_unavailable_exact_independence() -> None:
    row = e3_summary.rows_for_record(synthetic_e3_record(include_exact=False))[0]
    assert row["exact_independence_mean_calls"] is None
    assert row["exact_independence_standard_all_miss_risk_mean"] is None
    assert row["exact_independence_certification_count"] is None
    assert row["exact_independence_certification_rate"] is None


def test_deterministic_json_and_record_discovery(tmp_path: Path) -> None:
    records = tmp_path / "records"
    records.mkdir()
    e1_path = records / "b_e1.json"
    e1_path.write_text(json.dumps(synthetic_e1_record()), encoding="utf-8")
    (records / "ignored.json").write_text(json.dumps({"design": "other"}), encoding="utf-8")
    loaded = e1_summary.discover_records([records])
    rows = e1_summary.summarize_records(loaded)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    e1_summary.write_json(rows, first)
    e1_summary.write_json(rows, second)
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["summary_kind"] == "repaired_e1_benchmark_runs"
    assert payload["row_count"] == 1

    e3_rows = e3_summary.rows_for_record(synthetic_e3_record())
    e3_first = tmp_path / "e3_first.json"
    e3_second = tmp_path / "e3_second.json"
    e3_summary.write_json(e3_rows, e3_first)
    e3_summary.write_json(e3_rows, e3_second)
    assert e3_first.read_bytes() == e3_second.read_bytes()
