"""Synthetic tests for learned-baseline common-metric reporting."""
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

import compare_learned_baselines as report  # noqa: E402


def _learned_row(block: str, method: str, task: str, **values) -> dict:
    row = {
        "block": block,
        "run_status": "available",
        "analysis_label": "primary",
        "method": method,
        "target_task": task,
        "pool_size": 100,
        "pool_seed": 7001,
        "depth": "",
        "tau": "",
        "alpha": "",
        "training_seed_count": 5,
        "redraw_count": 40,
        "observation_count": 200,
        "success_mean": "",
        "success_se": "",
        "standard_all_miss_risk_mean": "",
        "standard_all_miss_risk_se": "",
        "mean_calls": "",
        "mean_calls_se": "",
        "median_calls_mean": "",
        "p90_calls_mean": "",
        "certification_rate": "",
        "certification_count": "",
        "non_certification_count": "",
    }
    row.update(values)
    return row


def learned_rows() -> list[dict]:
    rows = []
    for task, offset in (("ifeval", 0.0), ("musr", 0.1)):
        for method, bonus in (("flexrouter", 0.04), ("barp", 0.02)):
            for depth in (1, 2):
                success = 0.5 + offset + bonus + 0.1 * depth
                rows.append(
                    _learned_row(
                        "native_fixed_depth",
                        method,
                        task,
                        depth=depth,
                        success_mean=success,
                        success_se=0.01,
                        standard_all_miss_risk_mean=1.0 - success,
                        standard_all_miss_risk_se=0.01,
                        mean_calls=float(depth),
                    )
                )
        for alpha in (0.05, 0.10):
            rows.append(
                _learned_row(
                    "adapted_calibrated",
                    "flexrouter",
                    task,
                    alpha=alpha,
                    standard_all_miss_risk_mean=alpha,
                    standard_all_miss_risk_se=0.005,
                    mean_calls=4.0 if alpha == 0.05 else 2.0,
                    mean_calls_se=0.2,
                    certification_rate=0.95,
                    certification_count=190,
                    non_certification_count=10,
                )
            )
            rows.append(
                _learned_row(
                    "adapted_calibrated",
                    "barp",
                    task,
                    alpha=alpha,
                    standard_all_miss_risk_mean=alpha + 0.005,
                    standard_all_miss_risk_se=0.005,
                    mean_calls=5.0 if alpha == 0.05 else 2.5,
                    mean_calls_se=0.25,
                    certification_rate=1.0,
                    certification_count=200,
                    non_certification_count=0,
                )
            )
        for tau, calls, risk in ((0.0, 10.0, 0.02), (0.1, 4.0, 0.06)):
            rows.append(
                _learned_row(
                    "native_adaptive",
                    "flexrouter",
                    task,
                    tau=tau,
                    success_mean=1.0 - risk,
                    success_se=0.01,
                    standard_all_miss_risk_mean=risk,
                    standard_all_miss_risk_se=0.005,
                    mean_calls=calls,
                    mean_calls_se=0.2,
                    median_calls_mean=calls,
                    p90_calls_mean=calls + 1.0,
                )
            )
    return rows


def query_rows() -> list[dict]:
    rows = []
    for alpha in (0.05, 0.10):
        for method, calls, risk, certification in (
            ("fixed_tally_factorized", 8.0 if alpha == 0.05 else 4.0, alpha - 0.005, 1.0),
            ("exact_independence", 9.0 if alpha == 0.05 else 4.5, alpha, 0.95),
            ("racer_g", 10.0 if alpha == 0.05 else 5.0, alpha + 0.001, 0.95),
        ):
            redraws = 20
            rows.append(
                {
                    "pool_size": 100,
                    "target_alpha": alpha,
                    "method": method,
                    "method_label": method,
                    "mean_calls": calls,
                    "reported_risk_name": "standard_all_miss_risk",
                    "reported_risk_mean": risk,
                    "reported_risk_se": 0.005,
                    "risk_classification": "within_2se",
                    "above_target_2se": False,
                    "standard_all_miss_risk_mean": risk,
                    "certification_rate": certification,
                    "non_certification_count": round(redraws * (1.0 - certification)),
                    "redraw_count": redraws,
                    "source_schema": "repaired",
                }
            )
    return rows


def e1_rows() -> list[dict]:
    rows = []
    for task, risk_offset in (("ifeval", 0.0), ("musr", 0.1)):
        for phase in ("design", "test"):
            for depth, risk in ((1, 0.4 + risk_offset), (2, 0.2 + risk_offset)):
                rows.append(
                    {
                        "source": "RE",
                        "dataset": task,
                        "method": "empirical_tally",
                        "phase": phase,
                        "depth": depth,
                        "mean_miss_rate": risk,
                        "sd_miss_rate": 0.02,
                    }
                )
    return rows


def e2_rows() -> list[dict]:
    return [
        {
            "source": "RE",
            "dataset": task,
            "method": method,
            "phase": "test",
            "depth": depth,
            "mean_miss_rate": risk,
            "sd_miss_rate": 0.01,
        }
        for task in ("ifeval", "musr")
        for method in ("marginal", "pairwise_count")
        for depth, risk in ((1, 0.4), (2, 0.2))
    ]


def barp_rows() -> list[dict]:
    rows = []
    for task in ("mbpp", "hellaswag", "macro_average"):
        for w_cost, quality, cost in ((0.0, 0.8, 0.003), (1.0, 0.4, 0.001)):
            rows.append(
                {
                    "block": "native_barp_sensitivity",
                    "run_status": "available",
                    "analysis_label": "sensitivity_without_degenerate_gsm8k",
                    "training_tasks": '["mmlu","arc-challenge","winogrande"]',
                    "method": "barp",
                    "target_task": task,
                    "w_quality": 1.0 - w_cost,
                    "w_cost": w_cost,
                    "training_seed_count": 5,
                    "observation_count": 5,
                    "mean_official_quality": quality,
                    "mean_official_quality_se": 0.01,
                    "mean_cost": cost,
                    "mean_cost_se": 0.0001,
                    "mean_scalar_reward": quality - cost,
                    "binary_coverage": quality,
                    "nondominated_rate": 1.0,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_fixed_depth_uses_success_metric_and_marks_unavailable_curves() -> None:
    rows, exclusions, inventory = report.fixed_depth_rows(
        learned_rows(), e1_rows(), e2_rows()
    )
    learned = [row for row in rows if row["method"] == "flexrouter"]
    assert len(learned) == 4
    assert {row["depth_cap_label"] for row in learned} == {
        "registered maximum set size 10"
    }
    assert all(row["training_seed_count"] == 5 for row in learned)
    assert all(row["pairing_status"] == "paired_within_learned_protocol" for row in learned)

    fixed = [
        row
        for row in rows
        if row["method"] == "fixed_tally"
        and row["target_task"] == "ifeval"
        and row["depth"] == 1
    ]
    assert len(fixed) == 1
    assert fixed[0]["success_mean"] == pytest.approx(0.6)
    assert fixed[0]["pairing_status"] == "descriptive_nonpaired"
    assert fixed[0]["source_phase"] == "test"

    unavailable = [row for row in rows if row["availability_status"] == "unavailable"]
    assert {(row["method"], row["target_task"]) for row in unavailable} == {
        ("exact_independence", "ifeval"),
        ("racer_g", "ifeval"),
        ("exact_independence", "musr"),
        ("racer_g", "musr"),
    }
    assert all(row["success_mean"] is None for row in unavailable)
    assert inventory["exact_independence_fixed_depth"] == "unavailable"
    assert inventory["racer_g_fixed_depth"] == "unavailable"
    assert inventory["e2_requested_curve_contribution"].startswith("none")
    assert exclusions == []


def test_incomplete_learned_seed_aggregate_is_excluded_explicitly() -> None:
    learned = learned_rows()
    learned[0]["training_seed_count"] = 4
    rows, exclusions, _ = report.fixed_depth_rows(learned, e1_rows(), e2_rows())
    assert not any(
        row["method"] == learned[0]["method"]
        and row["target_task"] == learned[0]["target_task"]
        and row["depth"] == learned[0]["depth"]
        and row["source_protocol"].startswith(report.LEARNED_PROTOCOL)
        for row in rows
    )
    assert exclusions == [
        {
            "source": "learned_summary",
            "block": "native_fixed_depth",
            "method": "flexrouter",
            "target_task": "ifeval",
            "pool_size": 100,
            "training_seed_count": 4,
            "reason": "incomplete_training_seed_aggregate",
            "required_training_seed_count": 5,
        }
    ]


def test_calibrated_table_preserves_certification_and_nonpairing() -> None:
    rows, exclusions = report.calibrated_rows(
        learned_rows(), query_rows(), query_task="ifeval"
    )
    assert exclusions == []
    assert len(rows) == 14
    assert all(row["pairing_status"] == "descriptive_nonpaired" for row in rows)

    learned = next(
        row
        for row in rows
        if row["method"] == "flexrouter"
        and row["target_task"] == "ifeval"
        and row["target_alpha"] == 0.05
    )
    assert learned["certification_count"] == 190
    assert learned["certification_denominator"] == 200
    assert learned["certification_unit"] == "training_seed_redraw"
    assert learned["risk_classification"] == "within_2se"
    assert learned["above_target_2se"] is False

    fixed = next(
        row
        for row in rows
        if row["method"] == "fixed_tally_factorized"
        and row["target_alpha"] == 0.05
    )
    assert fixed["target_task"] == "ifeval"
    assert fixed["certification_count"] == 20
    assert fixed["certification_denominator"] == 20
    assert fixed["certification_unit"] == "redraw"
    assert fixed["pool_seed"] is None


def test_call_ratios_are_descriptive_and_missing_task_is_not_silently_joined() -> None:
    calibrated, _ = report.calibrated_rows(
        learned_rows(), query_rows(), query_task="ifeval"
    )
    ratios = report.learned_fixed_tally_ratios(calibrated, query_task="ifeval")
    assert len(ratios) == 8
    ifeval_flex = next(
        row
        for row in ratios
        if row["target_task"] == "ifeval"
        and row["learned_method"] == "flexrouter"
        and row["target_alpha"] == 0.05
    )
    assert ifeval_flex["availability_status"] == "available"
    assert ifeval_flex["pairing_status"] == "descriptive_nonpaired"
    assert ifeval_flex["learned_to_fixed_tally_call_ratio"] == pytest.approx(0.5)
    assert ifeval_flex["call_reduction_vs_fixed_tally"] == pytest.approx(0.5)

    musr = [row for row in ratios if row["target_task"] == "musr"]
    assert len(musr) == 4
    assert all(row["availability_status"] == "unavailable" for row in musr)
    assert all(row["fixed_tally_mean_calls"] is None for row in musr)
    assert all("No fixed-tally cell" in row["comparison_note"] for row in musr)


def test_all_variants_are_retained_without_test_selected_best() -> None:
    learned = learned_rows()
    variant = dict(
        next(
            row
            for row in learned
            if row["block"] == "adapted_calibrated"
            and row["method"] == "flexrouter"
            and row["target_task"] == "ifeval"
            and row["alpha"] == 0.05
        )
    )
    variant["analysis_label"] = "registered_sensitivity"
    variant["mean_calls"] = 1.0
    learned.append(variant)
    built = report.build_report(
        learned,
        query_rows(),
        e1_rows(),
        e2_rows(),
        query_task="ifeval",
    )
    labels = {
        row["analysis_label"]
        for row in built["blocks"]["calibrated_primary"]
        if row["method"] == "flexrouter"
        and row["target_task"] == "ifeval"
        and row["target_alpha"] == 0.05
    }
    assert labels == {"primary", "registered_sensitivity"}
    assert built["selection_policy"] == "all registered rows; no test-selected best variant"


def test_native_appendix_blocks_keep_complete_registered_grids() -> None:
    learned = learned_rows()
    incomplete = dict(
        next(row for row in learned if row["block"] == "native_adaptive")
    )
    incomplete["analysis_label"] = "incomplete"
    incomplete["training_seed_count"] = 3
    learned.append(incomplete)
    flex, flex_exclusions = report.native_flex_rows(learned)
    assert len(flex) == 4
    assert {row["tau"] for row in flex} == {0.0, 0.1}
    assert all(row["risk_certificate_claimed"] is False for row in flex)
    assert all(row["pairing_status"] == "not_applicable" for row in flex)
    assert len(flex_exclusions) == 1

    native_barp = barp_rows()
    incomplete_barp = dict(native_barp[0])
    incomplete_barp["training_seed_count"] = 4
    native_barp.append(incomplete_barp)
    barp, barp_exclusions = report.native_barp_rows(native_barp)
    assert len(barp) == 6
    assert {row["w_cost"] for row in barp} == {0.0, 1.0}
    assert all(row["analysis_label"] == "sensitivity_without_degenerate_gsm8k" for row in barp)
    assert len(barp_exclusions) == 1


def test_optional_native_barp_is_explicit_when_not_supplied() -> None:
    built = report.build_report(
        learned_rows(), query_rows(), e1_rows(), e2_rows(), query_task="ifeval"
    )
    assert built["blocks"]["native_barp_sensitivity_appendix"] == []
    assert built["optional_native_barp_status"] == "not_supplied_or_no_complete_rows"


def test_cli_writes_csv_json_pdf_png_and_source_hashes(tmp_path: Path) -> None:
    inputs = {
        "learned": tmp_path / "learned.csv",
        "query": tmp_path / "query.csv",
        "e1": tmp_path / "e1.csv",
        "e2": tmp_path / "e2.csv",
        "barp": tmp_path / "barp.csv",
    }
    _write_csv(inputs["learned"], learned_rows())
    _write_csv(inputs["query"], query_rows())
    _write_csv(inputs["e1"], e1_rows())
    _write_csv(inputs["e2"], e2_rows())
    _write_csv(inputs["barp"], barp_rows())
    output = tmp_path / "output"

    status = report.main(
        [
            "--learned-summary",
            str(inputs["learned"]),
            "--query-table",
            str(inputs["query"]),
            "--e1-fixed-depth",
            str(inputs["e1"]),
            "--e2-fixed-depth",
            str(inputs["e2"]),
            "--native-barp-summary",
            str(inputs["barp"]),
            "--query-task",
            "ifeval",
            "--output-dir",
            str(output),
        ]
    )
    assert status == 0
    expected = {
        "fixed_depth_success.csv",
        "fixed_depth_success.pdf",
        "fixed_depth_success.png",
        "calibrated_primary.csv",
        "calibrated_primary.pdf",
        "calibrated_primary.png",
        "learned_vs_fixed_tally_call_ratios.csv",
        "native_flex_adaptive.csv",
        "native_flex_adaptive.pdf",
        "native_flex_adaptive.png",
        "native_barp_sensitivity.csv",
        "native_barp_sensitivity.pdf",
        "native_barp_sensitivity.png",
        "common_metric_report.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert all((output / name).stat().st_size > 0 for name in expected)

    payload = json.loads((output / "common_metric_report.json").read_text())
    assert payload["schema_version"] == report.SCHEMA_VERSION
    assert payload["cross_protocol_comparison"] == "descriptive_nonpaired"
    assert payload["row_counts"] == {
        "fixed_depth_success": 16,
        "calibrated_primary": 14,
        "learned_vs_fixed_tally_call_ratios": 8,
        "native_flex_adaptive_appendix": 4,
        "native_barp_sensitivity_appendix": 6,
    }
    provenance_keys = {
        "learned": "learned_summary",
        "query": "query_table",
        "e1": "e1_fixed_depth",
        "e2": "e2_fixed_depth",
        "barp": "native_barp_summary",
    }
    for key, path in inputs.items():
        assert payload["inputs"][provenance_keys[key]]["sha256"] == report.sha256_file(path)

    csv_rows = list(
        csv.DictReader((output / "learned_vs_fixed_tally_call_ratios.csv").open())
    )
    assert {row["pairing_status"] for row in csv_rows} == {
        "descriptive_nonpaired",
        "unavailable",
    }

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        report.main(
            [
                "--learned-summary",
                str(inputs["learned"]),
                "--query-table",
                str(inputs["query"]),
                "--e1-fixed-depth",
                str(inputs["e1"]),
                "--e2-fixed-depth",
                str(inputs["e2"]),
                "--output-dir",
                str(output),
            ]
        )


def test_missing_required_schema_column_fails_closed() -> None:
    broken = query_rows()
    for row in broken:
        row.pop("standard_all_miss_risk_mean")
    with pytest.raises(report.CommonMetricError, match="standard_all_miss_risk_mean"):
        report.calibrated_rows(learned_rows(), broken, query_task="ifeval")
