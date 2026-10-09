"""Regenerate the benchmark-level taxonomy survival table.

The risk bar uses the same classifier as every other repaired report: a cell is
above target only when mean risk is greater than alpha plus two standard errors.
A method passes the risk bar when at least four of six unique RouterEval
benchmarks have at least one budget cell that is not above target. Budget cells
are never counted as distinct benchmarks.

The former shuffle bar is reported as an information-destruction sensitivity
bar. It summarizes a preregistered heuristic after design-label information is
destroyed. It is not labeled as a causal decomposition, a significance test, or
an artifact detector. The output includes its defined-cell denominator range.

Both legacy ``dr_bp_4_pass`` fields and repaired
``information_destruction_sensitivity_pass`` fields are accepted.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from plot_calls_vs_risk import risk_above_target


OUTPUT_COLUMNS = [
    "target_alpha",
    "method_count",
    "risk_bar_pass_count",
    "risk_bar_fail_methods",
    "risk_bar_unknown_methods",
    "information_destruction_sensitivity_pass_count",
    "information_destruction_sensitivity_fail_methods",
    "information_destruction_sensitivity_defined_cells_min",
    "information_destruction_sensitivity_defined_cells_max",
]


def _canonical_benchmark_method(name: str) -> str:
    aliases = {
        "local_empirical": "empirical_tally",
        "fixed_empirical_tally": "empirical_tally",
        "pairwise_heuristic": "pairwise_count",
        "correlation_heuristic": "error_correlation",
        "regularized_correlation": "gaussian_copula",
    }
    return aliases.get(str(name), str(name))


def _alpha_key(mapping: dict[str, Any], alpha: float) -> str | None:
    for key in mapping:
        try:
            if math.isclose(float(key), alpha, rel_tol=0.0, abs_tol=1e-12):
                return key
        except (TypeError, ValueError):
            continue
    return None


def _method_info(record: dict[str, Any], alpha: float, method: str) -> dict[str, Any] | None:
    aggregate = record.get("aggregate", {})
    key = _alpha_key(aggregate, alpha) if isinstance(aggregate, dict) else None
    if key is None or not isinstance(aggregate[key], dict):
        return None
    methods = aggregate[key].get("methods", {})
    if not isinstance(methods, dict):
        return None
    for raw_name, info in methods.items():
        if _canonical_benchmark_method(raw_name) == method and isinstance(info, dict):
            return info
    return None


def _risk_series(record: dict[str, Any], alpha: float, method: str) -> list[float]:
    values: list[float] = []
    for seed_record in record.get("seed_results", []):
        per_alpha = seed_record.get("per_alpha", {})
        key = _alpha_key(per_alpha, alpha) if isinstance(per_alpha, dict) else None
        if key is None or not isinstance(per_alpha[key], dict):
            continue
        tuned = per_alpha[key].get("tuned", {})
        if not isinstance(tuned, dict):
            continue
        for raw_name, result in tuned.items():
            if _canonical_benchmark_method(raw_name) != method or not isinstance(result, dict):
                continue
            value = result.get("test_risk", result.get("standard_all_miss_risk"))
            if value is not None:
                values.append(float(value))
    return values


def _risk_status(record: dict[str, Any], alpha: float, method: str) -> str:
    values = _risk_series(record, alpha, method)
    info = _method_info(record, alpha, method)
    if not values:
        if info is None:
            return "unavailable"
        explicit_mean = info.get("test_risk_mean", info.get("standard_all_miss_risk_mean"))
        explicit_se = info.get("test_risk_se", info.get("standard_all_miss_risk_se"))
        if explicit_mean is None or explicit_se is None:
            return "unavailable"
        mean = float(explicit_mean)
        standard_error = float(explicit_se)
    else:
        mean = sum(values) / len(values)
        if len(values) == 1:
            standard_error = 0.0
        else:
            mean_square_error = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            standard_error = math.sqrt(mean_square_error / len(values))
    return "above" if risk_above_target(mean, alpha, standard_error) else "within"


def _information_sensitivity_flag(info: dict[str, Any] | None) -> bool | None:
    if info is None:
        return None
    for name in (
        "information_destruction_sensitivity_pass",
        "information_destruction_pass",
        "dr_bp_4_pass",
    ):
        if name in info and info[name] is not None:
            return bool(info[name])
    return None


def load_run_records(directory: str | Path) -> dict[str, dict[str, Any]]:
    records = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(directory).glob("*.json"))
    }
    if not records:
        raise ValueError(f"no benchmark-level run records found under {directory}")
    return records


def survival_rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    routereval = {
        name: record
        for name, record in records.items()
        if record.get("protocol") == "routereval"
    }
    method_names = set()
    alphas = set()
    for record in records.values():
        aggregate = record.get("aggregate", {})
        if not isinstance(aggregate, dict):
            continue
        for alpha_key, block in aggregate.items():
            alphas.add(float(alpha_key))
            if isinstance(block, dict) and isinstance(block.get("methods"), dict):
                method_names.update(
                    _canonical_benchmark_method(name) for name in block["methods"]
                )
    methods = sorted(method_names)
    rows = []
    for alpha in sorted(alphas):
        risk_fail: list[str] = []
        risk_unknown: list[str] = []
        sensitivity_fail: list[str] = []
        denominators: list[int] = []
        for method in methods:
            benchmark_status: dict[str, list[str]] = collections.defaultdict(list)
            for record in routereval.values():
                benchmark_status[str(record["benchmark"])].append(
                    _risk_status(record, alpha, method)
                )
            benchmarks_within = sum(
                any(status == "within" for status in statuses)
                for statuses in benchmark_status.values()
            )
            benchmarks_known = sum(
                any(status != "unavailable" for status in statuses)
                for statuses in benchmark_status.values()
            )
            if benchmarks_known < 4:
                risk_unknown.append(method)
            elif benchmarks_within < 4:
                risk_fail.append(method)

            sensitivity_flags = []
            for record in records.values():
                flag = _information_sensitivity_flag(_method_info(record, alpha, method))
                if flag is not None:
                    sensitivity_flags.append(flag)
            denominators.append(len(sensitivity_flags))
            if not sensitivity_flags or sum(sensitivity_flags) * 2 < len(sensitivity_flags):
                sensitivity_fail.append(method)

        rows.append(
            {
                "target_alpha": alpha,
                "method_count": len(methods),
                "risk_bar_pass_count": len(methods) - len(risk_fail) - len(risk_unknown),
                "risk_bar_fail_methods": "|".join(risk_fail) or "-",
                "risk_bar_unknown_methods": "|".join(risk_unknown) or "-",
                "information_destruction_sensitivity_pass_count": (
                    len(methods) - len(sensitivity_fail)
                ),
                "information_destruction_sensitivity_fail_methods": (
                    "|".join(sensitivity_fail) or "-"
                ),
                "information_destruction_sensitivity_defined_cells_min": (
                    min(denominators) if denominators else 0
                ),
                "information_destruction_sensitivity_defined_cells_max": (
                    max(denominators) if denominators else 0
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", help="optional CSV output; stdout is used when omitted")
    arguments = parser.parse_args()

    rows = survival_rows(load_run_records(arguments.runs))
    if arguments.out:
        output_path = Path(arguments.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", newline="", encoding="utf-8")
    else:
        handle = sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if arguments.out:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
