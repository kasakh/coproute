"""Summarize repaired Design B prime benchmark-level run records.

The output has one row per protocol, benchmark, design budget, target level,
and dependence method. Statistics use only seeds with a certified operating
point. Missing source values remain missing in CSV and JSON output.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

DESIGN = "B_prime_per_method_tuning"

COLUMNS = [
    "protocol",
    "benchmark",
    "pool_size",
    "design_budget",
    "target_alpha",
    "method",
    "hyperparameter_name",
    "seed_count",
    "feasible_seed_count",
    "mean_calls",
    "mean_calls_se",
    "standard_all_miss_risk_mean",
    "standard_all_miss_risk_se",
    "certification_count",
    "certification_rate",
    "certification_status_available_count",
    "baseline_mean_calls",
    "baseline_standard_all_miss_risk_mean",
    "baseline_certification_count",
    "baseline_certification_rate",
    "selected_hyperparameters",
    "selected_hyperparameters_by_seed",
    "validation_fallback_count",
    "validation_fallback_rate",
    "validation_fallback_status_available_count",
    "information_destruction_permutation_distribution",
    "information_destruction_permutation_count",
    "information_destruction_feasible_count",
    "information_destruction_unavailable_count",
    "information_destruction_shuffled_calls_mean",
    "information_destruction_shuffled_calls_se",
    "real_minus_shuffled_calls",
    "source_record",
]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _number(value)) is not None]
    return statistics.fmean(numbers) if numbers else None


def _se(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _number(value)) is not None]
    if len(numbers) < 2:
        return None
    return statistics.stdev(numbers) / math.sqrt(len(numbers))


def _alpha_entry(mapping: Any, alpha: float) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    direct = mapping.get(str(alpha))
    if isinstance(direct, dict):
        return direct
    for key, value in mapping.items():
        try:
            matches = float(key) == alpha
        except (TypeError, ValueError):
            matches = False
        if matches and isinstance(value, dict):
            return value
    return {}


def _certification(record: Any) -> bool | None:
    if not isinstance(record, dict):
        return None
    for key in ("final_certified", "certified"):
        if key in record and record[key] is not None:
            return bool(record[key])
    if "infeasible" in record and record["infeasible"] is not None:
        return not bool(record["infeasible"])
    return None


def _fallback(record: Any) -> bool | None:
    if not isinstance(record, dict):
        return None
    for key in ("validation_fallback", "fallback"):
        if key in record and record[key] is not None:
            return bool(record[key])
    if "selection_constrained_by_val" in record:
        value = record["selection_constrained_by_val"]
        return None if value is None else not bool(value)
    if "validation_certified" in record:
        value = record["validation_certified"]
        return None if value is None else not bool(value)
    return None


def _calls(record: Any) -> float | None:
    if not isinstance(record, dict):
        return None
    for key in ("test_calls", "model_calls", "calls"):
        if key in record:
            return _number(record[key])
    return None


def _risk(record: Any) -> float | None:
    if not isinstance(record, dict):
        return None
    for key in ("standard_all_miss_risk", "test_risk"):
        if key in record:
            return _number(record[key])
    return None


def _feasible(record: Any) -> bool:
    certified = _certification(record)
    if certified is not None:
        return certified
    return _calls(record) is not None and _risk(record) is not None


def _method_record(container: Any, method: str) -> dict[str, Any] | None:
    if not isinstance(container, dict):
        return None
    for candidate in (
        container.get(method),
        container.get("tuned", {}).get(method) if isinstance(container.get("tuned"), dict) else None,
        container.get("methods", {}).get(method) if isinstance(container.get("methods"), dict) else None,
    ):
        if isinstance(candidate, dict):
            return candidate
    return None


def _permutation_records(cell: dict[str, Any], method: str) -> list[tuple[Any, dict[str, Any] | None]]:
    """Return every available information-destruction record for one seed."""

    results: list[tuple[Any, dict[str, Any] | None]] = []
    for key in (
        "information_destruction_permutations",
        "null_shuffled_permutations",
        "permutation_results",
    ):
        values = cell.get(key)
        if isinstance(values, list):
            for index, value in enumerate(values):
                identity = value.get("permutation", value.get("permutation_index", index)) if isinstance(value, dict) else index
                results.append((identity, _method_record(value, method)))
            return results

    shuffled = cell.get("null_shuffled")
    if isinstance(shuffled, list):
        for index, value in enumerate(shuffled):
            identity = value.get("permutation", value.get("permutation_index", index)) if isinstance(value, dict) else index
            results.append((identity, _method_record(value, method)))
    elif isinstance(shuffled, dict):
        if isinstance(shuffled.get("permutations"), list):
            for index, value in enumerate(shuffled["permutations"]):
                identity = value.get("permutation", value.get("permutation_index", index)) if isinstance(value, dict) else index
                results.append((identity, _method_record(value, method)))
        else:
            results.append((0, _method_record(shuffled, method)))
    return results


def _status_summary(values: list[bool | None]) -> tuple[int | None, float | None, int]:
    known = [value for value in values if value is not None]
    if not known:
        return None, None, 0
    count = sum(bool(value) for value in known)
    rate = count / len(known) if len(known) == len(values) else None
    return count, rate, len(known)


def _methods(record: dict[str, Any], alpha: float) -> list[str]:
    names: set[str] = set()
    aggregate = _alpha_entry(record.get("aggregate"), alpha)
    if isinstance(aggregate.get("methods"), dict):
        names.update(str(name) for name in aggregate["methods"])
    for seed in record.get("seed_results", []):
        cell = _alpha_entry(seed.get("per_alpha"), alpha)
        if isinstance(cell.get("tuned"), dict):
            names.update(str(name) for name in cell["tuned"])
    return sorted(names)


def rows_for_record(record: dict[str, Any], source_record: str = "") -> list[dict[str, Any]]:
    """Create canonical rows for one repaired Design B prime record."""

    if record.get("design") != DESIGN:
        raise ValueError(f"expected design {DESIGN!r}, found {record.get('design')!r}")
    required = ("protocol", "benchmark", "design_budget")
    missing = [key for key in required if record.get(key) is None]
    if missing:
        raise ValueError(f"record lacks identity fields: {', '.join(missing)}")

    alphas = sorted(float(alpha) for alpha in record.get("alphas", []))
    if not alphas and isinstance(record.get("aggregate"), dict):
        alphas = sorted(float(alpha) for alpha in record["aggregate"])
    seeds = sorted(record.get("seed_results", []), key=lambda item: (str(item.get("seed")),))
    rows: list[dict[str, Any]] = []

    for alpha in alphas:
        for method in _methods(record, alpha):
            method_records: list[dict[str, Any] | None] = []
            baseline_records: list[dict[str, Any] | None] = []
            seed_ids: list[Any] = []
            permutation_distribution: list[dict[str, Any]] = []

            for seed_index, seed in enumerate(seeds):
                seed_id = seed.get("seed", seed_index)
                seed_ids.append(seed_id)
                cell = _alpha_entry(seed.get("per_alpha"), alpha)
                method_record = _method_record(cell.get("tuned"), method)
                method_records.append(method_record)
                baseline_records.append(_method_record(cell, "marginal"))
                for permutation, shuffled_record in _permutation_records(cell, method):
                    permutation_distribution.append({
                        "seed": seed_id,
                        "permutation": permutation,
                        "calls": _calls(shuffled_record),
                        "standard_all_miss_risk": _risk(shuffled_record),
                        "certified": _certification(shuffled_record),
                    })

            aggregate = _alpha_entry(record.get("aggregate"), alpha)
            aggregate_method = _method_record(aggregate.get("methods"), method) or {}
            aggregate_baseline = aggregate
            feasible = [value for value in method_records if _feasible(value)]
            feasible_baseline = [value for value in baseline_records if _feasible(value)]
            calls = [_calls(value) for value in feasible]
            risks = [_risk(value) for value in feasible]
            baseline_calls = [_calls(value) for value in feasible_baseline]
            baseline_risks = [_risk(value) for value in feasible_baseline]

            if not seeds:
                calls = [aggregate_method.get("test_calls_mean")]
                risks = [aggregate_method.get("standard_all_miss_risk_mean", aggregate_method.get("test_risk_mean"))]
                baseline_calls = [aggregate_baseline.get("marginal_test_calls_mean")]
                baseline_risks = [aggregate_baseline.get("marginal_standard_all_miss_risk_mean", aggregate_baseline.get("marginal_test_risk_mean"))]

            certifications = [_certification(value) for value in method_records]
            certification_count, certification_rate, certification_available = _status_summary(certifications)
            baseline_certifications = [_certification(value) for value in baseline_records]
            baseline_cert_count, baseline_cert_rate, _ = _status_summary(baseline_certifications)
            fallbacks = [_fallback(value) for value in method_records]
            fallback_count, fallback_rate, fallback_available = _status_summary(fallbacks)

            selected_values = [
                value.get("chosen_hyper") if isinstance(value, dict) else None
                for value in method_records
            ]
            if not seeds and isinstance(aggregate_method.get("chosen_hyper_by_seed"), list):
                selected_values = list(aggregate_method["chosen_hyper_by_seed"])
                seed_ids = list(record.get("seeds", range(len(selected_values))))
            selected_by_seed = [
                {"seed": seed_id, "value": value}
                for seed_id, value in zip(seed_ids, selected_values)
            ]
            shuffled_calls = [
                item["calls"] for item in permutation_distribution
                if item["certified"] is True and item["calls"] is not None
            ]
            unavailable_permutations = sum(
                item["calls"] is None or item["certified"] is not True
                for item in permutation_distribution
            )
            real_mean = _mean(calls)
            shuffled_mean = _mean(shuffled_calls)
            row = {
                "protocol": record["protocol"],
                "benchmark": record["benchmark"],
                "pool_size": record.get("pool_size"),
                "design_budget": record["design_budget"],
                "target_alpha": alpha,
                "method": method,
                "hyperparameter_name": aggregate_method.get("hyper_name", next((value.get("hyper_name") for value in method_records if isinstance(value, dict) and value.get("hyper_name") is not None), None)),
                "seed_count": len(seeds) if seeds else (len(record.get("seeds", [])) or None),
                "feasible_seed_count": len(feasible) if seeds else aggregate_method.get("n_seeds_feasible"),
                "mean_calls": real_mean,
                "mean_calls_se": _se(calls) if seeds else None,
                "standard_all_miss_risk_mean": _mean(risks),
                "standard_all_miss_risk_se": _se(risks) if seeds else None,
                "certification_count": certification_count,
                "certification_rate": certification_rate,
                "certification_status_available_count": certification_available,
                "baseline_mean_calls": _mean(baseline_calls),
                "baseline_standard_all_miss_risk_mean": _mean(baseline_risks),
                "baseline_certification_count": baseline_cert_count,
                "baseline_certification_rate": baseline_cert_rate,
                "selected_hyperparameters": _json(selected_values),
                "selected_hyperparameters_by_seed": _json(selected_by_seed),
                "validation_fallback_count": fallback_count,
                "validation_fallback_rate": fallback_rate,
                "validation_fallback_status_available_count": fallback_available,
                "information_destruction_permutation_distribution": _json(permutation_distribution),
                "information_destruction_permutation_count": len(permutation_distribution),
                "information_destruction_feasible_count": len(shuffled_calls),
                "information_destruction_unavailable_count": unavailable_permutations,
                "information_destruction_shuffled_calls_mean": shuffled_mean,
                "information_destruction_shuffled_calls_se": _se(shuffled_calls),
                "real_minus_shuffled_calls": (real_mean - shuffled_mean if real_mean is not None and shuffled_mean is not None else None),
                "source_record": source_record,
            }
            rows.append(row)
    return rows


def discover_records(inputs: Iterable[str | Path]) -> list[tuple[str, dict[str, Any]]]:
    """Load matching records from files or directories in stable path order."""

    paths: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.update(path.rglob("*.json"))
        elif path.is_file():
            paths.add(path)
        else:
            raise FileNotFoundError(path)
    records = []
    for path in sorted(paths, key=lambda value: str(value)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("design") == DESIGN:
            records.append((str(path), data))
    if not records:
        raise ValueError(f"no {DESIGN} records found")
    return records


def summarize_records(records: Iterable[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    identities: set[tuple[Any, ...]] = set()
    for source, record in records:
        for row in rows_for_record(record, source):
            identity = tuple(row[key] for key in ("protocol", "benchmark", "design_budget", "target_alpha", "method"))
            if identity in identities:
                raise ValueError(f"duplicate canonical E1 row {identity}")
            identities.add(identity)
            rows.append(row)
    return sorted(rows, key=lambda row: (str(row["protocol"]), str(row["benchmark"]), str(row["design_budget"]), float(row["target_alpha"]), str(row["method"])))


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "summary_kind": "repaired_e1_benchmark_runs", "row_count": len(rows), "rows": rows}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="JSON files or directories")
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--json-out", required=True)
    arguments = parser.parse_args()
    rows = summarize_records(discover_records(arguments.runs))
    write_csv(rows, arguments.csv_out)
    write_json(rows, arguments.json_out)
    print(f"wrote {len(rows)} rows to {arguments.csv_out} and {arguments.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
