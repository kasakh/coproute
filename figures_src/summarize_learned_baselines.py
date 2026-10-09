"""Aggregate immutable learned-baseline runs across training seeds.

The output keeps native fixed-depth, native adaptive, adapted calibrated, and
native BaRP results in separate blocks. RouterEval training seeds are averaged
within each target redraw before uncertainty is computed across redraws.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROTOCOL_NAME = "preregistered_learned_baselines_v1"
SUMMARY_SCHEMA_VERSION = 1

COLUMNS = (
    "block",
    "run_status",
    "analysis_label",
    "failure_stage",
    "failure_reason",
    "training_tasks",
    "mode",
    "method",
    "target_task",
    "pool_size",
    "pool_seed",
    "depth",
    "tau",
    "alpha",
    "w_quality",
    "w_cost",
    "training_seed_count",
    "redraw_count",
    "observation_count",
    "success_mean",
    "success_se",
    "success_auc_mean",
    "success_auc_se",
    "standard_all_miss_risk_mean",
    "standard_all_miss_risk_se",
    "mean_calls",
    "mean_calls_se",
    "median_calls_mean",
    "p90_calls_mean",
    "certification_rate",
    "certification_count",
    "non_certification_count",
    "mean_official_quality",
    "mean_official_quality_se",
    "mean_cost",
    "mean_cost_se",
    "mean_scalar_reward",
    "binary_coverage",
    "nondominated_rate",
    "source_runs",
)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    return statistics.fmean(numbers) if numbers else None


def _se(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    if len(numbers) < 2:
        return None
    return statistics.stdev(numbers) / math.sqrt(len(numbers))


def discover_results(inputs: Iterable[str | Path]) -> list[tuple[str, dict[str, Any]]]:
    """Find legacy and shared-checkpoint learned-baseline result files."""

    paths: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            direct = [candidate for candidate in (path / "results.json", *sorted(path.glob("results_*.json"))) if candidate.is_file()]
            if direct:
                paths.update(direct)
            else:
                paths.update(candidate for candidate in path.rglob("results.json") if candidate.is_file())
                paths.update(candidate for candidate in path.rglob("results_*.json") if candidate.is_file())
        else:
            raise FileNotFoundError(path)
    records = []
    for path in sorted(paths, key=lambda value: str(value)):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("protocol") != PROTOCOL_NAME or value.get("mode") not in (
            "routereval",
            "routerbench_native_barp",
        ):
            continue
        records.append((str(path), value))
    if not records:
        raise ValueError("no preregistered learned-baseline results found")
    return records

def _identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    identity = record["identity"]
    return (
        record["mode"],
        identity.get("method"),
        identity.get("variant"),
        identity.get("target_task"),
        identity.get("pool_size"),
        identity.get("pool_seed"),
    )


def _base_row(key: tuple[Any, ...], block: str) -> dict[str, Any]:
    mode, method, variant, target_task, pool_size, pool_seed = key
    return {
        column: None for column in COLUMNS
    } | {
        "block": block,
        "run_status": "available",
        "analysis_label": variant,
        "mode": mode,
        "method": method,
        "target_task": target_task,
        "pool_size": pool_size,
        "pool_seed": pool_seed,
    }


def _source_runs(records: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        sorted({str(record["source"]) for record in records}),
        sort_keys=True,
        separators=(",", ":"),
    )


def _seed_redraw_reduce(
    observations: Sequence[dict[str, Any]], value_keys: Sequence[str]
) -> list[dict[str, Any]]:
    """Average seeds inside each redraw, retaining certification fractions."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[int(observation["redraw"])].append(observation)
    reduced = []
    for redraw, values in sorted(grouped.items()):
        row = {"redraw": redraw, "seed_count": len(values)}
        for key in value_keys:
            row[key] = _mean(value.get(key) for value in values)
        row["certification_rate"] = _mean(
            1.0 if value.get("certified") is True else 0.0 for value in values
        )
        row["certification_count"] = sum(value.get("certified") is True for value in values)
        row["non_certification_count"] = sum(value.get("certified") is False for value in values)
        reduced.append(row)
    return reduced


def _fixed_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    auc_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        record = item["record"]
        if record["mode"] != "routereval":
            continue
        key = _identity(record)
        seed = record["identity"]["training_seed"]
        for redraw in record["per_redraw"]:
            fixed = redraw["native_fixed_depth"]
            auc_grouped[key].append(
                {
                    "redraw": redraw["redraw"],
                    "training_seed": seed,
                    "success_auc": fixed["success_auc"],
                    "source": item["source"],
                }
            )
            for depth in fixed["depths"]:
                grouped[key + (depth["depth"],)].append(
                    {
                        "redraw": redraw["redraw"],
                        "training_seed": seed,
                        "success": depth["success"],
                        "risk": depth["zero_correct_rate"],
                        "calls": depth["mean_calls"],
                        "source": item["source"],
                    }
                )
    rows = []
    for composite, observations in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        key, depth = composite[:-1], composite[-1]
        reduced = _seed_redraw_reduce(observations, ("success", "risk", "calls"))
        reduced_auc = _seed_redraw_reduce(auc_grouped[key], ("success_auc",))
        row = _base_row(key, "native_fixed_depth")
        row.update(
            {
                "depth": depth,
                "training_seed_count": len({value["training_seed"] for value in observations}),
                "redraw_count": len(reduced),
                "observation_count": len(observations),
                "success_mean": _mean(value["success"] for value in reduced),
                "success_se": _se(value["success"] for value in reduced),
                "success_auc_mean": _mean(value["success_auc"] for value in reduced_auc),
                "success_auc_se": _se(value["success_auc"] for value in reduced_auc),
                "standard_all_miss_risk_mean": _mean(value["risk"] for value in reduced),
                "standard_all_miss_risk_se": _se(value["risk"] for value in reduced),
                "mean_calls": _mean(value["calls"] for value in reduced),
                "mean_calls_se": _se(value["calls"] for value in reduced),
                "source_runs": _source_runs(observations),
            }
        )
        rows.append(row)
    return rows


def _adaptive_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        record = item["record"]
        if record["mode"] != "routereval":
            continue
        key = _identity(record)
        seed = record["identity"]["training_seed"]
        for redraw in record["per_redraw"]:
            for point in redraw["native_adaptive"]:
                grouped[key + (point["tau"],)].append(
                    {
                        "redraw": redraw["redraw"],
                        "training_seed": seed,
                        "success": point["success"],
                        "risk": point["standard_all_miss_risk"],
                        "calls": point["mean_calls"],
                        "median_calls": point["median_calls"],
                        "p90_calls": point["p90_calls"],
                        "source": item["source"],
                    }
                )
    rows = []
    for composite, observations in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        key, tau = composite[:-1], composite[-1]
        reduced = _seed_redraw_reduce(
            observations, ("success", "risk", "calls", "median_calls", "p90_calls")
        )
        row = _base_row(key, "native_adaptive")
        row.update(
            {
                "tau": tau,
                "training_seed_count": len({value["training_seed"] for value in observations}),
                "redraw_count": len(reduced),
                "observation_count": len(observations),
                "success_mean": _mean(value["success"] for value in reduced),
                "success_se": _se(value["success"] for value in reduced),
                "standard_all_miss_risk_mean": _mean(value["risk"] for value in reduced),
                "standard_all_miss_risk_se": _se(value["risk"] for value in reduced),
                "mean_calls": _mean(value["calls"] for value in reduced),
                "mean_calls_se": _se(value["calls"] for value in reduced),
                "median_calls_mean": _mean(value["median_calls"] for value in reduced),
                "p90_calls_mean": _mean(value["p90_calls"] for value in reduced),
                "source_runs": _source_runs(observations),
            }
        )
        rows.append(row)
    return rows


def _calibrated_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        record = item["record"]
        if record["mode"] != "routereval":
            continue
        key = _identity(record)
        seed = record["identity"]["training_seed"]
        for redraw in record["per_redraw"]:
            for point in redraw["adapted_calibrated"]:
                grouped[key + (point["alpha"],)].append(
                    {
                        "redraw": redraw["redraw"],
                        "training_seed": seed,
                        "certified": point["certified"],
                        "risk": point["standard_all_miss_risk"],
                        "calls": point["mean_calls"],
                        "median_calls": point["median_calls"],
                        "p90_calls": point["p90_calls"],
                        "source": item["source"],
                    }
                )
    rows = []
    for composite, observations in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        key, alpha = composite[:-1], composite[-1]
        reduced = _seed_redraw_reduce(
            observations, ("risk", "calls", "median_calls", "p90_calls")
        )
        row = _base_row(key, "adapted_calibrated")
        row.update(
            {
                "alpha": alpha,
                "training_seed_count": len({value["training_seed"] for value in observations}),
                "redraw_count": len(reduced),
                "observation_count": len(observations),
                "standard_all_miss_risk_mean": _mean(value["risk"] for value in reduced),
                "standard_all_miss_risk_se": _se(value["risk"] for value in reduced),
                "mean_calls": _mean(value["calls"] for value in reduced),
                "mean_calls_se": _se(value["calls"] for value in reduced),
                "median_calls_mean": _mean(value["median_calls"] for value in reduced),
                "p90_calls_mean": _mean(value["p90_calls"] for value in reduced),
                "certification_rate": _mean(value["certification_rate"] for value in reduced),
                "certification_count": sum(value["certified"] is True for value in observations),
                "non_certification_count": sum(value["certified"] is False for value in observations),
                "source_runs": _source_runs(observations),
            }
        )
        rows.append(row)
    return rows


def _barp_rows(
    records: Sequence[dict[str, Any]], *, sensitivity: bool = False
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        record = item["record"]
        if record["mode"] != "routerbench_native_barp" or record.get("status") == "unavailable":
            continue
        is_sensitivity = record.get("analysis_label") == "sensitivity_without_degenerate_gsm8k"
        if is_sensitivity != sensitivity:
            continue
        identity = _identity(record)
        seed = record["identity"]["training_seed"]
        blocks = dict(record["native_barp"]["tasks"])
        blocks["macro_average"] = record["native_barp"]["macro_average"]
        for task, points in blocks.items():
            for point in points:
                grouped[identity + (task, point["preference_index"])].append(
                    {
                        "training_seed": seed,
                        "quality": point["mean_official_quality"],
                        "cost": point["mean_cost"],
                        "reward": point["mean_scalar_reward"],
                        "coverage": point["binary_coverage"],
                        "nondominated": bool(point["nondominated"]),
                        "w_quality": point["w_quality"],
                        "w_cost": point["w_cost"],
                        "training_tasks": list(record["configuration"]["training_tasks"]),
                        "source": item["source"],
                    }
                )
    rows = []
    for composite, observations in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        key, task, _ = composite[:-2], composite[-2], composite[-1]
        row = _base_row(
            key, "native_barp_sensitivity" if sensitivity else "native_barp"
        )
        row.update(
            {
                "target_task": task,
                "training_tasks": json.dumps(
                    observations[0]["training_tasks"], separators=(",", ":")
                ),
                "w_quality": observations[0]["w_quality"],
                "w_cost": observations[0]["w_cost"],
                "training_seed_count": len({value["training_seed"] for value in observations}),
                "redraw_count": 0,
                "observation_count": len(observations),
                "mean_official_quality": _mean(value["quality"] for value in observations),
                "mean_official_quality_se": _se(value["quality"] for value in observations),
                "mean_cost": _mean(value["cost"] for value in observations),
                "mean_cost_se": _se(value["cost"] for value in observations),
                "mean_scalar_reward": _mean(value["reward"] for value in observations),
                "binary_coverage": _mean(value["coverage"] for value in observations),
                "nondominated_rate": _mean(float(value["nondominated"]) for value in observations),
                "source_runs": _source_runs(observations),
            }
        )
        rows.append(row)
    return rows


def _unavailable_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in records:
        record = item["record"]
        if record.get("status") != "unavailable":
            continue
        key = _identity(record)
        failure = record.get("failure", {})
        training_tasks = list(record.get("configuration", {}).get("training_tasks", ()))
        target_tasks = list(record.get("configuration", {}).get("target_tasks", (None,)))
        for target_task in target_tasks:
            row = _base_row(key, "unavailable")
            row.update(
                {
                    "run_status": "unavailable",
                    "target_task": target_task,
                    "failure_stage": failure.get("stage"),
                    "failure_reason": failure.get("reason"),
                    "training_tasks": json.dumps(training_tasks, separators=(",", ":")),
                    "training_seed_count": 1,
                    "redraw_count": 0,
                    "observation_count": 0,
                    "source_runs": json.dumps([item["source"]], separators=(",", ":")),
                }
            )
            rows.append(row)
    return rows


def summarize_results(records: Iterable[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    materialized = [{"source": source, "record": record} for source, record in records]
    identities = set()
    for item in materialized:
        record = item["record"]
        identity = (
            record["mode"],
            tuple(sorted(record["identity"].items())),
        )
        if identity in identities:
            raise ValueError(f"duplicate immutable run identity {identity}")
        identities.add(identity)
    blocks = {
        "native_fixed_depth": _fixed_rows(materialized),
        "native_adaptive": _adaptive_rows(materialized),
        "adapted_calibrated": _calibrated_rows(materialized),
        "native_barp": _barp_rows(materialized),
        "native_barp_sensitivity": _barp_rows(materialized, sensitivity=True),
        "unavailable": _unavailable_rows(materialized),
    }
    rows = [row for name in blocks for row in blocks[name]]
    rows.sort(
        key=lambda row: tuple(
            str(row.get(key))
            for key in ("block", "mode", "method", "target_task", "pool_size", "pool_seed", "depth", "tau", "alpha", "w_cost")
        )
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "summary_kind": "preregistered_learned_baselines",
        "source_run_count": len(materialized),
        "row_count": len(rows),
        "aggregation": "training_seeds_within_redraw_then_redraw_summary",
        "blocks": blocks,
        "rows": rows,
    }


def _refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing summary {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    _refuse_existing(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(summary: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    _refuse_existing(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--json-out", required=True)
    arguments = parser.parse_args(argv)
    summary = summarize_results(discover_results(arguments.records))
    write_csv(summary["rows"], arguments.csv_out)
    write_json(summary, arguments.json_out)
    print(f"wrote {summary['row_count']} rows to {arguments.csv_out} and {arguments.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
