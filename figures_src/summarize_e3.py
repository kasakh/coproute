"""Summarize repaired E3 pool and class-penalty run records.

The output has one row per protocol, benchmark, pool size, target level, and
penalty setting. Statistics use certified redraws only. Missing controls remain
missing rather than being reconstructed from validation-only candidates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

DESIGN = "E3_unified_frozen_selection"

COLUMNS = [
    "protocol",
    "benchmark",
    "pool_size",
    "target_alpha",
    "redraw_count",
    "selected_mean_calls",
    "selected_mean_calls_se",
    "selected_standard_all_miss_risk_mean",
    "selected_standard_all_miss_risk_se",
    "selected_certification_count",
    "selected_certification_rate",
    "selected_certification_status_available_count",
    "selected_non_certification_count",
    "class_win_counts",
    "variant_win_counts",
    "validation_fallback_count",
    "validation_fallback_rate",
    "racer_g_mean_calls",
    "racer_g_mean_calls_se",
    "racer_g_standard_all_miss_risk_mean",
    "racer_g_standard_all_miss_risk_se",
    "racer_g_certification_count",
    "racer_g_certification_rate",
    "racer_p_mean_calls",
    "racer_p_mean_calls_se",
    "racer_p_standard_all_miss_risk_mean",
    "racer_p_standard_all_miss_risk_se",
    "racer_p_certification_count",
    "racer_p_certification_rate",
    "exact_independence_mean_calls",
    "exact_independence_mean_calls_se",
    "exact_independence_standard_all_miss_risk_mean",
    "exact_independence_standard_all_miss_risk_se",
    "exact_independence_certification_count",
    "exact_independence_certification_rate",
    "class_penalty_enabled",
    "class_penalty_margin",
    "penalty_setting",
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


def _alpha_entry(mapping: Any, alpha: float) -> Any:
    if not isinstance(mapping, dict):
        return None
    direct = mapping.get(str(alpha))
    if direct is not None:
        return direct
    for key, value in mapping.items():
        try:
            matches = float(key) == alpha
        except (TypeError, ValueError):
            matches = False
        if matches:
            return value
    return None


def _certification(record: Any) -> bool | None:
    if not isinstance(record, dict):
        return None
    for key in ("final_certified", "certified"):
        if key in record and record[key] is not None:
            return bool(record[key])
    if "non_certification" in record and record["non_certification"] is not None:
        return not bool(record["non_certification"])
    return None


def _calls(record: Any) -> float | None:
    if not isinstance(record, dict):
        return None
    for key in ("test_calls", "model_calls", "calls_mean", "mean_calls"):
        if key in record:
            return _number(record[key])
    return None


def _risk(record: Any) -> float | None:
    if not isinstance(record, dict):
        return None
    for key in (
        "standard_all_miss_risk",
        "standard_all_miss_risk_mean",
        "test_risk",
        "risk_mean",
    ):
        if key in record:
            return _number(record[key])
    return None


def _feasible(record: Any) -> bool:
    certified = _certification(record)
    if certified is not None:
        return certified
    return _calls(record) is not None and _risk(record) is not None


def _control(record: Any, names: tuple[str, ...]) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    containers = [
        record,
        record.get("racer_controls"),
        record.get("controls"),
        record.get("baselines"),
        record.get("predeclared_baselines"),
        record.get("wrapper_controls"),
    ]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for name in names:
            value = container.get(name)
            if isinstance(value, dict):
                return value
    return None


def _status_summary(records: list[dict[str, Any] | None]) -> tuple[int | None, float | None, int]:
    statuses = [_certification(record) for record in records]
    known = [status for status in statuses if status is not None]
    if not known:
        return None, None, 0
    count = sum(bool(status) for status in known)
    rate = count / len(known) if len(known) == len(statuses) else None
    return count, rate, len(known)


def _metric_summary(records: list[dict[str, Any] | None]) -> dict[str, Any]:
    feasible = [record for record in records if _feasible(record)]
    calls = [_calls(record) for record in feasible]
    risks = [_risk(record) for record in feasible]
    certification_count, certification_rate, certification_available = _status_summary(records)
    return {
        "mean_calls": _mean(calls),
        "mean_calls_se": _se(calls),
        "risk_mean": _mean(risks),
        "risk_se": _se(risks),
        "certification_count": certification_count,
        "certification_rate": certification_rate,
        "certification_available": certification_available,
    }


def _aggregate_control(entry: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any] | None:
    control = _control(entry, names)
    if control is None:
        return None
    return control


def _aggregate_metric(control: dict[str, Any] | None) -> dict[str, Any]:
    if control is None:
        return {
            "mean_calls": None,
            "mean_calls_se": None,
            "risk_mean": None,
            "risk_se": None,
            "certification_count": None,
            "certification_rate": None,
            "certification_available": 0,
        }
    return {
        "mean_calls": _calls(control),
        "mean_calls_se": _number(control.get("calls_se", control.get("mean_calls_se"))),
        "risk_mean": _risk(control),
        "risk_se": _number(control.get("standard_all_miss_risk_se", control.get("risk_se"))),
        "certification_count": control.get("certification_count"),
        "certification_rate": _number(control.get("certification_rate")),
        "certification_available": 0,
    }


def _counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        selected = record.get("selected")
        value = selected.get(key) if isinstance(selected, dict) else None
        if value is not None:
            name = str(value)
            result[name] = result.get(name, 0) + 1
    return dict(sorted(result.items()))


def rows_for_record(record: dict[str, Any], source_record: str = "") -> list[dict[str, Any]]:
    """Create canonical rows for one repaired E3 record."""

    if record.get("design") != DESIGN:
        raise ValueError(f"expected design {DESIGN!r}, found {record.get('design')!r}")
    required = ("protocol", "benchmark", "pool_size")
    missing = [key for key in required if record.get(key) is None]
    if missing:
        raise ValueError(f"record lacks identity fields: {', '.join(missing)}")

    alphas = sorted(float(alpha) for alpha in record.get("alphas", []))
    if not alphas and isinstance(record.get("aggregate"), dict):
        alphas = sorted(float(alpha) for alpha in record["aggregate"])
    penalty_enabled = record.get("class_penalty_enabled")
    penalty_margin = _number(record.get("class_penalty_margin"))
    if penalty_enabled is True:
        penalty_setting = f"enabled:{penalty_margin:g}" if penalty_margin is not None else "enabled:margin_unavailable"
    elif penalty_enabled is False:
        penalty_setting = "disabled"
    else:
        penalty_setting = "unavailable"

    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        raw_records = _alpha_entry(record.get("per_alpha_records"), alpha)
        redraws = raw_records if isinstance(raw_records, list) else []
        aggregate = _alpha_entry(record.get("aggregate"), alpha)
        aggregate = aggregate if isinstance(aggregate, dict) else {}

        selected_records = [entry.get("selected") for entry in redraws]
        racer_g_records = [_control(entry, ("racer_G", "racer_g", "RACER-G")) for entry in redraws]
        racer_p_records = [_control(entry, ("racer_P", "racer_p", "RACER-P")) for entry in redraws]
        exact_records = [
            _control(entry, ("exact_independence", "factorized_independence", "Exact independence"))
            for entry in redraws
        ]

        if redraws:
            selected = _metric_summary(selected_records)
            racer_g = _metric_summary(racer_g_records)
            racer_p = _metric_summary(racer_p_records)
            exact = _metric_summary(exact_records)
            class_counts = _counts(redraws, "policy_class")
            variant_counts = _counts(redraws, "variant")
            fallback_values = [entry.get("fallback_used") for entry in redraws]
            fallback_known = [value for value in fallback_values if value is not None]
            fallback_count = sum(bool(value) for value in fallback_known) if fallback_known else None
            fallback_rate = (
                fallback_count / len(fallback_known)
                if fallback_count is not None and len(fallback_known) == len(fallback_values)
                else None
            )
        else:
            selected = {
                "mean_calls": _number(aggregate.get("selected_test_calls_mean")),
                "mean_calls_se": _number(aggregate.get("selected_test_calls_se")),
                "risk_mean": _number(aggregate.get("selected_standard_all_miss_risk_mean")),
                "risk_se": _number(aggregate.get("selected_standard_all_miss_risk_se")),
                "certification_count": aggregate.get("selected_certification_count"),
                "certification_rate": _number(aggregate.get("selected_certification_rate")),
                "certification_available": 0,
            }
            racer_g = _aggregate_metric(_aggregate_control(aggregate, ("racer_G", "racer_g", "RACER-G")))
            racer_p = _aggregate_metric(_aggregate_control(aggregate, ("racer_P", "racer_p", "RACER-P")))
            exact = _aggregate_metric(_aggregate_control(aggregate, ("exact_independence", "factorized_independence", "Exact independence")))
            class_counts = dict(sorted((aggregate.get("class_win_counts") or {}).items()))
            variant_counts = dict(sorted((aggregate.get("variant_win_counts") or {}).items()))
            fallback_fraction = _number(aggregate.get("fallback_fraction"))
            redraw_count = aggregate.get("n_redraws", record.get("redraws"))
            fallback_count = (
                int(round(fallback_fraction * int(redraw_count)))
                if fallback_fraction is not None and redraw_count is not None
                else None
            )
            fallback_rate = fallback_fraction

        redraw_count = len(redraws) if redraws else aggregate.get("n_redraws", record.get("redraws"))
        selected_noncertification = (
            int(redraw_count) - int(selected["certification_count"])
            if (
                redraw_count is not None
                and selected["certification_count"] is not None
                and selected["certification_available"] == int(redraw_count)
            )
            else aggregate.get("selected_non_certification_count")
        )
        row = {
            "protocol": record["protocol"],
            "benchmark": record["benchmark"],
            "pool_size": record["pool_size"],
            "target_alpha": alpha,
            "redraw_count": redraw_count,
            "selected_mean_calls": selected["mean_calls"],
            "selected_mean_calls_se": selected["mean_calls_se"],
            "selected_standard_all_miss_risk_mean": selected["risk_mean"],
            "selected_standard_all_miss_risk_se": selected["risk_se"],
            "selected_certification_count": selected["certification_count"],
            "selected_certification_rate": selected["certification_rate"],
            "selected_certification_status_available_count": selected["certification_available"],
            "selected_non_certification_count": selected_noncertification,
            "class_win_counts": _json(class_counts),
            "variant_win_counts": _json(variant_counts),
            "validation_fallback_count": fallback_count,
            "validation_fallback_rate": fallback_rate,
            "racer_g_mean_calls": racer_g["mean_calls"],
            "racer_g_mean_calls_se": racer_g["mean_calls_se"],
            "racer_g_standard_all_miss_risk_mean": racer_g["risk_mean"],
            "racer_g_standard_all_miss_risk_se": racer_g["risk_se"],
            "racer_g_certification_count": racer_g["certification_count"],
            "racer_g_certification_rate": racer_g["certification_rate"],
            "racer_p_mean_calls": racer_p["mean_calls"],
            "racer_p_mean_calls_se": racer_p["mean_calls_se"],
            "racer_p_standard_all_miss_risk_mean": racer_p["risk_mean"],
            "racer_p_standard_all_miss_risk_se": racer_p["risk_se"],
            "racer_p_certification_count": racer_p["certification_count"],
            "racer_p_certification_rate": racer_p["certification_rate"],
            "exact_independence_mean_calls": exact["mean_calls"],
            "exact_independence_mean_calls_se": exact["mean_calls_se"],
            "exact_independence_standard_all_miss_risk_mean": exact["risk_mean"],
            "exact_independence_standard_all_miss_risk_se": exact["risk_se"],
            "exact_independence_certification_count": exact["certification_count"],
            "exact_independence_certification_rate": exact["certification_rate"],
            "class_penalty_enabled": penalty_enabled,
            "class_penalty_margin": penalty_margin,
            "penalty_setting": penalty_setting,
            "source_record": source_record,
        }
        rows.append(row)
    return rows


def discover_records(inputs: Iterable[str | Path]) -> list[tuple[str, dict[str, Any]]]:
    """Load matching E3 records from files or directories in stable order."""

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
            identity = tuple(row[key] for key in ("protocol", "benchmark", "pool_size", "target_alpha", "penalty_setting"))
            if identity in identities:
                raise ValueError(f"duplicate canonical E3 row {identity}")
            identities.add(identity)
            rows.append(row)
    return sorted(rows, key=lambda row: (str(row["protocol"]), str(row["benchmark"]), int(row["pool_size"]), float(row["target_alpha"]), str(row["penalty_setting"])))


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
    payload = {"schema_version": 1, "summary_kind": "repaired_e3_pool_penalty_runs", "row_count": len(rows), "rows": rows}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True, help="JSON files or directories")
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--json-out", required=True)
    arguments = parser.parse_args()
    rows = summarize_records(discover_records(arguments.records))
    write_csv(rows, arguments.csv_out)
    write_json(rows, arguments.json_out)
    print(f"wrote {len(rows)} rows to {arguments.csv_out} and {arguments.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
