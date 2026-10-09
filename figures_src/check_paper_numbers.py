"""Deterministically verify generated paper tables and figure metadata.

The checker reconstructs the canonical query-level table from source JSON
records, compares every generated CSV value, recomputes every risk flag using
``mean > alpha + 2 * SE``, validates event counts, and confirms that JSON figure
ranges contain every plotted point and every RACER baseline point.

Exit status is zero only when every check passes. Errors are sorted before they
are printed, so repeated runs produce identical diagnostics.

Example:

  python figures_src/check_paper_numbers.py \
      --records results/query_level_calibrated \
      --table out/query_level_table.csv \
      --figure-metadata out/calls_vs_risk.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from plot_calls_vs_risk import (
    TABLE_COLUMNS,
    load_records,
    risk_above_target,
    rows_for,
)


FLOAT_COLUMNS = {
    "target_alpha",
    "mean_calls",
    "reported_risk_mean",
    "reported_risk_se",
    "safe_action_risk_mean",
    "standard_all_miss_risk_mean",
    "augmented_loss_mean",
    "certification_rate",
}
INTEGER_COLUMNS = {
    "pool_size",
    "reported_event_count",
    "reported_trial_count",
    "safe_action_event_count",
    "standard_all_miss_event_count",
    "augmented_loss_event_count",
    "non_certification_count",
    "redraw_count",
}
KEY_COLUMNS = ("pool_size", "target_alpha", "method")


def _empty(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _float(value: Any) -> float | None:
    if _empty(value):
        return None
    return float(value)


def _integer(value: Any) -> int | None:
    if _empty(value):
        return None
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"expected integer, found {value!r}")
    return int(numeric)


def _boolean(value: Any) -> bool | None:
    if _empty(value):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0"}:
        return False
    raise ValueError(f"expected boolean, found {value!r}")


def _key(row: dict[str, Any]) -> tuple[int, float, str]:
    return int(float(row["pool_size"])), float(row["target_alpha"]), str(row["method"])


def _same_float(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _format_key(key: tuple[int, float, str]) -> str:
    return f"pool={key[0]}, alpha={key[1]:g}, method={key[2]}"


def _read_table(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def check_table(
    records_directory: str | Path,
    table_path: str | Path,
    *,
    tolerance: float = 1e-10,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return deterministic CSV errors and reconstructed expected rows."""

    errors: list[str] = []
    records = load_records(records_directory)
    expected_rows = [row for record in records.values() for row in rows_for(record)]
    headers, actual_rows = _read_table(table_path)
    if headers != TABLE_COLUMNS:
        errors.append(
            "table headers differ from canonical columns: "
            f"expected {TABLE_COLUMNS!r}, found {headers!r}"
        )

    try:
        expected = {_key(row): row for row in expected_rows}
        actual = {_key(row): row for row in actual_rows}
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"table identity parsing failed: {error}")
        return sorted(errors), expected_rows
    if len(expected) != len(expected_rows):
        errors.append("source records produce duplicate canonical table identities")
    if len(actual) != len(actual_rows):
        errors.append("generated table contains duplicate canonical row identities")

    for key in sorted(expected.keys() - actual.keys()):
        errors.append(f"table missing row {_format_key(key)}")
    for key in sorted(actual.keys() - expected.keys()):
        errors.append(f"table has unexpected row {_format_key(key)}")

    for key in sorted(expected.keys() & actual.keys()):
        expected_row = expected[key]
        actual_row = actual[key]
        identity = _format_key(key)
        for column in TABLE_COLUMNS:
            expected_value = expected_row.get(column)
            actual_value = actual_row.get(column)
            try:
                if column in FLOAT_COLUMNS:
                    matches = _same_float(
                        _float(expected_value), _float(actual_value), tolerance
                    )
                elif column in INTEGER_COLUMNS:
                    matches = _integer(expected_value) == _integer(actual_value)
                elif column == "above_target_2se":
                    matches = _boolean(expected_value) == _boolean(actual_value)
                else:
                    normalized_expected = "" if expected_value is None else str(expected_value)
                    normalized_actual = "" if actual_value is None else str(actual_value)
                    matches = normalized_expected == normalized_actual
            except (TypeError, ValueError) as error:
                errors.append(f"{identity}: invalid {column}: {error}")
                continue
            if not matches:
                errors.append(
                    f"{identity}: {column} mismatch, expected {expected_value!r}, "
                    f"found {actual_value!r}"
                )

        mean = _float(actual_row.get("reported_risk_mean"))
        standard_error = _float(actual_row.get("reported_risk_se"))
        flag = None
        try:
            flag = _boolean(actual_row.get("above_target_2se"))
        except ValueError as error:
            errors.append(f"{identity}: invalid above_target_2se: {error}")
        classification = actual_row.get("risk_classification")
        if mean is None or standard_error is None:
            if flag is not None or classification != "unavailable":
                errors.append(
                    f"{identity}: unavailable SE must have blank flag and unavailable classification"
                )
        else:
            recomputed = risk_above_target(mean, key[1], standard_error)
            expected_classification = "above_target_2se" if recomputed else "within_2se"
            if flag != recomputed:
                errors.append(
                    f"{identity}: risk flag does not equal mean > alpha + 2*SE"
                )
            if classification != expected_classification:
                errors.append(
                    f"{identity}: risk classification mismatch, expected "
                    f"{expected_classification!r}"
                )

        trials = _integer(actual_row.get("reported_trial_count"))
        events = _integer(actual_row.get("reported_event_count"))
        if trials is not None and events is not None and mean is not None:
            if abs(events - mean * trials) > max(1e-7, tolerance * trials):
                errors.append(
                    f"{identity}: reported event count {events} is inconsistent with "
                    f"mean {mean} and trials {trials}"
                )
        elif actual_row.get("event_count_status") != "unavailable_in_source":
            errors.append(
                f"{identity}: missing event count must be labeled unavailable_in_source"
            )

    return sorted(errors), expected_rows


def _range_contains(value: float, lower: float, upper: float, tolerance: float) -> bool:
    return lower - tolerance <= value <= upper + tolerance


def check_figure_metadata(
    path: str | Path,
    *,
    table_rows: list[dict[str, Any]] | None = None,
    tolerance: float = 1e-10,
) -> list[str]:
    """Verify a generated figure JSON range and, when available, source points."""

    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    kind = metadata.get("figure_kind")
    panels = metadata.get("panels")
    if not isinstance(panels, list):
        return [f"{path}: metadata panels must be a list"]

    table_by_pool: dict[int, list[dict[str, Any]]] = {}
    if table_rows is not None:
        for row in table_rows:
            table_by_pool.setdefault(int(row["pool_size"]), []).append(row)

    for index, panel in enumerate(panels):
        context = f"{path}: panel {index}"
        try:
            lower = float(panel["axis_x_min"])
            upper = float(panel["axis_x_max"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{context}: invalid axis range: {error}")
            continue
        if not lower < upper:
            errors.append(f"{context}: axis_x_min must be below axis_x_max")
            continue

        if kind == "calls_vs_risk":
            points = panel.get("points", [])
            point_calls = [float(point["mean_calls"]) for point in points]
            baseline_calls = [float(value) for value in panel.get("baseline_calls", [])]
            for value in point_calls:
                if not _range_contains(value, lower, upper, tolerance):
                    errors.append(f"{context}: plotted call value {value} lies outside x range")
            for value in baseline_calls:
                if not _range_contains(value, lower, upper, tolerance):
                    errors.append(f"{context}: baseline call value {value} lies outside x range")
            pool_size = int(panel["pool_size"])
            expected_rows = table_by_pool.get(pool_size, [])
            expected_baselines = sorted(
                float(row["mean_calls"])
                for row in expected_rows
                if row["method"] in {"racer_g", "racer_p"}
            )
            if sorted(baseline_calls) != expected_baselines:
                errors.append(
                    f"{context}: baseline call list does not match canonical table"
                )
            expected_points = sorted(
                (
                    str(row["method"]),
                    float(row["target_alpha"]),
                    float(row["mean_calls"]),
                )
                for row in expected_rows
            )
            actual_points = sorted(
                (
                    str(point["method"]),
                    float(point["target_alpha"]),
                    float(point["mean_calls"]),
                )
                for point in points
            )
            if actual_points != expected_points:
                errors.append(f"{context}: plotted points do not match canonical table")
        elif kind == "benchmark_level_sweep":
            baseline_points = panel.get("baseline_points", [])
            method_points = panel.get("method_points", {})
            alpha_values = [float(point["target_alpha"]) for point in baseline_points]
            for points in method_points.values():
                alpha_values.extend(float(point["target_alpha"]) for point in points)
            for value in alpha_values:
                if not _range_contains(value, lower, upper, tolerance):
                    errors.append(f"{context}: target alpha {value} lies outside x range")
        else:
            errors.append(f"{context}: unknown figure_kind {kind!r}")
    return sorted(errors)


def run_checks(
    records_directory: str | Path,
    table_path: str | Path,
    figure_metadata_paths: list[str | Path],
    *,
    tolerance: float = 1e-10,
) -> list[str]:
    errors, rows = check_table(records_directory, table_path, tolerance=tolerance)
    for path in figure_metadata_paths:
        errors.extend(
            check_figure_metadata(path, table_rows=rows, tolerance=tolerance)
        )
    return sorted(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument(
        "--figure-metadata",
        action="append",
        required=True,
        help="generated JSON metadata; repeat for multiple figures",
    )
    parser.add_argument("--tolerance", type=float, default=1e-10)
    arguments = parser.parse_args()

    errors = run_checks(
        arguments.records,
        arguments.table,
        arguments.figure_metadata,
        tolerance=arguments.tolerance,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"number check failed with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("number check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
