"""Summarize ISSUE-45 ordering and sizing controls from revised E2 records.

Rows retain every arm declared by each E2 record. Calls and standard all-miss
risk use certified redraws, while certification rates use every redraw. Paired
deltas are arm minus baseline on shared redraw IDs where both arms certified.
Score diagnostics summarize all evaluation-pool queries carried by the runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_calls_vs_risk import METHOD_SPECS, canonical_method_name  # noqa: E402

DESIGN = "B_prime_E2_frozen_selection"
SUMMARY_SCHEMA_VERSION = 1

PAIR_BASELINES = OrderedDict(
    [
        ("fixed_tally_factorized", "fixed_tally"),
        ("exact_independence", "exact_independence"),
        ("fixed_design_marginal_factorized", "fixed_design_marginal"),
    ]
)

FACTORIAL_METHODS = (
    "exact_independence",
    "fixed_design_marginal_factorized",
    "fixed_tally_factorized",
    "local_empirical",
    "local_order_factorized",
)

# Paul Tol's bright qualitative palette.
TOL_COLORS = {
    "exact_independence": "#4477AA",
    "fixed_design_marginal_factorized": "#66CCEE",
    "fixed_tally_factorized": "#228833",
    "local_empirical": "#EE6677",
    "local_order_factorized": "#AA3377",
}

BASE_COLUMNS = [
    "protocol",
    "benchmark",
    "pool_size",
    "neighbor_k",
    "target_alpha",
    "method",
    "method_label",
    "ordering_source",
    "sizing_source",
    "redraw_count",
    "certified_redraw_count",
    "certification_rate",
    "mean_calls",
    "mean_calls_se",
    "standard_all_miss_risk_mean",
    "standard_all_miss_risk_se",
    "diagnostic_redraw_count",
    "diagnostic_query_count",
    "distinct_finite_score_count_mean",
    "distinct_finite_score_count_min",
    "distinct_finite_score_count_max",
    "first_zero_miss_depth_mean",
    "first_zero_miss_depth_median",
    "first_zero_miss_depth_defined_rate",
    "plateau_depth_mean",
    "plateau_depth_median",
    "plateau_depth_defined_rate",
    "largest_tied_block_mean",
    "largest_tied_block_max",
    "score_one_tail_size_mean",
    "score_one_tail_size_max",
    "selected_depth_mean",
    "full_pool_call_rate",
    "factorized_underflow_count",
    "factorized_saturation_count",
    "source_records",
]

PAIR_COLUMNS = [
    column
    for suffix in PAIR_BASELINES.values()
    for column in (
        f"paired_calls_count_vs_{suffix}",
        f"calls_delta_mean_vs_{suffix}",
        f"calls_delta_se_vs_{suffix}",
        f"paired_risk_count_vs_{suffix}",
        f"standard_all_miss_risk_delta_mean_vs_{suffix}",
        f"standard_all_miss_risk_delta_se_vs_{suffix}",
    )
]

COLUMNS = BASE_COLUMNS + PAIR_COLUMNS


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


def _median(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    return statistics.median(numbers) if numbers else None


def _se(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    if len(numbers) < 2:
        return None
    return statistics.stdev(numbers) / math.sqrt(len(numbers))


def _alpha_entry(mapping: Any, alpha: float) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    for key, value in mapping.items():
        try:
            matches = math.isclose(float(key), alpha, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            matches = False
        if matches and isinstance(value, dict):
            return value
    return {}


def _canonical_or_error(raw_name: str) -> str:
    method = canonical_method_name(raw_name)
    if method is None:
        raise ValueError(f"E2 arm has no canonical public name: {raw_name}")
    return method


def _declared_raw_arms(record: Mapping[str, Any]) -> list[str]:
    controls = [str(value) for value in record.get("racer_controls", ())]
    wrappers = [f"wrapper_{value}_tuned" for value in record.get("wrapper_arms", ())]
    return controls + wrappers


def _actual_arm_map(cell: Mapping[str, Any]) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    result: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for raw_name, values in cell.items():
        if not isinstance(values, list):
            continue
        method = _canonical_or_error(str(raw_name))
        if method in result:
            raise ValueError(
                f"multiple E2 arm names map to canonical method {method!r}: "
                f"{result[method][0]!r} and {raw_name!r}"
            )
        result[method] = (
            str(raw_name),
            [value for value in values if isinstance(value, dict)],
        )
    return result


def _arm_metadata(record: Mapping[str, Any], raw_name: str) -> dict[str, Any]:
    name = raw_name
    if name.startswith("wrapper_"):
        name = name[len("wrapper_") :]
    if name.endswith("_tuned"):
        name = name[: -len("_tuned")]
    metadata = record.get("wrapper_arm_metadata", {})
    value = metadata.get(name, {}) if isinstance(metadata, dict) else {}
    return value if isinstance(value, dict) else {}


def discover_records(inputs: Iterable[str | Path]) -> list[tuple[str, dict[str, Any]]]:
    """Load revised E2 JSON records from files or directories."""

    paths: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            paths.update(candidate for candidate in path.rglob("*.json") if candidate.is_file())
        else:
            raise FileNotFoundError(path)
    records = []
    for path in sorted(paths, key=lambda value: str(value)):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("design") == DESIGN and isinstance(record.get("per_redraw"), dict):
            records.append((str(path), record))
    if not records:
        raise ValueError(f"no {DESIGN} E2 records found")
    return records


def _collect_observations(
    records: Iterable[tuple[str, dict[str, Any]]],
) -> tuple[
    dict[tuple[str, str, int, int | None, float, str], list[dict[str, Any]]],
    dict[tuple[str, str, int, int | None, float], set[str]],
]:
    grouped: dict[tuple[str, str, int, int | None, float, str], list[dict[str, Any]]] = defaultdict(list)
    expected_by_cell: dict[tuple[str, str, int, int | None, float], set[str]] = {}
    seen: set[tuple[str, str, int, int | None, float, str, int]] = set()

    for source, record in records:
        protocol = str(record.get("protocol", "routereval"))
        benchmark = str(record.get("benchmark", ""))
        pool_size = int(record["pool_size"])
        neighbor_value = record.get("primary_neighbor_k", record.get("knn_k"))
        neighbor_k = int(neighbor_value) if neighbor_value is not None else None
        alphas = [float(value) for value in record.get("alphas", ())]
        if not alphas:
            alphas = sorted(float(value) for value in record["per_redraw"])

        declared_raw = _declared_raw_arms(record)
        declared = {_canonical_or_error(name) for name in declared_raw}
        if len(declared) != len(declared_raw):
            raise ValueError("declared E2 arms do not have unique canonical public names")

        for alpha in alphas:
            cell_key = (protocol, benchmark, pool_size, neighbor_k, alpha)
            raw_cell = _alpha_entry(record["per_redraw"], alpha)
            actual = _actual_arm_map(raw_cell)
            expected = declared or set(actual)
            missing = sorted(expected - set(actual))
            unexpected = sorted(set(actual) - expected)
            if missing or unexpected:
                raise ValueError(
                    f"arm mismatch for pool={pool_size}, alpha={alpha}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            if cell_key in expected_by_cell and expected_by_cell[cell_key] != expected:
                raise ValueError(f"inconsistent declared arms for E2 cell {cell_key}")
            expected_by_cell[cell_key] = set(expected)

            for method in sorted(expected):
                raw_name, arm_records = actual[method]
                metadata = _arm_metadata(record, raw_name)
                for index, arm_record in enumerate(arm_records):
                    if "redraw" not in arm_record:
                        raise ValueError(
                            f"paired E2 reporting requires a redraw ID for {raw_name!r} "
                            f"at index {index}"
                        )
                    redraw = int(arm_record["redraw"])
                    identity = cell_key + (method, redraw)
                    if identity in seen:
                        raise ValueError(f"duplicate E2 redraw identity {identity}")
                    seen.add(identity)
                    grouped[cell_key + (method,)].append(
                        {
                            "source": source,
                            "redraw": redraw,
                            "record": arm_record,
                            "ordering_source": metadata.get("ordering_source"),
                            "sizing_source": metadata.get("sizing_source"),
                        }
                    )

    for values in grouped.values():
        values.sort(key=lambda value: value["redraw"])
    return grouped, expected_by_cell


def _record_calls(record: Mapping[str, Any]) -> float | None:
    for name in ("calls_test", "calls"):
        value = _finite(record.get(name))
        if value is not None:
            return value
    return None


def _record_risk(record: Mapping[str, Any]) -> float | None:
    for name in ("standard_all_miss_risk", "risk_test"):
        value = _finite(record.get(name))
        if value is not None:
            return value
    return None


def _diagnostic_summary(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diagnostics: list[Mapping[str, Any]] = []
    diagnostic_redraw_count = 0
    for observation in observations:
        values = observation["record"].get("score_diagnostics")
        if isinstance(values, list):
            diagnostic_redraw_count += 1
            diagnostics.extend(value for value in values if isinstance(value, dict))

    distinct = [value.get("distinct_finite_score_count") for value in diagnostics]
    first_zero = [value.get("first_zero_miss_depth") for value in diagnostics]
    plateau = [value.get("plateau_depth") for value in diagnostics]
    largest_tied = [
        max([1, *[int(size) for size in value.get("tied_block_sizes", ())]])
        for value in diagnostics
    ]
    score_one = [value.get("score_one_tail_size") for value in diagnostics]
    selected_depth = [value.get("selected_depth") for value in diagnostics]
    full_pool = [
        bool(value["full_pool_call_indicator"])
        for value in diagnostics
        if value.get("full_pool_call_indicator") is not None
    ]
    zero_known = [value for value in first_zero if _finite(value) is not None]
    plateau_known = [value for value in plateau if _finite(value) is not None]
    query_count = len(diagnostics)

    return {
        "diagnostic_redraw_count": diagnostic_redraw_count,
        "diagnostic_query_count": query_count,
        "distinct_finite_score_count_mean": _mean(distinct),
        "distinct_finite_score_count_min": min(distinct) if distinct else None,
        "distinct_finite_score_count_max": max(distinct) if distinct else None,
        "first_zero_miss_depth_mean": _mean(zero_known),
        "first_zero_miss_depth_median": _median(zero_known),
        "first_zero_miss_depth_defined_rate": len(zero_known) / query_count if query_count else None,
        "plateau_depth_mean": _mean(plateau_known),
        "plateau_depth_median": _median(plateau_known),
        "plateau_depth_defined_rate": len(plateau_known) / query_count if query_count else None,
        "largest_tied_block_mean": _mean(largest_tied),
        "largest_tied_block_max": max(largest_tied) if largest_tied else None,
        "score_one_tail_size_mean": _mean(score_one),
        "score_one_tail_size_max": max(score_one) if score_one else None,
        "selected_depth_mean": _mean(selected_depth),
        "full_pool_call_rate": _mean(float(value) for value in full_pool),
        "factorized_underflow_count": sum(
            int(value.get("factorized_underflow_count", 0)) for value in diagnostics
        ),
        "factorized_saturation_count": sum(
            int(value.get("factorized_saturation_count", 0)) for value in diagnostics
        ),
    }


def _paired_values(
    observations: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    extractor: Any,
) -> list[float]:
    arm_by_redraw = {
        int(value["redraw"]): value["record"]
        for value in observations
        if value["record"].get("certified") is True
    }
    baseline_by_redraw = {
        int(value["redraw"]): value["record"]
        for value in baseline
        if value["record"].get("certified") is True
    }
    differences = []
    for redraw in sorted(set(arm_by_redraw) & set(baseline_by_redraw)):
        arm_value = extractor(arm_by_redraw[redraw])
        baseline_value = extractor(baseline_by_redraw[redraw])
        if arm_value is not None and baseline_value is not None:
            differences.append(arm_value - baseline_value)
    return differences


def summarize_records(
    records: Iterable[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Return complete canonical rows and paired ISSUE-45 comparisons."""

    materialized = list(records)
    grouped, expected_by_cell = _collect_observations(materialized)
    rows = []
    method_order = {name: index for index, name in enumerate(METHOD_SPECS)}

    for key, observations in grouped.items():
        protocol, benchmark, pool_size, neighbor_k, alpha, method = key
        cell_key = key[:-1]
        records_for_arm = [value["record"] for value in observations]
        certified = [value for value in records_for_arm if value.get("certified") is True]
        calls = [_record_calls(value) for value in certified]
        risks = [_record_risk(value) for value in certified]
        ordering_sources = {
            value["ordering_source"] for value in observations if value["ordering_source"] is not None
        }
        sizing_sources = {
            value["sizing_source"] for value in observations if value["sizing_source"] is not None
        }
        if len(ordering_sources) > 1 or len(sizing_sources) > 1:
            raise ValueError(f"inconsistent arm metadata for {key}")

        row = {
            "protocol": protocol,
            "benchmark": benchmark,
            "pool_size": pool_size,
            "neighbor_k": neighbor_k,
            "target_alpha": alpha,
            "method": method,
            "method_label": METHOD_SPECS[method]["label"],
            "ordering_source": next(iter(ordering_sources), None),
            "sizing_source": next(iter(sizing_sources), None),
            "redraw_count": len(observations),
            "certified_redraw_count": len(certified),
            "certification_rate": len(certified) / len(observations) if observations else None,
            "mean_calls": _mean(calls),
            "mean_calls_se": _se(calls),
            "standard_all_miss_risk_mean": _mean(risks),
            "standard_all_miss_risk_se": _se(risks),
            "source_records": json.dumps(
                sorted({str(value["source"]) for value in observations}),
                separators=(",", ":"),
            ),
            **_diagnostic_summary(observations),
        }

        for baseline_method, suffix in PAIR_BASELINES.items():
            baseline = grouped.get(cell_key + (baseline_method,), ())
            call_differences = _paired_values(observations, baseline, _record_calls)
            risk_differences = _paired_values(observations, baseline, _record_risk)
            row.update(
                {
                    f"paired_calls_count_vs_{suffix}": len(call_differences),
                    f"calls_delta_mean_vs_{suffix}": _mean(call_differences),
                    f"calls_delta_se_vs_{suffix}": _se(call_differences),
                    f"paired_risk_count_vs_{suffix}": len(risk_differences),
                    f"standard_all_miss_risk_delta_mean_vs_{suffix}": _mean(risk_differences),
                    f"standard_all_miss_risk_delta_se_vs_{suffix}": _se(risk_differences),
                }
            )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["protocol"],
            row["benchmark"],
            row["pool_size"],
            -1 if row["neighbor_k"] is None else row["neighbor_k"],
            row["target_alpha"],
            method_order[row["method"]],
        )
    )

    for cell_key, expected in expected_by_cell.items():
        reported = {
            row["method"]
            for row in rows
            if (
                row["protocol"],
                row["benchmark"],
                row["pool_size"],
                row["neighbor_k"],
                row["target_alpha"],
            )
            == cell_key
        }
        if reported != expected:
            raise AssertionError(f"summary omitted E2 arms for {cell_key}: {sorted(expected - reported)}")

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "summary_kind": "issue_45_factorial_controls",
        "source_record_count": len(materialized),
        "row_count": len(rows),
        "delta_definition": "arm_minus_baseline_on_shared_certified_redraws",
        "diagnostic_scope": "all_evaluation_pool_queries_when_present",
        "pair_baselines": list(PAIR_BASELINES),
        "factorial_methods": list(FACTORIAL_METHODS),
        "rows": rows,
    }


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(summary: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def render_figure(rows: Sequence[Mapping[str, Any]], output_prefix: str | Path) -> Path:
    """Render compact calls and risk panels for the factorial methods."""

    plotted = [row for row in rows if row["method"] in FACTORIAL_METHODS]
    if not plotted:
        raise ValueError("no factorial-control rows available to plot")
    pools = sorted({int(row["pool_size"]) for row in plotted})
    figure, axes = plt.subplots(
        2,
        len(pools),
        figsize=(max(4.2, 2.55 * len(pools)), 5.6),
        squeeze=False,
        sharex="col",
    )

    for column, pool_size in enumerate(pools):
        pool_rows = [row for row in plotted if int(row["pool_size"]) == pool_size]
        for method in FACTORIAL_METHODS:
            method_rows = sorted(
                (row for row in pool_rows if row["method"] == method),
                key=lambda row: float(row["target_alpha"]),
            )
            calls = [row for row in method_rows if _finite(row.get("mean_calls")) is not None]
            risks = [
                row
                for row in method_rows
                if _finite(row.get("standard_all_miss_risk_mean")) is not None
            ]
            label = METHOD_SPECS[method]["label"]
            if calls:
                axes[0, column].plot(
                    [float(row["target_alpha"]) for row in calls],
                    [float(row["mean_calls"]) for row in calls],
                    marker="o",
                    markersize=2.8,
                    linewidth=1.35,
                    color=TOL_COLORS[method],
                    label=label,
                )
            if risks:
                axes[1, column].plot(
                    [float(row["target_alpha"]) for row in risks],
                    [float(row["standard_all_miss_risk_mean"]) for row in risks],
                    marker="o",
                    markersize=2.8,
                    linewidth=1.35,
                    color=TOL_COLORS[method],
                    label=label,
                )

        alpha_values = sorted({float(row["target_alpha"]) for row in pool_rows})
        if alpha_values:
            axes[1, column].plot(
                alpha_values,
                alpha_values,
                color="#BBBBBB",
                linestyle=":",
                linewidth=1.0,
                label="Target risk",
            )
        axes[0, column].set_title(f"Pool {pool_size}", fontsize=9)
        axes[0, column].grid(alpha=0.18, linewidth=0.5)
        axes[1, column].grid(alpha=0.18, linewidth=0.5)
        axes[1, column].set_xlabel("Target all-miss risk", fontsize=8)
        axes[0, column].tick_params(labelsize=7)
        axes[1, column].tick_params(labelsize=7)

    axes[0, 0].set_ylabel("Mean calls", fontsize=8)
    axes[1, 0].set_ylabel("Standard all-miss risk", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    figure.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=min(5, max(1, len(by_label))),
        frameon=False,
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 1), h_pad=1.0, w_pad=0.8)

    prefix = Path(output_prefix)
    pdf = prefix.with_suffix(".pdf")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf, dpi=300, bbox_inches="tight")
    figure.savefig(prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)
    return pdf


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--fig-out", required=True)
    arguments = parser.parse_args(argv)

    summary = summarize_records(discover_records(arguments.records))
    write_csv(summary["rows"], arguments.csv_out)
    write_json(summary, arguments.json_out)
    pdf = render_figure(summary["rows"], arguments.fig_out)
    print(f"wrote {summary['row_count']} rows to {arguments.csv_out} and {arguments.json_out}")
    print(f"wrote {pdf} and {pdf.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
