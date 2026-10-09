"""Regenerate benchmark-level calibrated sweep figures.

Legacy ``bprime_summary.csv`` columns and repaired canonical columns are both
accepted. Internally every row is normalized to ``design_budget``,
``target_alpha``, ``mean_calls``, and ``reported_risk_mean``. Every available
cell is shown, including cells with one or zero certified baseline points.
Sparse cells receive an explicit annotation instead of being omitted.

Each PDF and PNG pair has a JSON companion containing all points and axis
ranges. The explicit x range includes every baseline target point.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TITLES = {
    ("routerbench", "mmlu", "100"): "MMLU (RouterBench, pool 11)",
    ("routerbench", "mmlu", "500"): "MMLU (RouterBench, pool 11, budget 500)",
    ("routerbench", "hellaswag", "100"): "HellaSwag (RouterBench, pool 11)",
    ("routerbench", "hellaswag", "500"): "HellaSwag (RouterBench, pool 11, budget 500)",
    ("routerbench", "winogrande", "100"): "Winogrande (RouterBench, pool 11)",
    ("routerbench", "arc-challenge", "100"): "ARC-Challenge (RouterBench, pool 11)",
    ("routerbench", "gsm8k", "100"): "GSM8K (RouterBench, pool 11)",
    ("routerbench", "mbpp", "100"): "MBPP (RouterBench, pool 11)",
    (
        "routereval",
        "bbh_boolean_expressions",
        "100",
    ): "BBH boolean expressions (RouterEval, pool 3811)",
    ("routereval", "bbh_navigate", "100"): "BBH navigate (RouterEval, pool 3811)",
    (
        "routereval",
        "gpqa_diamond",
        "50",
    ): "GPQA diamond (RouterEval, pool 3811, budget 50)",
    (
        "routereval",
        "ifeval",
        "100",
    ): "Instruction following, IFEval (RouterEval, pool 3811)",
    (
        "routereval",
        "ifeval",
        "250",
    ): "Instruction following, IFEval (RouterEval, budget 250)",
    (
        "routereval",
        "math_algebra_hard",
        "100",
    ): "MATH algebra hard (RouterEval, pool 3811)",
    (
        "routereval",
        "math_intermediate_algebra_hard",
        "100",
    ): "MATH intermediate algebra hard (RouterEval, pool 3811)",
}

METHOD_LABELS = {
    "marginal": "Marginal ordering",
    "empirical_tally": "Empirical tally",
    "error_correlation": "Failure correlation",
    "gaussian_copula": "Regularized correlation",
    "factorized_independence": "Exact independence",
    "local_empirical": "Local empirical",
    "fixed_tally_factorized": "Fixed-tally factorized sizing",
    "pairwise_heuristic": "Pairwise heuristic",
    "correlation_heuristic": "Correlation heuristic",
    "regularized_correlation": "Regularized correlation",
}

TOL_COLORS = [
    "#0077BB",
    "#33BBEE",
    "#009988",
    "#EE7733",
    "#CC3311",
    "#EE3377",
    "#BBBBBB",
    "#AA4499",
]


def _first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _canonical_method(name: str) -> str:
    aliases = {
        "local_empirical_coverage": "local_empirical",
        "fixed_empirical_tally": "fixed_tally_factorized",
        "fixed_tally_factorized_sizing": "fixed_tally_factorized",
        "pairwise_count": "pairwise_heuristic",
        "error_correlation_heuristic": "correlation_heuristic",
    }
    return aliases.get(str(name), str(name))


def normalize_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one legacy or repaired benchmark summary row."""

    protocol = str(_first(row, "protocol", "source_protocol"))
    benchmark = str(_first(row, "benchmark", "dataset"))
    budget = str(_first(row, "design_budget", "budget"))
    alpha = _as_float(_first(row, "target_alpha", "alpha"))
    method = _canonical_method(str(_first(row, "method", "arm")))
    calls = _as_float(_first(row, "mean_calls", "test_calls", "calls_mean"))
    risk = _as_float(
        _first(
            row,
            "reported_risk_mean",
            "standard_all_miss_risk_mean",
            "test_risk",
            "risk_mean",
        )
    )
    baseline_calls = _as_float(
        _first(
            row,
            "baseline_mean_calls",
            "marginal_calls",
            "marginal_test_calls_mean",
        )
    )
    baseline_risk = _as_float(
        _first(
            row,
            "baseline_reported_risk_mean",
            "marginal_risk",
            "marginal_test_risk_mean",
        )
    )
    if not protocol or not benchmark or not budget or alpha is None:
        raise ValueError(f"row lacks canonical identity fields: {row}")
    return {
        "protocol": protocol,
        "benchmark": benchmark,
        "design_budget": budget,
        "target_alpha": alpha,
        "method": method,
        "mean_calls": calls,
        "reported_risk_mean": risk,
        "baseline_mean_calls": baseline_calls,
        "baseline_reported_risk_mean": baseline_risk,
    }


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [normalize_csv_row(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"no benchmark-level rows found in {path}")
    return rows


def _axis_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lower = min(values)
    upper = max(values)
    span = max(upper - lower, 0.01)
    return max(0.0, lower - 0.04 * span), upper + 0.04 * span


def _group_rows(rows: list[dict[str, Any]]):
    curves = collections.defaultdict(lambda: collections.defaultdict(dict))
    baselines = collections.defaultdict(dict)
    for row in rows:
        key = (row["protocol"], row["benchmark"], row["design_budget"])
        alpha = float(row["target_alpha"])
        if row["mean_calls"] is not None:
            curves[key][row["method"]][alpha] = float(row["mean_calls"])
        if row["baseline_mean_calls"] is not None:
            baselines[key][alpha] = float(row["baseline_mean_calls"])
    return curves, baselines


def _panel_metadata(key, curves, baseline) -> dict[str, Any]:
    alpha_values = sorted(
        set(baseline).union(*(set(curve) for curve in curves.values()))
    )
    axis_min, axis_max = _axis_range(alpha_values)
    return {
        "protocol": key[0],
        "benchmark": key[1],
        "design_budget": key[2],
        "axis_x_min": axis_min,
        "axis_x_max": axis_max,
        "baseline_points": [
            {"target_alpha": alpha, "mean_calls": baseline[alpha]}
            for alpha in sorted(baseline)
        ],
        "method_points": {
            method: [
                {"target_alpha": alpha, "mean_calls": value}
                for alpha, value in sorted(curve.items())
            ]
            for method, curve in sorted(curves.items())
        },
    }


def render_protocol(rows: list[dict[str, Any]], protocol: str, output_prefix: str | Path) -> Path:
    curves, baselines = _group_rows(rows)
    keys = sorted(key for key in set(curves).union(baselines) if key[0] == protocol)
    if not keys:
        raise ValueError(f"no rows for protocol {protocol}")

    ncols = 3
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows), squeeze=False)
    for ax in axes.flat[len(keys) :]:
        ax.axis("off")

    panels = []
    all_methods = sorted({method for key in keys for method in curves[key]})
    colors = {method: TOL_COLORS[index % len(TOL_COLORS)] for index, method in enumerate(all_methods)}
    for ax, key in zip(axes.flat, keys):
        panel = _panel_metadata(key, curves[key], baselines[key])
        panels.append(panel)
        for method, curve in sorted(curves[key].items()):
            points = sorted(curve.items())
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                "-o",
                markersize=2.5,
                linewidth=1.0,
                alpha=0.85,
                color=colors[method],
                label=METHOD_LABELS.get(method, method.replace("_", " ")),
            )
        baseline_points = sorted(baselines[key].items())
        if baseline_points:
            ax.plot(
                [point[0] for point in baseline_points],
                [point[1] for point in baseline_points],
                "-o",
                color="#CC3311",
                linewidth=2.6,
                markersize=3.5,
                label="Marginal ordering",
                zorder=5,
            )
        certified_count = len(baseline_points)
        if certified_count < 2:
            message = (
                "No certified baseline point"
                if certified_count == 0
                else "One certified baseline point"
            )
            ax.text(
                0.98,
                0.95,
                message,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="dimgray",
            )
        ax.set_xlim(panel["axis_x_min"], panel["axis_x_max"])
        ax.set_title(TITLES.get(key, "/".join(key)), fontsize=9)
        ax.set_xlabel("Target level alpha", fontsize=8)
        ax.set_ylabel("Mean test calls", fontsize=8)
        ax.tick_params(labelsize=7)

    handles: dict[str, Any] = {}
    for ax in axes.flat[: len(keys)]:
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        handles.update(zip(panel_labels, panel_handles))
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        ncol=5,
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output = Path(f"{output_prefix}_{protocol}.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "schema_version": 1,
        "figure_kind": "benchmark_level_sweep",
        "protocol": protocol,
        "panels": panels,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True, help="output path prefix")
    arguments = parser.parse_args()

    rows = load_rows(arguments.csv)
    for protocol in sorted({row["protocol"] for row in rows}):
        output = render_protocol(rows, protocol, arguments.out)
        print(f"wrote {output}, {output.with_suffix('.png')}, and {output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
