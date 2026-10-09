"""Regenerate query-level tables and calls-versus-risk figures.

Both the packaged legacy records and repaired records are accepted. Output always
uses the canonical columns in ``TABLE_COLUMNS`` and the canonical method names in
``METHOD_SPECS``. Repaired records report standard all-miss risk as the primary
paper objective, with the published augmented-null loss and legacy safe-action
fields retained as diagnostics. Legacy records retain their original reported
risk and explicitly mark unavailable semantics instead of silently relabeling it.

The two-standard-error classification is implemented only by
``risk_above_target``. If repetition-level risk values are unavailable, the
classification is unavailable rather than reconstructed from an old verdict.

Alongside PDF and PNG output, the script writes ``<fig-out>.json``. That file
records every plotted point and each panel's axis range so
``check_paper_numbers.py`` can verify that all RACER baseline points are visible.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METHOD_SPECS = OrderedDict(
    [
        ("racer_g", {"label": "RACER-G", "color": "#CC3311"}),
        ("racer_p", {"label": "RACER-P", "color": "#EE7733"}),
        ("exact_independence", {"label": "Exact independence", "color": "#0077BB"}),
        ("local_empirical", {"label": "Local empirical", "color": "#009988"}),
        (
            "local_order_factorized",
            {"label": "Local order + factorized sizing", "color": "#CCBB44"},
        ),
        (
            "fixed_tally_factorized",
            {"label": "Fixed-tally factorized sizing", "color": "#33BBEE"},
        ),
        (
            "fixed_design_marginal_factorized",
            {"label": "Fixed design marginal + factorized sizing", "color": "#000000"},
        ),
        ("pairwise_heuristic", {"label": "Pairwise heuristic", "color": "#EE3377"}),
        (
            "correlation_heuristic",
            {"label": "Correlation heuristic", "color": "#AA4499"},
        ),
        (
            "regularized_correlation",
            {"label": "Regularized correlation", "color": "#BBBBBB"},
        ),
    ]
)

TABLE_COLUMNS = [
    "pool_size",
    "target_alpha",
    "method",
    "method_label",
    "mean_calls",
    "reported_risk_name",
    "reported_risk_mean",
    "reported_risk_se",
    "reported_event_count",
    "reported_trial_count",
    "event_count_status",
    "risk_classification",
    "above_target_2se",
    "safe_action_risk_mean",
    "safe_action_event_count",
    "standard_all_miss_risk_mean",
    "standard_all_miss_event_count",
    "augmented_loss_mean",
    "augmented_loss_event_count",
    "certification_rate",
    "non_certification_count",
    "redraw_count",
    "source_schema",
]


def risk_above_target(mean: float, alpha: float, standard_error: float) -> bool:
    """Return the sole paper risk flag: mean exceeds alpha plus two SE."""

    values = (float(mean), float(alpha), float(standard_error))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("risk classification requires finite mean, alpha, and SE")
    if standard_error < 0.0:
        raise ValueError("standard error must be nonnegative")
    return mean > alpha + 2.0 * standard_error


def canonical_method_name(raw_name: str) -> str | None:
    """Map old and repaired result arm names to one public vocabulary."""

    name = str(raw_name).strip()
    lowered = name.lower()
    if lowered.startswith("wrapper_"):
        lowered = lowered[len("wrapper_") :]
    if lowered.endswith("_tuned"):
        lowered = lowered[: -len("_tuned")]

    exact = {
        "racer_g": "racer_g",
        "racer_marginal_g": "racer_g",
        "racer_p": "racer_p",
        "racer_marginal_p": "racer_p",
        "factorized_independence": "exact_independence",
        "exact_independence": "exact_independence",
        "independence": "exact_independence",
        "local_empirical": "local_empirical",
        "local_empirical_coverage": "local_empirical",
        "local_empirical_joint": "local_empirical",
        "local_order_factorized": "local_order_factorized",
        "local_order_factorized_sizing": "local_order_factorized",
        "local_empirical_order_factorized": "local_order_factorized",
        "fixed_tally_factorized": "fixed_tally_factorized",
        "fixed_tally_factorized_sizing": "fixed_tally_factorized",
        "fixed_empirical_tally": "fixed_tally_factorized",
        "empirical_tally_factorized": "fixed_tally_factorized",
        "fixed_design_marginal_factorized": "fixed_design_marginal_factorized",
        "fixed_design_marginal_factorized_sizing": "fixed_design_marginal_factorized",
        "fixed_marginal_factorized": "fixed_design_marginal_factorized",
        "pairwise_count": "pairwise_heuristic",
        "pairwise_heuristic": "pairwise_heuristic",
        "error_correlation": "correlation_heuristic",
        "correlation_heuristic": "correlation_heuristic",
        "gaussian_copula": "regularized_correlation",
        "regularized_correlation": "regularized_correlation",
    }
    if lowered in exact:
        return exact[lowered]
    if "local" in lowered and "order" in lowered and "factor" in lowered:
        return "local_order_factorized"
    if "fixed" in lowered and "design" in lowered and "marginal" in lowered:
        return "fixed_design_marginal_factorized"
    if "local" in lowered and "empirical" in lowered:
        return "local_empirical"
    if "fixed" in lowered and "tally" in lowered:
        return "fixed_tally_factorized"
    return None


def _alpha_key(mapping: dict[str, Any], alpha: float | str) -> str | None:
    wanted = float(alpha)
    for key in mapping:
        try:
            if math.isclose(float(key), wanted, rel_tol=0.0, abs_tol=1e-12):
                return key
        except (TypeError, ValueError):
            continue
    return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_number(mapping: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = _as_float(mapping.get(name))
        if value is not None:
            return value
    return None


def _series(info: dict[str, Any], details: list[dict[str, Any]], names: Iterable[str]) -> list[float]:
    for name in names:
        values = info.get(name)
        if isinstance(values, list):
            converted = [_as_float(value) for value in values]
            return [value for value in converted if value is not None]
    for name in names:
        converted = [_as_float(detail.get(name)) for detail in details]
        found = [value for value in converted if value is not None]
        if found:
            return found
    return []


def _standard_error(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _legacy_test_count_from_rate_grid(record: dict[str, Any]) -> int | None:
    """Recover an old record's exact test denominator from its risk lattice."""

    rates: list[float] = []
    summary = record.get("summary", {})
    if isinstance(summary, dict):
        for alpha_block in summary.values():
            if not isinstance(alpha_block, dict):
                continue
            arms = alpha_block.get("arms", {})
            if not isinstance(arms, dict):
                continue
            for info in arms.values():
                if not isinstance(info, dict):
                    continue
                values = info.get("per_redraw_risk")
                if isinstance(values, list):
                    rates.extend(
                        value
                        for value in (_as_float(item) for item in values)
                        if value is not None and value > 0.0
                    )
    n_queries = record.get("n_queries")
    if not rates or not isinstance(n_queries, int):
        return None
    for denominator in range(1, n_queries + 1):
        if all(
            abs(rate * denominator - round(rate * denominator))
            <= max(1e-8, 1e-8 * denominator)
            for rate in rates
        ):
            return denominator
    return None


def _infer_test_count(record: dict[str, Any]) -> int | None:
    n_queries = record.get("n_queries")
    fractions = record.get("partition_fractions")
    if isinstance(n_queries, int) and isinstance(fractions, dict):
        design = _as_float(fractions.get("design"))
        router = _as_float(fractions.get("router"))
        calibration = _first_number(fractions, ("final_calibration", "calibration", "cal"))
        validation = _as_float(fractions.get("validation"))
        if None not in (design, router, calibration, validation):
            n_design = int(round(n_queries * design))
            n_router = int(round(n_queries * router))
            n_pool = n_queries - n_design - n_router
            remaining = 1.0 - design - router
            test_fraction = remaining - calibration - validation
            denominator = calibration + validation + test_fraction
            if denominator > 0.0:
                n_cal = int(round(n_pool * calibration / denominator))
                n_val = int(round(n_pool * validation / denominator))
                n_test = n_pool - n_cal - n_val
                if n_test > 0:
                    return n_test
    return _legacy_test_count_from_rate_grid(record)


def _event_count(mean: float | None, trials: int | None) -> int | None:
    if mean is None or trials is None:
        return None
    estimate = mean * trials
    rounded = int(round(estimate))
    tolerance = max(1e-7, 1e-7 * trials)
    return rounded if abs(estimate - rounded) <= tolerance else None


def _details_for(
    record: dict[str, Any], alpha: float, source_name: str, canonical_name: str
) -> list[dict[str, Any]]:
    per_redraw = record.get("per_redraw")
    if not isinstance(per_redraw, dict):
        return []
    key = _alpha_key(per_redraw, alpha)
    if key is None or not isinstance(per_redraw[key], dict):
        return []
    block = per_redraw[key]
    if isinstance(block.get(source_name), list):
        return [value for value in block[source_name] if isinstance(value, dict)]
    for raw_name, values in block.items():
        if canonical_method_name(raw_name) == canonical_name and isinstance(values, list):
            return [value for value in values if isinstance(value, dict)]
    return []


def _arm_blocks(record: dict[str, Any], alpha: float) -> list[tuple[str, dict[str, Any]]]:
    summary = record.get("summary", {})
    key = _alpha_key(summary, alpha) if isinstance(summary, dict) else None
    block = summary.get(key, {}) if key is not None else {}
    result: dict[str, dict[str, Any]] = {}
    if isinstance(block, dict):
        for section_name in ("controls", "arms"):
            section = block.get(section_name, {})
            if not isinstance(section, dict):
                continue
            for raw_name, info in section.items():
                canonical = canonical_method_name(raw_name)
                if canonical is None or not isinstance(info, dict):
                    continue
                is_reported_arm = (
                    canonical.startswith("racer_")
                    or raw_name.endswith("_tuned")
                    or "::" not in raw_name
                )
                if is_reported_arm:
                    result[raw_name] = info
    tuned = record.get("per_variant_tuned", {})
    tuned_key = _alpha_key(tuned, alpha) if isinstance(tuned, dict) else None
    if tuned_key is not None and isinstance(tuned[tuned_key], dict):
        for raw_name, info in tuned[tuned_key].items():
            if canonical_method_name(raw_name) is not None and isinstance(info, dict):
                result[raw_name] = info
    return list(result.items())


def _metric_mean(
    info: dict[str, Any], details: list[dict[str, Any]], names: Iterable[str]
) -> float | None:
    direct = _first_number(info, names)
    if direct is not None:
        return direct
    values = _series(info, details, names)
    return statistics.fmean(values) if values else None


def rows_for(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize one old or repaired query-level record into canonical rows."""

    pool_size = int(record["pool_size"])
    alphas = record.get("alphas", list(record.get("summary", {})))
    test_count = _infer_test_count(record)
    output: dict[tuple[float, str], dict[str, Any]] = {}

    for alpha_value in alphas:
        alpha = float(alpha_value)
        for source_name, info in _arm_blocks(record, alpha):
            method = canonical_method_name(source_name)
            if method is None:
                continue
            details = _details_for(record, alpha, source_name, method)
            calls = _metric_mean(info, details, ("calls_mean", "calls_test_mean", "calls_test", "calls"))
            if calls is None:
                continue

            safe_mean = _metric_mean(
                info, details, ("safe_action_risk_mean", "safe_action_risk")
            )
            standard_mean = _metric_mean(
                info,
                details,
                ("standard_all_miss_risk_mean", "standard_all_miss_risk"),
            )
            augmented_mean = _metric_mean(
                info, details, ("augmented_loss_mean", "augmented_loss")
            )
            legacy_mean = _metric_mean(
                info, details, ("risk_mean", "risk_test_mean", "risk_test", "risk")
            )
            repaired = standard_mean is not None or bool(details)
            if standard_mean is not None:
                reported_name = "standard_all_miss_risk"
                reported_mean = standard_mean
            elif safe_mean is not None:
                reported_name = "safe_action_risk"
                reported_mean = safe_mean
            else:
                reported_name = "legacy_reported_risk"
                reported_mean = legacy_mean
            reported_series = _series(
                info,
                details,
                (
                    "per_redraw_standard_all_miss_risk",
                    "standard_all_miss_risk",
                    "per_redraw_risk",
                    "safe_action_risk",
                    "risk_test",
                    "risk",
                ),
            )
            reported_se = _standard_error(reported_series)
            redraw_count = len(reported_series) or len(details) or int(record.get("redraws", 0))
            total_trials = test_count * redraw_count if test_count and redraw_count else None

            counts = {
                "reported": _event_count(reported_mean, total_trials),
                "safe": _event_count(safe_mean, total_trials),
                "standard": _event_count(standard_mean, total_trials),
                "augmented": _event_count(augmented_mean, total_trials),
            }
            count_status = "available" if counts["reported"] is not None else "unavailable_in_source"
            if reported_mean is None or reported_se is None:
                classification = "unavailable"
                above: bool | str = ""
            else:
                above = risk_above_target(reported_mean, alpha, reported_se)
                classification = "above_target_2se" if above else "within_2se"

            certification_rate = _metric_mean(
                info, details, ("certification_rate", "certified")
            )
            non_certification_count = None
            if details:
                flags = [
                    bool(detail.get("non_certification", not detail.get("certified", True)))
                    for detail in details
                ]
                non_certification_count = sum(flags)
            elif certification_rate is not None and redraw_count:
                non_certification_count = int(round((1.0 - certification_rate) * redraw_count))

            output[(alpha, method)] = {
                "pool_size": pool_size,
                "target_alpha": alpha,
                "method": method,
                "method_label": METHOD_SPECS[method]["label"],
                "mean_calls": calls,
                "reported_risk_name": reported_name,
                "reported_risk_mean": reported_mean,
                "reported_risk_se": reported_se,
                "reported_event_count": counts["reported"],
                "reported_trial_count": total_trials if counts["reported"] is not None else None,
                "event_count_status": count_status,
                "risk_classification": classification,
                "above_target_2se": above,
                "safe_action_risk_mean": safe_mean,
                "safe_action_event_count": counts["safe"],
                "standard_all_miss_risk_mean": standard_mean,
                "standard_all_miss_event_count": counts["standard"],
                "augmented_loss_mean": augmented_mean,
                "augmented_loss_event_count": counts["augmented"],
                "certification_rate": certification_rate,
                "non_certification_count": non_certification_count,
                "redraw_count": redraw_count,
                "source_schema": "repaired" if repaired else "legacy",
            }

    method_order = {name: index for index, name in enumerate(METHOD_SPECS)}
    return [
        output[key]
        for key in sorted(output, key=lambda value: (value[0], method_order[value[1]]))
    ]


def load_records(records_directory: str | Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(Path(records_directory).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "pool_size" in record and "summary" in record:
            records[int(record["pool_size"])] = record
    if not records:
        raise ValueError(f"no query-level JSON records found under {records_directory}")
    return records


def write_table(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty query-level table")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _axis_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lower = min(0.0, min(values))
    upper = max(values)
    span = max(upper - lower, 1.0)
    return lower, upper + 0.04 * span


def figure_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    panels = []
    for pool_size in sorted({int(row["pool_size"]) for row in rows}):
        panel_rows = [row for row in rows if int(row["pool_size"]) == pool_size]
        calls = [float(row["mean_calls"]) for row in panel_rows]
        axis_min, axis_max = _axis_range(calls)
        baseline_calls = [
            float(row["mean_calls"])
            for row in panel_rows
            if row["method"] in {"racer_g", "racer_p"}
        ]
        panels.append(
            {
                "pool_size": pool_size,
                "axis_x_min": axis_min,
                "axis_x_max": axis_max,
                "data_x_min": min(calls),
                "data_x_max": max(calls),
                "baseline_calls": baseline_calls,
                "methods": sorted({row["method"] for row in panel_rows}),
                "points": [
                    {
                        "method": row["method"],
                        "target_alpha": float(row["target_alpha"]),
                        "mean_calls": float(row["mean_calls"]),
                        "reported_risk_mean": _as_float(row["reported_risk_mean"]),
                    }
                    for row in panel_rows
                ],
            }
        )
    return {"schema_version": 1, "figure_kind": "calls_vs_risk", "panels": panels}


def render_figure(rows: list[dict[str, Any]], output_prefix: str | Path) -> Path:
    pools = sorted({int(row["pool_size"]) for row in rows})
    ncols = 2
    nrows = max(1, math.ceil(len(pools) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows), squeeze=False)
    metadata = figure_metadata(rows)
    metadata_by_pool = {panel["pool_size"]: panel for panel in metadata["panels"]}

    for ax in axes.flat[len(pools) :]:
        ax.axis("off")
    for ax, pool_size in zip(axes.flat, pools):
        panel_rows = [row for row in rows if int(row["pool_size"]) == pool_size]
        for method, spec in METHOD_SPECS.items():
            points = sorted(
                (
                    (float(row["mean_calls"]), float(row["reported_risk_mean"]))
                    for row in panel_rows
                    if row["method"] == method and row["reported_risk_mean"] is not None
                ),
                key=lambda point: point[0],
            )
            if not points:
                continue
            linewidth = 2.2 if method in {"racer_g", "racer_p"} else 1.4
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                "-o",
                color=spec["color"],
                markersize=3,
                linewidth=linewidth,
                label=spec["label"],
            )
        for alpha in (0.02, 0.05, 0.10):
            ax.axhline(alpha, color="gray", linestyle=":", linewidth=0.8)
        panel_metadata = metadata_by_pool[pool_size]
        ax.set_xlim(panel_metadata["axis_x_min"], panel_metadata["axis_x_max"])
        ax.set_title(f"Instruction following (IFEval), pool = {pool_size}", fontsize=10)
        ax.set_xlabel("Mean test calls", fontsize=9)
        ax.set_ylabel("Reported test risk", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output = Path(output_prefix).with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    Path(output_prefix).with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--table-out", required=True)
    parser.add_argument("--fig-out", required=True)
    arguments = parser.parse_args()

    records = load_records(arguments.records)
    rows = [row for record in records.values() for row in rows_for(record)]
    write_table(rows, arguments.table_out)
    figure_path = render_figure(rows, arguments.fig_out)
    print(f"wrote {arguments.table_out} ({len(rows)} rows)")
    print(f"wrote {figure_path}, {figure_path.with_suffix('.png')}, and {Path(arguments.fig_out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
