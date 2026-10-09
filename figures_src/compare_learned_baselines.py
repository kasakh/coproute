"""Create common-metric learned-baseline reports without inventing pairings.

The learned summary and fixed-tally tables were produced by different
protocols. This module places their available metrics in common tables, but it
labels cross-protocol contrasts as descriptive and nonpaired. Missing
fixed-depth RACER-G and exact-independence curves are emitted as explicit
unavailable rows rather than reconstructed from calibrated operating points.

The primary command writes deterministic CSV and JSON artifacts plus PDF and
PNG views for fixed-depth Success@k, calibrated calls/risk/certification,
learned-to-fixed-tally call ratios, native FlexRouter adaptive stopping, and the
optional native BaRP sensitivity grid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCHEMA_VERSION = "learned_baseline_common_metrics_v1"
LEARNED_PROTOCOL = "preregistered_learned_baselines_v1"
QUERY_PROTOCOL = "repaired_e2_query_level_standard_loss"
E1_DEPTH_PROTOCOL = "legacy_e1_benchmark_level_fixed_prefix"
E2_DEPTH_PROTOCOL = "legacy_e2_query_level_fixed_prefix"
EXPECTED_TRAINING_SEEDS = 5
PRIMARY_ALPHAS = (0.05, 0.10)
LEARNED_DEPTH_CAP = 10

METHOD_LABELS = {
    "flexrouter": "FlexRouter",
    "barp": "BaRP",
    "fixed_tally": "Fixed tally",
    "fixed_tally_factorized": "Fixed tally",
    "exact_independence": "Exact independence",
    "racer_g": "RACER-G",
}
METHOD_ORDER = {
    method: index
    for index, method in enumerate(
        ("flexrouter", "barp", "fixed_tally", "fixed_tally_factorized", "exact_independence", "racer_g")
    )
}
COLORS = {
    "flexrouter": "#0077BB",
    "barp": "#EE7733",
    "fixed_tally": "#009988",
    "fixed_tally_factorized": "#009988",
    "exact_independence": "#CC3311",
    "racer_g": "#AA4499",
}

FIXED_DEPTH_COLUMNS = (
    "section",
    "availability_status",
    "source_protocol",
    "source_phase",
    "pairing_status",
    "method",
    "method_label",
    "analysis_label",
    "target_task",
    "pool_size",
    "pool_seed",
    "depth",
    "depth_cap",
    "depth_cap_label",
    "success_mean",
    "success_se",
    "all_miss_risk_mean",
    "all_miss_risk_se",
    "training_seed_count",
    "redraw_count",
    "observation_count",
    "comparison_note",
)

CALIBRATED_COLUMNS = (
    "section",
    "availability_status",
    "source_protocol",
    "pairing_status",
    "method",
    "method_label",
    "analysis_label",
    "target_task",
    "pool_size",
    "pool_seed",
    "target_alpha",
    "mean_calls",
    "mean_calls_se",
    "standard_all_miss_risk_mean",
    "standard_all_miss_risk_se",
    "risk_classification",
    "above_target_2se",
    "certification_rate",
    "certification_count",
    "certification_denominator",
    "certification_unit",
    "non_certification_count",
    "training_seed_count",
    "redraw_count",
    "observation_count",
    "reported_risk_name",
    "comparison_note",
)

RATIO_COLUMNS = (
    "section",
    "availability_status",
    "pairing_status",
    "learned_source_protocol",
    "reference_source_protocol",
    "learned_method",
    "learned_method_label",
    "analysis_label",
    "target_task",
    "pool_size",
    "pool_seed",
    "target_alpha",
    "learned_mean_calls",
    "fixed_tally_mean_calls",
    "learned_to_fixed_tally_call_ratio",
    "call_reduction_vs_fixed_tally",
    "learned_certification_rate",
    "fixed_tally_certification_rate",
    "comparison_note",
)

FLEX_COLUMNS = (
    "section",
    "availability_status",
    "source_protocol",
    "pairing_status",
    "method",
    "method_label",
    "analysis_label",
    "target_task",
    "pool_size",
    "pool_seed",
    "tau",
    "depth_cap",
    "depth_cap_label",
    "success_mean",
    "success_se",
    "standard_all_miss_risk_mean",
    "standard_all_miss_risk_se",
    "mean_calls",
    "mean_calls_se",
    "median_calls_mean",
    "p90_calls_mean",
    "training_seed_count",
    "redraw_count",
    "observation_count",
    "risk_certificate_claimed",
    "comparison_note",
)

BARP_COLUMNS = (
    "section",
    "availability_status",
    "source_protocol",
    "pairing_status",
    "method",
    "method_label",
    "analysis_label",
    "training_tasks",
    "target_task",
    "w_quality",
    "w_cost",
    "mean_official_quality",
    "mean_official_quality_se",
    "mean_cost",
    "mean_cost_se",
    "mean_scalar_reward",
    "binary_coverage",
    "nondominated_rate",
    "training_seed_count",
    "observation_count",
    "comparison_note",
)


class CommonMetricError(ValueError):
    """Raised when an input cannot support an auditable common-metric report."""


def _finite(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _finite(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    left_number = _finite(left)
    right_number = _finite(right)
    return (
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=tolerance)
    )


def _method_sort(method: str) -> tuple[int, str]:
    return METHOD_ORDER.get(method, len(METHOD_ORDER)), method


def load_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CommonMetricError(f"CSV has no header: {source}")
        return [dict(row) for row in reader]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(rows: Sequence[Mapping[str, Any]], required: Iterable[str], name: str) -> None:
    if not rows:
        raise CommonMetricError(f"{name} is empty")
    missing = sorted(set(required) - set(rows[0]))
    if missing:
        raise CommonMetricError(f"{name} is missing columns: {', '.join(missing)}")


def _is_available(row: Mapping[str, Any]) -> bool:
    return _text(row.get("run_status") or "available").lower() == "available"


def _is_complete_learned(row: Mapping[str, Any]) -> bool:
    return _integer(row.get("training_seed_count")) == EXPECTED_TRAINING_SEEDS


def _learned_exclusion(row: Mapping[str, Any], source_name: str) -> dict[str, Any]:
    return {
        "source": source_name,
        "block": _text(row.get("block")),
        "method": _text(row.get("method")),
        "target_task": _text(row.get("target_task")),
        "pool_size": _integer(row.get("pool_size")),
        "training_seed_count": _integer(row.get("training_seed_count")),
        "reason": "incomplete_training_seed_aggregate",
        "required_training_seed_count": EXPECTED_TRAINING_SEEDS,
    }


def _classify_risk(mean: Any, alpha: Any, standard_error: Any) -> tuple[str, bool | str]:
    risk = _finite(mean)
    target = _finite(alpha)
    se = _finite(standard_error)
    if risk is None or target is None or se is None or se < 0.0:
        return "unavailable", ""
    above = risk > target + 2.0 * se
    return ("above_target_2se" if above else "within_2se"), above


def _normalize_certification_counts(
    redraw_count: int | None,
    non_certification_count: int | None,
    certification_rate: float | None,
) -> tuple[int | None, int | None]:
    """Resolve source schemas where redraw_count means total or certified rows."""
    if redraw_count is None or non_certification_count is None:
        return None, redraw_count
    candidates: list[tuple[int, int]] = []
    if non_certification_count <= redraw_count:
        candidates.append((redraw_count - non_certification_count, redraw_count))
    candidates.append((redraw_count, redraw_count + non_certification_count))
    unique = list(dict.fromkeys(candidates))
    if certification_rate is None:
        if non_certification_count == 0:
            return unique[0]
        raise CommonMetricError("certification rate is required to resolve redraw-count semantics")
    matches = [
        candidate
        for candidate in unique
        if candidate[1] > 0
        and math.isclose(candidate[0] / candidate[1], certification_rate, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise CommonMetricError(
            "certification count, non-certification count, and rate are inconsistent"
        )
    return matches[0]


def fixed_depth_rows(
    learned_rows: Sequence[Mapping[str, Any]],
    e1_rows: Sequence[Mapping[str, Any]],
    e2_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Normalize all genuinely available fixed-depth curves.

    E1 empirical-tally rows provide the fixed-tally ordering curve. Neither
    supplied fixed-depth file contains exact-independence or RACER-G curves, so
    those methods receive explicit unavailable entries.
    """

    _require_columns(
        learned_rows,
        (
            "block",
            "method",
            "target_task",
            "pool_size",
            "depth",
            "training_seed_count",
            "success_mean",
            "standard_all_miss_risk_mean",
        ),
        "learned summary",
    )
    _require_columns(
        e1_rows,
        ("source", "dataset", "method", "phase", "depth", "mean_miss_rate"),
        "E1 fixed-depth table",
    )
    _require_columns(
        e2_rows,
        ("source", "dataset", "method", "phase", "depth", "mean_miss_rate"),
        "E2 fixed-depth table",
    )

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for source in learned_rows:
        if _text(source.get("block")) != "native_fixed_depth" or not _is_available(source):
            continue
        method = _text(source.get("method")).lower()
        if method not in ("flexrouter", "barp"):
            continue
        if not _is_complete_learned(source):
            exclusions.append(_learned_exclusion(source, "learned_summary"))
            continue
        depth = _integer(source.get("depth"))
        if depth is None or not 1 <= depth <= LEARNED_DEPTH_CAP:
            raise CommonMetricError("learned native fixed-depth rows must have depth 1 through 10")
        rows.append(
            {
                "section": "fixed_depth_success",
                "availability_status": "available",
                "source_protocol": f"{LEARNED_PROTOCOL}:native_fixed_depth",
                "source_phase": "test",
                "pairing_status": "paired_within_learned_protocol",
                "method": method,
                "method_label": METHOD_LABELS[method],
                "analysis_label": _text(source.get("analysis_label")),
                "target_task": _text(source.get("target_task")).lower(),
                "pool_size": _integer(source.get("pool_size")),
                "pool_seed": _integer(source.get("pool_seed")),
                "depth": depth,
                "depth_cap": LEARNED_DEPTH_CAP,
                "depth_cap_label": "registered maximum set size 10",
                "success_mean": _finite(source.get("success_mean")),
                "success_se": _finite(source.get("success_se")),
                "all_miss_risk_mean": _finite(source.get("standard_all_miss_risk_mean")),
                "all_miss_risk_se": _finite(source.get("standard_all_miss_risk_se")),
                "training_seed_count": EXPECTED_TRAINING_SEEDS,
                "redraw_count": _integer(source.get("redraw_count")),
                "observation_count": _integer(source.get("observation_count")),
                "comparison_note": "Five-seed learned aggregate. Cross-protocol overlays are descriptive nonpaired comparisons.",
            }
        )

    learned_tasks = {
        row["target_task"] for row in rows if row["method"] in ("flexrouter", "barp")
    }
    fixed_tasks: set[str] = set()
    for source in e1_rows:
        if (
            _text(source.get("source")).upper() != "RE"
            or _text(source.get("phase")).lower() != "test"
            or _text(source.get("method")).lower() != "empirical_tally"
        ):
            continue
        task = _text(source.get("dataset")).lower()
        if task not in learned_tasks:
            continue
        depth = _integer(source.get("depth"))
        risk = _finite(source.get("mean_miss_rate"))
        if not task or depth is None or risk is None or depth > LEARNED_DEPTH_CAP:
            continue
        fixed_tasks.add(task)
        sd = _finite(source.get("sd_miss_rate"))
        rows.append(
            {
                "section": "fixed_depth_success",
                "availability_status": "available",
                "source_protocol": E1_DEPTH_PROTOCOL,
                "source_phase": "test",
                "pairing_status": "descriptive_nonpaired",
                "method": "fixed_tally",
                "method_label": METHOD_LABELS["fixed_tally"],
                "analysis_label": "empirical_tally_order",
                "target_task": task,
                "pool_size": 100,
                "pool_seed": None,
                "depth": depth,
                "depth_cap": None,
                "depth_cap_label": "available reference depths only",
                "success_mean": 1.0 - risk,
                "success_se": None,
                "all_miss_risk_mean": risk,
                "all_miss_risk_se": None,
                "training_seed_count": None,
                "redraw_count": None,
                "observation_count": None,
                "comparison_note": "The fixed-tally reference lacks shared pool, redraw, query, and seed identities. Comparison to learned curves is descriptive and unpaired.",
            }
        )

    unavailable_tasks = sorted(learned_tasks | fixed_tasks)
    e1_methods = sorted({_text(row.get("method")).lower() for row in e1_rows})
    e2_methods = sorted({_text(row.get("method")).lower() for row in e2_rows})
    for task in unavailable_tasks:
        for method in ("exact_independence", "racer_g"):
            rows.append(
                {
                    "section": "fixed_depth_success",
                    "availability_status": "unavailable",
                    "source_protocol": f"{E1_DEPTH_PROTOCOL}+{E2_DEPTH_PROTOCOL}",
                    "source_phase": "test",
                    "pairing_status": "unavailable",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "analysis_label": "",
                    "target_task": task,
                    "pool_size": 100,
                    "pool_seed": None,
                    "depth": None,
                    "depth_cap": None,
                    "depth_cap_label": "unavailable",
                    "success_mean": None,
                    "success_se": None,
                    "all_miss_risk_mean": None,
                    "all_miss_risk_se": None,
                    "training_seed_count": None,
                    "redraw_count": None,
                    "observation_count": None,
                    "comparison_note": "No fixed-depth curve for this method exists in either supplied fixed-depth CSV.",
                }
            )

    rows.sort(
        key=lambda row: (
            row["target_task"],
            row["pool_size"] if row["pool_size"] is not None else -1,
            _method_sort(row["method"]),
            row["depth"] if row["depth"] is not None else 10_000,
            row["source_protocol"],
        )
    )
    inventory = {
        "e1_source_protocol": E1_DEPTH_PROTOCOL,
        "e1_methods": e1_methods,
        "e2_source_protocol": E2_DEPTH_PROTOCOL,
        "e2_methods": e2_methods,
        "fixed_tally_curve_source": "E1 empirical_tally test curve",
        "exact_independence_fixed_depth": "unavailable",
        "racer_g_fixed_depth": "unavailable",
        "e2_requested_curve_contribution": "none; no requested fixed-depth comparator is present",
    }
    return rows, exclusions, inventory


def calibrated_rows(
    learned_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    *,
    query_task: str,
    primary_alphas: Sequence[float] = PRIMARY_ALPHAS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the primary calibrated table at registered target risks."""

    _require_columns(
        query_rows,
        (
            "pool_size",
            "target_alpha",
            "method",
            "mean_calls",
            "standard_all_miss_risk_mean",
            "certification_rate",
            "redraw_count",
        ),
        "repaired query table",
    )
    alpha_values = tuple(float(value) for value in primary_alphas)
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for source in learned_rows:
        if _text(source.get("block")) != "adapted_calibrated" or not _is_available(source):
            continue
        method = _text(source.get("method")).lower()
        alpha = _finite(source.get("alpha"))
        if method not in ("flexrouter", "barp") or not any(
            _same_number(alpha, wanted) for wanted in alpha_values
        ):
            continue
        if not _is_complete_learned(source):
            exclusions.append(_learned_exclusion(source, "learned_summary"))
            continue
        risk = _finite(source.get("standard_all_miss_risk_mean"))
        risk_se = _finite(source.get("standard_all_miss_risk_se"))
        classification, above = _classify_risk(risk, alpha, risk_se)
        observation_count = _integer(source.get("observation_count"))
        certification_count = _integer(source.get("certification_count"))
        non_certification_count = _integer(source.get("non_certification_count"))
        denominator = observation_count
        if denominator is None and certification_count is not None and non_certification_count is not None:
            denominator = certification_count + non_certification_count
        rows.append(
            {
                "section": "calibrated_primary",
                "availability_status": "available",
                "source_protocol": f"{LEARNED_PROTOCOL}:adapted_standard_loss_calibration",
                "pairing_status": "descriptive_nonpaired",
                "method": method,
                "method_label": METHOD_LABELS[method],
                "analysis_label": _text(source.get("analysis_label")),
                "target_task": _text(source.get("target_task")).lower(),
                "pool_size": _integer(source.get("pool_size")),
                "pool_seed": _integer(source.get("pool_seed")),
                "target_alpha": alpha,
                "mean_calls": _finite(source.get("mean_calls")),
                "mean_calls_se": _finite(source.get("mean_calls_se")),
                "standard_all_miss_risk_mean": risk,
                "standard_all_miss_risk_se": risk_se,
                "risk_classification": classification,
                "above_target_2se": above,
                "certification_rate": _finite(source.get("certification_rate")),
                "certification_count": certification_count,
                "certification_denominator": denominator,
                "certification_unit": "training_seed_redraw",
                "non_certification_count": non_certification_count,
                "training_seed_count": EXPECTED_TRAINING_SEEDS,
                "redraw_count": _integer(source.get("redraw_count")),
                "observation_count": observation_count,
                "reported_risk_name": "standard_all_miss_risk",
                "comparison_note": "Learned five-seed aggregate. Its 40-redraw protocol is not paired to the fixed-tally query-table aggregate.",
            }
        )

    requested_query_methods = {"racer_g", "exact_independence", "fixed_tally_factorized"}
    for source in query_rows:
        method = _text(source.get("method")).lower()
        alpha = _finite(source.get("target_alpha"))
        if method not in requested_query_methods or not any(
            _same_number(alpha, wanted) for wanted in alpha_values
        ):
            continue
        redraw_count = _integer(source.get("redraw_count"))
        non_certification_count = _integer(source.get("non_certification_count"))
        certification_rate = _finite(source.get("certification_rate"))
        certification_count, certification_denominator = _normalize_certification_counts(
            redraw_count, non_certification_count, certification_rate
        )
        above_text = _text(source.get("above_target_2se")).lower()
        if above_text in ("true", "1"):
            above: bool | str = True
        elif above_text in ("false", "0"):
            above = False
        else:
            _, above = _classify_risk(
                source.get("standard_all_miss_risk_mean"), alpha, source.get("reported_risk_se")
            )
        classification = _text(source.get("risk_classification")) or (
            "above_target_2se" if above is True else "within_2se" if above is False else "unavailable"
        )
        rows.append(
            {
                "section": "calibrated_primary",
                "availability_status": "available",
                "source_protocol": QUERY_PROTOCOL,
                "pairing_status": "descriptive_nonpaired",
                "method": method,
                "method_label": METHOD_LABELS[method],
                "analysis_label": "frozen_fixed_tally_arm",
                "target_task": query_task.lower(),
                "pool_size": _integer(source.get("pool_size")),
                "pool_seed": None,
                "target_alpha": alpha,
                "mean_calls": _finite(source.get("mean_calls")),
                "mean_calls_se": None,
                "standard_all_miss_risk_mean": _finite(source.get("standard_all_miss_risk_mean")),
                "standard_all_miss_risk_se": _finite(source.get("reported_risk_se")),
                "risk_classification": classification,
                "above_target_2se": above,
                "certification_rate": certification_rate,
                "certification_count": certification_count,
                "certification_denominator": certification_denominator,
                "certification_unit": "redraw",
                "non_certification_count": non_certification_count,
                "training_seed_count": None,
                "redraw_count": certification_denominator,
                "observation_count": None,
                "reported_risk_name": _text(source.get("reported_risk_name")) or "standard_all_miss_risk",
                "comparison_note": "The fixed-tally IFEval query-table aggregate lacks shared query, redraw, pool-seed, and task identities. Task is supplied explicitly; learned contrasts remain descriptive and unpaired.",
            }
        )

    rows.sort(
        key=lambda row: (
            row["target_task"],
            row["pool_size"] if row["pool_size"] is not None else -1,
            row["target_alpha"] if row["target_alpha"] is not None else -1.0,
            _method_sort(row["method"]),
            row["analysis_label"],
        )
    )
    return rows, exclusions


def learned_fixed_tally_ratios(
    calibrated: Sequence[Mapping[str, Any]], *, query_task: str
) -> list[dict[str, Any]]:
    references: dict[tuple[str, int, float], Mapping[str, Any]] = {}
    for row in calibrated:
        if row.get("method") != "fixed_tally_factorized":
            continue
        task = _text(row.get("target_task")).lower()
        pool = _integer(row.get("pool_size"))
        alpha = _finite(row.get("target_alpha"))
        if task and pool is not None and alpha is not None:
            references[(task, pool, alpha)] = row

    output: list[dict[str, Any]] = []
    for row in calibrated:
        method = _text(row.get("method"))
        if method not in ("flexrouter", "barp"):
            continue
        task = _text(row.get("target_task")).lower()
        pool = _integer(row.get("pool_size"))
        alpha = _finite(row.get("target_alpha"))
        reference = references.get((task, pool, alpha)) if pool is not None and alpha is not None else None
        learned_calls = _finite(row.get("mean_calls"))
        fixed_calls = _finite(reference.get("mean_calls")) if reference is not None else None
        available = learned_calls is not None and fixed_calls is not None and fixed_calls > 0.0
        ratio = learned_calls / fixed_calls if available else None
        output.append(
            {
                "section": "learned_vs_fixed_tally_call_ratios",
                "availability_status": "available" if available else "unavailable",
                "pairing_status": "descriptive_nonpaired" if available else "unavailable",
                "learned_source_protocol": row.get("source_protocol"),
                "reference_source_protocol": reference.get("source_protocol") if reference else QUERY_PROTOCOL,
                "learned_method": method,
                "learned_method_label": METHOD_LABELS[method],
                "analysis_label": _text(row.get("analysis_label")),
                "target_task": task,
                "pool_size": pool,
                "pool_seed": _integer(row.get("pool_seed")),
                "target_alpha": alpha,
                "learned_mean_calls": learned_calls,
                "fixed_tally_mean_calls": fixed_calls,
                "learned_to_fixed_tally_call_ratio": ratio,
                "call_reduction_vs_fixed_tally": 1.0 - ratio if ratio is not None else None,
                "learned_certification_rate": _finite(row.get("certification_rate")),
                "fixed_tally_certification_rate": _finite(reference.get("certification_rate")) if reference else None,
                "comparison_note": (
                    "Ratio aligns task, pool size, and alpha only. Source redraws and pool identities differ, so this is descriptive nonpaired."
                    if available
                    else f"No fixed-tally cell is available for {task}; the supplied query table is labeled {query_task.lower()}."
                ),
            }
        )
    output.sort(
        key=lambda row: (
            row["target_task"],
            row["pool_size"] if row["pool_size"] is not None else -1,
            row["target_alpha"] if row["target_alpha"] is not None else -1.0,
            _method_sort(row["learned_method"]),
        )
    )
    return output


def native_flex_rows(
    learned_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for source in learned_rows:
        if (
            _text(source.get("block")) != "native_adaptive"
            or _text(source.get("method")).lower() != "flexrouter"
            or not _is_available(source)
        ):
            continue
        if not _is_complete_learned(source):
            exclusions.append(_learned_exclusion(source, "learned_summary"))
            continue
        rows.append(
            {
                "section": "native_flex_adaptive_appendix",
                "availability_status": "available",
                "source_protocol": f"{LEARNED_PROTOCOL}:native_flex_adaptive",
                "pairing_status": "not_applicable",
                "method": "flexrouter",
                "method_label": METHOD_LABELS["flexrouter"],
                "analysis_label": _text(source.get("analysis_label")),
                "target_task": _text(source.get("target_task")).lower(),
                "pool_size": _integer(source.get("pool_size")),
                "pool_seed": _integer(source.get("pool_seed")),
                "tau": _finite(source.get("tau")),
                "depth_cap": LEARNED_DEPTH_CAP,
                "depth_cap_label": "registered maximum set size 10",
                "success_mean": _finite(source.get("success_mean")),
                "success_se": _finite(source.get("success_se")),
                "standard_all_miss_risk_mean": _finite(source.get("standard_all_miss_risk_mean")),
                "standard_all_miss_risk_se": _finite(source.get("standard_all_miss_risk_se")),
                "mean_calls": _finite(source.get("mean_calls")),
                "mean_calls_se": _finite(source.get("mean_calls_se")),
                "median_calls_mean": _finite(source.get("median_calls_mean")),
                "p90_calls_mean": _finite(source.get("p90_calls_mean")),
                "training_seed_count": EXPECTED_TRAINING_SEEDS,
                "redraw_count": _integer(source.get("redraw_count")),
                "observation_count": _integer(source.get("observation_count")),
                "risk_certificate_claimed": False,
                "comparison_note": "Native adaptive stopping is an appendix frontier and does not claim conformal certification.",
            }
        )
    rows.sort(
        key=lambda row: (
            row["target_task"],
            row["pool_size"] if row["pool_size"] is not None else -1,
            row["tau"] if row["tau"] is not None else -1.0,
        )
    )
    return rows, exclusions


def native_barp_rows(
    source_rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if source_rows is None:
        return [], []
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for source in source_rows:
        if _text(source.get("block")) != "native_barp_sensitivity" or not _is_available(source):
            continue
        if not _is_complete_learned(source):
            exclusions.append(_learned_exclusion(source, "native_barp_summary"))
            continue
        rows.append(
            {
                "section": "native_barp_sensitivity_appendix",
                "availability_status": "available",
                "source_protocol": f"{LEARNED_PROTOCOL}:native_routerbench_barp_sensitivity",
                "pairing_status": "not_applicable",
                "method": "barp",
                "method_label": METHOD_LABELS["barp"],
                "analysis_label": _text(source.get("analysis_label")),
                "training_tasks": _text(source.get("training_tasks")),
                "target_task": _text(source.get("target_task")).lower(),
                "w_quality": _finite(source.get("w_quality")),
                "w_cost": _finite(source.get("w_cost")),
                "mean_official_quality": _finite(source.get("mean_official_quality")),
                "mean_official_quality_se": _finite(source.get("mean_official_quality_se")),
                "mean_cost": _finite(source.get("mean_cost")),
                "mean_cost_se": _finite(source.get("mean_cost_se")),
                "mean_scalar_reward": _finite(source.get("mean_scalar_reward")),
                "binary_coverage": _finite(source.get("binary_coverage")),
                "nondominated_rate": _finite(source.get("nondominated_rate")),
                "training_seed_count": EXPECTED_TRAINING_SEEDS,
                "observation_count": _integer(source.get("observation_count")),
                "comparison_note": "Complete registered preference grid. No preference or reconstruction variant is selected using test outcomes.",
            }
        )
    rows.sort(
        key=lambda row: (
            row["target_task"],
            row["w_cost"] if row["w_cost"] is not None else -1.0,
            row["analysis_label"],
        )
    )
    return rows, exclusions


def build_report(
    learned_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    e1_rows: Sequence[Mapping[str, Any]],
    e2_rows: Sequence[Mapping[str, Any]],
    native_barp_summary_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    query_task: str = "ifeval",
    primary_alphas: Sequence[float] = PRIMARY_ALPHAS,
) -> dict[str, Any]:
    """Build every report block from already loaded CSV rows."""

    fixed, excluded_fixed, inventory = fixed_depth_rows(learned_rows, e1_rows, e2_rows)
    calibrated, excluded_calibrated = calibrated_rows(
        learned_rows, query_rows, query_task=query_task, primary_alphas=primary_alphas
    )
    ratios = learned_fixed_tally_ratios(calibrated, query_task=query_task)
    flex, excluded_flex = native_flex_rows(learned_rows)
    barp, excluded_barp = native_barp_rows(native_barp_summary_rows)
    exclusions = excluded_fixed + excluded_calibrated + excluded_flex + excluded_barp

    return {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": "all registered rows; no test-selected best variant",
        "expected_training_seed_count": EXPECTED_TRAINING_SEEDS,
        "primary_alphas": [float(value) for value in primary_alphas],
        "query_table_task_label": query_task.lower(),
        "cross_protocol_comparison": "descriptive_nonpaired",
        "fixed_depth_inventory": inventory,
        "optional_native_barp_status": "available" if barp else "not_supplied_or_no_complete_rows",
        "excluded_incomplete_learned_rows": exclusions,
        "blocks": {
            "fixed_depth_success": fixed,
            "calibrated_primary": calibrated,
            "learned_vs_fixed_tally_call_ratios": ratios,
            "native_flex_adaptive_appendix": flex,
            "native_barp_sensitivity_appendix": barp,
        },
        "row_counts": {
            "fixed_depth_success": len(fixed),
            "calibrated_primary": len(calibrated),
            "learned_vs_fixed_tally_call_ratios": len(ratios),
            "native_flex_adaptive_appendix": len(flex),
            "native_barp_sensitivity_appendix": len(barp),
        },
    }


def _refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: str | Path) -> None:
    destination = Path(path)
    _refuse_existing(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(value: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    _refuse_existing(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_figure(fig: Any, pdf_path: Path) -> tuple[str, str]:
    png_path = pdf_path.with_suffix(".png")
    _refuse_existing(pdf_path)
    _refuse_existing(png_path)
    temporary_pdf = pdf_path.with_name(pdf_path.stem + ".tmp.pdf")
    temporary_png = png_path.with_name(png_path.stem + ".tmp.png")
    try:
        fig.savefig(temporary_pdf, dpi=300, bbox_inches="tight")
        fig.savefig(temporary_png, dpi=200, bbox_inches="tight")
        os.replace(temporary_pdf, pdf_path)
        os.replace(temporary_png, png_path)
    finally:
        for temporary in (temporary_pdf, temporary_png):
            if temporary.exists():
                temporary.unlink()
        plt.close(fig)
    return str(pdf_path), str(png_path)


def render_fixed_depth(rows: Sequence[Mapping[str, Any]], pdf_path: str | Path) -> tuple[str, str]:
    available = [row for row in rows if row.get("availability_status") == "available"]
    learned = [row for row in available if row.get("method") in ("flexrouter", "barp")]
    cells = sorted(
        {(str(row["target_task"]), int(row["pool_size"])) for row in learned},
        key=lambda value: (value[0], value[1]),
    )
    if not cells:
        raise CommonMetricError("no fixed-depth learned curves to render")
    tasks = sorted({task for task, _ in cells})
    pools = sorted({pool for _, pool in cells})
    fig, axes = plt.subplots(
        len(tasks), len(pools), figsize=(3.4 * len(pools), 3.0 * len(tasks)), squeeze=False
    )
    for task_index, task in enumerate(tasks):
        for pool_index, pool in enumerate(pools):
            axis = axes[task_index][pool_index]
            for method in ("flexrouter", "barp"):
                curve = sorted(
                    (
                        row
                        for row in learned
                        if row["target_task"] == task
                        and row["pool_size"] == pool
                        and row["method"] == method
                    ),
                    key=lambda row: int(row["depth"]),
                )
                if curve:
                    axis.plot(
                        [row["depth"] for row in curve],
                        [row["success_mean"] for row in curve],
                        marker="o",
                        linewidth=1.7,
                        markersize=3,
                        color=COLORS[method],
                        label=METHOD_LABELS[method],
                    )
            if pool == 100:
                fixed = sorted(
                    (
                        row
                        for row in available
                        if row["target_task"] == task and row["method"] == "fixed_tally"
                    ),
                    key=lambda row: int(row["depth"]),
                )
                if fixed:
                    axis.plot(
                        [row["depth"] for row in fixed],
                        [row["success_mean"] for row in fixed],
                        linestyle=":",
                        marker="s",
                        linewidth=1.5,
                        markersize=3,
                        color=COLORS["fixed_tally"],
                        label="Fixed tally, descriptive nonpaired",
                    )
            axis.set_title(f"{task}, pool {pool}", fontsize=9)
            axis.set_xlabel("Calls k")
            axis.set_ylabel("Success@k")
            axis.set_xlim(1, LEARNED_DEPTH_CAP)
            axis.set_ylim(0.0, 1.02)
            axis.grid(alpha=0.2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(handles)), frameon=False)
    fig.suptitle("Fixed-depth Success@k, registered maximum set size 10", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _save_figure(fig, Path(pdf_path))


def _format_cell(row: Mapping[str, Any] | None) -> str:
    if row is None or _finite(row.get("mean_calls")) is None:
        return "NA"
    calls = float(row["mean_calls"])
    risk = _finite(row.get("standard_all_miss_risk_mean"))
    cert = _finite(row.get("certification_rate"))
    risk_text = "NA" if risk is None else f"{risk:.3f}"
    cert_text = "NA" if cert is None else f"{cert:.2f}"
    return f"{calls:.2f} / {risk_text} / {cert_text}"


def render_calibrated_table(rows: Sequence[Mapping[str, Any]], pdf_path: str | Path) -> tuple[str, str]:
    tasks = sorted({_text(row.get("target_task")) for row in rows if row.get("availability_status") == "available"})
    alphas = sorted({_finite(row.get("target_alpha")) for row in rows if _finite(row.get("target_alpha")) is not None})
    if not tasks or not alphas:
        raise CommonMetricError("no calibrated rows to render")
    methods = ("flexrouter", "barp", "fixed_tally_factorized", "exact_independence", "racer_g")
    pools = sorted({_integer(row.get("pool_size")) for row in rows if _integer(row.get("pool_size")) is not None})
    lookup: dict[tuple[str, int, float, str], Mapping[str, Any]] = {}
    for row in rows:
        task = _text(row.get("target_task"))
        pool = _integer(row.get("pool_size"))
        alpha = _finite(row.get("target_alpha"))
        method = _text(row.get("method"))
        if task and pool is not None and alpha is not None:
            lookup[(task, pool, alpha, method)] = row

    fig, axes = plt.subplots(
        len(tasks), len(alphas), figsize=(7.0 * len(alphas), 3.0 * len(tasks)), squeeze=False
    )
    for task_index, task in enumerate(tasks):
        for alpha_index, alpha in enumerate(alphas):
            axis = axes[task_index][alpha_index]
            axis.axis("off")
            cell_text = [
                [_format_cell(lookup.get((task, pool, alpha, method))) for pool in pools]
                for method in methods
            ]
            table = axis.table(
                cellText=cell_text,
                rowLabels=[METHOD_LABELS[method] for method in methods],
                colLabels=[f"Pool {pool}" for pool in pools],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1.0, 1.35)
            axis.set_title(f"{task}, alpha {alpha:.2f}\ncell: calls / risk / certification", fontsize=9)
    fig.suptitle("Calibrated standard all-miss reporting", fontsize=12)
    fig.text(
        0.5,
        0.01,
        "Learned and fixed-tally cells are descriptive, unpaired aggregates from different source protocols.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save_figure(fig, Path(pdf_path))


def render_native_flex(rows: Sequence[Mapping[str, Any]], pdf_path: str | Path) -> tuple[str, str]:
    if not rows:
        raise CommonMetricError("no native FlexRouter rows to render")
    tasks = sorted({_text(row.get("target_task")) for row in rows})
    pools = sorted({_integer(row.get("pool_size")) for row in rows if _integer(row.get("pool_size")) is not None})
    fig, axes = plt.subplots(
        len(tasks), len(pools), figsize=(3.4 * len(pools), 3.0 * len(tasks)), squeeze=False
    )
    for task_index, task in enumerate(tasks):
        for pool_index, pool in enumerate(pools):
            axis = axes[task_index][pool_index]
            points = sorted(
                (row for row in rows if row["target_task"] == task and row["pool_size"] == pool),
                key=lambda row: float(row["tau"]),
            )
            axis.plot(
                [row["mean_calls"] for row in points],
                [row["standard_all_miss_risk_mean"] for row in points],
                marker="o",
                color=COLORS["flexrouter"],
            )
            for row in points:
                axis.annotate(
                    f"{row['tau']:.2g}",
                    (row["mean_calls"], row["standard_all_miss_risk_mean"]),
                    fontsize=6,
                )
            axis.set_title(f"{task}, pool {pool}", fontsize=9)
            axis.set_xlabel("Mean calls")
            axis.set_ylabel("Standard all-miss risk")
            axis.grid(alpha=0.2)
    fig.suptitle("FlexRouter adaptive frontier, maximum set size 10", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(fig, Path(pdf_path))


def render_native_barp(rows: Sequence[Mapping[str, Any]], pdf_path: str | Path) -> tuple[str, str]:
    if not rows:
        raise CommonMetricError("no native BaRP sensitivity rows to render")
    tasks = sorted({_text(row.get("target_task")) for row in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.5), squeeze=False)
    for index, task in enumerate(tasks):
        axis = axes[0][index]
        points = sorted(
            (row for row in rows if row["target_task"] == task),
            key=lambda row: float(row["w_cost"]),
        )
        axis.plot(
            [row["mean_cost"] for row in points],
            [row["mean_official_quality"] for row in points],
            marker="o",
            color=COLORS["barp"],
        )
        for row in points:
            axis.annotate(
                f"{row['w_cost']:.1f}",
                (row["mean_cost"], row["mean_official_quality"]),
                fontsize=6,
            )
        axis.set_title(task, fontsize=9)
        axis.set_xlabel("Mean cost")
        axis.set_ylabel("Mean official quality")
        axis.grid(alpha=0.2)
    fig.suptitle("Native BaRP registered preference sensitivity", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save_figure(fig, Path(pdf_path))


def write_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    blocks = report["blocks"]
    paths = {
        "fixed_depth_csv": destination / "fixed_depth_success.csv",
        "calibrated_csv": destination / "calibrated_primary.csv",
        "ratios_csv": destination / "learned_vs_fixed_tally_call_ratios.csv",
        "native_flex_csv": destination / "native_flex_adaptive.csv",
        "native_barp_csv": destination / "native_barp_sensitivity.csv",
        "report_json": destination / "common_metric_report.json",
        "fixed_depth_pdf": destination / "fixed_depth_success.pdf",
        "calibrated_pdf": destination / "calibrated_primary.pdf",
        "native_flex_pdf": destination / "native_flex_adaptive.pdf",
        "native_barp_pdf": destination / "native_barp_sensitivity.pdf",
    }
    required = [
        paths["fixed_depth_csv"],
        paths["calibrated_csv"],
        paths["ratios_csv"],
        paths["native_flex_csv"],
        paths["native_barp_csv"],
        paths["report_json"],
        paths["fixed_depth_pdf"],
        paths["fixed_depth_pdf"].with_suffix(".png"),
        paths["calibrated_pdf"],
        paths["calibrated_pdf"].with_suffix(".png"),
        paths["native_flex_pdf"],
        paths["native_flex_pdf"].with_suffix(".png"),
    ]
    if blocks["native_barp_sensitivity_appendix"]:
        required.extend([paths["native_barp_pdf"], paths["native_barp_pdf"].with_suffix(".png")])
    for path in required:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output {path}")

    write_csv(blocks["fixed_depth_success"], FIXED_DEPTH_COLUMNS, paths["fixed_depth_csv"])
    write_csv(blocks["calibrated_primary"], CALIBRATED_COLUMNS, paths["calibrated_csv"])
    write_csv(blocks["learned_vs_fixed_tally_call_ratios"], RATIO_COLUMNS, paths["ratios_csv"])
    write_csv(blocks["native_flex_adaptive_appendix"], FLEX_COLUMNS, paths["native_flex_csv"])
    write_csv(blocks["native_barp_sensitivity_appendix"], BARP_COLUMNS, paths["native_barp_csv"])
    render_fixed_depth(blocks["fixed_depth_success"], paths["fixed_depth_pdf"])
    render_calibrated_table(blocks["calibrated_primary"], paths["calibrated_pdf"])
    render_native_flex(blocks["native_flex_adaptive_appendix"], paths["native_flex_pdf"])
    if blocks["native_barp_sensitivity_appendix"]:
        render_native_barp(blocks["native_barp_sensitivity_appendix"], paths["native_barp_pdf"])

    payload = dict(report)
    payload["outputs"] = {
        key: str(path)
        for key, path in paths.items()
        if key != "native_barp_pdf" or blocks["native_barp_sensitivity_appendix"]
    }
    payload["outputs"].update(
        {
            "fixed_depth_png": str(paths["fixed_depth_pdf"].with_suffix(".png")),
            "calibrated_png": str(paths["calibrated_pdf"].with_suffix(".png")),
            "native_flex_png": str(paths["native_flex_pdf"].with_suffix(".png")),
        }
    )
    if blocks["native_barp_sensitivity_appendix"]:
        payload["outputs"]["native_barp_png"] = str(paths["native_barp_pdf"].with_suffix(".png"))
    write_json(payload, paths["report_json"])
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-summary", required=True)
    parser.add_argument("--query-table", required=True)
    parser.add_argument("--e1-fixed-depth", required=True)
    parser.add_argument("--e2-fixed-depth", required=True)
    parser.add_argument("--native-barp-summary")
    parser.add_argument("--query-task", default="ifeval", help="Explicit task label for the task-free fixed-tally query table")
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", required=True)
    arguments = parser.parse_args(argv)

    input_paths = {
        "learned_summary": Path(arguments.learned_summary),
        "query_table": Path(arguments.query_table),
        "e1_fixed_depth": Path(arguments.e1_fixed_depth),
        "e2_fixed_depth": Path(arguments.e2_fixed_depth),
    }
    native_path = Path(arguments.native_barp_summary) if arguments.native_barp_summary else None
    if native_path is not None:
        input_paths["native_barp_summary"] = native_path

    report = build_report(
        load_csv(input_paths["learned_summary"]),
        load_csv(input_paths["query_table"]),
        load_csv(input_paths["e1_fixed_depth"]),
        load_csv(input_paths["e2_fixed_depth"]),
        load_csv(native_path) if native_path is not None else None,
        query_task=arguments.query_task,
    )
    report["inputs"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in input_paths.items()
    }
    payload = write_report(report, arguments.output_dir)
    counts = payload["row_counts"]
    print(
        "wrote common-metric report with "
        f"{counts['fixed_depth_success']} fixed-depth rows and "
        f"{counts['calibrated_primary']} calibrated rows to {arguments.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
