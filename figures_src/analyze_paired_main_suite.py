"""Paired uncertainty and feasibility analysis for canonical E2/E3 records.

The analyzer never selects, retunes, or risk-matches a method on test data. It
compares frozen methods at the same cell and redraw identities. Canonical input
uses this compact shape::

    {
      "run_id": "...", "protocol": "...", "benchmark": "...",
      "pool_id": "...", "pool_size": 3,
      "query_ids": [...], "model_ids": [...],
      "cells": [{"alpha": 0.05, "redraws": [{
        "redraw_id": 0, "split_ids": {"design": [...], ...},
        "full_pool": {"model_ids": [...], "calibration_misses": 0,
                      "calibration_count": 100, "test_misses": 1,
                      "test_count": 100},
        "methods": {"racer_G": {...}, "exact_independence": {...}}
      }]}]
    }

A certified method record must contain exact per-query test calls and standard
all-miss indicators. It must also contain final-calibration counts and endpoint
metadata. E2 ``per_redraw`` and E3 ``per_alpha_records`` containers are accepted
when they carry the same canonical identity fields.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BOOTSTRAP_SEED = 20260904
DEFAULT_BOOTSTRAP_REPETITIONS = 10_000
CONFIDENCE_LEVEL = 0.95
_WILSON_Z = 1.959963984540054

IDENTITY_COLUMNS = (
    "protocol",
    "benchmark",
    "pool_id",
    "pool_size",
    "target_alpha",
    "method",
    "reference_method",
)

CSV_COLUMNS = [
    *IDENTITY_COLUMNS,
    "redraw_count",
    "paired_certified_redraw_count",
    "method_certified_redraw_count",
    "reference_certified_redraw_count",
    "method_certified_mean_calls",
    "method_certified_calls_se",
    "reference_certified_mean_calls",
    "reference_certified_calls_se",
    "paired_mean_calls_difference",
    "paired_calls_difference_se",
    "paired_calls_redraw_ci95_low",
    "paired_calls_redraw_ci95_high",
    "paired_calls_query_cluster_ci95_low",
    "paired_calls_query_cluster_ci95_high",
    "method_certified_standard_risk",
    "method_certified_standard_risk_se",
    "reference_certified_standard_risk",
    "reference_certified_standard_risk_se",
    "paired_mean_standard_risk_difference",
    "paired_standard_risk_difference_se",
    "paired_standard_risk_redraw_ci95_low",
    "paired_standard_risk_redraw_ci95_high",
    "paired_standard_risk_query_cluster_ci95_low",
    "paired_standard_risk_query_cluster_ci95_high",
    "query_cluster_count",
    "method_test_miss_numerator",
    "method_test_miss_denominator",
    "reference_test_miss_numerator",
    "reference_test_miss_denominator",
    "method_certification_numerator",
    "method_certification_denominator",
    "method_certification_rate",
    "method_certification_wilson_ci95_low",
    "method_certification_wilson_ci95_high",
    "reference_certification_numerator",
    "reference_certification_denominator",
    "reference_certification_rate",
    "reference_certification_wilson_ci95_low",
    "reference_certification_wilson_ci95_high",
    "method_empirical_joint_event_j_hat",
    "reference_empirical_joint_event_j_hat",
    "method_full_pool_calibration_misses",
    "method_full_pool_calibration_denominator",
    "method_full_pool_test_misses",
    "method_full_pool_test_denominator",
    "reference_full_pool_calibration_misses",
    "reference_full_pool_calibration_denominator",
    "reference_full_pool_test_misses",
    "reference_full_pool_test_denominator",
    "shared_endpoint_redraw_count",
    "shared_endpoint_certification_match_count",
    "shared_feasibility_equivalent",
    "endpoint_difference_count",
    "certification_disagreement_count",
    "certification_disagreement_reasons",
    "method_per_redraw",
    "reference_per_redraw",
    "paired_redraw_ids",
    "source_run_ids",
    "source_hashes",
    "bootstrap_seed",
    "bootstrap_repetitions",
    "redraw_interval_method",
    "query_interval_method",
    "split_pool_variability_label",
    "query_population_uncertainty_label",
    "difference_definition",
    "pairing_rule",
    "test_retuning_used",
]


class PairedAnalysisError(ValueError):
    """Raised when canonical inputs violate the pairing contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_record(record: Mapping[str, Any]) -> str:
    return _hash_bytes(_canonical_json(record).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first(mapping: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _finite_number(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise PairedAnalysisError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise PairedAnalysisError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise PairedAnalysisError(f"{name} must be a finite number")
    return number


def _nonnegative_int(value: Any, name: str) -> int:
    number = _finite_number(value, name)
    integer = int(number)
    if number != integer or integer < 0:
        raise PairedAnalysisError(f"{name} must be a nonnegative integer")
    return integer


def _bool_value(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (0, 1):
        return bool(value)
    raise PairedAnalysisError(f"{name} must be boolean")


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _se(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values) / math.sqrt(len(values)))


def _wilson(successes: int, trials: int) -> tuple[float | None, float | None, float | None]:
    if trials == 0:
        return None, None, None
    proportion = successes / trials
    z2 = _WILSON_Z * _WILSON_Z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    radius = (
        _WILSON_Z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z2 / (4.0 * trials * trials)
        )
        / denominator
    )
    return proportion, max(0.0, center - radius), min(1.0, center + radius)


def _bootstrap_mean_interval(
    values: Sequence[float], *, repetitions: int, seed: int
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 1024):
        size = min(1024, repetitions - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        means[start : start + size] = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _cluster_interval(
    observations: Sequence[tuple[str, float]], *, repetitions: int, seed: int
) -> tuple[float | None, float | None, int]:
    if not observations:
        return None, None, 0
    sums: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for query_id, value in observations:
        sums[query_id] += float(value)
        counts[query_id] += 1
    query_ids = sorted(sums)
    cluster_sums = np.asarray([sums[query_id] for query_id in query_ids], dtype=float)
    cluster_counts = np.asarray([counts[query_id] for query_id in query_ids], dtype=float)
    if len(query_ids) == 1:
        value = float(cluster_sums[0] / cluster_counts[0])
        return value, value, 1
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 512):
        size = min(512, repetitions - start)
        indices = rng.integers(0, len(query_ids), size=(size, len(query_ids)))
        estimates[start : start + size] = (
            cluster_sums[indices].sum(axis=1) / cluster_counts[indices].sum(axis=1)
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high), len(query_ids)


def canonical_method_name(name: Any) -> str:
    value = str(name).strip().replace("-", "_").replace(" ", "_").lower()
    aliases = {
        "racer_g": "racer_g",
        "racer_p": "racer_p",
        "factorized_independence": "exact_independence",
        "wrapper_factorized_independence": "exact_independence",
        "wrapper_factorized_independence_tuned": "exact_independence",
        "exact_factorized": "exact_independence",
        "selected": "e3_selected",
        "e3": "e3_selected",
        "baseline_fixed_marginal": "fixed_marginal",
    }
    if value.endswith("_tuned") and value.startswith("wrapper_"):
        value = value[: -len("_tuned")]
    return aliases.get(value, value)


def _extract_sequence(mapping: Mapping[str, Any], names: Sequence[str], name: str) -> list[str]:
    value = _first(mapping, names)
    if value is None:
        raise PairedAnalysisError(f"canonical record lacks {name}")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PairedAnalysisError(f"{name} must be a sequence")
    result = [str(item) for item in value]
    if len(result) != len(set(result)):
        raise PairedAnalysisError(f"{name} must contain unique IDs")
    return result


def _record_model_ids(record: Mapping[str, Any]) -> list[str]:
    nested = record.get("model_pool") if isinstance(record.get("model_pool"), Mapping) else {}
    value = _first(record, ("model_ids", "pool_model_ids"), _first(nested, ("model_ids",)))
    if value is None:
        raise PairedAnalysisError("canonical record lacks model_ids")
    return _extract_sequence({"value": value}, ("value",), "model_ids")


def _record_query_ids(record: Mapping[str, Any]) -> list[str]:
    nested = record.get("query_pool") if isinstance(record.get("query_pool"), Mapping) else {}
    value = _first(record, ("query_ids", "query_pool_ids", "all_query_ids"), _first(nested, ("query_ids",)))
    if value is None:
        raise PairedAnalysisError("canonical record lacks query_ids")
    return _extract_sequence({"value": value}, ("value",), "query_ids")


def _split_ids(redraw: Mapping[str, Any], record: Mapping[str, Any], redraw_id: Any) -> dict[str, list[str]]:
    value = redraw.get("split_ids")
    if value is None:
        containers = _first(record, ("split_ids_by_redraw", "redraw_splits"), {})
        if isinstance(containers, Mapping):
            value = containers.get(str(redraw_id), containers.get(redraw_id))
        elif isinstance(containers, Sequence) and not isinstance(containers, (str, bytes)):
            matches = [item for item in containers if isinstance(item, Mapping) and _first(item, ("redraw_id", "redraw")) == redraw_id]
            value = matches[0].get("split_ids") if len(matches) == 1 else None
    if not isinstance(value, Mapping) or not value:
        raise PairedAnalysisError(f"redraw {redraw_id!r} lacks split_ids")
    result: dict[str, list[str]] = {}
    seen: set[str] = set()
    for role, ids in value.items():
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
            raise PairedAnalysisError(f"split_ids.{role} must be a sequence")
        normalized = [str(item) for item in ids]
        if len(normalized) != len(set(normalized)):
            raise PairedAnalysisError(f"split_ids.{role} contains duplicate IDs")
        overlap = seen.intersection(normalized)
        if overlap:
            raise PairedAnalysisError(f"split roles overlap at query IDs {sorted(overlap)[:3]}")
        seen.update(normalized)
        result[str(role)] = normalized
    required = {"design", "router", "validation", "final_calibration", "test"}
    missing = required - set(result)
    if missing:
        raise PairedAnalysisError(f"redraw {redraw_id!r} lacks split roles {sorted(missing)}")
    return result


def _top_split_for_redraw(record: Mapping[str, Any], redraw_id: Any) -> Mapping[str, Any] | None:
    for key in ("redraw_metadata", "redraws"):
        values = record.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for item in values:
                if isinstance(item, Mapping) and _first(item, ("redraw_id", "redraw")) == redraw_id:
                    return item
    return None


def _methods_from_e3(redraw: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    methods: dict[str, Mapping[str, Any]] = {}
    if isinstance(redraw.get("selected"), Mapping):
        methods["e3_selected"] = redraw["selected"]
    if isinstance(redraw.get("baseline_fixed_marginal"), Mapping):
        methods["fixed_marginal"] = redraw["baseline_fixed_marginal"]
    for key in ("racer_controls", "controls", "methods"):
        values = redraw.get(key)
        if isinstance(values, Mapping):
            for name, value in values.items():
                if isinstance(value, Mapping):
                    methods[canonical_method_name(name)] = value
    return methods


def _cells(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct = record.get("cells")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        cells = []
        for cell in direct:
            if not isinstance(cell, Mapping):
                raise PairedAnalysisError("cells entries must be mappings")
            alpha = _finite_number(_first(cell, ("alpha", "target_alpha")), "target_alpha")
            redraws = cell.get("redraws")
            if isinstance(redraws, Mapping):
                redraws = list(redraws.values())
            if not isinstance(redraws, Sequence) or isinstance(redraws, (str, bytes)):
                raise PairedAnalysisError(f"cell alpha {alpha} lacks redraws")
            cells.append({"alpha": alpha, "cell": cell, "redraws": list(redraws)})
        return cells

    per_redraw = record.get("per_redraw")
    if isinstance(per_redraw, Mapping):
        cells = []
        for alpha_key, by_method in per_redraw.items():
            if not isinstance(by_method, Mapping):
                continue
            redraws: dict[Any, dict[str, Any]] = {}
            for method_name, values in by_method.items():
                if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                    continue
                for index, method_record in enumerate(values):
                    if not isinstance(method_record, Mapping):
                        raise PairedAnalysisError("per_redraw method entries must be mappings")
                    redraw_id = _first(method_record, ("redraw_id", "redraw"), index)
                    holder = redraws.setdefault(redraw_id, {"redraw_id": redraw_id, "methods": {}})
                    metadata = _top_split_for_redraw(record, redraw_id)
                    if metadata:
                        holder.update(metadata)
                    holder["methods"][canonical_method_name(method_name)] = method_record
            cells.append({"alpha": _finite_number(alpha_key, "target_alpha"), "cell": {}, "redraws": list(redraws.values())})
        return cells

    per_alpha = record.get("per_alpha_records")
    if isinstance(per_alpha, Mapping):
        cells = []
        for alpha_key, values in per_alpha.items():
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            redraws = []
            for index, entry in enumerate(values):
                if not isinstance(entry, Mapping):
                    raise PairedAnalysisError("per_alpha_records entries must be mappings")
                holder = dict(entry)
                holder.setdefault("redraw_id", _first(entry, ("redraw",), index))
                holder["methods"] = _methods_from_e3(entry)
                metadata = _top_split_for_redraw(record, holder["redraw_id"])
                if metadata:
                    for key, value in metadata.items():
                        holder.setdefault(key, value)
                redraws.append(holder)
            cells.append({"alpha": _finite_number(alpha_key, "target_alpha"), "cell": {}, "redraws": redraws})
        return cells
    raise PairedAnalysisError("canonical record lacks cells, per_redraw, or per_alpha_records")


def _endpoint(method: Mapping[str, Any], model_ids: Sequence[str]) -> dict[str, Any]:
    raw = _first(method, ("eligibility_endpoint", "endpoint"), {})
    raw = raw if isinstance(raw, Mapping) else {}
    ids_value = _first(raw, ("model_ids", "endpoint_model_ids"), _first(method, ("endpoint_model_ids", "eligible_model_ids")))
    cap_value = _first(raw, ("eligibility_cap", "max_depth", "depth"), _first(method, ("eligibility_cap", "max_depth")))
    if ids_value is not None:
        if isinstance(ids_value, (str, bytes)) or not isinstance(ids_value, Sequence):
            raise PairedAnalysisError("endpoint model_ids must be a sequence")
        ids = [str(value) for value in ids_value]
        if len(ids) != len(set(ids)) or any(value not in model_ids for value in ids):
            raise PairedAnalysisError("endpoint model_ids must be unique members of model_ids")
        cap = len(ids)
        signature: Any = {"model_ids": sorted(ids)}
    elif cap_value is not None:
        cap = _nonnegative_int(cap_value, "eligibility_cap")
        if cap > len(model_ids):
            raise PairedAnalysisError("eligibility_cap exceeds pool_size")
        order = _first(method, ("model_order", "order"))
        if isinstance(order, Sequence) and not isinstance(order, (str, bytes)):
            ids = [str(value) for value in order[:cap]]
            signature = {"model_ids": sorted(ids)}
        elif cap == len(model_ids):
            ids = list(model_ids)
            signature = {"model_ids": sorted(ids)}
        else:
            ids = []
            signature = {"eligibility_cap": cap}
    else:
        ids = list(model_ids)
        cap = len(ids)
        signature = {"model_ids": sorted(ids)}
    explicit_full = _first(raw, ("reaches_full_pool",), _first(method, ("reaches_full_pool",)))
    reaches_full = cap == len(model_ids) and (not ids or set(ids) == set(model_ids))
    if explicit_full is not None and _bool_value(explicit_full, "reaches_full_pool") != reaches_full:
        raise PairedAnalysisError("reaches_full_pool disagrees with endpoint identity")
    return {"model_ids": ids, "cap": cap, "signature": _canonical_json(signature), "reaches_full_pool": reaches_full}


def _query_observations(method: Mapping[str, Any], test_ids: Sequence[str]) -> tuple[list[float], list[int]]:
    rows = _first(method, ("test_queries", "per_query", "query_metrics"))
    calls_by_id: dict[str, float] = {}
    miss_by_id: dict[str, int] = {}
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if not isinstance(row, Mapping):
                raise PairedAnalysisError("per-query entries must be mappings")
            query_id = str(_first(row, ("query_id", "id")))
            if query_id in calls_by_id:
                raise PairedAnalysisError(f"duplicate per-query record for {query_id}")
            calls_value = _first(row, ("calls", "model_calls", "test_calls"))
            if calls_value is None:
                selected = _first(row, ("selected_model_ids", "selected_set"))
                if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
                    calls_value = len(selected)
            calls_by_id[query_id] = _finite_number(calls_value, f"calls for query {query_id}")
            miss_value = _first(row, ("standard_all_miss", "test_miss", "miss", "coverage_loss"))
            miss_by_id[query_id] = int(_bool_value(miss_value, f"standard miss for query {query_id}"))
    else:
        ids = _first(method, ("test_query_ids", "query_ids"))
        calls = _first(method, ("test_calls_by_query", "per_query_calls", "calls_by_query"))
        misses = _first(method, ("test_miss_by_query", "per_query_standard_all_miss", "standard_all_miss_by_query", "test_misses_by_query"))
        if not all(isinstance(value, Sequence) and not isinstance(value, (str, bytes)) for value in (ids, calls, misses)):
            raise PairedAnalysisError("certified method lacks exact per-query calls and standard miss indicators")
        if not (len(ids) == len(calls) == len(misses)):
            raise PairedAnalysisError("per-query arrays have different lengths")
        for query_id, calls_value, miss_value in zip(ids, calls, misses):
            key = str(query_id)
            if key in calls_by_id:
                raise PairedAnalysisError(f"duplicate per-query record for {key}")
            calls_by_id[key] = _finite_number(calls_value, f"calls for query {key}")
            miss_by_id[key] = int(_bool_value(miss_value, f"standard miss for query {key}"))
    if set(calls_by_id) != set(test_ids):
        raise PairedAnalysisError("per-query IDs do not equal split_ids.test")
    return [calls_by_id[value] for value in test_ids], [miss_by_id[value] for value in test_ids]


def _common_full_pool(redraw: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _first(redraw, ("full_pool", "full_pool_metrics", "shared_full_pool"), {})
    return value if isinstance(value, Mapping) else {}


def _count_field(primary: Mapping[str, Any], common: Mapping[str, Any], names: Sequence[str], label: str, *, required: bool = True) -> int | None:
    value = _first(primary, names, _first(common, names))
    if value is None:
        if required:
            raise PairedAnalysisError(f"method lacks {label}")
        return None
    return _nonnegative_int(value, label)


def _normalize_method(
    method: Mapping[str, Any], *, test_ids: Sequence[str], model_ids: Sequence[str], alpha: float
) -> dict[str, Any]:
    certified_value = _first(method, ("final_certified", "certified"))
    if certified_value is None:
        noncertified = _first(method, ("non_certification", "infeasible"))
        if noncertified is None:
            raise PairedAnalysisError("method lacks certification indicator")
        certified_value = not _bool_value(noncertified, "non_certification")
    certified = _bool_value(certified_value, "certified")
    endpoint = _endpoint(method, model_ids)
    common = method.get("_common_full_pool", {})
    common = common if isinstance(common, Mapping) else {}
    calibration_count = _count_field(method, common, ("n_final_calibration", "calibration_count", "final_calibration_denominator"), "final calibration denominator")
    calibration_misses = _count_field(method, common, ("calibration_misses", "final_calibration_misses"), "final calibration misses")
    if calibration_misses > calibration_count:
        raise PairedAnalysisError("calibration misses exceed denominator")
    corrected_bound = (calibration_misses + 1) / (calibration_count + 1)
    supplied_bound = _first(method, ("corrected_calibration_bound", "crc_bound"))
    if supplied_bound is not None and not math.isclose(_finite_number(supplied_bound, "corrected calibration bound"), corrected_bound, rel_tol=1e-9, abs_tol=1e-12):
        raise PairedAnalysisError("corrected calibration bound disagrees with exact calibration counts")

    endpoint_cal_misses = _count_field(method, common, ("endpoint_calibration_misses", "unavoidable_calibration_misses", "full_pool_calibration_misses"), "endpoint calibration misses")
    endpoint_cal_count = _count_field(method, common, ("endpoint_calibration_count", "full_pool_calibration_denominator", "full_pool_calibration_count", "n_final_calibration", "calibration_count"), "endpoint calibration denominator")
    endpoint_bound = (endpoint_cal_misses + 1) / (endpoint_cal_count + 1)
    implied_certified = endpoint_bound <= alpha + 1e-15
    if certified != implied_certified:
        raise PairedAnalysisError("certification indicator disagrees with corrected endpoint feasibility")

    full_cal_raw = _first(
        method,
        ("full_pool_calibration_misses",),
        _first(common, ("full_pool_calibration_misses", "calibration_misses")),
    )
    full_cal_count_raw = _first(
        method,
        ("full_pool_calibration_denominator", "full_pool_calibration_count"),
        _first(
            common,
            (
                "full_pool_calibration_denominator",
                "full_pool_calibration_count",
                "calibration_denominator",
                "calibration_count",
            ),
        ),
    )
    full_test_raw = _first(
        method,
        ("full_pool_test_misses",),
        _first(common, ("full_pool_test_misses", "test_misses")),
    )
    full_test_count_raw = _first(
        method,
        ("full_pool_test_denominator", "full_pool_test_count"),
        _first(
            common,
            (
                "full_pool_test_denominator",
                "full_pool_test_count",
                "test_denominator",
                "test_count",
            ),
        ),
    )
    full_cal_misses = (
        _nonnegative_int(full_cal_raw, "full-pool calibration misses")
        if full_cal_raw is not None
        else None
    )
    full_cal_count = (
        _nonnegative_int(full_cal_count_raw, "full-pool calibration denominator")
        if full_cal_count_raw is not None
        else None
    )
    full_test_misses = (
        _nonnegative_int(full_test_raw, "full-pool test misses")
        if full_test_raw is not None
        else None
    )
    full_test_count = (
        _nonnegative_int(full_test_count_raw, "full-pool test denominator")
        if full_test_count_raw is not None
        else None
    )
    if endpoint["reaches_full_pool"]:
        full_cal_misses = endpoint_cal_misses if full_cal_misses is None else full_cal_misses
        full_cal_count = endpoint_cal_count if full_cal_count is None else full_cal_count
    if None in (full_cal_misses, full_cal_count, full_test_misses, full_test_count):
        raise PairedAnalysisError("method lacks exact full-pool calibration or test counts")
    if full_cal_misses > full_cal_count or full_test_misses > full_test_count:
        raise PairedAnalysisError("full-pool misses exceed denominator")

    calls: list[float] = []
    misses: list[int] = []
    if certified:
        calls, misses = _query_observations(method, test_ids)
        aggregate_calls = _first(method, ("calls_test", "test_calls", "model_calls", "calls"))
        if aggregate_calls is not None and not math.isclose(_finite_number(aggregate_calls, "test calls"), statistics.fmean(calls), rel_tol=1e-9, abs_tol=1e-12):
            raise PairedAnalysisError("test calls disagree with per-query calls")
        aggregate_risk = _first(method, ("standard_all_miss_risk", "test_risk", "risk"))
        risk = sum(misses) / len(misses)
        if aggregate_risk is not None and not math.isclose(_finite_number(aggregate_risk, "standard risk"), risk, rel_tol=1e-9, abs_tol=1e-12):
            raise PairedAnalysisError("standard risk disagrees with exact test misses")
    reason = _first(method, ("non_certification_reason", "certification_reason"))
    if not certified and reason is None:
        if endpoint_bound <= 1.0 / (endpoint_cal_count + 1) + 1e-15 and alpha < 1.0 / (endpoint_cal_count + 1):
            reason = "finite_calibration_resolution"
        elif endpoint["reaches_full_pool"]:
            reason = "full_pool_endpoint_infeasible"
        else:
            reason = "shallow_eligibility_endpoint_infeasible"
    return {
        "certified": certified,
        "calls_by_query": calls,
        "miss_by_query": misses,
        "test_misses": int(sum(misses)) if certified else 0,
        "test_count": len(misses) if certified else 0,
        "calls": float(statistics.fmean(calls)) if certified else None,
        "risk": float(sum(misses) / len(misses)) if certified else None,
        "calibration_misses": calibration_misses,
        "calibration_count": calibration_count,
        "corrected_calibration_bound": corrected_bound,
        "endpoint_calibration_misses": endpoint_cal_misses,
        "endpoint_calibration_count": endpoint_cal_count,
        "endpoint_corrected_bound": endpoint_bound,
        "full_pool_calibration_misses": full_cal_misses,
        "full_pool_calibration_count": full_cal_count,
        "full_pool_test_misses": full_test_misses,
        "full_pool_test_count": full_test_count,
        "endpoint": endpoint,
        "non_certification_reason": reason,
    }


def _normalize_inputs(records: Sequence[tuple[str, str, Mapping[str, Any]]]) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[str, str], list[str]]:
    if not records:
        raise PairedAnalysisError("at least one canonical record is required")
    expected_query_ids: list[str] | None = None
    expected_model_ids: list[str] | None = None
    expected_pool_id: str | None = None
    expected_identity: tuple[str, str, int] | None = None
    cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    run_ids: list[str] = []

    for source_id, source_hash, record in records:
        query_ids = _record_query_ids(record)
        model_ids = _record_model_ids(record)
        pool_size = _nonnegative_int(_first(record, ("pool_size",), len(model_ids)), "pool_size")
        if pool_size != len(model_ids):
            raise PairedAnalysisError("pool_size does not equal len(model_ids)")
        pool_id = str(_first(record, ("pool_id", "model_pool_id"), _hash_bytes(_canonical_json(model_ids).encode("utf-8"))[:16]))
        protocol = str(_first(record, ("protocol", "source_protocol"), ""))
        benchmark = str(_first(record, ("benchmark", "task", "dataset"), ""))
        if not protocol or not benchmark:
            raise PairedAnalysisError("canonical record lacks protocol or benchmark")
        identity = (protocol, benchmark, pool_size)
        if expected_query_ids is None:
            expected_query_ids, expected_model_ids = query_ids, model_ids
            expected_pool_id, expected_identity = pool_id, identity
        else:
            if query_ids != expected_query_ids:
                raise PairedAnalysisError("input records have nonpaired query_ids")
            if model_ids != expected_model_ids:
                raise PairedAnalysisError("input records have nonpaired model_ids or model order")
            if pool_id != expected_pool_id or identity != expected_identity:
                raise PairedAnalysisError("input records have nonpaired pools, protocol, or benchmark")
        source_hashes[source_id] = source_hash
        run_id = str(_first(record, ("run_id", "source_run_id"), source_id))
        run_ids.append(run_id)

        for raw_cell in _cells(record):
            alpha = raw_cell["alpha"]
            if not 0.0 < alpha < 1.0:
                raise PairedAnalysisError("target_alpha must be strictly between zero and one")
            cell_key = (protocol, benchmark, pool_id, pool_size, alpha)
            cell = cells.setdefault(cell_key, {"redraws": {}, "source_run_ids": set(), "source_ids": set()})
            cell["source_run_ids"].add(run_id)
            cell["source_ids"].add(source_id)
            for index, raw_redraw in enumerate(raw_cell["redraws"]):
                if not isinstance(raw_redraw, Mapping):
                    raise PairedAnalysisError("redraw entries must be mappings")
                redraw_id = _first(raw_redraw, ("redraw_id", "redraw", "seed"), index)
                splits = _split_ids(raw_redraw, record, redraw_id)
                if any(query_id not in set(query_ids) for ids in splits.values() for query_id in ids):
                    raise PairedAnalysisError("split_ids contain IDs outside query_ids")
                holder = cell["redraws"].setdefault(redraw_id, {"split_ids": splits, "methods": {}})
                if holder["split_ids"] != splits:
                    raise PairedAnalysisError(f"redraw {redraw_id!r} has nonpaired split IDs")
                raw_methods = raw_redraw.get("methods")
                if not isinstance(raw_methods, Mapping) or not raw_methods:
                    raise PairedAnalysisError(f"redraw {redraw_id!r} lacks methods")
                common_full_pool = _common_full_pool(raw_redraw)
                common_model_ids = _first(
                    common_full_pool, ("model_ids", "pool_model_ids")
                )
                if common_model_ids is not None:
                    if (
                        isinstance(common_model_ids, (str, bytes))
                        or not isinstance(common_model_ids, Sequence)
                        or [str(value) for value in common_model_ids] != model_ids
                    ):
                        raise PairedAnalysisError(
                            "full-pool model IDs disagree with canonical model_ids"
                        )
                for raw_name, raw_method in raw_methods.items():
                    if not isinstance(raw_method, Mapping):
                        raise PairedAnalysisError("method records must be mappings")
                    name = canonical_method_name(raw_name)
                    method_copy = dict(raw_method)
                    method_copy["_common_full_pool"] = common_full_pool
                    normalized = _normalize_method(method_copy, test_ids=splits["test"], model_ids=model_ids, alpha=alpha)
                    normalized["source_id"] = source_id
                    if name in holder["methods"]:
                        existing = dict(holder["methods"][name])
                        existing.pop("source_id", None)
                        candidate = dict(normalized)
                        candidate.pop("source_id", None)
                        if _canonical_json(existing) != _canonical_json(candidate):
                            raise PairedAnalysisError(f"duplicate method {name!r} disagrees across records")
                    else:
                        holder["methods"][name] = normalized

    for cell_key, cell in cells.items():
        redraws = cell["redraws"]
        method_sets = [set(redraw["methods"]) for redraw in redraws.values()]
        if not method_sets or any(methods != method_sets[0] for methods in method_sets[1:]):
            raise PairedAnalysisError(f"cell {cell_key} has methods missing from some redraws")
        first_splits = None
        for redraw_id, redraw in sorted(redraws.items(), key=lambda item: str(item[0])):
            signature = set(redraw["split_ids"])
            if first_splits is None:
                first_splits = signature
            elif signature != first_splits:
                raise PairedAnalysisError(f"cell {cell_key} redraw {redraw_id!r} has nonpaired split roles")
            full_pool_counts = {
                (
                    method["full_pool_calibration_misses"],
                    method["full_pool_calibration_count"],
                    method["full_pool_test_misses"],
                    method["full_pool_test_count"],
                )
                for method in redraw["methods"].values()
            }
            if len(full_pool_counts) != 1:
                raise PairedAnalysisError(
                    f"cell {cell_key} redraw {redraw_id!r} has nonpaired full-pool event counts"
                )
    return cells, source_hashes, sorted(set(run_ids))


def _arm_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    certified = [record for record in records if record["certified"]]
    calls = [record["calls"] for record in certified]
    risks = [record["risk"] for record in certified]
    certification_count = len(certified)
    rate, low, high = _wilson(certification_count, len(records))
    return {
        "certification_count": certification_count,
        "certification_total": len(records),
        "certification_rate": rate,
        "certification_low": low,
        "certification_high": high,
        "calls_mean": _mean(calls),
        "calls_se": _se(calls),
        "risk_mean": _mean(risks),
        "risk_se": _se(risks),
        "test_misses": sum(record["test_misses"] for record in certified),
        "test_count": sum(record["test_count"] for record in certified),
        "joint_event": sum((record["test_misses"] / record["test_count"]) if record["certified"] else 0.0 for record in records) / len(records),
        "full_pool_calibration_misses": sum(record["full_pool_calibration_misses"] for record in records),
        "full_pool_calibration_count": sum(record["full_pool_calibration_count"] for record in records),
        "full_pool_test_misses": sum(record["full_pool_test_misses"] for record in records),
        "full_pool_test_count": sum(record["full_pool_test_count"] for record in records),
    }


def _endpoint_reason(method: dict[str, Any], reference: dict[str, Any]) -> str:
    if method["endpoint"]["signature"] == reference["endpoint"]["signature"]:
        return "shared_endpoint"
    method_full = method["endpoint"]["reaches_full_pool"]
    reference_full = reference["endpoint"]["reaches_full_pool"]
    if not method_full and reference_full:
        return "method_shallow_eligibility_cap"
    if method_full and not reference_full:
        return "reference_shallow_eligibility_cap"
    return "different_eligibility_endpoints"


def _paired_row(
    cell_key: tuple[Any, ...],
    cell: Mapping[str, Any],
    method_name: str,
    reference_name: str,
    *,
    source_hashes: Mapping[str, str],
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    redraw_items = sorted(cell["redraws"].items(), key=lambda item: str(item[0]))
    methods = [redraw["methods"][method_name] for _, redraw in redraw_items]
    references = [redraw["methods"][reference_name] for _, redraw in redraw_items]
    method_summary = _arm_summary(methods)
    reference_summary = _arm_summary(references)
    paired_calls: list[float] = []
    paired_risks: list[float] = []
    call_clusters: list[tuple[str, float]] = []
    risk_clusters: list[tuple[str, float]] = []
    paired_redraw_ids: list[Any] = []
    disagreement_reasons: Counter[str] = Counter()
    shared_count = 0
    shared_matches = 0
    endpoint_difference_count = 0
    disagreement_count = 0
    method_redraw_details = []
    reference_redraw_details = []

    for (redraw_id, redraw), method, reference in zip(redraw_items, methods, references):
        endpoint_reason = _endpoint_reason(method, reference)
        if endpoint_reason == "shared_endpoint":
            shared_count += 1
            if (
                method["endpoint_calibration_misses"] != reference["endpoint_calibration_misses"]
                or method["endpoint_calibration_count"] != reference["endpoint_calibration_count"]
            ):
                raise PairedAnalysisError("shared endpoint methods disagree on calibration event counts")
            if method["certified"] != reference["certified"]:
                raise PairedAnalysisError("shared endpoint methods disagree on certification")
            shared_matches += 1
        else:
            endpoint_difference_count += 1
        if method["certified"] != reference["certified"]:
            disagreement_count += 1
            disagreement_reasons[endpoint_reason] += 1
        test_ids = redraw["split_ids"]["test"]
        if method["certified"] and reference["certified"]:
            paired_redraw_ids.append(redraw_id)
            call_difference = method["calls"] - reference["calls"]
            risk_difference = method["risk"] - reference["risk"]
            paired_calls.append(call_difference)
            paired_risks.append(risk_difference)
            if not (len(method["calls_by_query"]) == len(reference["calls_by_query"]) == len(test_ids)):
                raise PairedAnalysisError("paired methods have different query observation counts")
            for index, query_id in enumerate(test_ids):
                call_clusters.append((query_id, method["calls_by_query"][index] - reference["calls_by_query"][index]))
                risk_clusters.append((query_id, method["miss_by_query"][index] - reference["miss_by_query"][index]))
        method_redraw_details.append({
            "redraw_id": redraw_id,
            "certified": method["certified"],
            "calls": method["calls"],
            "standard_all_miss_risk": method["risk"],
            "test_misses": method["test_misses"],
            "test_count": method["test_count"],
            "calibration_misses": method["calibration_misses"],
            "calibration_count": method["calibration_count"],
            "corrected_calibration_bound": method["corrected_calibration_bound"],
            "endpoint_calibration_misses": method["endpoint_calibration_misses"],
            "endpoint_calibration_count": method["endpoint_calibration_count"],
            "endpoint_corrected_bound": method["endpoint_corrected_bound"],
            "full_pool_calibration_misses": method["full_pool_calibration_misses"],
            "full_pool_calibration_count": method["full_pool_calibration_count"],
            "full_pool_test_misses": method["full_pool_test_misses"],
            "full_pool_test_count": method["full_pool_test_count"],
            "endpoint": method["endpoint"],
            "non_certification_reason": method["non_certification_reason"],
        })
        reference_redraw_details.append({
            "redraw_id": redraw_id,
            "certified": reference["certified"],
            "calls": reference["calls"],
            "standard_all_miss_risk": reference["risk"],
            "test_misses": reference["test_misses"],
            "test_count": reference["test_count"],
            "calibration_misses": reference["calibration_misses"],
            "calibration_count": reference["calibration_count"],
            "corrected_calibration_bound": reference["corrected_calibration_bound"],
            "endpoint_calibration_misses": reference["endpoint_calibration_misses"],
            "endpoint_calibration_count": reference["endpoint_calibration_count"],
            "endpoint_corrected_bound": reference["endpoint_corrected_bound"],
            "full_pool_calibration_misses": reference["full_pool_calibration_misses"],
            "full_pool_calibration_count": reference["full_pool_calibration_count"],
            "full_pool_test_misses": reference["full_pool_test_misses"],
            "full_pool_test_count": reference["full_pool_test_count"],
            "endpoint": reference["endpoint"],
            "non_certification_reason": reference["non_certification_reason"],
        })

    calls_redraw_low, calls_redraw_high = _bootstrap_mean_interval(paired_calls, repetitions=bootstrap_repetitions, seed=seed)
    risk_redraw_low, risk_redraw_high = _bootstrap_mean_interval(paired_risks, repetitions=bootstrap_repetitions, seed=seed + 1)
    calls_cluster_low, calls_cluster_high, query_count = _cluster_interval(call_clusters, repetitions=bootstrap_repetitions, seed=seed + 2)
    risk_cluster_low, risk_cluster_high, risk_query_count = _cluster_interval(risk_clusters, repetitions=bootstrap_repetitions, seed=seed + 3)
    if risk_query_count != query_count:
        raise AssertionError("call and risk query clusters differ")
    protocol, benchmark, pool_id, pool_size, alpha = cell_key
    selected_hashes = {source_id: source_hashes[source_id] for source_id in sorted(cell["source_ids"])}
    return {
        "protocol": protocol,
        "benchmark": benchmark,
        "pool_id": pool_id,
        "pool_size": pool_size,
        "target_alpha": alpha,
        "method": method_name,
        "reference_method": reference_name,
        "redraw_count": len(redraw_items),
        "paired_certified_redraw_count": len(paired_calls),
        "method_certified_redraw_count": method_summary["certification_count"],
        "reference_certified_redraw_count": reference_summary["certification_count"],
        "method_certified_mean_calls": method_summary["calls_mean"],
        "method_certified_calls_se": method_summary["calls_se"],
        "reference_certified_mean_calls": reference_summary["calls_mean"],
        "reference_certified_calls_se": reference_summary["calls_se"],
        "paired_mean_calls_difference": _mean(paired_calls),
        "paired_calls_difference_se": _se(paired_calls),
        "paired_calls_redraw_ci95_low": calls_redraw_low,
        "paired_calls_redraw_ci95_high": calls_redraw_high,
        "paired_calls_query_cluster_ci95_low": calls_cluster_low,
        "paired_calls_query_cluster_ci95_high": calls_cluster_high,
        "method_certified_standard_risk": method_summary["risk_mean"],
        "method_certified_standard_risk_se": method_summary["risk_se"],
        "reference_certified_standard_risk": reference_summary["risk_mean"],
        "reference_certified_standard_risk_se": reference_summary["risk_se"],
        "paired_mean_standard_risk_difference": _mean(paired_risks),
        "paired_standard_risk_difference_se": _se(paired_risks),
        "paired_standard_risk_redraw_ci95_low": risk_redraw_low,
        "paired_standard_risk_redraw_ci95_high": risk_redraw_high,
        "paired_standard_risk_query_cluster_ci95_low": risk_cluster_low,
        "paired_standard_risk_query_cluster_ci95_high": risk_cluster_high,
        "query_cluster_count": query_count,
        "method_test_miss_numerator": method_summary["test_misses"],
        "method_test_miss_denominator": method_summary["test_count"],
        "reference_test_miss_numerator": reference_summary["test_misses"],
        "reference_test_miss_denominator": reference_summary["test_count"],
        "method_certification_numerator": method_summary["certification_count"],
        "method_certification_denominator": method_summary["certification_total"],
        "method_certification_rate": method_summary["certification_rate"],
        "method_certification_wilson_ci95_low": method_summary["certification_low"],
        "method_certification_wilson_ci95_high": method_summary["certification_high"],
        "reference_certification_numerator": reference_summary["certification_count"],
        "reference_certification_denominator": reference_summary["certification_total"],
        "reference_certification_rate": reference_summary["certification_rate"],
        "reference_certification_wilson_ci95_low": reference_summary["certification_low"],
        "reference_certification_wilson_ci95_high": reference_summary["certification_high"],
        "method_empirical_joint_event_j_hat": method_summary["joint_event"],
        "reference_empirical_joint_event_j_hat": reference_summary["joint_event"],
        "method_full_pool_calibration_misses": method_summary["full_pool_calibration_misses"],
        "method_full_pool_calibration_denominator": method_summary["full_pool_calibration_count"],
        "method_full_pool_test_misses": method_summary["full_pool_test_misses"],
        "method_full_pool_test_denominator": method_summary["full_pool_test_count"],
        "reference_full_pool_calibration_misses": reference_summary["full_pool_calibration_misses"],
        "reference_full_pool_calibration_denominator": reference_summary["full_pool_calibration_count"],
        "reference_full_pool_test_misses": reference_summary["full_pool_test_misses"],
        "reference_full_pool_test_denominator": reference_summary["full_pool_test_count"],
        "shared_endpoint_redraw_count": shared_count,
        "shared_endpoint_certification_match_count": shared_matches,
        "shared_feasibility_equivalent": shared_count == len(redraw_items) and shared_matches == shared_count,
        "endpoint_difference_count": endpoint_difference_count,
        "certification_disagreement_count": disagreement_count,
        "certification_disagreement_reasons": dict(sorted(disagreement_reasons.items())),
        "method_per_redraw": method_redraw_details,
        "reference_per_redraw": reference_redraw_details,
        "paired_redraw_ids": paired_redraw_ids,
        "source_run_ids": sorted(cell["source_run_ids"]),
        "source_hashes": selected_hashes,
        "bootstrap_seed": seed,
        "bootstrap_repetitions": bootstrap_repetitions,
        "redraw_interval_method": "paired_redraw_percentile_bootstrap",
        "query_interval_method": "query_id_cluster_percentile_bootstrap",
        "split_pool_variability_label": "paired redraw variability over split and pool resampling",
        "query_population_uncertainty_label": "query-ID cluster bootstrap across all paired redraw occurrences",
        "difference_definition": "method_minus_reference",
        "pairing_rule": "same cell, redraw, split IDs, model IDs, and test query IDs",
        "test_retuning_used": False,
    }


def _default_pairs(methods: Sequence[str], references: Sequence[str]) -> list[tuple[str, str]]:
    available_references = [canonical_method_name(value) for value in references if canonical_method_name(value) in methods]
    if not available_references:
        raise PairedAnalysisError(f"none of the reference methods are available: {list(references)}")
    pairs: list[tuple[str, str]] = []
    nonreferences = [method for method in methods if method not in available_references]
    for method in nonreferences:
        pairs.extend((method, reference) for reference in available_references)
    if len(available_references) > 1:
        pairs.append((available_references[1], available_references[0]))
    if not pairs:
        raise PairedAnalysisError("at least two methods are required for paired analysis")
    return pairs


def analyze_records(
    records: Sequence[Mapping[str, Any] | tuple[str, Mapping[str, Any]] | tuple[str, str, Mapping[str, Any]]],
    *,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    reference_methods: Sequence[str] = ("racer_g", "exact_independence"),
    comparison_pairs: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Analyze canonical records and return deterministic JSON-ready output."""
    if isinstance(bootstrap_repetitions, bool) or not isinstance(bootstrap_repetitions, (int, np.integer)) or bootstrap_repetitions < 2:
        raise PairedAnalysisError("bootstrap_repetitions must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise PairedAnalysisError("seed must be a nonnegative integer")
    prepared: list[tuple[str, str, Mapping[str, Any]]] = []
    for index, item in enumerate(records):
        if isinstance(item, Mapping):
            source_id, record = f"memory:{index}", item
            source_hash = _hash_record(record)
        elif len(item) == 2:
            source_id, record = item
            source_id = str(source_id)
            source_hash = _hash_record(record)
        elif len(item) == 3:
            source_id, source_hash, record = item
            source_id, source_hash = str(source_id), str(source_hash)
        else:
            raise PairedAnalysisError("record entries must be mappings or source tuples")
        if not isinstance(record, Mapping):
            raise PairedAnalysisError("record payload must be a mapping")
        prepared.append((source_id, source_hash, record))
    cells, source_hashes, source_run_ids = _normalize_inputs(prepared)
    rows: list[dict[str, Any]] = []
    for cell_key in sorted(cells, key=lambda key: (str(key[0]), str(key[1]), str(key[2]), int(key[3]), float(key[4]))):
        cell = cells[cell_key]
        first_redraw = next(iter(cell["redraws"].values()))
        methods = sorted(first_redraw["methods"])
        pairs = (
            [(canonical_method_name(method), canonical_method_name(reference)) for method, reference in comparison_pairs]
            if comparison_pairs is not None
            else _default_pairs(methods, reference_methods)
        )
        for method, reference in pairs:
            if method == reference:
                raise PairedAnalysisError("comparison pair cannot contain the same method twice")
            if method not in methods or reference not in methods:
                raise PairedAnalysisError(f"comparison pair {(method, reference)} is unavailable")
            rows.append(_paired_row(cell_key, cell, method, reference, source_hashes=source_hashes, bootstrap_repetitions=int(bootstrap_repetitions), seed=int(seed)))
    return {
        "schema_version": "paired_main_suite_analysis_v1",
        "analysis_kind": "paired_uncertainty_joint_event_and_feasibility",
        "row_count": len(rows),
        "bootstrap": {
            "seed": int(seed),
            "repetitions": int(bootstrap_repetitions),
            "confidence_level": CONFIDENCE_LEVEL,
            "redraw_resampling_unit": "paired_redraw",
            "query_resampling_unit": "query_id_cluster",
        },
        "difference_definition": "method_minus_reference",
        "test_retuning_used": False,
        "source_run_ids": source_run_ids,
        "source_hashes": source_hashes,
        "rows": rows,
    }


def load_records(paths: Iterable[str | Path]) -> list[tuple[str, str, Mapping[str, Any]]]:
    loaded = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise PairedAnalysisError(f"{path} does not contain a JSON object")
        loaded.append((str(path), sha256_file(path), payload))
    if not loaded:
        raise PairedAnalysisError("at least one input path is required")
    return loaded


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for raw_row in rows:
            row = {}
            for key in CSV_COLUMNS:
                value = raw_row.get(key)
                row[key] = _canonical_json(value) if isinstance(value, (dict, list)) else value
            writer.writerow(row)
    temporary.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="canonical E2/E3 JSON records")
    parser.add_argument("--input", action="append", default=[], dest="extra_inputs", help="additional canonical JSON record; repeatable")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=DEFAULT_BOOTSTRAP_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--reference-method", action="append", dest="reference_methods")
    arguments = parser.parse_args(argv)
    input_paths = [*arguments.inputs, *arguments.extra_inputs]
    references = tuple(arguments.reference_methods or ("racer_g", "exact_independence"))
    payload = analyze_records(load_records(input_paths), bootstrap_repetitions=arguments.bootstrap_repetitions, seed=arguments.seed, reference_methods=references)
    write_csv(payload["rows"], arguments.output_csv)
    write_json(payload, arguments.output_json)
    return 0


analyze = analyze_records


if __name__ == "__main__":
    raise SystemExit(main())
