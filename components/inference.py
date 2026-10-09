"""Statistical inference for paired DACR evaluations.

The query is the resampling unit. Baseline and corrected observations are
always sampled together, optionally within benchmark/task strata. Repeated
train/test split outcomes are deliberately not treated as independent trials.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SUPPORTED_METRICS = ("calls", "risk", "accuracy", "cost")
DEFAULT_ALTERNATIVES = {
    "calls": "less",
    "risk": "less",
    "accuracy": "greater",
    "cost": "less",
}
DEFAULT_ENDPOINT_REGISTRY = {
    "calls": {"role": "primary", "alternative": "less"},
    "risk": {"role": "secondary", "alternative": "less"},
    "accuracy": {"role": "secondary", "alternative": "greater"},
    "cost": {"role": "secondary", "alternative": "less"},
}
_VALID_ROLES = {"primary", "secondary"}
_VALID_ALTERNATIVES = {"less", "greater", "two-sided"}


def holm_adjust(p_values: Mapping[str, float] | Sequence[float]):
    """Return Holm adjusted p-values while preserving input shape and order."""
    is_mapping = isinstance(p_values, Mapping)
    if is_mapping:
        keys = list(p_values)
        raw = [p_values[key] for key in keys]
    else:
        if isinstance(p_values, (str, bytes)):
            raise TypeError("p_values must be a mapping or a numeric sequence")
        keys = list(range(len(p_values)))
        raw = list(p_values)

    values = np.asarray(raw, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and between 0 and 1")
    if len(values) == 0:
        return {} if is_mapping else []

    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running_max = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(values[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max

    if is_mapping:
        return {key: float(adjusted[index]) for index, key in enumerate(keys)}
    return [float(value) for value in adjusted]


holm_correction = holm_adjust


def validate_primary_endpoint_registry(
    registry: Mapping[str, Any] | Sequence[str],
    metrics: Sequence[str] = SUPPORTED_METRICS,
) -> dict[str, dict[str, str]]:
    """Validate and normalize a preregistered endpoint family.

    A mapping must register every analyzed metric. Values may be role strings,
    primary booleans, or mappings with ``role``/``primary`` and ``alternative``.
    A sequence is interpreted as the names of primary endpoints; all other
    analyzed metrics become secondary endpoints.
    """
    metric_names = list(metrics)
    if not metric_names:
        raise ValueError("at least one metric is required")
    if len(set(metric_names)) != len(metric_names):
        raise ValueError("metric names must be unique")
    unknown_metrics = set(metric_names) - set(SUPPORTED_METRICS)
    if unknown_metrics:
        raise ValueError(f"unsupported metrics: {sorted(unknown_metrics)}")

    normalized: dict[str, dict[str, str]] = {}
    if isinstance(registry, Mapping):
        missing = set(metric_names) - set(registry)
        extra = set(registry) - set(metric_names)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing endpoints: {sorted(missing)}")
            if extra:
                details.append(f"unknown endpoints: {sorted(extra)}")
            raise ValueError("; ".join(details))

        for metric in metric_names:
            specification = registry[metric]
            alternative = DEFAULT_ALTERNATIVES[metric]
            if isinstance(specification, bool):
                role = "primary" if specification else "secondary"
            elif isinstance(specification, str):
                role = specification
            elif isinstance(specification, Mapping):
                if "role" in specification and "primary" in specification:
                    expected = "primary" if specification["primary"] else "secondary"
                    if specification["role"] != expected:
                        raise ValueError(f"conflicting role declarations for {metric}")
                if "role" in specification:
                    role = specification["role"]
                elif "primary" in specification:
                    if not isinstance(specification["primary"], bool):
                        raise TypeError(f"primary flag for {metric} must be boolean")
                    role = "primary" if specification["primary"] else "secondary"
                else:
                    raise ValueError(f"endpoint {metric} must declare role or primary")
                alternative = specification.get("alternative", alternative)
            else:
                raise TypeError(f"invalid endpoint specification for {metric}")

            if role not in _VALID_ROLES:
                raise ValueError(f"invalid role for {metric}: {role}")
            if alternative not in _VALID_ALTERNATIVES:
                raise ValueError(f"invalid alternative for {metric}: {alternative}")
            normalized[metric] = {"role": role, "alternative": alternative}
    else:
        if isinstance(registry, (str, bytes)):
            raise TypeError("registry must not be a string")
        primary_names = list(registry)
        if len(primary_names) != len(set(primary_names)):
            raise ValueError("primary endpoint names must be unique")
        unknown = set(primary_names) - set(metric_names)
        if unknown:
            raise ValueError(f"unknown primary endpoints: {sorted(unknown)}")
        for metric in metric_names:
            normalized[metric] = {
                "role": "primary" if metric in primary_names else "secondary",
                "alternative": DEFAULT_ALTERNATIVES[metric],
            }

    if not any(spec["role"] == "primary" for spec in normalized.values()):
        raise ValueError("registry must contain at least one primary endpoint")
    return normalized


validate_endpoint_registry = validate_primary_endpoint_registry


def _numeric_vector(values: Any, name: str, metric: str | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if metric in {"risk", "accuracy"} and np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be between 0 and 1")
    if metric in {"calls", "cost"} and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _validate_bootstrap_options(
    n_bootstrap: int, confidence_level: float, seed: int
) -> None:
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise TypeError("n_bootstrap must be an integer")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")


def _is_missing_label(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.asarray(value != value).item())
    except (TypeError, ValueError):
        return False


def _stratum_groups(strata: Any, n_queries: int) -> list[np.ndarray]:
    if strata is None:
        return [np.arange(n_queries, dtype=int)]

    if isinstance(strata, Mapping):
        if not strata:
            raise ValueError("strata mapping must not be empty")
        names = sorted(strata)
        columns = []
        for name in names:
            column = np.asarray(strata[name], dtype=object)
            if column.ndim != 1 or len(column) != n_queries:
                raise ValueError(f"stratum column {name} must match query count")
            columns.append(column.tolist())
        labels = [tuple(column[index] for column in columns) for index in range(n_queries)]
    else:
        array = np.asarray(strata, dtype=object)
        if array.ndim != 1 or len(array) != n_queries:
            raise ValueError("strata must be one-dimensional and match query count")
        labels = array.tolist()

    grouped: dict[Any, list[int]] = {}
    for index, label in enumerate(labels):
        values = label if isinstance(label, tuple) else (label,)
        if any(_is_missing_label(value) for value in values):
            raise ValueError("strata must not contain missing labels")
        try:
            grouped.setdefault(label, []).append(index)
        except TypeError as error:
            raise TypeError("stratum labels must be hashable") from error
    return [np.asarray(indices, dtype=int) for indices in grouped.values()]


def _interval(samples: np.ndarray, confidence_level: float) -> tuple[float, float]:
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return float(low), float(high)


def _bootstrap_p_value(
    bootstrap_differences: np.ndarray,
    estimate: float,
    alternative: str,
) -> float:
    centered = bootstrap_differences - estimate
    if alternative == "less":
        extreme = np.count_nonzero(centered <= estimate)
    elif alternative == "greater":
        extreme = np.count_nonzero(centered >= estimate)
    else:
        extreme = np.count_nonzero(np.abs(centered) >= abs(estimate))
    return float((extreme + 1) / (len(centered) + 1))


def _summary(estimate: float, bootstrap: np.ndarray, confidence_level: float) -> dict[str, float]:
    low, high = _interval(bootstrap, confidence_level)
    return {"estimate": float(estimate), "ci_low": low, "ci_high": high}


def infer_paired_metrics(
    baseline: Mapping[str, Any],
    corrected: Mapping[str, Any],
    *,
    endpoint_registry: Mapping[str, Any] | Sequence[str] = DEFAULT_ENDPOINT_REGISTRY,
    metrics: Sequence[str] = SUPPORTED_METRICS,
    strata: Any = None,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Infer paired arm estimates and corrected-minus-baseline differences.

    Primary p-values are reported without multiplicity adjustment. Holm's
    procedure is applied once across the registered secondary endpoint family.
    """
    _validate_bootstrap_options(n_bootstrap, confidence_level, seed)
    registry = validate_primary_endpoint_registry(endpoint_registry, metrics)

    baseline_columns: dict[str, np.ndarray] = {}
    corrected_columns: dict[str, np.ndarray] = {}
    query_count: int | None = None
    for metric in metrics:
        if metric not in baseline or metric not in corrected:
            raise ValueError(f"both arms must contain metric {metric}")
        baseline_column = _numeric_vector(baseline[metric], f"baseline {metric}", metric)
        corrected_column = _numeric_vector(corrected[metric], f"corrected {metric}", metric)
        if len(baseline_column) != len(corrected_column):
            raise ValueError(f"paired arms have different lengths for {metric}")
        if query_count is None:
            query_count = len(baseline_column)
        elif len(baseline_column) != query_count:
            raise ValueError("all metrics must use the same paired queries")
        baseline_columns[metric] = baseline_column
        corrected_columns[metric] = corrected_column

    assert query_count is not None
    groups = _stratum_groups(strata, query_count)
    metric_names = list(metrics)
    baseline_matrix = np.column_stack([baseline_columns[name] for name in metric_names])
    corrected_matrix = np.column_stack([corrected_columns[name] for name in metric_names])
    baseline_bootstrap = np.empty((n_bootstrap, len(metric_names)), dtype=float)
    corrected_bootstrap = np.empty_like(baseline_bootstrap)

    rng = np.random.default_rng(int(seed))
    for draw in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        baseline_bootstrap[draw] = baseline_matrix[sampled].mean(axis=0)
        corrected_bootstrap[draw] = corrected_matrix[sampled].mean(axis=0)

    results: dict[str, dict[str, Any]] = {}
    secondary_raw: dict[str, float] = {}
    for index, metric in enumerate(metric_names):
        baseline_estimate = float(baseline_matrix[:, index].mean())
        corrected_estimate = float(corrected_matrix[:, index].mean())
        difference_estimate = corrected_estimate - baseline_estimate
        difference_bootstrap = corrected_bootstrap[:, index] - baseline_bootstrap[:, index]
        specification = registry[metric]
        p_value = _bootstrap_p_value(
            difference_bootstrap,
            difference_estimate,
            specification["alternative"],
        )
        if specification["role"] == "secondary":
            secondary_raw[metric] = p_value

        if difference_estimate < 0.0:
            observed_direction = "corrected_lower"
        elif difference_estimate > 0.0:
            observed_direction = "corrected_higher"
        else:
            observed_direction = "identical"

        results[metric] = {
            "baseline": _summary(
                baseline_estimate, baseline_bootstrap[:, index], confidence_level
            ),
            "corrected": _summary(
                corrected_estimate, corrected_bootstrap[:, index], confidence_level
            ),
            "difference": _summary(
                difference_estimate, difference_bootstrap, confidence_level
            ),
            "difference_definition": "corrected_minus_baseline",
            "observed_direction": observed_direction,
            "role": specification["role"],
            "alternative": specification["alternative"],
            "p_value": p_value,
            "p_value_holm": None,
        }

    for metric, adjusted in holm_adjust(secondary_raw).items():
        results[metric]["p_value_holm"] = adjusted

    return {
        "metrics": results,
        "n_queries": query_count,
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": float(confidence_level),
        "seed": int(seed),
        "stratified": strata is not None,
        "n_strata": len(groups),
        "resampling_unit": "query_pair",
        "method": "paired_query_bootstrap",
        "endpoint_registry": registry,
        "secondary_correction": "holm",
    }


def paired_query_bootstrap(
    baseline: Any,
    corrected: Any,
    *,
    metric: str,
    strata: Any = None,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    alternative: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for one paired endpoint."""
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"unsupported metric: {metric}")
    selected_alternative = alternative or DEFAULT_ALTERNATIVES[metric]
    registry = {
        metric: {"role": "primary", "alternative": selected_alternative}
    }
    report = infer_paired_metrics(
        {metric: baseline},
        {metric: corrected},
        endpoint_registry=registry,
        metrics=(metric,),
        strata=strata,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    result = dict(report["metrics"][metric])
    result.update(
        {
            "metric": metric,
            "n_queries": report["n_queries"],
            "n_bootstrap": report["n_bootstrap"],
            "confidence_level": report["confidence_level"],
            "seed": report["seed"],
            "stratified": report["stratified"],
            "n_strata": report["n_strata"],
            "resampling_unit": report["resampling_unit"],
            "method": report["method"],
        }
    )
    return result


paired_bootstrap_ci = paired_query_bootstrap
stratified_paired_bootstrap = paired_query_bootstrap
