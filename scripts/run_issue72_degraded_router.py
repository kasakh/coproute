#!/usr/bin/env python3
"""Run the ISSUE-72 degraded-router sensitivity on archived labels.

This runner closes only the degraded and noisy router direction. The improved
KNN-neighborhood, MLP, and RouterDC direction is reused from ISSUE-63 and is not
rerun here. One process evaluates one registered dataset and one contiguous
redraw shard with immutable, host and time stamped output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.chain_discount import chain_discount_diagnostics  # noqa: E402
from components.racer_protocol import nonconformity_scores  # noqa: E402
from components.racer_wrapper import (  # noqa: E402
    FACTORIZED_INDEPENDENCE,
    fixed_order_factorized_scores,
    sequential_scores,
)
from components.routereval_protocol import subsample_pool  # noqa: E402
from components.scale_selectors import build_order  # noqa: E402
from scripts.run_paired_main_suite import (  # noqa: E402
    ALPHA_GRID,
    BASE_SEED,
    SPLIT_STEP,
    _knn,
    _load_task,
    _score_chain_discount_candidates,
    _select_chain_discount_candidate,
    _smooth,
    _split_query_ids,
    _target_split,
    atomic_write_json,
    evaluate_score_family,
    hash_json,
)

SCHEMA_VERSION = "issue72_degraded_router_v1"
REDRAWS = 20
KNN_K = 40
POOL_SEED = BASE_SEED
POOL_SIZE_BY_PROTOCOL = {"routereval": 100, "routerbench": 11}
BENCHMARK_BY_PROTOCOL = {"routereval": "ifeval", "routerbench": "mmlu"}
CHAIN_GRID = (0, 5, 20, 50)
NOISE_SIGMAS = (0.0, 0.05, 0.10, 0.20, 0.40)
OVERCONFIDENCE_LOGIT_SCALE = 2.0
UNDERCONFIDENCE_LOGIT_SCALE = 0.5
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260920
ECE_BINS = 15
D1_METHOD = "fixed_tally_factorized"
D2_METHOD = "fixed_tally_chain_discounted"
PRIMARY_CONTROL_METHODS = ("exact_independence", "racer_g")
MATCHED_CONTROL_METHODS = (
    "exact_independence_matched_budget",
    "racer_g_matched_budget",
)
CONTROL_METHODS = (*PRIMARY_CONTROL_METHODS, *MATCHED_CONTROL_METHODS)
REGISTERED_METHODS = (D1_METHOD, D2_METHOD, *CONTROL_METHODS)
RAW_VARIANT = "raw"
SIGMA_ZERO_VARIANT = "noise_sigma_0"
VARIANT_IDS = (
    RAW_VARIANT,
    "temperature_scaled",
    "systematic_overconfidence",
    "systematic_underconfidence",
    SIGMA_ZERO_VARIANT,
    "noise_sigma_0.05",
    "noise_sigma_0.10",
    "noise_sigma_0.20",
    "noise_sigma_0.40",
)


def _probability_matrix(values: np.ndarray, name: str = "probabilities") -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return matrix


def _binary_matrix(values: np.ndarray, name: str = "labels") -> np.ndarray:
    matrix = np.asarray(values, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError(f"{name} must be binary")
    return matrix


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = _probability_matrix(probabilities)
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return _probability_matrix(output)


def scale_logits(probabilities: np.ndarray, scale: float) -> np.ndarray:
    """Multiply finite logits by a positive deterministic scale."""

    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("logit scale must be finite and positive")
    return _sigmoid(_logit(probabilities) * float(scale))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar Bernoulli temperature scaling."""

    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return _sigmoid(_logit(probabilities) / float(temperature))


def _bernoulli_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values = np.clip(_probability_matrix(probabilities), 1e-12, 1.0 - 1e-12)
    outcomes = _binary_matrix(labels)
    if values.shape != outcomes.shape:
        raise ValueError("probabilities and labels must have the same shape")
    return float(-np.mean(outcomes * np.log(values) + (1 - outcomes) * np.log1p(-values)))


def fit_validation_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    validation_rows: Sequence[int],
) -> dict[str, Any]:
    """Fit one temperature using validation labels and no other rows."""

    values = _probability_matrix(probabilities)
    outcomes = _binary_matrix(labels)
    if values.shape != outcomes.shape:
        raise ValueError("probabilities and labels must have the same shape")
    rows = np.asarray(validation_rows, dtype=int)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("validation rows must be a non-empty vector")
    if len(rows) != len(np.unique(rows)) or np.any((rows < 0) | (rows >= len(values))):
        raise ValueError("validation rows must be unique valid indices")
    selected_probabilities = values[rows]
    selected_labels = outcomes[rows]

    def objective(log_temperature: float) -> float:
        return _bernoulli_nll(
            apply_temperature(selected_probabilities, math.exp(log_temperature)),
            selected_labels,
        )

    lower = math.log(0.125)
    upper = math.log(8.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = objective(left)
    right_value = objective(right)
    for _ in range(96):
        if left_value <= right_value:
            upper = right
            right = left
            right_value = left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        else:
            lower = left
            left = right
            left_value = right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)
    log_temperature = (lower + upper) / 2.0
    temperature = float(math.exp(log_temperature))
    return {
        "temperature": temperature,
        "validation_nll_before": _bernoulli_nll(selected_probabilities, selected_labels),
        "validation_nll_after": objective(log_temperature),
        "fit_roles": ["validation"],
        "fit_row_count": int(len(rows)),
        "fit_rows_sha256": hash_json([int(value) for value in rows]),
        "held_out_rows_used": False,
        "selection_source": "validation_only",
        "search_log_temperature_bounds": [math.log(0.125), math.log(8.0)],
        "search_iterations": 96,
    }


def deterministic_logit_noise(
    probabilities: np.ndarray,
    sigma: float,
    *,
    seed: int,
) -> np.ndarray:
    """Add deterministic Gaussian noise to logits without touching sigma zero."""

    values = _probability_matrix(probabilities)
    if not math.isfinite(float(sigma)) or float(sigma) < 0.0:
        raise ValueError("noise sigma must be finite and non-negative")
    if float(sigma) == 0.0:
        return np.array(values, copy=True)
    noise = np.random.default_rng(int(seed)).normal(size=values.shape)
    return _sigmoid(_logit(values) + float(sigma) * noise)


def calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    rows: Sequence[int],
    *,
    bins: int = ECE_BINS,
) -> dict[str, Any]:
    """Compute Bernoulli NLL, Brier score, and equal-width ECE."""

    values = _probability_matrix(probabilities)
    outcomes = _binary_matrix(labels)
    selected_rows = np.asarray(rows, dtype=int)
    if values.shape != outcomes.shape:
        raise ValueError("probabilities and labels must have the same shape")
    if selected_rows.ndim != 1 or len(selected_rows) == 0:
        raise ValueError("metric rows must be a non-empty vector")
    if bins < 1:
        raise ValueError("ECE bin count must be positive")
    selected_probabilities = values[selected_rows].ravel()
    selected_labels = outcomes[selected_rows].ravel().astype(float)
    bin_index = np.minimum((selected_probabilities * bins).astype(int), bins - 1)
    ece = 0.0
    event_counts = []
    for index in range(bins):
        mask = bin_index == index
        count = int(np.count_nonzero(mask))
        positives = int(selected_labels[mask].sum()) if count else 0
        probability_sum = float(selected_probabilities[mask].sum()) if count else 0.0
        if count:
            ece += count / len(selected_probabilities) * abs(
                probability_sum / count - positives / count
            )
        event_counts.append({
            "bin": index,
            "count": count,
            "positive_count": positives,
            "probability_sum": probability_sum,
        })
    return {
        "row_count": int(len(selected_rows)),
        "bernoulli_event_count": int(len(selected_probabilities)),
        "positive_count": int(selected_labels.sum()),
        "nll": _bernoulli_nll(values[selected_rows], outcomes[selected_rows]),
        "brier": float(np.mean((selected_probabilities - selected_labels) ** 2)),
        "ece": float(ece),
        "ece_bins": int(bins),
        "ece_bin_events": event_counts,
    }


def _noise_seed(redraw: int) -> int:
    material = f"issue72|{int(redraw)}|shared_logit_noise".encode("utf-8")
    return BOOTSTRAP_SEED + int(hashlib.sha256(material).hexdigest()[:8], 16)


def build_probability_variants(
    primary_raw: np.ndarray,
    matched_raw: np.ndarray,
    labels: np.ndarray,
    validation_rows: Sequence[int],
    *,
    redraw: int,
) -> list[dict[str, Any]]:
    """Build every predeclared variant for both isolated probability streams."""

    primary = _probability_matrix(primary_raw, "primary probabilities")
    matched = _probability_matrix(matched_raw, "matched control probabilities")
    outcomes = _binary_matrix(labels)
    if primary.shape != outcomes.shape or matched.shape != outcomes.shape:
        raise ValueError("both probability streams must align with labels")
    primary_temperature = fit_validation_temperature(primary, outcomes, validation_rows)
    matched_temperature = fit_validation_temperature(matched, outcomes, validation_rows)
    seed = _noise_seed(redraw)
    variants: list[dict[str, Any]] = [
        {
            "variant": RAW_VARIANT,
            "kind": "raw_probabilities",
            "primary_probabilities": np.array(primary, copy=True),
            "matched_control_probabilities": np.array(matched, copy=True),
            "fit_metadata": {"selection_source": "none", "fit_roles": []},
        },
        {
            "variant": "temperature_scaled",
            "kind": "validation_only_temperature_scaling",
            "primary_probabilities": apply_temperature(
                primary, primary_temperature["temperature"]
            ),
            "matched_control_probabilities": apply_temperature(
                matched, matched_temperature["temperature"]
            ),
            "fit_metadata": {
                "selection_source": "validation_only",
                "fit_roles": ["validation"],
                "primary": primary_temperature,
                "matched_control": matched_temperature,
            },
        },
        {
            "variant": "systematic_overconfidence",
            "kind": "deterministic_logit_scale",
            "primary_probabilities": scale_logits(primary, OVERCONFIDENCE_LOGIT_SCALE),
            "matched_control_probabilities": scale_logits(
                matched, OVERCONFIDENCE_LOGIT_SCALE
            ),
            "fit_metadata": {
                "selection_source": "predeclared",
                "fit_roles": [],
                "logit_scale": OVERCONFIDENCE_LOGIT_SCALE,
            },
        },
        {
            "variant": "systematic_underconfidence",
            "kind": "deterministic_logit_scale",
            "primary_probabilities": scale_logits(primary, UNDERCONFIDENCE_LOGIT_SCALE),
            "matched_control_probabilities": scale_logits(
                matched, UNDERCONFIDENCE_LOGIT_SCALE
            ),
            "fit_metadata": {
                "selection_source": "predeclared",
                "fit_roles": [],
                "logit_scale": UNDERCONFIDENCE_LOGIT_SCALE,
            },
        },
    ]
    for sigma in NOISE_SIGMAS:
        variants.append({
            "variant": "noise_sigma_0" if sigma == 0.0 else f"noise_sigma_{sigma:.2f}",
            "kind": "deterministic_gaussian_logit_noise",
            "primary_probabilities": deterministic_logit_noise(
                primary, sigma, seed=seed
            ),
            "matched_control_probabilities": deterministic_logit_noise(
                matched, sigma, seed=seed
            ),
            "fit_metadata": {
                "selection_source": "predeclared",
                "fit_roles": [],
                "sigma": float(sigma),
                "noise_seed": int(seed),
                "shared_noise_across_probability_streams": True,
            },
        })
    if tuple(record["variant"] for record in variants) != VARIANT_IDS:
        raise RuntimeError("router variant registry drift")
    sigma_zero = next(record for record in variants if record["variant"] == SIGMA_ZERO_VARIANT)
    if not np.array_equal(sigma_zero["primary_probabilities"], primary):
        raise RuntimeError("sigma zero does not reproduce raw primary probabilities exactly")
    if not np.array_equal(sigma_zero["matched_control_probabilities"], matched):
        raise RuntimeError("sigma zero does not reproduce raw matched probabilities exactly")
    return variants


def fit_probability_streams(
    embeddings: np.ndarray,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    *,
    knn_k: int = KNN_K,
    knn_function: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the primary and budget-matched control KNN streams."""

    features = np.asarray(embeddings, dtype=float)
    outcomes = _binary_matrix(labels)
    if features.ndim != 2 or features.shape[0] != outcomes.shape[0]:
        raise ValueError("embeddings and labels must align")
    if len(query_ids) != len(outcomes):
        raise ValueError("query IDs and labels must align")
    design_rows = np.asarray(split["design"], dtype=int)
    router_rows = np.asarray(split["router"], dtype=int)
    if set(map(int, design_rows)).intersection(map(int, router_rows)):
        raise ValueError("design and router rows must be disjoint")
    matched_rows = np.concatenate([design_rows, router_rows])
    expected_union = set(map(int, design_rows)).union(map(int, router_rows))
    if set(map(int, matched_rows)) != expected_union or len(matched_rows) != len(expected_union):
        raise RuntimeError("matched control rows are not exactly design plus router")
    knn_impl = knn_function or _knn
    primary, _ = knn_impl(
        features[router_rows], outcomes[router_rows], features, knn_k
    )
    matched, _ = knn_impl(
        features[matched_rows], outcomes[matched_rows], features, knn_k
    )
    primary = _probability_matrix(primary, "primary KNN probabilities")
    matched = _probability_matrix(matched, "matched control KNN probabilities")
    if primary.shape != outcomes.shape or matched.shape != outcomes.shape:
        raise ValueError("KNN probability shape differs from labels")
    design_ids = [str(query_ids[int(row)]) for row in design_rows]
    router_ids = [str(query_ids[int(row)]) for row in router_rows]
    matched_ids = [str(query_ids[int(row)]) for row in matched_rows]
    provenance = {
        "primary_router": {
            "used_by_methods": [D1_METHOD, D2_METHOD, *PRIMARY_CONTROL_METHODS],
            "excluded_from_methods": list(MATCHED_CONTROL_METHODS),
            "fit_roles": ["router"],
            "fit_rows": [int(value) for value in router_rows],
            "fit_query_ids": router_ids,
            "fit_query_ids_sha256": hash_json(router_ids),
            "fit_row_count": int(len(router_rows)),
            "knn_k": int(knn_k),
        },
        "data_budget_matched_marginal_controls": {
            "used_by_methods": list(MATCHED_CONTROL_METHODS),
            "excluded_from_methods": [D1_METHOD, D2_METHOD, *PRIMARY_CONTROL_METHODS],
            "fit_roles": ["design", "router"],
            "design_rows": [int(value) for value in design_rows],
            "router_rows": [int(value) for value in router_rows],
            "fit_rows": [int(value) for value in matched_rows],
            "design_query_ids": design_ids,
            "router_query_ids": router_ids,
            "fit_query_ids": matched_ids,
            "fit_query_ids_sha256": hash_json(matched_ids),
            "fit_row_count": int(len(matched_rows)),
            "exact_union_verified": True,
            "knn_k": int(knn_k),
        },
    }
    return primary, matched, provenance


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(result)
    output.pop("selected_model_ids_by_query", None)
    calls = output.get("test_calls_by_query")
    misses = output.get("test_miss_by_query")
    output["test_query_count"] = len(output["test_query_ids"])
    output["test_call_sum"] = None if calls is None else int(sum(calls))
    output["test_miss_count"] = None if misses is None else int(sum(misses))
    if output["certified"]:
        if len(calls) != output["test_query_count"] or len(misses) != output["test_query_count"]:
            raise RuntimeError("test event vectors differ from test query count")
        if output["test_calls"] != float(np.mean(calls)):
            raise RuntimeError("test call vector arithmetic mismatch")
        if output["standard_all_miss_risk"] != float(np.mean(misses)):
            raise RuntimeError("test miss vector arithmetic mismatch")
        if output["test_miss_count"] != int(
            round(output["standard_all_miss_risk"] * output["test_query_count"])
        ):
            raise RuntimeError("test event count arithmetic mismatch")
    elif calls is not None or misses is not None:
        raise RuntimeError("non-certified arm emitted test event vectors")
    return output


def _evaluate_method(
    scores: np.ndarray,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    alpha: float,
    method: str,
) -> dict[str, Any]:
    return _compact_result(
        evaluate_score_family(
            scores,
            labels,
            split["final_calibration"],
            split["test"],
            query_ids,
            model_ids,
            alpha,
            method_family=method,
        )
    )


def _decomposition(methods: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = (
        "racer_g",
        "exact_independence",
        D1_METHOD,
        D2_METHOD,
        "racer_g_matched_budget",
        "exact_independence_matched_budget",
    )
    if tuple(name for name in required if name not in methods):
        raise RuntimeError("decomposition is missing a registered method")

    def difference(left: str, right: str, label: str) -> dict[str, Any]:
        jointly_certified = bool(methods[left]["certified"] and methods[right]["certified"])
        value = (
            float(methods[left]["test_calls"] - methods[right]["test_calls"])
            if jointly_certified
            else None
        )
        return {
            "definition": label,
            "left_method": left,
            "right_method": right,
            "jointly_certified": jointly_certified,
            "calls": value,
        }

    return {
        "sizing_gain": difference(
            "racer_g", "exact_independence",
            "primary-router RACER-G calls minus primary-router exact-independence calls",
        ),
        "ordering_gain": difference(
            "exact_independence", D1_METHOD,
            "primary-router exact-independence calls minus primary-router fixed-tally D1 calls",
        ),
        "d2_increment": difference(
            D2_METHOD, D1_METHOD, "fixed-tally D2 calls minus fixed-tally D1 calls"
        ),
        "matched_budget_sizing_gain": difference(
            "racer_g_matched_budget", "exact_independence_matched_budget",
            "matched-budget RACER-G calls minus matched-budget exact-independence calls",
        ),
        "exact_independence_budget_effect": difference(
            "exact_independence", "exact_independence_matched_budget",
            "primary-router exact-independence calls minus matched-budget exact-independence calls",
        ),
        "racer_g_budget_effect": difference(
            "racer_g", "racer_g_matched_budget",
            "primary-router RACER-G calls minus matched-budget RACER-G calls",
        ),
    }


def _score_matched_controls(
    probabilities: np.ndarray,
    *,
    split_seed: int,
) -> dict[str, np.ndarray]:
    """Score only the two addendum controls from the matched KNN stream."""

    values = _probability_matrix(probabilities, "matched control probabilities")
    _, augmented = nonconformity_scores(
        values, "G", seed=BASE_SEED, split_seed=split_seed
    )
    exact_raw = sequential_scores(
        values,
        dependence=FACTORIZED_INDEPENDENCE,
        max_depth=values.shape[1],
    )
    return {
        "racer_g": augmented[:, :-1],
        "exact_independence": _smooth(
            exact_raw,
            base_seed=BASE_SEED,
            split_seed=split_seed,
            preserve_prefixes=True,
        ),
    }




def _score_primary_methods(
    probabilities: np.ndarray,
    design_labels: np.ndarray,
    model_ids: Sequence[str],
    *,
    split_seed: int,
) -> dict[str, np.ndarray]:
    """Score only the three primary-router families required by ISSUE-72."""

    values = _probability_matrix(probabilities, "primary router probabilities")
    order = build_order(
        "empirical_tally",
        design_labels,
        model_names=list(model_ids),
        max_depth=values.shape[1],
        seed=BASE_SEED,
    )
    order_array = np.asarray(order, dtype=int)
    raw_d1 = fixed_order_factorized_scores(
        values, order, max_depth=values.shape[1]
    )
    ordered = raw_d1[:, order_array]
    smoothed = _smooth(
        ordered,
        base_seed=BASE_SEED,
        split_seed=split_seed,
        preserve_prefixes=True,
    )
    d1_scores = np.empty_like(smoothed)
    d1_scores[:, order_array] = smoothed
    if np.any(np.diff(d1_scores[:, order_array], axis=1) < 0.0):
        raise RuntimeError("primary D1 prefix order was inverted")
    controls = _score_matched_controls(values, split_seed=split_seed)
    return {
        D1_METHOD: d1_scores,
        "exact_independence": controls["exact_independence"],
        "racer_g": controls["racer_g"],
    }


def _prepare_variant_scores(
    primary_probabilities: np.ndarray,
    matched_probabilities: np.ndarray,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    model_ids: Sequence[str],
    *,
    split_seed: int,
) -> dict[str, Any]:
    primary_scores = _score_primary_methods(
        primary_probabilities,
        labels[split["design"]],
        model_ids,
        split_seed=split_seed,
    )
    matched_scores = _score_matched_controls(
        matched_probabilities,
        split_seed=split_seed,
    )
    d1_scores = primary_scores[D1_METHOD]
    candidates, profile = _score_chain_discount_candidates(
        primary_probabilities,
        labels[split["design"]],
        model_ids,
        d1_scores,
        base_seed=BASE_SEED,
        split_seed=split_seed,
    )
    if tuple(candidates) != CHAIN_GRID:
        raise RuntimeError("D2 chain grid drift")
    if not np.array_equal(candidates[0], d1_scores):
        raise RuntimeError("D2 discount-off endpoint score mismatch")
    return {
        "primary_scores": primary_scores,
        "matched_scores": matched_scores,
        "d1_scores": d1_scores,
        "candidates": candidates,
        "profile": profile,
        "score_identity": {
            "primary_d1_score_sha256": hash_json(d1_scores.tolist()),
            "d2_endpoint_score_sha256": hash_json(candidates[0].tolist()),
        },
    }
def _score_variant(
    primary_probabilities: np.ndarray,
    matched_probabilities: np.ndarray,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    model_ids: Sequence[str],
    alpha: float,
    *,
    split_seed: int,
    prepared: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cache = dict(prepared) if prepared is not None else _prepare_variant_scores(
        primary_probabilities,
        matched_probabilities,
        labels,
        split,
        model_ids,
        split_seed=split_seed,
    )
    primary_scores = cache["primary_scores"]
    matched_scores = cache["matched_scores"]
    d1_scores = cache["d1_scores"]
    candidates = cache["candidates"]
    profile = cache["profile"]
    selected = _select_chain_discount_candidate(
        candidates, labels, split["validation"], alpha
    )
    endpoint_raw = evaluate_score_family(
        candidates[0], labels, split["final_calibration"], split["test"],
        [str(value) for value in range(len(labels))], model_ids, alpha,
        method_family=D1_METHOD,
    )
    d1_raw = evaluate_score_family(
        d1_scores, labels, split["final_calibration"], split["test"],
        [str(value) for value in range(len(labels))], model_ids, alpha,
        method_family=D1_METHOD,
    )
    if endpoint_raw != d1_raw:
        raise RuntimeError("D2 discount-off endpoint calibrated result mismatch")
    return {
        D1_METHOD: {"scores": d1_scores, "probability_stream": "primary_router"},
        D2_METHOD: {
            "scores": selected["scores"],
            "probability_stream": "primary_router",
            "chosen_hyper": dict(selected["chosen"]),
            "selection_source": "validation_only",
            "selection_fallback_used": bool(selected["selection_fallback_used"]),
            "validation_trajectory": selected["validation_trajectory"],
            "endpoint_reference_method": D1_METHOD,
            "endpoint_score_identity_verified": True,
            "endpoint_result_identity_verified": True,
            "chain_discount_profile": chain_discount_diagnostics(
                profile,
                minimum_survivor_count=int(
                    selected["chosen"]["minimum_survivor_count"]
                ),
            ),
        },
        "exact_independence": {
            "scores": primary_scores["exact_independence"],
            "probability_stream": "primary_router",
        },
        "racer_g": {
            "scores": primary_scores["racer_g"],
            "probability_stream": "primary_router",
        },
        "exact_independence_matched_budget": {
            "scores": matched_scores["exact_independence"],
            "probability_stream": "data_budget_matched_marginal_controls",
        },
        "racer_g_matched_budget": {
            "scores": matched_scores["racer_g"],
            "probability_stream": "data_budget_matched_marginal_controls",
        },
    }, dict(cache["score_identity"])


def _full_pool_record(
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    split_ids: Mapping[str, Sequence[str]],
    model_ids: Sequence[str],
) -> dict[str, Any]:
    calibration = np.all(labels[split["final_calibration"]] == 0, axis=1).astype(int)
    test = np.all(labels[split["test"]] == 0, axis=1).astype(int)
    return {
        "model_ids": list(model_ids),
        "model_ids_sha256": hash_json(list(model_ids)),
        "calibration_query_ids": list(split_ids["final_calibration"]),
        "calibration_miss_by_query": calibration.tolist(),
        "calibration_miss_count": int(calibration.sum()),
        "calibration_query_count": int(len(calibration)),
        "calibration_standard_all_miss_floor": float(np.mean(calibration)),
        "test_query_ids": list(split_ids["test"]),
        "test_miss_by_query": test.tolist(),
        "test_miss_count": int(test.sum()),
        "test_query_count": int(len(test)),
        "test_standard_all_miss_floor": float(np.mean(test)),
    }


def assert_complete_output(cells: Sequence[Mapping[str, Any]]) -> None:
    """Fail on drift, missing arms, invalid arithmetic, or surprising no-certificate cells."""

    if len(cells) != len(ALPHA_GRID):
        raise RuntimeError("missing alpha cells")
    context_by_redraw: dict[int, tuple[str, str]] = {}
    for cell in cells:
        if not cell.get("redraws"):
            raise RuntimeError("alpha cell contains no redraws")
        for redraw in cell["redraws"]:
            redraw_id = int(redraw["redraw_id"])
            identity = (str(redraw["split_ids_sha256"]), str(redraw["pool_id"]))
            prior = context_by_redraw.setdefault(redraw_id, identity)
            if prior != identity:
                raise RuntimeError("split or pool drift across alpha cells")
            variants = redraw.get("variants", [])
            if tuple(record.get("variant") for record in variants) != VARIANT_IDS:
                raise RuntimeError("missing or reordered router variants")
            variant_context = {
                (record["split_ids_sha256"], record["pool_id"]) for record in variants
            }
            if variant_context != {identity}:
                raise RuntimeError("split or pool drift across router variants")
            raw = next(record for record in variants if record["variant"] == RAW_VARIANT)
            sigma_zero = next(
                record for record in variants if record["variant"] == SIGMA_ZERO_VARIANT
            )
            if raw["probability_hashes"] != sigma_zero["probability_hashes"]:
                raise RuntimeError("sigma zero probability hashes differ from raw")
            if raw["router_metrics"] != sigma_zero["router_metrics"]:
                raise RuntimeError("sigma zero router metrics differ from raw")
            if raw["score_identity"] != sigma_zero["score_identity"]:
                raise RuntimeError("sigma zero score identities differ from raw")
            if raw["methods"] != sigma_zero["methods"]:
                raise RuntimeError("sigma zero calibrated outcomes differ from raw")
            for variant in variants:
                methods = variant.get("methods", {})
                if set(methods) != set(REGISTERED_METHODS):
                    raise RuntimeError("missing or unexpected ISSUE-72 method arm")
                if (
                    methods[D1_METHOD]["probability_stream"] != "primary_router"
                    or methods[D2_METHOD]["probability_stream"] != "primary_router"
                    or methods["exact_independence"]["probability_stream"]
                    != "primary_router"
                    or methods["racer_g"]["probability_stream"]
                    != "primary_router"
                    or methods["exact_independence_matched_budget"]["probability_stream"]
                    != "data_budget_matched_marginal_controls"
                    or methods["racer_g_matched_budget"]["probability_stream"]
                    != "data_budget_matched_marginal_controls"
                ):
                    raise RuntimeError("probability stream isolation failed")
                if not methods[D2_METHOD]["endpoint_score_identity_verified"]:
                    raise RuntimeError("D2 endpoint was not verified")
                certified_count = sum(bool(record["certified"]) for record in methods.values())
                full_pool = redraw["full_pool"]
                minimum_bound = (
                    full_pool["calibration_miss_count"] + 1.0
                ) / (full_pool["calibration_query_count"] + 1.0)
                if minimum_bound <= float(cell["alpha"]) and certified_count == 0:
                    raise RuntimeError("unexpected zero certification in a certifiable cell")
                expected = _decomposition(methods)
                if variant["decomposition"] != expected:
                    raise RuntimeError("decomposition arithmetic mismatch")
                for result in methods.values():
                    if result["certified"]:
                        if result["test_call_sum"] != sum(result["test_calls_by_query"]):
                            raise RuntimeError("test call event arithmetic mismatch")
                        if result["test_miss_count"] != sum(result["test_miss_by_query"]):
                            raise RuntimeError("test miss event arithmetic mismatch")


def deterministic_paired_bootstrap(
    values: Sequence[float],
    *,
    seed: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any] | None:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("paired bootstrap values must be a finite vector")
    if isinstance(repetitions, bool) or int(repetitions) < 2:
        raise ValueError("bootstrap repetitions must be at least two")
    if len(array) == 0:
        return None
    generator = np.random.default_rng(int(seed))
    sampled = array[
        generator.integers(0, len(array), size=(int(repetitions), len(array)))
    ].mean(axis=1)
    return {
        "paired_redraw_count": int(len(array)),
        "mean_difference": float(np.mean(array)),
        "ci95": [float(value) for value in np.quantile(sampled, [0.025, 0.975])],
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "difference_direction": "variant_minus_noise_sigma_0",
    }


def _bootstrap_seed(protocol: str, alpha: float, variant: str, method: str, metric: str) -> int:
    material = f"{protocol}|{alpha:.12g}|{variant}|{method}|{metric}".encode("utf-8")
    return BOOTSTRAP_SEED + int(hashlib.sha256(material).hexdigest()[:8], 16)


def build_bootstrap_summaries(
    protocol: str,
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build 10,000-repetition paired degradation intervals from sigma zero."""

    summaries = []
    for cell in cells:
        alpha = float(cell["alpha"])
        for variant_id in VARIANT_IDS:
            if variant_id == SIGMA_ZERO_VARIANT:
                continue
            for method in REGISTERED_METHODS:
                calls: list[float] = []
                risks: list[float] = []
                paired_ids: list[int] = []
                for redraw in cell["redraws"]:
                    variants = {row["variant"]: row for row in redraw["variants"]}
                    candidate = variants[variant_id]["methods"][method]
                    reference = variants[SIGMA_ZERO_VARIANT]["methods"][method]
                    if candidate["certified"] and reference["certified"]:
                        calls.append(float(candidate["test_calls"] - reference["test_calls"]))
                        risks.append(float(
                            candidate["standard_all_miss_risk"]
                            - reference["standard_all_miss_risk"]
                        ))
                        paired_ids.append(int(redraw["redraw_id"]))
                seed = _bootstrap_seed(protocol, alpha, variant_id, method, "calls")
                summaries.append({
                    "protocol": protocol,
                    "target_alpha": alpha,
                    "variant": variant_id,
                    "reference_variant": SIGMA_ZERO_VARIANT,
                    "method": method,
                    "jointly_certified_redraw_ids": paired_ids,
                    "calls": deterministic_paired_bootstrap(calls, seed=seed),
                    "standard_all_miss_risk": deterministic_paired_bootstrap(
                        risks, seed=seed + 1
                    ),
                })
    return summaries



def recompute_aggregates(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate calls, events, certification, and decomposition from exact vectors."""

    output: dict[str, Any] = {}
    for cell in cells:
        variants: dict[str, Any] = {}
        for variant_id in VARIANT_IDS:
            records = [
                next(row for row in redraw["variants"] if row["variant"] == variant_id)
                for redraw in cell["redraws"]
            ]
            methods = {}
            for method in REGISTERED_METHODS:
                method_records = [record["methods"][method] for record in records]
                certified = [record for record in method_records if record["certified"]]
                calls = [
                    int(value)
                    for record in certified
                    for value in record["test_calls_by_query"]
                ]
                misses = [
                    int(value)
                    for record in certified
                    for value in record["test_miss_by_query"]
                ]
                methods[method] = {
                    "redraw_count": len(method_records),
                    "certified_redraw_count": len(certified),
                    "certification_rate": len(certified) / len(method_records),
                    "test_query_count": len(calls),
                    "test_call_sum": int(sum(calls)),
                    "mean_calls": float(np.mean(calls)) if calls else None,
                    "test_miss_count": int(sum(misses)),
                    "standard_all_miss_risk": float(np.mean(misses)) if misses else None,
                }
            decomposition = {}
            for name in ("sizing_gain", "ordering_gain", "d2_increment"):
                values = [
                    float(record["decomposition"][name]["calls"])
                    for record in records
                    if record["decomposition"][name]["calls"] is not None
                ]
                decomposition[name] = {
                    "jointly_certified_redraw_count": len(values),
                    "mean_calls": float(np.mean(values)) if values else None,
                }
            variants[variant_id] = {
                "methods": methods,
                "decomposition": decomposition,
            }
        output[str(cell["alpha"])] = {"variants": variants}
    return output


def recompute_router_metric_aggregates(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate router diagnostics once because they do not depend on alpha."""

    if not cells:
        return {}
    output: dict[str, Any] = {}
    redraws = cells[0]["redraws"]
    for variant_id in VARIANT_IDS:
        records = [
            next(row for row in redraw["variants"] if row["variant"] == variant_id)
            for redraw in redraws
        ]
        streams = {}
        for stream in ("primary_router", "data_budget_matched_marginal_controls"):
            splits = {}
            for split_name in ("validation", "test"):
                metrics = {}
                for metric in ("nll", "brier", "ece"):
                    values = [
                        float(record["router_metrics"][stream][split_name][metric])
                        for record in records
                    ]
                    metrics[metric] = {
                        "redraw_count": len(values),
                        "mean": float(np.mean(values)),
                    }
                splits[split_name] = metrics
            streams[stream] = splits
        output[variant_id] = streams
    return output


def build_router_metric_bootstrap_summaries(
    protocol: str,
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair each router diagnostic against sigma zero over redraws."""

    if not cells:
        return []
    redraws = cells[0]["redraws"]
    summaries = []
    for variant_id in VARIANT_IDS:
        if variant_id == SIGMA_ZERO_VARIANT:
            continue
        for stream in ("primary_router", "data_budget_matched_marginal_controls"):
            for split_name in ("validation", "test"):
                for metric in ("nll", "brier", "ece"):
                    differences = []
                    paired_ids = []
                    for redraw in redraws:
                        variants = {row["variant"]: row for row in redraw["variants"]}
                        candidate = variants[variant_id]["router_metrics"][stream][split_name][metric]
                        reference = variants[SIGMA_ZERO_VARIANT]["router_metrics"][stream][split_name][metric]
                        differences.append(float(candidate - reference))
                        paired_ids.append(int(redraw["redraw_id"]))
                    seed = _bootstrap_seed(
                        protocol,
                        0.0,
                        variant_id,
                        stream,
                        f"{split_name}_{metric}",
                    )
                    summaries.append({
                        "protocol": protocol,
                        "variant": variant_id,
                        "reference_variant": SIGMA_ZERO_VARIANT,
                        "probability_stream": stream,
                        "split": split_name,
                        "metric": metric,
                        "paired_redraw_ids": paired_ids,
                        "difference": deterministic_paired_bootstrap(
                            differences, seed=seed
                        ),
                    })
    return summaries

def run_issue72(
    task: Any,
    *,
    protocol: str,
    host: str,
    stamp: str,
    redraw_start: int = 0,
    count: int = REDRAWS,
    knn_function: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Run one registered dataset and one contiguous redraw shard."""

    if protocol not in BENCHMARK_BY_PROTOCOL:
        raise ValueError("protocol must be routereval or routerbench")
    expected_benchmark = BENCHMARK_BY_PROTOCOL[protocol]
    if str(task.task) != expected_benchmark:
        raise ValueError(
            f"ISSUE-72 {protocol} is registered only for {expected_benchmark}"
        )
    if redraw_start < 0 or count < 1 or redraw_start + count > REDRAWS:
        raise ValueError(f"redraw shard must lie inside the registered 0 to {REDRAWS} range")
    labels_all = _binary_matrix(task.correctness, "task correctness")
    embeddings = np.asarray(task.embeddings, dtype=float)
    query_ids = [str(value) for value in task.query_ids]
    all_model_ids = [str(value) for value in task.model_ids]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("task query IDs must be unique")
    if len(all_model_ids) != len(set(all_model_ids)):
        raise ValueError("task model IDs must be unique")
    if labels_all.shape[0] != len(query_ids) or embeddings.shape[0] != len(query_ids):
        raise ValueError("task labels, embeddings, and query IDs must align")
    pool_size = POOL_SIZE_BY_PROTOCOL[protocol]
    if protocol == "routereval":
        columns = subsample_pool(len(all_model_ids), pool_size, seed=POOL_SEED)
    else:
        if len(all_model_ids) != pool_size:
            raise ValueError("RouterBench MMLU must use the complete registered 11-model pool")
        columns = np.arange(pool_size, dtype=int)
    labels = labels_all[:, columns]
    model_ids = [all_model_ids[int(index)] for index in columns]
    pool_id = hash_json({"pool_seed": POOL_SEED, "model_ids": model_ids})
    cells = [{"alpha": float(alpha), "redraws": []} for alpha in ALPHA_GRID]
    redraw_contexts = []

    for local_redraw in range(count):
        redraw = redraw_start + local_redraw
        split = _target_split(len(query_ids), redraw, BASE_SEED)
        split_ids = _split_query_ids(task, split)
        split_hash = hash_json(split_ids)
        primary_raw, matched_raw, stream_provenance = fit_probability_streams(
            embeddings,
            labels,
            split,
            query_ids,
            knn_k=KNN_K,
            knn_function=knn_function,
        )
        variants = build_probability_variants(
            primary_raw,
            matched_raw,
            labels,
            split["validation"],
            redraw=redraw,
        )
        full_pool = _full_pool_record(labels, split, split_ids, model_ids)
        context = {
            "redraw_id": redraw,
            "split_seed": BASE_SEED + SPLIT_STEP * redraw,
            "split_ids": split_ids,
            "split_ids_sha256": split_hash,
            "pool_id": pool_id,
            "probability_streams": stream_provenance,
        }
        context["context_sha256"] = hash_json(context)
        redraw_contexts.append(context)

        prepared_variants = {}
        for variant in variants:
            prepared_variants[variant["variant"]] = _prepare_variant_scores(
                _probability_matrix(variant["primary_probabilities"]),
                _probability_matrix(variant["matched_control_probabilities"]),
                labels,
                split,
                model_ids,
                split_seed=context["split_seed"],
            )

        for cell in cells:
            alpha = float(cell["alpha"])
            variant_records = []
            for variant in variants:
                primary = _probability_matrix(variant["primary_probabilities"])
                matched = _probability_matrix(variant["matched_control_probabilities"])
                fitted, score_identity = _score_variant(
                    primary,
                    matched,
                    labels,
                    split,
                    model_ids,
                    alpha,
                    split_seed=context["split_seed"],
                    prepared=prepared_variants[variant["variant"]],
                )
                methods = {}
                for method in REGISTERED_METHODS:
                    metadata = dict(fitted[method])
                    scores = metadata.pop("scores")
                    result = _evaluate_method(
                        scores, labels, split, query_ids, model_ids, alpha, method
                    )
                    result.update(metadata)
                    methods[method] = result
                if set(methods) != set(REGISTERED_METHODS):
                    raise RuntimeError("ISSUE-72 method registry mismatch")
                variant_records.append({
                    "variant": variant["variant"],
                    "kind": variant["kind"],
                    "split_ids_sha256": split_hash,
                    "pool_id": pool_id,
                    "fit_metadata": variant["fit_metadata"],
                    "probability_hashes": {
                        "primary_router": hash_json(primary.tolist()),
                        "data_budget_matched_marginal_controls": hash_json(matched.tolist()),
                    },
                    "router_metrics": {
                        "primary_router": {
                            "validation": calibration_metrics(
                                primary, labels, split["validation"]
                            ),
                            "test": calibration_metrics(primary, labels, split["test"]),
                        },
                        "data_budget_matched_marginal_controls": {
                            "validation": calibration_metrics(
                                matched, labels, split["validation"]
                            ),
                            "test": calibration_metrics(matched, labels, split["test"]),
                        },
                    },
                    "score_identity": score_identity,
                    "methods": methods,
                    "decomposition": _decomposition(methods),
                })
            cell["redraws"].append({
                "redraw_id": redraw,
                "split_seed": context["split_seed"],
                "split_ids": split_ids,
                "split_ids_sha256": split_hash,
                "pool_id": pool_id,
                "redraw_context_sha256": context["context_sha256"],
                "full_pool": full_pool,
                "variants": variant_records,
            })

    assert_complete_output(cells)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": 72,
        "design": "degraded_router_probability_sensitivity",
        "scope_statement": (
            "This runner closes only the degraded and noisy router remainder. "
            "The KNN-neighborhood, MLP, and RouterDC improved-router direction is "
            "reused from ISSUE-63 and is not rerun here."
        ),
        "host": str(host),
        "stamp": str(stamp),
        "protocol": protocol,
        "benchmark": expected_benchmark,
        "pool_size": pool_size,
        "pool_seed": POOL_SEED,
        "pool_id": pool_id,
        "model_ids": model_ids,
        "model_ids_sha256": hash_json(model_ids),
        "sampled_model_column_indices": [int(value) for value in columns],
        "query_ids_sha256": hash_json(query_ids),
        "redraw_start": redraw_start,
        "redraw_count": count,
        "redraw_end": redraw_start + count,
        "registered_total_redraws": REDRAWS,
        "base_seed": BASE_SEED,
        "knn_k": KNN_K,
        "target_grid": list(ALPHA_GRID),
        "preserve_prefixes": True,
        "chain_discount_grid": list(CHAIN_GRID),
        "noise_sigmas": list(NOISE_SIGMAS),
        "variant_identities": list(VARIANT_IDS),
        "method_identities": list(REGISTERED_METHODS),
        "primary_probability_stream_methods": [
            D1_METHOD, D2_METHOD, *PRIMARY_CONTROL_METHODS
        ],
        "data_budget_matched_control_methods": list(MATCHED_CONTROL_METHODS),
        "data_budget_statement": (
            "Separately named matched-budget RACER-G and exact-independence controls use "
            "a KNN fit on the exact union of design and router labels. Primary RACER-G, "
            "primary exact independence, and both dependence methods use only the "
            "primary router-split KNN probabilities."
        ),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "source_hashes": dict(task.source_hashes),
        "feature_audit": dict(task.feature_audit),
        "redraw_contexts": redraw_contexts,
        "cells": cells,
    }
    payload["aggregates"] = recompute_aggregates(cells)
    payload["router_metric_aggregates"] = recompute_router_metric_aggregates(cells)
    payload["paired_degradation_bootstrap"] = build_bootstrap_summaries(protocol, cells)
    payload["paired_router_metric_degradation_bootstrap"] = (
        build_router_metric_bootstrap_summaries(protocol, cells)
    )
    return payload


def _safe_identity(value: str, name: str) -> str:
    text = str(value)
    if not text or re.fullmatch(r"[A-Za-z0-9_.-]+", text) is None:
        raise ValueError(f"{name} must use only letters, digits, dot, underscore, or hyphen")
    return text


def publish_output(out: str | Path, payload: Mapping[str, Any]) -> Path:
    """Publish one immutable stamped result tree through an atomic rename."""

    destination = Path(out)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output collision {temporary}")
    temporary.mkdir()
    try:
        identity = (
            f"issue72_{payload['host']}_{payload['stamp']}_{payload['protocol']}_"
            f"{payload['benchmark']}_r{payload['redraw_start']:02d}_"
            f"r{payload['redraw_end']:02d}"
        )
        result_path = temporary / f"{identity}_results.json"
        atomic_write_json(result_path, payload)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "issue": 72,
            "host": payload["host"],
            "stamp": payload["stamp"],
            "protocol": payload["protocol"],
            "benchmark": payload["benchmark"],
            "redraw_start": payload["redraw_start"],
            "redraw_end": payload["redraw_end"],
            "result_file": result_path.name,
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        }
        atomic_write_json(temporary / f"{identity}_manifest.json", manifest)
        done = temporary / f"{identity}_DONE"
        with done.open("x", encoding="utf-8") as handle:
            handle.write("complete\n")
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite immutable output {destination}")
        os.rename(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, choices=("routereval", "routerbench"))
    parser.add_argument("--leaderboard", type=Path)
    parser.add_argument("--router-dir", type=Path)
    parser.add_argument("--routerbench-pickle", type=Path)
    parser.add_argument("--embedding-archive", type=Path)
    parser.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--redraw-start", required=True, type=int)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        host = _safe_identity(args.host, "host")
        stamp = _safe_identity(args.stamp, "stamp")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite immutable output {args.out}")
    benchmark = BENCHMARK_BY_PROTOCOL[args.protocol]
    try:
        task = _load_task(
            args.protocol,
            benchmark,
            leaderboard=args.leaderboard,
            router_dir=args.router_dir,
            routerbench_pickle=args.routerbench_pickle,
            embedding_archive=args.embedding_archive,
            embedding_manifest=args.embedding_manifest,
        )
        payload = run_issue72(
            task,
            protocol=args.protocol,
            host=host,
            stamp=stamp,
            redraw_start=args.redraw_start,
            count=args.count,
        )
        published = publish_output(args.out, payload)
    except (FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"status": "complete", "output": str(published)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
