#!/usr/bin/env python3
"""Run ISSUE-67 RouterEval pool composition and availability stress tests.

The runner builds duplicate-growth and specialist-growth pool paths from design
labels only. Each declared pool is frozen before validation, final calibration,
and test evaluation. Availability interventions are emitted separately. The
unchanged deployment path keeps the original fitted scores and threshold. The
refit and recalibrated path is clearly marked and uses the changed pool's
legitimate design, router, validation, and final calibration labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from components.chain_discount import (  # noqa: E402
    chain_discount_diagnostics,
    chain_discounted_scores,
    fit_chain_discount,
)
from components.racer_protocol import select_standard_crc_threshold  # noqa: E402
from scripts.run_issue78_o2_full_matrix import _select_candidate, _smooth_fixed  # noqa: E402
from scripts.run_paired_main_suite import (  # noqa: E402
    ALPHA_GRID,
    BASE_SEED,
    SPLIT_STEP,
    _knn,
    _load_task,
    _score_nonlearned,
    _split_query_ids,
    _target_split,
    atomic_write_json,
    evaluate_score_family,
    hash_json,
)

SCHEMA_VERSION = "issue67_pool_shift_v1"
POOL_SIZES = (2, 4, 8, 16, 32)
REDRAWS = 20
KNN_K = 40
CHAIN_GRID = (0, 5, 20, 50)
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260920
REGISTERED_METHODS = (
    "fixed_tally_factorized",
    "fixed_tally_chain_discounted",
    "exact_independence",
    "fixed_design_marginal_factorized",
    "racer_g",
    "racer_p",
)
PATH_NAMES = ("duplicate_growth", "specialist_growth")
STRESS_KINDS = ("remove_core_model", "add_held_out_specialist")


def normalize_family_label(model_id: str) -> str:
    """Return a stable lowercase RouterEval model-family label."""

    text = str(model_id).strip().lower()
    if not text:
        raise ValueError("model IDs must be non-empty")
    tail = text.rsplit("/", 1)[-1]
    tail = re.sub(r"(?:[-_.](?:chat|instruct|it|sft|dpo|rlhf|checkpoint|ckpt)).*$", "", tail)
    tail = re.sub(r"(?:[-_.](?:\d+(?:\.\d+)?b|v\d+)).*$", "", tail)
    tokens = [token for token in re.split(r"[-_.]+", tail) if token]
    if not tokens:
        raise ValueError(f"cannot normalize model family for {model_id!r}")
    return tokens[0]


def _binary_labels(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int8)
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("design labels must be a non-empty matrix")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("design labels must be binary")
    return labels


def _quality_filter(
    labels: np.ndarray,
    needed: int,
    family_labels: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    accuracy = labels.mean(axis=0)
    positive = accuracy[accuracy > 0.0]
    if len(positive) < needed:
        raise ValueError(f"only {len(positive)} models have positive design accuracy; need {needed}")
    family_best: dict[str, float] = {}
    for family, value in zip(family_labels, accuracy):
        family_best[family] = max(family_best.get(family, 0.0), float(value))
    positive_family = sorted((value for value in family_best.values() if value > 0.0), reverse=True)
    if len(positive_family) < needed:
        raise ValueError(
            f"only {len(positive_family)} families have positive design accuracy; need {needed}"
        )
    base_floor = max(0.05, float(np.quantile(positive, 0.25)))
    model_rank_floor = float(np.sort(positive)[-needed])
    family_rank_floor = float(positive_family[needed - 1])
    floor = min(base_floor, model_rank_floor, family_rank_floor)
    eligible = np.flatnonzero(accuracy >= floor)
    if len(eligible) < needed:
        raise RuntimeError("deterministic quality filter retained too few models")
    return eligible, {
        "rule": (
            "accuracy at least min(max(0.05, positive q25), needed-th best model, "
            "needed-th best family representative)"
        ),
        "positive_model_count": int(len(positive)),
        "positive_family_count": int(len(positive_family)),
        "needed_model_count": int(needed),
        "base_floor": base_floor,
        "model_rank_floor": model_rank_floor,
        "family_rank_floor": family_rank_floor,
        "applied_floor": floor,
        "eligible_model_count": int(len(eligible)),
        "eligible_family_count": len({family_labels[int(index)] for index in eligible}),
    }


def _argmax_tie(values: np.ndarray, candidates: Sequence[int], secondary: np.ndarray) -> int:
    return min(
        (int(index) for index in candidates),
        key=lambda index: (-float(values[index]), -float(secondary[index]), index),
    )


def construct_pool_paths(
    design_labels: np.ndarray,
    model_ids: Sequence[str],
    *,
    pool_sizes: Sequence[int] = POOL_SIZES,
) -> dict[str, Any]:
    """Construct both nested paths using only the supplied design labels."""

    labels = _binary_labels(design_labels)
    ids = [str(value) for value in model_ids]
    if labels.shape[1] != len(ids):
        raise ValueError("design-label width differs from model IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate model IDs in the real candidate pool")
    sizes = tuple(int(value) for value in pool_sizes)
    if not sizes or sizes[0] != 2 or any(a >= b for a, b in zip(sizes, sizes[1:])):
        raise ValueError("pool sizes must be strictly increasing and start at two")
    needed = max(sizes) + 1
    families = [normalize_family_label(value) for value in ids]
    eligible, quality = _quality_filter(labels, needed, families)
    accuracy = labels.mean(axis=0)

    first = _argmax_tie(accuracy, eligible, accuracy)
    residual = labels[:, first] == 0
    second_candidates = [
        int(index) for index in eligible if int(index) != first and families[int(index)] != families[first]
    ]
    if not second_candidates:
        raise ValueError("competent core requires two eligible model families")
    gains = labels[residual].sum(axis=0).astype(float)
    second = _argmax_tie(gains, second_candidates, accuracy)
    core = [first, second]

    duplicate = list(core)
    while len(duplicate) < max(sizes):
        candidates = [int(index) for index in eligible if int(index) not in duplicate]
        core_matrix = labels[:, duplicate]
        agreement = np.zeros(labels.shape[1], dtype=float)
        for index in candidates:
            agreement[index] = float(np.mean(labels[:, index, None] == core_matrix))
        duplicate.append(_argmax_tie(agreement, candidates, accuracy))

    specialist = list(core)
    while len(specialist) < needed:
        covered = np.any(labels[:, specialist] == 1, axis=1)
        residual = ~covered
        used_families = {families[index] for index in specialist}
        candidates = [
            int(index)
            for index in eligible
            if int(index) not in specialist and families[int(index)] not in used_families
        ]
        if not candidates:
            raise ValueError(f"specialist path requires {needed} quality-filtered distinct families")
        gains = labels[residual].sum(axis=0).astype(float)
        specialist.append(_argmax_tie(gains, candidates, accuracy))

    def record(name: str, order: Sequence[int]) -> dict[str, Any]:
        pools = {}
        for size in sizes:
            columns = [int(value) for value in order[:size]]
            pool_ids = [ids[index] for index in columns]
            if len(pool_ids) != len(set(pool_ids)):
                raise RuntimeError(f"{name} size {size} contains duplicate model IDs")
            pools[str(size)] = {
                "size": size,
                "column_indices": columns,
                "model_ids": pool_ids,
                "family_labels": [families[index] for index in columns],
                "pool_sha256": hash_json(pool_ids),
            }
        return {
            "name": name,
            "construction_roles": ["design"],
            "held_out_roles_used": False,
            "order_through_largest_pool": [int(value) for value in order[: max(sizes)]],
            "pools": pools,
        }

    output = {
        "construction_source": "design_labels_only",
        "design_label_sha256": hash_json(labels.tolist()),
        "model_ids_sha256": hash_json(ids),
        "normalized_family_labels": families,
        "quality_filter": quality,
        "competent_core_columns": core,
        "competent_core_model_ids": [ids[index] for index in core],
        "competent_core_family_labels": [families[index] for index in core],
        "paths": {
            "duplicate_growth": record("duplicate_growth", duplicate),
            "specialist_growth": record("specialist_growth", specialist),
        },
        "held_out_specialist_order": [int(value) for value in specialist],
    }
    validate_pool_paths(output, sizes)
    return output


def validate_pool_paths(paths: Mapping[str, Any], pool_sizes: Sequence[int] = POOL_SIZES) -> None:
    sizes = tuple(int(value) for value in pool_sizes)
    if set(paths.get("paths", {})) != set(PATH_NAMES):
        raise RuntimeError("missing duplicate-growth or specialist-growth pool path")
    core = list(paths["competent_core_model_ids"])
    for name in PATH_NAMES:
        pools = paths["paths"][name]["pools"]
        if set(pools) != {str(value) for value in sizes}:
            raise RuntimeError(f"{name} pool sizes differ from the registry")
        previous: list[str] = []
        for size in sizes:
            record = pools[str(size)]
            ids = list(record["model_ids"])
            if len(ids) != size or len(ids) != len(set(ids)):
                raise RuntimeError(f"{name} size {size} has invalid model identity")
            if ids[: len(previous)] != previous:
                raise RuntimeError(f"{name} pools are not nested at size {size}")
            if record["pool_sha256"] != hash_json(ids):
                raise RuntimeError(f"{name} size {size} pool hash drift")
            if ids[:2] != core:
                raise RuntimeError(f"{name} does not preserve the shared core")
            previous = ids


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
    if not len(array):
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
        "difference_direction": "specialist_growth_minus_duplicate_growth",
    }


def _bootstrap_seed(alpha: float, size: int, method: str) -> int:
    material = f"{alpha:.12g}|{size}|{method}".encode("utf-8")
    return BOOTSTRAP_SEED + int(hashlib.sha256(material).hexdigest()[:8], 16)


def _fit_scores(
    probabilities: np.ndarray,
    labels: np.ndarray,
    design_rows: np.ndarray,
    validation_rows: np.ndarray,
    model_ids: Sequence[str],
    alpha: float,
    *,
    split_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    design = labels[design_rows]
    scores = _score_nonlearned(
        probabilities,
        design,
        model_ids,
        base_seed=BASE_SEED,
        split_seed=split_seed,
        preserve_score_prefixes=True,
    )
    tally = scores["fixed_tally_factorized"]
    from components.scale_selectors import build_order

    order = build_order(
        "empirical_tally", design, model_names=list(model_ids), max_depth=len(model_ids), seed=BASE_SEED
    )
    profile = fit_chain_discount(design, order)
    candidates = []
    for grid_index, minimum in enumerate(CHAIN_GRID):
        raw = chain_discounted_scores(probabilities, profile, minimum_survivor_count=minimum)
        candidates.append({
            "grid_index": grid_index,
            "hyper": {"minimum_survivor_count": minimum},
            "scores": _smooth_fixed(raw, order, base_seed=BASE_SEED, split_seed=split_seed),
        })
    if not np.array_equal(candidates[0]["scores"], tally):
        error = float(np.max(np.abs(candidates[0]["scores"] - tally)))
        raise RuntimeError(f"chain discount endpoint mismatch: {error}")
    chosen, trajectory, fallback = _select_candidate(candidates, labels, validation_rows, alpha)
    scores["fixed_tally_chain_discounted"] = np.asarray(chosen["scores"])
    metadata = {
        "fixed_tally_factorized": {
            "chosen_order": list(order),
            "chosen_order_sha256": hash_json(list(order)),
            "preserve_prefixes": True,
        },
        "fixed_tally_chain_discounted": {
            "chosen_order": list(order),
            "chosen_order_sha256": hash_json(list(order)),
            "preserve_prefixes": True,
            "chosen_hyper": dict(chosen["hyper"]),
            "selection_source": "validation_only",
            "selection_fallback_used": fallback,
            "validation_trajectory": trajectory,
            "endpoint_reference_method": "fixed_tally_factorized",
            "endpoint_score_identity_verified": True,
            "endpoint_score_sha256": hash_json(candidates[0]["scores"].tolist()),
            "chain_discount_profile": chain_discount_diagnostics(
                profile,
                minimum_survivor_count=int(chosen["hyper"]["minimum_survivor_count"]),
            ),
        },
    }
    selected = {name: scores[name] for name in REGISTERED_METHODS}
    if set(selected) != set(REGISTERED_METHODS):
        raise RuntimeError("ISSUE-67 score registry mismatch")
    return selected, metadata


def _event_enrichment(result: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(result)
    vector = output.get("test_miss_by_query")
    calls = output.get("test_calls_by_query")
    output["test_query_count"] = len(output["test_query_ids"])
    output["test_miss_count"] = None if vector is None else int(sum(vector))
    output["test_call_sum"] = None if calls is None else int(sum(calls))
    if vector is not None and output["standard_all_miss_risk"] != float(np.mean(vector)):
        raise RuntimeError("test event vector and risk differ")
    return output


def _evaluate_methods(
    scores: Mapping[str, np.ndarray],
    metadata: Mapping[str, Mapping[str, Any]],
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    alpha: float,
) -> dict[str, dict[str, Any]]:
    methods = {}
    for method in REGISTERED_METHODS:
        result = evaluate_score_family(
            scores[method], labels, split["final_calibration"], split["test"],
            query_ids, model_ids, alpha, method_family=method,
        )
        result.update(metadata.get(method, {}))
        methods[method] = _event_enrichment(result)
    endpoint = evaluate_score_family(
        scores["fixed_tally_factorized"], labels, split["final_calibration"], split["test"],
        query_ids, model_ids, alpha, method_family="fixed_tally_factorized",
    )
    off = evaluate_score_family(
        _score_chain_endpoint(scores, metadata), labels, split["final_calibration"], split["test"],
        query_ids, model_ids, alpha, method_family="fixed_tally_factorized",
    )
    if endpoint != off:
        raise RuntimeError("chain discount calibrated endpoint mismatch")
    methods["fixed_tally_chain_discounted"]["endpoint_result_identity_verified"] = True
    if set(methods) != set(REGISTERED_METHODS):
        raise RuntimeError("missing ISSUE-67 method arm")
    return methods


def _score_chain_endpoint(
    scores: Mapping[str, np.ndarray], metadata: Mapping[str, Mapping[str, Any]]
) -> np.ndarray:
    if not metadata["fixed_tally_chain_discounted"]["endpoint_score_identity_verified"]:
        raise RuntimeError("chain endpoint was not verified")
    return scores["fixed_tally_factorized"]


def _full_pool_record(
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    split_ids: Mapping[str, Sequence[str]],
    model_ids: Sequence[str],
) -> dict[str, Any]:
    cal = np.all(labels[split["final_calibration"]] == 0, axis=1).astype(int)
    test = np.all(labels[split["test"]] == 0, axis=1).astype(int)
    return {
        "model_ids": list(model_ids),
        "model_ids_sha256": hash_json(list(model_ids)),
        "calibration_query_ids": list(split_ids["final_calibration"]),
        "calibration_miss_by_query": cal.tolist(),
        "calibration_miss_count": int(cal.sum()),
        "calibration_query_count": int(len(cal)),
        "calibration_standard_all_miss_floor": float(np.mean(cal)),
        "test_query_ids": list(split_ids["test"]),
        "test_miss_by_query": test.tolist(),
        "test_miss_count": int(test.sum()),
        "test_query_count": int(len(test)),
        "test_standard_all_miss_floor": float(np.mean(test)),
    }


def _difficulty_context(
    probabilities: np.ndarray,
    split: Mapping[str, np.ndarray],
    split_ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    difficulty = 1.0 - np.asarray(probabilities, dtype=float).mean(axis=1)
    thresholds = np.quantile(difficulty[split["design"]], [1.0 / 3.0, 2.0 / 3.0])
    names = np.asarray(["easy", "medium", "hard"], dtype=object)
    strata = names[np.digitize(difficulty, thresholds, right=True)]
    return {
        "score": "one minus mean router probability over declared pool",
        "threshold_fit_roles": ["design"],
        "held_out_threshold_rows_used": False,
        "thresholds": [float(value) for value in thresholds],
        "design_query_ids_sha256": hash_json(list(split_ids["design"])),
        "test_stratum_by_query": [str(strata[int(row)]) for row in split["test"]],
    }


def _attach_difficulty(result: dict[str, Any], context: Mapping[str, Any]) -> None:
    misses = result.get("test_miss_by_query")
    calls = result.get("test_calls_by_query")
    if misses is None or calls is None:
        result["difficulty_strata"] = None
        return
    strata = list(context["test_stratum_by_query"])
    result["difficulty_strata"] = {}
    for name in ("easy", "medium", "hard"):
        rows = [index for index, value in enumerate(strata) if value == name]
        result["difficulty_strata"][name] = {
            "query_count": len(rows),
            "miss_count": int(sum(misses[index] for index in rows)),
            "standard_all_miss_risk": float(np.mean([misses[index] for index in rows])) if rows else None,
            "mean_calls": float(np.mean([calls[index] for index in rows])) if rows else None,
        }


def _subgroup_context(task: Any, split: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Expose archived groups when aligned metadata exists, otherwise explain why."""

    for attribute in ("subgroup_labels", "task_groups", "groups"):
        values = getattr(task, attribute, None)
        if values is None:
            continue
        labels = [str(value) for value in values]
        if len(labels) != len(task.query_ids):
            raise ValueError(f"{attribute} does not align with task query rows")
        test_values = [labels[int(row)] for row in split["test"]]
        return {
            "available": True,
            "source_attribute": attribute,
            "fit_roles": [],
            "test_group_by_query": test_values,
            "test_group_counts": {
                name: test_values.count(name) for name in sorted(set(test_values))
            },
        }
    return {
        "available": False,
        "reason": "archived aligned IFEval task exposes prompts but no task or subgroup labels",
    }


def _unchanged_intervention(
    base: Mapping[str, Any],
    scores: np.ndarray,
    original_labels: np.ndarray,
    changed_labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    original_model_ids: Sequence[str],
    changed_model_ids: Sequence[str],
    *,
    kind: str,
    changed_model_id: str,
) -> dict[str, Any]:
    if kind not in STRESS_KINDS:
        raise ValueError(f"unknown intervention {kind}")
    test_ids = [str(query_ids[int(row)]) for row in split["test"]]
    record = {
        "evaluation_path": "unchanged_deployment_certificate",
        "intervention": kind,
        "changed_model_id": changed_model_id,
        "original_pool_model_ids": list(original_model_ids),
        "changed_pool_model_ids": list(changed_model_ids),
        "original_pool_sha256": hash_json(list(original_model_ids)),
        "changed_pool_sha256": hash_json(list(changed_model_ids)),
        "test_query_ids": test_ids,
        "test_query_ids_sha256": hash_json(test_ids),
        "same_test_rows_as_declared_pool": test_ids == list(base["test_query_ids"]),
        "refit_performed": False,
        "recalibration_performed": False,
        "changed_pool_calibration_labels_recomputed": False,
    }
    if not base["certified"]:
        record.update({
            "original_certificate_available": False,
            "certificate_survives": None,
            "test_calls_by_query": None,
            "test_miss_by_query": None,
            "test_query_count": len(test_ids),
            "test_call_sum": None,
            "test_miss_count": None,
            "mean_calls": None,
            "standard_all_miss_risk": None,
            "call_premium_vs_declared_pool": None,
        })
    else:
        threshold = float(base["threshold"])
        selected_cal = scores[split["final_calibration"]] <= threshold
        selected_test = scores[split["test"]] <= threshold
        if kind == "remove_core_model":
            removed = list(original_model_ids).index(changed_model_id)
            selected_cal = np.delete(selected_cal, removed, axis=1)
            selected_test = np.delete(selected_test, removed, axis=1)
        else:
            selected_cal = np.column_stack([selected_cal, np.zeros(len(selected_cal), dtype=bool)])
            selected_test = np.column_stack([selected_test, np.zeros(len(selected_test), dtype=bool)])
        cal_miss = (~np.any(selected_cal & changed_labels[split["final_calibration"]].astype(bool), axis=1)).astype(int)
        test_miss = (~np.any(selected_test & changed_labels[split["test"]].astype(bool), axis=1)).astype(int)
        calls = selected_test.sum(axis=1).astype(int)
        record.update({
            "original_certificate_available": True,
            "certificate_survives": bool(int(cal_miss.sum()) <= int(base["maximum_allowed_misses"])),
            "unchanged_threshold": threshold,
            "maximum_allowed_misses": int(base["maximum_allowed_misses"]),
            "changed_pool_calibration_miss_by_query": cal_miss.tolist(),
            "changed_pool_calibration_miss_count": int(cal_miss.sum()),
            "test_calls_by_query": calls.tolist(),
            "test_miss_by_query": test_miss.tolist(),
            "test_query_count": int(len(test_miss)),
            "test_call_sum": int(calls.sum()),
            "test_miss_count": int(test_miss.sum()),
            "mean_calls": float(np.mean(calls)),
            "standard_all_miss_risk": float(np.mean(test_miss)),
            "call_premium_vs_declared_pool": float(np.mean(calls) - float(base["test_calls"])),
        })
    changed_pool = _full_pool_record(
        changed_labels, split, _split_ids_from_query_ids(query_ids, split), changed_model_ids
    )
    record["changed_complete_pool"] = changed_pool
    return record


def _split_ids_from_query_ids(
    query_ids: Sequence[str], split: Mapping[str, np.ndarray]
) -> dict[str, list[str]]:
    return {
        name: [str(query_ids[int(row)]) for row in rows]
        for name, rows in split.items()
    }


def _recalibrated_intervention(
    method: str,
    changed_scores: Mapping[str, np.ndarray],
    changed_metadata: Mapping[str, Mapping[str, Any]],
    changed_labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    changed_model_ids: Sequence[str],
    alpha: float,
    *,
    kind: str,
    changed_model_id: str,
) -> dict[str, Any]:
    result = _event_enrichment(evaluate_score_family(
        changed_scores[method], changed_labels, split["final_calibration"], split["test"],
        query_ids, changed_model_ids, alpha, method_family=method,
    ))
    result.update(changed_metadata.get(method, {}))
    result.update({
        "evaluation_path": "refit_and_recalibrated_changed_pool",
        "intervention": kind,
        "changed_model_id": changed_model_id,
        "changed_pool_model_ids": list(changed_model_ids),
        "changed_pool_sha256": hash_json(list(changed_model_ids)),
        "test_query_ids_sha256": hash_json(result["test_query_ids"]),
        "same_test_rows_as_declared_pool": True,
        "refit_performed": True,
        "recalibration_performed": True,
        "changed_pool_calibration_labels_recomputed": True,
        "changed_pool_calibration_query_ids": [str(query_ids[int(row)]) for row in split["final_calibration"]],
    })
    return result


def build_bootstrap_summaries(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for cell in cells:
        alpha = float(cell["alpha"])
        for size in POOL_SIZES:
            for method in REGISTERED_METHODS:
                duplicate = {}
                specialist = {}
                for redraw in cell["redraws"]:
                    for path_record in redraw["paths"]:
                        if path_record["pool_size"] != size:
                            continue
                        target = duplicate if path_record["path"] == "duplicate_growth" else specialist
                        target[int(redraw["redraw_id"])] = path_record["methods"][method]
                ids = sorted(set(duplicate).intersection(specialist))
                ids = [index for index in ids if duplicate[index]["certified"] and specialist[index]["certified"]]
                calls = [specialist[index]["test_calls"] - duplicate[index]["test_calls"] for index in ids]
                risks = [
                    specialist[index]["standard_all_miss_risk"]
                    - duplicate[index]["standard_all_miss_risk"]
                    for index in ids
                ]
                seed = _bootstrap_seed(alpha, size, method)
                summaries.append({
                    "target_alpha": alpha,
                    "pool_size": size,
                    "method": method,
                    "paired_redraw_ids": ids,
                    "calls": deterministic_paired_bootstrap(calls, seed=seed),
                    "standard_all_miss_risk": deterministic_paired_bootstrap(risks, seed=seed + 1),
                })
    return summaries


def assert_complete_output(cells: Sequence[Mapping[str, Any]]) -> None:
    expected = set(REGISTERED_METHODS)
    if len(cells) != len(ALPHA_GRID):
        raise RuntimeError("missing alpha cells")
    pool_hashes: dict[tuple[int, str, int], str] = {}
    for cell in cells:
        for redraw in cell["redraws"]:
            if len(redraw["paths"]) != len(PATH_NAMES) * len(POOL_SIZES):
                raise RuntimeError("missing pool path or pool size")
            for path in redraw["paths"]:
                if set(path["methods"]) != expected:
                    raise RuntimeError("missing method arm")
                if not any(result["certified"] for result in path["methods"].values()):
                    full_pool = path["full_pool"]
                    minimum_bound = (
                        full_pool["calibration_miss_count"] + 1.0
                    ) / (full_pool["calibration_query_count"] + 1.0)
                    if minimum_bound <= float(cell["alpha"]):
                        raise RuntimeError("zero certified methods in a certifiable pool cell")
                key = (int(redraw["redraw_id"]), str(path["path"]), int(path["pool_size"]))
                old = pool_hashes.setdefault(key, str(path["pool"]["pool_sha256"]))
                if old != path["pool"]["pool_sha256"]:
                    raise RuntimeError("pool hash drift across alpha cells")
                stress = path["availability_stress"]
                expected_stress = 0 if path["pool_size"] == 2 else len(STRESS_KINDS)
                if len(stress) != expected_stress:
                    raise RuntimeError("availability interventions were hidden or missing")
                if any(record["intervention"] not in STRESS_KINDS for record in stress):
                    raise RuntimeError("unknown availability intervention")
                if any(set(record["unchanged_deployment"]) != expected for record in stress):
                    raise RuntimeError("missing unchanged-deployment stress arm")
                if any(set(record["refit_and_recalibrated"]) != expected for record in stress):
                    raise RuntimeError("missing recalibrated stress arm")


def run_issue67(
    task: Any,
    *,
    host: str,
    stamp: str,
    redraws: int = REDRAWS,
    redraw_start: int = 0,
    knn_k: int = KNN_K,
    knn_function: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    pool_sizes: Sequence[int] = POOL_SIZES,
    alphas: Sequence[float] = ALPHA_GRID,
) -> dict[str, Any]:
    if task.task != "ifeval":
        raise ValueError("ISSUE-67 is registered for archived RouterEval IFEval")
    if redraws < 1 or redraw_start < 0:
        raise ValueError("redraw count must be positive and redraw start non-negative")
    if tuple(int(value) for value in pool_sizes) != POOL_SIZES:
        raise ValueError(f"ISSUE-67 requires pool sizes {POOL_SIZES}")
    if tuple(float(value) for value in alphas) != ALPHA_GRID:
        raise ValueError(f"ISSUE-67 requires alpha grid {ALPHA_GRID}")
    labels_all = _binary_labels(task.correctness)
    model_ids_all = [str(value) for value in task.model_ids]
    query_ids = [str(value) for value in task.query_ids]
    embeddings = np.asarray(task.embeddings, dtype=float)
    if labels_all.shape[1] < 33:
        raise ValueError("ISSUE-67 requires at least 33 real model columns")
    if len(model_ids_all) != len(set(model_ids_all)):
        raise ValueError("duplicate model IDs in the real RouterEval pool")
    if labels_all.shape[0] != len(query_ids) or embeddings.shape[0] != len(query_ids):
        raise ValueError("task rows, query IDs, and embeddings must align")

    cells = [{"alpha": float(alpha), "redraws": []} for alpha in alphas]
    contexts = []
    knn_impl = knn_function or _knn
    for local_redraw in range(redraws):
        redraw = redraw_start + local_redraw
        split = _target_split(len(query_ids), redraw, BASE_SEED)
        split_ids = _split_query_ids(task, split)
        pool_paths = construct_pool_paths(labels_all[split["design"]], model_ids_all)
        validate_pool_paths(pool_paths)
        context = {
            "redraw_id": redraw,
            "split_seed": BASE_SEED + SPLIT_STEP * redraw,
            "split_ids": split_ids,
            "pool_construction": pool_paths,
        }
        context["context_sha256"] = hash_json(context)
        contexts.append(context)

        path_cache: dict[tuple[str, int], dict[str, Any]] = {}
        specialist_order = pool_paths["held_out_specialist_order"]
        for path_name in PATH_NAMES:
            for size in POOL_SIZES:
                pool = pool_paths["paths"][path_name]["pools"][str(size)]
                columns = np.asarray(pool["column_indices"], dtype=int)
                labels = labels_all[:, columns]
                ids = [model_ids_all[index] for index in columns]
                probabilities, _ = knn_impl(
                    embeddings[split["router"]], labels[split["router"]], embeddings, knn_k
                )
                probabilities = np.asarray(probabilities, dtype=float)
                if probabilities.shape != labels.shape:
                    raise ValueError("KNN probabilities do not match a declared pool")
                path_cache[(path_name, size)] = {
                    "pool": pool,
                    "columns": columns,
                    "labels": labels,
                    "ids": ids,
                    "probabilities": probabilities,
                    "full_pool": _full_pool_record(labels, split, split_ids, ids),
                    "difficulty": _difficulty_context(probabilities, split, split_ids),
                }

        for cell in cells:
            alpha = float(cell["alpha"])
            path_records = []
            for path_name in PATH_NAMES:
                for size in POOL_SIZES:
                    cached = path_cache[(path_name, size)]
                    scores, metadata = _fit_scores(
                        cached["probabilities"], cached["labels"], split["design"],
                        split["validation"], cached["ids"], alpha,
                        split_seed=context["split_seed"],
                    )
                    methods = _evaluate_methods(
                        scores, metadata, cached["labels"], split, query_ids,
                        cached["ids"], alpha,
                    )
                    for result in methods.values():
                        _attach_difficulty(result, cached["difficulty"])

                    stress_records = []
                    if size >= 4:
                        remove_id = pool_paths["competent_core_model_ids"][0]
                        original_columns = list(map(int, cached["columns"]))
                        remove_position = cached["ids"].index(remove_id)
                        remove_columns = original_columns[:remove_position] + original_columns[remove_position + 1 :]
                        add_candidates = [int(index) for index in specialist_order if int(index) not in original_columns]
                        if not add_candidates:
                            raise RuntimeError("no held-out specialist remains for add intervention")
                        add_column = add_candidates[0]
                        interventions = (
                            ("remove_core_model", remove_id, remove_columns),
                            ("add_held_out_specialist", model_ids_all[add_column], original_columns + [add_column]),
                        )
                        for kind, changed_id, changed_columns_list in interventions:
                            changed_columns = np.asarray(changed_columns_list, dtype=int)
                            changed_ids = [model_ids_all[index] for index in changed_columns]
                            changed_labels = labels_all[:, changed_columns]
                            changed_probabilities, _ = knn_impl(
                                embeddings[split["router"]], changed_labels[split["router"]],
                                embeddings, knn_k,
                            )
                            changed_scores, changed_metadata = _fit_scores(
                                np.asarray(changed_probabilities, dtype=float), changed_labels,
                                split["design"], split["validation"], changed_ids, alpha,
                                split_seed=context["split_seed"],
                            )
                            unchanged = {}
                            recalibrated = {}
                            for method in REGISTERED_METHODS:
                                unchanged[method] = _unchanged_intervention(
                                    methods[method], scores[method], cached["labels"], changed_labels,
                                    split, query_ids, cached["ids"], changed_ids,
                                    kind=kind, changed_model_id=changed_id,
                                )
                                recalibrated[method] = _recalibrated_intervention(
                                    method, changed_scores, changed_metadata, changed_labels, split,
                                    query_ids, changed_ids, alpha,
                                    kind=kind, changed_model_id=changed_id,
                                )
                                if methods[method]["certified"] and recalibrated[method]["certified"]:
                                    recalibrated[method]["call_premium_vs_declared_pool"] = float(
                                        recalibrated[method]["test_calls"] - methods[method]["test_calls"]
                                    )
                                else:
                                    recalibrated[method]["call_premium_vs_declared_pool"] = None
                            changed_full_pool = _full_pool_record(
                                changed_labels, split, split_ids, changed_ids
                            )
                            stress_records.append({
                                "intervention": kind,
                                "changed_model_id": changed_id,
                                "original_pool_sha256": cached["pool"]["pool_sha256"],
                                "changed_pool_sha256": hash_json(changed_ids),
                                "changed_complete_pool": changed_full_pool,
                                "same_test_query_rows": True,
                                "test_query_ids_sha256": hash_json(split_ids["test"]),
                                "unchanged_deployment": unchanged,
                                "refit_and_recalibrated": recalibrated,
                            })
                    path_records.append({
                        "path": path_name,
                        "pool_size": size,
                        "pool": cached["pool"],
                        "full_pool": cached["full_pool"],
                        "difficulty_definition": cached["difficulty"],
                        "task_subgroup_shift": _subgroup_context(task, split),
                        "methods": methods,
                        "availability_stress": stress_records,
                    })
            cell["redraws"].append({
                "redraw_id": redraw,
                "split_seed": context["split_seed"],
                "split_ids": split_ids,
                "test_query_ids_sha256": hash_json(split_ids["test"]),
                "pool_context_sha256": context["context_sha256"],
                "paths": path_records,
            })

    assert_complete_output(cells)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": 67,
        "design": "pool_composition_and_deployment_shift",
        "protocol": "routereval",
        "benchmark": "ifeval",
        "host": str(host),
        "stamp": str(stamp),
        "redraw_count": redraws,
        "redraw_start": redraw_start,
        "redraw_end": redraw_start + redraws,
        "knn_k": knn_k,
        "target_grid": list(ALPHA_GRID),
        "pool_sizes": list(POOL_SIZES),
        "path_identities": list(PATH_NAMES),
        "method_identities": list(REGISTERED_METHODS),
        "availability_interventions": list(STRESS_KINDS),
        "preserve_prefixes": True,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "query_ids_sha256": hash_json(query_ids),
        "candidate_model_ids_sha256": hash_json(model_ids_all),
        "source_hashes": dict(task.source_hashes),
        "feature_audit": dict(task.feature_audit),
        "pool_contexts": contexts,
        "cells": cells,
    }
    payload["paired_bootstrap_summaries"] = build_bootstrap_summaries(cells)
    return payload


def _safe_identity(value: str, name: str) -> str:
    text = str(value)
    if not text or re.fullmatch(r"[A-Za-z0-9_.-]+", text) is None:
        raise ValueError(f"{name} must use only letters, digits, dot, underscore, or hyphen")
    return text


def publish_output(out: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(out)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output collision {temporary}")
    temporary.mkdir()
    try:
        identity = f"issue67_{payload['host']}_{payload['stamp']}"
        result = temporary / f"{identity}_results.json"
        atomic_write_json(result, payload)
        atomic_write_json(temporary / f"{identity}_manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "issue": 67,
            "host": payload["host"],
            "stamp": payload["stamp"],
            "benchmark": payload["benchmark"],
            "results_file": result.name,
            "results_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        })
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
    parser.add_argument("--leaderboard", required=True, type=Path)
    parser.add_argument("--router-dir", required=True, type=Path)
    parser.add_argument("--benchmark", default="ifeval")
    parser.add_argument("--host", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--redraws", type=int, default=REDRAWS)
    parser.add_argument("--redraw-start", type=int, default=0)
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
    if args.benchmark != "ifeval":
        raise SystemExit("ISSUE-67 is registered only for archived RouterEval IFEval")
    task = _load_task(
        "routereval", args.benchmark, leaderboard=args.leaderboard, router_dir=args.router_dir
    )
    try:
        payload = run_issue67(
            task, host=host, stamp=stamp, redraws=args.redraws,
            redraw_start=args.redraw_start,
        )
        published = publish_output(args.out, payload)
    except (FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"status": "complete", "output": str(published)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
