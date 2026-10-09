"""Run ISSUE-70 heterogeneous public-price cost sensitivity on RouterBench.

The registered run uses four approved RouterBench tasks, the complete eleven
model pool, twenty paired redraws, and the full Stage 1 alpha grid. Costs used
for fitting are per-model means computed only from design and router rows.
Validation, final calibration, and test costs never enter a fitted order. Test
cost is the sum of the archived per-response public-price values for the exact
models selected on each test query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from components.chain_discount import (  # noqa: E402
    chain_discount_diagnostics,
    chain_discounted_scores,
    fit_chain_discount,
)
from components.data_protocol import load_routerbench  # noqa: E402
from components.racer_protocol import (  # noqa: E402
    select_standard_crc_threshold,
    smooth_nonconformity_scores,
)
from components.racer_wrapper import fixed_order_factorized_scores  # noqa: E402
from components.routerbench_protocol import (  # noqa: E402
    ROUTERBENCH_MODEL_COLUMNS,
    _ensure_pandas_pickle_compat,
)
from components.scale_selectors import build_order  # noqa: E402
from scripts.run_issue78_o2_full_matrix import _fallback_standard  # noqa: E402
from scripts.run_learned_baselines import _routerbench_costs  # noqa: E402
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

SCHEMA_VERSION = "issue70_cost_sensitivity_v1"
APPROVED_TASKS = ("mmlu", "hellaswag", "winogrande", "arc-challenge")
POOL_SIZE = 11
REDRAWS = 20
KNN_K = 40
CHAIN_GRID = (0, 5, 20, 50)
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260920

COUNT_D1 = "fixed_tally_factorized"
COUNT_D2 = "fixed_tally_chain_discounted"
COST_D1 = "cost_weighted_tally_factorized"
COST_D2 = "cost_weighted_tally_chain_discounted"
REGISTERED_METHODS = (
    COUNT_D1,
    COUNT_D2,
    COST_D1,
    COST_D2,
    "racer_g",
    "racer_p",
    "exact_independence",
    "fixed_design_marginal_factorized",
)
COMPARISON_PAIRS = (
    (COST_D1, COUNT_D1),
    (COST_D2, COUNT_D2),
    (COUNT_D1, "racer_g"),
    (COUNT_D1, "racer_p"),
    (COUNT_D2, "racer_g"),
    (COUNT_D2, "racer_p"),
    (COST_D1, "racer_g"),
    (COST_D1, "racer_p"),
    (COST_D2, "racer_g"),
    (COST_D2, "racer_p"),
)
FIT_COST_ROLES = ("design", "router")
HELD_OUT_COST_ROLES = ("final_calibration", "validation", "test")


def _validate_cost_matrix(costs: np.ndarray, *, n_models: int | None = None) -> np.ndarray:
    values = np.asarray(costs, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("cost matrix must be a non-empty two-dimensional array")
    if n_models is not None and values.shape[1] != n_models:
        raise ValueError("cost matrix model width differs from the task")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("cost matrix must contain finite non-negative values")
    return values


def _validated_split_rows(
    split: Mapping[str, np.ndarray],
    *,
    n_rows: int,
) -> dict[str, np.ndarray]:
    required = FIT_COST_ROLES + HELD_OUT_COST_ROLES
    if tuple(split) != required:
        raise ValueError(f"split roles must be exactly {required}")
    result: dict[str, np.ndarray] = {}
    seen: set[int] = set()
    for role in required:
        rows = np.asarray(split[role], dtype=int)
        if rows.ndim != 1 or np.any((rows < 0) | (rows >= n_rows)):
            raise ValueError(f"invalid rows for split role {role}")
        if len(rows) != len(np.unique(rows)):
            raise ValueError(f"duplicate rows inside split role {role}")
        overlap = seen.intersection(int(value) for value in rows)
        if overlap:
            raise ValueError(f"split roles overlap at rows {sorted(overlap)}")
        seen.update(int(value) for value in rows)
        result[role] = rows
    if len(seen) != n_rows:
        raise ValueError("split roles do not account for every task row")
    return result


def freeze_cost_vector(
    cost_matrix: np.ndarray,
    split: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Freeze one public-price vector without held-out row access."""

    costs = _validate_cost_matrix(cost_matrix, n_models=len(model_ids))
    if costs.shape[0] != len(query_ids):
        raise ValueError("cost matrix row count differs from query IDs")
    rows = _validated_split_rows(split, n_rows=len(query_ids))
    fit_rows = np.concatenate([rows[role] for role in FIT_COST_ROLES])
    vector = costs[fit_rows].mean(axis=0)
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise RuntimeError("frozen public-price vector is invalid")
    fit_ids = [str(query_ids[int(row)]) for row in fit_rows]
    held_out_ids = {
        role: [str(query_ids[int(row)]) for row in rows[role]]
        for role in HELD_OUT_COST_ROLES
    }
    provenance = {
        "source": "RouterBench archived public-price total_cost",
        "source_fields": [str(model) + "|total_cost" for model in model_ids],
        "aggregation": "arithmetic mean by model over design and router rows",
        "fit_roles": list(FIT_COST_ROLES),
        "fit_row_count": int(len(fit_rows)),
        "fit_query_ids": fit_ids,
        "fit_query_ids_sha256": hash_json(fit_ids),
        "held_out_roles": list(HELD_OUT_COST_ROLES),
        "held_out_query_ids_sha256": {
            role: hash_json(ids) for role, ids in held_out_ids.items()
        },
        "held_out_rows_used": False,
        "frozen_before_validation_calibration_and_test": True,
        "model_ids": [str(value) for value in model_ids],
        "mean_cost_by_model": [float(value) for value in vector],
        "cost_vector_sha256": hash_json([float(value) for value in vector]),
    }
    fit_set = set(fit_ids)
    for role, ids in held_out_ids.items():
        if fit_set.intersection(ids):
            raise RuntimeError(f"cost fit leaked rows from {role}")
    return vector, provenance


def cost_weighted_tally_order(
    design_labels: np.ndarray,
    frozen_costs: Sequence[float],
    *,
    max_depth: int | None = None,
) -> tuple[int, ...]:
    """Greedily maximize newly covered design failures per frozen mean cost.

    Ties use overall design accuracy and then the lower model index. When every
    cost is equal this is byte-for-byte the existing empirical tally order.
    """

    labels = np.asarray(design_labels)
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("design labels must be a non-empty two-dimensional array")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("design labels must be binary")
    costs = np.asarray(frozen_costs, dtype=float)
    if costs.shape != (labels.shape[1],):
        raise ValueError("frozen costs must have one value per model")
    if not np.all(np.isfinite(costs)) or np.any(costs < 0.0):
        raise ValueError("frozen costs must be finite and non-negative")
    depth = labels.shape[1] if max_depth is None else min(int(max_depth), labels.shape[1])
    if depth < 1:
        raise ValueError("max_depth must be positive")

    alive = np.ones(labels.shape[0], dtype=bool)
    remaining = np.ones(labels.shape[1], dtype=bool)
    accuracy = labels.mean(axis=0)
    order: list[int] = []
    for _ in range(depth):
        if not np.any(alive):
            break
        benefit = labels[alive].sum(axis=0).astype(float)
        ratio = np.full(labels.shape[1], -np.inf, dtype=float)
        positive_cost = costs > 0.0
        ratio[remaining & positive_cost] = (
            benefit[remaining & positive_cost] / costs[remaining & positive_cost]
        )
        free = remaining & ~positive_cost
        ratio[free & (benefit > 0.0)] = np.inf
        ratio[free & (benefit == 0.0)] = 0.0
        best_ratio = np.max(ratio)
        tied = np.flatnonzero(remaining & (ratio == best_ratio))
        best_accuracy = np.max(accuracy[tied])
        choice = int(tied[accuracy[tied] == best_accuracy][0])
        order.append(choice)
        remaining[choice] = False
        alive &= labels[:, choice] == 0
    if len(order) < depth:
        residual = np.flatnonzero(remaining)
        residual = residual[np.lexsort((residual, -accuracy[residual]))]
        order.extend(int(value) for value in residual[: depth - len(order)])
    return tuple(order)


def _smooth_fixed(
    scores: np.ndarray,
    order: Sequence[int],
    *,
    base_seed: int,
    split_seed: int,
) -> np.ndarray:
    order_array = np.asarray(order, dtype=int)
    ordered = np.asarray(scores, dtype=float)[:, order_array]
    smoothed = smooth_nonconformity_scores(
        ordered,
        seed=base_seed,
        split_seed=split_seed,
        preserve_prefixes=True,
    )
    result = np.empty_like(smoothed)
    result[:, order_array] = smoothed
    if np.any(np.diff(result[:, order_array], axis=1) < 0.0):
        raise RuntimeError("prefix-preserving smoothing inverted a fixed order")
    return result


def _fixed_order_scores(
    probabilities: np.ndarray,
    order: Sequence[int],
    *,
    base_seed: int,
    split_seed: int,
) -> np.ndarray:
    raw = fixed_order_factorized_scores(
        probabilities, order, max_depth=probabilities.shape[1]
    )
    return _smooth_fixed(raw, order, base_seed=base_seed, split_seed=split_seed)


def _chain_candidates(
    probabilities: np.ndarray,
    design_labels: np.ndarray,
    order: Sequence[int],
    d1_scores: np.ndarray,
    *,
    base_seed: int,
    split_seed: int,
) -> tuple[list[dict[str, Any]], Any]:
    profile = fit_chain_discount(design_labels, order)
    candidates = []
    for grid_index, minimum in enumerate(CHAIN_GRID):
        raw = chain_discounted_scores(
            probabilities,
            profile,
            minimum_survivor_count=minimum,
        )
        candidates.append({
            "grid_index": grid_index,
            "hyper": {"minimum_survivor_count": minimum},
            "scores": _smooth_fixed(
                raw, order, base_seed=base_seed, split_seed=split_seed
            ),
        })
    if not np.array_equal(candidates[0]["scores"], d1_scores):
        error = float(np.max(np.abs(candidates[0]["scores"] - d1_scores)))
        raise RuntimeError(f"D2 discount-off endpoint differs from D1 by {error}")
    return candidates, profile


def _fallback_with_cost(
    scores: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    frozen_costs: np.ndarray,
) -> tuple[float, float, float, float]:
    """Scan score events once and minimize risk, frozen cost, calls."""

    values = np.asarray(scores[rows], dtype=float)
    outcomes = np.asarray(labels[rows], dtype=bool)
    n_queries, n_models = values.shape
    flat_scores = values.ravel()
    flat_correct = outcomes.ravel()
    sequence = np.argsort(flat_scores, kind="stable")
    first = float(flat_scores[sequence[0]])
    best = (1.0, 0.0, 0.0, float(np.nextafter(first, -np.inf)))
    successful = np.zeros(n_queries, dtype=bool)
    calls_total = 0
    cost_total = 0.0
    start = 0
    while start < len(sequence):
        threshold = float(flat_scores[sequence[start]])
        end = start + 1
        while end < len(sequence) and flat_scores[sequence[end]] == threshold:
            end += 1
        selected = sequence[start:end]
        calls_total += end - start
        cost_total += float(np.sum(frozen_costs[selected % n_models]))
        correct = selected[flat_correct[selected]]
        if len(correct):
            successful[np.unique(correct // n_models)] = True
        best = min(
            best,
            (
                float(np.mean(~successful)),
                cost_total / n_queries,
                calls_total / n_queries,
                threshold,
            ),
        )
        start = end
    return best


def select_chain_candidate(
    candidates: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    validation_rows: np.ndarray,
    alpha: float,
    frozen_costs: np.ndarray,
    *,
    objective: str,
) -> tuple[Mapping[str, Any], list[dict[str, Any]], bool]:
    """Tune one D2 grid on validation by calls or frozen expected cost."""

    if objective not in {"calls", "cost"}:
        raise ValueError("objective must be calls or cost")
    trajectory = []
    for candidate in candidates:
        scores = np.asarray(candidate["scores"], dtype=float)
        calibration = select_standard_crc_threshold(
            scores[validation_rows], labels[validation_rows], alpha
        )
        if calibration.feasible:
            threshold = float(calibration.threshold)
            selected = scores[validation_rows] <= threshold
            risk = float(
                np.mean(
                    ~np.any(
                        selected & labels[validation_rows].astype(bool), axis=1
                    )
                )
            )
            calls = float(np.mean(selected.sum(axis=1)))
            cost = float(np.mean(selected @ frozen_costs))
        else:
            if objective == "cost":
                risk, cost, calls, threshold = _fallback_with_cost(
                    scores, labels, validation_rows, frozen_costs
                )
            else:
                risk, calls, threshold = _fallback_standard(
                    scores, labels, validation_rows
                )
                selected = scores[validation_rows] <= threshold
                cost = float(np.mean(selected @ frozen_costs))
        trajectory.append({
            "grid_index": int(candidate["grid_index"]),
            "hyper": dict(candidate["hyper"]),
            "validation_certified": bool(calibration.feasible),
            "validation_threshold": float(threshold),
            "validation_calls": calls,
            "validation_frozen_mean_cost": cost,
            "validation_standard_all_miss_risk": risk,
            "validation_non_certification_reason": calibration.non_certification_reason,
            "validation_crc_bound": float(calibration.crc_bound),
        })
    certified = [row for row in trajectory if row["validation_certified"]]
    pool = certified if certified else trajectory
    if objective == "cost":
        key = lambda row: (
            row["validation_frozen_mean_cost"],
            row["validation_standard_all_miss_risk"],
            row["validation_calls"],
            row["grid_index"],
        ) if certified else (
            row["validation_standard_all_miss_risk"],
            row["validation_frozen_mean_cost"],
            row["validation_calls"],
            row["grid_index"],
        )
    else:
        key = lambda row: (
            row["validation_calls"],
            row["validation_standard_all_miss_risk"],
            row["grid_index"],
        ) if certified else (
            row["validation_standard_all_miss_risk"],
            row["validation_calls"],
            row["grid_index"],
        )
    chosen_row = min(pool, key=key)
    chosen = candidates[int(chosen_row["grid_index"])]
    return chosen, trajectory, not bool(certified)


def attach_archived_test_costs(
    result: Mapping[str, Any],
    cost_matrix: np.ndarray,
    test_rows: np.ndarray,
    model_ids: Sequence[str],
) -> dict[str, Any]:
    """Attach exact archived test cost arithmetic to one calibrated result."""

    output = dict(result)
    if not output["certified"]:
        output.update({
            "test_query_count": int(len(test_rows)),
            "test_call_sum": None,
            "test_miss_count": None,
            "test_total_cost_by_query": None,
            "test_total_cost_sum": None,
            "test_mean_total_cost": None,
        })
        return output
    costs = _validate_cost_matrix(cost_matrix, n_models=len(model_ids))
    index = {str(model): position for position, model in enumerate(model_ids)}
    selected = output["selected_model_ids_by_query"]
    calls = output["test_calls_by_query"]
    if len(selected) != len(test_rows) or len(calls) != len(test_rows):
        raise RuntimeError("test selection vectors do not align with test rows")
    totals: list[float] = []
    for offset, (row, selected_ids) in enumerate(zip(test_rows, selected)):
        try:
            selected_indices = [index[str(model)] for model in selected_ids]
        except KeyError as error:
            raise RuntimeError(f"unknown selected model ID {error.args[0]}") from error
        if len(selected_indices) != int(calls[offset]):
            raise RuntimeError("selected model IDs and test call count differ")
        total = float(costs[int(row), selected_indices].sum()) if selected_indices else 0.0
        totals.append(total)
    output.update({
        "test_query_count": int(len(test_rows)),
        "test_call_sum": int(sum(int(value) for value in calls)),
        "test_miss_count": int(sum(int(value) for value in output["test_miss_by_query"])),
        "test_total_cost_by_query": totals,
        "test_total_cost_sum": float(sum(totals)),
        "test_mean_total_cost": float(np.mean(totals)),
    })
    if output["test_miss_count"] != int(
        round(output["standard_all_miss_risk"] * output["test_query_count"])
    ):
        raise RuntimeError("standard all-miss event count failed its arithmetic check")
    if not np.isclose(output["test_total_cost_sum"], sum(totals), rtol=0.0, atol=1e-15):
        raise RuntimeError("archived test cost sum failed its arithmetic check")
    return output


def assert_arm_presence(cells: Sequence[Mapping[str, Any]]) -> None:
    expected = set(REGISTERED_METHODS)
    if not cells:
        raise RuntimeError("ISSUE-70 emitted no alpha cells")
    for cell in cells:
        if not cell.get("redraws"):
            raise RuntimeError(f"ISSUE-70 emitted no redraws for alpha {cell.get('alpha')}")
        for redraw in cell["redraws"]:
            present = set(redraw.get("methods", {}))
            if present != expected:
                raise RuntimeError(
                    "ISSUE-70 arm presence mismatch at "
                    f"alpha={cell.get('alpha')}, redraw={redraw.get('redraw_id')}: "
                    f"missing={sorted(expected - present)}, "
                    f"unexpected={sorted(present - expected)}"
                )


def deterministic_paired_bootstrap(
    values: Sequence[float],
    *,
    seed: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any] | None:
    """Return a deterministic paired-redraw mean difference summary."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("paired bootstrap values must be a finite vector")
    if isinstance(repetitions, bool) or int(repetitions) < 2:
        raise ValueError("bootstrap repetitions must be at least two")
    if len(array) == 0:
        return None
    generator = np.random.default_rng(int(seed))
    samples = array[
        generator.integers(0, len(array), size=(int(repetitions), len(array)))
    ].mean(axis=1)
    return {
        "paired_redraw_count": int(len(array)),
        "mean_difference": float(np.mean(array)),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "difference_direction": "candidate_minus_reference",
    }


def _comparison_seed(benchmark: str, alpha: float, candidate: str, reference: str) -> int:
    material = f"{benchmark}|{alpha:.12g}|{candidate}|{reference}".encode("utf-8")
    return BOOTSTRAP_SEED + int(hashlib.sha256(material).hexdigest()[:8], 16)


def build_bootstrap_summaries(
    benchmark: str,
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        alpha = float(cell["alpha"])
        by_redraw = {int(row["redraw_id"]): row for row in cell["redraws"]}
        for candidate, reference in COMPARISON_PAIRS:
            call_diffs = []
            cost_diffs = []
            risk_diffs = []
            paired_ids = []
            for redraw_id in sorted(by_redraw):
                methods = by_redraw[redraw_id]["methods"]
                left = methods[candidate]
                right = methods[reference]
                if not left["certified"] or not right["certified"]:
                    continue
                call_diffs.append(float(left["test_calls"] - right["test_calls"]))
                cost_diffs.append(
                    float(left["test_mean_total_cost"] - right["test_mean_total_cost"])
                )
                risk_diffs.append(
                    float(
                        left["standard_all_miss_risk"]
                        - right["standard_all_miss_risk"]
                    )
                )
                paired_ids.append(redraw_id)
            seed = _comparison_seed(benchmark, alpha, candidate, reference)
            rows.append({
                "benchmark": benchmark,
                "target_alpha": alpha,
                "candidate": candidate,
                "reference": reference,
                "jointly_certified_redraw_ids": paired_ids,
                "jointly_certified_redraw_count": len(paired_ids),
                "calls": deterministic_paired_bootstrap(call_diffs, seed=seed),
                "actual_archived_total_cost": deterministic_paired_bootstrap(
                    cost_diffs, seed=seed + 1
                ),
                "standard_all_miss_risk": deterministic_paired_bootstrap(
                    risk_diffs, seed=seed + 2
                ),
            })
    return rows


def recompute_aggregates(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for cell in cells:
        methods: dict[str, Any] = {}
        for method in REGISTERED_METHODS:
            records = [redraw["methods"][method] for redraw in cell["redraws"]]
            certified = [record for record in records if record["certified"]]
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
            costs = [
                float(value)
                for record in certified
                for value in record["test_total_cost_by_query"]
            ]
            methods[method] = {
                "redraw_count": len(records),
                "certified_redraw_count": len(certified),
                "certification_rate": len(certified) / len(records),
                "test_query_count": len(calls),
                "test_call_sum": int(sum(calls)),
                "mean_calls": float(np.mean(calls)) if calls else None,
                "test_miss_count": int(sum(misses)),
                "standard_all_miss_risk": float(np.mean(misses)) if misses else None,
                "actual_archived_total_cost_sum": float(sum(costs)) if costs else None,
                "actual_archived_mean_cost": float(np.mean(costs)) if costs else None,
            }
        statuses = [
            redraw["structural_feasibility"]["test_floor_above_target"]
            for redraw in cell["redraws"]
        ]
        if all(statuses):
            status = "all_redraws_structurally_infeasible"
        elif any(statuses):
            status = "some_redraws_structurally_infeasible"
        else:
            status = "no_redraw_structurally_infeasible"
        output[str(cell["alpha"])] = {
            "methods": methods,
            "structural_feasibility": {
                "status": status,
                "redraw_count": len(statuses),
                "test_floor_above_target_redraw_count": int(sum(statuses)),
            },
        }
    return output


def _evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    alpha: float,
    method: str,
    cost_matrix: np.ndarray,
) -> dict[str, Any]:
    base = evaluate_score_family(
        scores,
        labels,
        split["final_calibration"],
        split["test"],
        query_ids,
        model_ids,
        alpha,
        method_family=method,
    )
    return attach_archived_test_costs(base, cost_matrix, split["test"], model_ids)


def run_issue70_task(
    task: Any,
    cost_matrix: np.ndarray,
    *,
    host_id: str,
    run_stamp: str,
    redraws: int = REDRAWS,
    redraw_start: int = 0,
    alphas: Sequence[float] = ALPHA_GRID,
    knn_k: int = KNN_K,
    base_seed: int = BASE_SEED,
    knn_function: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Run one task with shared splits and all eight registered arms."""

    if redraws < 1:
        raise ValueError("redraw count must be positive")
    if redraw_start < 0:
        raise ValueError("redraw start must be non-negative")
    if tuple(float(value) for value in alphas) != ALPHA_GRID:
        raise ValueError(f"ISSUE-70 requires the full alpha grid {ALPHA_GRID}")
    labels = np.asarray(task.correctness, dtype=np.int8)
    if labels.ndim != 2 or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("task correctness must be a binary matrix")
    embeddings = np.asarray(task.embeddings, dtype=float)
    model_ids = [str(value) for value in task.model_ids]
    query_ids = [str(value) for value in task.query_ids]
    costs = _validate_cost_matrix(cost_matrix, n_models=len(model_ids))
    if costs.shape[0] != len(query_ids) or labels.shape != costs.shape:
        raise ValueError("task labels, query IDs, and archived costs must align")
    if embeddings.shape[0] != len(query_ids):
        raise ValueError("task embeddings and query IDs must align")

    cells = [{"alpha": float(alpha), "redraws": []} for alpha in alphas]
    redraw_contexts = []
    knn_impl = knn_function or _knn
    for local_redraw in range(redraws):
        redraw = redraw_start + local_redraw
        split = _target_split(len(query_ids), redraw, base_seed)
        split_ids = _split_query_ids(task, split)
        probabilities, _ = knn_impl(
            embeddings[split["router"]],
            labels[split["router"]],
            embeddings,
            knn_k,
        )
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != labels.shape:
            raise ValueError("KNN probability shape differs from task labels")
        split_seed = base_seed + SPLIT_STEP * redraw
        frozen_costs, cost_provenance = freeze_cost_vector(
            costs, split, query_ids, model_ids
        )
        count_order = build_order(
            "empirical_tally",
            labels[split["design"]],
            model_names=model_ids,
            max_depth=len(model_ids),
            seed=base_seed,
        )
        cost_order = cost_weighted_tally_order(
            labels[split["design"]], frozen_costs, max_depth=len(model_ids)
        )
        score_families = _score_nonlearned(
            probabilities,
            labels[split["design"]],
            model_ids,
            base_seed=base_seed,
            split_seed=split_seed,
            preserve_score_prefixes=True,
        )
        count_d1_scores = score_families[COUNT_D1]
        expected_count_scores = _fixed_order_scores(
            probabilities,
            count_order,
            base_seed=base_seed,
            split_seed=split_seed,
        )
        if not np.array_equal(count_d1_scores, expected_count_scores):
            raise RuntimeError("registered count tally scores changed unexpectedly")
        cost_d1_scores = _fixed_order_scores(
            probabilities,
            cost_order,
            base_seed=base_seed,
            split_seed=split_seed,
        )
        count_chain, count_profile = _chain_candidates(
            probabilities,
            labels[split["design"]],
            count_order,
            count_d1_scores,
            base_seed=base_seed,
            split_seed=split_seed,
        )
        cost_chain, cost_profile = _chain_candidates(
            probabilities,
            labels[split["design"]],
            cost_order,
            cost_d1_scores,
            base_seed=base_seed,
            split_seed=split_seed,
        )

        cost_context = {
            "redraw_id": redraw,
            "split_seed": split_seed,
            "split_ids": split_ids,
            "cost_vector": cost_provenance,
            "count_tally_order": list(count_order),
            "cost_weighted_tally_order": list(cost_order),
        }
        cost_context["context_sha256"] = hash_json(cost_context)
        redraw_contexts.append(cost_context)

        calibration_rows = split["final_calibration"]
        test_rows = split["test"]
        calibration_floor = np.all(labels[calibration_rows] == 0, axis=1).astype(int)
        test_floor = np.all(labels[test_rows] == 0, axis=1).astype(int)
        full_pool = {
            "model_ids": model_ids,
            "calibration_query_ids": split_ids["final_calibration"],
            "calibration_miss_by_query": calibration_floor.tolist(),
            "calibration_miss_count": int(calibration_floor.sum()),
            "calibration_query_count": int(len(calibration_floor)),
            "calibration_standard_all_miss_floor": float(np.mean(calibration_floor)),
            "test_query_ids": split_ids["test"],
            "test_miss_by_query": test_floor.tolist(),
            "test_miss_count": int(test_floor.sum()),
            "test_query_count": int(len(test_floor)),
            "test_standard_all_miss_floor": float(np.mean(test_floor)),
            "test_total_cost_sum": float(costs[test_rows].sum()),
            "test_mean_total_cost": float(costs[test_rows].sum(axis=1).mean()),
        }

        for cell in cells:
            alpha = float(cell["alpha"])
            methods: dict[str, dict[str, Any]] = {}
            for method in (
                "racer_g",
                "racer_p",
                "exact_independence",
                "fixed_design_marginal_factorized",
            ):
                methods[method] = _evaluate(
                    score_families[method],
                    labels,
                    split,
                    query_ids,
                    model_ids,
                    alpha,
                    method,
                    costs,
                )

            count_d1_base = _evaluate(
                count_d1_scores,
                labels,
                split,
                query_ids,
                model_ids,
                alpha,
                COUNT_D1,
                costs,
            )
            count_d1_base.update({
                "ordering_objective": "newly covered design failures",
                "chosen_order": list(count_order),
                "chosen_order_sha256": hash_json(list(count_order)),
                "stopping_family": "D1_independence_product",
                "preserve_prefixes": True,
            })
            methods[COUNT_D1] = count_d1_base

            cost_d1_base = _evaluate(
                cost_d1_scores,
                labels,
                split,
                query_ids,
                model_ids,
                alpha,
                COST_D1,
                costs,
            )
            cost_d1_base.update({
                "ordering_objective": "newly covered design failures per frozen mean cost",
                "chosen_order": list(cost_order),
                "chosen_order_sha256": hash_json(list(cost_order)),
                "stopping_family": "D1_independence_product",
                "preserve_prefixes": True,
                "cost_vector_sha256": cost_provenance["cost_vector_sha256"],
            })
            methods[COST_D1] = cost_d1_base

            for d1_method, d2_method, d1_scores, candidates, profile, order, objective in (
                (
                    COUNT_D1,
                    COUNT_D2,
                    count_d1_scores,
                    count_chain,
                    count_profile,
                    count_order,
                    "calls",
                ),
                (
                    COST_D1,
                    COST_D2,
                    cost_d1_scores,
                    cost_chain,
                    cost_profile,
                    cost_order,
                    "cost",
                ),
            ):
                endpoint = _evaluate(
                    candidates[0]["scores"],
                    labels,
                    split,
                    query_ids,
                    model_ids,
                    alpha,
                    d1_method,
                    costs,
                )
                d1_comparable = {
                    key: value
                    for key, value in methods[d1_method].items()
                    if key not in {
                        "ordering_objective",
                        "chosen_order",
                        "chosen_order_sha256",
                        "stopping_family",
                        "preserve_prefixes",
                        "cost_vector_sha256",
                    }
                }
                if endpoint != d1_comparable:
                    raise RuntimeError(f"{d2_method} calibrated off endpoint differs from {d1_method}")
                chosen, trajectory, fallback = select_chain_candidate(
                    candidates,
                    labels,
                    split["validation"],
                    alpha,
                    frozen_costs,
                    objective=objective,
                )
                result = _evaluate(
                    chosen["scores"],
                    labels,
                    split,
                    query_ids,
                    model_ids,
                    alpha,
                    d2_method,
                    costs,
                )
                minimum = int(chosen["hyper"]["minimum_survivor_count"])
                result.update({
                    "ordering_objective": methods[d1_method]["ordering_objective"],
                    "chosen_order": list(order),
                    "chosen_order_sha256": hash_json(list(order)),
                    "stopping_family": "D2_chain_discounted",
                    "preserve_prefixes": True,
                    "chosen_hyper": dict(chosen["hyper"]),
                    "selection_source": "validation_only",
                    "selection_objective": objective,
                    "selection_fallback_used": fallback,
                    "validation_trajectory": trajectory,
                    "endpoint_reference_method": d1_method,
                    "endpoint_score_identity_verified": True,
                    "endpoint_result_identity_verified": True,
                    "chain_discount_profile": chain_discount_diagnostics(
                        profile, minimum_survivor_count=minimum
                    ),
                })
                if objective == "cost":
                    result["cost_vector_sha256"] = cost_provenance["cost_vector_sha256"]
                methods[d2_method] = result

            if set(methods) != set(REGISTERED_METHODS):
                raise RuntimeError("ISSUE-70 method registry mismatch")
            structural = {
                "target_alpha": alpha,
                "complete_pool_calibration_floor": full_pool[
                    "calibration_standard_all_miss_floor"
                ],
                "complete_pool_test_floor": full_pool["test_standard_all_miss_floor"],
                "calibration_floor_above_target": bool(
                    full_pool["calibration_standard_all_miss_floor"] > alpha
                ),
                "test_floor_above_target": bool(
                    full_pool["test_standard_all_miss_floor"] > alpha
                ),
                "interpretation": (
                    "structurally infeasible on the archived test population"
                    if full_pool["test_standard_all_miss_floor"] > alpha
                    else "not structurally infeasible on the archived test population"
                ),
            }
            cell["redraws"].append({
                "redraw_id": redraw,
                "split_seed": split_seed,
                "split_ids": split_ids,
                "redraw_context_sha256": cost_context["context_sha256"],
                "full_pool": full_pool,
                "structural_feasibility": structural,
                "methods": methods,
            })

    assert_arm_presence(cells)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": 70,
        "design": "heterogeneous_public_price_cost_sensitivity",
        "host": str(host_id),
        "stamp": str(run_stamp),
        "protocol": "routerbench",
        "benchmark": str(task.task),
        "pool_size": len(model_ids),
        "pool_id": hash_json(model_ids),
        "model_ids": model_ids,
        "model_ids_sha256": hash_json(model_ids),
        "query_ids": query_ids,
        "query_ids_sha256": hash_json(query_ids),
        "redraw_count": redraws,
        "redraw_start": redraw_start,
        "redraw_end": redraw_start + redraws,
        "base_seed": base_seed,
        "knn_k": knn_k,
        "target_grid": list(ALPHA_GRID),
        "preserve_prefixes": True,
        "chain_discount_grid": list(CHAIN_GRID),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "method_identities": list(REGISTERED_METHODS),
        "comparison_pairs": [list(pair) for pair in COMPARISON_PAIRS],
        "cost_fit_roles": list(FIT_COST_ROLES),
        "cost_held_out_roles": list(HELD_OUT_COST_ROLES),
        "cost_fit_statement": (
            "Per-model mean public-price costs use only design and router rows. "
            "Actual test costs are used only after frozen selection for reporting."
        ),
        "source_hashes": dict(task.source_hashes),
        "feature_audit": dict(task.feature_audit),
        "redraw_contexts": redraw_contexts,
        "cells": cells,
    }
    payload["aggregates"] = recompute_aggregates(cells)
    payload["paired_bootstrap_summaries"] = build_bootstrap_summaries(
        str(task.task), cells
    )
    return payload


def align_routerbench_costs(data: Any, task: Any) -> np.ndarray:
    """Align audited RouterBench costs to one registered task by exact IDs."""

    if list(data.model_columns) != list(task.model_ids):
        raise ValueError("RouterBench public-price model IDs differ from task model IDs")
    all_costs = _routerbench_costs(data)
    index = {str(query_id): row for row, query_id in enumerate(data.query_ids)}
    if len(index) != len(data.query_ids):
        raise ValueError("RouterBench cost source query IDs are not unique")
    try:
        rows = [index[str(query_id)] for query_id in task.query_ids]
    except KeyError as error:
        raise ValueError(f"task query ID missing from cost source: {error.args[0]}") from error
    aligned = all_costs[np.asarray(rows, dtype=int)]
    tasks = data.frame.iloc[rows]["task"].astype(str).tolist()
    if any(value != str(task.task) for value in tasks):
        raise ValueError("cost alignment crossed RouterBench task boundaries")
    return aligned


def _safe_identity(value: str, name: str) -> str:
    text = str(value)
    if not text or re.fullmatch(r"[A-Za-z0-9_.-]+", text) is None:
        raise ValueError(f"{name} must use only letters, digits, dot, underscore, or hyphen")
    return text


def publish_output_directory(
    out_dir: str | Path,
    task_payloads: Mapping[str, Mapping[str, Any]],
    *,
    host: str,
    stamp: str,
) -> Path:
    """Publish a complete immutable output tree with one atomic rename."""

    destination = Path(out_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output collision {temporary}")
    temporary.mkdir()
    try:
        files = {}
        for task_name in APPROVED_TASKS:
            if task_name not in task_payloads:
                raise ValueError(f"missing approved task payload {task_name}")
            filename = f"issue70_{host}_{stamp}_{task_name}.json"
            path = temporary / filename
            atomic_write_json(path, task_payloads[task_name])
            files[task_name] = {
                "filename": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "issue": 70,
            "host": host,
            "stamp": stamp,
            "out_identity": str(destination),
            "approved_tasks": list(APPROVED_TASKS),
            "pool_size": POOL_SIZE,
            "redraw_count": REDRAWS,
            "target_grid": list(ALPHA_GRID),
            "files": files,
        }
        atomic_write_json(
            temporary / f"issue70_{host}_{stamp}_manifest.json", manifest
        )
        done = temporary / f"issue70_{host}_{stamp}_DONE"
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
    parser.add_argument("--routerbench-pickle", required=True, type=Path)
    parser.add_argument("--embedding-archive", required=True, type=Path)
    parser.add_argument("--embedding-manifest", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        host = _safe_identity(args.host, "host")
        stamp = _safe_identity(args.stamp, "stamp")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite immutable output {args.out_dir}")

    _ensure_pandas_pickle_compat()
    cost_data = load_routerbench(
        args.routerbench_pickle,
        shot="0shot",
        model_columns=ROUTERBENCH_MODEL_COLUMNS,
    )
    payloads = {}
    for benchmark in APPROVED_TASKS:
        task = _load_task(
            "routerbench",
            benchmark,
            routerbench_pickle=args.routerbench_pickle,
            embedding_archive=args.embedding_archive,
            embedding_manifest=args.embedding_manifest,
        )
        if len(task.model_ids) != POOL_SIZE:
            raise SystemExit(
                f"{benchmark}: registered ISSUE-70 pool must contain {POOL_SIZE} models"
            )
        costs = align_routerbench_costs(cost_data, task)
        payloads[benchmark] = run_issue70_task(
            task,
            costs,
            host_id=host,
            run_stamp=stamp,
        )
    try:
        published = publish_output_directory(
            args.out_dir, payloads, host=host, stamp=stamp
        )
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"status": "complete", "output": str(published)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
