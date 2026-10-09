"""Run every fixed O2 order with D1 product and D2 chain stopping."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.chain_discount import (
    chain_discount_diagnostics,
    chain_discounted_scores,
    fit_chain_discount,
)
from components.racer_protocol import select_standard_crc_threshold, smooth_nonconformity_scores
from components.racer_wrapper import fixed_order_factorized_scores
from components.scale_selectors import HYPERPARAMETER_GRIDS, HYPERPARAMETER_NAMES, build_order
from scripts.run_paired_main_suite import (
    ALPHA_GRID,
    BASE_SEED,
    SCHEMA_VERSION,
    SPLIT_STEP,
    _knn,
    _load_task,
    _split_query_ids,
    _target_split,
    assert_registered_arm_presence,
    atomic_write_json,
    evaluate_score_family,
    hash_json,
    recompute_aggregates,
)

CHAIN_GRID = (0, 5, 20, 50)
FIXED_METHODS = OrderedDict((
    ("fixed_tally_factorized", "empirical_tally"),
    ("fixed_double_fault_factorized", "double_fault"),
    ("fixed_error_correlation_factorized", "error_correlation"),
    ("fixed_rank_correlation_factorized", "rank_correlation"),
    ("fixed_gaussian_copula_factorized", "gaussian_copula"),
    ("fixed_student_t_copula_factorized", "student_t_copula"),
    ("fixed_factor_copula_factorized", "factor_copula"),
    ("fixed_bernoulli_mixture_factorized", "bernoulli_mixture"),
    ("fixed_shrinkage_indep_factorized", "shrinkage_indep"),
    ("fixed_shrinkage_factor_factorized", "shrinkage_factor"),
    ("fixed_lineage_factorized", "lineage"),
    ("fixed_design_marginal_factorized", "marginal"),
    ("fixed_seeded_random_factorized", "seeded_random"),
    ("fixed_accuracy_weighted_random_factorized", "accuracy_weighted_random"),
))
D1_METHODS = tuple(FIXED_METHODS)
D2_METHODS = tuple(name.replace("_factorized", "_chain_discounted") for name in D1_METHODS)
REGISTERED_METHODS = tuple(name for pair in zip(D1_METHODS, D2_METHODS) for name in pair)


def _smooth_fixed(
    raw_scores: np.ndarray,
    order: Sequence[int],
    *,
    base_seed: int,
    split_seed: int,
) -> np.ndarray:
    """Call T1's shared smoothing in the declared fixed-order sequence."""

    order_array = np.asarray(order, dtype=int)
    values = np.asarray(raw_scores, dtype=float)
    ordered = values[:, order_array]
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


def _fallback_standard(
    scores: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
) -> tuple[float, float, float]:
    values = np.asarray(scores[rows], dtype=float)
    outcomes = np.asarray(labels[rows], dtype=bool)
    n_queries, n_models = values.shape
    flat_scores = values.ravel()
    flat_correct = outcomes.ravel()
    sequence = np.argsort(flat_scores, kind="stable")
    first = float(flat_scores[sequence[0]])
    best = (1.0, 0.0, float(np.nextafter(first, -np.inf)))
    successful = np.zeros(n_queries, dtype=bool)
    calls_total = 0
    start = 0
    while start < len(sequence):
        threshold = float(flat_scores[sequence[start]])
        end = start + 1
        while end < len(sequence) and flat_scores[sequence[end]] == threshold:
            end += 1
        selected = sequence[start:end]
        calls_total += end - start
        correct = selected[flat_correct[selected]]
        if len(correct):
            successful[np.unique(correct // n_models)] = True
        best = min(
            best,
            (float(np.mean(~successful)), calls_total / n_queries, threshold),
        )
        start = end
    return best


def _select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    validation_rows: np.ndarray,
    alpha: float,
) -> tuple[Mapping[str, Any], list[dict[str, Any]], bool]:
    trajectory = []
    for candidate in candidates:
        scores = np.asarray(candidate["scores"])
        temporary = select_standard_crc_threshold(
            scores[validation_rows], labels[validation_rows], alpha
        )
        if temporary.feasible:
            threshold = float(temporary.threshold)
            chosen = scores[validation_rows] <= threshold
            calls = float(chosen.sum(axis=1).mean())
            risk = float(
                np.mean(~np.any(chosen & labels[validation_rows].astype(bool), axis=1))
            )
        else:
            risk, calls, threshold = _fallback_standard(
                scores, labels, validation_rows
            )
        trajectory.append({
            "grid_index": int(candidate["grid_index"]),
            "hyper": candidate["hyper"],
            "validation_certified": bool(temporary.feasible),
            "validation_threshold": threshold,
            "validation_calls": calls,
            "validation_standard_all_miss_risk": risk,
            "validation_non_certification_reason": temporary.non_certification_reason,
            "validation_crc_bound": float(temporary.crc_bound),
        })
    certified = [row for row in trajectory if row["validation_certified"]]
    pool = certified if certified else trajectory
    if certified:
        chosen_row = min(
            pool,
            key=lambda row: (
                row["validation_calls"],
                row["validation_standard_all_miss_risk"],
                row["grid_index"],
            ),
        )
    else:
        chosen_row = min(
            pool,
            key=lambda row: (
                row["validation_standard_all_miss_risk"],
                row["validation_calls"],
                row["grid_index"],
            ),
        )
    return candidates[int(chosen_row["grid_index"])], trajectory, not bool(certified)


def _fit_d1_candidates(
    probabilities: np.ndarray,
    design_labels: np.ndarray,
    model_ids: Sequence[str],
    *,
    base_seed: int,
    split_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    n_models = probabilities.shape[1]
    fitted: dict[str, list[dict[str, Any]]] = {}
    for method, arm in FIXED_METHODS.items():
        candidates = []
        for grid_index, hyper in enumerate(HYPERPARAMETER_GRIDS[arm]):
            order = build_order(
                arm,
                design_labels,
                model_names=list(model_ids),
                max_depth=n_models,
                seed=split_seed,
                n_draws=2000,
                hyper=hyper,
            )
            if len(order) != n_models or len(set(order)) != n_models:
                raise RuntimeError(f"{method} did not return a full permutation")
            raw = fixed_order_factorized_scores(
                probabilities, order, max_depth=n_models
            )
            candidates.append({
                "grid_index": grid_index,
                "hyper": hyper,
                "order": tuple(int(index) for index in order),
                "scores": _smooth_fixed(
                    raw,
                    order,
                    base_seed=base_seed,
                    split_seed=split_seed,
                ),
            })
        fitted[method] = candidates
    if set(fitted) != set(D1_METHODS):
        raise RuntimeError("D1 fitted family registry mismatch")
    return fitted


def _evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    alpha: float,
    method: str,
) -> dict[str, Any]:
    return evaluate_score_family(
        scores,
        labels,
        split["final_calibration"],
        split["test"],
        query_ids,
        model_ids,
        alpha,
        method_family=method,
    )


def run_issue78(
    task: Any,
    *,
    protocol: str,
    pool_size: int,
    pool_seed: int,
    redraw_start: int,
    redraw_end: int,
    knn_k: int,
    host_id: str,
    run_stamp: str,
    worker_count: int,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    if protocol not in {"routereval", "routerbench"}:
        raise ValueError("protocol must be routereval or routerbench")
    if redraw_start < 0 or redraw_end <= redraw_start:
        raise ValueError("redraw range must be non-empty and non-negative")
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    from components.routereval_protocol import subsample_pool

    columns = subsample_pool(len(task.model_ids), pool_size, seed=pool_seed)
    model_ids = [task.model_ids[int(index)] for index in columns]
    labels = np.asarray(task.correctness[:, columns], dtype=np.int8)
    embeddings = np.asarray(task.embeddings, dtype=float)
    cells = [{"alpha": float(alpha), "redraws": []} for alpha in ALPHA_GRID]
    split_roles = ("design", "router", "final_calibration", "validation", "test")

    for redraw in range(redraw_start, redraw_end):
        split = _target_split(len(task.query_ids), redraw, base_seed)
        if tuple(split) != split_roles:
            raise RuntimeError("registered split roles changed")
        split_ids = _split_query_ids(task, split)
        probabilities, _ = _knn(
            embeddings[split["router"]],
            labels[split["router"]],
            embeddings,
            knn_k,
        )
        split_seed = base_seed + SPLIT_STEP * redraw
        fitted = _fit_d1_candidates(
            probabilities,
            labels[split["design"]],
            model_ids,
            base_seed=base_seed,
            split_seed=split_seed,
        )
        calibration_rows = split["final_calibration"]
        test_rows = split["test"]
        full_pool_cal_miss = np.all(labels[calibration_rows] == 0, axis=1).astype(int)
        full_pool_test_miss = np.all(labels[test_rows] == 0, axis=1).astype(int)
        full_pool = {
            "model_ids": model_ids,
            "model_ids_sha256": hash_json(model_ids),
            "calibration_query_ids": split_ids["final_calibration"],
            "calibration_miss_by_query": [int(value) for value in full_pool_cal_miss],
            "calibration_misses": int(full_pool_cal_miss.sum()),
            "calibration_count": int(len(calibration_rows)),
            "test_query_ids": split_ids["test"],
            "test_miss_by_query": [int(value) for value in full_pool_test_miss],
            "test_misses": int(full_pool_test_miss.sum()),
            "test_count": int(len(test_rows)),
        }

        for cell in cells:
            alpha = float(cell["alpha"])
            methods: dict[str, dict[str, Any]] = {}
            for d1_method in D1_METHODS:
                chosen, order_trajectory, order_fallback = _select_candidate(
                    fitted[d1_method], labels, split["validation"], alpha
                )
                order = chosen["order"]
                d1_base = _evaluate(
                    chosen["scores"],
                    labels,
                    split,
                    task.query_ids,
                    model_ids,
                    alpha,
                    d1_method,
                )
                d1_result = dict(d1_base)
                d1_result.update({
                    "stopping_family": "D1_independence_product",
                    "ordering_arm": FIXED_METHODS[d1_method],
                    "chosen_hyper": chosen["hyper"],
                    "chosen_order": list(order),
                    "chosen_order_sha256": hash_json(list(order)),
                    "selection_source": "validation_only",
                    "selection_fallback_used": order_fallback,
                    "validation_trajectory": order_trajectory,
                })
                methods[d1_method] = d1_result

                profile = fit_chain_discount(labels[split["design"]], order)
                chain_candidates = []
                for grid_index, minimum in enumerate(CHAIN_GRID):
                    raw = chain_discounted_scores(
                        probabilities,
                        profile,
                        minimum_survivor_count=minimum,
                    )
                    chain_candidates.append({
                        "grid_index": grid_index,
                        "hyper": {"minimum_survivor_count": minimum},
                        "order": order,
                        "scores": _smooth_fixed(
                            raw,
                            order,
                            base_seed=base_seed,
                            split_seed=split_seed,
                        ),
                    })
                if not np.array_equal(chain_candidates[0]["scores"], chosen["scores"]):
                    maximum_error = float(
                        np.max(np.abs(chain_candidates[0]["scores"] - chosen["scores"]))
                    )
                    raise RuntimeError(
                        f"{d1_method} D2 off endpoint differs from D1 by {maximum_error}"
                    )
                endpoint_result = _evaluate(
                    chain_candidates[0]["scores"],
                    labels,
                    split,
                    task.query_ids,
                    model_ids,
                    alpha,
                    d1_method,
                )
                if endpoint_result != d1_base:
                    raise RuntimeError(
                        f"{d1_method} calibrated D2 off endpoint differs from D1"
                    )
                chosen_chain, chain_trajectory, chain_fallback = _select_candidate(
                    chain_candidates, labels, split["validation"], alpha
                )
                d2_method = d1_method.replace("_factorized", "_chain_discounted")
                d2_result = _evaluate(
                    chosen_chain["scores"],
                    labels,
                    split,
                    task.query_ids,
                    model_ids,
                    alpha,
                    d2_method,
                )
                minimum = int(chosen_chain["hyper"]["minimum_survivor_count"])
                d2_result.update({
                    "stopping_family": "D2_chain_discounted",
                    "ordering_arm": FIXED_METHODS[d1_method],
                    "chosen_hyper": chosen_chain["hyper"],
                    "chosen_order": list(order),
                    "chosen_order_sha256": hash_json(list(order)),
                    "selection_source": "validation_only",
                    "selection_fallback_used": chain_fallback,
                    "validation_trajectory": chain_trajectory,
                    "endpoint_reference_method": d1_method,
                    "endpoint_score_identity_verified": True,
                    "endpoint_result_identity_verified": True,
                    "chain_discount_profile": chain_discount_diagnostics(
                        profile,
                        minimum_survivor_count=minimum,
                    ),
                })
                methods[d2_method] = d2_result
            if set(methods) != set(REGISTERED_METHODS):
                raise RuntimeError("ISSUE-78 method registry mismatch")
            cell["redraws"].append({
                "redraw_id": int(redraw),
                "split_seed": int(split_seed),
                "split_ids": split_ids,
                "full_pool": full_pool,
                "methods": methods,
            })

    assert_registered_arm_presence(cells, REGISTERED_METHODS)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "design": "ISSUE_78_full_matrix_O2_D1_D2",
        "issue": 78,
        "host": host_id,
        "stamp": run_stamp,
        "worker_count": int(worker_count),
        "protocol": protocol,
        "benchmark": task.task,
        "pool_id": hash_json({"pool_seed": pool_seed, "model_ids": model_ids}),
        "pool_size": int(pool_size),
        "pool_seed": int(pool_seed),
        "base_seed": int(base_seed),
        "knn_k": int(knn_k),
        "redraw_start": int(redraw_start),
        "redraw_end": int(redraw_end),
        "redraw_range": {
            "start_inclusive": int(redraw_start),
            "end_exclusive": int(redraw_end),
            "count": int(redraw_end - redraw_start),
        },
        "target_grid": list(ALPHA_GRID),
        "preserve_prefixes": True,
        "chain_discount_grid": list(CHAIN_GRID),
        "d1_methods": list(D1_METHODS),
        "d2_methods": list(D2_METHODS),
        "method_identities": list(REGISTERED_METHODS),
        "ordering_hyperparameter_grids": {
            arm: list(HYPERPARAMETER_GRIDS[arm]) for arm in FIXED_METHODS.values()
        },
        "ordering_hyperparameter_names": {
            arm: HYPERPARAMETER_NAMES[arm] for arm in FIXED_METHODS.values()
        },
        "query_ids": list(task.query_ids),
        "query_ids_sha256": hash_json(list(task.query_ids)),
        "model_ids": model_ids,
        "model_ids_sha256": hash_json(model_ids),
        "sampled_model_column_indices": [int(value) for value in columns],
        "source_hashes": dict(task.source_hashes),
        "feature_audit": dict(task.feature_audit),
        "cells": cells,
    }
    payload["aggregates"] = recompute_aggregates(cells)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, choices=("routereval", "routerbench"))
    parser.add_argument("--leaderboard", type=Path)
    parser.add_argument("--router-dir", type=Path)
    parser.add_argument("--routerbench-pickle", type=Path)
    parser.add_argument("--embedding-archive", type=Path)
    parser.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pool-size", required=True, type=int)
    parser.add_argument("--pool-seed", type=int, default=BASE_SEED)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--knn-k", type=int, default=40)
    parser.add_argument("--redraw-start", required=True, type=int)
    parser.add_argument("--redraw-end", required=True, type=int)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--worker-count", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite immutable result {args.out}")
    try:
        task = _load_task(
            args.protocol,
            args.benchmark,
            leaderboard=args.leaderboard,
            router_dir=args.router_dir,
            routerbench_pickle=args.routerbench_pickle,
            embedding_archive=args.embedding_archive,
            embedding_manifest=args.embedding_manifest,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    payload = run_issue78(
        task,
        protocol=args.protocol,
        pool_size=args.pool_size,
        pool_seed=args.pool_seed,
        redraw_start=args.redraw_start,
        redraw_end=args.redraw_end,
        knn_k=args.knn_k,
        host_id=args.host_id,
        run_stamp=args.run_stamp,
        worker_count=args.worker_count,
        base_seed=args.base_seed,
    )
    atomic_write_json(args.out, payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
