"""Run ISSUE-65 exact, beam, and local-swap min-sum ordering comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.racer_wrapper import fixed_order_factorized_scores
from components.scale_selectors import empirical_tally_order
from components.set_optimizers import (
    beam_min_sum_order,
    exact_min_sum_order,
    local_swap_min_sum_order,
    min_sum_objective,
    prefix_miss_counts,
)
from scripts.run_issue78_o2_full_matrix import _select_candidate, _smooth_fixed
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

EXACT = "fixed_exact_minsum_factorized"
BEAM = "fixed_beam_minsum_factorized"
SWAP = "fixed_local_swap_minsum_factorized"
BEAM_GRID = (1, 4, 16)
SWAP_GRID = (1, 10, 25, 50)


def _score(
    probabilities: np.ndarray,
    order: Sequence[int],
    *,
    base_seed: int,
    split_seed: int,
) -> np.ndarray:
    raw = fixed_order_factorized_scores(probabilities, order, max_depth=len(order))
    return _smooth_fixed(raw, order, base_seed=base_seed, split_seed=split_seed)


def _subset_miss(labels: np.ndarray, rows: np.ndarray, subset: Sequence[int]) -> int:
    values = np.asarray(labels[np.asarray(rows, dtype=int)][:, list(subset)], dtype=bool)
    return int(np.count_nonzero(~np.any(values, axis=1)))


def _fit_candidates(
    probabilities: np.ndarray,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    *,
    base_seed: int,
    split_seed: int,
    include_exact: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    design = labels[split["design"]]
    n_models = labels.shape[1]
    started = perf_counter()
    tally_order = empirical_tally_order(design, max_depth=n_models)
    tally_scores = _score(
        probabilities, tally_order, base_seed=base_seed, split_seed=split_seed
    )
    tally_fit_seconds = perf_counter() - started
    fitted: dict[str, list[dict[str, Any]]] = {BEAM: [], SWAP: []}
    audit: dict[str, Any] = {
        "tally_order": list(tally_order),
        "tally_order_sha256": hash_json(list(tally_order)),
        "tally_design_objective": min_sum_objective(design, tally_order),
        "tally_design_prefix_misses": list(prefix_miss_counts(design, tally_order)),
        "tally_fit_seconds": float(tally_fit_seconds),
    }

    for grid_index, width in enumerate(BEAM_GRID):
        if width == 1:
            order = tally_order
            diagnostics = {
                "optimizer": "beam_min_sum",
                "beam_width": 1,
                "greedy_endpoint": True,
                "objective": audit["tally_design_objective"],
                "elapsed_seconds": 0.0,
            }
        else:
            order, diagnostics = beam_min_sum_order(design, beam_width=width)
            diagnostics["greedy_endpoint"] = False
        fitted[BEAM].append({
            "grid_index": int(grid_index),
            "hyper": {"beam_width": int(width)},
            "order": order,
            "scores": _score(probabilities, order, base_seed=base_seed, split_seed=split_seed),
            "optimizer_diagnostics": diagnostics,
        })

    for grid_index, maximum in enumerate(SWAP_GRID):
        if maximum == 1:
            order = tally_order
            diagnostics = {
                "optimizer": "local_swap_min_sum",
                "maximum_positions": 1,
                "passes": 0,
                "greedy_endpoint": True,
                "initial_objective": audit["tally_design_objective"],
                "objective": audit["tally_design_objective"],
                "objective_improvement": 0,
                "elapsed_seconds": 0.0,
            }
        else:
            order, diagnostics = local_swap_min_sum_order(
                design, maximum_positions=min(maximum, n_models), passes=2
            )
            diagnostics["greedy_endpoint"] = False
        fitted[SWAP].append({
            "grid_index": int(grid_index),
            "hyper": {"maximum_positions": int(maximum)},
            "order": order,
            "scores": _score(probabilities, order, base_seed=base_seed, split_seed=split_seed),
            "optimizer_diagnostics": diagnostics,
        })

    if not np.array_equal(fitted[BEAM][0]["scores"], tally_scores):
        raise RuntimeError("beam endpoint does not reproduce greedy tally scores")
    if not np.array_equal(fitted[SWAP][0]["scores"], tally_scores):
        raise RuntimeError("local-swap endpoint does not reproduce greedy tally scores")
    audit["beam_greedy_endpoint_verified"] = True
    audit["swap_greedy_endpoint_verified"] = True

    if include_exact:
        exact = exact_min_sum_order(design, maximum_models=11)
        exact_order = exact.order
        exact_diagnostics = {
            "optimizer": "exact_subset_dynamic_program",
            "state_count": exact.state_count,
            "elapsed_seconds": exact.elapsed_seconds,
            "exact_design_objective": exact.objective,
            "greedy_design_objective": exact.greedy_objective,
            "nested_objective_gap": exact.greedy_objective - exact.objective,
            "nested_relative_gap": (
                (exact.greedy_objective - exact.objective) / exact.objective
                if exact.objective > 0 else 0.0
            ),
            "exact_depth_misses": list(exact.exact_depth_misses),
            "exact_depth_subsets": [list(subset) for subset in exact.exact_depth_subsets],
            "greedy_depth_misses": list(prefix_miss_counts(design, exact.greedy_order)),
            "fixed_depth_evaluation": [],
        }
        for depth, subset in enumerate(exact.exact_depth_subsets, start=1):
            greedy_subset = exact.greedy_order[:depth]
            exact_diagnostics["fixed_depth_evaluation"].append({
                "depth": depth,
                "design_exact_misses": _subset_miss(labels, split["design"], subset),
                "design_greedy_misses": _subset_miss(labels, split["design"], greedy_subset),
                "validation_exact_misses": _subset_miss(labels, split["validation"], subset),
                "validation_greedy_misses": _subset_miss(labels, split["validation"], greedy_subset),
                "test_exact_misses": _subset_miss(labels, split["test"], subset),
                "test_greedy_misses": _subset_miss(labels, split["test"], greedy_subset),
                "validation_count": int(len(split["validation"])),
                "test_count": int(len(split["test"])),
            })
        fitted[EXACT] = [{
            "grid_index": 0,
            "hyper": {"optimizer": "exact_subset_dp"},
            "order": exact_order,
            "scores": _score(
                probabilities, exact_order, base_seed=base_seed, split_seed=split_seed
            ),
            "optimizer_diagnostics": exact_diagnostics,
        }]
        audit["exact"] = exact_diagnostics
    return fitted, audit


def _evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    task: Any,
    model_ids: Sequence[str],
    alpha: float,
    method: str,
) -> dict[str, Any]:
    return evaluate_score_family(
        scores,
        labels,
        split["final_calibration"],
        split["test"],
        task.query_ids,
        model_ids,
        alpha,
        method_family=method,
    )


def run_issue65(
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
    include_exact = pool_size <= 11
    registered = (BEAM, SWAP, EXACT) if include_exact else (BEAM, SWAP)
    cells = [{"alpha": float(alpha), "redraws": []} for alpha in ALPHA_GRID]

    for redraw in range(redraw_start, redraw_end):
        split = _target_split(len(task.query_ids), redraw, base_seed)
        probabilities, _ = _knn(
            embeddings[split["router"]], labels[split["router"]], embeddings, knn_k
        )
        split_seed = base_seed + SPLIT_STEP * redraw
        fitted, optimization_audit = _fit_candidates(
            probabilities,
            labels,
            split,
            base_seed=base_seed,
            split_seed=split_seed,
            include_exact=include_exact,
        )
        split_ids = _split_query_ids(task, split)
        for cell in cells:
            alpha = float(cell["alpha"])
            methods: dict[str, dict[str, Any]] = {}
            for method in registered:
                chosen, trajectory, fallback = _select_candidate(
                    fitted[method], labels, split["validation"], alpha
                )
                result = _evaluate(
                    chosen["scores"], labels, split, task, model_ids, alpha, method
                )
                result.update({
                    "ordering_optimizer": chosen["optimizer_diagnostics"]["optimizer"],
                    "chosen_hyper": chosen["hyper"],
                    "chosen_order": list(chosen["order"]),
                    "chosen_order_sha256": hash_json(list(chosen["order"])),
                    "selection_source": "validation_only",
                    "selection_fallback_used": fallback,
                    "validation_trajectory": trajectory,
                    "optimizer_diagnostics": chosen["optimizer_diagnostics"],
                    "greedy_endpoint_available": method in {BEAM, SWAP},
                    "greedy_endpoint_score_identity_verified": method in {BEAM, SWAP},
                })
                methods[method] = result
            cell["redraws"].append({
                "redraw_id": int(redraw),
                "split_seed": int(split_seed),
                "split_ids": split_ids,
                "optimization_audit": optimization_audit,
                "methods": methods,
            })
    assert_registered_arm_presence(cells, registered)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "design": "ISSUE_65_set_optimizer_D1_comparison",
        "issue": 65,
        "host": host_id,
        "stamp": run_stamp,
        "worker_count": int(worker_count),
        "protocol": protocol,
        "benchmark": task.task,
        "pool_size": int(pool_size),
        "pool_seed": int(pool_seed),
        "base_seed": int(base_seed),
        "knn_k": int(knn_k),
        "redraw_start": int(redraw_start),
        "redraw_end": int(redraw_end),
        "target_grid": list(ALPHA_GRID),
        "preserve_prefixes": True,
        "include_exact": include_exact,
        "method_identities": list(registered),
        "beam_grid": list(BEAM_GRID),
        "swap_grid": list(SWAP_GRID),
        "model_ids": model_ids,
        "model_ids_sha256": hash_json(model_ids),
        "sampled_model_column_indices": [int(value) for value in columns],
        "query_ids_sha256": hash_json(list(task.query_ids)),
        "source_hashes": dict(task.source_hashes),
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
    task = _load_task(
        args.protocol,
        args.benchmark,
        leaderboard=args.leaderboard,
        router_dir=args.router_dir,
        routerbench_pickle=args.routerbench_pickle,
        embedding_archive=args.embedding_archive,
        embedding_manifest=args.embedding_manifest,
    )
    payload = run_issue65(
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
