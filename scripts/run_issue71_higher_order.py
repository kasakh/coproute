"""Run ISSUE-71 higher-order fixed orders in the ISSUE-78 D1/D2 framework."""

from __future__ import annotations

import argparse
from collections import OrderedDict
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
from components.higher_order_selectors import (
    TETRACHORIC_RIDGE_GRID,
    VINE_LAPLACE_GRID,
    discrete_vine_tree_order,
    tetrachoric_diagnostics,
    tetrachoric_normal_score_order,
    vine_diagnostics,
)
from components.racer_wrapper import fixed_order_factorized_scores
from scripts.run_issue78_o2_full_matrix import CHAIN_GRID, _select_candidate, _smooth_fixed
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

HIGHER_ORDER_METHODS = OrderedDict((
    ("fixed_tetrachoric_normal_score_factorized", "tetrachoric_normal_score"),
    ("fixed_discrete_vine_tree_factorized", "discrete_vine_tree"),
))
D1_METHODS = tuple(HIGHER_ORDER_METHODS)
D2_METHODS = tuple(name.replace("_factorized", "_chain_discounted") for name in D1_METHODS)
REGISTERED_METHODS = tuple(name for pair in zip(D1_METHODS, D2_METHODS) for name in pair)
HYPER_GRIDS = {
    "tetrachoric_normal_score": TETRACHORIC_RIDGE_GRID,
    "discrete_vine_tree": VINE_LAPLACE_GRID,
}
HYPER_NAMES = {
    "tetrachoric_normal_score": "latent_correlation_ridge",
    "discrete_vine_tree": "laplace_smoothing",
}


def _fit_candidates(
    probabilities: np.ndarray,
    design_labels: np.ndarray,
    model_ids: Sequence[str],
    *,
    base_seed: int,
    split_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    del model_ids
    n_models = probabilities.shape[1]
    fitted: dict[str, list[dict[str, Any]]] = {}
    for method, arm in HIGHER_ORDER_METHODS.items():
        candidates = []
        for grid_index, hyper in enumerate(HYPER_GRIDS[arm]):
            if arm == "tetrachoric_normal_score":
                order = tetrachoric_normal_score_order(
                    design_labels, n_models, ridge=float(hyper), n_draws=2000, seed=split_seed
                )
                diagnostics = tetrachoric_diagnostics(design_labels, ridge=float(hyper))
            else:
                order = discrete_vine_tree_order(
                    design_labels, n_models, laplace=float(hyper), n_draws=2000, seed=split_seed
                )
                diagnostics = vine_diagnostics(design_labels, laplace=float(hyper))
            if len(order) != n_models or len(set(order)) != n_models:
                raise RuntimeError(f"{method} did not return a full permutation")
            raw = fixed_order_factorized_scores(probabilities, order, max_depth=n_models)
            candidates.append({
                "grid_index": int(grid_index),
                "hyper": {HYPER_NAMES[arm]: hyper},
                "order": tuple(int(index) for index in order),
                "scores": _smooth_fixed(
                    raw, order, base_seed=base_seed, split_seed=split_seed
                ),
                "fit_diagnostics": diagnostics,
            })
        fitted[method] = candidates
    if set(fitted) != set(D1_METHODS):
        raise RuntimeError("ISSUE-71 fitted family registry mismatch")
    return fitted


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


def run_issue71(
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

    for redraw in range(redraw_start, redraw_end):
        split = _target_split(len(task.query_ids), redraw, base_seed)
        probabilities, _ = _knn(
            embeddings[split["router"]], labels[split["router"]], embeddings, knn_k
        )
        split_seed = base_seed + SPLIT_STEP * redraw
        fitted = _fit_candidates(
            probabilities,
            labels[split["design"]],
            model_ids,
            base_seed=base_seed,
            split_seed=split_seed,
        )
        split_ids = _split_query_ids(task, split)
        for cell in cells:
            alpha = float(cell["alpha"])
            methods: dict[str, dict[str, Any]] = {}
            for d1_method in D1_METHODS:
                chosen, trajectory, fallback = _select_candidate(
                    fitted[d1_method], labels, split["validation"], alpha
                )
                order = chosen["order"]
                d1_base = _evaluate(
                    chosen["scores"], labels, split, task, model_ids, alpha, d1_method
                )
                d1_result = dict(d1_base)
                d1_result.update({
                    "stopping_family": "D1_independence_product",
                    "ordering_arm": HIGHER_ORDER_METHODS[d1_method],
                    "chosen_hyper": chosen["hyper"],
                    "chosen_order": list(order),
                    "chosen_order_sha256": hash_json(list(order)),
                    "selection_source": "validation_only",
                    "selection_fallback_used": fallback,
                    "validation_trajectory": trajectory,
                    "fit_diagnostics": chosen["fit_diagnostics"],
                })
                methods[d1_method] = d1_result

                profile = fit_chain_discount(labels[split["design"]], order)
                chain_candidates = []
                for grid_index, minimum in enumerate(CHAIN_GRID):
                    raw = chain_discounted_scores(
                        probabilities, profile, minimum_survivor_count=minimum
                    )
                    chain_candidates.append({
                        "grid_index": int(grid_index),
                        "hyper": {"minimum_survivor_count": int(minimum)},
                        "order": order,
                        "scores": _smooth_fixed(
                            raw, order, base_seed=base_seed, split_seed=split_seed
                        ),
                    })
                if not np.array_equal(chain_candidates[0]["scores"], chosen["scores"]):
                    raise RuntimeError(f"{d1_method} D2 off score differs from D1")
                endpoint = _evaluate(
                    chain_candidates[0]["scores"], labels, split, task, model_ids, alpha, d1_method
                )
                if endpoint != d1_base:
                    raise RuntimeError(f"{d1_method} D2 off result differs from D1")
                chosen_chain, chain_trajectory, chain_fallback = _select_candidate(
                    chain_candidates, labels, split["validation"], alpha
                )
                d2_method = d1_method.replace("_factorized", "_chain_discounted")
                d2_result = _evaluate(
                    chosen_chain["scores"], labels, split, task, model_ids, alpha, d2_method
                )
                minimum = int(chosen_chain["hyper"]["minimum_survivor_count"])
                d2_result.update({
                    "stopping_family": "D2_chain_discounted",
                    "ordering_arm": HIGHER_ORDER_METHODS[d1_method],
                    "chosen_hyper": chosen_chain["hyper"],
                    "chosen_order": list(order),
                    "chosen_order_sha256": hash_json(list(order)),
                    "selection_source": "validation_only",
                    "selection_fallback_used": chain_fallback,
                    "validation_trajectory": chain_trajectory,
                    "fit_diagnostics": chosen["fit_diagnostics"],
                    "endpoint_reference_method": d1_method,
                    "endpoint_score_identity_verified": True,
                    "endpoint_result_identity_verified": True,
                    "chain_discount_profile": chain_discount_diagnostics(
                        profile, minimum_survivor_count=minimum
                    ),
                })
                methods[d2_method] = d2_result
            if set(methods) != set(REGISTERED_METHODS):
                raise RuntimeError("ISSUE-71 arm registry mismatch")
            cell["redraws"].append({
                "redraw_id": int(redraw),
                "split_seed": int(split_seed),
                "split_ids": split_ids,
                "methods": methods,
            })

    assert_registered_arm_presence(cells, REGISTERED_METHODS)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "design": "ISSUE_71_higher_order_O2_D1_D2",
        "issue": 71,
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
        "chain_discount_grid": list(CHAIN_GRID),
        "d1_methods": list(D1_METHODS),
        "d2_methods": list(D2_METHODS),
        "method_identities": list(REGISTERED_METHODS),
        "ordering_hyperparameter_grids": {
            arm: list(HYPER_GRIDS[arm]) for arm in HIGHER_ORDER_METHODS.values()
        },
        "ordering_hyperparameter_names": dict(HYPER_NAMES),
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
    payload = run_issue71(
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
