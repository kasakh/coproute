"""Cross ISSUE-77 chain discount with selected non-tally ISSUE-74 orders."""

from __future__ import annotations

import argparse
import json
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
from components.racer_wrapper import augment_with_racer_null, fixed_order_factorized_scores
from scripts.run_issue74_ordering_factorial import (
    _evaluate_routerbench,
    _evaluate_routereval,
    _knn_probabilities,
    _load_routereval_task,
    _routerbench_task,
    _smooth,
    _split,
)
from scripts.run_paired_main_suite import ALPHA_GRID, BASE_SEED, atomic_write_json, hash_json

SCHEMA_VERSION = "issue77_selected_order_cross_v1"
CHAIN_GRID = (0, 5, 20, 50)
DEFAULT_METHODS = (
    "fixed_error_correlation_factorized",
    "fixed_bernoulli_mixture_factorized",
)


def result_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical fields that establish calibrated endpoint identity."""

    return {
        "certified": record.get("certified"),
        "final_calibration_count": record.get("final_calibration_count"),
        "calibration_misses": record.get("calibration_misses"),
        "unavoidable_calibration_misses": record.get("unavoidable_calibration_misses"),
        "calibration_bound": record.get("corrected_calibration_bound", record.get("crc_bound")),
        "threshold": record.get("threshold", record.get("final_threshold")),
        "test_calls": record.get("test_calls", record.get("calls_test")),
        "standard_all_miss_risk": record.get("standard_all_miss_risk"),
        "test_calls_by_query": record.get("test_calls_by_query"),
        "test_miss_by_query": record.get("test_miss_by_query"),
    }


def _score(
    raw: np.ndarray,
    probabilities: np.ndarray,
    order: Sequence[int],
    *,
    protocol: str,
    base_seed: int,
    redraw: int,
) -> np.ndarray:
    if protocol == "routerbench":
        raw = augment_with_racer_null(probabilities, raw)
        split_seed = redraw + 1
    else:
        split_seed = base_seed + 13 * redraw
    return _smooth(
        raw,
        seed=base_seed,
        split_seed=split_seed,
        preserve_score_prefixes=True,
        fixed_order=order,
    )


def _evaluate(
    method: str,
    candidates: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    task: Any,
    model_ids: Sequence[str],
    protocol: str,
    alpha: float,
) -> dict[str, Any]:
    if protocol == "routereval":
        return _evaluate_routereval(
            method,
            candidates,
            labels,
            split,
            task.query_ids,
            model_ids,
            alpha,
        )
    return _evaluate_routerbench(
        method,
        candidates,
        probabilities,
        labels,
        split,
        alpha,
    )


def run_cross(
    task: Any,
    reference: Mapping[str, Any],
    *,
    methods: Sequence[str],
    protocol: str,
    pool_size: int,
    pool_seed: int,
    redraw_start: int,
    redraw_end: int,
    knn_k: int,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    selected_methods = tuple(methods)
    if redraw_start < 0 or redraw_end <= redraw_start:
        raise ValueError("redraw range must be non-empty and non-negative")
    if not selected_methods or len(selected_methods) != len(set(selected_methods)):
        raise ValueError("selected methods must be non-empty and unique")
    if tuple(reference.get("selected_methods", ())) != selected_methods:
        raise ValueError("reference manifest selected methods differ")
    if reference.get("protocol") != protocol or reference.get("benchmark") != task.task:
        raise ValueError("reference manifest task identity differs")
    if int(reference.get("pool_size")) != pool_size:
        raise ValueError("reference manifest pool size differs")

    if protocol == "routereval":
        from components.routereval_protocol import subsample_pool
        columns = subsample_pool(len(task.model_ids), pool_size, seed=pool_seed)
        model_ids = [task.model_ids[int(index)] for index in columns]
        labels = np.asarray(task.correctness[:, columns], dtype=np.int8)
    else:
        if pool_size != len(task.model_ids):
            raise ValueError("RouterBench requires its complete model pool")
        columns = np.arange(pool_size)
        model_ids = list(task.model_ids)
        labels = np.asarray(task.correctness, dtype=np.int8)
    if hash_json(model_ids) != reference.get("model_ids_sha256"):
        raise ValueError("reference manifest model identity differs")
    if hash_json(list(task.query_ids)) != reference.get("query_ids_sha256"):
        raise ValueError("reference manifest query identity differs")

    output_methods = tuple(
        name
        for method in selected_methods
        for name in (method, method.replace("_factorized", "_chain_discounted"))
    )
    cells = [{"alpha": float(alpha), "redraws": []} for alpha in ALPHA_GRID]
    reference_redraws = reference["redraws"]
    for redraw in range(redraw_start, redraw_end):
        split = _split(task, protocol, redraw, base_seed)
        if protocol == "routereval":
            probability_rows = np.arange(len(task.query_ids))
        else:
            probability_rows = np.concatenate((
                split["final_calibration"], split["validation"], split["test"]
            ))
        probabilities, _ = _knn_probabilities(
            task.embeddings[split["router"]],
            labels[split["router"]],
            task.embeddings[probability_rows],
            knn_k,
            return_support=True,
        )
        split_ids = {
            name: [task.query_ids[int(index)] for index in rows]
            for name, rows in split.items()
        }
        for cell in cells:
            alpha = float(cell["alpha"])
            alpha_key = str(alpha)
            methods_out = {}
            for method in selected_methods:
                ref = reference_redraws[str(redraw)][alpha_key][method]
                order = tuple(int(index) for index in ref["chosen_order"])
                if len(order) != pool_size or len(set(order)) != pool_size:
                    raise RuntimeError(f"{method} reference order is not a full permutation")
                raw_product = fixed_order_factorized_scores(
                    probabilities, order, max_depth=pool_size
                )
                product_scores = _score(
                    raw_product,
                    probabilities,
                    order,
                    protocol=protocol,
                    base_seed=base_seed,
                    redraw=redraw,
                )
                product_candidate = [{
                    "grid_index": 0,
                    "hyper": ref.get("chosen_hyper"),
                    "order": list(order),
                    "scores": product_scores,
                }]
                product_result = _evaluate(
                    method,
                    product_candidate,
                    probabilities,
                    labels,
                    split,
                    task,
                    model_ids,
                    protocol,
                    alpha,
                )
                actual_signature = result_signature(product_result)
                if actual_signature != ref["result_signature"]:
                    raise RuntimeError(
                        f"{method} redraw {redraw} alpha {alpha} product endpoint differs"
                    )
                product_result.update({
                    "endpoint_reference_manifest_verified": True,
                    "reference_issue": "ISSUE-74",
                })
                methods_out[method] = product_result

                profile = fit_chain_discount(labels[split["design"]], order)
                chain_candidates = []
                for grid_index, minimum in enumerate(CHAIN_GRID):
                    raw = chain_discounted_scores(
                        probabilities,
                        profile,
                        minimum_survivor_count=minimum,
                    )
                    scores = _score(
                        raw,
                        probabilities,
                        order,
                        protocol=protocol,
                        base_seed=base_seed,
                        redraw=redraw,
                    )
                    chain_candidates.append({
                        "grid_index": grid_index,
                        "hyper": {"minimum_survivor_count": minimum},
                        "order": list(order),
                        "scores": scores,
                    })
                if not np.array_equal(chain_candidates[0]["scores"], product_scores):
                    error = float(np.max(np.abs(chain_candidates[0]["scores"] - product_scores)))
                    raise RuntimeError(f"{method} discount-off score endpoint differs by {error}")
                chain_name = method.replace("_factorized", "_chain_discounted")
                chain_result = _evaluate(
                    chain_name,
                    chain_candidates,
                    probabilities,
                    labels,
                    split,
                    task,
                    model_ids,
                    protocol,
                    alpha,
                )
                minimum = int(chain_result["chosen_hyper"]["minimum_survivor_count"])
                chain_result.update({
                    "endpoint_reference_method": method,
                    "endpoint_score_identity_verified": True,
                    "endpoint_result_identity_verified": True,
                    "chain_discount_profile": chain_discount_diagnostics(
                        profile, minimum_survivor_count=minimum
                    ),
                })
                methods_out[chain_name] = chain_result
            if set(methods_out) != set(output_methods):
                raise RuntimeError("selected order cross has a missing output arm")
            cell["redraws"].append({
                "redraw_id": redraw,
                "split_seed": base_seed + 13 * redraw,
                "split_ids": split_ids,
                "methods": methods_out,
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": "ISSUE-77",
        "host": os.environ.get("DACR_MACHINE_ALIAS", os.uname().nodename),
        "utc_stamp": os.environ.get("DACR_RUN_STAMP", ""),
        "protocol": protocol,
        "benchmark": task.task,
        "pool_size": pool_size,
        "pool_seed": pool_seed,
        "base_seed": base_seed,
        "knn_k": knn_k,
        "redraw_start": redraw_start,
        "redraw_end": redraw_end,
        "target_grid": list(ALPHA_GRID),
        "preserve_score_prefixes": True,
        "selected_methods": list(selected_methods),
        "registered_methods": list(output_methods),
        "chain_discount_grid": list(CHAIN_GRID),
        "reference_manifest_sha256": reference.get("manifest_sha256"),
        "query_ids": list(task.query_ids),
        "query_ids_sha256": hash_json(list(task.query_ids)),
        "model_ids": model_ids,
        "model_ids_sha256": hash_json(model_ids),
        "source_hashes": dict(task.source_hashes),
        "feature_audit": dict(task.feature_audit),
        "cells": cells,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, choices=("routereval", "routerbench"))
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pool-size", required=True, type=int)
    parser.add_argument("--pool-seed", type=int, default=BASE_SEED)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--knn-k", type=int, default=40)
    parser.add_argument("--redraw-start", required=True, type=int)
    parser.add_argument("--redraw-end", required=True, type=int)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--leaderboard", type=Path)
    parser.add_argument("--router-dir", type=Path)
    parser.add_argument("--routerbench-pkl", type=Path)
    parser.add_argument("--embedding-archive", type=Path)
    parser.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    if args.protocol == "routereval":
        if args.leaderboard is None or args.router_dir is None:
            parser.error("RouterEval requires --leaderboard and --router-dir")
        task = _load_routereval_task(
            args.leaderboard,
            args.router_dir,
            args.benchmark,
        )
    else:
        if any(value is None for value in (args.routerbench_pkl, args.embedding_archive, args.embedding_manifest)):
            parser.error("RouterBench requires its pickle and embedding files")
        task = _routerbench_task(
            args.routerbench_pkl,
            args.embedding_archive,
            args.embedding_manifest,
            args.benchmark,
        )
    payload = run_cross(
        task,
        reference,
        methods=methods,
        protocol=args.protocol,
        pool_size=args.pool_size,
        pool_seed=args.pool_seed,
        redraw_start=args.redraw_start,
        redraw_end=args.redraw_end,
        knn_k=args.knn_k,
        base_seed=args.base_seed,
    )
    atomic_write_json(args.out, payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
