"""Run the leakage-free validation-based fixed-order selector for ISSUE-79."""

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

from scripts.run_issue78_o2_full_matrix import (
    FIXED_METHODS,
    _fit_d1_candidates,
)
from scripts.run_paired_main_suite import (
    ALPHA_GRID,
    BASE_SEED,
    SCHEMA_VERSION,
    SPLIT_STEP,
    _knn,
    _load_task,
    _split_query_ids,
    _target_split,
    atomic_write_json,
    evaluate_score_family,
    hash_json,
)

SCIENTIFIC_METHODS = tuple(
    method
    for method in FIXED_METHODS
    if FIXED_METHODS[method] not in {"seeded_random", "accuracy_weighted_random"}
)
RANDOM_CONTROL_METHODS = tuple(
    method for method in FIXED_METHODS if method not in SCIENTIFIC_METHODS
)
if len(SCIENTIFIC_METHODS) != 12:
    raise RuntimeError("ISSUE-79 scientific family registry must contain 12 methods")
FAMILY_BIAS_MULTIPLIER = float(np.sqrt(2.0 * np.log(len(SCIENTIFIC_METHODS))))
BASELINE_METHOD = "fixed_tally_factorized"
ERROR_CORRELATION_METHOD = "fixed_error_correlation_factorized"
OUTPUT_METHODS = (
    "validation_selected_factorized",
    BASELINE_METHOD,
    ERROR_CORRELATION_METHOD,
)


def split_validation_rows(validation_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the registered validation permutation into disjoint tune/select halves."""

    rows = np.asarray(validation_rows, dtype=int)
    if rows.ndim != 1 or len(rows) < 4 or len(np.unique(rows)) != len(rows):
        raise ValueError("validation rows must contain at least four unique indices")
    tune = rows[::2].copy()
    select = rows[1::2].copy()
    if len(tune) == 0 or len(select) == 0:
        raise RuntimeError("validation halves must both be non-empty")
    if np.intersect1d(tune, select).size:
        raise RuntimeError("validation tune/select overlap")
    if set(np.concatenate((tune, select))) != set(rows):
        raise RuntimeError("validation tune/select partition is incomplete")
    return tune, select


def discovery_losses(
    labels: np.ndarray,
    rows: np.ndarray,
    order: Sequence[int],
) -> np.ndarray:
    """Return first-correct prefix rank, with pool size plus one for full misses."""

    outcomes = np.asarray(labels[np.asarray(rows, dtype=int)], dtype=bool)
    permutation = np.asarray(order, dtype=int)
    if permutation.ndim != 1 or len(permutation) != outcomes.shape[1]:
        raise ValueError("order must be a full-pool permutation")
    if len(np.unique(permutation)) != len(permutation):
        raise ValueError("order contains duplicate model indices")
    ordered = outcomes[:, permutation]
    has_success = np.any(ordered, axis=1)
    ranks = np.argmax(ordered, axis=1).astype(float) + 1.0
    ranks[~has_success] = float(outcomes.shape[1] + 1)
    return ranks


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("selection losses must be a non-empty finite vector")
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, se


def tune_family(
    candidates: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    tune_rows: np.ndarray,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Choose one distinct order within a family using penalized tune loss."""

    unique: OrderedDict[str, Mapping[str, Any]] = OrderedDict()
    for candidate in candidates:
        order_hash = hash_json(list(candidate["order"]))
        unique.setdefault(order_hash, candidate)
    effective_k = len(unique)
    if effective_k < 1:
        raise RuntimeError("family has no candidate order")
    multiplier = float(np.sqrt(2.0 * np.log(effective_k))) if effective_k > 1 else 0.0
    trajectory: list[dict[str, Any]] = []
    for order_hash, candidate in unique.items():
        losses = discovery_losses(labels, tune_rows, candidate["order"])
        mean, se = _mean_se(losses)
        bias = multiplier * se
        trajectory.append({
            "grid_index": int(candidate["grid_index"]),
            "hyper": candidate["hyper"],
            "order_sha256": order_hash,
            "mean_discovery_calls": mean,
            "discovery_calls_se": se,
            "bias_multiplier": multiplier,
            "winner_bias_budget_calls": bias,
            "penalized_discovery_calls": mean + bias,
            "query_count": int(len(losses)),
        })
    chosen_row = min(
        trajectory,
        key=lambda row: (
            row["penalized_discovery_calls"],
            row["mean_discovery_calls"],
            row["grid_index"],
        ),
    )
    chosen = next(
        candidate
        for candidate in unique.values()
        if int(candidate["grid_index"]) == int(chosen_row["grid_index"])
    )
    return chosen, {
        "nominal_candidate_count": int(len(candidates)),
        "effective_candidate_count": int(effective_k),
        "bias_multiplier": multiplier,
        "selected_grid_index": int(chosen_row["grid_index"]),
        "selected_hyper": chosen_row["hyper"],
        "selected_order_sha256": chosen_row["order_sha256"],
        "trajectory": trajectory,
    }


def select_family(
    tuned: Mapping[str, Mapping[str, Any]],
    labels: np.ndarray,
    select_rows: np.ndarray,
) -> tuple[str, list[dict[str, Any]]]:
    """Select one scientific family on the validation-select half."""

    if set(tuned) != set(FIXED_METHODS):
        raise RuntimeError("tuned family registry mismatch")
    trajectory: list[dict[str, Any]] = []
    for registry_index, method in enumerate(SCIENTIFIC_METHODS):
        candidate = tuned[method]["candidate"]
        losses = discovery_losses(labels, select_rows, candidate["order"])
        mean, se = _mean_se(losses)
        bias = FAMILY_BIAS_MULTIPLIER * se
        trajectory.append({
            "method": method,
            "ordering_arm": FIXED_METHODS[method],
            "registry_index": int(registry_index),
            "mean_discovery_calls": mean,
            "discovery_calls_se": se,
            "bias_multiplier": FAMILY_BIAS_MULTIPLIER,
            "winner_bias_budget_calls": bias,
            "penalized_discovery_calls": mean + bias,
            "query_count": int(len(losses)),
        })
    chosen = min(
        trajectory,
        key=lambda row: (
            row["penalized_discovery_calls"],
            row["mean_discovery_calls"],
            0 if row["method"] == BASELINE_METHOD else 1,
            row["registry_index"],
        ),
    )
    return str(chosen["method"]), trajectory


def _assert_split_isolation(split: Mapping[str, np.ndarray]) -> None:
    roles = tuple(split)
    if roles != ("design", "router", "final_calibration", "validation", "test"):
        raise RuntimeError("registered split roles changed")
    sets = {name: set(map(int, rows)) for name, rows in split.items()}
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            if sets[left] & sets[right]:
                raise RuntimeError(f"split overlap between {left} and {right}")


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


def run_issue79(
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
    redraws: list[dict[str, Any]] = []

    for redraw in range(redraw_start, redraw_end):
        split = _target_split(len(task.query_ids), redraw, base_seed)
        _assert_split_isolation(split)
        validation_tune, validation_select = split_validation_rows(split["validation"])
        forbidden = set(map(int, split["final_calibration"])) | set(map(int, split["test"]))
        if forbidden & set(map(int, validation_tune)) or forbidden & set(map(int, validation_select)):
            raise RuntimeError("final-calibration or test query entered order selection")
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
        tuned: dict[str, dict[str, Any]] = {}
        for method in FIXED_METHODS:
            candidate, tuning = tune_family(
                fitted[method], labels, validation_tune
            )
            tuned[method] = {"candidate": candidate, "tuning": tuning}
        selected_method, family_trajectory = select_family(
            tuned, labels, validation_select
        )
        selected_candidate = tuned[selected_method]["candidate"]
        tally_candidate = tuned[BASELINE_METHOD]["candidate"]
        error_candidate = tuned[ERROR_CORRELATION_METHOD]["candidate"]
        methods_by_alpha: list[dict[str, Any]] = []
        for alpha in ALPHA_GRID:
            methods: dict[str, dict[str, Any]] = {}
            for output_method, source_method, candidate in (
                ("validation_selected_factorized", selected_method, selected_candidate),
                (BASELINE_METHOD, BASELINE_METHOD, tally_candidate),
                (ERROR_CORRELATION_METHOD, ERROR_CORRELATION_METHOD, error_candidate),
            ):
                result = _evaluate(
                    candidate["scores"],
                    labels,
                    split,
                    task,
                    model_ids,
                    float(alpha),
                    output_method,
                )
                if result["final_calibration_count"] != 1:
                    raise RuntimeError("method did not use exactly one final calibration")
                result.update({
                    "source_order_method": source_method,
                    "ordering_arm": FIXED_METHODS[source_method],
                    "chosen_hyper": candidate["hyper"],
                    "chosen_order": list(candidate["order"]),
                    "chosen_order_sha256": hash_json(list(candidate["order"])),
                    "selection_source": (
                        "validation_tune_then_validation_select"
                        if output_method == "validation_selected_factorized"
                        else "validation_tune_only"
                    ),
                })
                methods[output_method] = result
            if set(methods) != set(OUTPUT_METHODS):
                raise RuntimeError("ISSUE-79 output arm registry mismatch")
            if methods["validation_selected_factorized"]["source_order_method"] != selected_method:
                raise RuntimeError("selected family changed across target levels")
            methods_by_alpha.append({"alpha": float(alpha), "methods": methods})

        split_ids = _split_query_ids(task, split)
        tune_ids = [task.query_ids[int(row)] for row in validation_tune]
        select_ids = [task.query_ids[int(row)] for row in validation_select]
        redraws.append({
            "redraw_id": int(redraw),
            "split_seed": int(split_seed),
            "split_ids": split_ids,
            "validation_partition": {
                "validation_tune_query_ids": tune_ids,
                "validation_select_query_ids": select_ids,
                "validation_tune_sha256": hash_json(tune_ids),
                "validation_select_sha256": hash_json(select_ids),
                "complete_and_disjoint_verified": True,
            },
            "selected_method": selected_method,
            "selected_ordering_arm": FIXED_METHODS[selected_method],
            "selected_hyper": selected_candidate["hyper"],
            "selected_order": list(selected_candidate["order"]),
            "selected_order_sha256": hash_json(list(selected_candidate["order"])),
            "target_free_selection_verified": True,
            "family_bias_multiplier": FAMILY_BIAS_MULTIPLIER,
            "family_menu_size": len(SCIENTIFIC_METHODS),
            "family_selection_trajectory": family_trajectory,
            "within_family_tuning": {
                method: tuned[method]["tuning"] for method in FIXED_METHODS
            },
            "random_controls_selection_eligible": False,
            "alpha_results": methods_by_alpha,
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "design": "ISSUE_79_validation_based_order_selector",
        "issue": 79,
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
        "scientific_methods": list(SCIENTIFIC_METHODS),
        "random_control_methods": list(RANDOM_CONTROL_METHODS),
        "output_methods": list(OUTPUT_METHODS),
        "family_menu_size": len(SCIENTIFIC_METHODS),
        "family_bias_multiplier": FAMILY_BIAS_MULTIPLIER,
        "selection_metric": "first_correct_prefix_rank_pool_plus_one_on_full_miss",
        "selection_target_free": True,
        "preserve_prefixes": True,
        "pool_id": hash_json({"pool_seed": pool_seed, "model_ids": model_ids}),
        "model_ids": model_ids,
        "model_ids_sha256": hash_json(model_ids),
        "sampled_model_column_indices": [int(value) for value in columns],
        "query_ids_sha256": hash_json(list(task.query_ids)),
        "source_hashes": dict(task.source_hashes),
        "redraws": redraws,
    }
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
    payload = run_issue79(
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
