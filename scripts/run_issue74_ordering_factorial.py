"""Run ISSUE-74: every fixed E1 order with identical per-query product sizing."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.data_protocol import PRIMARY_COVERAGE_POLICY, derive_coverage_labels, load_routerbench
from components.racer_protocol import (
    augment_ground_truth,
    augment_router_scores,
    select_standard_crc_threshold,
    smooth_nonconformity_scores,
)
from components.racer_wrapper import (
    FACTORIZED_INDEPENDENCE,
    augment_with_racer_null,
    fixed_design_marginal_factorized_scores,
    fixed_order_factorized_scores,
    sequential_scores,
)
from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS, _ensure_pandas_pickle_compat
from components.scale_selectors import HYPERPARAMETER_GRIDS, HYPERPARAMETER_NAMES, build_order
from scripts.build_routerbench_embedding_archive import sha256_file
from scripts.design_bprime_e2_routerbench_run import load_aligned_embedding_archive, partition_task_rows
from scripts.design_bprime_e2_run import (
    _final_calibrate_and_test,
    _knn_probabilities,
    _select_validation_candidate,
    _temporary_validation_record,
)
from scripts.run_paired_main_suite import (
    ALPHA_GRID,
    BASE_SEED,
    SPLIT_STEP,
    _load_task as _paired_load_task,
    _split_query_ids,
    _target_split,
    atomic_write_json,
    evaluate_score_family,
    hash_json,
)

SCHEMA_VERSION = "issue74_ordering_factorial_v1"
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
REGISTERED_METHODS = tuple(FIXED_METHODS) + ("exact_independence",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_routereval_task(leaderboard: Path, router_dir: Path, benchmark: str) -> Any:
    try:
        return _paired_load_task(
            "routereval",
            benchmark,
            leaderboard=leaderboard,
            router_dir=router_dir,
        )
    except TypeError:
        return _paired_load_task(
            leaderboard,
            router_dir / f"{benchmark}_router_dataset.pkl",
            benchmark,
        )


def _routerbench_task(
    pickle_path: Path,
    embedding_archive: Path,
    embedding_manifest: Path,
    benchmark: str,
) -> SimpleNamespace:
    _ensure_pandas_pickle_compat()
    data = load_routerbench(
        pickle_path,
        shot="0shot",
        model_columns=ROUTERBENCH_MODEL_COLUMNS,
    )
    labels = derive_coverage_labels(data, policy=PRIMARY_COVERAGE_POLICY).standard.astype(np.int8)
    embeddings, embedding_audit = load_aligned_embedding_archive(
        embedding_archive,
        data.query_ids,
        manifest_path=embedding_manifest,
    )
    task_rows = np.flatnonzero(data.frame["task"].astype(str).to_numpy() == benchmark)
    if not len(task_rows):
        raise ValueError(f"RouterBench task {benchmark!r} has no rows")
    prompt_audit = {
        str(row["task"]): dict(row) for row in data.prompt_degeneracy
    }.get(benchmark)
    if prompt_audit is None or float(prompt_audit["duplicate_rate"]) > 0.10:
        raise ValueError(f"RouterBench task {benchmark!r} lacks clean prompt features")
    return SimpleNamespace(
        task=benchmark,
        query_ids=tuple(data.query_ids[int(row)] for row in task_rows),
        embeddings=np.asarray(embeddings[task_rows], dtype=float),
        correctness=np.asarray(labels[task_rows], dtype=np.int8),
        model_ids=tuple(data.model_columns),
        source_hashes={
            "routerbench_pickle": sha256_file(pickle_path),
            "embedding_archive": sha256_file(embedding_archive),
            "embedding_manifest": sha256_file(embedding_manifest),
        },
        feature_audit={**prompt_audit, **embedding_audit},
    )


def _split(task: Any, protocol: str, redraw: int, base_seed: int) -> dict[str, np.ndarray]:
    if protocol == "routereval":
        return _target_split(len(task.query_ids), redraw, base_seed)
    return partition_task_rows(len(task.query_ids), seed=base_seed, redraw=redraw)


def _smooth(
    raw: np.ndarray,
    *,
    seed: int,
    split_seed: int,
    preserve_score_prefixes: bool,
    fixed_order: Sequence[int] | None = None,
) -> np.ndarray:
    values = np.asarray(raw)
    if preserve_score_prefixes and fixed_order is not None:
        order = [int(index) for index in fixed_order]
        if values.shape[1] == len(order) + 1:
            order.append(len(order))
        sequence = np.asarray(order, dtype=int)
        ordered = values[:, sequence]
        smoothed = smooth_nonconformity_scores(
            ordered,
            seed=seed,
            split_seed=split_seed,
            preserve_prefixes=True,
        )
        result = np.empty_like(smoothed)
        result[:, sequence] = smoothed
        return result
    return smooth_nonconformity_scores(
        values,
        seed=seed,
        split_seed=split_seed,
        preserve_prefixes=preserve_score_prefixes,
    )


def build_score_candidates(
    probabilities: np.ndarray,
    design_labels: np.ndarray,
    model_ids: Sequence[str],
    *,
    protocol: str,
    base_seed: int,
    redraw: int,
    preserve_score_prefixes: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Fit every ordering candidate using design labels only."""

    n_models = probabilities.shape[1]
    split_seed = base_seed + SPLIT_STEP * redraw if protocol == "routereval" else redraw + 1
    candidate_seed = base_seed + SPLIT_STEP * redraw
    fitted: dict[str, list[dict[str, Any]]] = {}
    for method, arm in FIXED_METHODS.items():
        candidates = []
        for grid_index, hyper in enumerate(HYPERPARAMETER_GRIDS[arm]):
            order = build_order(
                arm,
                design_labels,
                model_names=list(model_ids),
                max_depth=n_models,
                seed=candidate_seed,
                n_draws=2000,
                hyper=hyper,
            )
            if len(order) != n_models or len(set(order)) != n_models:
                raise RuntimeError(f"{method} did not return a complete permutation")
            if method == "fixed_design_marginal_factorized":
                raw = fixed_design_marginal_factorized_scores(
                    probabilities, design_labels, max_depth=n_models
                )
            else:
                raw = fixed_order_factorized_scores(
                    probabilities, order, max_depth=n_models
                )
            if protocol == "routerbench":
                raw = augment_with_racer_null(probabilities, raw)
            scores = _smooth(
                raw,
                seed=base_seed,
                split_seed=split_seed,
                preserve_score_prefixes=preserve_score_prefixes,
                fixed_order=order,
            )
            candidates.append({
                "grid_index": grid_index,
                "hyper": hyper,
                "order": [int(index) for index in order],
                "scores": scores,
            })
        fitted[method] = candidates

    exact = sequential_scores(
        probabilities,
        dependence=FACTORIZED_INDEPENDENCE,
        max_depth=n_models,
    )
    if protocol == "routerbench":
        exact = augment_with_racer_null(probabilities, exact)
    fitted["exact_independence"] = [{
        "grid_index": 0,
        "hyper": None,
        "order": None,
        "scores": _smooth(
            exact,
            seed=base_seed,
            split_seed=split_seed,
            preserve_score_prefixes=preserve_score_prefixes,
        ),
    }]
    if set(fitted) != set(REGISTERED_METHODS):
        raise RuntimeError("registered factorial methods and fitted methods differ")
    return fitted


def _fallback_standard(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(scores, dtype=float)
    outcomes = np.asarray(labels, dtype=bool)
    if values.shape != outcomes.shape or values.ndim != 2:
        raise ValueError("fallback scores and labels must have one shared matrix shape")
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
        chosen = sequence[start:end]
        calls_total += end - start
        correct = chosen[flat_correct[chosen]]
        if len(correct):
            successful[np.unique(correct // n_models)] = True
        risk = float(np.mean(~successful))
        calls = calls_total / n_queries
        best = min(best, (risk, calls, threshold))
        start = end
    return best


def _select_routereval_candidate(
    candidates: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    validation_rows: np.ndarray,
    alpha: float,
) -> tuple[Mapping[str, Any], list[dict[str, Any]], bool]:
    trajectory = []
    for candidate in candidates:
        scores = np.asarray(candidate["scores"])
        result = select_standard_crc_threshold(
            scores[validation_rows], labels[validation_rows], alpha
        )
        if result.feasible:
            threshold = float(result.threshold)
            mask = scores[validation_rows] <= threshold
            calls = float(mask.sum(axis=1).mean())
            risk = float(np.mean(~np.any(mask & labels[validation_rows].astype(bool), axis=1)))
        else:
            risk, calls, threshold = _fallback_standard(
                scores[validation_rows], labels[validation_rows]
            )
        trajectory.append({
            "grid_index": int(candidate["grid_index"]),
            "hyper": candidate["hyper"],
            "validation_certified": bool(result.feasible),
            "validation_threshold": threshold,
            "validation_calls": calls,
            "validation_standard_all_miss_risk": risk,
            "validation_non_certification_reason": result.non_certification_reason,
        })
    certified = [row for row in trajectory if row["validation_certified"]]
    pool = certified if certified else trajectory
    chosen_row = min(
        pool,
        key=lambda row: (
            row["validation_calls"] if certified else row["validation_standard_all_miss_risk"],
            row["validation_standard_all_miss_risk"] if certified else row["validation_calls"],
            row["grid_index"],
        ),
    )
    chosen = candidates[chosen_row["grid_index"]]
    return chosen, trajectory, not bool(certified)


def _evaluate_routereval(
    method: str,
    candidates: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    alpha: float,
) -> dict[str, Any]:
    chosen, trajectory, fallback = _select_routereval_candidate(
        candidates, labels, split["validation"], alpha
    )
    result = evaluate_score_family(
        np.asarray(chosen["scores"]),
        labels,
        split["final_calibration"],
        split["test"],
        query_ids,
        model_ids,
        alpha,
        method_family=method,
    )
    result.update({
        "selected_arm": method,
        "chosen_hyper": chosen["hyper"],
        "chosen_order": chosen["order"],
        "selection_source": "validation_only",
        "selection_fallback_used": fallback,
        "validation_trajectory": trajectory,
    })
    return result


def _evaluate_routerbench(
    method: str,
    candidates: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    alpha: float,
) -> dict[str, Any]:
    evaluation_rows = np.concatenate((
        split["final_calibration"], split["validation"], split["test"]
    ))
    n_cal = len(split["final_calibration"])
    n_val = len(split["validation"])
    cal_idx = np.arange(n_cal)
    val_idx = np.arange(n_cal, n_cal + n_val)
    test_idx = np.arange(n_cal + n_val, len(evaluation_rows))
    standard_labels = labels[evaluation_rows].astype(bool)
    if probabilities.shape[0] != len(evaluation_rows):
        raise ValueError("RouterBench probabilities must cover the evaluation pool only")
    standard_candidates = [
        {**candidate, "scores": np.asarray(candidate["scores"])[:, :-1]}
        for candidate in candidates
    ]
    chosen, trajectory, fallback = _select_routereval_candidate(
        standard_candidates, standard_labels, val_idx, alpha
    )
    scores = np.asarray(chosen["scores"])
    final = select_standard_crc_threshold(scores[cal_idx], standard_labels[cal_idx], alpha)
    result: dict[str, Any] = {
        "primary_risk_name": "standard_all_miss_risk",
        "final_threshold": float(final.threshold) if final.feasible else None,
        "certified": bool(final.feasible),
        "non_certification": not bool(final.feasible),
        "non_certification_reason": final.non_certification_reason,
        "risk_validity_claimed": bool(final.feasible),
        "final_calibration_count": 1,
        "n_final_calibration": int(len(cal_idx)),
        "calibration_misses": int(final.calibration_misses),
        "unavoidable_calibration_misses": int(final.unavoidable_misses),
        "crc_bound": float(final.crc_bound),
        "selected_arm": method,
        "chosen_hyper": chosen["hyper"],
        "chosen_order": chosen["order"],
        "selection_source": "validation_only",
        "selection_fallback_used": fallback,
        "validation_trajectory": trajectory,
    }
    if not final.feasible:
        result.update({
            "calls": None,
            "calls_test": None,
            "test_calls": None,
            "risk_test": None,
            "standard_all_miss_risk": None,
            "test_calls_by_query": None,
            "test_miss_by_query": None,
        })
        return result
    selected = scores[test_idx] <= float(final.threshold)
    calls = selected.sum(axis=1, dtype=int)
    misses = (~np.any(selected & standard_labels[test_idx], axis=1)).astype(int)
    result.update({
        "calls": float(np.mean(calls)),
        "calls_test": float(np.mean(calls)),
        "test_calls": float(np.mean(calls)),
        "risk_test": float(np.mean(misses)),
        "standard_all_miss_risk": float(np.mean(misses)),
        "test_calls_by_query": [int(value) for value in calls],
        "test_miss_by_query": [int(value) for value in misses],
    })
    return result


def run_factorial(
    task: Any,
    *,
    protocol: str,
    pool_size: int,
    pool_seed: int,
    redraw_start: int,
    redraw_end: int,
    knn_k: int,
    base_seed: int = BASE_SEED,
    preserve_score_prefixes: bool = True,
) -> dict[str, Any]:
    if redraw_start < 0 or redraw_end <= redraw_start:
        raise ValueError("redraw range must be non-empty and non-negative")
    if protocol == "routereval":
        from components.routereval_protocol import subsample_pool
        columns = subsample_pool(len(task.model_ids), pool_size, seed=pool_seed)
        model_ids = [task.model_ids[int(index)] for index in columns]
        labels = np.asarray(task.correctness[:, columns], dtype=np.int8)
    else:
        if pool_size != len(task.model_ids):
            raise ValueError("RouterBench factorial requires its complete 11-model pool")
        columns = np.arange(pool_size)
        model_ids = list(task.model_ids)
        labels = np.asarray(task.correctness, dtype=np.int8)
    cells = [{"alpha": float(alpha), "redraws": []} for alpha in ALPHA_GRID]
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
        candidates = build_score_candidates(
            probabilities,
            labels[split["design"]],
            model_ids,
            protocol=protocol,
            base_seed=base_seed,
            redraw=redraw,
            preserve_score_prefixes=preserve_score_prefixes,
        )
        split_ids = {
            name: [task.query_ids[int(index)] for index in rows]
            for name, rows in split.items()
        }
        for cell in cells:
            alpha = float(cell["alpha"])
            methods = {}
            for method in REGISTERED_METHODS:
                if protocol == "routereval":
                    methods[method] = _evaluate_routereval(
                        method, candidates[method], labels, split,
                        task.query_ids, model_ids, alpha,
                    )
                else:
                    methods[method] = _evaluate_routerbench(
                        method, candidates[method], probabilities, labels, split, alpha
                    )
            if set(methods) != set(REGISTERED_METHODS):
                raise RuntimeError("registered method produced no output row")
            cell["redraws"].append({
                "redraw_id": redraw,
                "split_seed": base_seed + SPLIT_STEP * redraw,
                "split_ids": split_ids,
                "methods": methods,
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": "ISSUE-74",
        "design": "B_prime_full_ordering_by_product_sizing",
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
        "preserve_score_prefixes": bool(preserve_score_prefixes),
        "registered_methods": list(REGISTERED_METHODS),
        "fixed_method_order_arms": dict(FIXED_METHODS),
        "hyperparameter_grids": {
            arm: list(HYPERPARAMETER_GRIDS[arm]) for arm in FIXED_METHODS.values()
        },
        "hyperparameter_names": {
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
    parser.add_argument("--leaderboard", type=Path)
    parser.add_argument("--router-dir", type=Path)
    parser.add_argument("--routerbench-pkl", type=Path)
    parser.add_argument("--embedding-archive", type=Path)
    parser.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--preserve-score-prefixes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite immutable result {args.out}")
    if args.protocol == "routereval":
        if args.leaderboard is None or args.router_dir is None:
            parser.error("RouterEval requires --leaderboard and --router-dir")
        task = _load_routereval_task(
            args.leaderboard,
            args.router_dir,
            args.benchmark,
        )
    else:
        required = (args.routerbench_pkl, args.embedding_archive, args.embedding_manifest)
        if any(value is None for value in required):
            parser.error("RouterBench requires its pickle, embedding archive, and manifest")
        task = _routerbench_task(
            args.routerbench_pkl,
            args.embedding_archive,
            args.embedding_manifest,
            args.benchmark,
        )
    payload = run_factorial(
        task,
        protocol=args.protocol,
        pool_size=args.pool_size,
        pool_seed=args.pool_seed,
        redraw_start=args.redraw_start,
        redraw_end=args.redraw_end,
        knn_k=args.knn_k,
        base_seed=args.base_seed,
        preserve_score_prefixes=args.preserve_score_prefixes,
    )
    atomic_write_json(args.out, payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
