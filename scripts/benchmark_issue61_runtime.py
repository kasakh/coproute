"""Measure one ISSUE-61 runtime phase on archived IFEval labels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.chain_discount import chain_discounted_scores, fit_chain_discount
from components.racer_protocol import nonconformity_scores, select_standard_crc_threshold
from components.racer_wrapper import (
    fixed_order_factorized_scores,
    query_local_empirical_order,
)
from components.scale_selectors import (
    empirical_tally_order,
    empirical_tally_order_lazy,
    marginal_order,
)
from scripts.design_bprime_e2_run import _knn_probabilities
from components.stage1_task_adapters import load_routereval_registered_task

PHASES = (
    "fit_marginal",
    "fit_tally_eager",
    "fit_tally_lazy",
    "fit_chain",
    "router_knn",
    "infer_d1",
    "infer_d2",
    "infer_racer_g",
    "infer_racer_p",
    "infer_query_local_eager",
    "infer_query_local_lazy",
    "calibrate_d1",
    "calibrate_d2",
    "calibrate_racer_g",
    "calibrate_racer_p",
)
COMPLEXITY = {
    "fit_marginal": "O(n_design * M + M log M)",
    "fit_tally_eager": "O(D * n_design * M)",
    "fit_tally_lazy": "O(n_design * M + E_lazy * n_design + E_lazy log M)",
    "fit_chain": "O(n_design * M)",
    "router_knn": "O(n_router * d + n_router * M + n_router log k)",
    "infer_d1": "O(M) per query",
    "infer_d2": "O(M) per query",
    "infer_racer_g": "O(M) per query",
    "infer_racer_p": "O(M) per query",
    "infer_query_local_eager": "O(D * k * M) per query",
    "infer_query_local_lazy": "O(k * M + E_lazy * k + E_lazy log M) per query",
    "calibrate_d1": "O(n_cal * M + n_cal log n_cal)",
    "calibrate_d2": "O(n_cal * M + n_cal log n_cal)",
    "calibrate_racer_g": "O(n_cal * M + n_cal log n_cal)",
    "calibrate_racer_p": "O(n_cal * M + n_cal log n_cal)",
}


def hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def intervals(values: Sequence[float], seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    samples = array[
        generator.integers(0, len(array), size=(2000, len(array)))
    ]
    medians = np.median(samples, axis=1)
    p90s = np.quantile(samples, 0.9, axis=1)
    return {
        "median_seconds": float(np.median(array)),
        "p90_seconds": float(np.quantile(array, 0.9)),
        "median_ci95_low": float(np.quantile(medians, 0.025)),
        "median_ci95_high": float(np.quantile(medians, 0.975)),
        "p90_ci95_low": float(np.quantile(p90s, 0.025)),
        "p90_ci95_high": float(np.quantile(p90s, 0.975)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard", required=True, type=Path)
    parser.add_argument("--router-dir", required=True, type=Path)
    parser.add_argument("--pool-size", required=True, type=int)
    parser.add_argument("--design-size", required=True, type=int)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--host", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--worker-count", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if args.repetitions < 3:
        raise SystemExit("repetitions must be at least 3")

    task = load_routereval_registered_task(
        args.leaderboard,
        args.router_dir,
        "ifeval",
    )
    if args.pool_size > len(task.model_ids):
        raise SystemExit("pool size exceeds archived model count")
    columns = np.random.default_rng(7001).choice(
        len(task.model_ids), size=args.pool_size, replace=False
    )
    labels = np.asarray(task.correctness[:, columns], dtype=np.int8)
    embeddings = np.asarray(task.embeddings, dtype=float)
    permutation = np.random.default_rng(101003).permutation(len(labels))
    if args.design_size > len(labels):
        raise SystemExit("design size exceeds archived query count")
    design_rows = permutation[: args.design_size]
    remaining = permutation[args.design_size :]
    router_count = min(100, max(40, len(remaining) // 3))
    calibration_count = min(100, max(30, (len(remaining) - router_count) // 2))
    query_count = min(64, len(remaining) - router_count - calibration_count)
    if query_count < 1:
        raise SystemExit("not enough rows for router, calibration, and query timing")
    router_rows = remaining[:router_count]
    calibration_rows = remaining[router_count : router_count + calibration_count]
    query_rows = remaining[
        router_count + calibration_count : router_count + calibration_count + query_count
    ]
    design = labels[design_rows]
    eager_order = empirical_tally_order(design, min(25, args.pool_size))
    full_tail = [index for index in np.argsort(-design.mean(axis=0), kind="stable") if index not in eager_order]
    full_order = tuple(eager_order) + tuple(int(index) for index in full_tail)
    probabilities, support = _knn_probabilities(
        embeddings[router_rows],
        labels[router_rows],
        embeddings,
        40,
        return_support=True,
    )
    query_probabilities = probabilities[query_rows]
    query_support = support[query_rows]
    calibration_labels = labels[calibration_rows]
    profile = None
    if args.phase in {"infer_d2", "calibrate_d2"}:
        profile = fit_chain_discount(design, full_order)
    d1_scores = (
        fixed_order_factorized_scores(probabilities, full_order)
        if args.phase == "calibrate_d1"
        else None
    )
    d2_scores = (
        chain_discounted_scores(probabilities, profile, minimum_survivor_count=20)
        if args.phase == "calibrate_d2" and profile is not None
        else None
    )
    racer_g_scores = None
    racer_p_scores = None
    if args.phase == "calibrate_racer_g":
        _, racer_g_scores = nonconformity_scores(
            probabilities, "G", seed=7001, split_seed=7001
        )
    if args.phase == "calibrate_racer_p":
        _, racer_p_scores = nonconformity_scores(
            probabilities, "P", seed=7001, split_seed=7001
        )

    state: dict[str, Any] = {}
    if args.phase == "fit_marginal":
        operation: Callable[[], Any] = lambda: marginal_order(design, min(25, args.pool_size))
        denominator = 1
    elif args.phase == "fit_tally_eager":
        operation = lambda: empirical_tally_order(design, min(25, args.pool_size))
        denominator = 1
    elif args.phase == "fit_tally_lazy":
        def lazy_fit():
            order, diagnostics = empirical_tally_order_lazy(
                design,
                min(25, args.pool_size),
                return_diagnostics=True,
            )
            if order != eager_order:
                raise RuntimeError("lazy greedy order differs from eager tally")
            state["lazy_diagnostics"] = diagnostics
            return order
        operation = lazy_fit
        denominator = 1
    elif args.phase == "fit_chain":
        operation = lambda: fit_chain_discount(design, full_order)
        denominator = 1
    elif args.phase == "router_knn":
        operation = lambda: _knn_probabilities(
            embeddings[router_rows], labels[router_rows], embeddings[query_rows], 40,
            return_support=True,
        )
        denominator = query_count
    elif args.phase == "infer_d1":
        operation = lambda: fixed_order_factorized_scores(query_probabilities, full_order)
        denominator = query_count
    elif args.phase == "infer_d2":
        operation = lambda: chain_discounted_scores(
            query_probabilities, profile, minimum_survivor_count=20
        )
        denominator = query_count
    elif args.phase == "infer_racer_g":
        operation = lambda: nonconformity_scores(
            query_probabilities, "G", seed=7001, split_seed=7001
        )
        denominator = query_count
    elif args.phase == "infer_racer_p":
        operation = lambda: nonconformity_scores(
            query_probabilities, "P", seed=7001, split_seed=7001
        )
        denominator = query_count
    elif args.phase == "infer_query_local_eager":
        operation = lambda: query_local_empirical_order(
            query_support, max_depth=min(25, args.pool_size)
        )
        denominator = query_count
    elif args.phase == "infer_query_local_lazy":
        def local_lazy():
            orders = []
            evaluation_count = 0
            eager = query_local_empirical_order(
                query_support, max_depth=min(25, args.pool_size)
            )
            for row in range(query_count):
                order, diagnostics = empirical_tally_order_lazy(
                    query_support[row],
                    min(25, args.pool_size),
                    return_diagnostics=True,
                )
                orders.append(order)
                evaluation_count += diagnostics["candidate_gain_evaluations"]
            matrix = np.asarray(orders, dtype=int)
            if not np.array_equal(matrix, eager):
                raise RuntimeError("query-local lazy order differs from eager")
            state["query_local_lazy_candidate_gain_evaluations"] = evaluation_count
            return matrix
        operation = local_lazy
        denominator = query_count
    elif args.phase == "calibrate_d1":
        operation = lambda: select_standard_crc_threshold(
            d1_scores[calibration_rows], calibration_labels, 0.10
        )
        denominator = 1
    elif args.phase == "calibrate_d2":
        operation = lambda: select_standard_crc_threshold(
            d2_scores[calibration_rows], calibration_labels, 0.10
        )
        denominator = 1
    elif args.phase == "calibrate_racer_g":
        operation = lambda: select_standard_crc_threshold(
            racer_g_scores[calibration_rows, :-1], calibration_labels, 0.10
        )
        denominator = 1
    elif args.phase == "calibrate_racer_p":
        operation = lambda: select_standard_crc_threshold(
            racer_p_scores[calibration_rows, :-1], calibration_labels, 0.10
        )
        denominator = 1
    else:
        raise AssertionError(args.phase)

    gc.collect()
    start = time.perf_counter_ns()
    cold_result = operation()
    cold_seconds = (time.perf_counter_ns() - start) / 1e9 / denominator
    timings = []
    warm_result = cold_result
    for _ in range(args.repetitions):
        start = time.perf_counter_ns()
        warm_result = operation()
        timings.append((time.perf_counter_ns() - start) / 1e9 / denominator)
    serializable = warm_result
    try:
        serialized_bytes = len(pickle.dumps(serializable, protocol=5))
    except Exception:
        serialized_bytes = None
    summary = intervals(timings, 20260920 + args.pool_size + args.design_size)
    payload = {
        "schema_version": 1,
        "issue": 61,
        "host": args.host,
        "stamp": args.stamp,
        "worker_count": args.worker_count,
        "benchmark": "ifeval",
        "pool_size": args.pool_size,
        "design_size": args.design_size,
        "phase": args.phase,
        "complexity": COMPLEXITY[args.phase],
        "repetitions": args.repetitions,
        "cold_seconds_per_unit": cold_seconds,
        "warm_seconds_per_unit": timings,
        "timing_unit": "per_query" if denominator > 1 else "per_operation",
        "query_batch_size": query_count,
        "router_rows": router_count,
        "calibration_rows": calibration_count,
        "max_order_depth": min(25, args.pool_size),
        "serialized_result_bytes": serialized_bytes,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "thread_caps": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "BLIS_NUM_THREADS": os.environ.get("BLIS_NUM_THREADS"),
        },
        "source_hashes": dict(task.source_hashes),
        "sampled_model_ids_sha256": hash_json([task.model_ids[int(index)] for index in columns]),
        "lazy_order_exact": (
            True if args.phase in {"fit_tally_lazy", "infer_query_local_lazy"} else None
        ),
        **summary,
        **state,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
