"""E1-P1: RouterEval headroom census and abort gate.

Question the gate answers: at each pool size, does the COMPLETE pool ever fail?

If the complete pool's all-miss rate exceeds the target alpha, that cell is
structurally infeasible and no estimator can create headroom, exactly as
RouterBench pool 3 was at 0.125. If the complete pool essentially never fails,
there is room for small-subset selection and the E1 grid is worth running.

Labels only. No features, no router, no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.routereval_protocol import (  # noqa: E402
    headroom_census,
    load_correctness_only,
    subsample_pool,
)

# Fixed before any result is read. Four from the plan plus two chosen for
# contrasting difficulty: bbh_boolean_expressions is among the easiest BBH
# subtasks and math_intermediate_algebra_hard among the hardest MATH ones.
DEFAULT_BENCHMARKS = (
    "gpqa_diamond",
    "math_algebra_hard",
    "ifeval",
    "bbh_navigate",
    "bbh_boolean_expressions",
    "math_intermediate_algebra_hard",
)
DEFAULT_POOLS = (11, 50, 500, 3811)
DEFAULT_ALPHAS = (0.05, 0.10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--pools", default=",".join(str(p) for p in DEFAULT_POOLS))
    parser.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--threshold", type=float, default=0.5)
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)

    benchmarks = [b for b in arguments.benchmarks.split(",") if b]
    pools = [int(p) for p in arguments.pools.split(",") if p]
    alphas = [float(a) for a in arguments.alphas.split(",") if a]
    seeds = [int(s) for s in arguments.seeds.split(",") if s]

    summary = {
        "machine": os.uname().nodename,
        "leaderboard": str(arguments.leaderboard),
        "threshold": arguments.threshold,
        "benchmarks": benchmarks,
        "pools": pools,
        "alphas": alphas,
        "seeds": seeds,
        "mode": "correctness_only, no features, no router",
        "census": {},
        "abort_gate": {},
    }

    print(
        "%-32s %6s %8s %10s %10s %10s"
        % ("benchmark", "pool", "queries", "all_miss", "best_model", "mean_model")
    )
    for benchmark in benchmarks:
        data = load_correctness_only(
            leaderboard_path=arguments.leaderboard, leaderboard_key=benchmark
        )
        per_seed = []
        for seed in seeds:
            per_seed.append(
                headroom_census(data, pools, seed=seed, threshold=arguments.threshold)
            )
        # Aggregate across design redraws, never a point value.
        aggregated = []
        for position, pool_size in enumerate(pools):
            rates = [c["records"][position]["complete_pool_all_miss_rate"] for c in per_seed]
            best = [c["records"][position]["best_single_model_accuracy"] for c in per_seed]
            mean_model = [c["records"][position]["mean_model_accuracy"] for c in per_seed]
            entry = {
                "pool_size": int(pool_size),
                "all_miss_median": float(np.median(rates)),
                "all_miss_min": float(np.min(rates)),
                "all_miss_max": float(np.max(rates)),
                "best_single_model_median": float(np.median(best)),
                "mean_model_accuracy_median": float(np.median(mean_model)),
                "feasible_at_alpha": {
                    str(alpha): bool(np.max(rates) <= alpha) for alpha in alphas
                },
            }
            aggregated.append(entry)
            print(
                "%-32s %6d %8d %10.5f %10.4f %10.4f"
                % (
                    benchmark,
                    pool_size,
                    data.n_queries,
                    entry["all_miss_median"],
                    entry["best_single_model_median"],
                    entry["mean_model_accuracy_median"],
                )
            )
        summary["census"][benchmark] = {
            "n_queries": data.n_queries,
            "n_models_available": data.n_models,
            "source_sha256": data.source_sha256,
            "by_pool": aggregated,
        }

    # Abort gate: a benchmark contributes usable cells only where the complete
    # pool is feasible at the target level, at EVERY seed.
    usable = {}
    for benchmark, record in summary["census"].items():
        cells = []
        for entry in record["by_pool"]:
            for alpha in alphas:
                if entry["feasible_at_alpha"][str(alpha)]:
                    cells.append({"pool_size": entry["pool_size"], "alpha": alpha})
        usable[benchmark] = cells
    total_cells = sum(len(v) for v in usable.values())
    summary["abort_gate"] = {
        "usable_cells_by_benchmark": {k: len(v) for k, v in usable.items()},
        "usable_cells_total": total_cells,
        "benchmarks_with_zero_usable_cells": [k for k, v in usable.items() if not v],
        "verdict": "PROCEED" if total_cells > 0 else "ABORT_NO_HEADROOM",
        "rule": (
            "PROCEED requires at least one benchmark, pool and alpha where the "
            "complete pool's all-miss rate is at or below alpha at every seed. "
            "Cells failing that are structurally infeasible and no estimator can "
            "create headroom in them."
        ),
    }
    print()
    print("ABORT GATE:", summary["abort_gate"]["verdict"])
    print("usable cells:", summary["abort_gate"]["usable_cells_by_benchmark"])
    if summary["abort_gate"]["benchmarks_with_zero_usable_cells"]:
        print(
            "no headroom at any pool/alpha:",
            summary["abort_gate"]["benchmarks_with_zero_usable_cells"],
        )

    (out_dir / "headroom_census.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "DONE").write_text("")
    print("WROTE", out_dir / "headroom_census.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
