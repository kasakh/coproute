"""E1-P2: the RouterEval scale grid over the estimator-family taxonomy.

One cell is (benchmark, pool size, alpha, design budget, design seed). Within a
cell every arm sees the SAME design labels, the SAME calibration split and the
SAME untouched test split, and every arm's ordering is frozen before calibration,
so all arms carry the identical conformal guarantee and differ only in how they
estimate the joint failure structure.

Depth calibration is RACER's CRC inequality on the calibration split:
    choose the smallest depth d with (misses_d + 1) / (n_cal + 1) <= alpha
where misses_d counts calibration queries on which every one of the first d
models fails. Endpoint is the calibrated depth, which IS the call count for a
fixed-depth policy, plus the realized all-miss rate on test.

Labels only. No features, no router, no GPU. Results are reported per seed so the
caller can present a distribution over design redraws rather than a point value.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.routereval_protocol import (  # noqa: E402
    load_correctness_only,
    subsample_pool,
)
from components.scale_selectors import ARMS, build_order  # noqa: E402


def calibrate_depth(chain_failures: np.ndarray, alpha: float) -> tuple[int, bool, float]:
    """Smallest depth whose CRC bound is at or below alpha.

    `chain_failures[i, d]` is True when every model in the first d+1 positions
    fails calibration query i. Returns depth, feasibility, and the bound.
    """

    n_cal, max_depth = chain_failures.shape
    for depth in range(max_depth):
        misses = int(chain_failures[:, depth].sum())
        bound = (misses + 1.0) / (n_cal + 1.0)
        if bound <= alpha:
            return depth + 1, True, bound
    misses = int(chain_failures[:, max_depth - 1].sum())
    return max_depth, False, (misses + 1.0) / (n_cal + 1.0)


def _prefix_failures(labels: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    """Cumulative all-miss indicator along a nested chain."""

    failures = ~labels[:, list(order)].astype(bool)
    return np.logical_and.accumulate(failures, axis=1)


def run_cell(task: dict) -> dict:
    labels = np.load(task["labels_path"], mmap_mode="r")
    binary = np.asarray(labels[:, task["columns"]]).astype(bool)
    n_queries = binary.shape[0]

    generator = np.random.default_rng(task["seed"] + 100_003)
    permutation = generator.permutation(n_queries)
    budget = int(task["design_budget"])
    remaining = n_queries - budget
    n_cal = remaining // 2
    design_idx = permutation[:budget]
    cal_idx = permutation[budget : budget + n_cal]
    test_idx = permutation[budget + n_cal :]

    design = binary[design_idx].astype(int)
    calibration = binary[cal_idx]
    test = binary[test_idx]

    result = {
        "benchmark": task["benchmark"],
        "pool_size": int(task["pool_size"]),
        "alpha": float(task["alpha"]),
        "design_budget": budget,
        "seed": int(task["seed"]),
        "n_design": int(len(design_idx)),
        "n_calibration": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "arms": {},
    }
    max_depth = int(task["max_depth"])
    for arm in ARMS:
        started = time.time()
        order = build_order(
            arm,
            design,
            model_names=task["model_names"],
            max_depth=max_depth,
            seed=task["seed"],
            n_draws=int(task["n_draws"]),
        )
        cal_prefix = _prefix_failures(calibration, order)
        depth, feasible, bound = calibrate_depth(cal_prefix, task["alpha"])
        test_prefix = _prefix_failures(test, order)
        realized = float(test_prefix[:, depth - 1].mean())
        result["arms"][arm] = {
            "calls": int(depth),
            "feasible": bool(feasible),
            "calibration_bound": float(bound),
            "realized_all_miss": realized,
            "seconds": round(time.time() - started, 3),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--census", required=True, help="E1-P1 headroom_census.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--budgets", default="25,50,100,250")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--max-depth", type=int, default=25)
    parser.add_argument("--n-draws", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    scratch = out_dir / "labels"
    scratch.mkdir()

    census = json.loads(Path(arguments.census).read_text())
    budgets = [int(b) for b in arguments.budgets.split(",") if b]
    seeds = [int(s) for s in arguments.seeds.split(",") if s]

    tasks: list[dict] = []
    skipped: list[dict] = []
    for benchmark, record in census["census"].items():
        data = load_correctness_only(
            leaderboard_path=arguments.leaderboard, leaderboard_key=benchmark
        )
        binary = (data.correctness >= arguments.threshold).astype(np.int8)
        labels_path = scratch / f"{benchmark}.npy"
        np.save(labels_path, binary)
        for entry in record["by_pool"]:
            pool_size = int(entry["pool_size"])
            for alpha_key, feasible in entry["feasible_at_alpha"].items():
                if not feasible:
                    skipped.append(
                        {
                            "benchmark": benchmark,
                            "pool_size": pool_size,
                            "alpha": float(alpha_key),
                            "reason": "complete pool infeasible at this alpha",
                        }
                    )
                    continue
                for budget in budgets:
                    # Need a calibration and a test split after the design budget.
                    if data.n_queries - budget < 20:
                        skipped.append(
                            {
                                "benchmark": benchmark,
                                "pool_size": pool_size,
                                "alpha": float(alpha_key),
                                "design_budget": budget,
                                "reason": "too few queries left for calibration and test",
                            }
                        )
                        continue
                    for seed in seeds:
                        columns = subsample_pool(data.n_models, pool_size, seed=seed)
                        tasks.append(
                            {
                                "benchmark": benchmark,
                                "labels_path": str(labels_path),
                                "columns": columns.tolist(),
                                "model_names": [data.model_names[c] for c in columns],
                                "pool_size": pool_size,
                                "alpha": float(alpha_key),
                                "design_budget": budget,
                                "seed": seed,
                                "max_depth": arguments.max_depth,
                                "n_draws": arguments.n_draws,
                            }
                        )

    print(f"cells to run: {len(tasks)}   skipped: {len(skipped)}", flush=True)
    (out_dir / "plan.json").write_text(
        json.dumps(
            {
                "machine": os.uname().nodename,
                "n_cells": len(tasks),
                "skipped": skipped,
                "budgets": budgets,
                "seeds": seeds,
                "arms": list(ARMS),
                "max_depth": arguments.max_depth,
                "n_draws": arguments.n_draws,
                "threshold": arguments.threshold,
                "census": str(arguments.census),
            },
            indent=1,
        )
    )

    results: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {pool.submit(run_cell, task): index for index, task in enumerate(tasks)}
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % 25 == 0 or done == len(tasks):
                elapsed = time.time() - started
                rate = elapsed / done
                print(
                    f"  {done}/{len(tasks)} cells  elapsed {elapsed/60:.1f}m  "
                    f"eta {(len(tasks)-done)*rate/60:.1f}m",
                    flush=True,
                )

    results.sort(
        key=lambda r: (r["benchmark"], r["pool_size"], r["alpha"], r["design_budget"], r["seed"])
    )
    (out_dir / "grid_results.json").write_text(json.dumps(results, indent=1))
    (out_dir / "DONE").write_text("")
    print("WROTE", out_dir / "grid_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
