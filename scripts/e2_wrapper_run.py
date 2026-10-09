"""E2 runner: dependence-aware scores inside RACER's policy class, on RouterBench.

Venue fixed in ISSUE-31 at 14:04Z: MMLU, HellaSwag, Winogrande and ARC-Challenge
only. Those four are confirmed duplication-free at the loader level. GSM8K is
excluded because its released prompt field is degenerate (7,450 rows, one distinct
prompt) and MBPP because its coverage sits at an infeasible ceiling.

Arms, all inside one policy class so the certificate is identical:
  racer_marginal_G / racer_marginal_P   RACER's own score, the control
  wrapper_pairwise_count               sequential score from counted co-failure
  wrapper_error_correlation            sequential score from failure correlation
  wrapper_gaussian_copula              sequential score from a latent correlation

Every arm shares the SAME design labels, calibration split and untouched test
split within a redraw, and every score is frozen before calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.data_protocol import (  # noqa: E402
    derive_coverage_labels,
    load_routerbench,
    resolve_coverage_policy,
)
from components.racer_protocol import (  # noqa: E402
    augment_router_scores,
    ground_truth_critical_scores,
    misrouting_loss,
    nonconformity_scores,
    prediction_sets,
    select_crc_threshold,
    smooth_nonconformity_scores,
)
from components.racer_wrapper import (  # noqa: E402
    augment_with_racer_null,
    sequential_scores,
)
from components.routers import KNNR, MLPR  # noqa: E402

CLEAN_TASKS = ("mmlu", "hellaswag", "winogrande", "arc-challenge")
WRAPPER_ARMS = ("pairwise_count", "error_correlation", "gaussian_copula")


def _load_router(router_dir: Path, task: str, name: str):
    path = router_dir / f"{task}_{name}.npz"
    return MLPR.load(path, device="cpu") if name == "mlpr" else KNNR.load(path, device="cpu")


def _evaluate(
    *,
    scores: np.ndarray,
    augmented_probabilities: np.ndarray,
    augmented_labels: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict:
    critical = ground_truth_critical_scores(scores[cal_idx], augmented_labels[cal_idx])
    calibration = select_crc_threshold(critical, alpha)
    routing = prediction_sets(
        augmented_probabilities[test_idx], scores[test_idx], calibration.threshold
    )
    loss = misrouting_loss(routing.augmented_mask, augmented_labels[test_idx])
    return {
        "calls_mean": float(np.mean(routing.size)),
        "augmented_set_size_mean": float(np.mean(routing.augmented_mask.sum(axis=1))),
        "risk": float(np.mean(loss)),
        "abstention_rate": float(np.mean(routing.abstain)),
        "threshold": float(calibration.threshold) if np.isfinite(calibration.threshold) else None,
        "feasible": bool(calibration.feasible),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="a completed primary run, for routers and splits")
    parser.add_argument("--routerbench-pkl", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alphas", default="0.05,0.10")
    parser.add_argument("--redraws", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--design-fraction", type=float, default=0.3)
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)

    run_dir = Path(arguments.run_dir)
    fits = json.loads((run_dir / "fits.json").read_text())
    splits = json.loads((run_dir / "split_assignments.json").read_text())
    record = splits["coverage_policy"]
    policy = resolve_coverage_policy(record["name"] if isinstance(record, dict) else record)

    data = load_routerbench(arguments.routerbench_pkl, shot="0shot")
    if data.degenerate_tasks:
        print("loader reports degenerate tasks (excluded by design):", data.degenerate_tasks)
    labels = derive_coverage_labels(data, policy=policy)
    embeddings = np.load(arguments.embeddings)
    row_of = {q: i for i, q in enumerate(data.query_ids)}
    label_row_of = {q: i for i, q in enumerate(labels.query_ids)}
    alphas = [float(a) for a in arguments.alphas.split(",") if a]

    summary = {
        "machine": os.uname().nodename,
        "run_dir": str(run_dir),
        "venue": list(CLEAN_TASKS),
        "excluded": {
            "gsm8k": "degenerate prompt field, 7450 rows with one distinct prompt",
            "mbpp": "coverage at an infeasible ceiling",
        },
        "redraws": arguments.redraws,
        "alphas": alphas,
        "cells": [],
    }

    for task in CLEAN_TASKS:
        if task not in fits["tasks"]:
            print(f"skip {task}: absent from run fits")
            continue
        task_fits = fits["tasks"][task]
        pool_ids = list(task_fits["split_ids"]["calibration"]) + list(task_fits["split_ids"]["test"])
        rows = np.asarray([row_of[q] for q in pool_ids], dtype=int)
        label_rows = np.asarray([label_row_of[q] for q in pool_ids], dtype=int)
        standard = labels.standard[label_rows]
        augmented_labels = labels.augmented[label_rows]

        router_name = task_fits["selected_router"]
        router = _load_router(run_dir / "routers", task, router_name)
        probabilities = np.clip(
            np.asarray(router.predict(embeddings[rows]), dtype=float), 0.0, 1.0
        )
        augmented_probabilities = augment_router_scores(probabilities)
        n_pool = len(pool_ids)
        n_design = int(n_pool * arguments.design_fraction)
        n_cal = (n_pool - n_design) // 2

        for alpha in alphas:
            per_arm: dict[str, list[dict]] = {}
            for redraw in range(arguments.redraws):
                generator = np.random.default_rng(arguments.seed + redraw)
                permutation = generator.permutation(n_pool)
                design_idx = permutation[:n_design]
                cal_idx = permutation[n_design : n_design + n_cal]
                test_idx = permutation[n_design + n_cal :]
                design_labels = standard[design_idx].astype(int)

                for variant in ("G", "P"):
                    _, smoothed = nonconformity_scores(
                        probabilities, variant, seed=arguments.seed, split_seed=redraw + 1
                    )
                    key = f"racer_marginal_{variant}"
                    per_arm.setdefault(key, []).append(
                        _evaluate(
                            scores=smoothed,
                            augmented_probabilities=augmented_probabilities,
                            augmented_labels=augmented_labels,
                            cal_idx=cal_idx,
                            test_idx=test_idx,
                            alpha=alpha,
                        )
                    )

                for arm in WRAPPER_ARMS:
                    raw = sequential_scores(
                        probabilities, dependence=arm, design=design_labels
                    )
                    augmented_raw = augment_with_racer_null(probabilities, raw)
                    smoothed = smooth_nonconformity_scores(
                        augmented_raw, seed=arguments.seed, split_seed=redraw + 1
                    )
                    key = f"wrapper_{arm}"
                    per_arm.setdefault(key, []).append(
                        _evaluate(
                            scores=smoothed,
                            augmented_probabilities=augmented_probabilities,
                            augmented_labels=augmented_labels,
                            cal_idx=cal_idx,
                            test_idx=test_idx,
                            alpha=alpha,
                        )
                    )

            cell = {
                "task": task,
                "alpha": alpha,
                "router": router_name,
                "n_pool": n_pool,
                "n_design": n_design,
                "n_calibration": n_cal,
                "n_test": n_pool - n_design - n_cal,
                "arms": {},
            }
            for key, records in per_arm.items():
                calls = [r["calls_mean"] for r in records]
                risks = [r["risk"] for r in records]
                cell["arms"][key] = {
                    "calls_median": float(np.median(calls)),
                    "calls_p25": float(np.percentile(calls, 25)),
                    "calls_p75": float(np.percentile(calls, 75)),
                    "risk_median": float(np.median(risks)),
                    "risk_max": float(np.max(risks)),
                    "abstention_median": float(np.median([r["abstention_rate"] for r in records])),
                    "all_feasible": bool(all(r["feasible"] for r in records)),
                    "n_redraws": len(records),
                }
            best_control = min(
                cell["arms"][k]["calls_median"] for k in ("racer_marginal_G", "racer_marginal_P")
            )
            wins = {}
            for arm in WRAPPER_ARMS:
                key = f"wrapper_{arm}"
                arm_calls = [r["calls_mean"] for r in per_arm[key]]
                control_calls = [
                    min(g["calls_mean"], p["calls_mean"])
                    for g, p in zip(per_arm["racer_marginal_G"], per_arm["racer_marginal_P"])
                ]
                redraw_wins = sum(1 for a, c in zip(arm_calls, control_calls) if a < c - 1e-12)
                wins[key] = {
                    "median_calls": cell["arms"][key]["calls_median"],
                    "control_median_calls": best_control,
                    "beats_control_on_median": bool(cell["arms"][key]["calls_median"] < best_control),
                    "redraw_wins": redraw_wins,
                    "redraws": len(arm_calls),
                    "risk_within_alpha": bool(cell["arms"][key]["risk_median"] <= alpha),
                }
            cell["comparison_vs_racer_control"] = wins
            summary["cells"].append(cell)
            print(
                f"{task:<14s} alpha={alpha:<5} control={best_control:.3f} "
                + "  ".join(
                    f"{a}={cell['arms']['wrapper_'+a]['calls_median']:.3f}"
                    f"({wins['wrapper_'+a]['redraw_wins']}/{wins['wrapper_'+a]['redraws']})"
                    for a in WRAPPER_ARMS
                ),
                flush=True,
            )

    total = len(summary["cells"])
    decisive = {
        arm: sum(
            1
            for c in summary["cells"]
            if c["comparison_vs_racer_control"][f"wrapper_{arm}"]["beats_control_on_median"]
            and c["comparison_vs_racer_control"][f"wrapper_{arm}"]["risk_within_alpha"]
        )
        for arm in WRAPPER_ARMS
    }
    summary["verdict"] = {
        "cells": total,
        "cells_won_with_risk_within_alpha_by_arm": decisive,
        "rule": (
            "E2 is claimable only if an arm beats the RACER control on median calls "
            "with risk within alpha in at least 3 of 4 tasks at BOTH alphas, "
            "including hellaswag"
        ),
    }
    print()
    print("VERDICT INPUT:", json.dumps(summary["verdict"], indent=1))
    (out_dir / "e2_results.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "DONE").write_text("")
    print("WROTE", out_dir / "e2_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
