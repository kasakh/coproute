"""E2 per-query wrapper frontier for RouterBench: generalization gap plot.

For each of the three dependence-aware wrapper variants, draws per-depth risk
curves on marginal baseline vs the tuned wrapper, on design and test splits.

RouterBench setup: uses the primary run's fitted routers (KNNR or MLPR trained
on external benchmarks per RACER protocol), precomputed mDeBERTa embeddings,
and RouterBench correctness matrix. Four clean tasks: mmlu, hellaswag,
winogrande, arc-challenge (gsm8k and mbpp excluded per data_protocol).

Splits: reuse the primary run's cal + test pool; sub-split into
design (0.30) / val (0.25) / test (0.45) for the per-query wrapper analysis.
Fit dependence on design; val-select hyper; evaluate per-depth on design + test.
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.data_protocol import (
    load_routerbench, derive_coverage_labels, resolve_coverage_policy,
)
from components.racer_wrapper import sequential_scores
from components.routers import KNNR, MLPR

DEPTHS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
WRAPPER_ARMS = ("pairwise_count", "error_correlation", "gaussian_copula")
WRAPPER_HYPER_GRIDS = {
    "pairwise_count":     [{"laplace_smoothing": v} for v in (0.0, 0.5, 1.0, 2.0)],
    "error_correlation":  [{"clip_magnitude": v}    for v in (0.90, 0.95, 0.99, 0.999)],
    "gaussian_copula":    [{"ridge": v}             for v in (0.0, 0.01, 0.05, 0.10, 0.20)],
}
CLEAN_TASKS = ("mmlu", "hellaswag", "winogrande", "arc-challenge")


def _load_router(router_dir: Path, task: str, name: str):
    path = router_dir / f"{task}_{name}.npz"
    return MLPR.load(path, device="cpu") if name == "mlpr" else KNNR.load(path, device="cpu")


def depth_risk(order_matrix: np.ndarray, correct: np.ndarray, depth: int) -> float:
    n_q = order_matrix.shape[0]
    top_k = order_matrix[:, :depth]
    fails = np.zeros(n_q, dtype=bool)
    for q in range(n_q):
        fails[q] = not np.any(correct[q, top_k[q]] == 1)
    return float(fails.mean())


def compute_task(task, run_dir, routerbench_pkl, embeddings, seeds, shot):
    fits = json.loads((run_dir / "fits.json").read_text())
    splits = json.loads((run_dir / "split_assignments.json").read_text())
    record = splits["coverage_policy"]
    policy = resolve_coverage_policy(record["name"] if isinstance(record, dict) else record)
    data = load_routerbench(routerbench_pkl, shot=shot)
    labels = derive_coverage_labels(data, policy=policy)
    row_of = {q: i for i, q in enumerate(data.query_ids)}
    label_row_of = {q: i for i, q in enumerate(labels.query_ids)}

    task_fits = fits["tasks"][task]
    pool_ids = list(task_fits["split_ids"]["calibration"]) + list(task_fits["split_ids"]["test"])
    rows = np.asarray([row_of[q] for q in pool_ids], dtype=int)
    label_rows = np.asarray([label_row_of[q] for q in pool_ids], dtype=int)
    standard = labels.standard[label_rows]  # correctness matrix, shape (n_q, n_models)
    router_name = task_fits["selected_router"]
    router = _load_router(run_dir / "routers", task, router_name)
    # Predict router probabilities on the whole pool
    probs = np.clip(np.asarray(router.predict(embeddings[rows]), dtype=float), 0.0, 1.0)

    n_q = standard.shape[0]
    result = {}
    result["marginal"] = {"test": {d: [] for d in DEPTHS}, "design": {d: [] for d in DEPTHS}, "val": {d: [] for d in DEPTHS}}
    for arm in WRAPPER_ARMS:
        result[arm] = {"test": {d: [] for d in DEPTHS}, "design": {d: [] for d in DEPTHS}, "val": {d: [] for d in DEPTHS}}
    hyper_pick_freq = {arm: {} for arm in WRAPPER_ARMS}

    for seed in seeds:
        rng = np.random.default_rng(seed + 100_003)
        perm = rng.permutation(n_q)
        n_des = int(n_q * 0.30)
        n_val = int(n_q * 0.25)
        des_idx = perm[:n_des]
        val_idx = perm[n_des:n_des+n_val]
        tst_idx = perm[n_des+n_val:]

        design_labels = standard[des_idx].astype(bool)
        val_labels    = standard[val_idx].astype(bool)
        test_labels   = standard[tst_idx].astype(bool)

        des_probs  = probs[des_idx]
        val_probs  = probs[val_idx]
        test_probs = probs[tst_idx]

        # Marginal baseline: top-k by router prob descending
        marg_order_test = np.argsort(-test_probs, axis=1)
        marg_order_des  = np.argsort(-des_probs,  axis=1)
        marg_order_val = np.argsort(-val_probs, axis=1)
        for d in DEPTHS:
            result["marginal"]["test"][d].append(depth_risk(marg_order_test, test_labels.astype(int), d))
            result["marginal"]["design"][d].append(depth_risk(marg_order_des,  design_labels.astype(int), d))
            result["marginal"]["val"][d].append(depth_risk(marg_order_val,   val_labels.astype(int), d))

        pool_full_depth = probs.shape[1]
        for arm in WRAPPER_ARMS:
            best_hyper = None; best_val_risk = float("inf")
            for hyper in WRAPPER_HYPER_GRIDS[arm]:
                val_scores = sequential_scores(val_probs, dependence=arm, design=design_labels, hyper=hyper, max_depth=pool_full_depth)
                val_order = np.argsort(val_scores, axis=1)
                risks = [depth_risk(val_order, val_labels.astype(int), d) for d in DEPTHS if d >= 3]
                mean_r = float(np.mean(risks))
                if mean_r < best_val_risk - 1e-9:
                    best_val_risk = mean_r; best_hyper = hyper
            key = json.dumps(best_hyper, sort_keys=True)
            hyper_pick_freq[arm][key] = hyper_pick_freq[arm].get(key, 0) + 1
            test_scores = sequential_scores(test_probs, dependence=arm, design=design_labels, hyper=best_hyper, max_depth=pool_full_depth)
            des_scores  = sequential_scores(des_probs,  dependence=arm, design=design_labels, hyper=best_hyper, max_depth=pool_full_depth)
            val_scores2 = sequential_scores(val_probs,  dependence=arm, design=design_labels, hyper=best_hyper, max_depth=pool_full_depth)
            test_order = np.argsort(test_scores, axis=1)
            des_order  = np.argsort(des_scores,  axis=1)
            val_order2 = np.argsort(val_scores2, axis=1)
            for d in DEPTHS:
                result[arm]["test"][d].append(depth_risk(test_order, test_labels.astype(int), d))
                result[arm]["design"][d].append(depth_risk(des_order, design_labels.astype(int), d))
                result[arm]["val"][d].append(depth_risk(val_order2, val_labels.astype(int), d))
    return result, hyper_pick_freq, probs.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--routerbench-pkl", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--shot", default="0shot")
    ap.add_argument("--seeds", default="7001,7002,7003,7004,7005")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(a.run_dir)
    embeddings = np.load(a.embeddings)
    for task in CLEAN_TASKS:
        print(f"== {task} ==", flush=True)
        curves, hyper_pick, n_models = compute_task(
            task, run_dir, a.routerbench_pkl, embeddings, seeds, a.shot,
        )
        payload = {
            "task": task, "n_models": n_models, "seeds": seeds,
            "depths": list(DEPTHS), "hyper_pick_freq": hyper_pick,
            "curves": {arm: {ph: {str(d): curves[arm][ph][d] for d in DEPTHS} for ph in ("test","design","val")}
                       for arm in curves},
        }
        (out / f"{task}.json").write_text(json.dumps(payload, indent=2))
        print(f"  wrote {task}.json")


if __name__ == "__main__":
    raise SystemExit(main())
