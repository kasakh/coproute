"""E2 per-query wrapper frontier: generalization gap plot in the style of
frontier_curves_v4_gengap. For each of the three dependence-aware wrapper
variants (pairwise_count, error_correlation, gaussian_copula), draw one
figure with one panel per benchmark. In each panel:
  - red solid: marginal baseline (RACER-G greedy top-k) on test
  - red dashed: marginal baseline on design/router-fit set
  - blue solid: tuned wrapper on test
  - blue dashed: tuned wrapper on design/router-fit set

Val-selected hyperparameter: minimize wrapper's mean all-miss risk on val
averaged over depths 3..12. Ties broken toward smaller regularization.

x-axis: call depth. y-axis: all-models-miss rate.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.routereval_protocol import subsample_pool
from components.routers import KNNR
from components.racer_wrapper import sequential_scores

DEPTHS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
WRAPPER_ARMS = ("pairwise_count", "error_correlation", "gaussian_copula")
WRAPPER_HYPER_GRIDS = {
    "pairwise_count":     [{"laplace_smoothing": v} for v in (0.0, 0.5, 1.0, 2.0)],
    "error_correlation":  [{"clip_magnitude": v}    for v in (0.90, 0.95, 0.99, 0.999)],
    "gaussian_copula":    [{"ridge": v}             for v in (0.0, 0.01, 0.05, 0.10, 0.20)],
}
# For direct wrapper score comparison the "marginal" baseline is the router
# prob top-k order (RACER-G's greedy).

# Aligned benchmarks (5 RouterEval aggregates)
BENCH_LB_KEYS = {
    "ifeval": ("ifeval",),
    "bbh": (
        "bbh_boolean_expressions","bbh_causal_judgement","bbh_date_understanding",
        "bbh_disambiguation_qa","bbh_formal_fallacies","bbh_geometric_shapes",
        "bbh_hyperbaton","bbh_logical_deduction_five_objects",
        "bbh_logical_deduction_seven_objects","bbh_logical_deduction_three_objects",
        "bbh_movie_recommendation","bbh_navigate","bbh_object_counting",
        "bbh_penguins_in_a_table","bbh_reasoning_about_colored_objects",
        "bbh_ruin_names","bbh_salient_translation_error_detection","bbh_snarks",
        "bbh_sports_understanding","bbh_temporal_sequences",
        "bbh_tracking_shuffled_objects_five_objects",
        "bbh_tracking_shuffled_objects_seven_objects",
        "bbh_tracking_shuffled_objects_three_objects","bbh_web_of_lies",
    ),
    "gpqa": ("gpqa_diamond","gpqa_extended","gpqa_main"),
    "math": ("math_algebra_hard","math_counting_and_prob_hard","math_geometry_hard",
             "math_intermediate_algebra_hard","math_num_theory_hard",
             "math_prealgebra_hard","math_precalculus_hard"),
    "musr": ("musr_murder_mysteries","musr_object_placements","musr_team_allocation"),
}


def load_aligned(lb_path, router_dir, bench):
    import pickle
    lb = pickle.load(open(lb_path, "rb"))
    keys = BENCH_LB_KEYS[bench]
    parts = []
    for k in keys:
        parts.append(np.asarray(lb["data"][k]["correctness"]))
    binary = (np.concatenate(parts, axis=0) >= 0.5).astype(int)
    router_name = bench
    rd = pickle.load(open(f"{router_dir}/{router_name}_router_dataset.pkl", "rb"))
    train_idx = np.asarray(rd["split_index"]["train_indices"])
    val_idx   = np.asarray(rd["split_index"]["val_indices"])
    test_idx  = np.asarray(rd["split_index"]["test_indices"])
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    train_e = np.asarray(rd["embedding"]["train_embed"])
    val_e   = np.asarray(rd["embedding"]["val_embed"])
    test_e  = np.asarray(rd["embedding"]["test_embed"])
    all_e = np.vstack([train_e, val_e, test_e])
    assert binary.shape[0] == all_e.shape[0], (
        f"row mismatch for {bench}: binary {binary.shape[0]} vs embed {all_e.shape[0]}"
    )
    return binary, all_e, all_idx


def depth_risk(order_matrix: np.ndarray, correct: np.ndarray, depth: int) -> float:
    """order_matrix[q, k] = model index chosen for query q at position k.
    correct[q, m] = 1 if model m answers query q correctly.
    Returns mean over queries of P(all top-k models miss).
    """
    n_q = order_matrix.shape[0]
    top_k = order_matrix[:, :depth]
    fails = np.zeros(n_q, dtype=bool)
    for q in range(n_q):
        fails[q] = not np.any(correct[q, top_k[q]] == 1)
    return float(fails.mean())


def compute_bench(binary, embeddings, bench, pool_size, seeds, knn_k, design_frac, router_frac, val_frac, max_depth):
    """Returns curves dict: {arm: {phase: {depth: [values across seeds]}}}."""
    n_q, n_m = binary.shape
    depths = list(DEPTHS)
    result = {}
    result["marginal"] = {"test": {d: [] for d in depths}, "design": {d: [] for d in depths}, "val": {d: [] for d in depths}}
    for arm in WRAPPER_ARMS:
        result[arm] = {"test": {d: [] for d in depths}, "design": {d: [] for d in depths}, "val": {d: [] for d in depths}}
    hyper_pick_freq = {arm: {} for arm in WRAPPER_ARMS}

    for seed in seeds:
        cols = subsample_pool(n_m, pool_size, seed=seed)
        pool_full_depth = pool_size
        sub = binary[:, cols]
        rng = np.random.default_rng(seed + 100_003)
        perm = rng.permutation(n_q)
        n_des = int(n_q * design_frac)
        n_rou = int(n_q * router_frac)
        n_val = int(n_q * val_frac)
        des_idx = perm[:n_des]
        rou_idx = perm[n_des:n_des+n_rou]
        val_idx = perm[n_des+n_rou:n_des+n_rou+n_val]
        tst_idx = perm[n_des+n_rou+n_val:]

        design_labels = sub[des_idx].astype(bool)
        router_labels = sub[rou_idx].astype(bool)
        val_labels    = sub[val_idx].astype(bool)
        test_labels   = sub[tst_idx].astype(bool)

        # Fit KNN router on router split
        knn = KNNR(k=knn_k)
        knn.fit(embeddings[rou_idx], router_labels.astype(float))
        val_probs  = np.clip(knn.predict(embeddings[val_idx]), 0.0, 1.0)
        test_probs = np.clip(knn.predict(embeddings[tst_idx]), 0.0, 1.0)
        des_probs  = np.clip(knn.predict(embeddings[des_idx]), 0.0, 1.0)  # for design-set eval

        # Marginal baseline: top-k by router prob descending
        marg_order_test  = np.argsort(-test_probs, axis=1)
        marg_order_val   = np.argsort(-val_probs,  axis=1)
        marg_order_des   = np.argsort(-des_probs,  axis=1)
        for d in depths:
            result["marginal"]["test"][d].append(depth_risk(marg_order_test, test_labels.astype(int), d))
            result["marginal"]["design"][d].append(depth_risk(marg_order_des, design_labels.astype(int), d))
            result["marginal"]["val"][d].append(depth_risk(marg_order_val, val_labels.astype(int), d))

        # Wrapper variants: val-select hyper, evaluate on test + design
        for arm in WRAPPER_ARMS:
            best_hyper = None; best_val_risk = float("inf")
            arm_score_cache = {}
            for hyper in WRAPPER_HYPER_GRIDS[arm]:
                # Fit dependence structure on design; compute per-query scores on val
                val_scores  = sequential_scores(val_probs,  dependence=arm, design=design_labels, hyper=hyper, max_depth=pool_full_depth)
                # depth-averaged val risk (depths 3..12)
                val_order = np.argsort(val_scores, axis=1)
                risks = [depth_risk(val_order, val_labels.astype(int), d) for d in depths if d >= 3]
                mean_r = float(np.mean(risks))
                if mean_r < best_val_risk - 1e-9:
                    best_val_risk = mean_r
                    best_hyper = hyper
                arm_score_cache[str(hyper)] = None  # placeholder
            # tally
            k = tuple(sorted(best_hyper.items()))
            hyper_pick_freq[arm][str(dict(k))] = hyper_pick_freq[arm].get(str(dict(k)), 0) + 1

            # score at position 1 is 0 (best), later positions have higher scores → ASCENDING sort gives greedy order
            test_scores = sequential_scores(test_probs, dependence=arm, design=design_labels, hyper=best_hyper, max_depth=pool_full_depth)
            des_scores  = sequential_scores(des_probs,  dependence=arm, design=design_labels, hyper=best_hyper, max_depth=pool_full_depth)
            val_scores2 = sequential_scores(val_probs,  dependence=arm, design=design_labels, hyper=best_hyper, max_depth=pool_full_depth)
            test_order = np.argsort(test_scores, axis=1)
            des_order  = np.argsort(des_scores,  axis=1)
            val_order2 = np.argsort(val_scores2, axis=1)
            for d in depths:
                result[arm]["test"][d].append(depth_risk(test_order, test_labels.astype(int), d))
                result[arm]["design"][d].append(depth_risk(des_order, design_labels.astype(int), d))
                result[arm]["val"][d].append(depth_risk(val_order2, val_labels.astype(int), d))
    return result, hyper_pick_freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboard", required=True)
    ap.add_argument("--router-dir", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--pool-size", type=int, default=100)
    ap.add_argument("--seeds", default="7001,7002,7003,7004,7005")
    ap.add_argument("--knn-k", type=int, default=40)
    ap.add_argument("--design-fraction", type=float, default=0.30)
    ap.add_argument("--router-fraction", type=float, default=0.20)
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument("--max-depth", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    binary, embeddings, _ = load_aligned(a.leaderboard, a.router_dir, a.benchmark)
    seeds = [int(s) for s in a.seeds.split(",") if s]
    curves, hyper_pick = compute_bench(
        binary, embeddings, a.benchmark, a.pool_size, seeds, a.knn_k,
        a.design_fraction, a.router_fraction, a.val_fraction, a.max_depth,
    )
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": a.benchmark, "pool_size": a.pool_size, "seeds": seeds,
        "depths": list(DEPTHS), "hyper_pick_freq": hyper_pick,
        "curves": {arm: {ph: {str(d): curves[arm][ph][d] for d in DEPTHS} for ph in ("test","design","val")}
                   for arm in curves},
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
