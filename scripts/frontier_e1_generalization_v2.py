"""E1 fixed-prefix dependence-aware ordering: generalization-gap frontiers
across all datasets. Mirrors frontier_e2_generalization for E1's policy class.

Datasets:
- RouterEval aggregate benchmarks (ifeval, bbh, gpqa, math, musr) at pool 100
  (subsampled from 3811 for direct comparison to E2 plots at same scale).
- RouterBench clean tasks (mmlu, hellaswag, winogrande, arc-challenge) at pool 11
  (full).

Three fixed-prefix dependence methods: empirical_tally, error_correlation,
gaussian_copula. Design B' hyperparameter tuning per method on val split.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.routereval_protocol import subsample_pool
from components.scale_selectors import build_order

DEPTHS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
E1_METHODS = ("empirical_tally", "error_correlation", "gaussian_copula")
# Small hyper grids for Design B' fairness
HYPER_GRIDS = {
    "empirical_tally":   [0.0, 0.5, 1.0, 2.0],
    "error_correlation": [0.90, 0.95, 0.99, 0.999],
    "gaussian_copula":   [0.0, 0.01, 0.05, 0.10, 0.20],
}

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


def load_routereval_aggregate(lb_path, bench):
    import pickle
    lb = pickle.load(open(lb_path, "rb"))
    parts = [np.asarray(lb["data"][k]["correctness"]) for k in BENCH_LB_KEYS[bench]]
    binary = (np.concatenate(parts, axis=0) >= 0.5).astype(int)
    return binary, [f"m{i}" for i in range(binary.shape[1])]


def compute_curves_from_binary(binary, model_names, pool_size, seeds, val_frac=0.20):
    """Compute per-depth risk curves for E1 methods + marginal on this binary matrix.
    Returns (curves_dict, hyper_pick_freq).
    """
    n_q, n_m = binary.shape
    depths = list(DEPTHS)
    result = {"marginal": {"test": {d: [] for d in depths}, "design": {d: [] for d in depths}, "val": {d: [] for d in depths}}}
    for m in E1_METHODS:
        result[m] = {"test": {d: [] for d in depths}, "design": {d: [] for d in depths}, "val": {d: [] for d in depths}}
    hyper_pick_freq = {m: {} for m in E1_METHODS}

    for seed in seeds:
        if pool_size < n_m:
            cols = subsample_pool(n_m, pool_size, seed=seed)
        else:
            cols = np.arange(n_m)
        sub = binary[:, cols]
        names = [model_names[c] for c in cols]
        rng = np.random.default_rng(seed + 100_003)
        perm = rng.permutation(n_q)
        n_des = int(n_q * 0.30)
        n_val = int(n_q * val_frac)
        des_idx = perm[:n_des]
        val_idx = perm[n_des:n_des+n_val]
        tst_idx = perm[n_des+n_val:]

        design = sub[des_idx].astype(bool)
        val    = sub[val_idx].astype(bool)
        test   = sub[tst_idx].astype(bool)

        max_depth = min(25, len(cols))
        # Marginal ordering: single, no hyper
        m_order = list(build_order("marginal", design, model_names=names, max_depth=max_depth, seed=seed))
        # Fill order to full pool length so all depths are addressable
        remaining = [i for i in range(len(cols)) if i not in m_order]
        m_full = m_order + remaining
        for d in depths:
            top = m_full[:d]
            fails_d = ~design[:, top].any(axis=1)
            fails_v = ~val[:, top].any(axis=1)
            fails_t = ~test[:, top].any(axis=1)
            result["marginal"]["design"][d].append(float(fails_d.mean()))
            result["marginal"]["val"][d].append(float(fails_v.mean()))
            result["marginal"]["test"][d].append(float(fails_t.mean()))

        # E1 methods with Design B' hyperparameter tuning on val
        for method in E1_METHODS:
            best_hyper = None; best_val_avg = float("inf"); best_order = None
            for hyper in HYPER_GRIDS[method]:
                order = list(build_order(method, design, model_names=names,
                                          max_depth=max_depth, seed=seed, n_draws=1500, hyper=hyper))
                rem = [i for i in range(len(cols)) if i not in order]
                full = order + rem
                # val risk averaged over depths 3..12
                val_risks = []
                for d in depths:
                    if d < 3: continue
                    top = full[:d]
                    val_risks.append(float((~val[:, top].any(axis=1)).mean()))
                mean_r = float(np.mean(val_risks))
                if mean_r < best_val_avg - 1e-9:
                    best_val_avg = mean_r; best_hyper = hyper; best_order = full
            key = json.dumps({"h": best_hyper})
            hyper_pick_freq[method][key] = hyper_pick_freq[method].get(key, 0) + 1
            for d in depths:
                top = best_order[:d]
                result[method]["design"][d].append(float((~design[:, top].any(axis=1)).mean()))
                result[method]["val"][d].append(float((~val[:, top].any(axis=1)).mean()))
                result[method]["test"][d].append(float((~test[:, top].any(axis=1)).mean()))
    return result, hyper_pick_freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("routereval","routerbench"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="7001,7002,7003,7004,7005")
    ap.add_argument("--pool-size", type=int, default=100)
    ap.add_argument("--leaderboard")
    ap.add_argument("--routerbench-pkl")
    ap.add_argument("--benchmark")
    ap.add_argument("--task")
    ap.add_argument("--primary-run-dir", help="directory of a primary RouterBench run containing fits.json and split_assignments.json (routerbench mode only)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]
    if a.mode == "routereval":
        binary, names = load_routereval_aggregate(a.leaderboard, a.benchmark)
        curves, hyper = compute_curves_from_binary(binary, names, a.pool_size, seeds)
        payload = {"mode":"routereval","benchmark":a.benchmark,"pool_size":a.pool_size,
                   "seeds":seeds,"depths":list(DEPTHS),"hyper_pick_freq":hyper,
                   "curves":{m:{ph:{str(d):curves[m][ph][d] for d in DEPTHS} for ph in ("test","design","val")}
                             for m in curves}}
    else:  # routerbench
        import pickle
        from components.data_protocol import load_routerbench, derive_coverage_labels, resolve_coverage_policy
        run_dir = Path(a.primary_run_dir)
        fits = json.loads((run_dir / "fits.json").read_text())
        splits = json.loads((run_dir / "split_assignments.json").read_text())
        pol_rec = splits["coverage_policy"]
        policy_name = pol_rec["name"] if isinstance(pol_rec, dict) else pol_rec
        data = load_routerbench(a.routerbench_pkl, shot="0shot")
        policy = resolve_coverage_policy(policy_name)
        labels = derive_coverage_labels(data, policy=policy)
        # get this task's query IDs from primary run
        task_fits = fits["tasks"][a.task]
        pool_ids = list(task_fits["split_ids"]["calibration"]) + list(task_fits["split_ids"]["test"])
        label_row_of = {q: i for i, q in enumerate(labels.query_ids)}
        rows = np.asarray([label_row_of[q] for q in pool_ids if q in label_row_of], dtype=int)
        binary = labels.standard[rows].astype(int)
        names = [f"m{i}" for i in range(binary.shape[1])]
        curves, hyper = compute_curves_from_binary(binary, names, binary.shape[1], seeds)
        payload = {"mode":"routerbench","task":a.task,"n_models":binary.shape[1],
                   "seeds":seeds,"depths":list(DEPTHS),"hyper_pick_freq":hyper,
                   "curves":{m:{ph:{str(d):curves[m][ph][d] for d in DEPTHS} for ph in ("test","design","val")}
                             for m in curves}}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
