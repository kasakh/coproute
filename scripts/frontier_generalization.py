"""Emit both in-sample (design-set) and out-of-sample (test-set) risk curves.

For each ordering, fits on a random design subset of size design_budget, then
evaluates the same ordering on both the design queries and the disjoint test
queries. Purpose: visualize the generalization gap between the risk on the
data used to fit the ordering and the risk on unseen queries.

Labels only. CPU only.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.routereval_protocol import load_correctness_only, subsample_pool  # noqa: E402
from components.scale_selectors import build_order  # noqa: E402

ORDERINGS = (
    "marginal","empirical_tally","error_correlation","gaussian_copula","student_t_copula",
    "factor_copula","bernoulli_mixture","rank_correlation","lineage","shrinkage_indep",
    "shrinkage_factor","double_fault",
)
DEPTHS = (1, 2, 3, 4, 5, 6, 8, 10, 12)


def compute(binary, model_names, design_budget, seeds, pool_size, max_depth):
    n_queries, n_models = binary.shape
    curves = {o: {"design": {d: [] for d in DEPTHS}, "test": {d: [] for d in DEPTHS}} for o in ORDERINGS}
    for seed in seeds:
        columns = subsample_pool(n_models, pool_size, seed=seed)
        sub = binary[:, columns]
        names = [model_names[c] for c in columns]
        rng = np.random.default_rng(seed + 100_003)
        perm = rng.permutation(n_queries)
        d_idx = perm[:design_budget]
        t_idx = perm[design_budget:]
        design = sub[d_idx]
        test = sub[t_idx].astype(bool)
        design_b = sub[d_idx].astype(bool)
        for ordering in ORDERINGS:
            order = list(build_order(ordering, design, model_names=names, max_depth=max_depth, seed=seed, n_draws=2000))
            # test-set risk
            fails_t = ~test[:, order]
            cum_t = np.logical_and.accumulate(fails_t, axis=1) if len(fails_t) else fails_t
            # design-set risk
            fails_d = ~design_b[:, order]
            cum_d = np.logical_and.accumulate(fails_d, axis=1) if len(fails_d) else fails_d
            for depth in DEPTHS:
                if len(cum_t) and depth <= cum_t.shape[1]:
                    curves[ordering]["test"][depth].append(float(cum_t[:, depth - 1].mean()))
                if len(cum_d) and depth <= cum_d.shape[1]:
                    curves[ordering]["design"][depth].append(float(cum_d[:, depth - 1].mean()))
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboard", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--pool-size", type=int, default=3811)
    ap.add_argument("--design-budget", type=int, default=100)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-depth", type=int, default=15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    data = load_correctness_only(leaderboard_path=a.leaderboard, leaderboard_key=a.benchmark)
    binary = (data.correctness >= a.threshold).astype(int)
    seeds = [int(s) for s in a.seeds.split(",") if s]

    curves = compute(binary, list(data.model_names), a.design_budget, seeds, a.pool_size, a.max_depth)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as h:
        json.dump({
            "benchmark": a.benchmark, "pool_size": a.pool_size,
            "design_budget": a.design_budget, "seeds": seeds, "depths": list(DEPTHS),
            "curves": {o: {"design": {str(d): curves[o]["design"][d] for d in DEPTHS},
                           "test":   {str(d): curves[o]["test"][d]   for d in DEPTHS}} for o in ORDERINGS},
        }, h, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
