"""Four-way frontier analysis: design / calibration / validation / test.

This driver turns the frontier envelope into a deployable policy question. For
each seed the queries are split disjointly into four partitions with ratios
(0.4, 0.3, 0.15, 0.15) after subtracting the absolute design budget from the
front. Every ordering in the taxonomy is fit on the design partition. For each
alpha the conformal risk control threshold per ordering is calibrated on the
calibration partition. Two selection modes then choose one winning ordering:

  per-alpha (primary). For each alpha, select the ordering that minimises the
    calibrated depth on cal subject to its risk on validation being at most
    alpha. If no ordering satisfies the validation constraint, fall back to the
    ordering with the lowest validation risk at its calibrated depth. Evaluate
    the winner on test.

  per-depth (frontier-analog). For each depth d, select the ordering that
    minimises the validation risk at depth d and evaluate that ordering's risk
    on test at the same depth.

The oracle envelope on test, computed as the pointwise minimum over all
orderings at each depth, is emitted alongside for comparison. The difference
between the selected policy's test risk and the envelope's test risk is the
selection penalty (DR-B1, DR-B2 in the plan).

Labels-only. No calibration by any external model. Deterministic given a seed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.routereval_protocol import (  # noqa: E402
    load_correctness_only as _load_routereval,
    subsample_pool as _subsample_routereval,
)
from components.routerbench_protocol import (  # noqa: E402
    load_correctness_only as _load_routerbench,
    subsample_pool as _subsample_routerbench,
)
from components.scale_selectors import build_order  # noqa: E402


ORDERINGS = (
    "marginal",
    "empirical_tally",
    "error_correlation",
    "gaussian_copula",
    "student_t_copula",
    "factor_copula",
    "bernoulli_mixture",
    "rank_correlation",
    "lineage",
    "shrinkage_indep",
    "shrinkage_factor",
    "double_fault",
)


def _load_data(protocol: str, data_path: str, benchmark: str):
    if protocol == "routereval":
        return _load_routereval(leaderboard_path=data_path, leaderboard_key=benchmark)
    if protocol == "routerbench":
        return _load_routerbench(pickle_path=data_path, benchmark=benchmark)
    raise ValueError(f"unknown protocol {protocol!r}")


def _subsample(protocol: str, n_models: int, pool_size: int, seed: int):
    if protocol == "routereval":
        return _subsample_routereval(n_models, pool_size, seed=seed)
    if protocol == "routerbench":
        return _subsample_routerbench(n_models, pool_size, seed=seed)
    raise ValueError(f"unknown protocol {protocol!r}")


def _load_transfer_design(
    protocol: str,
    data_path: str,
    target_benchmark: str,
    transfer_benchmarks: list[str],
    threshold: float,
    design_budget: int,
    seed: int,
) -> np.ndarray:
    """Variant TF design partition: labels drawn from OTHER benchmarks entirely.

    Plan Section 11 distinguishes two legitimate training-data sources. Variant IN
    fits the dependence structure on a slice of the target benchmark, which licenses
    a within-protocol comparison against the marginal baseline because both are fit
    the same way. Variant TF instead fits it on a pool of DIFFERENT benchmarks, which
    is the source RACER's own KNN scoring function uses, and is therefore the only
    variant that supports a comparison against RACER's published numbers.

    Alignment is trivial here and worth stating: within one dataset every benchmark
    shares the same model columns in the same order (3,811 for RouterEval, 11 for
    RouterBench), so correctness matrices from different benchmarks concatenate along
    the query axis with no reindexing. The target benchmark is excluded from the
    transfer pool, which is the entire point of the variant.
    """
    rows = []
    for name in transfer_benchmarks:
        if name == target_benchmark:
            continue
        data = _load_data(protocol, data_path, name)
        rows.append((data.correctness >= threshold).astype(int))
    if not rows:
        raise ValueError(
            f"transfer pool empty after excluding the target {target_benchmark!r}"
        )
    widths = {block.shape[1] for block in rows}
    if len(widths) != 1:
        raise ValueError(
            f"transfer benchmarks disagree on model count: {sorted(widths)}; "
            "they cannot be concatenated"
        )
    pooled = np.concatenate(rows, axis=0)
    generator = np.random.default_rng(seed + 555_001)
    if design_budget >= pooled.shape[0]:
        return pooled
    picked = generator.choice(pooled.shape[0], size=design_budget, replace=False)
    return pooled[np.sort(picked)]


def _split_indices(
    n_queries: int,
    design_budget: int,
    cal_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    """Deterministically partition indices into design/cal/val/test.

    design is an absolute count; cal, val, test are ratios of the remaining
    queries after design is cut off. The three ratios need not sum to 1;
    they are normalised so the three post-design partitions consume everything
    that is left.
    """
    if design_budget >= n_queries:
        raise ValueError(f"design_budget {design_budget} >= n_queries {n_queries}")
    generator = np.random.default_rng(seed + 100_003)
    permutation = generator.permutation(n_queries)
    design_idx = permutation[:design_budget]
    rest = permutation[design_budget:]
    n_rest = len(rest)
    denom = cal_ratio + val_ratio + test_ratio
    n_cal = int(round(n_rest * (cal_ratio / denom)))
    n_val = int(round(n_rest * (val_ratio / denom)))
    n_cal = max(n_cal, 1)
    n_val = max(n_val, 1)
    n_test = n_rest - n_cal - n_val
    if n_test < 1:
        raise ValueError(
            f"partitioning yielded n_test={n_test}; ratios cal={cal_ratio}, "
            f"val={val_ratio}, test={test_ratio}, n_rest={n_rest}"
        )
    cal_idx = rest[:n_cal]
    val_idx = rest[n_cal:n_cal + n_val]
    test_idx = rest[n_cal + n_val:]
    return {
        "design": design_idx,
        "cal": cal_idx,
        "val": val_idx,
        "test": test_idx,
    }


def _cumulative_miss(matrix: np.ndarray, order: list[int]) -> np.ndarray:
    """Return per-query cumulative-all-miss booleans of shape (n_queries, n_ordered).

    `matrix` is (n_queries, n_models) BINARY correctness with 1=correct.
    Order specifies the model column indices in the fitted prefix ordering.
    """
    failures = matrix[:, order] == 0
    return np.logical_and.accumulate(failures, axis=1)


def _calibrated_depth(
    cal_miss: np.ndarray,
    alpha: float,
) -> int | None:
    """Split-conformal CRC over a fixed nested prefix.

    cal_miss is (n_cal, n_depths) boolean, where cal_miss[i, d-1] is true iff
    the first d models in the ordering all miss query i.

    Uses the finite-sample correction: pick smallest k such that
    (n_cal * empirical_miss_at_k + 1) / (n_cal + 1) <= alpha.
    Returns None if no depth satisfies the bound (target is infeasible).
    """
    if cal_miss.size == 0:
        return None
    n_cal, n_depths = cal_miss.shape
    empirical = cal_miss.mean(axis=0)
    target = (alpha * (n_cal + 1) - 1.0) / n_cal
    if target < 0:
        return None
    for k in range(n_depths):
        if empirical[k] <= target + 1e-12:
            return k + 1
    return None


def _fit_one_seed(
    binary: np.ndarray,
    model_names: list[str],
    design_budget: int,
    cal_ratio: float,
    val_ratio: float,
    test_ratio: float,
    pool_size: int,
    max_depth: int,
    seed: int,
    depths: tuple[int, ...],
    alphas: list[float],
    protocol: str,
    transfer_design_full: np.ndarray | None = None,
) -> dict:
    n_queries, n_models = binary.shape
    columns = _subsample(protocol, n_models, pool_size, seed=seed)
    sub = binary[:, columns]
    names = [model_names[c] for c in columns]

    parts = _split_indices(
        n_queries=n_queries,
        design_budget=design_budget,
        cal_ratio=cal_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    if transfer_design_full is None:
        # Variant IN: the design partition is a disjoint slice of the target benchmark.
        design = sub[parts["design"]]
    else:
        # Variant TF: the design partition comes from other benchmarks entirely, so the
        # target's own design slice is UNUSED and its cal/val/test are untouched. The
        # same model columns are taken so the ordering indexes the same pool.
        design = transfer_design_full[:, columns]
    cal = sub[parts["cal"]]
    val = sub[parts["val"]]
    test = sub[parts["test"]]

    per_ordering: dict[str, dict] = {}
    for ordering in ORDERINGS:
        order = build_order(
            ordering,
            design,
            model_names=names,
            max_depth=max_depth,
            seed=seed,
            n_draws=2000,
        )
        order = list(order)
        design_miss = _cumulative_miss(design, order)
        cal_miss = _cumulative_miss(cal, order)
        val_miss = _cumulative_miss(val, order)
        test_miss = _cumulative_miss(test, order)
        max_d = min(max_depth, len(order))
        depth_slice = [d for d in depths if d <= max_d]
        per_ordering[ordering] = {
            "order": order,
            "depths": depth_slice,
            "design": {d: float(design_miss[:, d - 1].mean()) for d in depth_slice},
            "cal": {d: float(cal_miss[:, d - 1].mean()) for d in depth_slice},
            "val": {d: float(val_miss[:, d - 1].mean()) for d in depth_slice},
            "test": {d: float(test_miss[:, d - 1].mean()) for d in depth_slice},
            "cal_miss_matrix": cal_miss,
            "val_miss_matrix": val_miss,
            "test_miss_matrix": test_miss,
        }

    # Per-alpha selection
    per_alpha: dict[str, dict] = {}
    for alpha in alphas:
        calibrated: dict[str, int | None] = {}
        val_risk_at_cal: dict[str, float | None] = {}
        test_risk_at_cal: dict[str, float | None] = {}
        for ordering, arm in per_ordering.items():
            k_star = _calibrated_depth(arm["cal_miss_matrix"], alpha)
            calibrated[ordering] = k_star
            if k_star is None or k_star > arm["val_miss_matrix"].shape[1]:
                val_risk_at_cal[ordering] = None
                test_risk_at_cal[ordering] = None
            else:
                val_risk_at_cal[ordering] = float(arm["val_miss_matrix"][:, k_star - 1].mean())
                test_risk_at_cal[ordering] = float(arm["test_miss_matrix"][:, k_star - 1].mean())

        # Primary selector: among orderings with val_risk <= alpha at their k*, pick lowest k*.
        # Fallback: pick lowest val_risk over orderings whose k* is defined.
        eligible = [
            (ordering, calibrated[ordering], val_risk_at_cal[ordering])
            for ordering in per_ordering
            if calibrated[ordering] is not None
            and val_risk_at_cal[ordering] is not None
            and val_risk_at_cal[ordering] <= alpha + 1e-12
        ]
        if eligible:
            eligible.sort(key=lambda triple: (triple[1], triple[2]))
            selected = eligible[0][0]
        else:
            defined = [
                (ordering, val_risk_at_cal[ordering], calibrated[ordering])
                for ordering in per_ordering
                if calibrated[ordering] is not None
                and val_risk_at_cal[ordering] is not None
            ]
            if defined:
                defined.sort(key=lambda t: (t[1], t[2]))
                selected = defined[0][0]
            else:
                selected = "marginal"

        selected_k = calibrated[selected]
        selected_val = val_risk_at_cal[selected]
        selected_test = test_risk_at_cal[selected]
        n_test = per_ordering[selected]["test_miss_matrix"].shape[0]
        per_alpha[f"{alpha}"] = {
            "selected_method": selected,
            "selected_k": selected_k,
            "selected_val_risk": selected_val,
            "selected_test_risk": selected_test,
            "test_set_size": n_test,
            "calibrated_depth_per_ordering": {o: calibrated[o] for o in per_ordering},
            "val_risk_at_calibrated_per_ordering": {
                o: val_risk_at_cal[o] for o in per_ordering
            },
            "test_risk_at_calibrated_per_ordering": {
                o: test_risk_at_cal[o] for o in per_ordering
            },
        }

    # Oracle envelope (min over dependence orderings, excluding marginal, at each depth on test)
    depth_union = sorted(set().union(*(set(a["depths"]) for a in per_ordering.values())))
    envelope: dict[int, dict] = {}
    for depth in depth_union:
        arm_vals = {
            ordering: per_ordering[ordering]["test"].get(depth)
            for ordering in per_ordering
            if ordering != "marginal"
        }
        arm_vals = {k: v for k, v in arm_vals.items() if v is not None}
        if not arm_vals:
            continue
        winner = min(arm_vals, key=arm_vals.get)
        envelope[depth] = {"winner": winner, "test_risk": arm_vals[winner]}

    # Per-depth val-selection (frontier-analog)
    per_depth_val_selected: dict[int, dict] = {}
    for depth in depth_union:
        val_at_depth = {
            ordering: per_ordering[ordering]["val"].get(depth)
            for ordering in per_ordering
            if ordering != "marginal"
        }
        val_at_depth = {k: v for k, v in val_at_depth.items() if v is not None}
        if not val_at_depth:
            continue
        winner = min(val_at_depth, key=val_at_depth.get)
        per_depth_val_selected[depth] = {
            "winner": winner,
            "val_risk": val_at_depth[winner],
            "test_risk": per_ordering[winner]["test"][depth],
        }

    # Strip the heavy boolean matrices before serialization
    for arm in per_ordering.values():
        arm.pop("cal_miss_matrix", None)
        arm.pop("val_miss_matrix", None)
        arm.pop("test_miss_matrix", None)

    return {
        "seed": seed,
        "n_design": int(len(parts["design"])),
        "n_cal": int(len(parts["cal"])),
        "n_val": int(len(parts["val"])),
        "n_test": int(len(parts["test"])),
        "per_ordering": per_ordering,
        "per_alpha": per_alpha,
        "envelope_per_depth": envelope,
        "per_depth_val_selected": per_depth_val_selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, choices=("routereval", "routerbench"))
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--design-budget", type=int, required=True)
    parser.add_argument("--alphas", default="0.05,0.10")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--split-ratios", default="0.4,0.3,0.15,0.15",
                        help="design,cal,val,test; design is absolute (--design-budget) and its "
                             "ratio is used only when the plan cites the ratios")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--training-source", default="in_domain",
                        choices=("in_domain", "transfer"),
                        help="Variant IN fits the dependence structure on a slice of the "
                             "target benchmark; Variant TF fits it on other benchmarks, "
                             "matching RACER's own training-data source")
    parser.add_argument("--transfer-benchmarks", default=None,
                        help="comma-separated benchmarks forming the transfer pool; the "
                             "target is excluded automatically. Required for transfer.")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    data = _load_data(arguments.protocol, arguments.data_path, arguments.benchmark)
    binary = (data.correctness >= arguments.threshold).astype(int)
    seeds = [int(s) for s in arguments.seeds.split(",") if s]
    alphas = [float(a) for a in arguments.alphas.split(",") if a]
    ratios = [float(r) for r in arguments.split_ratios.split(",") if r]
    if len(ratios) != 4:
        parser.error("--split-ratios must have four comma-separated values")
    _, cal_ratio, val_ratio, test_ratio = ratios

    transfer_design_full = None
    transfer_pool_used: list[str] = []
    if arguments.training_source == "transfer":
        if not arguments.transfer_benchmarks:
            parser.error("--training-source transfer requires --transfer-benchmarks")
        transfer_pool_used = [
            b.strip() for b in arguments.transfer_benchmarks.split(",")
            if b.strip() and b.strip() != arguments.benchmark
        ]
        transfer_design_full = _load_transfer_design(
            protocol=arguments.protocol,
            data_path=arguments.data_path,
            target_benchmark=arguments.benchmark,
            transfer_benchmarks=transfer_pool_used,
            threshold=arguments.threshold,
            design_budget=arguments.design_budget,
            seed=seeds[0],
        )
        print(f"Variant TF: design drawn from {len(transfer_pool_used)} other benchmarks, "
              f"{transfer_design_full.shape[0]} rows")

    if arguments.depths:
        depths = tuple(int(d) for d in arguments.depths.split(",") if d)
    elif arguments.protocol == "routerbench":
        depths = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    else:
        depths = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)

    seed_results = []
    for seed in seeds:
        result = _fit_one_seed(
            binary=binary,
            model_names=list(data.model_names),
            design_budget=arguments.design_budget,
            cal_ratio=cal_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            pool_size=arguments.pool_size,
            max_depth=arguments.max_depth,
            seed=seed,
            depths=depths,
            alphas=alphas,
            protocol=arguments.protocol,
            transfer_design_full=transfer_design_full,
        )
        seed_results.append(result)

    # Aggregate
    aggregated_per_alpha: dict[str, dict] = {}
    for alpha in alphas:
        akey = f"{alpha}"
        per_seed = [r["per_alpha"][akey] for r in seed_results]
        risks = [ps["selected_test_risk"] for ps in per_seed if ps["selected_test_risk"] is not None]
        calls = [ps["selected_k"] for ps in per_seed if ps["selected_k"] is not None]
        methods = [ps["selected_method"] for ps in per_seed]
        aggregated_per_alpha[akey] = {
            "median_selected_test_risk": float(np.median(risks)) if risks else None,
            "mean_selected_test_risk": float(np.mean(risks)) if risks else None,
            "median_selected_k": float(np.median(calls)) if calls else None,
            "mean_selected_k": float(np.mean(calls)) if calls else None,
            "selected_method_by_seed": methods,
        }

    envelope_agg: dict = {}
    depth_union = sorted(set().union(*(set(r["envelope_per_depth"].keys()) for r in seed_results)))
    for depth in depth_union:
        vals = [r["envelope_per_depth"][depth]["test_risk"]
                for r in seed_results if depth in r["envelope_per_depth"]]
        winners = [r["envelope_per_depth"][depth]["winner"]
                   for r in seed_results if depth in r["envelope_per_depth"]]
        envelope_agg[depth] = {
            "mean_test_risk": float(np.mean(vals)),
            "median_test_risk": float(np.median(vals)),
            "winner_by_seed": winners,
        }

    output = {
        "protocol": arguments.protocol,
        "machine": os.uname().nodename,
        "machine_alias": os.environ.get("DACR_MACHINE_ALIAS", ""),
        "run_finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_source": arguments.training_source,
        "transfer_pool": transfer_pool_used,
        "transfer_design_rows": (
            int(transfer_design_full.shape[0]) if transfer_design_full is not None else None
        ),
        "benchmark": arguments.benchmark,
        "pool_size": arguments.pool_size,
        "design_budget": arguments.design_budget,
        "split_ratios": ratios,
        "alphas": alphas,
        "seeds": seeds,
        "depths": list(depths),
        "max_depth": arguments.max_depth,
        "feature_audit": dict(data.feature_audit),
        "source_sha256": data.source_sha256,
        "seed_results": seed_results,
        "aggregated_per_alpha": aggregated_per_alpha,
        "envelope_aggregated": envelope_agg,
    }

    out_path = Path(arguments.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(output, handle, indent=2, default=str)
    tmp_path.replace(out_path)

    print(
        f"protocol {arguments.protocol}  benchmark {arguments.benchmark}  "
        f"pool {arguments.pool_size}  budget {arguments.design_budget}  seeds {len(seeds)}"
    )
    for alpha in alphas:
        akey = f"{alpha}"
        agg = aggregated_per_alpha[akey]
        print(
            f"  alpha {alpha}: selected_method_mode="
            f"{max(set(agg['selected_method_by_seed']), key=agg['selected_method_by_seed'].count)}"
            f"  median calls={agg['median_selected_k']}"
            f"  median test risk={agg['median_selected_test_risk']}"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
