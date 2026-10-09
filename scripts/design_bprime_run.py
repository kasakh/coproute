"""Design B prime (ISSUE-34): per-method hyperparameter tuning, no cross-method selection.

Design A picks the single best of eleven methods on a validation partition and
reports one test number, so its comparison against a single-candidate marginal
baseline carries cross-method winner's curse over K = 11, about
`sigma * sqrt(2 ln 11) = 2.19 * sigma`. Design B prime removes that entirely by
never comparing across methods inside the pipeline. Each method independently
tunes its own single hyperparameter on validation and is then evaluated on test,
and the paper reports eleven separate per-method-against-marginal comparisons.
Within-method curse is bounded at `sigma * sqrt(2 ln 5) = 1.79 * sigma` and
cross-method bias is zero by construction.

Grids are frozen in `components/scale_selectors.py::HYPERPARAMETER_GRIDS` before
any result here is read, per PLAN_DESIGN_BPRIME_PER_METHOD_TUNING_20260904.md
Section 2.

Three grids are INERT and this runner says so rather than hiding it. Laplace
smoothing on `empirical_tally` and the correlation clips on `error_correlation`
and `rank_correlation` are monotone transforms of a ranking whose candidates all
share one denominator, so they cannot reorder it. Those arms therefore have an
EFFECTIVE K of 1 and carry zero within-method selection bias, which is reported
per arm as `effective_k` and is a strictly favourable fact for the fairness
argument rather than a defect.

DR-Bp-4 null shuffling. For each method the whole tuning procedure is repeated on
a design partition whose model columns have been independently shuffled per row,
which destroys co-failure structure while preserving every marginal accuracy. Any
apparent improvement that survives that is an artifact of tuning against noise.
"""

from __future__ import annotations

import argparse
import json
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
from components.scale_selectors import (  # noqa: E402
    HYPERPARAMETER_GRIDS,
    HYPERPARAMETER_NAMES,
    build_order,
)
from scripts.frontier_fourway import _calibrated_depth, _cumulative_miss, _split_indices  # noqa: E402

DEPENDENCE_ARMS = tuple(a for a in HYPERPARAMETER_GRIDS if a != "marginal")


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


def _shuffle_columns_within_rows(design: np.ndarray, seed: int) -> np.ndarray:
    """Destroy co-failure structure, preserve every model's marginal accuracy.

    Shuffling each COLUMN independently across rows keeps each model's own accuracy
    exactly and removes which queries it shares failures with, which is precisely
    the signal the dependence estimators claim to use. The marginal ordering is
    therefore unchanged in distribution while every dependence arm loses its input.
    """
    generator = np.random.default_rng(seed)
    shuffled = np.array(design, copy=True)
    for column in range(shuffled.shape[1]):
        generator.shuffle(shuffled[:, column])
    return shuffled


def _standard_error(values: list[float | int]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=float), ddof=1) / np.sqrt(len(values)))


def _tune_one_arm(
    *,
    arm: str,
    design: np.ndarray,
    cal: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    names: list[str],
    max_depth: int,
    seed: int,
    alpha: float,
) -> dict:
    """Select on design plus validation, then calibrate the frozen arm once.

    Validation supplies a temporary finite-sample depth for every hyperparameter.
    That depth is used only to compare candidates and is discarded. The chosen
    hyperparameter is then frozen. Only that frozen policy sees ``cal``, exactly
    once, and its final calibrated depth is evaluated on test.
    """

    grid = HYPERPARAMETER_GRIDS[arm]
    trajectory = []
    distinct_orders: set[tuple[int, ...]] = set()
    orders: dict[str, list[int]] = {}
    for grid_index, value in enumerate(grid):
        order = list(
            build_order(
                arm, design, model_names=names, max_depth=max_depth,
                seed=seed, n_draws=2000, hyper=value,
            )
        )
        orders[str(grid_index)] = order
        distinct_orders.add(tuple(order))
        val_miss = _cumulative_miss(val, order)
        selection_depth = _calibrated_depth(val_miss, alpha)
        if selection_depth is None:
            fallback_depth = val_miss.shape[1]
            trajectory.append({
                "grid_index": grid_index,
                "hyper": value,
                "selection_depth": fallback_depth,
                "val_risk": float(val_miss[:, fallback_depth - 1].mean()),
                "validation_certified": False,
            })
        else:
            trajectory.append({
                "grid_index": grid_index,
                "hyper": value,
                "selection_depth": int(selection_depth),
                "val_risk": float(val_miss[:, selection_depth - 1].mean()),
                "validation_certified": True,
            })

    validation_certified = [t for t in trajectory if t["validation_certified"]]
    pool = validation_certified if validation_certified else trajectory
    if validation_certified:
        chosen = min(pool, key=lambda t: (t["selection_depth"], t["val_risk"], t["grid_index"]))
    else:
        chosen = min(pool, key=lambda t: (t["val_risk"], t["selection_depth"], t["grid_index"]))

    chosen_order = orders[str(chosen["grid_index"])]
    test_miss = _cumulative_miss(test, chosen_order)
    fixed_depth_curve = [
        {
            "depth": depth,
            "standard_all_miss_risk": float(test_miss[:, depth - 1].mean()),
            "miss_count": int(test_miss[:, depth - 1].sum()),
            "trial_count": int(test_miss.shape[0]),
        }
        for depth in range(1, test_miss.shape[1] + 1)
    ]
    final_depth = _calibrated_depth(_cumulative_miss(cal, chosen_order), alpha)
    base = {
        "arm": arm,
        "hyper_name": HYPERPARAMETER_NAMES[arm],
        "grid_size": len(grid),
        "effective_k": len(distinct_orders),
        "trajectory": trajectory,
        "chosen_hyper": chosen["hyper"],
        "selection_depth": int(chosen["selection_depth"]),
        "chosen_val_risk": chosen["val_risk"],
        "selection_constrained_by_val": bool(validation_certified),
        "final_calibration_count": 1,
        "chosen_order_indices": [int(index) for index in chosen_order],
        "chosen_order_model_names": [names[int(index)] for index in chosen_order],
        "fixed_depth_test_curve": fixed_depth_curve,
    }
    if final_depth is None:
        return {
            **base,
            "chosen_depth": None,
            "final_calibrated_depth": None,
            "test_risk": None,
            "standard_all_miss_risk": None,
            "test_calls": None,
            "certified": False,
            "infeasible": True,
        }

    final_risk = float(test_miss[:, final_depth - 1].mean())
    return {
        **base,
        "chosen_depth": int(final_depth),
        "final_calibrated_depth": int(final_depth),
        "test_risk": final_risk,
        "standard_all_miss_risk": final_risk,
        "test_calls": int(final_depth),
        "certified": True,
        "infeasible": False,
    }

def _assert_arm_presence(seed_results: list[dict], alphas: list[float]) -> None:
    expected_tuned = set(DEPENDENCE_ARMS)
    for seed_record in seed_results:
        seed = seed_record["seed"]
        for alpha in alphas:
            key = f"{alpha}"
            if key not in seed_record["per_alpha"]:
                raise RuntimeError(f"seed {seed} has no output row for alpha {alpha}")
            row = seed_record["per_alpha"][key]
            if "marginal" not in row:
                raise RuntimeError(f"seed {seed} alpha {alpha} is missing marginal")
            actual = set(row.get("tuned", {}))
            if actual != expected_tuned:
                missing = sorted(expected_tuned - actual)
                extra = sorted(actual - expected_tuned)
                raise RuntimeError(
                    f"seed {seed} alpha {alpha} arm mismatch: missing={missing}, extra={extra}"
                )
            for arm in ("marginal", *DEPENDENCE_ARMS):
                record = row["marginal"] if arm == "marginal" else row["tuned"][arm]
                if not record.get("fixed_depth_test_curve"):
                    raise RuntimeError(
                        f"seed {seed} alpha {alpha} arm {arm} wrote no fixed-depth rows"
                    )


def main() -> int:
    run_started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, choices=("routereval", "routerbench"))
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--design-budget", type=int, required=True)
    parser.add_argument("--alphas", default="0.05,0.10")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--split-ratios", default="0.4,0.3,0.15,0.15")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--null-shuffle", action="store_true",
                        help="also run the DR-Bp-4 column-shuffled control")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    data = _load_data(arguments.protocol, arguments.data_path, arguments.benchmark)
    binary = (data.correctness >= arguments.threshold).astype(int)
    seeds = [int(s) for s in arguments.seeds.split(",") if s]
    alphas = [float(a) for a in arguments.alphas.split(",") if a]
    ratios = [float(r) for r in arguments.split_ratios.split(",") if r]
    _, cal_ratio, val_ratio, test_ratio = ratios
    n_queries, n_models_full = binary.shape

    seed_results = []
    for seed in seeds:
        columns = _subsample(arguments.protocol, n_models_full, arguments.pool_size, seed=seed)
        sub = binary[:, columns]
        names = [data.model_names[c] for c in columns]
        parts = _split_indices(
            n_queries=n_queries, design_budget=arguments.design_budget,
            cal_ratio=cal_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
        )
        design = sub[parts["design"]]
        cal = sub[parts["cal"]]
        val = sub[parts["val"]]
        test = sub[parts["test"]]

        per_alpha = {}
        for alpha in alphas:
            # Marginal baseline: K = 1, no tuning, evaluated the same way.
            baseline = _tune_one_arm(
                arm="marginal", design=design, cal=cal, val=val, test=test,
                names=names, max_depth=arguments.max_depth, seed=seed, alpha=alpha,
            )
            tuned = {
                arm: _tune_one_arm(
                    arm=arm, design=design, cal=cal, val=val, test=test,
                    names=names, max_depth=arguments.max_depth, seed=seed, alpha=alpha,
                )
                for arm in DEPENDENCE_ARMS
            }
            null_tuned = None
            if arguments.null_shuffle:
                shuffled_design = _shuffle_columns_within_rows(design, seed + 777)
                null_tuned = {
                    arm: _tune_one_arm(
                        arm=arm, design=shuffled_design, cal=cal, val=val, test=test,
                        names=names, max_depth=arguments.max_depth, seed=seed, alpha=alpha,
                    )
                    for arm in DEPENDENCE_ARMS
                }
            per_alpha[f"{alpha}"] = {
                "marginal": baseline,
                "tuned": tuned,
                "null_shuffled": null_tuned,
            }

        seed_results.append({
            "seed": seed,
            "n_design": int(len(parts["design"])),
            "n_cal": int(len(parts["cal"])),
            "n_val": int(len(parts["val"])),
            "n_test": int(len(parts["test"])),
            "per_alpha": per_alpha,
        })

    _assert_arm_presence(seed_results, alphas)

    # Aggregate per method per alpha across seeds
    aggregate = {}
    for alpha in alphas:
        akey = f"{alpha}"
        base_calls = [
            r["per_alpha"][akey]["marginal"]["test_calls"] for r in seed_results
            if r["per_alpha"][akey]["marginal"]["test_calls"] is not None
        ]
        base_risk = [
            r["per_alpha"][akey]["marginal"]["test_risk"] for r in seed_results
            if r["per_alpha"][akey]["marginal"]["test_risk"] is not None
        ]
        entry = {
            "marginal_test_calls_mean": float(np.mean(base_calls)) if base_calls else None,
            "marginal_test_calls_se": _standard_error(base_calls),
            "marginal_test_risk_mean": float(np.mean(base_risk)) if base_risk else None,
            "marginal_standard_all_miss_risk_mean": (
                float(np.mean(base_risk)) if base_risk else None
            ),
            "marginal_standard_all_miss_risk_se": _standard_error(base_risk),
            "marginal_certification_count": len(base_calls),
            "marginal_certification_rate": (
                len(base_calls) / len(seed_results) if seed_results else None
            ),
            "methods": {},
        }
        for arm in DEPENDENCE_ARMS:
            records = [r["per_alpha"][akey]["tuned"][arm] for r in seed_results]
            calls = [x["test_calls"] for x in records if x["test_calls"] is not None]
            risks = [x["test_risk"] for x in records if x["test_risk"] is not None]
            chosen = [x["chosen_hyper"] for x in records]
            effective_k = max((x["effective_k"] for x in records), default=0)
            info = {
                "hyper_name": HYPERPARAMETER_NAMES[arm],
                "grid_size": len(HYPERPARAMETER_GRIDS[arm]),
                "effective_k": effective_k,
                "chosen_hyper_by_seed": chosen,
                "test_calls_mean": float(np.mean(calls)) if calls else None,
                "test_calls_se": _standard_error(calls),
                "test_risk_mean": float(np.mean(risks)) if risks else None,
                "standard_all_miss_risk_mean": float(np.mean(risks)) if risks else None,
                "standard_all_miss_risk_se": _standard_error(risks),
                "n_seeds_feasible": len(calls),
                "certification_count": len(calls),
                "certification_rate": (
                    len(calls) / len(seed_results) if seed_results else None
                ),
                "risk_valid": (float(np.mean(risks)) <= alpha + 1e-12) if risks else None,
                "beats_marginal_on_calls": (
                    float(np.mean(calls)) < float(np.mean(base_calls)) - 1e-12
                    if calls and base_calls else None
                ),
            }
            if arguments.null_shuffle:
                null_records = [r["per_alpha"][akey]["null_shuffled"][arm] for r in seed_results]
                null_calls = [x["test_calls"] for x in null_records if x["test_calls"] is not None]
                real_improvement = (
                    float(np.mean(base_calls)) - float(np.mean(calls))
                    if calls and base_calls else None
                )
                null_improvement = (
                    float(np.mean(base_calls)) - float(np.mean(null_calls))
                    if null_calls and base_calls else None
                )
                drbp4 = None
                if real_improvement is not None and null_improvement is not None:
                    drbp4 = bool(
                        null_improvement <= 0.3
                        and real_improvement >= 3.0 * max(null_improvement, 0.0)
                    )
                info["null_shuffled_test_calls_mean"] = (
                    float(np.mean(null_calls)) if null_calls else None
                )
                info["real_improvement_calls"] = real_improvement
                info["null_improvement_calls"] = null_improvement
                info["dr_bp_4_pass"] = drbp4
            entry["methods"][arm] = info
        aggregate[akey] = entry

    output = {
        "design": "B_prime_per_method_tuning",
        "schema_version": "issue57_corrected_e1_v1",
        "issue": "ISSUE-57",
        "registered_arms": ["marginal", *DEPENDENCE_ARMS],
        "machine": os.uname().nodename,
        "machine_alias": os.environ.get("DACR_MACHINE_ALIAS", ""),
        "run_started_utc": run_started_utc,
        "run_finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": arguments.protocol,
        "benchmark": arguments.benchmark,
        "pool_size": arguments.pool_size,
        "design_budget": arguments.design_budget,
        "alphas": alphas,
        "seeds": seeds,
        "split_ratios": ratios,
        "max_depth": arguments.max_depth,
        "null_shuffle": bool(arguments.null_shuffle),
        "hyperparameter_grids": {k: list(v) for k, v in HYPERPARAMETER_GRIDS.items()},
        "feature_audit": dict(data.feature_audit),
        "source_sha256": data.source_sha256,
        "seed_results": seed_results,
        "aggregate": aggregate,
    }
    out_path = Path(arguments.out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite immutable result {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(output, indent=2, default=str))
    tmp.replace(out_path)

    print(f"{arguments.protocol} {arguments.benchmark} pool {arguments.pool_size} "
          f"budget {arguments.design_budget}")
    for alpha in alphas:
        akey = f"{alpha}"
        entry = aggregate[akey]
        mc = entry["marginal_test_calls_mean"]
        mr = entry["marginal_test_risk_mean"]
        mc_s = f"{mc:.2f}" if mc is not None else "n/a"
        mr_s = f"{mr:.4f}" if mr is not None else "n/a"
        print(f"  alpha {alpha}: marginal {mc_s} calls at risk {mr_s}")
        for arm, info in entry["methods"].items():
            tc = info["test_calls_mean"]
            tr = info["test_risk_mean"]
            tc_s = f"{tc:6.2f}" if tc is not None else "   n/a"
            tr_s = f"{tr:.4f}" if tr is not None else "  n/a "
            flag = "" if info["risk_valid"] else "  RISK-INVALID"
            beats = "beats" if info["beats_marginal_on_calls"] else "     "
            extra = ""
            if arguments.null_shuffle and info.get("dr_bp_4_pass") is not None:
                extra = f"  DR-Bp-4 {'PASS' if info['dr_bp_4_pass'] else 'FAIL'}"
            print(f"    {arm:<20s} K={info['grid_size']} eff={info['effective_k']}  "
                  f"calls {tc_s}  risk {tr_s}  {beats}{flag}{extra}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
