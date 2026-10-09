"""Section 12: the dependence-aware per-query wrapper inside RACER's policy class, on RouterEval.

Scope is deliberately narrow and the reason is recorded rather than glossed. A
per-query wrapper needs, for each query, a router's predicted correctness
probability per model, which means the query's FEATURES must be matched to that
query's own correctness row. On RouterEval that correspondence is only certified
for a benchmark whose leaderboard key is unique:
`routereval_alignment_verify.py` established that `ifeval` holds 541 leaderboard
rows and its `router_dataset` holds exactly 541 indices, so index i addresses row
i with no ordering assumption at all. The aggregated benchmarks (bbh, gpqa, math,
musr) address a CONCATENATION whose sub-task order the release does not state and
which the alignment probe could not certify, so they are excluded here rather
than run on a guess.

Arms, all inside one policy class so the certificate is identical:
  racer_marginal          RACER's own one-minus-probability score, the control
  wrapper_pairwise_count  sequential score from counted co-failure
  wrapper_error_correlation   sequential score from failure correlation
  wrapper_gaussian_copula     sequential score from a latent correlation

Protocol per redraw. The benchmark's queries are permuted, then split into four
disjoint partitions: design fits the dependence statistic, router-train fits the
nearest-neighbour router, calibration selects the CRC threshold, and test is
untouched. Every score is frozen before calibration. RACER's null column,
nonconformity map, smoothing and threshold scan are reused verbatim, so a change
in abstention is attributable to the real-model scores alone.

Reporting. Calls and realized risk are reported as a distribution over redraws,
never a point value, and a wrapper arm counts as VALID only when its mean realized
risk is at or below the target while the control's is too.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.racer_protocol import (  # noqa: E402
    augment_ground_truth,
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

WRAPPER_ARMS = ("pairwise_count", "error_correlation", "gaussian_copula")

# Benchmarks whose router_dataset to leaderboard alignment is EXACT by construction,
# established by scripts/routereval_alignment_verify.py. Extend only with evidence.
CERTIFIED_ALIGNED = {
    "ifeval": {"router_name": "ifeval", "leaderboard_keys": ("ifeval",)},
}


def _load_aligned(
    router_dir: Path,
    leaderboard_path: Path,
    benchmark: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (correctness, embeddings) with rows in the SAME order.

    Refuses any benchmark not in CERTIFIED_ALIGNED so an uncertified ordering
    can never silently reach a result table.
    """
    if benchmark not in CERTIFIED_ALIGNED:
        raise SystemExit(
            f"benchmark {benchmark!r} has no certified feature-to-label alignment; "
            f"certified set is {sorted(CERTIFIED_ALIGNED)}. Run "
            f"scripts/routereval_alignment_verify.py and extend the set only on evidence."
        )
    spec = CERTIFIED_ALIGNED[benchmark]
    with (router_dir / f"{spec['router_name']}_router_dataset.pkl").open("rb") as handle:
        rd = pickle.load(handle)
    with leaderboard_path.open("rb") as handle:
        leaderboard = pickle.load(handle)

    blocks = [
        np.asarray(leaderboard["data"][key]["correctness"], dtype=float)
        for key in spec["leaderboard_keys"]
    ]
    correctness_full = np.concatenate(blocks, axis=0)
    binary = (correctness_full >= threshold).astype(int)

    split = rd["split_index"]
    embed = rd["embedding"]
    index_parts = []
    embed_parts = []
    for index_key, embed_key in (
        ("train_indices", "train_embed"),
        ("val_indices", "val_embed"),
        ("test_indices", "test_embed"),
    ):
        index_parts.append(np.asarray(split[index_key], dtype=int))
        embed_parts.append(np.asarray(embed[embed_key], dtype=float))
    all_indices = np.concatenate(index_parts)
    all_embeddings = np.concatenate(embed_parts, axis=0)

    if len(all_indices) != binary.shape[0]:
        raise SystemExit(
            f"{benchmark}: {len(all_indices)} router_dataset indices against "
            f"{binary.shape[0]} leaderboard rows; refusing to proceed"
        )
    if len(np.unique(all_indices)) != len(all_indices):
        raise SystemExit(f"{benchmark}: router_dataset indices are not unique")

    # Reorder so row i of both arrays is the SAME query
    order = np.argsort(all_indices)
    embeddings_aligned = all_embeddings[order]
    # all_indices[order] is now 0..n-1, so binary needs no reordering
    return binary, embeddings_aligned


def _knn_probabilities(
    train_embed: np.ndarray,
    train_correct: np.ndarray,
    query_embed: np.ndarray,
    k: int,
) -> np.ndarray:
    """Neighbourhood-average predicted correctness per model, in [0, 1]."""

    def _normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    train_n = _normalise(train_embed)
    query_n = _normalise(query_embed)
    similarity = query_n @ train_n.T
    k = int(min(k, train_n.shape[0]))
    neighbour_idx = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    predictions = np.empty((query_n.shape[0], train_correct.shape[1]), dtype=float)
    for row in range(query_n.shape[0]):
        predictions[row] = train_correct[neighbour_idx[row]].mean(axis=0)
    return np.clip(predictions, 1e-6, 1.0 - 1e-6)


def _evaluate_arm(
    *,
    scores: np.ndarray,
    augmented_probabilities: np.ndarray,
    augmented_labels: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict:
    """Calibrate on cal rows, evaluate on the untouched test rows.

    Mirrors `scripts/e2_wrapper_run.py::_evaluate` so the RouterEval numbers are
    produced by the same code path as the RouterBench numbers already in the record.
    """
    critical = ground_truth_critical_scores(scores[cal_idx], augmented_labels[cal_idx])
    calibration = select_crc_threshold(critical, alpha)
    routing = prediction_sets(
        augmented_probabilities[test_idx], scores[test_idx], calibration.threshold
    )
    loss = misrouting_loss(routing.augmented_mask, augmented_labels[test_idx])
    return {
        "calls": float(np.mean(routing.size)),
        "risk": float(np.mean(loss)),
        "abstention_rate": float(np.mean(routing.abstain)),
        "feasible": bool(calibration.feasible),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--benchmark", default="ifeval")
    parser.add_argument("--pool-size", type=int, default=100,
                        help="model columns kept; a larger pool lowers the structural floor")
    parser.add_argument("--max-depth", type=int, default=25,
                        help="greedy depth cap inside the wrapper; the score is O(depth x pool) "
                             "per query, and any depth well above the calibrated call count is "
                             "equivalent to no cap while costing far less")
    parser.add_argument("--alphas", default="0.05,0.10")
    parser.add_argument("--redraws", type=int, default=40)
    parser.add_argument("--knn-k", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--design-fraction", type=float, default=0.25)
    parser.add_argument("--router-fraction", type=float, default=0.25)
    parser.add_argument("--cal-fraction", type=float, default=0.25)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    binary, embeddings = _load_aligned(
        Path(arguments.router_dir),
        Path(arguments.leaderboard),
        arguments.benchmark,
        arguments.threshold,
    )
    n_queries, n_models_full = binary.shape
    alphas = [float(a) for a in arguments.alphas.split(",") if a]

    generator = np.random.default_rng(arguments.seed)
    columns = np.sort(
        generator.choice(n_models_full, size=min(arguments.pool_size, n_models_full), replace=False)
    )
    correctness = binary[:, columns]

    # Structural floor: the fraction of queries no model in this pool can answer.
    # No policy of any kind can drive realized risk below this, so when several arms
    # report an IDENTICAL risk it is usually this floor and not a coincidence. Recorded
    # so a reader can tell a genuine call saving at matched risk from a saturated cell.
    complete_pool_miss = float((correctness.sum(axis=1) == 0).mean())
    top1_miss = float(1.0 - correctness.mean(axis=0).max())

    print(
        f"benchmark {arguments.benchmark}  queries {n_queries}  "
        f"pool {len(columns)} of {n_models_full}  redraws {arguments.redraws}"
    )
    print(
        f"  structural floor: complete-pool miss {complete_pool_miss:.4f}, "
        f"top-1 model miss {top1_miss:.4f}"
    )
    for alpha in alphas:
        if complete_pool_miss > alpha:
            print(f"  WARNING alpha {alpha} is BELOW the complete-pool floor "
                  f"{complete_pool_miss:.4f}; no policy can meet it on this pool")

    arms = ["racer_marginal_G", "racer_marginal_P"] + [f"wrapper_{name}" for name in WRAPPER_ARMS]
    per_redraw: dict[str, dict[str, list[dict]]] = {
        f"{a}": {arm: [] for arm in arms} for a in alphas
    }

    n_design = int(round(n_queries * arguments.design_fraction))
    n_router = int(round(n_queries * arguments.router_fraction))

    for redraw in range(arguments.redraws):
        rng = np.random.default_rng(arguments.seed + 13 * redraw)
        permutation = rng.permutation(n_queries)
        design_rows = permutation[:n_design]
        router_rows = permutation[n_design:n_design + n_router]
        pool_rows = permutation[n_design + n_router:]
        if min(len(design_rows), len(router_rows), len(pool_rows)) < 40:
            raise SystemExit("a partition fell below 40 queries; adjust the fractions")

        design_labels = correctness[design_rows].astype(int)

        # The router is fit ONLY on the router partition, and probabilities are
        # produced ONLY for the evaluation pool, which is disjoint from both the
        # design and the router partitions. So no query's own label informs its
        # own probability, and the dependence fit never sees an evaluated query.
        probabilities = _knn_probabilities(
            embeddings[router_rows],
            correctness[router_rows],
            embeddings[pool_rows],
            arguments.knn_k,
        )
        augmented_probabilities = augment_router_scores(probabilities)
        augmented_labels = augment_ground_truth(correctness[pool_rows].astype(bool))

        n_pool = len(pool_rows)
        n_cal = n_pool // 2
        cal_idx = np.arange(n_cal)
        test_idx = np.arange(n_cal, n_pool)

        arm_scores: dict[str, np.ndarray] = {}
        for variant in ("G", "P"):
            _, smoothed = nonconformity_scores(
                probabilities, variant, seed=arguments.seed, split_seed=redraw + 1
            )
            arm_scores[f"racer_marginal_{variant}"] = smoothed
        for name in WRAPPER_ARMS:
            raw = sequential_scores(
                probabilities, dependence=name, design=design_labels,
                max_depth=arguments.max_depth,
            )
            augmented_raw = augment_with_racer_null(probabilities, raw)
            arm_scores[f"wrapper_{name}"] = smooth_nonconformity_scores(
                augmented_raw, seed=arguments.seed, split_seed=redraw + 1
            )

        for alpha in alphas:
            for arm, scores in arm_scores.items():
                per_redraw[f"{alpha}"][arm].append(
                    _evaluate_arm(
                        scores=scores,
                        augmented_probabilities=augmented_probabilities,
                        augmented_labels=augmented_labels,
                        cal_idx=cal_idx,
                        test_idx=test_idx,
                        alpha=alpha,
                    )
                )

        if (redraw + 1) % 10 == 0:
            print(f"  completed redraw {redraw + 1} of {arguments.redraws}")

    # Aggregate and assign verdicts. The control is RACER-G, matching the
    # RouterBench end-to-end table already in the record.
    summary: dict[str, dict] = {}
    for alpha in alphas:
        akey = f"{alpha}"
        control = per_redraw[akey]["racer_marginal_G"]
        control_risk = float(np.mean([r["risk"] for r in control]))
        control_calls = float(np.mean([r["calls"] for r in control]))
        entry = {
            "control_arm": "racer_marginal_G",
            "control_calls_mean": control_calls,
            "control_risk_mean": control_risk,
            "control_verdict": "VALID" if control_risk <= alpha + 1e-12 else "OPTIMISTIC",
            "control_per_redraw_calls": [r["calls"] for r in control],
            "control_per_redraw_risk": [r["risk"] for r in control],
            "arms": {},
        }
        for arm in arms:
            records = per_redraw[akey][arm]
            risks = [r["risk"] for r in records]
            calls = [r["calls"] for r in records]
            risk_mean = float(np.mean(risks))
            calls_mean = float(np.mean(calls))
            entry["arms"][arm] = {
                "calls_mean": calls_mean,
                "calls_median": float(np.median(calls)),
                "calls_p10": float(np.percentile(calls, 10)),
                "calls_p90": float(np.percentile(calls, 90)),
                "risk_mean": risk_mean,
                "risk_median": float(np.median(risks)),
                "risk_p90": float(np.percentile(risks, 90)),
                "abstention_mean": float(np.mean([r["abstention_rate"] for r in records])),
                "verdict": "VALID" if risk_mean <= alpha + 1e-12 else "OPTIMISTIC",
                "call_reduction_vs_control": (
                    (control_calls - calls_mean) / control_calls if control_calls > 0 else None
                ),
                "per_redraw_calls": calls,
                "per_redraw_risk": risks,
            }

        # Paired, same-redraw comparison against the control. This is the test the
        # paper's question actually asks: inside RACER's own policy class, does a
        # dependence-aware score beat RACER's own score on the SAME data split?
        # It does not require the control to hit its nominal target, because both
        # arms are calibrated and evaluated identically, so a shared overshoot
        # cancels. A "strict dominance" redraw is one where the wrapper used fewer
        # calls AND realized no more risk than the control on that same redraw.
        control_calls_series = [r["calls"] for r in control]
        control_risk_series = [r["risk"] for r in control]
        for arm in arms:
            if arm.startswith("racer_marginal"):
                continue
            arm_calls = entry["arms"][arm]["per_redraw_calls"]
            arm_risk = entry["arms"][arm]["per_redraw_risk"]
            fewer_calls = 0
            no_more_risk = 0
            strict_dominance = 0
            for ac, ar, cc, cr in zip(arm_calls, arm_risk, control_calls_series, control_risk_series):
                if ac < cc - 1e-12:
                    fewer_calls += 1
                if ar <= cr + 1e-12:
                    no_more_risk += 1
                if ac < cc - 1e-12 and ar <= cr + 1e-12:
                    strict_dominance += 1
            n = len(arm_calls)
            entry["arms"][arm]["paired_vs_control"] = {
                "n_redraws": n,
                "fewer_calls": fewer_calls,
                "no_more_risk": no_more_risk,
                "strict_dominance": strict_dominance,
                "strict_dominance_fraction": strict_dominance / n if n else None,
                "mean_call_delta": float(np.mean(np.asarray(arm_calls) - np.asarray(control_calls_series))),
                "mean_risk_delta": float(np.mean(np.asarray(arm_risk) - np.asarray(control_risk_series))),
            }
        summary[akey] = entry

    output = {
        "benchmark": arguments.benchmark,
        "protocol": "routereval",
        "alignment": "exact_by_construction (single leaderboard key)",
        "n_queries": int(n_queries),
        "pool_size": int(len(columns)),
        "n_models_full": int(n_models_full),
        "complete_pool_miss": complete_pool_miss,
        "top1_model_miss": top1_miss,
        "redraws": arguments.redraws,
        "knn_k": arguments.knn_k,
        "wrapper_max_depth": arguments.max_depth,
        "alphas": alphas,
        "partition_fractions": {
            "design": arguments.design_fraction,
            "router": arguments.router_fraction,
            "cal_and_test": "remainder split in half",
        },
        "summary": summary,
    }
    out_path = Path(arguments.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(output, indent=2))
    tmp.replace(out_path)

    print()
    for alpha in alphas:
        akey = f"{alpha}"
        entry = summary[akey]
        print(f"alpha {alpha}: control (RACER-G) {entry['control_calls_mean']:.3f} calls, "
              f"risk {entry['control_risk_mean']:.4f} [{entry['control_verdict']}]")
        for arm in arms:
            if arm == "racer_marginal_G":
                continue
            info = entry["arms"][arm]
            reduction = info["call_reduction_vs_control"]
            reduction_text = f"{100 * reduction:+.1f} pct" if reduction is not None else "n/a"
            print(f"    {arm:<28s} calls {info['calls_mean']:.3f}  risk {info['risk_mean']:.4f}  "
                  f"[{info['verdict']}]  reduction {reduction_text}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
