"""E3 unified selection with a frozen pre-calibration winner.

All candidate identities and temporary operating points are chosen from design
and validation data. The winning method is frozen before final calibration. The
final calibration labels are read exactly once for that winner, then the policy
is evaluated on test. Every policy class uses standard real-model all-miss loss,
where empty sets and full-pool failures are misses. RACER's augmented-null loss
is diagnostic only. RACER-G and RACER-P remain separate controls and are never
selected over one another. If no candidate certifies on validation, E3 freezes
the deterministic lowest-risk fallback and still gives only final calibration the
power to certify or reject the selected policy.

The query-level menu uses honest public names. ``factorized_independence`` is the
exact no-dependence coverage control. ``pairwise_heuristic`` and
``correlation_heuristic`` are pairwise ranking heuristics. The old
``gaussian_copula`` query-level label is accepted only as a read alias for
``regularized_correlation`` because this implementation does not evaluate
Gaussian orthant probabilities.
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

from components.racer_protocol import (  # noqa: E402
    augment_ground_truth,
    augment_router_scores,
    nonconformity_scores,
    prediction_sets,
    select_standard_crc_threshold,
    smooth_nonconformity_scores,
)
from components.racer_wrapper import (  # noqa: E402
    CORRELATION_HEURISTIC,
    FACTORIZED_INDEPENDENCE,
    PAIRWISE_HEURISTIC,
    REGULARIZED_CORRELATION,
    QUERY_LOCAL_EMPIRICAL_COVERAGE,
    FIXED_TALLY_FACTORIZED,
    augment_with_racer_null,
    sequential_scores,
)
from components.scale_selectors import build_order  # noqa: E402
from scripts.design_bprime_e2_run import (  # noqa: E402
    CERTIFIED_ALIGNED,
    _fallback_validation_operating_point,
    _knn_probabilities,
    _load_aligned,
    _routing_metrics,
    WRAPPER_ARM_METADATA,
)
from scripts.frontier_fourway import _calibrated_depth, _cumulative_miss  # noqa: E402

FIXED_PREFIX_ARMS = (
    "marginal", "empirical_tally", "error_correlation", "gaussian_copula",
    "student_t_copula", "factor_copula", "bernoulli_mixture", "rank_correlation",
    "lineage", "shrinkage_indep", "shrinkage_factor",
)
RACER_CONTROLS = ("racer_G", "racer_P")

WRAPPER_ARMS = (
    FACTORIZED_INDEPENDENCE,
    PAIRWISE_HEURISTIC,
    CORRELATION_HEURISTIC,
    REGULARIZED_CORRELATION,
    QUERY_LOCAL_EMPIRICAL_COVERAGE,
    FIXED_TALLY_FACTORIZED,
)
CLASS_PENALTY_MARGIN = 0.05


def _prepare_fixed_prefix_candidate(
    *,
    arm: str,
    design: np.ndarray,
    validation: np.ndarray,
    names: list[str],
    max_depth: int,
    seed: int,
    alpha: float,
) -> dict:
    """Fit and score a fixed-prefix candidate using no final calibration labels."""

    order = list(build_order(
        arm, design, model_names=names, max_depth=max_depth,
        seed=seed, n_draws=2000,
    ))
    validation_miss = _cumulative_miss(validation, order)
    temporary_depth = _calibrated_depth(validation_miss, alpha)
    validation_certified = temporary_depth is not None
    if validation_certified:
        validation_depth = int(temporary_depth)
    else:
        validation_depth = min(
            range(1, validation_miss.shape[1] + 1),
            key=lambda depth: (
                float(validation_miss[:, depth - 1].mean()),
                depth,
            ),
        )
    val_calls = float(validation_depth)
    val_standard_all_miss_risk = float(
        validation_miss[:, validation_depth - 1].mean()
    )
    return {
        "candidate": f"fixed:{arm}",
        "policy_class": "fixed_prefix",
        "variant": arm,
        "validation_operating_point": float(validation_depth),
        "validation_certified": validation_certified,
        "validation_fallback": not validation_certified,
        "fallback": not validation_certified,
        "validation_non_certification": not validation_certified,
        "validation_non_certification_reason": (
            None if validation_certified
            else "no_validation_depth_satisfies_corrected_bound"
        ),
        "val_calls": val_calls,
        "val_risk": val_standard_all_miss_risk,
        "val_standard_all_miss_risk": val_standard_all_miss_risk,
        "val_safe_action_risk": val_standard_all_miss_risk,
        "val_augmented_loss": None,
        "validation_empty_set_rate": None,
        "primary_risk_name": "standard_all_miss_risk",
        "selection_source": "design_and_validation_only",
        "_order": order,
    }


def _prepare_wrapper_candidate(
    *,
    arm: str,
    design_labels: np.ndarray,
    probabilities: np.ndarray,
    neighbor_correctness: np.ndarray,
    fixed_tally_order: tuple[int, ...],
    augmented_probabilities: np.ndarray,
    standard_labels: np.ndarray,
    augmented_labels: np.ndarray,
    val_idx: np.ndarray,
    alpha: float,
    seed: int,
    split_seed: int,
    max_depth: int,
) -> dict:
    """Fit and score a query-level candidate using validation only."""

    raw = sequential_scores(
        probabilities,
        dependence=arm,
        design=(
            design_labels
            if arm in (PAIRWISE_HEURISTIC, CORRELATION_HEURISTIC, REGULARIZED_CORRELATION)
            else None
        ),
        max_depth=max_depth,
        neighbor_correctness=(
            neighbor_correctness if arm == QUERY_LOCAL_EMPIRICAL_COVERAGE else None
        ),
        fixed_order=(fixed_tally_order if arm == FIXED_TALLY_FACTORIZED else None),
    )
    scores = smooth_nonconformity_scores(
        augment_with_racer_null(probabilities, raw),
        seed=seed,
        split_seed=split_seed,
    )
    temporary = select_standard_crc_threshold(
        scores[val_idx, :-1], standard_labels[val_idx], alpha
    )
    if temporary.feasible:
        validation_threshold = float(temporary.threshold)
        routing = prediction_sets(
            augmented_probabilities[val_idx], scores[val_idx], validation_threshold
        )
        metrics = _routing_metrics(
            routing, standard_labels[val_idx], augmented_labels[val_idx]
        )
    else:
        validation_threshold, metrics = _fallback_validation_operating_point(
            scores=scores,
            augmented_probabilities=augmented_probabilities,
            standard_labels=standard_labels,
            augmented_labels=augmented_labels,
            val_idx=val_idx,
        )
    base = {
        "candidate": f"wrapper:{arm}",
        "policy_class": "per_query_wrapper",
        "variant": arm,
        "policy_metadata": {
            **WRAPPER_ARM_METADATA[arm],
            "knn_support_size": int(neighbor_correctness.shape[1]),
        },
        "validation_operating_point": validation_threshold,
        "validation_certified": bool(temporary.feasible),
        "validation_fallback": not bool(temporary.feasible),
        "fallback": not bool(temporary.feasible),
        "validation_non_certification": not bool(temporary.feasible),
        "validation_non_certification_reason": temporary.non_certification_reason,
        "validation_crc_bound": float(temporary.crc_bound),
        "primary_risk_name": "standard_all_miss_risk",
        "selection_source": "design_and_validation_only",
        "_scores": scores,
    }
    return {
        **base,
        "val_calls": metrics["calls"],
        "val_risk": metrics["standard_all_miss_risk"],
        "val_standard_all_miss_risk": metrics["standard_all_miss_risk"],
        "val_safe_action_risk": metrics["standard_all_miss_risk"],
        "val_augmented_loss": metrics["published_augmented_null_loss"],
        "validation_empty_set_rate": metrics["empty_set_rate"],
    }


def _noncertification_candidate(reason: str) -> dict:
    """Return an explicit no-policy record for failed validation certification."""

    return {
        "candidate": "non_certification",
        "policy_class": "non_certification",
        "variant": "non_certification",
        "validation_operating_point": None,
        "validation_certified": False,
        "validation_non_certification": True,
        "val_calls": None,
        "val_risk": None,
        "val_standard_all_miss_risk": None,
        "val_safe_action_risk": None,
        "val_augmented_loss": None,
        "validation_empty_set_rate": None,
        "primary_risk_name": "standard_all_miss_risk",
        "selection_source": "explicit_non_certification",
        "non_certification_reason": reason,
    }


def _select(candidates: list[dict], alpha: float, class_penalty: bool) -> dict:
    """Prefer certified candidates, otherwise select the lowest-risk fallback."""

    del alpha  # Certification status already encodes the corrected bound.
    if not candidates:
        raise ValueError("at least one validation candidate is required")
    certified = [
        candidate for candidate in candidates
        if candidate.get("validation_certified", True)
    ]
    fallback_used = not bool(certified)
    if certified:
        pool = sorted(
            certified,
            key=lambda candidate: (
                candidate["val_calls"],
                candidate["val_risk"],
                candidate["candidate"],
            ),
        )
    else:
        fallbacks = [
            candidate for candidate in candidates
            if candidate.get("validation_fallback", candidate.get("fallback", False))
        ]
        if not fallbacks:
            raise ValueError("validation candidates have neither certificates nor fallbacks")
        pool = sorted(
            fallbacks,
            key=lambda candidate: (
                candidate["val_risk"],
                candidate["val_calls"],
                candidate["candidate"],
            ),
        )
    best = pool[0]
    tie_break_applied = False
    if not fallback_used and class_penalty and best["policy_class"] == "per_query_wrapper":
        threshold = best["val_calls"] * (1.0 + CLASS_PENALTY_MARGIN)
        simpler = [
            candidate for candidate in pool
            if candidate["policy_class"] == "fixed_prefix"
            and candidate["val_calls"] <= threshold + 1e-12
        ]
        if simpler:
            best = min(
                simpler,
                key=lambda candidate: (
                    candidate["val_calls"], candidate["val_risk"], candidate["candidate"]
                ),
            )
            tie_break_applied = True
    return {
        "selected": best,
        "fallback_used": fallback_used,
        "fallback_reason": (
            "no_validation_certified_candidate" if fallback_used else None
        ),
        "selection_non_certification": fallback_used,
        "class_penalty_applied": tie_break_applied,
    }


def _finalize_noncertification(candidate: dict, *, reason: str) -> dict:
    """Finalize an explicit no-policy result without a risk or call claim."""

    public = {key: value for key, value in candidate.items() if not key.startswith("_")}
    return {
        **public,
        "final_operating_point": None,
        "test_calls": None,
        "model_calls": None,
        "test_risk": None,
        "standard_all_miss_risk": None,
        "safe_action_risk": None,
        "published_augmented_null_loss": None,
        "augmented_loss": None,
        "empty_set_rate": None,
        "abstention_rate": None,
        "certified": False,
        "final_certified": False,
        "non_certification": True,
        "non_certification_reason": reason,
        "risk_validity_claimed": False,
        "final_calibration_count": 0,
    }


def _finalize_fixed_prefix(
    *,
    candidate: dict,
    calibration_labels: np.ndarray,
    test_labels: np.ndarray,
    alpha: float,
) -> dict:
    """Calibrate one frozen fixed prefix once, then evaluate test once."""

    order = candidate["_order"]
    depth = _calibrated_depth(_cumulative_miss(calibration_labels, order), alpha)
    public = {key: value for key, value in candidate.items() if not key.startswith("_")}
    if depth is None:
        return {
            **public,
            "final_operating_point": None,
            "test_calls": None,
            "model_calls": None,
            "test_risk": None,
            "standard_all_miss_risk": None,
            "safe_action_risk": None,
            "published_augmented_null_loss": None,
            "augmented_loss": None,
            "empty_set_rate": None,
            "abstention_rate": None,
            "certified": False,
            "final_certified": False,
            "non_certification": True,
            "non_certification_reason": "final_calibration_non_certification",
            "risk_validity_claimed": False,
            "final_calibration_count": 1,
        }
    test_miss = _cumulative_miss(test_labels, order)
    risk = float(test_miss[:, depth - 1].mean())
    return {
        **public,
        "primary_risk_name": "standard_all_miss_risk",
        "final_operating_point": float(depth),
        "test_calls": float(depth),
        "model_calls": float(depth),
        "test_risk": risk,
        "standard_all_miss_risk": risk,
        "safe_action_risk": risk,
        "published_augmented_null_loss": None,
        "augmented_loss": None,
        "empty_set_rate": 0.0,
        "abstention_rate": 0.0,
        "certified": True,
        "final_certified": True,
        "non_certification": False,
        "non_certification_reason": None,
        "risk_validity_claimed": True,
        "final_calibration_count": 1,
    }


def _finalize_wrapper(
    *,
    candidate: dict,
    augmented_probabilities: np.ndarray,
    standard_labels: np.ndarray,
    augmented_labels: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict:
    """Calibrate one frozen query-level score once, then evaluate test once."""

    scores = candidate["_scores"]
    final = select_standard_crc_threshold(
        scores[cal_idx, :-1], standard_labels[cal_idx], alpha
    )
    public = {key: value for key, value in candidate.items() if not key.startswith("_")}
    base = {
        **public,
        "primary_risk_name": "standard_all_miss_risk",
        "final_operating_point": float(final.threshold) if final.feasible else None,
        "certified": bool(final.feasible),
        "final_certified": bool(final.feasible),
        "non_certification": not bool(final.feasible),
        "non_certification_reason": final.non_certification_reason,
        "risk_validity_claimed": bool(final.feasible),
        "final_calibration_count": 1,
        "calibration_misses": int(final.calibration_misses),
        "unavoidable_calibration_misses": int(final.unavoidable_misses),
        "crc_bound": float(final.crc_bound),
    }
    if not final.feasible:
        return {
            **base,
            "test_calls": None,
            "model_calls": None,
            "test_risk": None,
            "standard_all_miss_risk": None,
            "safe_action_risk": None,
            "published_augmented_null_loss": None,
            "augmented_loss": None,
            "empty_set_rate": None,
            "abstention_rate": None,
        }
    routing = prediction_sets(
        augmented_probabilities[test_idx], scores[test_idx], final.threshold
    )
    metrics = _routing_metrics(
        routing, standard_labels[test_idx], augmented_labels[test_idx]
    )
    return {
        **base,
        "test_calls": metrics["calls"],
        "model_calls": metrics["calls"],
        "test_risk": metrics["standard_all_miss_risk"],
        **metrics,
    }


def _evaluate_racer_control(
    *,
    variant: str,
    probabilities: np.ndarray,
    augmented_probabilities: np.ndarray,
    standard_labels: np.ndarray,
    augmented_labels: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
    seed: int,
    split_seed: int,
) -> dict:
    """Calibrate and report one predeclared RACER control independently."""

    _, scores = nonconformity_scores(
        probabilities, variant, seed=seed, split_seed=split_seed
    )
    final = select_standard_crc_threshold(
        scores[cal_idx, :-1], standard_labels[cal_idx], alpha
    )
    base = {
        "candidate": f"racer_{variant}",
        "policy_class": "racer_control",
        "variant": f"racer_{variant}",
        "primary_risk_name": "standard_all_miss_risk",
        "final_operating_point": float(final.threshold) if final.feasible else None,
        "validation_certified": None,
        "validation_non_certification": None,
        "certified": bool(final.feasible),
        "final_certified": bool(final.feasible),
        "non_certification": not bool(final.feasible),
        "non_certification_reason": final.non_certification_reason,
        "risk_validity_claimed": bool(final.feasible),
        "final_calibration_count": 1,
        "calibration_misses": int(final.calibration_misses),
        "unavoidable_calibration_misses": int(final.unavoidable_misses),
        "crc_bound": float(final.crc_bound),
    }
    if not final.feasible:
        return {
            **base,
            "test_calls": None,
            "model_calls": None,
            "test_risk": None,
            "standard_all_miss_risk": None,
            "safe_action_risk": None,
            "published_augmented_null_loss": None,
            "augmented_loss": None,
            "empty_set_rate": None,
            "abstention_rate": None,
        }
    routing = prediction_sets(
        augmented_probabilities[test_idx], scores[test_idx], final.threshold
    )
    metrics = _routing_metrics(
        routing, standard_labels[test_idx], augmented_labels[test_idx]
    )
    return {
        **base,
        "test_calls": metrics["calls"],
        "model_calls": metrics["calls"],
        "test_risk": metrics["standard_all_miss_risk"],
        **metrics,
    }


def _finalize_predeclared_fixed_baseline(
    *,
    candidates: list[dict],
    candidate_id: str,
    variant: str,
    calibration_labels: np.ndarray,
    test_labels: np.ndarray,
    alpha: float,
) -> dict:
    """Always emit a record for a predeclared fixed-prefix baseline."""

    candidate_by_id = {
        candidate["candidate"]: candidate for candidate in candidates
    }
    candidate = candidate_by_id.get(candidate_id)
    if candidate is None:
        missing = {
            **_noncertification_candidate("predeclared_candidate_record_missing"),
            "candidate": candidate_id,
            "policy_class": "fixed_prefix",
            "variant": variant,
        }
        return _finalize_noncertification(
            missing, reason="predeclared_candidate_record_missing"
        )
    return _finalize_fixed_prefix(
        candidate=candidate,
        calibration_labels=calibration_labels,
        test_labels=test_labels,
        alpha=alpha,
    )


def main() -> int:
    run_started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--benchmark", default="ifeval")
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=100)
    parser.add_argument("--alphas", default="0.05,0.10")
    parser.add_argument("--redraws", type=int, default=20)
    parser.add_argument("--knn-k", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--design-fraction", type=float, default=0.30)
    parser.add_argument("--router-fraction", type=float, default=0.25)
    parser.add_argument("--no-class-penalty", action="store_true")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    if arguments.benchmark not in CERTIFIED_ALIGNED:
        raise SystemExit(f"benchmark {arguments.benchmark!r} has no certified alignment")
    binary, embeddings = _load_aligned(
        Path(arguments.router_dir), Path(arguments.leaderboard),
        arguments.benchmark, arguments.threshold,
    )
    n_queries, n_models_full = binary.shape
    alphas = [float(value) for value in arguments.alphas.split(",") if value]
    class_penalty = not arguments.no_class_penalty
    generator = np.random.default_rng(arguments.seed)
    columns = np.sort(generator.choice(
        n_models_full, size=min(arguments.pool_size, n_models_full), replace=False
    ))
    correctness = binary[:, columns]
    names = [str(index) for index in range(len(columns))]
    complete_pool_miss = float((correctness.sum(axis=1) == 0).mean())
    n_design = int(round(n_queries * arguments.design_fraction))
    n_router = int(round(n_queries * arguments.router_fraction))
    per_alpha = {f"{alpha}": [] for alpha in alphas}

    for redraw in range(arguments.redraws):
        rng = np.random.default_rng(arguments.seed + 13 * redraw)
        permutation = rng.permutation(n_queries)
        design_rows = permutation[:n_design]
        router_rows = permutation[n_design:n_design + n_router]
        pool_rows = permutation[n_design + n_router:]
        design_labels = correctness[design_rows].astype(int)
        probabilities, neighbor_correctness = _knn_probabilities(
            embeddings[router_rows], correctness[router_rows],
            embeddings[pool_rows], arguments.knn_k, return_support=True,
        )
        standard_labels = correctness[pool_rows].astype(bool)
        augmented_probabilities = augment_router_scores(probabilities)
        augmented_labels = augment_ground_truth(standard_labels)
        fixed_tally_order = build_order(
            "empirical_tally",
            design_labels,
            model_names=names,
            max_depth=arguments.max_depth,
        )

        n_pool = len(pool_rows)
        n_cal = int(n_pool * 0.5)
        n_val = int(n_pool * 0.25)
        cal_idx = np.arange(n_cal)
        val_idx = np.arange(n_cal, n_cal + n_val)
        test_idx = np.arange(n_cal + n_val, n_pool)
        calibration_labels = standard_labels[cal_idx]
        validation_labels = standard_labels[val_idx]
        test_labels = standard_labels[test_idx]

        for alpha in alphas:
            candidates = []
            for arm in FIXED_PREFIX_ARMS:
                candidate = _prepare_fixed_prefix_candidate(
                    arm=arm,
                    design=design_labels,
                    validation=validation_labels,
                    names=names,
                    max_depth=arguments.max_depth,
                    seed=arguments.seed + redraw,
                    alpha=alpha,
                )
                candidates.append(candidate)
            for arm in WRAPPER_ARMS:
                candidate = _prepare_wrapper_candidate(
                    arm=arm,
                    design_labels=design_labels,
                    probabilities=probabilities,
                    neighbor_correctness=neighbor_correctness,
                    fixed_tally_order=fixed_tally_order,
                    augmented_probabilities=augmented_probabilities,
                    standard_labels=standard_labels,
                    augmented_labels=augmented_labels,
                    val_idx=val_idx,
                    alpha=alpha,
                    seed=arguments.seed,
                    split_seed=redraw + 1,
                    max_depth=arguments.max_depth,
                )
                candidates.append(candidate)

            choice = _select(candidates, alpha, class_penalty)
            frozen = choice["selected"]
            if frozen["policy_class"] == "fixed_prefix":
                selected = _finalize_fixed_prefix(
                    candidate=frozen,
                    calibration_labels=calibration_labels,
                    test_labels=test_labels,
                    alpha=alpha,
                )
            elif frozen["policy_class"] == "per_query_wrapper":
                selected = _finalize_wrapper(
                    candidate=frozen,
                    augmented_probabilities=augmented_probabilities,
                    standard_labels=standard_labels,
                    augmented_labels=augmented_labels,
                    cal_idx=cal_idx,
                    test_idx=test_idx,
                    alpha=alpha,
                )
            elif frozen["policy_class"] == "non_certification":
                selected = _finalize_noncertification(
                    frozen, reason=choice["fallback_reason"]
                )
            else:
                raise AssertionError(f"unknown policy class {frozen['policy_class']!r}")

            controls = {
                f"racer_{variant}": _evaluate_racer_control(
                    variant=variant,
                    probabilities=probabilities,
                    augmented_probabilities=augmented_probabilities,
                    standard_labels=standard_labels,
                    augmented_labels=augmented_labels,
                    cal_idx=cal_idx,
                    test_idx=test_idx,
                    alpha=alpha,
                    seed=arguments.seed,
                    split_seed=redraw + 1,
                )
                for variant in ("G", "P")
            }
            marginal = _finalize_predeclared_fixed_baseline(
                candidates=candidates,
                candidate_id="fixed:marginal",
                variant="marginal",
                calibration_labels=calibration_labels,
                test_labels=test_labels,
                alpha=alpha,
            )
            per_alpha[f"{alpha}"].append({
                "redraw": redraw,
                "selected": selected,
                "selected_identity_before_final_calibration": frozen["candidate"],
                "fallback_used": choice["fallback_used"],
                "fallback_reason": choice["fallback_reason"],
                "selection_non_certification": choice["selection_non_certification"],
                "selection_validation_certified": bool(
                    frozen.get("validation_certified", False)
                ),
                "risk_validity_claimed": selected["risk_validity_claimed"],
                "class_penalty_applied": choice["class_penalty_applied"],
                "baseline_fixed_marginal": marginal,
                "racer_controls": controls,
                "validation_candidates": [
                    {key: value for key, value in candidate.items() if not key.startswith("_")}
                    for candidate in candidates
                ],
                "n_candidates": len(candidates),
            })

    aggregate = {}
    for alpha in alphas:
        records = per_alpha[f"{alpha}"]
        if not records:
            aggregate[f"{alpha}"] = {"n_redraws": 0}
            continue
        certified_selected = [
            record["selected"] for record in records
            if record["selected"]["final_certified"]
        ]
        selected_calls = [value["test_calls"] for value in certified_selected]
        selected_standard = [
            value["standard_all_miss_risk"] for value in certified_selected
        ]
        classes = [record["selected"]["policy_class"] for record in records]
        variants = [record["selected"]["variant"] for record in records]
        controls = {}
        for control_name in RACER_CONTROLS:
            values = [record["racer_controls"][control_name] for record in records]
            certified_values = [value for value in values if value["certified"]]
            calls = [value["test_calls"] for value in certified_values]
            standard = [
                value["standard_all_miss_risk"] for value in certified_values
            ]
            augmented = [
                value["published_augmented_null_loss"] for value in certified_values
            ]
            controls[control_name] = {
                "primary_risk_name": "standard_all_miss_risk",
                "calls_mean": float(np.mean(calls)) if calls else None,
                "standard_all_miss_risk_mean": (
                    float(np.mean(standard)) if standard else None
                ),
                "safe_action_risk_mean": (
                    float(np.mean(standard)) if standard else None
                ),
                "published_augmented_null_loss_mean": (
                    float(np.mean(augmented)) if augmented else None
                ),
                "augmented_loss_mean": (
                    float(np.mean(augmented)) if augmented else None
                ),
                "certification_rate": float(np.mean([
                    value["certified"] for value in values
                ])),
                "non_certification_count": int(sum(
                    value["non_certification"] for value in values
                )),
            }
        aggregate[f"{alpha}"] = {
            "n_redraws": len(records),
            "primary_risk_name": "standard_all_miss_risk",
            "selected_test_calls_mean": (
                float(np.mean(selected_calls)) if selected_calls else None
            ),
            "selected_standard_all_miss_risk_mean": (
                float(np.mean(selected_standard)) if selected_standard else None
            ),
            "selected_safe_action_risk_mean": (
                float(np.mean(selected_standard)) if selected_standard else None
            ),
            "selected_certification_rate": float(np.mean([
                record["selected"]["final_certified"] for record in records
            ])),
            "selected_non_certification_count": int(sum(
                record["selected"]["non_certification"] for record in records
            )),
            "selected_risk_validity_claimed_rate": float(np.mean([
                record["selected"]["risk_validity_claimed"] for record in records
            ])),
            "class_win_counts": {
                "fixed_prefix": classes.count("fixed_prefix"),
                "per_query_wrapper": classes.count("per_query_wrapper"),
                "non_certification": classes.count("non_certification"),
            },
            "variant_win_counts": {variant: variants.count(variant) for variant in sorted(set(variants))},
            "fallback_fraction": float(np.mean([record["fallback_used"] for record in records])),
            "class_penalty_applied_fraction": float(np.mean([
                record["class_penalty_applied"] for record in records
            ])),
            "racer_controls": controls,
        }

    output = {
        "design": "E3_unified_frozen_selection",
        "machine": os.uname().nodename,
        "machine_alias": os.environ.get("DACR_MACHINE_ALIAS", ""),
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_protocol": "design_and_validation_then_freeze_then_one_final_calibration_then_test",
        "primary_risk_name": "standard_all_miss_risk",
        "published_racer_diagnostic_name": "published_augmented_null_loss",
        "protocol": "routereval",
        "benchmark": arguments.benchmark,
        "pool_size": int(len(columns)),
        "n_models_full": int(n_models_full),
        "complete_pool_miss": complete_pool_miss,
        "redraws": arguments.redraws,
        "alphas": alphas,
        "class_penalty_enabled": class_penalty,
        "class_penalty_margin": CLASS_PENALTY_MARGIN,
        "fixed_prefix_arms": list(FIXED_PREFIX_ARMS),
        "wrapper_arms": list(WRAPPER_ARMS),
        "wrapper_arm_metadata": WRAPPER_ARM_METADATA,
        "racer_controls": list(RACER_CONTROLS),
        "n_candidates_nominal": len(FIXED_PREFIX_ARMS) + len(WRAPPER_ARMS),
        "per_alpha_records": per_alpha,
        "aggregate": aggregate,
    }
    out_path = Path(arguments.out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite immutable result {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, default=str))
    temporary.replace(out_path)

    for alpha in alphas:
        entry = aggregate[f"{alpha}"]
        if not entry.get("n_redraws"):
            print(f"alpha {alpha}: no validation-feasible candidate")
            continue
        if entry["selected_test_calls_mean"] is None:
            print(f"alpha {alpha}: selected policy NON-CERTIFIED")
        else:
            print(
                f"alpha {alpha}: selected {entry['selected_test_calls_mean']:.3f} calls, "
                f"standard miss "
                f"{entry['selected_standard_all_miss_risk_mean']:.4f}, "
                f"cert {entry['selected_certification_rate']:.3f}"
            )
        for name, control in entry["racer_controls"].items():
            if control["calls_mean"] is None:
                print(f"  {name}: NON-CERTIFIED")
                continue
            print(
                f"  {name}: {control['calls_mean']:.3f} calls, "
                f"standard miss {control['standard_all_miss_risk_mean']:.4f}, "
                f"published augmented-null diagnostic "
                f"{control['published_augmented_null_loss_mean']:.4f}"
            )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
