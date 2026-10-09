"""Design B prime E2 with frozen selection and one final calibration.

The query-level arms are exact factorized independence coverage, two explicitly
named pairwise heuristics, and regularized correlation. The old query-level name
``gaussian_copula`` remains a read alias in ``components.racer_wrapper`` but is
never emitted by this runner because no Gaussian orthant probability is used.

For every redraw and target level, this runner follows one data flow:

1. Fit router and heuristic statistics using design-side data.
2. Select each arm's hyperparameter using validation labels only.
3. Freeze the arm and discard its temporary validation threshold.
4. Calibrate the frozen arm exactly once on untouched final calibration labels.
5. Evaluate that final policy once on test labels.

RACER-G and RACER-P are separate predeclared controls. Neither is selected over
the other. Every arm calibrates real-model scores against standard all-miss
loss. Empty sets and rows unanswerable by the full real pool are misses. The
published augmented-null loss is retained only as a separately named diagnostic.
A non-certified arm has no deployed operating point, call count, or risk claim.
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
    augmented_misrouting_loss,
    augment_ground_truth,
    augment_router_scores,
    nonconformity_scores,
    prediction_sets,
    select_standard_crc_threshold,
    smooth_nonconformity_scores,
    standard_all_miss_loss,
)
from components.racer_wrapper import (  # noqa: E402
    CORRELATION_HEURISTIC,
    FACTORIZED_INDEPENDENCE,
    FIXED_DESIGN_MARGINAL_FACTORIZED,
    FIXED_TALLY_FACTORIZED,
    LOCAL_ORDER_FACTORIZED,
    PAIRWISE_HEURISTIC,
    QUERY_LOCAL_EMPIRICAL_COVERAGE,
    REGULARIZED_CORRELATION,
    augment_with_racer_null,
    local_order_factorized_scores,
    per_query_score_diagnostics,
    query_local_empirical_scores,
    sequential_scores,
)
from components.scale_selectors import build_order  # noqa: E402

RACER_CONTROLS = ("racer_G", "racer_P")

WRAPPER_ARMS = (
    FACTORIZED_INDEPENDENCE,
    PAIRWISE_HEURISTIC,
    CORRELATION_HEURISTIC,
    REGULARIZED_CORRELATION,
    QUERY_LOCAL_EMPIRICAL_COVERAGE,
    LOCAL_ORDER_FACTORIZED,
    FIXED_TALLY_FACTORIZED,
    FIXED_DESIGN_MARGINAL_FACTORIZED,
)

WRAPPER_HYPER_GRIDS: dict[str, list[dict]] = {
    FACTORIZED_INDEPENDENCE: [{}],
    PAIRWISE_HEURISTIC: [
        {"adjustment_strength": 0.0},
        *[
            {"adjustment_strength": 1.0, "laplace_smoothing": value}
            for value in (0.0, 0.5, 1.0, 2.0)
        ],
    ],
    CORRELATION_HEURISTIC: [
        {"adjustment_strength": 0.0},
        *[
            {"adjustment_strength": 1.0, "clip_magnitude": value}
            for value in (0.90, 0.95, 0.99, 0.999)
        ],
    ],
    REGULARIZED_CORRELATION: [
        {"ridge": value} for value in (0.0, 0.01, 0.05, 0.10, 0.20, 1.0)
    ],
    QUERY_LOCAL_EMPIRICAL_COVERAGE: [{}],
    LOCAL_ORDER_FACTORIZED: [{}],
    FIXED_TALLY_FACTORIZED: [{}],
    FIXED_DESIGN_MARGINAL_FACTORIZED: [{}],
}

WRAPPER_ARM_METADATA = {
    FACTORIZED_INDEPENDENCE: {
        "ordering_source": "query_router_marginals",
        "sizing_source": "query_factorized_cumulative_coverage",
    },
    PAIRWISE_HEURISTIC: {
        "ordering_source": "query_router_marginals_plus_design_pairwise_heuristic",
        "sizing_source": "query_pairwise_residual_heuristic",
    },
    CORRELATION_HEURISTIC: {
        "ordering_source": "query_router_marginals_plus_design_correlation_heuristic",
        "sizing_source": "query_correlation_residual_heuristic",
    },
    REGULARIZED_CORRELATION: {
        "ordering_source": "query_router_marginals_plus_design_regularized_correlation",
        "sizing_source": "query_regularized_correlation_residual_heuristic",
    },
    QUERY_LOCAL_EMPIRICAL_COVERAGE: {
        "ordering_source": "query_local_frozen_knn_empirical_joint",
        "sizing_source": "query_local_predecessor_empirical_coverage",
        "neighbor_weighting": "uniform",
    },
    LOCAL_ORDER_FACTORIZED: {
        "ordering_source": "query_local_frozen_knn_empirical_joint",
        "sizing_source": "query_factorized_cumulative_coverage",
        "neighbor_weighting": "uniform",
    },
    FIXED_TALLY_FACTORIZED: {
        "ordering_source": "design_empirical_tally_fixed",
        "sizing_source": "query_factorized_cumulative_coverage",
    },
    FIXED_DESIGN_MARGINAL_FACTORIZED: {
        "ordering_source": "design_marginal_accuracy_fixed",
        "sizing_source": "query_factorized_cumulative_coverage",
    },
}


def _parse_neighbor_k_grid(value: str | None, primary_k: int) -> tuple[int, ...]:
    """Return unique positive neighbor sizes with the primary setting first."""

    if primary_k < 1:
        raise ValueError("primary neighbor k must be positive")
    parsed = []
    if value:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            candidate = int(token)
            if candidate < 1:
                raise ValueError("neighbor-k grid values must be positive")
            if candidate not in parsed:
                parsed.append(candidate)
    return tuple(
        [primary_k, *[candidate for candidate in parsed if candidate != primary_k]]
    )


DESIGN_DEPENDENT_ARMS = (
    PAIRWISE_HEURISTIC,
    CORRELATION_HEURISTIC,
    REGULARIZED_CORRELATION,
    FIXED_TALLY_FACTORIZED,
    FIXED_DESIGN_MARGINAL_FACTORIZED,
)

DESIGN_PERMUTATION_OMISSIONS = {
    FACTORIZED_INDEPENDENCE: (
        "omitted because exact independence has no shuffled-design dependence"
    ),
    QUERY_LOCAL_EMPIRICAL_COVERAGE: (
        "omitted because query-local empirical coverage uses frozen router-neighbor "
        "support, not design labels"
    ),
    LOCAL_ORDER_FACTORIZED: (
        "omitted because local ordering uses frozen router-neighbor support, not "
        "design labels"
    ),
}


def _shuffle_design_columns(design_labels: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute each design column while preserving its marginal."""

    shuffled = np.array(design_labels, copy=True)
    generator = np.random.default_rng(seed)
    for column in range(shuffled.shape[1]):
        generator.shuffle(shuffled[:, column])
    return shuffled


def _design_permutation_seed(base_seed: int, redraw: int, permutation_index: int) -> int:
    """Return a deterministic seed isolated from the main experiment random stream."""

    sequence = np.random.SeedSequence([base_seed, redraw, permutation_index, 0xD3516E])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])

# Benchmarks whose feature rows and correctness rows were verified to align.
CERTIFIED_ALIGNED = {
    "ifeval": {"router_name": "ifeval", "leaderboard_keys": ("ifeval",)},
    "bbh": {"router_name": "bbh", "leaderboard_keys": (
        "bbh_boolean_expressions", "bbh_causal_judgement", "bbh_date_understanding",
        "bbh_disambiguation_qa", "bbh_formal_fallacies", "bbh_geometric_shapes",
        "bbh_hyperbaton", "bbh_logical_deduction_five_objects",
        "bbh_logical_deduction_seven_objects", "bbh_logical_deduction_three_objects",
        "bbh_movie_recommendation", "bbh_navigate", "bbh_object_counting",
        "bbh_penguins_in_a_table", "bbh_reasoning_about_colored_objects",
        "bbh_ruin_names", "bbh_salient_translation_error_detection", "bbh_snarks",
        "bbh_sports_understanding", "bbh_temporal_sequences",
        "bbh_tracking_shuffled_objects_five_objects",
        "bbh_tracking_shuffled_objects_seven_objects",
        "bbh_tracking_shuffled_objects_three_objects", "bbh_web_of_lies",
    )},
    "gpqa": {"router_name": "gpqa", "leaderboard_keys": (
        "gpqa_diamond", "gpqa_extended", "gpqa_main",
    )},
    "math": {"router_name": "math", "leaderboard_keys": (
        "math_algebra_hard", "math_counting_and_prob_hard", "math_geometry_hard",
        "math_intermediate_algebra_hard", "math_num_theory_hard",
        "math_prealgebra_hard", "math_precalculus_hard",
    )},
    "musr": {"router_name": "musr", "leaderboard_keys": (
        "musr_murder_mysteries", "musr_object_placements", "musr_team_allocation",
    )},
}


def _load_aligned(
    router_dir: Path,
    leaderboard_path: Path,
    benchmark: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return correctness and embeddings with exactly aligned query rows."""

    if benchmark not in CERTIFIED_ALIGNED:
        raise SystemExit(
            f"benchmark {benchmark!r} has no certified feature-to-label alignment; "
            f"certified set is {sorted(CERTIFIED_ALIGNED)}"
        )
    spec = CERTIFIED_ALIGNED[benchmark]
    with (router_dir / f"{spec['router_name']}_router_dataset.pkl").open("rb") as handle:
        router_data = pickle.load(handle)
    with leaderboard_path.open("rb") as handle:
        leaderboard = pickle.load(handle)

    correctness = np.concatenate([
        np.asarray(leaderboard["data"][key]["correctness"], dtype=float)
        for key in spec["leaderboard_keys"]
    ], axis=0)
    binary = (correctness >= threshold).astype(int)

    split = router_data["split_index"]
    embedding = router_data["embedding"]
    indices = np.concatenate([
        np.asarray(split[key], dtype=int)
        for key in ("train_indices", "val_indices", "test_indices")
    ])
    embeddings = np.concatenate([
        np.asarray(embedding[key], dtype=float)
        for key in ("train_embed", "val_embed", "test_embed")
    ], axis=0)
    if len(indices) != binary.shape[0] or len(np.unique(indices)) != len(indices):
        raise SystemExit(f"{benchmark}: router rows do not uniquely cover correctness rows")
    order = np.argsort(indices)
    if not np.array_equal(indices[order], np.arange(binary.shape[0])):
        raise SystemExit(f"{benchmark}: router indices are not exactly 0 through n minus 1")
    return binary, embeddings[order]


def _knn_probabilities(
    train_embed: np.ndarray,
    train_correct: np.ndarray,
    query_embed: np.ndarray,
    k: int,
    *,
    return_support: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return marginal predictions and optionally their exact frozen support."""

    def normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    train = normalise(train_embed)
    query = normalise(query_embed)
    k = min(int(k), train.shape[0])
    if k < 1:
        raise ValueError("knn-k must be positive")
    similarity = query @ train.T
    neighbours = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    neighbor_correctness = np.asarray(train_correct[neighbours])
    predictions = neighbor_correctness.mean(axis=1)
    clipped = np.clip(predictions, 1e-6, 1.0 - 1e-6)
    if return_support:
        return clipped, neighbor_correctness
    return clipped


def _routing_metrics(
    routing,
    standard_labels: np.ndarray,
    augmented_labels: np.ndarray,
) -> dict:
    """Return standard risk first and augmented-null diagnostics separately."""

    standard = standard_all_miss_loss(routing.standard_mask, standard_labels)
    published_augmented = augmented_misrouting_loss(
        routing.augmented_mask, augmented_labels
    )
    standard_risk = float(np.mean(standard))
    published_risk = float(np.mean(published_augmented))
    empty_rate = float(np.mean(routing.size == 0))
    return {
        "calls": float(np.mean(routing.size)),
        "standard_all_miss_risk": standard_risk,
        "risk_test": standard_risk,
        "safe_action_risk": standard_risk,
        "published_augmented_null_loss": published_risk,
        "augmented_loss": published_risk,
        "null_selection_rate": float(np.mean(routing.null_selected)),
        "empty_set_rate": empty_rate,
        "abstention_rate": empty_rate,
    }


def _fallback_validation_operating_point(
    *,
    scores: np.ndarray,
    augmented_probabilities: np.ndarray,
    standard_labels: np.ndarray,
    augmented_labels: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[float, dict]:
    """Return the lowest-risk empirical validation point deterministically.

    The corrected finite-sample bound may reject every threshold even though
    validation still has to choose a candidate identity. Scan every distinct
    real-model threshold, minimize standard all-miss risk first, then calls,
    then the threshold itself as a stable operating-point identity.
    """

    validation_scores = np.asarray(scores[val_idx, :-1], dtype=float)
    observed_thresholds = np.unique(validation_scores)
    if observed_thresholds.size == 0 or not np.all(np.isfinite(observed_thresholds)):
        raise ValueError("validation scores must contain finite real-model thresholds")
    empty_threshold = np.nextafter(observed_thresholds[0], -np.inf)
    thresholds = np.concatenate(([empty_threshold], observed_thresholds))

    operating_points = []
    for threshold in thresholds:
        routing = prediction_sets(
            augmented_probabilities[val_idx], scores[val_idx], float(threshold)
        )
        metrics = _routing_metrics(
            routing, standard_labels[val_idx], augmented_labels[val_idx]
        )
        operating_points.append((float(threshold), metrics))
    return min(
        operating_points,
        key=lambda point: (
            point[1]["standard_all_miss_risk"],
            point[1]["calls"],
            point[0],
        ),
    )


def _temporary_validation_record(
    *,
    arm_key: str,
    hyper: dict,
    scores: np.ndarray,
    augmented_probabilities: np.ndarray,
    standard_labels: np.ndarray,
    augmented_labels: np.ndarray,
    val_idx: np.ndarray,
    alpha: float,
) -> dict:
    """Build one certified or deterministic fallback validation point."""

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
    return {
        "arm_key": arm_key,
        "hyper": dict(hyper),
        "validation_threshold": validation_threshold,
        "validation_certified": bool(temporary.feasible),
        "validation_fallback": not bool(temporary.feasible),
        "fallback": not bool(temporary.feasible),
        "validation_non_certification_reason": temporary.non_certification_reason,
        "validation_crc_bound": float(temporary.crc_bound),
        "validation_unavoidable_misses": int(temporary.unavoidable_misses),
        "primary_risk_name": "standard_all_miss_risk",
        **metrics,
    }


def _select_validation_candidate(records: list[dict], alpha: float) -> dict:
    """Prefer certified candidates, otherwise choose the lowest-risk fallback."""

    del alpha  # Certification status already encodes the corrected bound.
    if not records:
        raise ValueError("at least one validation candidate is required")
    certified = [record for record in records if record["validation_certified"]]
    if certified:
        chosen = min(
            certified,
            key=lambda record: (
                record["calls"],
                record["standard_all_miss_risk"],
                record["arm_key"],
            ),
        )
        return {
            "chosen": chosen,
            "fallback_used": False,
            "selection_non_certification": False,
            "non_certification_reason": None,
        }

    fallbacks = [record for record in records if record["validation_fallback"]]
    if not fallbacks:
        raise ValueError("validation candidates have neither certificates nor fallbacks")
    chosen = min(
        fallbacks,
        key=lambda record: (
            record["standard_all_miss_risk"],
            record["calls"],
            record["arm_key"],
        ),
    )
    return {
        "chosen": chosen,
        "fallback_used": True,
        "selection_non_certification": True,
        "non_certification_reason": "no_validation_standard_risk_certificate",
    }


def _final_calibrate_and_test(
    *,
    scores: np.ndarray,
    augmented_probabilities: np.ndarray,
    standard_labels: np.ndarray,
    augmented_labels: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict:
    """Calibrate one frozen score on standard loss, then test if certified."""

    final = select_standard_crc_threshold(
        scores[cal_idx, :-1], standard_labels[cal_idx], alpha
    )
    base = {
        "primary_risk_name": "standard_all_miss_risk",
        "final_threshold": float(final.threshold) if final.feasible else None,
        "certified": bool(final.feasible),
        "non_certification": not bool(final.feasible),
        "non_certification_reason": final.non_certification_reason,
        "risk_validity_claimed": bool(final.feasible),
        "final_calibration_count": 1,
        "n_final_calibration": int(len(cal_idx)),
        "calibration_misses": int(final.calibration_misses),
        "unavoidable_calibration_misses": int(final.unavoidable_misses),
        "crc_bound": float(final.crc_bound),
    }
    if not final.feasible:
        return {
            **base,
            "calls": None,
            "calls_test": None,
            "risk_test": None,
            "standard_all_miss_risk": None,
            "safe_action_risk": None,
            "published_augmented_null_loss": None,
            "augmented_loss": None,
            "null_selection_rate": None,
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
        **metrics,
        "risk_test": metrics["standard_all_miss_risk"],
        "calls_test": metrics["calls"],
    }


def _sequential_scores_for_grid_candidate(
    probabilities: np.ndarray,
    *,
    arm: str,
    design_labels: np.ndarray,
    max_depth: int,
    hyper: dict,
    fixed_tally_order: tuple[int, ...],
) -> np.ndarray:
    """Score one grid candidate, including exact adjustment-off endpoints."""

    strength = float(hyper.get("adjustment_strength", 1.0))
    if not np.isfinite(strength) or strength not in (0.0, 1.0):
        raise ValueError("adjustment_strength must be either 0 or 1")
    if arm in (PAIRWISE_HEURISTIC, CORRELATION_HEURISTIC) and strength == 0.0:
        return sequential_scores(
            probabilities,
            dependence=FACTORIZED_INDEPENDENCE,
            max_depth=max_depth,
        )
    effective_hyper = {
        key: value for key, value in hyper.items() if key != "adjustment_strength"
    }
    return sequential_scores(
        probabilities,
        dependence=arm,
        design=(design_labels if arm in DESIGN_DEPENDENT_ARMS else None),
        max_depth=max_depth,
        hyper=effective_hyper,
        fixed_order=(fixed_tally_order if arm == FIXED_TALLY_FACTORIZED else None),
    )


def _score_wrapper_grid(
    *,
    probabilities: np.ndarray,
    design_labels: np.ndarray,
    neighbor_correctness: np.ndarray,
    fixed_tally_order: tuple[int, ...],
    max_depth: int,
    seed: int,
    split_seed: int,
    arms: tuple[str, ...] = WRAPPER_ARMS,
    diagnostic_traces: dict[str, dict] | None = None,
    preserve_score_prefixes: bool = True,
) -> dict[str, tuple[str, dict, np.ndarray]]:
    """Fit requested query-level candidates without validation or calibration labels."""

    fitted = {}
    for arm in arms:
        for index, hyper in enumerate(WRAPPER_HYPER_GRIDS[arm]):
            trace = None
            if arm == QUERY_LOCAL_EMPIRICAL_COVERAGE:
                raw, trace = query_local_empirical_scores(
                    neighbor_correctness,
                    max_depth=max_depth,
                    return_trace=True,
                )
            elif arm == LOCAL_ORDER_FACTORIZED:
                raw, trace = local_order_factorized_scores(
                    probabilities,
                    neighbor_correctness,
                    max_depth=max_depth,
                    return_trace=True,
                )
            else:
                raw = _sequential_scores_for_grid_candidate(
                    probabilities,
                    arm=arm,
                    design_labels=design_labels,
                    max_depth=max_depth,
                    hyper=hyper,
                    fixed_tally_order=fixed_tally_order,
                )
            scores = smooth_nonconformity_scores(
                augment_with_racer_null(probabilities, raw),
                seed=seed,
                split_seed=split_seed,
                preserve_prefixes=preserve_score_prefixes,
            )
            key = f"wrapper_{arm}::{index}"
            fitted[key] = (arm, hyper, scores)
            if diagnostic_traces is not None and trace is not None:
                diagnostic_traces[key] = {
                    "raw_scores": raw,
                    "trace": trace,
                    "selection_scores": scores[:, :-1],
                }
    return fitted


def _select_and_finalize_wrapper_arm(
    *,
    arm: str,
    fitted: dict[str, tuple[str, dict, np.ndarray]],
    augmented_probabilities: np.ndarray,
    standard_labels: np.ndarray,
    augmented_labels: np.ndarray,
    val_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
    neighbor_support_size: int,
    diagnostic_traces: dict[str, dict] | None = None,
) -> dict:
    """Select on validation, freeze, then calibrate the selected arm once."""

    validation_records = []
    for key, (candidate_arm, hyper, scores) in fitted.items():
        if candidate_arm != arm:
            continue
        validation_records.append(_temporary_validation_record(
            arm_key=key,
            hyper=hyper,
            scores=scores,
            augmented_probabilities=augmented_probabilities,
            standard_labels=standard_labels,
            augmented_labels=augmented_labels,
            val_idx=val_idx,
            alpha=alpha,
        ))
    selection = _select_validation_candidate(validation_records, alpha)
    chosen = selection["chosen"]
    if chosen is None:
        return {
            "selected_arm": arm,
            "chosen_hyper": None,
            "policy_metadata": {
                **WRAPPER_ARM_METADATA[arm],
                "knn_support_size": int(neighbor_support_size),
            },
            "selection_source": "validation_only",
            "selection_fallback_used": False,
            "selection_non_certification": True,
            "validation_calls": None,
            "validation_standard_all_miss_risk": None,
            "validation_safe_action_risk": None,
            "validation_augmented_loss": None,
            "validation_trajectory": validation_records,
            "primary_risk_name": "standard_all_miss_risk",
            "calls": None,
            "calls_test": None,
            "risk_test": None,
            "standard_all_miss_risk": None,
            "safe_action_risk": None,
            "published_augmented_null_loss": None,
            "augmented_loss": None,
            "null_selection_rate": None,
            "empty_set_rate": None,
            "abstention_rate": None,
            "certified": False,
            "non_certification": True,
            "non_certification_reason": selection["non_certification_reason"],
            "risk_validity_claimed": False,
            "final_calibration_count": 0,
            "n_final_calibration": 0,
            "calibration_misses": None,
            "unavoidable_calibration_misses": None,
            "crc_bound": None,
            "final_threshold": None,
        }
    _, chosen_hyper, chosen_scores = fitted[chosen["arm_key"]]
    result = _final_calibrate_and_test(
        scores=chosen_scores,
        augmented_probabilities=augmented_probabilities,
        standard_labels=standard_labels,
        augmented_labels=augmented_labels,
        cal_idx=cal_idx,
        test_idx=test_idx,
        alpha=alpha,
    )
    result.update({
        "selected_arm": arm,
        "chosen_hyper": dict(chosen_hyper),
        "policy_metadata": {
            **WRAPPER_ARM_METADATA[arm],
            "knn_support_size": int(neighbor_support_size),
        },
        "selection_source": "validation_only",
        "selection_fallback_used": selection["fallback_used"],
        "selection_non_certification": selection["selection_non_certification"],
        "selection_non_certification_reason": selection["non_certification_reason"],
        "validation_calls": chosen["calls"],
        "validation_standard_all_miss_risk": chosen["standard_all_miss_risk"],
        "validation_safe_action_risk": chosen["standard_all_miss_risk"],
        "validation_augmented_loss": chosen["published_augmented_null_loss"],
        "validation_trajectory": validation_records,
    })
    if diagnostic_traces is not None and chosen["arm_key"] in diagnostic_traces:
        bundle = diagnostic_traces[chosen["arm_key"]]
        diagnostics = per_query_score_diagnostics(
            bundle["raw_scores"],
            bundle["trace"],
            threshold=result["final_threshold"],
            selection_scores=bundle["selection_scores"],
        )
        split_by_offset = np.full(len(diagnostics), "unused", dtype=object)
        split_by_offset[val_idx] = "validation"
        split_by_offset[cal_idx] = "final_calibration"
        split_by_offset[test_idx] = "test"
        for offset, record in enumerate(diagnostics):
            record["evaluation_split"] = str(split_by_offset[offset])
        result["score_diagnostics"] = diagnostics
        result["score_diagnostics_scope"] = "all_evaluation_pool_queries"
    return result


def _evaluate_neighbor_k_redraw(
    *,
    train_embeddings: np.ndarray,
    train_correctness: np.ndarray,
    query_embeddings: np.ndarray,
    design_labels: np.ndarray,
    fixed_tally_order: tuple[int, ...],
    standard_labels: np.ndarray,
    val_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alphas: list[float],
    neighbor_k: int,
    max_depth: int,
    seed: int,
    split_seed: int,
    redraw: int,
) -> dict[str, dict[str, dict]]:
    """Evaluate one secondary neighbor size on the frozen primary split."""

    probabilities, neighbor_correctness = _knn_probabilities(
        train_embeddings,
        train_correctness,
        query_embeddings,
        neighbor_k,
        return_support=True,
    )
    augmented_probabilities = augment_router_scores(probabilities)
    augmented_labels = augment_ground_truth(standard_labels)
    controls = {}
    for variant in ("G", "P"):
        _, scores = nonconformity_scores(
            probabilities,
            variant,
            seed=seed,
            split_seed=split_seed,
        )
        controls[f"racer_{variant}"] = scores
    diagnostic_traces = {}
    fitted = _score_wrapper_grid(
        probabilities=probabilities,
        design_labels=design_labels,
        neighbor_correctness=neighbor_correctness,
        fixed_tally_order=fixed_tally_order,
        max_depth=max_depth,
        seed=seed,
        split_seed=split_seed,
        diagnostic_traces=diagnostic_traces,
    )

    results = {}
    for alpha in alphas:
        alpha_key = f"{alpha}"
        records = {}
        for control_name, scores in controls.items():
            record = _final_calibrate_and_test(
                scores=scores,
                augmented_probabilities=augmented_probabilities,
                standard_labels=standard_labels,
                augmented_labels=augmented_labels,
                cal_idx=cal_idx,
                test_idx=test_idx,
                alpha=alpha,
            )
            record.update({
                "redraw": int(redraw),
                "selected_arm": control_name,
                "chosen_hyper": None,
                "selection_source": "predeclared_control",
                "neighbor_k": int(neighbor_k),
            })
            records[control_name] = record
        for arm in WRAPPER_ARMS:
            record = _select_and_finalize_wrapper_arm(
                arm=arm,
                fitted=fitted,
                augmented_probabilities=augmented_probabilities,
                standard_labels=standard_labels,
                augmented_labels=augmented_labels,
                val_idx=val_idx,
                cal_idx=cal_idx,
                test_idx=test_idx,
                alpha=alpha,
                neighbor_support_size=neighbor_correctness.shape[1],
                diagnostic_traces=diagnostic_traces,
            )
            record.update({"redraw": int(redraw), "neighbor_k": int(neighbor_k)})
            records[f"wrapper_{arm}_tuned"] = record
        results[alpha_key] = records
    return results


def _evaluate_design_permutation(
    *,
    design_labels: np.ndarray,
    permutation_seed: int,
    permutation_index: int,
    probabilities: np.ndarray,
    neighbor_correctness: np.ndarray,
    augmented_probabilities: np.ndarray,
    standard_labels: np.ndarray,
    augmented_labels: np.ndarray,
    val_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alphas: list[float],
    real_results_by_alpha: dict[str, dict[str, dict]],
    max_depth: int,
    seed: int,
    split_seed: int,
) -> dict:
    """Refit design-dependent arms after one column-wise design permutation."""

    shuffled_design = _shuffle_design_columns(design_labels, permutation_seed)
    shuffled_fixed_order = build_order(
        "empirical_tally",
        shuffled_design,
        model_names=[str(index) for index in range(design_labels.shape[1])],
        max_depth=max_depth,
    )
    fitted = _score_wrapper_grid(
        probabilities=probabilities,
        design_labels=shuffled_design,
        neighbor_correctness=neighbor_correctness,
        fixed_tally_order=shuffled_fixed_order,
        max_depth=max_depth,
        seed=seed,
        split_seed=split_seed,
        arms=DESIGN_DEPENDENT_ARMS,
    )

    per_alpha = {}
    for alpha in alphas:
        alpha_key = f"{alpha}"
        arm_records = {}
        for arm in DESIGN_DEPENDENT_ARMS:
            shuffled_result = _select_and_finalize_wrapper_arm(
                arm=arm,
                fitted=fitted,
                augmented_probabilities=augmented_probabilities,
                standard_labels=standard_labels,
                augmented_labels=augmented_labels,
                val_idx=val_idx,
                cal_idx=cal_idx,
                test_idx=test_idx,
                alpha=alpha,
                neighbor_support_size=neighbor_correctness.shape[1],
            )
            real_calls = real_results_by_alpha[alpha_key][arm]["calls_test"]
            shuffled_calls = shuffled_result["calls_test"]
            comparable = real_calls is not None and shuffled_calls is not None
            shuffled_result.update({
                "permutation_seed": int(permutation_seed),
                "real_calls_test": real_calls,
                "shuffled_calls_test": shuffled_calls,
                "real_minus_shuffled_calls": (
                    float(real_calls) - float(shuffled_calls) if comparable else None
                ),
                "call_difference_available": comparable,
            })
            arm_records[arm] = shuffled_result
        per_alpha[alpha_key] = {
            "arms": arm_records,
            "omitted_arms": dict(DESIGN_PERMUTATION_OMISSIONS),
        }

    return {
        "permutation_index": int(permutation_index),
        "permutation_seed": int(permutation_seed),
        "per_alpha": per_alpha,
    }


def build_summary(per_redraw: dict, alphas: list[float]) -> dict:
    """Recompute all E2 aggregates from ordered raw redraw records."""

    control_names = RACER_CONTROLS
    tuned_names = tuple(f"wrapper_{arm}_tuned" for arm in WRAPPER_ARMS)
    summary = {}
    for alpha in alphas:
        alpha_key = f"{alpha}"
        entry = {"controls": {}, "arms": {}}
        for name, records in per_redraw[alpha_key].items():
            certified_records = [record for record in records if record["certified"]]
            calls = [record["calls_test"] for record in certified_records]
            standard = [
                record["standard_all_miss_risk"] for record in certified_records
            ]
            augmented = [
                record["published_augmented_null_loss"]
                for record in certified_records
            ]
            info = {
                "primary_risk_name": "standard_all_miss_risk",
                "calls_mean": float(np.mean(calls)) if calls else None,
                "calls_median": float(np.median(calls)) if calls else None,
                "standard_all_miss_risk_mean": (
                    float(np.mean(standard)) if standard else None
                ),
                "risk_mean": float(np.mean(standard)) if standard else None,
                "safe_action_risk_mean": (
                    float(np.mean(standard)) if standard else None
                ),
                "published_augmented_null_loss_mean": (
                    float(np.mean(augmented)) if augmented else None
                ),
                "augmented_loss_mean": (
                    float(np.mean(augmented)) if augmented else None
                ),
                "empty_set_mean": float(np.mean([
                    record["empty_set_rate"] for record in certified_records
                ])) if certified_records else None,
                "abstention_mean": float(np.mean([
                    record["empty_set_rate"] for record in certified_records
                ])) if certified_records else None,
                "certification_rate": float(np.mean([
                    record["certified"] for record in records
                ])),
                "non_certification_count": int(sum(
                    record["non_certification"] for record in records
                )),
                "certified_redraw_count": len(certified_records),
                "per_redraw_calls": [record["calls_test"] for record in records],
                "per_redraw_risk": [
                    record["standard_all_miss_risk"] for record in records
                ],
            }
            entry["arms"][name] = info
            if name in control_names:
                entry["controls"][name] = info
        for name in tuned_names:
            arm_info = entry["arms"][name]
            for control_name in RACER_CONTROLS:
                control_calls = entry["controls"][control_name]["calls_mean"]
                arm_calls = arm_info["calls_mean"]
                arm_info[f"call_reduction_vs_{control_name}"] = (
                    (control_calls - arm_calls) / control_calls
                    if control_calls is not None
                    and arm_calls is not None
                    and control_calls > 0
                    else None
                )
        summary[alpha_key] = entry
    return summary


def build_design_permutation_summary(
    design_permutation_redraws: list[dict],
    alphas: list[float],
) -> dict:
    """Recompute information-destruction aggregates from raw permutations."""

    summary = {}
    for alpha in alphas:
        alpha_key = f"{alpha}"
        arm_summary = {}
        for arm in DESIGN_DEPENDENT_ARMS:
            differences = [
                value
                for redraw_record in design_permutation_redraws
                for permutation_record in redraw_record["permutations"]
                for value in [
                    permutation_record["per_alpha"][alpha_key]["arms"][arm][
                        "real_minus_shuffled_calls"
                    ]
                ]
                if value is not None
            ]
            arm_summary[arm] = {
                "real_minus_shuffled_calls_mean": (
                    float(np.mean(differences)) if differences else None
                ),
                "real_minus_shuffled_calls": differences,
                "record_count": len(differences),
            }
        summary[alpha_key] = arm_summary
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--benchmark", default="ifeval")
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=25)
    parser.add_argument("--alphas", default="0.05,0.10")
    parser.add_argument("--redraws", type=int, default=40)
    parser.add_argument("--redraw-start", type=int, default=0)
    parser.add_argument("--knn-k", type=int, default=40)
    parser.add_argument(
        "--neighbor-k-grid",
        default=None,
        help=(
            "optional comma-separated sensitivity grid, for example 10,20,40,80; "
            "--knn-k remains the primary result and is evaluated first"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--design-fraction", type=float, default=0.25)
    parser.add_argument("--router-fraction", type=float, default=0.25)
    parser.add_argument("--cal-fraction", type=float, default=0.20)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--design-permutations",
        type=int,
        default=0,
        help="number of independent within-column design-label permutations per redraw",
    )
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    if arguments.redraws <= 0:
        parser.error("--redraws must be positive")
    if arguments.redraw_start < 0:
        parser.error("--redraw-start must be nonnegative")
    if arguments.design_permutations < 0:
        parser.error("--design-permutations must be nonnegative")
    try:
        neighbor_k_grid = _parse_neighbor_k_grid(
            arguments.neighbor_k_grid, arguments.knn_k
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    binary, embeddings = _load_aligned(
        Path(arguments.router_dir), Path(arguments.leaderboard),
        arguments.benchmark, arguments.threshold,
    )
    n_queries, n_models_full = binary.shape
    alphas = [float(value) for value in arguments.alphas.split(",") if value]
    generator = np.random.default_rng(arguments.seed)
    columns = np.sort(generator.choice(
        n_models_full,
        size=min(arguments.pool_size, n_models_full),
        replace=False,
    ))
    correctness = binary[:, columns]
    complete_pool_miss = float((correctness.sum(axis=1) == 0).mean())
    top1_miss = float(1.0 - correctness.mean(axis=0).max())

    n_design = int(round(n_queries * arguments.design_fraction))
    n_router = int(round(n_queries * arguments.router_fraction))
    control_names = RACER_CONTROLS
    tuned_names = tuple(f"wrapper_{arm}_tuned" for arm in WRAPPER_ARMS)
    per_redraw = {
        f"{alpha}": {name: [] for name in control_names + tuned_names}
        for alpha in alphas
    }
    neighbor_k_per_redraw = {
        neighbor_k: {
            f"{alpha}": {name: [] for name in control_names + tuned_names}
            for alpha in alphas
        }
        for neighbor_k in neighbor_k_grid
        if neighbor_k != arguments.knn_k
    }
    design_permutation_redraws = []

    for local_redraw in range(arguments.redraws):
        global_redraw = arguments.redraw_start + local_redraw
        rng = np.random.default_rng(arguments.seed + 13 * global_redraw)
        permutation = rng.permutation(n_queries)
        design_rows = permutation[:n_design]
        router_rows = permutation[n_design:n_design + n_router]
        pool_rows = permutation[n_design + n_router:]
        if min(len(design_rows), len(router_rows), len(pool_rows)) < 40:
            raise SystemExit("a partition fell below 40 queries")

        probabilities, neighbor_correctness = _knn_probabilities(
            embeddings[router_rows], correctness[router_rows],
            embeddings[pool_rows], arguments.knn_k, return_support=True,
        )
        standard_labels = correctness[pool_rows].astype(bool)
        augmented_labels = augment_ground_truth(standard_labels)
        augmented_probabilities = augment_router_scores(probabilities)
        design_labels = correctness[design_rows].astype(int)
        fixed_tally_order = build_order(
            "empirical_tally",
            design_labels,
            model_names=[str(index) for index in range(correctness.shape[1])],
            max_depth=arguments.max_depth,
        )

        n_pool = len(pool_rows)
        remaining_fraction = 1.0 - arguments.design_fraction - arguments.router_fraction
        test_fraction = remaining_fraction - arguments.cal_fraction - arguments.val_fraction
        denominator = arguments.cal_fraction + arguments.val_fraction + test_fraction
        n_cal = int(round(n_pool * arguments.cal_fraction / denominator))
        n_val = int(round(n_pool * arguments.val_fraction / denominator))
        n_test = n_pool - n_cal - n_val
        if min(n_cal, n_val, n_test) < 20:
            raise SystemExit(f"partition too small: cal={n_cal} val={n_val} test={n_test}")
        cal_idx = np.arange(n_cal)
        val_idx = np.arange(n_cal, n_cal + n_val)
        test_idx = np.arange(n_cal + n_val, n_pool)

        controls = {}
        for variant in ("G", "P"):
            _, scores = nonconformity_scores(
                probabilities,
                variant,
                seed=arguments.seed,
                split_seed=global_redraw + 1,
            )
            controls[f"racer_{variant}"] = scores
        diagnostic_traces = {}
        fitted = _score_wrapper_grid(
            probabilities=probabilities,
            design_labels=design_labels,
            neighbor_correctness=neighbor_correctness,
            fixed_tally_order=fixed_tally_order,
            max_depth=arguments.max_depth,
            seed=arguments.seed,
            split_seed=global_redraw + 1,
            diagnostic_traces=diagnostic_traces,
        )

        for alpha in alphas:
            alpha_key = f"{alpha}"
            for control_name, scores in controls.items():
                result = _final_calibrate_and_test(
                    scores=scores,
                    augmented_probabilities=augmented_probabilities,
                    standard_labels=standard_labels,
                    augmented_labels=augmented_labels,
                    cal_idx=cal_idx,
                    test_idx=test_idx,
                    alpha=alpha,
                )
                result.update({
                    "redraw": int(global_redraw),
                    "selected_arm": control_name,
                    "chosen_hyper": None,
                    "selection_source": "predeclared_control",
                    "neighbor_k": int(arguments.knn_k),
                })
                per_redraw[alpha_key][control_name].append(result)

            for arm in WRAPPER_ARMS:
                result = _select_and_finalize_wrapper_arm(
                    arm=arm,
                    fitted=fitted,
                    augmented_probabilities=augmented_probabilities,
                    standard_labels=standard_labels,
                    augmented_labels=augmented_labels,
                    val_idx=val_idx,
                    cal_idx=cal_idx,
                    test_idx=test_idx,
                    alpha=alpha,
                    neighbor_support_size=neighbor_correctness.shape[1],
                    diagnostic_traces=diagnostic_traces,
                )
                result["redraw"] = int(global_redraw)
                result["neighbor_k"] = int(arguments.knn_k)
                per_redraw[alpha_key][f"wrapper_{arm}_tuned"].append(result)

        for neighbor_k, sensitivity_records in neighbor_k_per_redraw.items():
            redraw_results = _evaluate_neighbor_k_redraw(
                train_embeddings=embeddings[router_rows],
                train_correctness=correctness[router_rows],
                query_embeddings=embeddings[pool_rows],
                design_labels=design_labels,
                fixed_tally_order=fixed_tally_order,
                standard_labels=standard_labels,
                val_idx=val_idx,
                cal_idx=cal_idx,
                test_idx=test_idx,
                alphas=alphas,
                neighbor_k=neighbor_k,
                max_depth=arguments.max_depth,
                seed=arguments.seed,
                split_seed=global_redraw + 1,
                redraw=global_redraw,
            )
            for alpha_key, arm_records in redraw_results.items():
                for name, record in arm_records.items():
                    sensitivity_records[alpha_key][name].append(record)

        permutation_records = []
        if arguments.design_permutations:
            real_results_by_alpha = {
                f"{alpha}": {
                    arm: per_redraw[f"{alpha}"][f"wrapper_{arm}_tuned"][-1]
                    for arm in DESIGN_DEPENDENT_ARMS
                }
                for alpha in alphas
            }
            for permutation_index in range(arguments.design_permutations):
                permutation_seed = _design_permutation_seed(
                    arguments.seed, global_redraw, permutation_index
                )
                permutation_records.append(_evaluate_design_permutation(
                    design_labels=design_labels,
                    permutation_seed=permutation_seed,
                    permutation_index=permutation_index,
                    probabilities=probabilities,
                    neighbor_correctness=neighbor_correctness,
                    augmented_probabilities=augmented_probabilities,
                    standard_labels=standard_labels,
                    augmented_labels=augmented_labels,
                    val_idx=val_idx,
                    cal_idx=cal_idx,
                    test_idx=test_idx,
                    alphas=alphas,
                    real_results_by_alpha=real_results_by_alpha,
                    max_depth=arguments.max_depth,
                    seed=arguments.seed,
                    split_seed=global_redraw + 1,
                ))
        design_permutation_redraws.append({
            "redraw": int(global_redraw),
            "permutations": permutation_records,
        })

    summary = build_summary(per_redraw, alphas)
    neighbor_k_sensitivity = {
        str(neighbor_k): {
            "neighbor_k": int(neighbor_k),
            "per_redraw": records,
            "summary": build_summary(records, alphas),
        }
        for neighbor_k, records in neighbor_k_per_redraw.items()
    }
    design_permutation_summary = build_design_permutation_summary(
        design_permutation_redraws, alphas
    )

    output = {
        "design": "B_prime_E2_frozen_selection",
        "benchmark": arguments.benchmark,
        "protocol": "routereval",
        "selection_protocol": "design_and_validation_then_freeze_then_one_final_calibration_then_test",
        "primary_risk_name": "standard_all_miss_risk",
        "published_racer_diagnostic_name": "published_augmented_null_loss",
        "n_queries": int(n_queries),
        "pool_size": int(len(columns)),
        "n_models_full": int(n_models_full),
        "complete_pool_miss": complete_pool_miss,
        "top1_model_miss": top1_miss,
        "redraws": arguments.redraws,
        "redraw_start": arguments.redraw_start,
        "redraw_end": arguments.redraw_start + arguments.redraws,
        "redraw_range": {
            "start_inclusive": arguments.redraw_start,
            "end_exclusive": arguments.redraw_start + arguments.redraws,
            "count": arguments.redraws,
        },
        "seed": arguments.seed,
        "threshold": arguments.threshold,
        "knn_k": arguments.knn_k,
        "primary_neighbor_k": arguments.knn_k,
        "neighbor_k_grid": list(neighbor_k_grid),
        "neighbor_k_sensitivity": neighbor_k_sensitivity,
        "wrapper_max_depth": arguments.max_depth,
        "alphas": alphas,
        "wrapper_arms": list(WRAPPER_ARMS),
        "wrapper_arm_metadata": WRAPPER_ARM_METADATA,
        "racer_controls": ["racer_G", "racer_P"],
        "hyperparameter_grids": WRAPPER_HYPER_GRIDS,
        "partition_fractions": {
            "design": arguments.design_fraction,
            "router": arguments.router_fraction,
            "final_calibration": arguments.cal_fraction,
            "validation": arguments.val_fraction,
            "test": 1.0 - arguments.design_fraction - arguments.router_fraction
                    - arguments.cal_fraction - arguments.val_fraction,
        },
        "per_redraw": per_redraw,
        "summary": summary,
        "information_destruction_control": {
            "requested_permutations_per_redraw": arguments.design_permutations,
            "shuffle_scope": "design_rows_only",
            "shuffle_operation": "independent_per_model_correctness_column_permutation",
            "preserved_quantity": "each_design_column_marginal",
            "refit_arms": list(DESIGN_DEPENDENT_ARMS),
            "omitted_arms": dict(DESIGN_PERMUTATION_OMISSIONS),
            "selection_protocol": "validation_only_then_one_final_calibration",
            "difference_definition": "real_calls_test_minus_shuffled_calls_test",
            "per_redraw": design_permutation_redraws,
            "summary": design_permutation_summary,
        },
    }
    out_path = Path(arguments.out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite immutable result {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(out_path)

    for alpha in alphas:
        entry = summary[f"{alpha}"]
        print(f"alpha {alpha}")
        for name in control_names + tuned_names:
            info = entry["arms"][name]
            if info["calls_mean"] is None:
                print(f"  {name:<42s} NON-CERTIFIED")
                continue
            print(
                f"  {name:<42s} calls {info['calls_mean']:.3f}  "
                f"standard miss {info['standard_all_miss_risk_mean']:.4f}  "
                f"published augmented-null diagnostic "
                f"{info['published_augmented_null_loss_mean']:.4f}  "
                f"cert {info['certification_rate']:.3f}"
            )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
