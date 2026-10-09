"""Frozen aggregation selection and per-query selector evaluation.

Aggregation is selected once on validation data and then reused unchanged for
all selectors. Model correctness labels are used only to measure whether a
selected set contains a correct model. They are never passed to response
aggregation. Final answer correctness is determined by canonical comparison
with the supplied reference answer through :mod:`components.scoring`.

The raw payload returned by :func:`evaluate_selector` contains every field
required by ``artifacts.validate_raw_artifacts``. Extra fields retain selector
name, frozen aggregator identity, depths, abstentions, latency, and summary
metrics without weakening the version 1 artifact schema.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import SCHEMA_VERSION, validate_raw_artifacts
from .inference import DEFAULT_ENDPOINT_REGISTRY, infer_paired_metrics
from .racer_protocol import CalibrationResult, RoutingResult
from .scoring import (
    STATUS_SCORED,
    STATUS_UNSUPPORTED,
    AggregationResult,
    aggregate_responses,
    canonicalize_answer,
    reference_comparison_supported,
)


AGGREGATOR_CANDIDATES = (
    "top_router",
    "canonical_majority",
    "score_weighted",
    "full_ensemble",
)

_AGGREGATOR_METHODS = {
    "top_router": "top_router",
    "canonical_majority": "majority",
    "score_weighted": "score_weighted",
    "full_ensemble": "full_ensemble",
}


@dataclass(frozen=True)
class AggregatorValidationScore:
    """Validation performance for one aggregation candidate."""

    name: str
    scoring_method: str
    correct: int
    scored: int
    total: int
    accuracy: float
    conditional_accuracy: float
    scored_rate: float


@dataclass(frozen=True)
class FrozenAggregator:
    """An immutable validation-selected aggregation policy."""

    name: str
    scoring_method: str
    validation_scores: tuple[AggregatorValidationScore, ...]
    selection_rule: str = "maximum end-to-end validation accuracy; stable candidate order ties"

    def __post_init__(self) -> None:
        if self.name not in _AGGREGATOR_METHODS:
            raise ValueError(f"unknown frozen aggregator: {self.name}")
        if self.scoring_method != _AGGREGATOR_METHODS[self.name]:
            raise ValueError("frozen aggregator name and scoring method disagree")


class EvaluationError(ValueError):
    """Raised when query-level evaluation inputs do not align."""


def _normalize_aggregator_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "majority": "canonical_majority",
        "majority_vote": "canonical_majority",
        "canonical_vote": "canonical_majority",
        "weighted": "score_weighted",
        "weighted_vote": "score_weighted",
        "top": "top_router",
        "router": "top_router",
        "full": "full_ensemble",
    }
    key = aliases.get(key, key)
    if key not in _AGGREGATOR_METHODS:
        raise ValueError(f"unknown aggregation candidate: {name}")
    return key


def _as_rows(values: Sequence[Any], name: str, expected: int | None = None) -> list[Any]:
    rows = list(values)
    if not rows:
        raise EvaluationError(f"{name} must not be empty")
    if expected is not None and len(rows) != expected:
        raise EvaluationError(f"{name} must contain {expected} rows")
    return rows


def _validate_query_inputs(
    tasks: Sequence[str],
    responses: Sequence[Sequence[Any]],
    router_scores: Sequence[Sequence[float]],
    reference_answers: Sequence[Any],
) -> tuple[list[str], list[list[Any]], list[list[float]], list[Any], int]:
    task_rows = [str(task) for task in _as_rows(tasks, "tasks")]
    n_queries = len(task_rows)
    response_rows = [list(row) for row in _as_rows(responses, "responses", n_queries)]
    score_rows = [list(row) for row in _as_rows(router_scores, "router_scores", n_queries)]
    references = _as_rows(reference_answers, "reference_answers", n_queries)

    n_models = len(response_rows[0])
    if n_models == 0:
        raise EvaluationError("responses must contain at least one model")
    for index, (response_row, score_row) in enumerate(zip(response_rows, score_rows)):
        if len(response_row) != n_models or len(score_row) != n_models:
            raise EvaluationError(
                f"query {index} must contain exactly {n_models} responses and scores"
            )
        numeric_scores = np.asarray(score_row, dtype=float)
        if np.any(~np.isfinite(numeric_scores)):
            raise EvaluationError("router_scores must be finite")
        score_rows[index] = numeric_scores.tolist()
    return task_rows, response_rows, score_rows, references, n_models


def _normalize_selected_sets(
    selected_sets: Sequence[Iterable[int]], n_queries: int, n_models: int
) -> list[tuple[int, ...]]:
    rows = _as_rows(selected_sets, "selected_sets", n_queries)
    normalized: list[tuple[int, ...]] = []
    for row_index, row in enumerate(rows):
        raw_indices = tuple(int(index) for index in row)
        indices = tuple(dict.fromkeys(raw_indices))
        if len(indices) != len(raw_indices):
            raise EvaluationError(f"selected_sets[{row_index}] contains duplicates")
        if any(index < 0 or index >= n_models for index in indices):
            raise EvaluationError(f"selected_sets[{row_index}] contains an invalid index")
        normalized.append(indices)
    return normalized


def _selected_set_groups(
    selected_sets: Sequence[Iterable[int]] | Mapping[str, Sequence[Iterable[int]]],
    n_queries: int,
    n_models: int,
) -> list[tuple[str, list[tuple[int, ...]]]]:
    if isinstance(selected_sets, Mapping):
        if not selected_sets:
            raise EvaluationError("selected_sets mapping must not be empty")
        return [
            (str(name), _normalize_selected_sets(rows, n_queries, n_models))
            for name, rows in selected_sets.items()
        ]
    return [("validation", _normalize_selected_sets(selected_sets, n_queries, n_models))]


def _aggregate_one(
    *,
    task: str,
    responses: Sequence[Any],
    router_scores: Sequence[float],
    selected: tuple[int, ...],
    aggregator: FrozenAggregator | str,
) -> AggregationResult | None:
    if not selected:
        return None
    name = aggregator.name if isinstance(aggregator, FrozenAggregator) else _normalize_aggregator_name(aggregator)
    return aggregate_responses(
        task,
        responses,
        router_scores,
        method=_AGGREGATOR_METHODS[name],
        selected_indices=selected,
        # Deliberately no per-model labels and no reference answer here. Answer
        # construction must be independent of oracle correctness information.
    )


def _score_against_reference(
    *,
    task: str,
    responses: Sequence[Any],
    router_scores: Sequence[float],
    selected: tuple[int, ...],
    aggregator: FrozenAggregator | str,
    reference_answer: Any,
) -> AggregationResult | None:
    """Aggregate without oracle information, then score the fixed answer."""

    result = _aggregate_one(
        task=task,
        responses=responses,
        router_scores=router_scores,
        selected=selected,
        aggregator=aggregator,
    )
    if result is None or result.answer is None:
        return result

    if not reference_comparison_supported(task):
        return replace(
            result,
            status=STATUS_UNSUPPORTED,
            correct=None,
            reason=(
                "task has no canonically comparable answer; equality between two "
                "generated programs is uninformative so correctness is not scored"
            ),
        )

    reference = canonicalize_answer(task, reference_answer)
    if not reference.supported:
        return replace(
            result,
            status=STATUS_UNSUPPORTED,
            correct=None,
            reason="reference answer could not be extracted",
        )
    return replace(
        result,
        status=STATUS_SCORED,
        correct=result.answer == reference.value,
        reason=None,
    )


def select_common_aggregator(
    *,
    tasks: Sequence[str],
    responses: Sequence[Sequence[Any]],
    router_scores: Sequence[Sequence[float]],
    reference_answers: Sequence[Any],
    selected_sets: Sequence[Iterable[int]] | Mapping[str, Sequence[Iterable[int]]],
    candidates: Sequence[str] = AGGREGATOR_CANDIDATES,
) -> FrozenAggregator:
    """Choose one common aggregator on validation data and freeze it.

    If ``selected_sets`` maps selector names to per-query sets, candidate
    performance is pooled over every selector and validation query. This makes
    it impossible to choose a different downstream aggregator for each
    selector. Abstentions and unscored answers count as incorrect in the main
    validation accuracy, preventing abstention from inflating the criterion.
    """

    task_rows, response_rows, score_rows, references, n_models = _validate_query_inputs(
        tasks, responses, router_scores, reference_answers
    )
    groups = _selected_set_groups(selected_sets, len(task_rows), n_models)
    normalized_candidates = tuple(_normalize_aggregator_name(name) for name in candidates)
    if not normalized_candidates:
        raise EvaluationError("candidates must not be empty")
    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise EvaluationError("candidates must be unique after alias normalization")

    validation_scores: list[AggregatorValidationScore] = []
    for candidate in normalized_candidates:
        total = len(task_rows) * len(groups)
        scored_count = 0
        correct_count = 0
        for _, selector_sets in groups:
            for index, selected in enumerate(selector_sets):
                result = _score_against_reference(
                    task=task_rows[index],
                    responses=response_rows[index],
                    router_scores=score_rows[index],
                    selected=selected,
                    aggregator=candidate,
                    reference_answer=references[index],
                )
                if result is not None and result.correct is not None:
                    scored_count += 1
                    correct_count += int(result.correct)
        validation_scores.append(
            AggregatorValidationScore(
                name=candidate,
                scoring_method=_AGGREGATOR_METHODS[candidate],
                correct=correct_count,
                scored=scored_count,
                total=total,
                accuracy=correct_count / total,
                conditional_accuracy=(correct_count / scored_count if scored_count else 0.0),
                scored_rate=scored_count / total,
            )
        )

    best_index = max(
        range(len(validation_scores)),
        key=lambda index: (
            validation_scores[index].accuracy,
            validation_scores[index].scored_rate,
            -index,
        ),
    )
    best = validation_scores[best_index]
    return FrozenAggregator(
        name=best.name,
        scoring_method=best.scoring_method,
        validation_scores=tuple(validation_scores),
    )


def _expand_numeric(values: float | int | Sequence[float | int], n: int, name: str) -> list[float]:
    if np.isscalar(values):
        rows = [float(values)] * n
    else:
        rows = [float(value) for value in values]
        if len(rows) != n:
            raise EvaluationError(f"{name} must contain {n} values")
    if any(not math.isfinite(value) for value in rows):
        raise EvaluationError(f"{name} must contain only finite values")
    return rows


def _expand_depths(values: int | Sequence[int], n: int) -> list[int]:
    rows = [int(values)] * n if np.isscalar(values) else [int(value) for value in values]
    if len(rows) != n or any(value < 0 for value in rows):
        raise EvaluationError("depths must contain one non-negative integer per query")
    return rows


def _cost_matrix(model_costs: Sequence[Any], n_queries: int, n_models: int) -> np.ndarray:
    costs = np.asarray(model_costs, dtype=float)
    if costs.shape == (n_models,):
        costs = np.broadcast_to(costs, (n_queries, n_models))
    if costs.shape != (n_queries, n_models):
        raise EvaluationError("model_costs must have shape (models,) or (queries, models)")
    if np.any(~np.isfinite(costs)) or np.any(costs < 0.0):
        raise EvaluationError("model_costs must be finite and non-negative")
    return costs


def _model_identifiers(model_ids: Sequence[str | int] | None, n_models: int) -> list[str | int]:
    identifiers: list[str | int] = list(range(n_models)) if model_ids is None else list(model_ids)
    if len(identifiers) != n_models:
        raise EvaluationError("model_ids must contain one identifier per model")
    if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in identifiers):
        raise EvaluationError("model_ids must contain only strings or integers")
    if len(set(identifiers)) != len(identifiers):
        raise EvaluationError("model_ids must be unique")
    return identifiers


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def evaluate_selector(
    selector_name: str,
    *,
    frozen_aggregator: FrozenAggregator,
    tasks: Sequence[str],
    responses: Sequence[Sequence[Any]],
    router_scores: Sequence[Sequence[float]],
    reference_answers: Sequence[Any],
    correctness: Sequence[Sequence[int | bool]],
    selected_sets: Sequence[Iterable[int]],
    thresholds: float | Sequence[float],
    depths: int | Sequence[int],
    model_costs: Sequence[Any],
    latencies: float | Sequence[float],
    split_ids: Mapping[str, Sequence[Any]],
    seeds: Mapping[str, int] | Sequence[int],
    correlations: Sequence[Sequence[float]],
    query_ids: Sequence[Any] | None = None,
    model_ids: Sequence[str | int] | None = None,
) -> dict[str, Any]:
    """Evaluate one selector and return a version 1 raw artifact payload.

    ``correctness`` is consulted only after selected sets are fixed and only to
    compute selection coverage. Aggregation receives only responses, router
    scores, and selected indices. Canonical reference scoring happens after the
    aggregate answer is fixed. ``latencies`` are observed end-to-end per-query
    values supplied by the caller rather than reconstructed from model timings.
    """

    if not isinstance(frozen_aggregator, FrozenAggregator):
        raise TypeError("frozen_aggregator must be a FrozenAggregator")
    if not str(selector_name):
        raise EvaluationError("selector_name must not be empty")

    task_rows, response_rows, score_rows, references, n_models = _validate_query_inputs(
        tasks, responses, router_scores, reference_answers
    )
    n_queries = len(task_rows)
    selected_rows = _normalize_selected_sets(selected_sets, n_queries, n_models)
    labels = np.asarray(correctness)
    if labels.shape != (n_queries, n_models) or not np.all((labels == 0) | (labels == 1)):
        raise EvaluationError("correctness must be a binary queries by models matrix")
    labels = labels.astype(bool, copy=False)

    threshold_rows = _expand_numeric(thresholds, n_queries, "thresholds")
    depth_rows = _expand_depths(depths, n_queries)
    if any(depth != len(selected) for depth, selected in zip(depth_rows, selected_rows)):
        raise EvaluationError("each depth must equal the corresponding selected set size")
    latency_rows = _expand_numeric(latencies, n_queries, "latencies")
    if any(value < 0.0 for value in latency_rows):
        raise EvaluationError("latencies must be non-negative")
    costs = _cost_matrix(model_costs, n_queries, n_models)
    identifiers = _model_identifiers(model_ids, n_models)

    ids = list(range(n_queries)) if query_ids is None else list(query_ids)
    if len(ids) != n_queries:
        raise EvaluationError("query_ids must contain one identifier per query")
    normalized_ids = [_json_scalar(value) for value in ids]
    test_ids = split_ids.get("test")
    if test_ids is None or [_json_scalar(value) for value in test_ids] != normalized_ids:
        raise EvaluationError("split_ids.test must equal query_ids in evaluation order")

    correlation_matrix = np.asarray(correlations, dtype=float)
    if correlation_matrix.shape != (n_models, n_models):
        raise EvaluationError("correlations must have one row and column per model")

    saved_sets: list[list[str | int]] = []
    abstentions: list[bool] = []
    risks: list[float] = []
    realized_costs: list[float] = []
    predictions: list[dict[str, Any]] = []
    final_correctness: list[bool] = []
    scored_count = 0

    for index, selected in enumerate(selected_rows):
        abstained = not selected
        coverage_loss = not any(labels[index, model] for model in selected)
        result = _score_against_reference(
            task=task_rows[index],
            responses=response_rows[index],
            router_scores=score_rows[index],
            selected=selected,
            aggregator=frozen_aggregator,
            reference_answer=references[index],
        )

        if result is None:
            used = ()
            answer = None
            status = "abstain"
            correct = False
            supporter_indices: tuple[int, ...] = ()
        else:
            used = result.used_indices
            answer = result.answer
            status = result.status
            correct = result.correct is True
            supporter_indices = result.supporter_indices
            if result.correct is not None:
                scored_count += 1

        selected_ids = [identifiers[model] for model in selected]
        used_ids = [identifiers[model] for model in used]
        supporter_ids = [identifiers[model] for model in supporter_indices]
        cost = float(costs[index, list(used)].sum()) if used else 0.0

        saved_sets.append(selected_ids)
        abstentions.append(abstained)
        risks.append(float(coverage_loss))
        realized_costs.append(cost)
        final_correctness.append(correct)
        predictions.append(
            {
                "query_id": normalized_ids[index],
                "selector": str(selector_name),
                "aggregator": frozen_aggregator.name,
                "selected_set": selected_ids,
                "aggregation_used_set": used_ids,
                "supporter_set": supporter_ids,
                "abstained": abstained,
                "threshold": threshold_rows[index],
                "depth": depth_rows[index],
                "selection_covered": not coverage_loss,
                "coverage_loss": coverage_loss,
                "canonical_answer": answer,
                "final_correct": correct,
                "aggregation_status": status,
                "cost": cost,
                "latency": latency_rows[index],
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selector": str(selector_name),
        "frozen_aggregator": {
            "name": frozen_aggregator.name,
            "scoring_method": frozen_aggregator.scoring_method,
            "selection_rule": frozen_aggregator.selection_rule,
            "validation_scores": [asdict(score) for score in frozen_aggregator.validation_scores],
        },
        "selected_sets": saved_sets,
        "abstentions": abstentions,
        "thresholds": threshold_rows,
        "depths": depth_rows,
        "predictions": predictions,
        "risks": risks,
        "costs": realized_costs,
        "latencies": latency_rows,
        "correlations": correlation_matrix.tolist(),
        "seeds": dict(seeds) if isinstance(seeds, Mapping) else list(seeds),
        "split_ids": {name: list(values) for name, values in split_ids.items()},
        "summary": {
            "n_queries": n_queries,
            "selection_coverage": float(1.0 - np.mean(risks)),
            "selection_coverage_loss": float(np.mean(risks)),
            "aggregation_accuracy": float(np.mean(final_correctness)),
            "aggregation_accuracy_scored": (
                float(sum(final_correctness) / scored_count) if scored_count else 0.0
            ),
            "aggregation_scored_rate": float(scored_count / n_queries),
            "abstention_rate": float(np.mean(abstentions)),
            "mean_cost": float(np.mean(realized_costs)),
            "mean_latency": float(np.mean(latency_rows)),
        },
    }
    validate_raw_artifacts(payload)
    return payload


def evaluate_selectors(
    selectors: Mapping[str, Sequence[Iterable[int]]],
    *,
    frozen_aggregator: FrozenAggregator,
    **common_inputs: Any,
) -> dict[str, dict[str, Any]]:
    """Evaluate multiple selectors with the exact same frozen aggregator.

    ``thresholds``, ``depths``, and ``latencies`` may be common values or
    mappings keyed by selector name. All other inputs are shared across arms.
    """

    if not selectors:
        raise EvaluationError("selectors must not be empty")

    reports: dict[str, dict[str, Any]] = {}
    for name, selected_sets in selectors.items():
        selector_name = str(name)
        scoped_inputs = dict(common_inputs)
        for field in ("thresholds", "depths", "latencies"):
            value = scoped_inputs.get(field)
            if isinstance(value, Mapping):
                if name not in value and selector_name not in value:
                    raise EvaluationError(
                        f"selector-specific {field} missing {selector_name}"
                    )
                scoped_inputs[field] = value.get(name, value.get(selector_name))
        reports[selector_name] = evaluate_selector(
            selector_name,
            frozen_aggregator=frozen_aggregator,
            selected_sets=selected_sets,
            **scoped_inputs,
        )
    return reports


def evaluate_routing_result(
    selector_name: str,
    *,
    routing: RoutingResult,
    calibration: CalibrationResult | float,
    frozen_aggregator: FrozenAggregator,
    **common_inputs: Any,
) -> dict[str, Any]:
    """Evaluate an exact RACER routing result through the common evaluator."""

    if not isinstance(routing, RoutingResult):
        raise TypeError("routing must be a RoutingResult")
    threshold = (
        calibration.threshold
        if isinstance(calibration, CalibrationResult)
        else float(calibration)
    )
    selected_sets = [
        tuple(int(index) for index in np.flatnonzero(row))
        for row in routing.standard_mask
    ]
    depths = [int(value) for value in routing.size]
    if [not selected for selected in selected_sets] != routing.abstain.tolist():
        raise RuntimeError("routing abstentions disagree with standard model sets")
    return evaluate_selector(
        selector_name,
        frozen_aggregator=frozen_aggregator,
        selected_sets=selected_sets,
        thresholds=threshold,
        depths=depths,
        **common_inputs,
    )


def compare_selector_evaluations(
    baseline: Mapping[str, Any],
    corrected: Mapping[str, Any],
    *,
    endpoint_registry: Mapping[str, Any] | Sequence[str] = DEFAULT_ENDPOINT_REGISTRY,
    strata: Any = None,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Run paired query-level inference on two selector artifact payloads."""

    validate_raw_artifacts(baseline)
    validate_raw_artifacts(corrected)
    baseline_ids = list(baseline["split_ids"]["test"])
    corrected_ids = list(corrected["split_ids"]["test"])
    if baseline_ids != corrected_ids:
        raise EvaluationError("selector evaluations must use identical test query order")

    def endpoint_values(payload: Mapping[str, Any]) -> dict[str, list[float]]:
        predictions = payload["predictions"]
        return {
            "calls": [
                float(len(row["aggregation_used_set"])) for row in predictions
            ],
            "risk": [float(value) for value in payload["risks"]],
            "accuracy": [float(row["final_correct"]) for row in predictions],
            "cost": [float(value) for value in payload["costs"]],
        }

    return infer_paired_metrics(
        endpoint_values(baseline),
        endpoint_values(corrected),
        endpoint_registry=endpoint_registry,
        strata=strata,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )


select_validation_aggregator = select_common_aggregator
freeze_validation_aggregator = select_common_aggregator
evaluate_selector_queries = evaluate_selector


__all__ = [
    "AGGREGATOR_CANDIDATES",
    "AggregatorValidationScore",
    "EvaluationError",
    "FrozenAggregator",
    "compare_selector_evaluations",
    "evaluate_routing_result",
    "evaluate_selector",
    "evaluate_selector_queries",
    "evaluate_selectors",
    "freeze_validation_aggregator",
    "select_common_aggregator",
    "select_validation_aggregator",
]
