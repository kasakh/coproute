"""Run ISSUE-52 fit and calibration transfer pairs.

One shard evaluates a source task and target task that share exactly the same
model columns. The fitted score family is frozen before either calibration
variant is applied. Source-calibrated and target-recalibrated records therefore
use the same target score matrix and target test rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.chain_discount import chain_discount_diagnostics  # noqa: E402
from components.paired_orders import per_query_orders_factorized_scores  # noqa: E402
from components.racer_protocol import select_standard_crc_threshold, smooth_nonconformity_scores  # noqa: E402
from components.routereval_protocol import subsample_pool  # noqa: E402
from scripts import run_paired_main_suite as paired  # noqa: E402

SCHEMA_VERSION = "issue52_transfer_v1"
REDRAW_COUNT = 20
FIT_MODES = ("source_fitted", "in_domain")
CALIBRATION_MODES = ("source_calibrated", "target_recalibrated")
CORE_METHODS = (
    "fixed_tally_factorized",
    paired.CHAIN_DISCOUNT_METHOD,
    "racer_g",
    "racer_p",
)
STAMP_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(paired.canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _task_family(protocol: str, benchmark: str) -> str:
    from components.stage1_task_adapters import (
        REGISTERED_ROUTERBENCH_TASKS,
        REGISTERED_ROUTEREVAL_TASKS,
    )

    if protocol == "routereval":
        try:
            return str(REGISTERED_ROUTEREVAL_TASKS[benchmark])
        except KeyError as error:
            raise ValueError(f"unregistered RouterEval transfer task {benchmark}") from error
    if protocol == "routerbench":
        if benchmark not in REGISTERED_ROUTERBENCH_TASKS:
            raise ValueError(f"unregistered RouterBench transfer task {benchmark}")
        return benchmark
    raise ValueError(f"unknown protocol {protocol}")


def _validate_task_shape(task: Any, name: str) -> None:
    query_ids = tuple(str(value) for value in task.query_ids)
    model_ids = tuple(str(value) for value in task.model_ids)
    embeddings = np.asarray(task.embeddings, dtype=float)
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError(f"{name} query IDs must be non-empty and unique")
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise ValueError(f"{name} model IDs must be non-empty and unique")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(query_ids):
        raise ValueError(f"{name} embeddings do not match query IDs")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError(f"{name} embeddings must be finite")
    shape = getattr(task.correctness, "shape", None)
    if shape != (len(query_ids), len(model_ids)):
        raise ValueError(f"{name} correctness shape does not match IDs")


def _labels_at(task: Any, rows: np.ndarray, columns: np.ndarray, role: str) -> np.ndarray:
    values = np.asarray(task.correctness[np.ix_(rows, columns)], dtype=np.int8)
    if values.shape != (len(rows), len(columns)):
        raise ValueError(f"{role} label shape is invalid")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{role} labels must be binary")
    return values


def _split_ids(task: Any, split: Mapping[str, np.ndarray]) -> dict[str, list[str]]:
    return {
        role: [str(task.query_ids[int(row)]) for row in rows]
        for role, rows in split.items()
    }


def _same_task(source_task: Any, target_task: Any) -> bool:
    return (
        str(source_task.task) == str(target_task.task)
        and tuple(source_task.query_ids) == tuple(target_task.query_ids)
        and tuple(source_task.model_ids) == tuple(target_task.model_ids)
    )


def _validate_ensembles(
    learned_ensembles: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    pool_size: int,
    pool_seed: int,
    model_ids: Sequence[str],
    protocol: str,
    target_family: str | None,
) -> None:
    if set(learned_ensembles) != set(paired.LEARNED_FAMILIES):
        raise ValueError("ISSUE-52 requires both FlexRouter and BaRP checkpoint ensembles")
    for family in paired.LEARNED_FAMILIES:
        members = learned_ensembles[family]
        seeds = [int(member["identity"]["training_seed"]) for member in members]
        if seeds != list(paired.TRAINING_SEEDS):
            raise ValueError(f"{family} checkpoints must use all five registered seeds in order")
        for member in members:
            identity = member["identity"]
            if identity.get("method") not in (None, family):
                raise ValueError(f"{family} checkpoint method identity is wrong")
            if identity.get("pool_size") != pool_size or identity.get("pool_seed") != pool_seed:
                raise ValueError(f"{family} checkpoint pool identity differs from transfer pool")
            if identity.get("model_ids") != list(model_ids):
                raise ValueError(f"{family} checkpoint model columns differ from transfer pool")
            if target_family is None:
                if any(identity.get(key) is not None for key in ("source", "target_family", "excluded_family")):
                    raise ValueError(f"{family} legacy checkpoint identity is inconsistent")
            else:
                if identity.get("source") != protocol:
                    raise ValueError(f"{family} checkpoint source protocol is wrong")
                if identity.get("target_family") != target_family:
                    raise ValueError(f"{family} checkpoint target family is wrong")
                if identity.get("excluded_family") != target_family:
                    raise ValueError(f"{family} checkpoint did not exclude the target family")


def registered_methods() -> tuple[str, ...]:
    learned = tuple(
        f"{family}_seed_{seed}"
        for family in paired.LEARNED_FAMILIES
        for seed in paired.TRAINING_SEEDS
    )
    return CORE_METHODS + learned


def _smooth_per_query_orders(
    raw_scores: np.ndarray,
    orders: np.ndarray,
    *,
    base_seed: int,
    split_seed: int,
) -> np.ndarray:
    ordered = np.take_along_axis(raw_scores, orders, axis=1)
    smoothed = smooth_nonconformity_scores(
        ordered,
        seed=base_seed,
        split_seed=split_seed,
        preserve_prefixes=True,
    )
    output = np.empty_like(smoothed)
    np.put_along_axis(output, orders, smoothed, axis=1)
    if np.any(np.diff(np.take_along_axis(output, orders, axis=1), axis=1) < 0.0):
        raise RuntimeError("prefix-preserving smoothing inverted a learned order")
    return output


def _learned_scores(
    ensembles: Mapping[str, Sequence[Mapping[str, Any]]],
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    *,
    base_seed: int,
    split_seed: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for family in paired.LEARNED_FAMILIES:
        for member in ensembles[family]:
            seed = int(member["identity"]["training_seed"])
            if family == "flexrouter":
                orders, _ = paired.full_flexrouter_orders(member["model"], embeddings)
            else:
                orders = paired.full_barp_orders(member["model"], embeddings)
            raw = per_query_orders_factorized_scores(probabilities, orders)
            output[f"{family}_seed_{seed}"] = _smooth_per_query_orders(
                raw,
                orders,
                base_seed=base_seed,
                split_seed=split_seed,
            )
    return output


def _score_all_queries(
    probabilities: np.ndarray,
    embeddings: np.ndarray,
    design_labels: np.ndarray,
    model_ids: Sequence[str],
    ensembles: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    base_seed: int,
    split_seed: int,
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray], Any]:
    nonlearned = paired._score_nonlearned(
        probabilities,
        design_labels,
        model_ids,
        base_seed=base_seed,
        split_seed=split_seed,
        preserve_score_prefixes=True,
    )
    wanted = {
        name: nonlearned[name]
        for name in ("fixed_tally_factorized", "racer_g", "racer_p")
    }
    candidates, profile = paired._score_chain_discount_candidates(
        probabilities,
        design_labels,
        model_ids,
        nonlearned["fixed_tally_factorized"],
        base_seed=base_seed,
        split_seed=split_seed,
    )
    if not np.array_equal(candidates[0], wanted["fixed_tally_factorized"]):
        raise RuntimeError("O2 x D2 discount-off endpoint mismatch")
    wanted.update(
        _learned_scores(
            ensembles,
            embeddings,
            probabilities,
            base_seed=base_seed,
            split_seed=split_seed,
        )
    )
    return wanted, candidates, profile


def _full_pool_record(labels: np.ndarray, query_ids: Sequence[str]) -> dict[str, Any]:
    misses = np.all(labels == 0, axis=1).astype(np.int8)
    return {
        "query_ids": list(query_ids),
        "miss_by_query": [int(value) for value in misses],
        "event_count": int(misses.sum()),
        "trial_count": int(len(misses)),
        "risk_floor": float(np.mean(misses)) if len(misses) else None,
    }


def _evaluate_calibration(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_ids: Sequence[str],
    target_test_scores: np.ndarray,
    target_test_labels: np.ndarray,
    target_test_ids: Sequence[str],
    model_ids: Sequence[str],
    alpha: float,
    *,
    calibration_origin: str,
    fit_origin: str,
    frozen_target_scores_sha256: str,
) -> dict[str, Any]:
    calibration = select_standard_crc_threshold(
        calibration_scores, calibration_labels, alpha
    )
    base: dict[str, Any] = {
        "fit_origin": fit_origin,
        "calibration_origin": calibration_origin,
        "calibration_query_ids": list(calibration_ids),
        "calibration_query_ids_sha256": paired.hash_json(list(calibration_ids)),
        "calibration_scores_sha256": _array_sha256(calibration_scores),
        "frozen_target_scores_sha256": frozen_target_scores_sha256,
        "target_test_query_ids": list(target_test_ids),
        "target_test_query_ids_sha256": paired.hash_json(list(target_test_ids)),
        "certified": bool(calibration.feasible),
        "non_certification_reason": calibration.non_certification_reason,
        "threshold": float(calibration.threshold) if calibration.feasible else None,
        "corrected_calibration_bound": float(calibration.crc_bound),
        "calibration_event_count": int(calibration.calibration_misses),
        "calibration_trial_count": int(len(calibration_labels)),
        "unavoidable_calibration_event_count": int(calibration.unavoidable_misses),
        "maximum_allowed_calibration_events": int(calibration.max_misses),
        "target_test_calls_by_query": None,
        "target_test_miss_by_query": None,
        "target_test_event_count": None,
        "target_test_trial_count": int(len(target_test_labels)),
        "target_test_mean_calls": None,
        "target_test_all_miss_risk": None,
        "selected_model_ids_by_query": None,
    }
    if not calibration.feasible:
        return base

    selected = target_test_scores <= float(calibration.threshold)
    calls = selected.sum(axis=1, dtype=int)
    misses = (~np.any(selected & target_test_labels.astype(bool), axis=1)).astype(np.int8)
    return {
        **base,
        "target_test_calls_by_query": [int(value) for value in calls],
        "target_test_miss_by_query": [int(value) for value in misses],
        "target_test_event_count": int(misses.sum()),
        "target_test_mean_calls": float(np.mean(calls)),
        "target_test_all_miss_risk": float(np.mean(misses)),
        "selected_model_ids_by_query": [
            [model_ids[int(index)] for index in np.flatnonzero(mask)]
            for mask in selected
        ],
    }


def _assert_identity_twins(source_result: Mapping[str, Any], target_result: Mapping[str, Any]) -> None:
    fields = (
        "certified",
        "non_certification_reason",
        "threshold",
        "corrected_calibration_bound",
        "calibration_event_count",
        "calibration_trial_count",
        "target_test_calls_by_query",
        "target_test_miss_by_query",
        "target_test_event_count",
        "target_test_mean_calls",
        "target_test_all_miss_risk",
        "selected_model_ids_by_query",
    )
    differences = [name for name in fields if source_result.get(name) != target_result.get(name)]
    if differences:
        raise RuntimeError(f"source equals target identity failed for fields {differences}")


def assert_registered_arm_presence(
    cells: Sequence[Mapping[str, Any]], expected_methods: Sequence[str]
) -> None:
    expected = set(expected_methods)
    if not cells:
        raise RuntimeError("ISSUE-52 emitted no alpha cells")
    for cell in cells:
        redraws = cell.get("redraws", [])
        if not redraws:
            raise RuntimeError(f"ISSUE-52 emitted no redraws for alpha={cell.get('alpha')}")
        for redraw in redraws:
            methods = redraw.get("methods", {})
            missing = sorted(expected - set(methods))
            unexpected = sorted(set(methods) - expected)
            if missing or unexpected:
                raise RuntimeError(
                    "ISSUE-52 registered arm mismatch at "
                    f"alpha={cell.get('alpha')}, redraw={redraw.get('redraw_id')}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            for name, record in methods.items():
                twins = record.get("calibrations", {})
                if set(twins) != set(CALIBRATION_MODES):
                    raise RuntimeError(f"{name} is missing a calibration transfer twin")
                hashes = {value.get("frozen_target_scores_sha256") for value in twins.values()}
                if len(hashes) != 1 or None in hashes:
                    raise RuntimeError(f"{name} calibration twins do not share frozen scores")


def run_transfer_pair(
    source_task: Any,
    target_task: Any,
    *,
    protocol: str,
    fit_mode: str,
    source_family: str,
    target_family: str,
    pool_size: int,
    pool_seed: int,
    redraw_start: int,
    redraw_end: int,
    knn_k: int,
    learned_ensembles: Mapping[str, Sequence[Mapping[str, Any]]],
    base_seed: int = paired.BASE_SEED,
    alphas: Sequence[float] = paired.ALPHA_GRID,
    knn_function: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    checkpoint_identity_target_family: str | None = "strict_target_family",
) -> dict[str, Any]:
    """Evaluate one ordered source and target pair over a redraw shard."""

    if protocol not in ("routereval", "routerbench"):
        raise ValueError("protocol must be routereval or routerbench")
    if fit_mode not in FIT_MODES:
        raise ValueError(f"fit_mode must be one of {FIT_MODES}")
    if not 0 <= redraw_start < redraw_end <= REDRAW_COUNT:
        raise ValueError(f"redraw range must lie within 0 through {REDRAW_COUNT}")
    if tuple(float(value) for value in alphas) != paired.ALPHA_GRID:
        raise ValueError(f"ISSUE-52 requires the canonical alpha grid {paired.ALPHA_GRID}")
    if base_seed != paired.BASE_SEED:
        raise ValueError(f"ISSUE-52 requires base_seed={paired.BASE_SEED}")
    if knn_k <= 0:
        raise ValueError("knn_k must be positive")

    _validate_task_shape(source_task, "source task")
    _validate_task_shape(target_task, "target task")
    if tuple(source_task.model_ids) != tuple(target_task.model_ids):
        raise ValueError("source and target model columns must be exactly identical")
    same_task = _same_task(source_task, target_task)
    if fit_mode == "in_domain" and not same_task:
        raise ValueError("in_domain fit requires source and target to be the same task")
    if fit_mode == "source_fitted" and source_family == target_family:
        raise ValueError("source_fitted mode must exclude the complete target task family")

    columns = subsample_pool(len(source_task.model_ids), pool_size, seed=pool_seed)
    model_ids = [str(source_task.model_ids[int(index)]) for index in columns]
    expected_checkpoint_family = (
        target_family
        if checkpoint_identity_target_family == "strict_target_family"
        else checkpoint_identity_target_family
    )
    _validate_ensembles(
        learned_ensembles,
        pool_size=pool_size,
        pool_seed=pool_seed,
        model_ids=model_ids,
        protocol=protocol,
        target_family=expected_checkpoint_family,
    )
    expected_methods = registered_methods()
    cells = [{"alpha": float(alpha), "redraws": []} for alpha in alphas]
    knn_impl = knn_function or paired._knn

    source_embeddings = np.asarray(source_task.embeddings, dtype=float)
    target_embeddings = np.asarray(target_task.embeddings, dtype=float)
    if source_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError("source and target embedding dimensions must be identical")

    for redraw_id in range(redraw_start, redraw_end):
        source_split = paired._target_split(len(source_task.query_ids), redraw_id, base_seed)
        target_split = paired._target_split(len(target_task.query_ids), redraw_id, base_seed)
        source_split_ids = _split_ids(source_task, source_split)
        target_split_ids = _split_ids(target_task, target_split)
        split_seed = base_seed + paired.SPLIT_STEP * redraw_id

        source_design_labels = _labels_at(
            source_task, source_split["design"], columns, "source design"
        )
        source_router_labels = _labels_at(
            source_task, source_split["router"], columns, "source router"
        )
        source_validation_labels = _labels_at(
            source_task, source_split["validation"], columns, "source validation"
        )
        source_calibration_labels = _labels_at(
            source_task,
            source_split["final_calibration"],
            columns,
            "source final calibration",
        )
        target_calibration_labels = _labels_at(
            target_task,
            target_split["final_calibration"],
            columns,
            "target final calibration",
        )
        target_test_labels = _labels_at(
            target_task, target_split["test"], columns, "target test"
        )

        source_probabilities, _ = knn_impl(
            source_embeddings[source_split["router"]],
            source_router_labels,
            source_embeddings,
            knn_k,
        )
        if same_task:
            target_probabilities = np.asarray(source_probabilities, dtype=float).copy()
        else:
            target_probabilities, _ = knn_impl(
                source_embeddings[source_split["router"]],
                source_router_labels,
                target_embeddings,
                knn_k,
            )
        source_probabilities = np.asarray(source_probabilities, dtype=float)
        target_probabilities = np.asarray(target_probabilities, dtype=float)
        if source_probabilities.shape != (len(source_task.query_ids), pool_size):
            raise ValueError("source KNN score shape differs from source task and pool")
        if target_probabilities.shape != (len(target_task.query_ids), pool_size):
            raise ValueError("target KNN score shape differs from target task and pool")

        source_scores, source_chain, source_profile = _score_all_queries(
            source_probabilities,
            source_embeddings,
            source_design_labels,
            model_ids,
            learned_ensembles,
            base_seed=base_seed,
            split_seed=split_seed,
        )
        if same_task:
            target_scores = {name: values.copy() for name, values in source_scores.items()}
            target_chain = {key: values.copy() for key, values in source_chain.items()}
            target_profile = source_profile
        else:
            target_scores, target_chain, target_profile = _score_all_queries(
                target_probabilities,
                target_embeddings,
                source_design_labels,
                model_ids,
                learned_ensembles,
                base_seed=base_seed,
                split_seed=split_seed,
            )
        if not np.array_equal(target_chain[0], target_scores["fixed_tally_factorized"]):
            raise RuntimeError("target O2 x D2 discount-off endpoint mismatch")
        if tuple(source_profile.order) != tuple(target_profile.order):
            raise RuntimeError("source and target chain profiles do not share one fitted order")

        source_full_pool = {
            "final_calibration": _full_pool_record(
                source_calibration_labels, source_split_ids["final_calibration"]
            )
        }
        target_full_pool = {
            "final_calibration": _full_pool_record(
                target_calibration_labels, target_split_ids["final_calibration"]
            ),
            "test": _full_pool_record(target_test_labels, target_split_ids["test"]),
        }

        for cell in cells:
            alpha = float(cell["alpha"])
            methods: dict[str, Any] = {}
            chosen_chain = paired._select_chain_discount_candidate(
                {
                    minimum: scores[source_split["validation"]]
                    for minimum, scores in source_chain.items()
                },
                source_validation_labels,
                np.arange(len(source_validation_labels), dtype=int),
                alpha,
            )
            minimum = int(chosen_chain["chosen"]["minimum_survivor_count"])
            selected_source_scores = dict(source_scores)
            selected_target_scores = dict(target_scores)
            selected_source_scores[paired.CHAIN_DISCOUNT_METHOD] = source_chain[minimum]
            selected_target_scores[paired.CHAIN_DISCOUNT_METHOD] = target_chain[minimum]

            for method in expected_methods:
                all_source_scores = selected_source_scores[method]
                all_target_scores = selected_target_scores[method]
                target_score_hash = _array_sha256(all_target_scores)
                if "_seed_" in method:
                    fit_origin = (
                        f"{protocol}:target_family_heldout_checkpoint:{target_family}"
                    )
                else:
                    fit_origin = (
                        f"{protocol}:{target_task.task}"
                        if fit_mode == "in_domain"
                        else f"{protocol}:{source_task.task}"
                    )
                common = {
                    "method_family": method.split("_seed_", 1)[0],
                    "training_seed": (
                        int(method.rsplit("_", 1)[1]) if "_seed_" in method else None
                    ),
                    "fit_origin": fit_origin,
                    "fit_mode": fit_mode,
                    "frozen_target_scores_sha256": target_score_hash,
                    "frozen_target_score_shape": list(all_target_scores.shape),
                    "preserve_prefixes": True,
                    "endpoint": {
                        "model_ids": model_ids,
                        "model_ids_sha256": paired.hash_json(model_ids),
                        "reaches_full_pool": True,
                    },
                }
                if method == paired.CHAIN_DISCOUNT_METHOD:
                    common.update({
                        "chosen_hyperparameter": {"minimum_survivor_count": minimum},
                        "hyperparameter_selection_origin": (
                            f"{protocol}:{source_task.task}:validation"
                        ),
                        "validation_trajectory": chosen_chain["validation_trajectory"],
                        "discount_off_endpoint_sha256": _array_sha256(target_chain[0]),
                        "endpoint_reference_method": "fixed_tally_factorized",
                        "endpoint_reproduction_exact": True,
                        "chain_discount_profile": chain_discount_diagnostics(
                            target_profile, minimum_survivor_count=minimum
                        ),
                    })
                source_calibrated = _evaluate_calibration(
                    all_source_scores[source_split["final_calibration"]],
                    source_calibration_labels,
                    source_split_ids["final_calibration"],
                    all_target_scores[target_split["test"]],
                    target_test_labels,
                    target_split_ids["test"],
                    model_ids,
                    alpha,
                    calibration_origin=f"{protocol}:{source_task.task}:final_calibration",
                    fit_origin=common["fit_origin"],
                    frozen_target_scores_sha256=target_score_hash,
                )
                target_recalibrated = _evaluate_calibration(
                    all_target_scores[target_split["final_calibration"]],
                    target_calibration_labels,
                    target_split_ids["final_calibration"],
                    all_target_scores[target_split["test"]],
                    target_test_labels,
                    target_split_ids["test"],
                    model_ids,
                    alpha,
                    calibration_origin=f"{protocol}:{target_task.task}:final_calibration",
                    fit_origin=common["fit_origin"],
                    frozen_target_scores_sha256=target_score_hash,
                )
                if same_task:
                    _assert_identity_twins(source_calibrated, target_recalibrated)
                methods[method] = {
                    **common,
                    "calibrations": {
                        "source_calibrated": source_calibrated,
                        "target_recalibrated": target_recalibrated,
                    },
                }

            cell["redraws"].append({
                "redraw_id": int(redraw_id),
                "split_seed": int(split_seed),
                "source_split_ids": source_split_ids,
                "target_split_ids": target_split_ids,
                "fit_label_access": {
                    "design_origin": f"{protocol}:{source_task.task}",
                    "router_origin": f"{protocol}:{source_task.task}",
                    "validation_origin": f"{protocol}:{source_task.task}",
                    "target_fit_label_roles": (
                        ["design", "router", "validation"]
                        if fit_mode == "in_domain"
                        else []
                    ),
                    "target_label_roles": ["final_calibration", "test"],
                },
                "source_complete_pool": source_full_pool,
                "target_complete_pool": target_full_pool,
                "methods": methods,
            })

    assert_registered_arm_presence(cells, expected_methods)
    return {
        "schema_version": SCHEMA_VERSION,
        "design": "ISSUE_52_fit_and_calibration_transfer",
        "protocol": protocol,
        "fit_mode": fit_mode,
        "source_benchmark": str(source_task.task),
        "target_benchmark": str(target_task.task),
        "source_family": source_family,
        "target_family": target_family,
        "source_equals_target": same_task,
        "pool_size": int(pool_size),
        "pool_seed": int(pool_seed),
        "model_ids": model_ids,
        "model_ids_sha256": paired.hash_json(model_ids),
        "sampled_model_column_indices": [int(value) for value in columns],
        "source_full_model_ids_sha256": paired.hash_json(list(source_task.model_ids)),
        "target_full_model_ids_sha256": paired.hash_json(list(target_task.model_ids)),
        "source_hashes": dict(source_task.source_hashes),
        "target_hashes": dict(target_task.source_hashes),
        "base_seed": int(base_seed),
        "knn_k": int(knn_k),
        "redraw_start": int(redraw_start),
        "redraw_end": int(redraw_end),
        "redraw_range": {
            "start_inclusive": int(redraw_start),
            "end_exclusive": int(redraw_end),
            "count": int(redraw_end - redraw_start),
            "registered_total": REDRAW_COUNT,
        },
        "target_grid": [float(value) for value in alphas],
        "calibration_modes": list(CALIBRATION_MODES),
        "method_identities": list(expected_methods),
        "preserve_prefixes": True,
        "checkpoint_identities": {
            family: [dict(member["identity"]) for member in learned_ensembles[family]]
            for family in paired.LEARNED_FAMILIES
        },
        "cells": cells,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("routereval", "routerbench"), required=True)
    parser.add_argument("--fit-mode", choices=FIT_MODES, required=True)
    parser.add_argument("--source-benchmark")
    parser.add_argument("--target-benchmark", required=True)
    parser.add_argument("--leaderboard", type=Path)
    parser.add_argument("--router-dir", type=Path)
    parser.add_argument("--routerbench-pickle", type=Path)
    parser.add_argument("--embedding-archive", type=Path)
    parser.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--pool-size", required=True, type=int)
    parser.add_argument("--pool-seed", type=int, default=paired.BASE_SEED)
    parser.add_argument("--knn-k", type=int, default=40)
    parser.add_argument("--redraw-start", required=True, type=int)
    parser.add_argument("--redraw-end", required=True, type=int)
    parser.add_argument("--flexrouter-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--barp-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def _validate_output_identity(path: Path, host_id: str, run_stamp: str) -> None:
    if not STAMP_PATTERN.fullmatch(run_stamp):
        raise ValueError("run stamp must use YYYYMMDDTHHMMSSZ UTC format")
    token = f"issue52_{host_id}_{run_stamp}"
    if token not in str(path):
        raise ValueError(f"output path must contain collision-safe token {token}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        _validate_output_identity(arguments.out, arguments.host_id, arguments.run_stamp)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if arguments.out.exists():
        raise SystemExit(f"refusing to overwrite immutable result {arguments.out}")

    source_benchmark = arguments.source_benchmark
    if arguments.fit_mode == "in_domain":
        if source_benchmark not in (None, arguments.target_benchmark):
            raise SystemExit("in_domain source benchmark must equal the target benchmark")
        source_benchmark = arguments.target_benchmark
    elif not source_benchmark:
        raise SystemExit("source_fitted mode requires --source-benchmark")

    load_options = {
        "leaderboard": arguments.leaderboard,
        "router_dir": arguments.router_dir,
        "routerbench_pickle": arguments.routerbench_pickle,
        "embedding_archive": arguments.embedding_archive,
        "embedding_manifest": arguments.embedding_manifest,
    }
    try:
        source_task = paired._load_task(arguments.protocol, source_benchmark, **load_options)
        target_task = (
            source_task
            if source_benchmark == arguments.target_benchmark
            else paired._load_task(arguments.protocol, arguments.target_benchmark, **load_options)
        )
        source_family = _task_family(arguments.protocol, source_benchmark)
        target_family = _task_family(arguments.protocol, arguments.target_benchmark)
        from components.stage1_task_adapters import checkpoint_target_family
        expected_checkpoint_family = checkpoint_target_family(
            arguments.protocol, arguments.target_benchmark
        )
        if tuple(source_task.model_ids) != tuple(target_task.model_ids):
            raise ValueError("source and target model columns must be exactly identical")
        columns = subsample_pool(
            len(source_task.model_ids), arguments.pool_size, seed=arguments.pool_seed
        )
        model_ids = [str(source_task.model_ids[int(index)]) for index in columns]
        ensembles = {
            family: paired.load_checkpoint_ensemble(
                directory,
                family,
                pool_size=arguments.pool_size,
                pool_seed=arguments.pool_seed,
                model_ids=model_ids,
                device=arguments.device,
                expected_source=(arguments.protocol if expected_checkpoint_family is not None else None),
                expected_target_family=expected_checkpoint_family,
            )
            for family, directory in (
                ("flexrouter", arguments.flexrouter_checkpoint_dir),
                ("barp", arguments.barp_checkpoint_dir),
            )
        }
        payload = run_transfer_pair(
            source_task,
            target_task,
            protocol=arguments.protocol,
            fit_mode=arguments.fit_mode,
            source_family=source_family,
            target_family=target_family,
            pool_size=arguments.pool_size,
            pool_seed=arguments.pool_seed,
            redraw_start=arguments.redraw_start,
            redraw_end=arguments.redraw_end,
            knn_k=arguments.knn_k,
            learned_ensembles=ensembles,
            checkpoint_identity_target_family=expected_checkpoint_family,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    payload["artifact_manifest"] = {
        "owning_host": arguments.host_id,
        "actual_hostname": socket.gethostname(),
        "run_stamp_utc": arguments.run_stamp,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_path": str(arguments.out),
    }
    paired.atomic_write_json(arguments.out, payload)
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
