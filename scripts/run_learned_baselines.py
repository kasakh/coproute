"""Run the preregistered FlexRouter and BaRP learned baselines.

The runner has two intentionally separate modes. RouterEval trains on the
external BBH, MATH, and GPQA tasks, freezes one learned router, and evaluates
native and final-calibrated views on IFEval or MuSR. RouterBench trains native
BaRP on the four registered in-distribution task families and reports its
single-action quality and cost frontier on MBPP and HellaSwag.

Every run is written through a temporary directory and atomically renamed.
Existing output directories are never replaced.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # NumPy-only helpers stay importable without torch.
    torch = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from components.barp_baseline import (  # noqa: E402
        BaRPBaseline,
        BaRPConfig,
        bandit_reward,
        chosen_action_feedback,
        reinforce_loss,
        sample_categorical_actions,
        sample_simplex_preferences,
    )
    from components.flexrouter_baseline import (  # noqa: E402
        FlexRouter,
        FlexRouterTrainingConfig,
    )
except ModuleNotFoundError:  # pragma: no cover - torch-free helper mode
    if torch is not None:
        raise
    BaRPBaseline = BaRPConfig = bandit_reward = chosen_action_feedback = None  # type: ignore[assignment]
    reinforce_loss = sample_categorical_actions = sample_simplex_preferences = None  # type: ignore[assignment]
    FlexRouter = FlexRouterTrainingConfig = None  # type: ignore[assignment]
from components.data_protocol import (  # noqa: E402
    PRIMARY_COVERAGE_POLICY,
    assign_racer_splits,
    derive_coverage_labels,
    load_routerbench,
)
from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS  # noqa: E402
from components.racer_protocol import select_crc_threshold  # noqa: E402
from components.routereval_protocol import audit_features, subsample_pool  # noqa: E402
from scripts.build_routerbench_embedding_archive import (  # noqa: E402
    MANIFEST_FILENAME as ROUTERBENCH_EMBEDDING_MANIFEST,
    PROTOCOL_NAME as ROUTERBENCH_EMBEDDING_PROTOCOL,
    load_archive_manifest,
)

SCHEMA_VERSION = 1
PROTOCOL_NAME = "preregistered_learned_baselines_v1"
EXTERNAL_TASKS = ("bbh", "math", "gpqa")
TARGET_TASKS = ("ifeval", "musr")
ROUTERBENCH_TRAIN_TASKS = ("mmlu", "arc-challenge", "winogrande", "gsm8k")
ROUTERBENCH_THREE_TASK_SENSITIVITY = ("mmlu", "arc-challenge", "winogrande")
ROUTERBENCH_TARGET_TASKS = ("mbpp", "hellaswag")
POOL_SIZES = (100, 200, 300, 500)
POOL_SEEDS = (7001, 7002, 7003, 7004, 7005)
TRAINING_SEEDS = (9101, 9102, 9103, 9104, 9105)
TARGET_REDRAWS = 40
TARGET_SPLIT_BASE_SEED = 7001
EXTERNAL_SPLIT_SEED = 20260903
MAX_DEPTH = 10
FLEX_TAU_GRID = (0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
ALPHA_GRID = (
    0.02,
    0.025,
    0.03,
    0.04,
    0.05,
    0.06,
    0.075,
    0.09,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.18,
)
PREFERENCE_GRID = tuple((1.0 - value, value) for value in np.linspace(0.0, 1.0, 11))

ROUTEREVAL_KEYS: Mapping[str, tuple[str, ...]] = {
    "bbh": (
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
    ),
    "math": (
        "math_algebra_hard", "math_counting_and_prob_hard", "math_geometry_hard",
        "math_intermediate_algebra_hard", "math_num_theory_hard",
        "math_prealgebra_hard", "math_precalculus_hard",
    ),
    "gpqa": ("gpqa_diamond", "gpqa_extended", "gpqa_main"),
    "ifeval": ("ifeval",),
    "musr": ("musr_murder_mysteries", "musr_object_placements", "musr_team_allocation"),
}


@dataclass(frozen=True)
class LearnedTaskData:
    """One aligned task with stable IDs and source provenance."""

    task: str
    query_ids: tuple[str, ...]
    embeddings: np.ndarray
    correctness: np.ndarray
    model_ids: tuple[str, ...]
    prompts: tuple[str, ...]
    feature_audit: Mapping[str, Any]
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class FrozenScores:
    """A frozen nested learned score family over at most ten models per query."""

    orders: tuple[tuple[int, ...], ...]
    scores: np.ndarray
    auxiliary: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class RouterEvalCheckpoint:
    """One externally trained checkpoint shared by all RouterEval targets."""

    model: FlexRouter | BaRPBaseline
    method: str
    columns: tuple[int, ...]
    model_ids: tuple[str, ...]
    full_model_ids: tuple[str, ...]
    training_metadata: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()




def implementation_source_hashes() -> dict[str, str]:
    """Hash every frozen protocol and implementation source used by this runner."""

    paths = {
        "runner": Path(__file__).resolve(),
        "protocol": ROOT / "docs" / "LEARNED_BASELINES_PROTOCOL.md",
        "flexrouter_component": ROOT / "components" / "flexrouter_baseline.py",
        "barp_component": ROOT / "components" / "barp_baseline.py",
        "calibration_component": ROOT / "components" / "racer_protocol.py",
        "routereval_component": ROOT / "components" / "routereval_protocol.py",
        "routerbench_component": ROOT / "components" / "data_protocol.py",
        "routerbench_pool_contract": ROOT / "components" / "routerbench_protocol.py",
        "routerbench_embedding_builder": ROOT / "scripts" / "build_routerbench_embedding_archive.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _largest_remainder_counts(total: int, fractions: Sequence[float]) -> np.ndarray:
    values = np.asarray(fractions, dtype=float)
    if total < 0 or values.ndim != 1 or np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("fractions must be non-negative and sum to one")
    exact = values * int(total)
    counts = np.floor(exact).astype(int)
    remainder = int(total) - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def stable_hash_split(
    query_ids: Sequence[str],
    fractions: Mapping[str, float],
    *,
    seed: int,
    namespace: str,
) -> dict[str, np.ndarray]:
    """Hash-rank stable IDs and allocate them by largest remainder."""

    ids = tuple(str(value) for value in query_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("query IDs must be unique")
    names = tuple(fractions)
    counts = _largest_remainder_counts(len(ids), tuple(float(fractions[name]) for name in names))

    def rank(item: tuple[int, str]) -> tuple[str, str]:
        _, query_id = item
        text = f"{namespace}\x1f{seed}\x1f{query_id}".encode("utf-8")
        return hashlib.sha256(text).hexdigest(), query_id

    ordered = [index for index, _ in sorted(enumerate(ids), key=rank)]
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name, count in zip(names, counts):
        result[name] = np.asarray(ordered[offset : offset + int(count)], dtype=int)
        offset += int(count)
    if offset != len(ids):
        raise RuntimeError("stable split did not allocate every query")
    return result


def target_redraw_split(n_queries: int, redraw: int) -> dict[str, np.ndarray]:
    """Create the exact 30/20/20/15/15 repaired target partition."""

    if n_queries <= 0 or redraw < 0:
        raise ValueError("n_queries must be positive and redraw must be non-negative")
    permutation = np.random.default_rng(TARGET_SPLIT_BASE_SEED + 13 * redraw).permutation(n_queries)
    n_design = int(round(0.30 * n_queries))
    n_router = int(round(0.20 * n_queries))
    remaining = n_queries - n_design - n_router
    n_calibration = int(round(remaining * (0.20 / 0.50)))
    n_validation = int(round(remaining * (0.15 / 0.50)))
    counts = (
        n_design,
        n_router,
        n_calibration,
        n_validation,
        remaining - n_calibration - n_validation,
    )
    names = ("design", "router", "final_calibration", "validation", "test")
    output: dict[str, np.ndarray] = {}
    offset = 0
    for name, count in zip(names, counts):
        output[name] = permutation[offset : offset + count]
        offset += count
    if offset != n_queries or len(np.unique(np.concatenate(tuple(output.values())))) != n_queries:
        raise RuntimeError("target redraw split is incomplete or overlapping")
    return output


def l2_normalize_rows(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embeddings contain a zero-norm row")
    return matrix / norms


def _routereval_query_ids(task: str, count: int) -> tuple[str, ...]:
    return tuple(f"routereval:{task}:{index}" for index in range(count))


def load_routereval_task(
    leaderboard_path: str | Path,
    router_dataset_path: str | Path,
    task: str,
    *,
    threshold: float = 0.5,
) -> LearnedTaskData:
    """Load one preregistered RouterEval task with exact feature row alignment."""

    if task not in ROUTEREVAL_KEYS:
        raise KeyError(f"task {task!r} has no preregistered RouterEval alignment map")
    leaderboard_path = Path(leaderboard_path)
    router_dataset_path = Path(router_dataset_path)
    with leaderboard_path.open("rb") as handle:
        leaderboard = pickle.load(handle)
    with router_dataset_path.open("rb") as handle:
        router = pickle.load(handle)

    missing = [key for key in ROUTEREVAL_KEYS[task] if key not in leaderboard["data"]]
    if missing:
        raise KeyError(f"{task}: missing leaderboard keys {missing}")
    correctness = np.concatenate(
        [np.asarray(leaderboard["data"][key]["correctness"], dtype=float) for key in ROUTEREVAL_KEYS[task]],
        axis=0,
    )
    if correctness.ndim != 2 or correctness.size == 0 or not np.all(np.isfinite(correctness)):
        raise ValueError(f"{task}: correctness must be finite and two-dimensional")
    binary = (correctness >= threshold).astype(np.int8)
    model_ids = tuple(str(value) for value in np.asarray(leaderboard["model"]).ravel())
    if len(model_ids) != binary.shape[1] or len(set(model_ids)) != len(model_ids):
        raise ValueError(f"{task}: invalid or duplicate model IDs")

    split = router["split_index"]
    embedding = router["embedding"]
    prompt = router["prompt"]
    indices = np.concatenate(
        [np.asarray(split[f"{name}_indices"], dtype=int) for name in ("train", "val", "test")]
    )
    embeddings = np.concatenate(
        [np.asarray(embedding[f"{name}_embed"], dtype=float) for name in ("train", "val", "test")],
        axis=0,
    )
    prompts = np.concatenate(
        [np.asarray(prompt[f"{name}_prompt"], dtype=object) for name in ("train", "val", "test")]
    )
    if len(indices) != binary.shape[0] or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{task}: feature indices do not uniquely cover correctness rows")
    order = np.argsort(indices, kind="stable")
    if not np.array_equal(indices[order], np.arange(binary.shape[0])):
        raise ValueError(f"{task}: feature indices are not exactly zero through n minus one")
    aligned_embeddings = embeddings[order]
    aligned_prompts = tuple(str(value) for value in prompts[order])
    feature_audit = audit_features(aligned_prompts)
    if feature_audit["degenerate"] or feature_audit["exceeds_threshold"]:
        raise ValueError(f"{task}: failed feature audit: {feature_audit}")
    return LearnedTaskData(
        task=task,
        query_ids=_routereval_query_ids(task, binary.shape[0]),
        embeddings=l2_normalize_rows(aligned_embeddings),
        correctness=binary,
        model_ids=model_ids,
        prompts=aligned_prompts,
        feature_audit={**feature_audit, "threshold": threshold, "alignment": "original_index_exact"},
        source_hashes={
            "leaderboard": sha256_file(leaderboard_path),
            "router_dataset": sha256_file(router_dataset_path),
        },
    )


def _validate_routereval_task_set(
    tasks: Mapping[str, LearnedTaskData],
    *,
    target_tasks: Sequence[str],
    expected_embedding_dim: int | None = 768,
) -> tuple[str, ...]:
    required = set(EXTERNAL_TASKS + tuple(target_tasks))
    if set(tasks) != required:
        raise ValueError(f"RouterEval bundle must contain exactly {sorted(required)}")
    first = tasks[EXTERNAL_TASKS[0]].model_ids
    dimensions = set()
    for name, task in tasks.items():
        if task.model_ids != first:
            raise ValueError(f"{name}: model columns differ from the shared RouterEval pool")
        if task.correctness.shape != (len(task.query_ids), len(first)):
            raise ValueError(f"{name}: correctness shape does not match IDs and model pool")
        if task.embeddings.shape[0] != len(task.query_ids):
            raise ValueError(f"{name}: embedding rows do not match query IDs")
        dimensions.add(task.embeddings.shape[1])
    if len(dimensions) != 1:
        raise ValueError("RouterEval tasks do not share one embedding dimension")
    dimension = next(iter(dimensions))
    if expected_embedding_dim is not None and dimension != expected_embedding_dim:
        raise ValueError(f"expected {expected_embedding_dim} embedding features, found {dimension}")
    return first


def _validate_routereval_tasks(
    tasks: Mapping[str, LearnedTaskData],
    *,
    target_task: str,
    expected_embedding_dim: int | None = 768,
) -> tuple[str, ...]:
    """Validate the legacy one-target RouterEval bundle."""

    return _validate_routereval_task_set(
        tasks,
        target_tasks=(target_task,),
        expected_embedding_dim=expected_embedding_dim,
    )

def _split_ids(task: LearnedTaskData, split: Mapping[str, np.ndarray]) -> dict[str, list[str]]:
    return {name: [task.query_ids[int(index)] for index in values] for name, values in split.items()}


def _barp_validation_reward(
    model: BaRPBaseline,
    embeddings: np.ndarray,
    qualities: np.ndarray,
    costs: np.ndarray,
    preferences: Sequence[Sequence[float]],
) -> float:
    scores = []
    for preference in preferences:
        actions = model.predict(embeddings, preference).numpy()
        rows = np.arange(len(actions))
        normalized_cost = np.clip(costs[rows, actions] / model.config.tau, 0.0, 1.0)
        reward = preference[0] * qualities[rows, actions] - preference[1] * normalized_cost
        scores.append(float(np.mean(reward)))
    return float(np.mean(scores))


def fit_barp_with_early_stopping(
    model: BaRPBaseline,
    train_embeddings: np.ndarray,
    train_qualities: np.ndarray,
    train_costs: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_qualities: np.ndarray,
    validation_costs: np.ndarray,
    *,
    validation_preferences: Sequence[Sequence[float]],
    patience: int = 15,
) -> BaRPBaseline:
    """Fit BaRP with persistent Adam state and validation-only early stopping."""

    if patience <= 0:
        raise ValueError("patience must be positive")
    features, qualities, costs = model._training_tensors(  # type: ignore[attr-defined]
        train_embeddings, train_qualities, train_costs
    )
    generator = torch.Generator(device=model.device.type)
    generator.manual_seed(model.config.seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=model.config.learning_rate)
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_reward = -math.inf
    best_epoch: int | None = None
    stale = 0

    for epoch in range(model.config.max_epochs):
        model.policy.train()
        permutation = torch.randperm(features.shape[0], generator=generator, device=model.device)
        totals = {"loss": 0.0, "reward": 0.0, "entropy": 0.0, "seen": 0}
        for start in range(0, features.shape[0], model.config.batch_size):
            indices = permutation[start : start + model.config.batch_size]
            batch_features = features.index_select(0, indices)
            batch_qualities = qualities.index_select(0, indices)
            batch_costs = costs if costs.ndim == 1 else costs.index_select(0, indices)
            preferences = sample_simplex_preferences(
                len(indices), generator=generator, device=model.device, dtype=batch_features.dtype
            )
            logits = model.policy(batch_features, preferences)
            actions = sample_categorical_actions(logits, generator=generator)
            selected_quality, selected_cost = chosen_action_feedback(batch_qualities, batch_costs, actions)
            rewards = bandit_reward(selected_quality, selected_cost, preferences, tau=model.config.tau)
            loss, diagnostics = reinforce_loss(
                logits, actions, rewards, entropy_beta=model.config.entropy_beta
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(len(indices))
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["reward"] += float(rewards.mean().detach().cpu()) * count
            totals["entropy"] += float(diagnostics["entropy"].cpu()) * count
            totals["seen"] += count
        model.is_fitted = True
        validation_reward = _barp_validation_reward(
            model,
            validation_embeddings,
            validation_qualities,
            validation_costs,
            validation_preferences,
        )
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / totals["seen"],
            "mean_reward": totals["reward"] / totals["seen"],
            "entropy": totals["entropy"] / totals["seen"],
            "validation_mean_reward": validation_reward,
        }
        if not all(math.isfinite(float(value)) for key, value in row.items() if key != "epoch"):
            raise ValueError("BaRP training produced a non-finite history value")
        history.append(row)
        if validation_reward > best_reward:
            best_reward = validation_reward
            best_epoch = epoch
            best_state = copy.deepcopy(model.policy.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_epoch is None:
        raise RuntimeError("BaRP training did not produce a checkpoint")
    model.policy.load_state_dict(best_state)
    model.policy.eval()
    model.training_history = history
    model.best_epoch = best_epoch  # type: ignore[attr-defined]
    model.best_validation_reward = best_reward  # type: ignore[attr-defined]
    model.observed_model_outcomes = int(len(train_embeddings) * len(history))  # type: ignore[attr-defined]
    return model


def train_learned_model(
    method: str,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_labels: np.ndarray,
    model_ids: Sequence[str],
    *,
    training_seed: int,
    train_costs: np.ndarray | None = None,
    validation_costs: np.ndarray | None = None,
    tau: float = 1.0,
    max_epochs: int = 100,
    patience: int = 15,
    device: str = "cpu",
    routerbench_preferences: bool = False,
) -> FlexRouter | BaRPBaseline:
    """Train one registered learned model without target-task labels."""

    if method == "flexrouter":
        config = FlexRouterTrainingConfig(max_epochs=max_epochs, patience=patience)
        model = FlexRouter(
            train_embeddings.shape[1],
            len(model_ids),
            seed=training_seed,
            device=device,
            embedding_metadata={
                "source": "audited_precomputed_embeddings",
                "row_normalization": "l2",
                "model_ids_sha256": _hash_json(list(model_ids)),
            },
        )
        return model.fit(
            train_embeddings,
            train_labels,
            validation_embeddings,
            validation_labels,
            config=config,
        )
    if method != "barp":
        raise ValueError("method must be flexrouter or barp")
    config = BaRPConfig(tau=tau, max_epochs=max_epochs, seed=training_seed)
    model = BaRPBaseline(
        train_embeddings.shape[1],
        len(model_ids),
        config=config,
        model_ids=model_ids,
        device=device,
    )
    train_cost_matrix = (
        np.zeros_like(train_labels, dtype=float) if train_costs is None else np.asarray(train_costs, dtype=float)
    )
    validation_cost_matrix = (
        np.zeros_like(validation_labels, dtype=float)
        if validation_costs is None
        else np.asarray(validation_costs, dtype=float)
    )
    preferences = PREFERENCE_GRID if routerbench_preferences else ((1.0, 0.0),)
    return fit_barp_with_early_stopping(
        model,
        train_embeddings,
        train_labels,
        train_cost_matrix,
        validation_embeddings,
        validation_labels,
        validation_cost_matrix,
        validation_preferences=preferences,
        patience=patience,
    )


def score_learned_model(
    model: FlexRouter | BaRPBaseline,
    method: str,
    embeddings: np.ndarray,
    n_models: int,
    *,
    max_depth: int = MAX_DEPTH,
) -> FrozenScores:
    """Freeze one learned order and nested score array for every query."""

    depth = min(max_depth, n_models)
    scores = np.full((len(embeddings), n_models), np.inf, dtype=float)
    orders: list[tuple[int, ...]] = []
    auxiliary: list[Mapping[str, Any]] = []
    if method == "flexrouter":
        if not isinstance(model, FlexRouter):
            raise TypeError("flexrouter scoring requires FlexRouter")
        qualities = model.predict_quality(embeddings)
        for row in range(len(embeddings)):
            path = model.select_from_qualities(qualities[row], max_size=depth)
            if len(path.order) != depth or not path.gains or path.gains[0] <= 0.0:
                raise ValueError(f"FlexRouter query {row} did not produce a complete positive path")
            order = tuple(int(value) for value in path.order)
            scores[row, list(order)] = np.asarray(path.scores, dtype=float)
            orders.append(order)
            auxiliary.append(
                {
                    "gains": [float(value) for value in path.gains],
                    "relative_gains": [float(value) for value in path.relative_gains],
                    "quality": [float(value) for value in qualities[row].detach().cpu().tolist()],
                }
            )
    elif method == "barp":
        if not isinstance(model, BaRPBaseline):
            raise TypeError("barp scoring requires BaRPBaseline")
        logits = model.logits(embeddings, [1.0, 0.0]).numpy()
        ranking = np.argsort(-logits, axis=1, kind="stable")[:, :depth]
        for row in range(len(embeddings)):
            order = tuple(int(value) for value in ranking[row])
            gaps = logits[row, order[0]] - logits[row, list(order)]
            scores[row, list(order)] = gaps
            orders.append(order)
            auxiliary.append(
                {
                    "eligible_logits": [float(logits[row, index]) for index in order],
                    "logit_gaps": [float(value) for value in gaps],
                }
            )
    else:
        raise ValueError("method must be flexrouter or barp")
    return FrozenScores(tuple(orders), scores, tuple(auxiliary))


def _success_events(
    orders: Sequence[Sequence[int]], labels: np.ndarray, rows: np.ndarray, depth: int
) -> list[bool]:
    return [bool(np.any(labels[index, list(orders[index][:depth])] == 1)) for index in rows]


def fixed_depth_record(
    frozen: FrozenScores,
    labels: np.ndarray,
    test_rows: np.ndarray,
    query_ids: Sequence[str],
    model_ids: Sequence[str],
) -> dict[str, Any]:
    depths = range(1, min(MAX_DEPTH, labels.shape[1]) + 1)
    records = []
    for depth in depths:
        events = _success_events(frozen.orders, labels, test_rows, depth)
        records.append(
            {
                "depth": depth,
                "success": float(np.mean(events)),
                "zero_correct_rate": float(1.0 - np.mean(events)),
                "mean_calls": float(depth),
                "event_numerator": int(sum(events)),
                "event_denominator": len(events),
                "query_events": [
                    {
                        "query_id": query_ids[int(row)],
                        "selected_model_ids": [model_ids[index] for index in frozen.orders[int(row)][:depth]],
                        "success": event,
                    }
                    for row, event in zip(test_rows, events)
                ],
            }
        )
    return {
        "comparison_kind": "native_fixed_depth",
        "risk_certificate": False,
        "depths": records,
        "success_auc": float(np.mean([record["success"] for record in records])),
    }


def flex_adaptive_record(
    frozen: FrozenScores,
    labels: np.ndarray,
    test_rows: np.ndarray,
    query_ids: Sequence[str],
    model_ids: Sequence[str],
) -> list[dict[str, Any]]:
    records = []
    for tau in FLEX_TAU_GRID:
        query_records = []
        calls = []
        events = []
        histogram: dict[str, int] = {}
        for row in test_rows:
            relatives = frozen.auxiliary[int(row)]["relative_gains"]
            count = sum(float(value) >= tau for value in relatives)
            selected = frozen.orders[int(row)][:count]
            success = bool(selected and np.any(labels[int(row), list(selected)] == 1))
            calls.append(count)
            events.append(success)
            histogram[str(count)] = histogram.get(str(count), 0) + 1
            query_records.append(
                {
                    "query_id": query_ids[int(row)],
                    "selected_model_ids": [model_ids[index] for index in selected],
                    "calls": count,
                    "success": success,
                }
            )
        records.append(
            {
                "tau": tau,
                "success": float(np.mean(events)),
                "standard_all_miss_risk": float(1.0 - np.mean(events)),
                "mean_calls": float(np.mean(calls)),
                "median_calls": float(np.median(calls)),
                "p90_calls": float(np.quantile(calls, 0.90)),
                "empty_set_rate": float(np.mean(np.asarray(calls) == 0)),
                "subset_size_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
                "event_numerator": int(sum(events)),
                "event_denominator": len(events),
                "risk_certificate": False,
                "query_events": query_records,
            }
        )
    return records


def calibrated_record(
    frozen: FrozenScores,
    labels: np.ndarray,
    calibration_rows: np.ndarray,
    test_rows: np.ndarray,
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    alpha: float,
) -> dict[str, Any]:
    """Calibrate one frozen score family once and evaluate standard all-miss loss."""

    critical = []
    calibration_events = []
    for row in calibration_rows:
        eligible = frozen.orders[int(row)]
        correct_scores = [frozen.scores[int(row), index] for index in eligible if labels[int(row), index] == 1]
        value = min(correct_scores) if correct_scores else math.inf
        critical.append(value)
        calibration_events.append(
            {
                "query_id": query_ids[int(row)],
                "critical_score": float(value) if math.isfinite(value) else None,
                "unavoidable_miss": not math.isfinite(value),
            }
        )
    calibration = select_crc_threshold(np.asarray(critical, dtype=float), alpha)
    base = {
        "alpha": alpha,
        "final_calibration_count": 1,
        "n_final_calibration": len(calibration_rows),
        "certified": bool(calibration.feasible),
        "risk_validity_claimed": bool(calibration.feasible),
        "threshold": float(calibration.threshold) if calibration.feasible else None,
        "maximum_allowed_misses": int(calibration.max_misses),
        "calibration_misses": int(calibration.calibration_misses),
        "unavoidable_calibration_misses": int(calibration.unavoidable_misses),
        "corrected_bound": float(calibration.crc_bound),
        "non_certification_reason": calibration.non_certification_reason,
        "calibration_events": calibration_events,
    }
    if not calibration.feasible:
        return {
            **base,
            "mean_calls": None,
            "median_calls": None,
            "p90_calls": None,
            "standard_all_miss_risk": None,
            "event_numerator": None,
            "event_denominator": len(test_rows),
            "query_events": [],
        }

    query_records = []
    calls = []
    misses = []
    threshold = calibration.threshold
    for row in test_rows:
        selected = tuple(
            index for index in frozen.orders[int(row)] if frozen.scores[int(row), index] <= threshold
        )
        miss = not bool(selected and np.any(labels[int(row), list(selected)] == 1))
        calls.append(len(selected))
        misses.append(miss)
        query_records.append(
            {
                "query_id": query_ids[int(row)],
                "selected_model_ids": [model_ids[index] for index in selected],
                "selected_scores": [float(frozen.scores[int(row), index]) for index in selected],
                "calls": len(selected),
                "all_miss": miss,
            }
        )
    return {
        **base,
        "mean_calls": float(np.mean(calls)),
        "median_calls": float(np.median(calls)),
        "p90_calls": float(np.quantile(calls, 0.90)),
        "standard_all_miss_risk": float(np.mean(misses)),
        "event_numerator": int(sum(misses)),
        "event_denominator": len(misses),
        "query_events": query_records,
    }


def _prediction_records(
    frozen: FrozenScores, query_ids: Sequence[str], model_ids: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        {
            "query_id": query_ids[row],
            "learned_order": [model_ids[index] for index in frozen.orders[row]],
            "learned_model_indices": list(frozen.orders[row]),
            "learned_scores": [float(frozen.scores[row, index]) for index in frozen.orders[row]],
            **dict(frozen.auxiliary[row]),
        }
        for row in range(len(query_ids))
    ]


def _save_checkpoint(model: FlexRouter | BaRPBaseline, path: Path) -> None:
    model.save(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_immutable_run(
    output_dir: str | Path,
    result: dict[str, Any],
    model: FlexRouter | BaRPBaseline,
) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable output directory {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=destination.parent))
    try:
        checkpoint = temporary / "checkpoint.pt"
        _save_checkpoint(model, checkpoint)
        history = list(model.training_history)
        _atomic_write_json(temporary / "training_history.json", history)
        _atomic_write_json(temporary / "results.json", result)
        files = {
            name: sha256_file(temporary / name)
            for name in ("checkpoint.pt", "training_history.json", "results.json")
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_NAME,
            "immutable": True,
            "files": files,
            "source_hashes": result["source_hashes"],
            "checkpoint_sha256": files["checkpoint.pt"],
            "result_identity_sha256": _hash_json(result["identity"]),
        }
        _atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _write_immutable_unavailable_run(
    output_dir: str | Path,
    result: dict[str, Any],
) -> Path:
    """Write an expected unavailable cell without inventing a checkpoint."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable output directory {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=destination.parent))
    try:
        _atomic_write_json(temporary / "results.json", result)
        files = {"results.json": sha256_file(temporary / "results.json")}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_NAME,
            "immutable": True,
            "run_status": "unavailable",
            "files": files,
            "source_hashes": result["source_hashes"],
            "result_identity_sha256": _hash_json(result["identity"]),
            "failure_sha256": _hash_json(result["failure"]),
            "checkpoint_sha256": None,
        }
        _atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _write_immutable_multi_target_run(
    output_dir: str | Path,
    results: Mapping[str, dict[str, Any]],
    training_metadata: Mapping[str, Any],
    model: FlexRouter | BaRPBaseline,
) -> Path:
    """Write one checkpoint and both target evaluations as one immutable run."""

    if set(results) != set(TARGET_TASKS):
        raise ValueError(f"multi-target results must contain exactly {TARGET_TASKS}")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable output directory {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=destination.parent))
    try:
        checkpoint = temporary / "checkpoint.pt"
        _save_checkpoint(model, checkpoint)
        _atomic_write_json(temporary / "training_history.json", list(model.training_history))
        _atomic_write_json(temporary / "training_metadata.json", training_metadata)
        for target_task in TARGET_TASKS:
            _atomic_write_json(temporary / f"results_{target_task}.json", results[target_task])
        names = (
            "checkpoint.pt",
            "training_history.json",
            "training_metadata.json",
            *(f"results_{target_task}.json" for target_task in TARGET_TASKS),
        )
        files = {name: sha256_file(temporary / name) for name in names}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_NAME,
            "layout": "shared_checkpoint_multi_target",
            "immutable": True,
            "files": files,
            "source_hashes": {
                "training": training_metadata["source_hashes"],
                "targets": {
                    target_task: results[target_task]["source_hashes"]["data"][target_task]
                    for target_task in TARGET_TASKS
                },
            },
            "checkpoint_sha256": files["checkpoint.pt"],
            "training_identity_sha256": _hash_json(training_metadata["identity"]),
            "result_identity_sha256": {
                target_task: _hash_json(results[target_task]["identity"])
                for target_task in TARGET_TASKS
            },
        }
        _atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _validate_routereval_training_options(
    *,
    method: str,
    pool_size: int,
    pool_seed: int,
    training_seed: int,
    expected_embedding_dim: int | None,
) -> None:
    if method not in ("flexrouter", "barp"):
        raise ValueError("method must be flexrouter or barp")
    if pool_size not in POOL_SIZES and expected_embedding_dim is not None:
        raise ValueError(f"pool_size must be one of {POOL_SIZES}")
    if pool_seed not in POOL_SEEDS:
        raise ValueError(f"pool_seed must be one of {POOL_SEEDS}")
    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"training_seed must be one of {TRAINING_SEEDS}")


def _validate_routereval_evaluation_options(
    *,
    target_task: str,
    redraws: int,
    alphas: Sequence[float],
    registered_execution: bool,
) -> None:
    if target_task not in TARGET_TASKS:
        raise ValueError(f"target_task must be one of {TARGET_TASKS}")
    if redraws <= 0 or redraws > TARGET_REDRAWS:
        raise ValueError(f"redraws must be between one and {TARGET_REDRAWS}")
    if registered_execution:
        if redraws != TARGET_REDRAWS:
            raise ValueError(f"registered RouterEval execution requires {TARGET_REDRAWS} redraws")
        if tuple(float(value) for value in alphas) != ALPHA_GRID:
            raise ValueError("registered RouterEval execution requires the complete alpha grid")


def train_routereval_checkpoint(
    *,
    tasks: Mapping[str, LearnedTaskData],
    method: str,
    pool_size: int,
    pool_seed: int,
    training_seed: int,
    max_epochs: int = 100,
    patience: int = 15,
    device: str = "cpu",
    expected_embedding_dim: int | None = 768,
) -> RouterEvalCheckpoint:
    """Train exactly one checkpoint using external tasks only."""

    _validate_routereval_training_options(
        method=method,
        pool_size=pool_size,
        pool_seed=pool_seed,
        training_seed=training_seed,
        expected_embedding_dim=expected_embedding_dim,
    )
    model_ids_full = _validate_routereval_task_set(
        tasks,
        target_tasks=(),
        expected_embedding_dim=expected_embedding_dim,
    )
    columns_array = subsample_pool(len(model_ids_full), pool_size, seed=pool_seed)
    columns = tuple(int(value) for value in columns_array)
    model_ids = tuple(model_ids_full[index] for index in columns)

    fit_embeddings = []
    fit_labels = []
    validation_embeddings = []
    validation_labels = []
    external_splits = {}
    for task_name in EXTERNAL_TASKS:
        task = tasks[task_name]
        split = stable_hash_split(
            task.query_ids,
            {"fit": 0.80, "validation": 0.20},
            seed=EXTERNAL_SPLIT_SEED,
            namespace=f"routereval_external:{task_name}",
        )
        external_splits[task_name] = _split_ids(task, split)
        fit_embeddings.append(task.embeddings[split["fit"]])
        fit_labels.append(task.correctness[split["fit"]][:, columns])
        validation_embeddings.append(task.embeddings[split["validation"]])
        validation_labels.append(task.correctness[split["validation"]][:, columns])
    train_x = np.concatenate(fit_embeddings, axis=0)
    train_y = np.concatenate(fit_labels, axis=0)
    validation_x = np.concatenate(validation_embeddings, axis=0)
    validation_y = np.concatenate(validation_labels, axis=0)
    model = train_learned_model(
        method,
        train_x,
        train_y,
        validation_x,
        validation_y,
        model_ids,
        training_seed=training_seed,
        tau=1.0,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )

    best_epoch = getattr(model, "best_epoch", None)
    observed_outcomes = (
        int(len(train_x) * len(model.training_history))
        if method == "barp"
        else int(len(train_x) * len(model_ids) * len(model.training_history))
    )
    method_label = (
        "FlexRouter matched-embedding reconstruction"
        if method == "flexrouter"
        else "BaRP quality-only set-ranking reconstruction"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "mode": "routereval_shared_training",
        "identity": {
            "method": method,
            "pool_size": pool_size,
            "pool_seed": pool_seed,
            "training_seed": training_seed,
        },
        "method_label": method_label,
        "source_hashes": {
            "data": {name: dict(task.source_hashes) for name, task in sorted(tasks.items())},
            "implementation": implementation_source_hashes(),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": device,
        },
        "configuration": {
            "external_tasks": list(EXTERNAL_TASKS),
            "external_split_seed": EXTERNAL_SPLIT_SEED,
            "external_split_fractions": {"fit": 0.80, "validation": 0.20},
            "target_split_fractions": {
                "design": 0.30,
                "router": 0.20,
                "final_calibration": 0.20,
                "validation": 0.15,
                "test": 0.15,
            },
            "flex_tau_grid": list(FLEX_TAU_GRID),
            "maximum_depth": MAX_DEPTH,
            "row_l2_normalization": True,
            "max_epochs": max_epochs,
            "patience": patience,
            "expected_embedding_dim": expected_embedding_dim,
        },
        "model_pool": {
            "column_indices": list(columns),
            "model_ids": list(model_ids),
            "model_ids_sha256": _hash_json(list(model_ids)),
            "fixed_across_tasks": True,
        },
        "feature_audits": {name: dict(task.feature_audit) for name, task in sorted(tasks.items())},
        "external_split_ids": external_splits,
        "training": {
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "epochs_completed": len(model.training_history),
            "fit_query_count": len(train_x),
            "validation_query_count": len(validation_x),
            "observed_model_outcome_count": observed_outcomes,
            "target_labels_used": False,
            "checkpoint_selected_without_target_labels": True,
            "history": list(model.training_history),
        },
        "information_budget": {
            "external_fit_labels_per_query": pool_size if method == "flexrouter" else 1,
            "target_design_router_validation_labels": 0,
            "final_calibration_labels_per_query": pool_size,
            "test_labels": "evaluation_only",
        },
    }
    return RouterEvalCheckpoint(
        model=model,
        method=method,
        columns=columns,
        model_ids=model_ids,
        full_model_ids=model_ids_full,
        training_metadata=metadata,
    )


def evaluate_routereval_target(
    *,
    checkpoint: RouterEvalCheckpoint,
    target: LearnedTaskData,
    redraws: int = TARGET_REDRAWS,
    alphas: Sequence[float] = ALPHA_GRID,
) -> dict[str, Any]:
    """Evaluate one target without fitting or selecting a checkpoint."""

    metadata = checkpoint.training_metadata
    registered_execution = metadata["configuration"]["expected_embedding_dim"] is not None
    _validate_routereval_evaluation_options(
        target_task=target.task,
        redraws=redraws,
        alphas=alphas,
        registered_execution=registered_execution,
    )
    if target.model_ids != checkpoint.full_model_ids:
        raise ValueError(f"{target.task}: model columns differ from the frozen checkpoint pool")
    if target.correctness.shape != (len(target.query_ids), len(checkpoint.full_model_ids)):
        raise ValueError(f"{target.task}: correctness shape does not match IDs and model pool")
    if target.embeddings.shape[0] != len(target.query_ids):
        raise ValueError(f"{target.task}: embedding rows do not match query IDs")
    expected_dimension = metadata["configuration"]["expected_embedding_dim"]
    if expected_dimension is not None and target.embeddings.shape[1] != expected_dimension:
        raise ValueError(
            f"expected {expected_dimension} embedding features, found {target.embeddings.shape[1]}"
        )

    frozen = score_learned_model(
        checkpoint.model, checkpoint.method, target.embeddings, len(checkpoint.model_ids)
    )
    # Target correctness is first touched only after checkpoint and score-family freeze.
    target_labels = target.correctness[:, checkpoint.columns]
    per_redraw = []
    for redraw in range(redraws):
        split = target_redraw_split(len(target.query_ids), redraw)
        test_rows = split["test"]
        calibration_rows = split["final_calibration"]
        fixed = fixed_depth_record(
            frozen, target_labels, test_rows, target.query_ids, checkpoint.model_ids
        )
        adaptive = (
            flex_adaptive_record(
                frozen, target_labels, test_rows, target.query_ids, checkpoint.model_ids
            )
            if checkpoint.method == "flexrouter"
            else []
        )
        calibrated = [
            calibrated_record(
                frozen,
                target_labels,
                calibration_rows,
                test_rows,
                target.query_ids,
                checkpoint.model_ids,
                float(alpha),
            )
            for alpha in alphas
        ]
        per_redraw.append(
            {
                "redraw": redraw,
                "split_seed": TARGET_SPLIT_BASE_SEED + 13 * redraw,
                "split_ids": _split_ids(target, split),
                "target_label_access": {
                    "design": False,
                    "router": False,
                    "validation": False,
                    "final_calibration": True,
                    "test": "evaluation_only_after_freeze",
                },
                "native_fixed_depth": fixed,
                "native_adaptive": adaptive,
                "adapted_calibrated": calibrated,
            }
        )

    source_hashes = copy.deepcopy(metadata["source_hashes"])
    source_hashes["data"][target.task] = dict(target.source_hashes)
    feature_audits = copy.deepcopy(metadata["feature_audits"])
    feature_audits[target.task] = dict(target.feature_audit)
    configuration = copy.deepcopy(metadata["configuration"])
    configuration.pop("expected_embedding_dim")
    configuration.update(
        {
            "target_task": target.task,
            "redraws": redraws,
            "alphas": [float(value) for value in alphas],
        }
    )
    identity = {
        **dict(metadata["identity"]),
        "target_task": target.task,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "mode": "routereval",
        "identity": identity,
        "method_label": metadata["method_label"],
        "source_hashes": source_hashes,
        "environment": copy.deepcopy(metadata["environment"]),
        "configuration": configuration,
        "model_pool": copy.deepcopy(metadata["model_pool"]),
        "feature_audits": feature_audits,
        "external_split_ids": copy.deepcopy(metadata["external_split_ids"]),
        "training": copy.deepcopy(metadata["training"]),
        "information_budget": copy.deepcopy(metadata["information_budget"]),
        "structural_metrics": {
            "complete_pool_all_miss_rate": float(np.mean(np.all(target_labels == 0, axis=1))),
            "best_single_model_miss_rate": float(1.0 - target_labels.mean(axis=0).max()),
        },
        "frozen_predictions": _prediction_records(
            frozen, target.query_ids, checkpoint.model_ids
        ),
        "per_redraw": per_redraw,
        "native_and_adapted_separated": True,
    }


def run_routereval(
    *,
    tasks: Mapping[str, LearnedTaskData],
    target_task: str,
    method: str,
    pool_size: int,
    pool_seed: int,
    training_seed: int,
    output_dir: str | Path,
    redraws: int = TARGET_REDRAWS,
    alphas: Sequence[float] = ALPHA_GRID,
    max_epochs: int = 100,
    patience: int = 15,
    device: str = "cpu",
    expected_embedding_dim: int | None = 768,
) -> dict[str, Any]:
    """Train and evaluate one RouterEval learned-baseline cell."""

    _validate_routereval_training_options(
        method=method,
        pool_size=pool_size,
        pool_seed=pool_seed,
        training_seed=training_seed,
        expected_embedding_dim=expected_embedding_dim,
    )
    _validate_routereval_evaluation_options(
        target_task=target_task,
        redraws=redraws,
        alphas=alphas,
        registered_execution=expected_embedding_dim is not None,
    )
    _validate_routereval_tasks(
        tasks,
        target_task=target_task,
        expected_embedding_dim=expected_embedding_dim,
    )
    external_tasks = {name: tasks[name] for name in EXTERNAL_TASKS}
    checkpoint = train_routereval_checkpoint(
        tasks=external_tasks,
        method=method,
        pool_size=pool_size,
        pool_seed=pool_seed,
        training_seed=training_seed,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        expected_embedding_dim=expected_embedding_dim,
    )
    result = evaluate_routereval_target(
        checkpoint=checkpoint,
        target=tasks[target_task],
        redraws=redraws,
        alphas=alphas,
    )
    destination = _write_immutable_run(output_dir, result, checkpoint.model)
    result["output_dir"] = str(destination)
    return result


def run_routereval_all(
    *,
    tasks: Mapping[str, LearnedTaskData],
    method: str,
    pool_size: int,
    pool_seed: int,
    training_seed: int,
    output_dir: str | Path,
    redraws: int = TARGET_REDRAWS,
    alphas: Sequence[float] = ALPHA_GRID,
    max_epochs: int = 100,
    patience: int = 15,
    device: str = "cpu",
    expected_embedding_dim: int | None = 768,
) -> dict[str, Any]:
    """Train once and evaluate the frozen checkpoint on both OOD targets."""

    _validate_routereval_training_options(
        method=method,
        pool_size=pool_size,
        pool_seed=pool_seed,
        training_seed=training_seed,
        expected_embedding_dim=expected_embedding_dim,
    )
    for target_task in TARGET_TASKS:
        _validate_routereval_evaluation_options(
            target_task=target_task,
            redraws=redraws,
            alphas=alphas,
            registered_execution=expected_embedding_dim is not None,
        )
    _validate_routereval_task_set(
        tasks,
        target_tasks=TARGET_TASKS,
        expected_embedding_dim=expected_embedding_dim,
    )
    external_tasks = {name: tasks[name] for name in EXTERNAL_TASKS}
    checkpoint = train_routereval_checkpoint(
        tasks=external_tasks,
        method=method,
        pool_size=pool_size,
        pool_seed=pool_seed,
        training_seed=training_seed,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        expected_embedding_dim=expected_embedding_dim,
    )
    results = {
        target_task: evaluate_routereval_target(
            checkpoint=checkpoint,
            target=tasks[target_task],
            redraws=redraws,
            alphas=alphas,
        )
        for target_task in TARGET_TASKS
    }
    destination = _write_immutable_multi_target_run(
        output_dir, results, checkpoint.training_metadata, checkpoint.model
    )
    return {
        "output_dir": str(destination),
        "training_metadata": copy.deepcopy(checkpoint.training_metadata),
        "results": results,
    }

def load_aligned_routerbench_embeddings(
    path: str | Path,
    query_ids: Sequence[str],
    *,
    manifest_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an ID-bearing archive and verify manifest plus exact row alignment."""

    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("RouterBench embeddings must be an npz with embeddings and query_ids")
    resolved_manifest = (
        Path(manifest_path) if manifest_path is not None else path.parent / ROUTERBENCH_EMBEDDING_MANIFEST
    )
    if not resolved_manifest.is_file():
        raise ValueError("RouterBench embedding archive requires its immutable manifest.json")
    manifest = load_archive_manifest(resolved_manifest, archive_path=path)
    if manifest.get("protocol") != ROUTERBENCH_EMBEDDING_PROTOCOL:
        raise ValueError("unexpected RouterBench embedding archive protocol")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"embeddings", "query_ids"}:
            raise ValueError("embedding npz must contain exactly embeddings and query_ids")
        embeddings = np.asarray(archive["embeddings"], dtype=float)
        embedded_ids = tuple(str(value) for value in archive["query_ids"].tolist())
    if len(embedded_ids) != len(set(embedded_ids)):
        raise ValueError("embedding query_ids contain duplicates")
    expected = tuple(str(value) for value in query_ids)
    if set(embedded_ids) != set(expected):
        raise ValueError("embedding query IDs do not exactly match audited RouterBench queries")
    if manifest["reconstruction"]["query_ids_sha256"] != _hash_json(list(embedded_ids)):
        raise ValueError("embedding query IDs do not match the archive manifest")
    row_of = {query_id: row for row, query_id in enumerate(embedded_ids)}
    order = np.asarray([row_of[query_id] for query_id in expected], dtype=int)
    aligned = l2_normalize_rows(embeddings[order])
    if manifest["output"]["n_rows"] != len(expected):
        raise ValueError("embedding archive row count does not match its manifest")
    if manifest["output"]["embedding_dim"] != aligned.shape[1]:
        raise ValueError("embedding archive dimension does not match its manifest")
    return aligned, {
        "source_sha256": sha256_file(path),
        "manifest_sha256": sha256_file(resolved_manifest),
        "manifest_content_sha256": manifest["manifest_sha256"],
        "archive_protocol": manifest["protocol"],
        "row_alignment": "exact_query_id_set_and_reorder",
        "n_rows": len(expected),
        "embedding_dim": aligned.shape[1],
        "row_l2_normalization": True,
        "reconstruction": dict(manifest["reconstruction"]),
        "source": dict(manifest["source"]),
    }


def _routerbench_costs(data: Any) -> np.ndarray:
    columns = [model + "|total_cost" for model in data.model_columns]
    values = data.frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    if values.shape != (len(data.frame), len(data.model_columns)):
        raise ValueError("RouterBench cost matrix shape is invalid")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("RouterBench costs must be finite and non-negative")
    return values


class RouterBenchFeatureAuditError(ValueError):
    """Expected protocol unavailability caused by unusable prompt features."""

    def __init__(self, failures: Sequence[str], audits: Mapping[str, Any]) -> None:
        self.failures = tuple(str(value) for value in failures)
        self.audits = dict(audits)
        super().__init__("RouterBench feature audit failed: " + "; ".join(self.failures))


def _audit_routerbench_features(data: Any, tasks: Sequence[str]) -> dict[str, Any]:
    by_task = {str(row["task"]): dict(row) for row in data.prompt_degeneracy}
    failures = []
    audits = {}
    for task in tasks:
        record = by_task.get(task)
        if record is None:
            failures.append(f"{task}: missing feature audit")
            continue
        audits[task] = record
        if int(record["unique_prompts"]) <= 1:
            failures.append(f"{task}: one or fewer distinct prompts")
        elif float(record["duplicate_rate"]) > 0.10:
            failures.append(f"{task}: duplicate prompt rate exceeds 0.10")
    if failures:
        raise RouterBenchFeatureAuditError(failures, audits)
    return audits


def nondominated_frontier(points: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return indices of quality-maximizing, cost-minimizing registered points."""

    keep = []
    for index, point in enumerate(points):
        quality = float(point["mean_official_quality"])
        cost = float(point["mean_cost"])
        dominated = False
        for other_index, other in enumerate(points):
            if index == other_index:
                continue
            other_quality = float(other["mean_official_quality"])
            other_cost = float(other["mean_cost"])
            if other_quality >= quality and other_cost <= cost and (
                other_quality > quality or other_cost < cost
            ):
                dominated = True
                break
        if not dominated:
            keep.append(index)
    return keep


def _routerbench_unavailable_result(
    *,
    data: Any,
    embedding_audit: Mapping[str, Any],
    training_seed: int,
    training_tasks: Sequence[str],
    analysis_label: str,
    error: RouterBenchFeatureAuditError,
    device: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "mode": "routerbench_native_barp",
        "status": "unavailable",
        "analysis_label": analysis_label,
        "identity": {
            "method": "barp",
            "training_seed": training_seed,
            "shot": "0shot",
            "variant": analysis_label,
        },
        "method_label": "BaRP native single-action quality-cost reconstruction",
        "source_hashes": {
            "data": {
                "routerbench": data.source_sha256,
                "embeddings": embedding_audit["source_sha256"],
                "embedding_manifest": embedding_audit.get("manifest_sha256"),
            },
            "implementation": implementation_source_hashes(),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": device,
        },
        "configuration": {
            "training_tasks": list(training_tasks),
            "target_tasks": list(ROUTERBENCH_TARGET_TASKS),
            "substituted_tasks": [],
            "split_seed": EXTERNAL_SPLIT_SEED,
        },
        "feature_audits": error.audits,
        "embedding_audit": dict(embedding_audit),
        "failure": {
            "stage": "prompt_feature_audit",
            "reason": str(error),
            "failures": list(error.failures),
            "training_started": False,
            "checkpoint_available": False,
        },
        "training": {
            "started": False,
            "ood_target_labels_used": False,
            "target_training_labels": 0,
        },
        "native_and_adapted_separated": True,
    }


def run_routerbench_native_barp(
    *,
    data: Any,
    embeddings: np.ndarray,
    embedding_audit: Mapping[str, Any],
    training_seed: int,
    output_dir: str | Path,
    max_epochs: int = 100,
    patience: int = 15,
    device: str = "cpu",
    three_task_sensitivity: bool = False,
) -> dict[str, Any]:
    """Train native BaRP or record why the preregistered cell is unavailable."""

    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"training_seed must be one of {TRAINING_SEEDS}")
    if data.shot != "0shot":
        raise ValueError("native RouterBench BaRP mode requires the 0-shot release")
    if embeddings.shape[0] != len(data.frame):
        raise ValueError("embedding rows do not match audited RouterBench rows")
    if tuple(data.model_columns) != ROUTERBENCH_MODEL_COLUMNS:
        raise ValueError("RouterBench must use its fixed complete 11-model pool")

    training_tasks = (
        ROUTERBENCH_THREE_TASK_SENSITIVITY
        if three_task_sensitivity
        else ROUTERBENCH_TRAIN_TASKS
    )
    analysis_label = (
        "sensitivity_without_degenerate_gsm8k"
        if three_task_sensitivity
        else "primary_four_training_tasks"
    )
    required_tasks = tuple(training_tasks) + ROUTERBENCH_TARGET_TASKS
    try:
        audits = _audit_routerbench_features(data, required_tasks)
    except RouterBenchFeatureAuditError as error:
        result = _routerbench_unavailable_result(
            data=data,
            embedding_audit=embedding_audit,
            training_seed=training_seed,
            training_tasks=training_tasks,
            analysis_label=analysis_label,
            error=error,
            device=device,
        )
        destination = _write_immutable_unavailable_run(output_dir, result)
        result["output_dir"] = str(destination)
        return result

    assignment = assign_racer_splits(data, seed=EXTERNAL_SPLIT_SEED)
    quality = data.frame.loc[:, data.model_columns].to_numpy(dtype=float, copy=True)
    if not np.all(np.isfinite(quality)) or np.any((quality < 0.0) | (quality > 1.0)):
        raise ValueError("official RouterBench quality must be finite and within zero and one")
    costs = _routerbench_costs(data)
    coverage = derive_coverage_labels(data, policy=PRIMARY_COVERAGE_POLICY)
    row_of = {str(query_id): row for row, query_id in enumerate(data.query_ids)}

    def rows_for(tasks: Sequence[str], roles: Sequence[str]) -> np.ndarray:
        ids = [
            query_id
            for task in tasks
            for role in roles
            for query_id in assignment.ids(task, role)
        ]
        return np.asarray([row_of[query_id] for query_id in ids], dtype=int)

    fit_rows = rows_for(training_tasks, ("train", "design"))
    validation_rows = rows_for(training_tasks, ("validation",))
    positive_costs = costs[fit_rows][costs[fit_rows] > 0.0]
    if positive_costs.size == 0:
        raise ValueError("training train/design rows contain no finite positive costs")
    tau = float(np.quantile(positive_costs, 0.95))
    model = train_learned_model(
        "barp",
        embeddings[fit_rows],
        quality[fit_rows],
        embeddings[validation_rows],
        quality[validation_rows],
        data.model_columns,
        training_seed=training_seed,
        train_costs=costs[fit_rows],
        validation_costs=costs[validation_rows],
        tau=tau,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        routerbench_preferences=True,
    )
    assert isinstance(model, BaRPBaseline)

    task_points: dict[str, list[dict[str, Any]]] = {}
    for task in ROUTERBENCH_TARGET_TASKS:
        test_rows = rows_for((task,), ("test",))
        points = []
        for preference_index, (quality_weight, cost_weight) in enumerate(PREFERENCE_GRID):
            actions = model.predict(embeddings[test_rows], [quality_weight, cost_weight]).numpy()
            selected_quality = quality[test_rows, actions]
            selected_cost = costs[test_rows, actions]
            selected_coverage = coverage.standard[test_rows, actions]
            reward = quality_weight * selected_quality - cost_weight * np.clip(selected_cost / tau, 0.0, 1.0)
            points.append(
                {
                    "preference_index": preference_index,
                    "w_quality": float(quality_weight),
                    "w_cost": float(cost_weight),
                    "mean_official_quality": float(np.mean(selected_quality)),
                    "mean_cost": float(np.mean(selected_cost)),
                    "mean_scalar_reward": float(np.mean(reward)),
                    "binary_coverage": float(np.mean(selected_coverage)),
                    "query_events": [
                        {
                            "query_id": data.query_ids[int(row)],
                            "selected_model_id": data.model_columns[int(action)],
                            "selected_model_index": int(action),
                            "official_quality": float(one_quality),
                            "cost": float(one_cost),
                            "binary_coverage": bool(one_coverage),
                            "scalar_reward": float(one_reward),
                        }
                        for row, action, one_quality, one_cost, one_coverage, one_reward in zip(
                            test_rows,
                            actions,
                            selected_quality,
                            selected_cost,
                            selected_coverage,
                            reward,
                        )
                    ],
                }
            )
        frontier = set(nondominated_frontier(points))
        for index, point in enumerate(points):
            point["nondominated"] = index in frontier
        task_points[task] = points

    macro_points = []
    for index, (quality_weight, cost_weight) in enumerate(PREFERENCE_GRID):
        point = {
            "preference_index": index,
            "w_quality": float(quality_weight),
            "w_cost": float(cost_weight),
            "mean_official_quality": float(np.mean([task_points[task][index]["mean_official_quality"] for task in ROUTERBENCH_TARGET_TASKS])),
            "mean_cost": float(np.mean([task_points[task][index]["mean_cost"] for task in ROUTERBENCH_TARGET_TASKS])),
            "mean_scalar_reward": float(np.mean([task_points[task][index]["mean_scalar_reward"] for task in ROUTERBENCH_TARGET_TASKS])),
            "binary_coverage": float(np.mean([task_points[task][index]["binary_coverage"] for task in ROUTERBENCH_TARGET_TASKS])),
        }
        macro_points.append(point)
    frontier = set(nondominated_frontier(macro_points))
    for index, point in enumerate(macro_points):
        point["nondominated"] = index in frontier

    split_ids = {
        task: {role: list(assignment.ids(task, role)) for role in ("train", "design", "calibration", "validation", "test")}
        for task in required_tasks
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "mode": "routerbench_native_barp",
        "status": "available",
        "analysis_label": analysis_label,
        "identity": {
            "method": "barp",
            "training_seed": training_seed,
            "shot": "0shot",
            "variant": analysis_label,
        },
        "method_label": "BaRP native single-action quality-cost reconstruction",
        "source_hashes": {
            "data": {
                "routerbench": data.source_sha256,
                "embeddings": embedding_audit["source_sha256"],
                "embedding_manifest": embedding_audit.get("manifest_sha256"),
            },
            "implementation": implementation_source_hashes(),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": device,
        },
        "configuration": {
            "training_tasks": list(training_tasks),
            "target_tasks": list(ROUTERBENCH_TARGET_TASKS),
            "substituted_tasks": [],
            "split_seed": EXTERNAL_SPLIT_SEED,
            "tau_definition": "95th_percentile_positive_train_design_cost",
            "tau": tau,
            "preference_grid": [list(value) for value in PREFERENCE_GRID],
            "max_epochs": max_epochs,
            "patience": patience,
            "coverage_policy": PRIMARY_COVERAGE_POLICY.to_dict(),
        },
        "feature_audits": audits,
        "embedding_audit": dict(embedding_audit),
        "model_pool": {
            "model_ids": list(data.model_columns),
            "fixed_complete_pool": True,
            "pool_size": len(data.model_columns),
        },
        "split_ids": split_ids,
        "training": {
            "best_epoch": int(getattr(model, "best_epoch", -1)),
            "epochs_completed": len(model.training_history),
            "fit_query_count": len(fit_rows),
            "validation_query_count": len(validation_rows),
            "observed_model_outcome_count": int(len(fit_rows) * len(model.training_history)),
            "ood_target_labels_used": False,
            "history": list(model.training_history),
        },
        "information_budget": {
            "chosen_action_outcomes_per_fit_query": len(model.training_history),
            "full_training_cost_table_use": "tau_quantile_only",
            "target_training_labels": 0,
            "final_calibration_labels": 0,
            "test_labels": "evaluation_only",
        },
        "native_barp": {
            "tasks": task_points,
            "macro_average": macro_points,
            "frontier_preference_indices": sorted(frontier),
            "objective": "single_action_quality_and_monetary_cost",
            "risk_certificate": False,
        },
        "native_and_adapted_separated": True,
    }
    destination = _write_immutable_run(output_dir, result, model)
    result["output_dir"] = str(destination)
    return result


def _parse_float_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(",") if value.strip())
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("alpha grid must contain values strictly between zero and one")
    return values


def _load_routereval_bundle(
    leaderboard: Path, router_dir: Path, target_task: str
) -> dict[str, LearnedTaskData]:
    targets = TARGET_TASKS if target_task == "all" else (target_task,)
    return {
        task: load_routereval_task(
            leaderboard,
            router_dir / f"{task}_router_dataset.pkl",
            task,
        )
        for task in EXTERNAL_TASKS + targets
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("routereval", "routerbench"))
    parser.add_argument("--method", default="barp", choices=("flexrouter", "barp"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--leaderboard")
    parser.add_argument("--router-dir")
    parser.add_argument("--target-task", choices=TARGET_TASKS + ("all",))
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--pool-seed", type=int, default=7001)
    parser.add_argument("--redraws", type=int, default=TARGET_REDRAWS)
    parser.add_argument("--alphas", default=",".join(str(value) for value in ALPHA_GRID))
    parser.add_argument("--routerbench-pickle")
    parser.add_argument("--routerbench-embeddings")
    parser.add_argument("--routerbench-embedding-manifest")
    parser.add_argument(
        "--routerbench-three-task-sensitivity",
        action="store_true",
        help="train only MMLU, ARC-Challenge, and Winogrande; substitute no task",
    )
    arguments = parser.parse_args(argv)
    if arguments.max_epochs != 100 or arguments.patience != 15:
        parser.error("registered execution requires --max-epochs 100 and --patience 15")

    if arguments.mode == "routereval":
        if arguments.routerbench_three_task_sensitivity:
            parser.error("--routerbench-three-task-sensitivity is valid only in routerbench mode")
        missing = [
            name
            for name, value in (
                ("--leaderboard", arguments.leaderboard),
                ("--router-dir", arguments.router_dir),
                ("--target-task", arguments.target_task),
                ("--pool-size", arguments.pool_size),
            )
            if value is None
        ]
        if missing:
            parser.error("routereval mode requires " + ", ".join(missing))
        tasks = _load_routereval_bundle(
            Path(arguments.leaderboard),
            Path(arguments.router_dir),
            arguments.target_task,
        )
        common = {
            "tasks": tasks,
            "method": arguments.method,
            "pool_size": arguments.pool_size,
            "pool_seed": arguments.pool_seed,
            "training_seed": arguments.training_seed,
            "output_dir": arguments.out_dir,
            "redraws": arguments.redraws,
            "alphas": _parse_float_grid(arguments.alphas),
            "max_epochs": arguments.max_epochs,
            "patience": arguments.patience,
            "device": arguments.device,
        }
        if arguments.target_task == "all":
            run_routereval_all(**common)
        else:
            run_routereval(target_task=arguments.target_task, **common)
    else:
        if arguments.method != "barp":
            parser.error("routerbench mode supports only --method barp")
        if not arguments.routerbench_pickle or not arguments.routerbench_embeddings:
            parser.error("routerbench mode requires --routerbench-pickle and --routerbench-embeddings")
        data = load_routerbench(
            arguments.routerbench_pickle,
            shot="0shot",
            model_columns=ROUTERBENCH_MODEL_COLUMNS,
        )
        embeddings, audit = load_aligned_routerbench_embeddings(
            arguments.routerbench_embeddings,
            data.query_ids,
            manifest_path=arguments.routerbench_embedding_manifest,
        )
        run_routerbench_native_barp(
            data=data,
            embeddings=embeddings,
            embedding_audit=audit,
            training_seed=arguments.training_seed,
            output_dir=arguments.out_dir,
            max_epochs=arguments.max_epochs,
            patience=arguments.patience,
            device=arguments.device,
            three_task_sensitivity=arguments.routerbench_three_task_sensitivity,
        )
    print(f"wrote immutable learned-baseline run to {arguments.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
