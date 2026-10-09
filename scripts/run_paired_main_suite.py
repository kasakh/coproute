"""Run the canonical ISSUE-50 paired RouterEval main suite.

The runner evaluates every arm on one aligned task, one sampled model pool, and
one contiguous range of global redraw identities.  It stores query-level calls
and miss indicators so every aggregate and paired comparison can be rebuilt
without rerunning a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.chain_discount import (  # noqa: E402
    chain_discount_diagnostics,
    chain_discounted_scores,
    fit_chain_discount,
)
from components.paired_orders import per_query_orders_factorized_scores  # noqa: E402
from components.racer_protocol import (  # noqa: E402
    nonconformity_scores,
    select_standard_crc_threshold,
    smooth_nonconformity_scores,
)
from components.racer_wrapper import (  # noqa: E402
    FACTORIZED_INDEPENDENCE,
    fixed_design_marginal_factorized_scores,
    fixed_order_factorized_scores,
    sequential_scores,
)
from components.routereval_protocol import subsample_pool  # noqa: E402
from components.scale_selectors import build_order  # noqa: E402

SCHEMA_VERSION = "paired_main_suite_v2"
ALPHA_GRID = (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
TRAINING_SEEDS = (9101, 9102, 9103, 9104, 9105)
BASE_SEED = 7001
SPLIT_STEP = 13
NONLEARNED_METHODS = (
    "racer_g",
    "racer_p",
    "exact_independence",
    "fixed_design_marginal_factorized",
    "fixed_tally_factorized",
)
E3_MENU = (
    "exact_independence",
    "fixed_design_marginal_factorized",
    "fixed_tally_factorized",
)
LEARNED_FAMILIES = ("flexrouter", "barp")
CHAIN_DISCOUNT_METHOD = "fixed_tally_chain_discounted"
CHAIN_DISCOUNT_MIN_SURVIVOR_GRID = (0, 5, 20, 50)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one byte representation used by runner and merger."""

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any, *, refuse_overwrite: bool = True) -> None:
    """Write canonical JSON through an fsynced temporary file and atomic replace."""

    destination = Path(path)
    if refuse_overwrite and destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable result {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        if refuse_overwrite and destination.exists():
            raise FileExistsError(f"refusing to overwrite immutable result {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_task(
    protocol: str,
    benchmark: str,
    *,
    leaderboard: Path | None = None,
    router_dir: Path | None = None,
    routerbench_pickle: Path | None = None,
    embedding_archive: Path | None = None,
    embedding_manifest: Path | None = None,
) -> Any:
    # Lazy import keeps NumPy-only helpers and tests usable when optional deps are absent.
    from components.stage1_task_adapters import (
        load_routerbench_registered_task,
        load_routereval_registered_task,
    )

    if protocol == "routereval":
        if leaderboard is None or router_dir is None:
            raise ValueError("RouterEval requires leaderboard and router directory")
        return load_routereval_registered_task(leaderboard, router_dir, benchmark)
    if protocol == "routerbench":
        if routerbench_pickle is None or embedding_archive is None:
            raise ValueError("RouterBench requires its pickle and embedding archive")
        return load_routerbench_registered_task(
            routerbench_pickle,
            embedding_archive,
            benchmark,
            embedding_manifest=embedding_manifest,
        )
    raise ValueError(f"unknown protocol {protocol}")


def _target_split(n_queries: int, global_redraw: int, base_seed: int) -> dict[str, np.ndarray]:
    """Use the registered split helper and assert its exact global seed identity."""

    if base_seed != BASE_SEED:
        raise ValueError(f"registered execution requires base_seed={BASE_SEED}")
    from scripts.run_learned_baselines import target_redraw_split

    split = target_redraw_split(n_queries, global_redraw)
    expected = np.random.default_rng(base_seed + SPLIT_STEP * global_redraw).permutation(n_queries)
    concatenated = np.concatenate(tuple(split.values()))
    if not np.array_equal(concatenated, expected):
        raise RuntimeError("target_redraw_split no longer matches the registered seed rule")
    return split


def _split_query_ids(task: Any, split: Mapping[str, np.ndarray]) -> dict[str, list[str]]:
    from scripts.run_learned_baselines import _split_ids

    return _split_ids(task, split)


def _knn(
    train_embeddings: np.ndarray,
    train_correctness: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    # This is intentionally the audited implementation, not a second KNN.
    from scripts.design_bprime_e2_run import _knn_probabilities

    probabilities, support = _knn_probabilities(
        train_embeddings,
        train_correctness,
        query_embeddings,
        k,
        return_support=True,
    )
    return np.asarray(probabilities, dtype=float), np.asarray(support)


def _ordered_quality_tail(quality: np.ndarray, excluded: set[int]) -> list[int]:
    indices = np.asarray([index for index in range(len(quality)) if index not in excluded], dtype=int)
    if not len(indices):
        return []
    order = np.lexsort((indices, -quality[indices]))
    return [int(index) for index in indices[order]]


def full_flexrouter_orders(model: Any, embeddings: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return full DPP-first orders and each query's native positive path length."""

    qualities_tensor = model.predict_quality(embeddings)
    qualities = np.asarray(qualities_tensor.detach().cpu().numpy(), dtype=float)
    n_queries, n_models = qualities.shape
    if n_models != int(model.n_models):
        raise ValueError("FlexRouter quality width differs from checkpoint model count")
    orders = np.empty((n_queries, n_models), dtype=int)
    native_lengths: list[int] = []
    for row in range(n_queries):
        path = model.select_from_qualities(
            qualities_tensor[row], max_size=n_models, relative_threshold=0.0, top_k0=None
        )
        native = [int(value) for value in path.order]
        if len(native) != len(set(native)) or any(index < 0 or index >= n_models for index in native):
            raise ValueError(f"FlexRouter query {row} produced an invalid DPP path")
        full = native + _ordered_quality_tail(qualities[row], set(native))
        if len(full) != n_models:
            raise RuntimeError("FlexRouter quality tail did not complete the pool")
        orders[row] = full
        native_lengths.append(len(native))
    return orders, native_lengths


def full_barp_orders(model: Any, embeddings: np.ndarray) -> np.ndarray:
    """Return BaRP's stable quality-only ranking over the complete pool."""

    ranking = model.rank_models(embeddings, [1.0, 0.0], top_k=int(model.n_models))
    return np.asarray(ranking.detach().cpu().numpy(), dtype=int)


def _manifest_identity(directory: Path, method: str) -> tuple[dict[str, Any], Path]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{directory}: missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or "checkpoint.pt" not in files:
        raise ValueError(f"{manifest_path}: manifest does not identify checkpoint.pt")
    for name, expected_hash in files.items():
        candidate = directory / name
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            raise ValueError(f"{manifest_path}: hash mismatch for {name}")
    checkpoint_hash = sha256_file(directory / "checkpoint.pt")
    if manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"{manifest_path}: checkpoint_sha256 mismatch")

    metadata_path = directory / "training_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        result_paths = sorted(directory.glob("results*.json"))
        if not result_paths:
            raise ValueError(f"{directory}: no training metadata or result identity")
        metadata = json.loads(result_paths[0].read_text(encoding="utf-8"))
    identity = metadata.get("identity")
    pool = metadata.get("model_pool")
    if not isinstance(identity, dict) or not isinstance(pool, dict):
        raise ValueError(f"{directory}: checkpoint metadata lacks identity or model_pool")
    if identity.get("method") != method:
        raise ValueError(f"{directory}: expected {method} checkpoint")
    return {
        "method": method,
        "pool_size": identity.get("pool_size"),
        "pool_seed": identity.get("pool_seed"),
        "training_seed": identity.get("training_seed"),
        "model_ids": pool.get("model_ids"),
        "model_ids_sha256": pool.get("model_ids_sha256"),
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "result_identity_sha256": hash_json(identity),
        "source": identity.get("source"),
        "target_family": identity.get("target_family"),
        "excluded_family": identity.get("excluded_family"),
    }, directory / "checkpoint.pt"


def load_checkpoint_ensemble(
    checkpoint_root: str | Path,
    method: str,
    *,
    pool_size: int,
    pool_seed: int,
    model_ids: Sequence[str],
    device: str = "cpu",
    expected_source: str | None = None,
    expected_target_family: str | None = None,
) -> list[dict[str, Any]]:
    """Discover and verify exactly the five registered checkpoints for one arm."""

    if method not in LEARNED_FAMILIES:
        raise ValueError(f"unknown learned method {method}")
    root = Path(checkpoint_root)
    if not root.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {root}")
    manifests = sorted(root.rglob("manifest.json"))
    matches: dict[int, tuple[dict[str, Any], Path]] = {}
    for manifest_path in manifests:
        identity, checkpoint = _manifest_identity(manifest_path.parent, method)
        if identity["pool_size"] != pool_size or identity["pool_seed"] != pool_seed:
            continue
        seed = identity["training_seed"]
        if seed not in TRAINING_SEEDS:
            continue
        if expected_source is not None and identity.get("source") != expected_source:
            continue
        if (
            expected_target_family is not None
            and identity.get("target_family") != expected_target_family
        ):
            continue
        if (
            expected_target_family is not None
            and identity.get("excluded_family") != expected_target_family
        ):
            continue
        if seed in matches:
            raise ValueError(f"duplicate {method} checkpoint for training seed {seed}")
        if identity["model_ids"] != list(model_ids):
            raise ValueError(f"{manifest_path.parent}: checkpoint model IDs differ from sampled pool")
        if identity["model_ids_sha256"] not in (None, hash_json(list(model_ids))):
            raise ValueError(f"{manifest_path.parent}: model ID hash mismatch")
        matches[int(seed)] = (identity, checkpoint)
    missing = sorted(set(TRAINING_SEEDS) - set(matches))
    if missing:
        raise ValueError(f"{method}: missing registered checkpoint seeds {missing}")

    if method == "flexrouter":
        from components.flexrouter_baseline import FlexRouter as checkpoint_class
    else:
        from components.barp_baseline import BaRPBaseline as checkpoint_class
    loaded = []
    for seed in TRAINING_SEEDS:
        identity, checkpoint = matches[seed]
        model = checkpoint_class.load(checkpoint, device=device)
        if int(model.n_models) != pool_size:
            raise ValueError(f"{checkpoint}: checkpoint model count differs from pool size")
        if method == "barp" and list(model.model_ids) != list(model_ids):
            raise ValueError(f"{checkpoint}: BaRP checkpoint model IDs differ from pool")
        loaded.append({"identity": identity, "model": model})
    return loaded


def _smooth(
    raw_scores: np.ndarray,
    *,
    base_seed: int,
    split_seed: int,
    preserve_prefixes: bool = False,
    fixed_order: Sequence[int] | None = None,
) -> np.ndarray:
    if fixed_order is None:
        return smooth_nonconformity_scores(
            raw_scores,
            seed=base_seed,
            split_seed=split_seed,
            preserve_prefixes=preserve_prefixes,
        )
    order = np.asarray(fixed_order, dtype=int)
    ordered = np.asarray(raw_scores)[:, order]
    smoothed = smooth_nonconformity_scores(
        ordered,
        seed=base_seed,
        split_seed=split_seed,
        preserve_prefixes=preserve_prefixes,
    )
    result = np.empty_like(smoothed)
    result[:, order] = smoothed
    return result


def _score_nonlearned(
    probabilities: np.ndarray,
    design_labels: np.ndarray,
    model_ids: Sequence[str],
    *,
    base_seed: int,
    split_seed: int,
    preserve_score_prefixes: bool = False,
) -> dict[str, np.ndarray]:
    n_models = probabilities.shape[1]
    scores: dict[str, np.ndarray] = {}
    for variant, name in (("G", "racer_g"), ("P", "racer_p")):
        _, augmented = nonconformity_scores(
            probabilities, variant, seed=base_seed, split_seed=split_seed
        )
        scores[name] = augmented[:, :-1]
    exact = sequential_scores(
        probabilities,
        dependence=FACTORIZED_INDEPENDENCE,
        max_depth=n_models,
    )
    scores["exact_independence"] = _smooth(
        exact,
        base_seed=base_seed,
        split_seed=split_seed,
        preserve_prefixes=preserve_score_prefixes,
    )
    design_marginal = fixed_design_marginal_factorized_scores(
        probabilities, design_labels, max_depth=n_models
    )
    scores["fixed_design_marginal_factorized"] = _smooth(
        design_marginal,
        base_seed=base_seed,
        split_seed=split_seed,
        preserve_prefixes=preserve_score_prefixes,
    )
    tally_order = build_order(
        "empirical_tally",
        design_labels,
        model_names=list(model_ids),
        max_depth=n_models,
        seed=base_seed,
    )
    if len(tally_order) != n_models:
        raise RuntimeError("fixed empirical tally did not return a full-pool order")
    tally = fixed_order_factorized_scores(
        probabilities, tally_order, max_depth=n_models
    )
    scores["fixed_tally_factorized"] = _smooth(
        tally,
        base_seed=base_seed,
        split_seed=split_seed,
        preserve_prefixes=preserve_score_prefixes,
        fixed_order=(tally_order if preserve_score_prefixes else None),
    )
    return scores


def _score_chain_discount_candidates(
    probabilities: np.ndarray,
    design_labels: np.ndarray,
    model_ids: Sequence[str],
    baseline_scores: np.ndarray,
    *,
    base_seed: int,
    split_seed: int,
) -> tuple[dict[int, np.ndarray], Any]:
    """Fit the ISSUE-77 chain once and score its frozen shrinkage grid."""

    n_models = probabilities.shape[1]
    tally_order = build_order(
        "empirical_tally",
        design_labels,
        model_names=list(model_ids),
        max_depth=n_models,
        seed=base_seed,
    )
    profile = fit_chain_discount(design_labels, tally_order)
    candidates: dict[int, np.ndarray] = {}
    for minimum in CHAIN_DISCOUNT_MIN_SURVIVOR_GRID:
        raw = chain_discounted_scores(
            probabilities,
            profile,
            minimum_survivor_count=minimum,
        )
        candidates[minimum] = _smooth(
            raw,
            base_seed=base_seed,
            split_seed=split_seed,
            preserve_prefixes=True,
            fixed_order=profile.order,
        )
        ordered_scores = candidates[minimum][:, profile.order]
        if np.any(np.diff(ordered_scores, axis=1) < 0.0):
            raise RuntimeError(
                "prefix-preserving smoothing inverted an ISSUE-77 fixed order"
            )

    endpoint = candidates[0]
    if not np.array_equal(endpoint, baseline_scores):
        maximum_error = float(np.max(np.abs(endpoint - baseline_scores)))
        raise RuntimeError(
            "ISSUE-77 discount-off endpoint does not reproduce "
            f"fixed_tally_factorized exactly; maximum error={maximum_error}"
        )
    return candidates, profile


def _fallback_score_operating_point(
    scores: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
) -> tuple[float, float, float]:
    observed = np.unique(scores[rows])
    candidates = [float(np.nextafter(observed[0], -np.inf)), *map(float, observed)]
    points = []
    for threshold in candidates:
        selected = scores[rows] <= threshold
        calls = float(selected.sum(axis=1).mean())
        risk = float(
            np.mean(~np.any(selected & labels[rows].astype(bool), axis=1))
        )
        points.append((risk, calls, threshold))
    risk, calls, threshold = min(points)
    return threshold, calls, risk


def _select_chain_discount_candidate(
    candidates: Mapping[int, np.ndarray],
    labels: np.ndarray,
    validation_rows: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    """Tune only the shrinkage schedule on validation, then freeze it."""

    trajectory = []
    for grid_index, minimum in enumerate(CHAIN_DISCOUNT_MIN_SURVIVOR_GRID):
        scores = candidates[minimum]
        temporary = select_standard_crc_threshold(
            scores[validation_rows], labels[validation_rows], alpha
        )
        if temporary.feasible:
            threshold = float(temporary.threshold)
            selected = scores[validation_rows] <= threshold
            calls = float(selected.sum(axis=1).mean())
            risk = float(
                np.mean(
                    ~np.any(
                        selected & labels[validation_rows].astype(bool), axis=1
                    )
                )
            )
        else:
            threshold, calls, risk = _fallback_score_operating_point(
                scores, labels, validation_rows
            )
        trajectory.append({
            "grid_index": grid_index,
            "minimum_survivor_count": int(minimum),
            "discount_fully_off": minimum == 0,
            "validation_certified": bool(temporary.feasible),
            "validation_fallback": not bool(temporary.feasible),
            "validation_non_certification_reason": temporary.non_certification_reason,
            "validation_crc_bound": float(temporary.crc_bound),
            "validation_threshold": threshold,
            "validation_calls": calls,
            "validation_standard_all_miss_risk": risk,
        })

    certified = [record for record in trajectory if record["validation_certified"]]
    if certified:
        chosen = min(
            certified,
            key=lambda record: (
                record["validation_calls"],
                record["validation_standard_all_miss_risk"],
                record["grid_index"],
            ),
        )
        fallback_used = False
    else:
        chosen = min(
            trajectory,
            key=lambda record: (
                record["validation_standard_all_miss_risk"],
                record["validation_calls"],
                record["grid_index"],
            ),
        )
        fallback_used = True
    return {
        "chosen": chosen,
        "scores": candidates[int(chosen["minimum_survivor_count"])],
        "selection_source": "validation_only",
        "selection_fallback_used": fallback_used,
        "validation_trajectory": trajectory,
    }


def assert_registered_arm_presence(
    cells: Sequence[Mapping[str, Any]],
    registered_methods: Sequence[str],
) -> None:
    """Fail if any registered arm is absent from any emitted redraw."""

    expected = set(registered_methods)
    if not cells:
        raise RuntimeError("paired suite emitted no target cells")
    redraw_count = 0
    for cell in cells:
        redraws = cell.get("redraws", [])
        if not redraws:
            raise RuntimeError(f"paired suite emitted no redraws for alpha={cell.get('alpha')}")
        for redraw in redraws:
            redraw_count += 1
            present = set(redraw.get("methods", {}))
            missing = sorted(expected - present)
            unexpected = sorted(present - expected)
            if missing or unexpected:
                raise RuntimeError(
                    "registered arm presence mismatch at "
                    f"alpha={cell.get('alpha')}, redraw={redraw.get('redraw_id')}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
    if redraw_count == 0:
        raise RuntimeError("paired suite emitted no method rows")


def _endpoint_identity(model_ids: Sequence[str], calibration_ids: Sequence[str], test_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "model_ids": list(model_ids),
        "model_ids_sha256": hash_json(list(model_ids)),
        "calibration_query_ids_sha256": hash_json(list(calibration_ids)),
        "test_query_ids_sha256": hash_json(list(test_ids)),
        "reaches_full_pool": True,
    }


def evaluate_score_family(
    scores: np.ndarray,
    labels: np.ndarray,
    calibration_rows: np.ndarray,
    test_rows: np.ndarray,
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    alpha: float,
    *,
    method_family: str,
    training_seed: int | None = None,
    native_dpp_prefix_length: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Apply the common standard-loss calibration and emit exact query vectors."""

    calibration = select_standard_crc_threshold(
        scores[calibration_rows], labels[calibration_rows], alpha
    )
    calibration_ids = [query_ids[int(row)] for row in calibration_rows]
    test_ids = [query_ids[int(row)] for row in test_rows]
    endpoint = _endpoint_identity(model_ids, calibration_ids, test_ids)
    endpoint_misses = int(np.count_nonzero(np.all(labels[calibration_rows] == 0, axis=1)))
    base: dict[str, Any] = {
        "method_family": method_family,
        "training_seed": training_seed,
        "certified": bool(calibration.feasible),
        "non_certification": not bool(calibration.feasible),
        "non_certification_reason": calibration.non_certification_reason,
        "risk_validity_claimed": bool(calibration.feasible),
        "final_calibration_count": 1,
        "n_final_calibration": int(len(calibration_rows)),
        "calibration_count": int(len(calibration_rows)),
        "calibration_misses": int(calibration.calibration_misses),
        "unavoidable_calibration_misses": int(calibration.unavoidable_misses),
        "maximum_allowed_misses": int(calibration.max_misses),
        "corrected_calibration_bound": float(calibration.crc_bound),
        "threshold": float(calibration.threshold) if calibration.feasible else None,
        "endpoint_calibration_misses": endpoint_misses,
        "endpoint_calibration_count": int(len(calibration_rows)),
        "endpoint": endpoint,
        "test_query_ids": test_ids,
    }
    if native_dpp_prefix_length is not None:
        base["native_dpp_prefix_length"] = [
            int(native_dpp_prefix_length[int(row)]) for row in test_rows
        ]
    if not calibration.feasible:
        return {
            **base,
            "test_calls": None,
            "standard_all_miss_risk": None,
            "test_calls_by_query": None,
            "test_miss_by_query": None,
            "selected_model_ids_by_query": None,
        }

    threshold = float(calibration.threshold)
    selected_mask = scores[test_rows] <= threshold
    calls = selected_mask.sum(axis=1, dtype=int)
    success = np.any(selected_mask & labels[test_rows].astype(bool), axis=1)
    misses = (~success).astype(int)
    selected_ids = [
        [model_ids[index] for index in np.flatnonzero(row_mask)]
        for row_mask in selected_mask
    ]
    return {
        **base,
        "test_calls": float(np.mean(calls)),
        "standard_all_miss_risk": float(np.mean(misses)),
        "test_calls_by_query": [int(value) for value in calls],
        "test_miss_by_query": [int(value) for value in misses],
        "selected_model_ids_by_query": selected_ids,
    }


def _validation_choice(
    score_families: Mapping[str, np.ndarray],
    labels: np.ndarray,
    validation_rows: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    """Choose one nonlearned family using validation labels only."""

    trajectory = []
    for name in E3_MENU:
        scores = score_families[name]
        result = select_standard_crc_threshold(
            scores[validation_rows], labels[validation_rows], alpha
        )
        if result.feasible:
            mask = scores[validation_rows] <= result.threshold
            calls = float(mask.sum(axis=1).mean())
            risk = float(np.mean(~np.any(mask & labels[validation_rows].astype(bool), axis=1)))
            threshold = float(result.threshold)
        else:
            observed = np.unique(scores[validation_rows])
            candidates = [float(np.nextafter(observed[0], -np.inf)), *map(float, observed)]
            points = []
            for threshold_value in candidates:
                mask = scores[validation_rows] <= threshold_value
                points.append((
                    float(np.mean(~np.any(mask & labels[validation_rows].astype(bool), axis=1))),
                    float(mask.sum(axis=1).mean()),
                    threshold_value,
                ))
            risk, calls, threshold = min(points)
        trajectory.append({
            "method": name,
            "validation_certified": bool(result.feasible),
            "validation_calls": calls,
            "validation_standard_all_miss_risk": risk,
            "validation_threshold": threshold,
            "validation_non_certification_reason": result.non_certification_reason,
        })
    certified = [record for record in trajectory if record["validation_certified"]]
    if certified:
        chosen = min(
            certified,
            key=lambda record: (
                record["validation_calls"],
                record["validation_standard_all_miss_risk"],
                record["method"],
            ),
        )
        fallback = False
    else:
        chosen = min(
            trajectory,
            key=lambda record: (
                record["validation_standard_all_miss_risk"],
                record["validation_calls"],
                record["method"],
            ),
        )
        fallback = True
    return {
        "selected_method": chosen["method"],
        "selection_source": "validation_only",
        "selection_fallback_used": fallback,
        "validation_trajectory": trajectory,
    }


def recompute_aggregates(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rebuild every summary solely from query-level redraw records."""

    output: dict[str, Any] = {}
    for cell in cells:
        alpha_key = str(cell["alpha"])
        method_names = sorted({
            name for redraw in cell["redraws"] for name in redraw["methods"]
        })
        methods: dict[str, Any] = {}
        for name in method_names:
            records = [redraw["methods"][name] for redraw in cell["redraws"]]
            certified = [record for record in records if record["certified"]]
            call_values = [
                value
                for record in certified
                for value in (record["test_calls_by_query"] or [])
            ]
            miss_values = [
                value
                for record in certified
                for value in (record["test_miss_by_query"] or [])
            ]
            methods[name] = {
                "redraw_count": len(records),
                "certified_redraw_count": len(certified),
                "certification_rate": len(certified) / len(records) if records else None,
                "test_call_sum": int(sum(call_values)),
                "test_query_count": len(call_values),
                "mean_calls": float(np.mean(call_values)) if call_values else None,
                "test_miss_count": int(sum(miss_values)),
                "standard_all_miss_risk": float(np.mean(miss_values)) if miss_values else None,
            }
        output[alpha_key] = {"methods": methods}
    return output


def run_paired_suite(
    task: Any,
    *,
    pool_size: int,
    pool_seed: int,
    redraw_start: int,
    redraw_end: int,
    knn_k: int,
    base_seed: int = BASE_SEED,
    alphas: Sequence[float] = ALPHA_GRID,
    include_e3: bool = False,
    include_chain_discount: bool = False,
    learned_ensembles: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    knn_function: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    protocol: str = "routereval",
) -> dict[str, Any]:
    """Run one deterministic benchmark, pool, and contiguous redraw shard."""

    if protocol not in {"routereval", "routerbench"}:
        raise ValueError("protocol must be routereval or routerbench")
    if redraw_start < 0 or redraw_end <= redraw_start:
        raise ValueError("redraw range must be non-empty and non-negative")
    if tuple(float(value) for value in alphas) != ALPHA_GRID:
        raise ValueError(f"canonical suite requires target grid {ALPHA_GRID}")
    if len(task.query_ids) != len(set(task.query_ids)):
        raise ValueError("task query IDs must be unique")
    columns = subsample_pool(len(task.model_ids), pool_size, seed=pool_seed)
    model_ids = [task.model_ids[int(index)] for index in columns]
    labels = np.asarray(task.correctness[:, columns], dtype=np.int8)
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("task correctness must be binary")
    embeddings = np.asarray(task.embeddings, dtype=float)
    if embeddings.shape[0] != len(task.query_ids):
        raise ValueError("task embeddings and query IDs are not aligned")

    ensembles = dict(learned_ensembles or {})
    for family, members in ensembles.items():
        if family not in LEARNED_FAMILIES:
            raise ValueError(f"unknown learned family {family}")
        seeds = [int(member["identity"]["training_seed"]) for member in members]
        if seeds != list(TRAINING_SEEDS):
            raise ValueError(f"{family} checkpoints must use seeds {TRAINING_SEEDS} in order")
        for member in members:
            identity = member["identity"]
            if identity.get("pool_size") != pool_size or identity.get("pool_seed") != pool_seed:
                raise ValueError(f"{family} checkpoint identity differs from suite pool")
            if identity.get("model_ids") != model_ids:
                raise ValueError(f"{family} checkpoint model IDs differ from suite pool")

    cells = [{"alpha": float(alpha), "redraws": []} for alpha in alphas]
    split_roles = ("design", "router", "final_calibration", "validation", "test")
    checkpoint_identities = {
        family: [dict(member["identity"]) for member in members]
        for family, members in sorted(ensembles.items())
    }
    knn_impl = knn_function or _knn

    for global_redraw in range(redraw_start, redraw_end):
        split = _target_split(len(task.query_ids), global_redraw, base_seed)
        if tuple(split) != split_roles:
            raise RuntimeError("registered split roles or order changed")
        split_ids = _split_query_ids(task, split)
        probabilities, _ = knn_impl(
            embeddings[split["router"]],
            labels[split["router"]],
            embeddings,
            knn_k,
        )
        if probabilities.shape != labels.shape:
            raise ValueError("KNN probability shape differs from sampled labels")
        split_seed = base_seed + SPLIT_STEP * global_redraw
        score_families = _score_nonlearned(
            probabilities,
            labels[split["design"]],
            model_ids,
            base_seed=base_seed,
            split_seed=split_seed,
            preserve_score_prefixes=include_chain_discount,
        )
        if include_chain_discount:
            chain_candidates, chain_profile = _score_chain_discount_candidates(
                probabilities,
                labels[split["design"]],
                model_ids,
                score_families["fixed_tally_factorized"],
                base_seed=base_seed,
                split_seed=split_seed,
            )
        else:
            chain_candidates, chain_profile = None, None
        learned_native_lengths: dict[str, list[int]] = {}
        for family, members in sorted(ensembles.items()):
            for member in members:
                seed = int(member["identity"]["training_seed"])
                if family == "flexrouter":
                    orders, native_lengths = full_flexrouter_orders(member["model"], embeddings)
                    learned_native_lengths[f"flexrouter_seed_{seed}"] = native_lengths
                else:
                    orders = full_barp_orders(member["model"], embeddings)
                raw = per_query_orders_factorized_scores(probabilities, orders)
                score_families[f"{family}_seed_{seed}"] = _smooth(
                    raw,
                    base_seed=base_seed,
                    split_seed=split_seed,
                    preserve_prefixes=include_chain_discount,
                )

        calibration_rows = split["final_calibration"]
        test_rows = split["test"]
        full_pool_cal_miss = np.all(labels[calibration_rows] == 0, axis=1).astype(int)
        full_pool_test_miss = np.all(labels[test_rows] == 0, axis=1).astype(int)
        full_pool = {
            "model_ids": model_ids,
            "model_ids_sha256": hash_json(model_ids),
            "calibration_query_ids": split_ids["final_calibration"],
            "calibration_miss_by_query": [int(value) for value in full_pool_cal_miss],
            "calibration_misses": int(full_pool_cal_miss.sum()),
            "calibration_count": int(len(calibration_rows)),
            "test_query_ids": split_ids["test"],
            "test_miss_by_query": [int(value) for value in full_pool_test_miss],
            "test_misses": int(full_pool_test_miss.sum()),
            "test_count": int(len(test_rows)),
        }

        for cell in cells:
            alpha = float(cell["alpha"])
            methods = {}
            for name, scores in score_families.items():
                family = name.split("_seed_", 1)[0] if "_seed_" in name else name
                training_seed = int(name.rsplit("_", 1)[1]) if "_seed_" in name else None
                methods[name] = evaluate_score_family(
                    scores,
                    labels,
                    calibration_rows,
                    test_rows,
                    task.query_ids,
                    model_ids,
                    alpha,
                    method_family=family,
                    training_seed=training_seed,
                    native_dpp_prefix_length=learned_native_lengths.get(name),
                )
            if include_chain_discount:
                if chain_candidates is None or chain_profile is None:
                    raise RuntimeError("ISSUE-77 chain candidates were not fitted")
                endpoint_result = evaluate_score_family(
                    chain_candidates[0],
                    labels,
                    calibration_rows,
                    test_rows,
                    task.query_ids,
                    model_ids,
                    alpha,
                    method_family="fixed_tally_factorized",
                )
                if endpoint_result != methods["fixed_tally_factorized"]:
                    raise RuntimeError(
                        "ISSUE-77 discount-off endpoint calibrated result differs "
                        "from fixed_tally_factorized"
                    )
                chain_selection = _select_chain_discount_candidate(
                    chain_candidates, labels, split["validation"], alpha
                )
                chosen = chain_selection["chosen"]
                chain_result = evaluate_score_family(
                    chain_selection["scores"],
                    labels,
                    calibration_rows,
                    test_rows,
                    task.query_ids,
                    model_ids,
                    alpha,
                    method_family=CHAIN_DISCOUNT_METHOD,
                )
                minimum = int(chosen["minimum_survivor_count"])
                chain_result.update({
                    "chosen_hyper": {"minimum_survivor_count": minimum},
                    "discount_fully_off": minimum == 0,
                    "selection_source": chain_selection["selection_source"],
                    "selection_fallback_used": chain_selection["selection_fallback_used"],
                    "endpoint_reference_method": "fixed_tally_factorized",
                    "endpoint_result_identity_verified": True,
                    "validation_calls": chosen["validation_calls"],
                    "validation_standard_all_miss_risk": chosen[
                        "validation_standard_all_miss_risk"
                    ],
                    "validation_trajectory": chain_selection["validation_trajectory"],
                    "chain_discount_profile": chain_discount_diagnostics(
                        chain_profile, minimum_survivor_count=minimum
                    ),
                })
                methods[CHAIN_DISCOUNT_METHOD] = chain_result
            if include_e3:
                selection = _validation_choice(
                    score_families, labels, split["validation"], alpha
                )
                selected = selection["selected_method"]
                e3 = evaluate_score_family(
                    score_families[selected],
                    labels,
                    calibration_rows,
                    test_rows,
                    task.query_ids,
                    model_ids,
                    alpha,
                    method_family="e3_selected_nonlearned",
                )
                e3.update(selection)
                methods["e3_selected_nonlearned"] = e3
            cell["redraws"].append({
                "redraw_id": int(global_redraw),
                "split_seed": int(split_seed),
                "split_ids": split_ids,
                "full_pool": full_pool,
                "methods": methods,
            })

    registered_methods = sorted(
        set(NONLEARNED_METHODS)
        | ({CHAIN_DISCOUNT_METHOD} if include_chain_discount else set())
        | {
            f"{family}_seed_{seed}"
            for family in ensembles
            for seed in TRAINING_SEEDS
        }
        | ({"e3_selected_nonlearned"} if include_e3 else set())
    )
    assert_registered_arm_presence(cells, registered_methods)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "design": (
            "ISSUE_77_chain_discounted_stopping"
            if include_chain_discount
            else "ISSUE_50_unified_paired_main_suite"
        ),
        "protocol": protocol,
        "benchmark": task.task,
        "pool_id": hash_json({"pool_seed": pool_seed, "model_ids": model_ids}),
        "pool_size": int(pool_size),
        "pool_seed": int(pool_seed),
        "base_seed": int(base_seed),
        "knn_k": int(knn_k),
        "redraw_start": int(redraw_start),
        "redraw_end": int(redraw_end),
        "redraw_range": {
            "start_inclusive": int(redraw_start),
            "end_exclusive": int(redraw_end),
            "count": int(redraw_end - redraw_start),
        },
        "target_grid": [float(value) for value in alphas],
        "include_e3": bool(include_e3),
        "include_chain_discount": bool(include_chain_discount),
        "chain_discount_min_survivor_grid": (
            list(CHAIN_DISCOUNT_MIN_SURVIVOR_GRID)
            if include_chain_discount
            else None
        ),
        "query_ids": list(task.query_ids),
        "query_ids_sha256": hash_json(list(task.query_ids)),
        "model_ids": model_ids,
        "model_ids_sha256": hash_json(model_ids),
        "sampled_model_column_indices": [int(value) for value in columns],
        "source_hashes": dict(task.source_hashes),
        "feature_audit": dict(task.feature_audit),
        "checkpoint_identities": checkpoint_identities,
        "method_identities": registered_methods,
        "cells": cells,
    }
    payload["aggregates"] = recompute_aggregates(cells)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("routereval", "routerbench"), default="routereval")
    parser.add_argument("--leaderboard", type=Path)
    parser.add_argument("--router-dir", type=Path)
    parser.add_argument("--routerbench-pickle", type=Path)
    parser.add_argument("--embedding-archive", type=Path)
    parser.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pool-size", required=True, type=int)
    parser.add_argument("--pool-seed", type=int, default=BASE_SEED)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--knn-k", type=int, default=40)
    parser.add_argument("--redraw-start", required=True, type=int)
    parser.add_argument("--redraw-end", required=True, type=int)
    parser.add_argument("--flexrouter-checkpoint-dir", type=Path)
    parser.add_argument("--barp-checkpoint-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-e3", action="store_true")
    parser.add_argument(
        "--include-chain-discount",
        action="store_true",
        help="register the ISSUE-77 fixed tally chain discounted arm",
    )
    parser.add_argument("--nonlearned-only", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.out.exists():
        raise SystemExit(f"refusing to overwrite immutable result {arguments.out}")
    if arguments.redraw_start < 0 or arguments.redraw_end <= arguments.redraw_start:
        raise SystemExit("--redraw-end must be greater than --redraw-start")
    if arguments.knn_k <= 0:
        raise SystemExit("--knn-k must be positive")
    try:
        task = _load_task(
            arguments.protocol,
            arguments.benchmark,
            leaderboard=arguments.leaderboard,
            router_dir=arguments.router_dir,
            routerbench_pickle=arguments.routerbench_pickle,
            embedding_archive=arguments.embedding_archive,
            embedding_manifest=arguments.embedding_manifest,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    columns = subsample_pool(len(task.model_ids), arguments.pool_size, seed=arguments.pool_seed)
    model_ids = [task.model_ids[int(index)] for index in columns]
    from components.stage1_task_adapters import checkpoint_target_family
    expected_target_family = checkpoint_target_family(arguments.protocol, arguments.benchmark)

    ensembles: dict[str, Sequence[Mapping[str, Any]]] = {}
    if not arguments.nonlearned_only:
        missing = [
            option
            for option, value in (
                ("--flexrouter-checkpoint-dir", arguments.flexrouter_checkpoint_dir),
                ("--barp-checkpoint-dir", arguments.barp_checkpoint_dir),
            )
            if value is None
        ]
        if missing:
            raise SystemExit("learned execution requires " + ", ".join(missing))
        ensembles = {
            "flexrouter": load_checkpoint_ensemble(
                arguments.flexrouter_checkpoint_dir,
                "flexrouter",
                pool_size=arguments.pool_size,
                pool_seed=arguments.pool_seed,
                model_ids=model_ids,
                device=arguments.device,
                expected_source=(arguments.protocol if expected_target_family is not None else None),
                expected_target_family=expected_target_family,
            ),
            "barp": load_checkpoint_ensemble(
                arguments.barp_checkpoint_dir,
                "barp",
                pool_size=arguments.pool_size,
                pool_seed=arguments.pool_seed,
                model_ids=model_ids,
                device=arguments.device,
                expected_source=(arguments.protocol if expected_target_family is not None else None),
                expected_target_family=expected_target_family,
            ),
        }
    payload = run_paired_suite(
        task,
        pool_size=arguments.pool_size,
        pool_seed=arguments.pool_seed,
        redraw_start=arguments.redraw_start,
        redraw_end=arguments.redraw_end,
        knn_k=arguments.knn_k,
        base_seed=arguments.base_seed,
        include_e3=arguments.include_e3,
        include_chain_discount=arguments.include_chain_discount,
        learned_ensembles=ensembles,
        protocol=arguments.protocol,
    )
    atomic_write_json(arguments.out, payload)
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
