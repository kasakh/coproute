"""Run ISSUE-52 fixed-depth fit-transfer curves.

Each shard freezes method scores from either the target task or a different
source task, then evaluates exact target-test prefixes through depth ten. No
calibration cutoff is fitted in this artifact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import socket
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.routereval_protocol import subsample_pool  # noqa: E402
from scripts import run_issue52_transfer as transfer  # noqa: E402
from scripts import run_paired_main_suite as paired  # noqa: E402

SCHEMA_VERSION = "issue52_fixed_depth_v1"
MAX_DEPTH = 10
REDRAW_COUNT = transfer.REDRAW_COUNT
FIT_MODES = transfer.FIT_MODES


def _stable_orders(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("frozen scores must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("frozen scores must be finite")
    return np.argsort(values, axis=1, kind="stable")


def _fixed_depth_record(
    scores: np.ndarray,
    target_test_labels: np.ndarray,
    target_test_rows: np.ndarray,
    target_query_ids: Sequence[str],
    model_ids: Sequence[str],
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(target_test_labels, dtype=np.int8)
    rows = np.asarray(target_test_rows, dtype=int)
    if values.shape[1] != len(model_ids):
        raise ValueError("frozen score columns differ from sampled model IDs")
    if labels.shape != (len(rows), len(model_ids)):
        raise ValueError("target test labels differ from test rows and sampled model IDs")
    orders = _stable_orders(values)
    test_orders = orders[rows]
    test_ids = [str(target_query_ids[int(row)]) for row in rows]
    records = []
    previous: list[list[str]] | None = None
    for depth in range(1, min(MAX_DEPTH, len(model_ids)) + 1):
        selected_indices = test_orders[:, :depth]
        selected_ids = [
            [str(model_ids[int(index)]) for index in order]
            for order in selected_indices
        ]
        if previous is not None:
            for old, new in zip(previous, selected_ids):
                if new[:-1] != old:
                    raise RuntimeError("fixed-depth selections are not nested prefixes")
        successes = np.asarray(
            [
                bool(np.any(labels[row_index, selected_indices[row_index]] == 1))
                for row_index in range(len(rows))
            ],
            dtype=bool,
        )
        misses = ~successes
        success_count = int(np.count_nonzero(successes))
        miss_count = int(np.count_nonzero(misses))
        trial_count = int(len(rows))
        if success_count + miss_count != trial_count:
            raise RuntimeError("fixed-depth event counts do not sum to the trial count")
        records.append(
            {
                "depth": int(depth),
                "target_query_ids": test_ids,
                "selected_model_ids_by_query": selected_ids,
                "target_success_by_query": [bool(value) for value in successes],
                "target_miss_by_query": [int(value) for value in misses],
                "target_success_count": success_count,
                "target_miss_count": miss_count,
                "target_trial_count": trial_count,
                "target_success": success_count / trial_count,
                "target_all_miss_risk": miss_count / trial_count,
                "mean_calls": float(depth),
            }
        )
        previous = selected_ids

    full_indices = test_orders
    full_selected_ids = [
        [str(model_ids[int(index)]) for index in order]
        for order in full_indices
    ]
    endpoint_misses = np.asarray(
        [
            not bool(np.any(labels[row_index, full_indices[row_index]] == 1))
            for row_index in range(len(rows))
        ],
        dtype=np.int8,
    )
    direct_misses = np.all(labels == 0, axis=1).astype(np.int8)
    if not np.array_equal(endpoint_misses, direct_misses):
        raise RuntimeError("full-pool endpoint does not reproduce direct target misses")
    return {
        "comparison_kind": "fixed_depth_transfer",
        "risk_certificate": False,
        "depths": records,
        "endpoint": {
            "depth": int(len(model_ids)),
            "target_query_ids": test_ids,
            "selected_model_ids_by_query": full_selected_ids,
            "target_miss_by_query": [int(value) for value in endpoint_misses],
            "target_miss_count": int(endpoint_misses.sum()),
            "target_trial_count": int(len(rows)),
            "target_all_miss_risk": float(np.mean(endpoint_misses)),
            "reproduces_complete_pool": True,
            "model_ids": list(model_ids),
            "model_ids_sha256": paired.hash_json(list(model_ids)),
        },
    }


def _assert_chain_rank_identity(candidates: Mapping[int, np.ndarray]) -> None:
    if set(candidates) != set(paired.CHAIN_DISCOUNT_MIN_SURVIVOR_GRID):
        raise RuntimeError("chain discount candidate grid is incomplete")
    endpoint_order = _stable_orders(candidates[0])
    for minimum, scores in candidates.items():
        if not np.array_equal(_stable_orders(scores), endpoint_order):
            raise RuntimeError(
                "chain discount fixed-depth order changed at "
                f"minimum_survivor_count={minimum}"
            )


def assert_registered_arm_presence(
    redraws: Sequence[Mapping[str, Any]], expected_methods: Sequence[str]
) -> None:
    expected = set(expected_methods)
    if not redraws:
        raise RuntimeError("ISSUE-52 fixed-depth runner emitted no redraws")
    for redraw in redraws:
        methods = redraw.get("methods", {})
        missing = sorted(expected - set(methods))
        unexpected = sorted(set(methods) - expected)
        if missing or unexpected:
            raise RuntimeError(
                "ISSUE-52 fixed-depth registered arm mismatch at "
                f"redraw={redraw.get('redraw_id')}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name, method in methods.items():
            depths = method.get("fixed_depth", {}).get("depths", [])
            if not depths:
                raise RuntimeError(f"{name} emitted no fixed-depth rows")


def run_fixed_depth_transfer(
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
    knn_function: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    checkpoint_identity_target_family: str | None = "strict_target_family",
) -> dict[str, Any]:
    """Evaluate one source and target pair over a fixed-depth redraw shard."""

    if protocol not in ("routereval", "routerbench"):
        raise ValueError("protocol must be routereval or routerbench")
    if fit_mode not in FIT_MODES:
        raise ValueError(f"fit_mode must be one of {FIT_MODES}")
    if not 0 <= redraw_start < redraw_end <= REDRAW_COUNT:
        raise ValueError(f"redraw range must lie within 0 through {REDRAW_COUNT}")
    if base_seed != paired.BASE_SEED:
        raise ValueError(f"ISSUE-52 requires base_seed={paired.BASE_SEED}")
    if knn_k <= 0:
        raise ValueError("knn_k must be positive")

    transfer._validate_task_shape(source_task, "source task")
    transfer._validate_task_shape(target_task, "target task")
    if tuple(source_task.model_ids) != tuple(target_task.model_ids):
        raise ValueError("source and target model columns must be exactly identical")
    same_task = transfer._same_task(source_task, target_task)
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
    transfer._validate_ensembles(
        learned_ensembles,
        pool_size=pool_size,
        pool_seed=pool_seed,
        model_ids=model_ids,
        protocol=protocol,
        target_family=expected_checkpoint_family,
    )
    expected_methods = transfer.registered_methods()
    knn_impl = knn_function or paired._knn
    source_embeddings = np.asarray(source_task.embeddings, dtype=float)
    target_embeddings = np.asarray(target_task.embeddings, dtype=float)
    if source_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError("source and target embedding dimensions must be identical")

    redraws = []
    for redraw_id in range(redraw_start, redraw_end):
        source_split = paired._target_split(len(source_task.query_ids), redraw_id, base_seed)
        target_split = paired._target_split(len(target_task.query_ids), redraw_id, base_seed)
        source_split_ids = transfer._split_ids(source_task, source_split)
        target_split_ids = transfer._split_ids(target_task, target_split)
        split_seed = base_seed + paired.SPLIT_STEP * redraw_id

        source_design_labels = transfer._labels_at(
            source_task, source_split["design"], columns, "source design"
        )
        source_router_labels = transfer._labels_at(
            source_task, source_split["router"], columns, "source router"
        )
        target_test_labels = transfer._labels_at(
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
        target_probabilities = np.asarray(target_probabilities, dtype=float)
        if target_probabilities.shape != (len(target_task.query_ids), pool_size):
            raise ValueError("target KNN score shape differs from target task and pool")

        target_scores, target_chain, target_profile = transfer._score_all_queries(
            target_probabilities,
            target_embeddings,
            source_design_labels,
            model_ids,
            learned_ensembles,
            base_seed=base_seed,
            split_seed=split_seed,
        )
        if not np.array_equal(
            target_chain[0], target_scores["fixed_tally_factorized"]
        ):
            raise RuntimeError("chain discount-off endpoint mismatch")
        _assert_chain_rank_identity(target_chain)
        target_scores[paired.CHAIN_DISCOUNT_METHOD] = target_chain[0]

        methods: dict[str, Any] = {}
        for method in expected_methods:
            scores = target_scores[method]
            training_seed = (
                int(method.rsplit("_", 1)[1]) if "_seed_" in method else None
            )
            if training_seed is None:
                fit_origin = (
                    f"{protocol}:{target_task.task}"
                    if fit_mode == "in_domain"
                    else f"{protocol}:{source_task.task}"
                )
                checkpoint_identity = None
            else:
                family = method.split("_seed_", 1)[0]
                member = next(
                    item
                    for item in learned_ensembles[family]
                    if int(item["identity"]["training_seed"]) == training_seed
                )
                checkpoint_identity = dict(member["identity"])
                fit_origin = (
                    f"{protocol}:target_family_heldout_checkpoint:{target_family}"
                )
            record = {
                "method_family": method.split("_seed_", 1)[0],
                "training_seed": training_seed,
                "fit_mode": fit_mode,
                "fit_origin": fit_origin,
                "checkpoint_identity": checkpoint_identity,
                "frozen_target_scores_sha256": transfer._array_sha256(scores),
                "frozen_target_score_shape": list(scores.shape),
                "target_test_scores_sha256": transfer._array_sha256(
                    scores[target_split["test"]]
                ),
                "model_ids": model_ids,
                "model_ids_sha256": paired.hash_json(model_ids),
                "preserve_prefixes": True,
                "fixed_depth": _fixed_depth_record(
                    scores,
                    target_test_labels,
                    target_split["test"],
                    target_task.query_ids,
                    model_ids,
                ),
            }
            if method == paired.CHAIN_DISCOUNT_METHOD:
                record.update(
                    {
                        "fixed_depth_score_origin": "discount_off_endpoint",
                        "endpoint_reference_method": "fixed_tally_factorized",
                        "endpoint_reproduction_exact": True,
                        "rank_identity_across_shrinkage_grid": True,
                        "chain_discount_min_survivor_grid": list(
                            paired.CHAIN_DISCOUNT_MIN_SURVIVOR_GRID
                        ),
                        "chain_candidate_score_sha256": {
                            str(minimum): transfer._array_sha256(candidate)
                            for minimum, candidate in target_chain.items()
                        },
                        "chain_order": list(target_profile.order),
                    }
                )
            methods[method] = record

        fixed = methods["fixed_tally_factorized"]
        chain = methods[paired.CHAIN_DISCOUNT_METHOD]
        if (
            fixed["frozen_target_scores_sha256"]
            != chain["frozen_target_scores_sha256"]
            or fixed["fixed_depth"] != chain["fixed_depth"]
        ):
            raise RuntimeError("chain endpoint did not reproduce fixed tally exactly")

        direct_full_pool = transfer._full_pool_record(
            target_test_labels, target_split_ids["test"]
        )
        for name, method in methods.items():
            endpoint = method["fixed_depth"]["endpoint"]
            if endpoint["target_miss_by_query"] != direct_full_pool["miss_by_query"]:
                raise RuntimeError(f"{name} endpoint differs from complete-pool misses")

        redraws.append(
            {
                "redraw_id": int(redraw_id),
                "split_seed": int(split_seed),
                "source_split_ids": source_split_ids,
                "target_split_ids": target_split_ids,
                "fit_label_access": {
                    "design_origin": f"{protocol}:{source_task.task}",
                    "router_origin": f"{protocol}:{source_task.task}",
                    "source_label_roles": ["design", "router"],
                    "target_fit_label_roles": (
                        ["design", "router"] if fit_mode == "in_domain" else []
                    ),
                    "target_evaluation_label_roles": ["test"],
                },
                "target_complete_pool": direct_full_pool,
                "methods": methods,
            }
        )

    assert_registered_arm_presence(redraws, expected_methods)
    return {
        "schema_version": SCHEMA_VERSION,
        "design": "ISSUE_52_fixed_depth_fit_transfer",
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
        "maximum_fixed_depth": int(min(MAX_DEPTH, pool_size)),
        "redraw_start": int(redraw_start),
        "redraw_end": int(redraw_end),
        "redraw_range": {
            "start_inclusive": int(redraw_start),
            "end_exclusive": int(redraw_end),
            "count": int(redraw_end - redraw_start),
            "registered_total": REDRAW_COUNT,
        },
        "method_identities": list(expected_methods),
        "preserve_prefixes": True,
        "stable_tie_rule": "sampled_model_column_order",
        "checkpoint_identities": {
            family: [dict(member["identity"]) for member in learned_ensembles[family]]
            for family in paired.LEARNED_FAMILIES
        },
        "redraws": redraws,
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
    if not transfer.STAMP_PATTERN.fullmatch(run_stamp):
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
            else paired._load_task(
                arguments.protocol, arguments.target_benchmark, **load_options
            )
        )
        source_family = transfer._task_family(arguments.protocol, source_benchmark)
        target_family = transfer._task_family(
            arguments.protocol, arguments.target_benchmark
        )
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
                expected_source=(
                    arguments.protocol if expected_checkpoint_family is not None else None
                ),
                expected_target_family=expected_checkpoint_family,
            )
            for family, directory in (
                ("flexrouter", arguments.flexrouter_checkpoint_dir),
                ("barp", arguments.barp_checkpoint_dir),
            )
        }
        payload = run_fixed_depth_transfer(
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
