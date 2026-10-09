"""Train immutable target-family-held-out FlexRouter and BaRP checkpoints.

The target family is never included in fitting or early stopping. This runner
writes one immutable checkpoint directory per method and seed. A separate
finalize action verifies the ten checkpoints in one family and pool group and
writes its immutable family manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from components.barp_baseline import BaRPBaseline
from components.data_protocol import (
    PRIMARY_COVERAGE_POLICY,
    SUPPORTED_TASKS,
    assign_racer_splits,
    derive_coverage_labels,
    load_routerbench,
)
from components.flexrouter_baseline import FlexRouter
from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS
from components.routereval_protocol import subsample_pool
from scripts.run_learned_baselines import (
    EXTERNAL_SPLIT_SEED,
    ROUTEREVAL_KEYS,
    TRAINING_SEEDS,
    LearnedTaskData,
    _hash_json,
    implementation_source_hashes,
    l2_normalize_rows,
    load_aligned_routerbench_embeddings,
    load_routereval_task,
    sha256_file,
    stable_hash_split,
    train_learned_model,
)

SCHEMA_VERSION = 1
PROTOCOL = "target_family_heldout_learned_checkpoints_v1"
METHODS = ("flexrouter", "barp")
ROUTEREVAL_FAMILIES = ("bbh", "gpqa", "math", "ifeval", "musr")
ROUTEREVAL_TARGETS = ("bbh", "gpqa", "math")
ROUTERBENCH_TARGETS = tuple(SUPPORTED_TASKS)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_arrays(metadata: Mapping[str, Any], arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(canonical_json(metadata))
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(canonical_json(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def split_task(task: LearnedTaskData, family: str) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    split = stable_hash_split(
        task.query_ids,
        {"fit": 0.80, "validation": 0.20},
        seed=EXTERNAL_SPLIT_SEED,
        namespace=f"target_heldout:{family}:{task.task}",
    )
    ids = {
        role: [task.query_ids[int(index)] for index in rows]
        for role, rows in split.items()
    }
    return split["fit"], split["validation"], ids


def cap_rows(rows: np.ndarray, limit: int | None) -> np.ndarray:
    return rows if limit is None else rows[: min(len(rows), limit)]


def routereval_training_data(
    leaderboard: Path,
    router_dir: Path,
    target_family: str,
    pool_size: int,
    pool_seed: int,
    smoke: bool,
) -> dict[str, Any]:
    if target_family not in ROUTEREVAL_TARGETS:
        raise ValueError(f"RouterEval target must be one of {ROUTEREVAL_TARGETS}")
    training_families = tuple(name for name in ROUTEREVAL_FAMILIES if name != target_family)
    tasks = {
        name: load_routereval_task(
            leaderboard,
            router_dir / f"{name}_router_dataset.pkl",
            name,
        )
        for name in training_families
    }
    first = tasks[training_families[0]]
    for name, task in tasks.items():
        if task.model_ids != first.model_ids:
            raise ValueError(f"{name}: model columns differ from the shared pool")
        if task.embeddings.shape[1] != first.embeddings.shape[1]:
            raise ValueError(f"{name}: embedding dimension differs from the shared pool")
    if pool_size <= 0 or pool_size > len(first.model_ids):
        raise ValueError("pool size is outside the shared RouterEval model pool")
    columns = (
        np.arange(pool_size, dtype=int)
        if pool_size == len(first.model_ids)
        else subsample_pool(len(first.model_ids), pool_size, seed=pool_seed)
    )
    fit_x: list[np.ndarray] = []
    fit_y: list[np.ndarray] = []
    validation_x: list[np.ndarray] = []
    validation_y: list[np.ndarray] = []
    split_ids: dict[str, Any] = {}
    for name in training_families:
        task = tasks[name]
        fit_rows, validation_rows, ids = split_task(task, target_family)
        fit_rows = cap_rows(fit_rows, 128 if smoke else None)
        validation_rows = cap_rows(validation_rows, 64 if smoke else None)
        split_ids[name] = {
            "fit": [task.query_ids[int(index)] for index in fit_rows],
            "validation": [task.query_ids[int(index)] for index in validation_rows],
        }
        fit_x.append(task.embeddings[fit_rows])
        fit_y.append(task.correctness[fit_rows][:, columns])
        validation_x.append(task.embeddings[validation_rows])
        validation_y.append(task.correctness[validation_rows][:, columns])
    train_embeddings = np.concatenate(fit_x, axis=0)
    train_labels = np.concatenate(fit_y, axis=0)
    validation_embeddings = np.concatenate(validation_x, axis=0)
    validation_labels = np.concatenate(validation_y, axis=0)
    model_ids = tuple(first.model_ids[int(index)] for index in columns)
    hash_metadata = {
        "protocol": PROTOCOL,
        "source": "routereval",
        "excluded_family": target_family,
        "training_families": list(training_families),
        "model_ids": list(model_ids),
        "split_ids": split_ids,
    }
    return {
        "train_embeddings": train_embeddings,
        "train_labels": train_labels,
        "validation_embeddings": validation_embeddings,
        "validation_labels": validation_labels,
        "model_ids": model_ids,
        "column_indices": [int(value) for value in columns],
        "training_families": training_families,
        "split_ids": split_ids,
        "feature_audits": {name: dict(tasks[name].feature_audit) for name in training_families},
        "source_hashes": {name: dict(tasks[name].source_hashes) for name in training_families},
        "training_data_sha256": hash_arrays(
            hash_metadata,
            (train_embeddings, train_labels, validation_embeddings, validation_labels),
        ),
        "embedding_dim": int(train_embeddings.shape[1]),
    }


def routerbench_training_data(
    routerbench_pickle: Path,
    embedding_archive: Path,
    embedding_manifest: Path | None,
    target_family: str,
    smoke: bool,
) -> dict[str, Any]:
    if target_family not in ROUTERBENCH_TARGETS:
        raise ValueError(f"RouterBench target must be one of {ROUTERBENCH_TARGETS}")
    data = load_routerbench(
        routerbench_pickle,
        shot="0shot",
        model_columns=ROUTERBENCH_MODEL_COLUMNS,
    )
    embeddings, embedding_audit = load_aligned_routerbench_embeddings(
        embedding_archive,
        data.query_ids,
        manifest_path=embedding_manifest,
    )
    labels = derive_coverage_labels(data, policy=PRIMARY_COVERAGE_POLICY).standard
    assignment = assign_racer_splits(data, seed=EXTERNAL_SPLIT_SEED)
    row_of = {query_id: row for row, query_id in enumerate(data.query_ids)}
    training_families = tuple(name for name in ROUTERBENCH_TARGETS if name != target_family)

    def rows_for(role_names: Sequence[str]) -> np.ndarray:
        ids = [
            query_id
            for family in training_families
            for role in role_names
            for query_id in assignment.ids(family, role)
        ]
        return np.asarray([row_of[query_id] for query_id in ids], dtype=int)

    fit_rows = cap_rows(rows_for(("train", "design")), 512 if smoke else None)
    validation_rows = cap_rows(rows_for(("validation",)), 256 if smoke else None)
    if np.any(data.frame.iloc[fit_rows]["task"].to_numpy() == target_family):
        raise RuntimeError("target family entered RouterBench fit rows")
    if np.any(data.frame.iloc[validation_rows]["task"].to_numpy() == target_family):
        raise RuntimeError("target family entered RouterBench validation rows")
    split_ids = {
        family: {
            role: list(assignment.ids(family, role))
            for role in ("train", "design", "validation")
        }
        for family in training_families
    }
    train_embeddings = embeddings[fit_rows]
    train_labels = labels[fit_rows]
    validation_embeddings = embeddings[validation_rows]
    validation_labels = labels[validation_rows]
    model_ids = tuple(data.model_columns)
    hash_metadata = {
        "protocol": PROTOCOL,
        "source": "routerbench",
        "excluded_family": target_family,
        "training_families": list(training_families),
        "model_ids": list(model_ids),
        "fit_query_ids": [data.query_ids[int(index)] for index in fit_rows],
        "validation_query_ids": [data.query_ids[int(index)] for index in validation_rows],
    }
    return {
        "train_embeddings": train_embeddings,
        "train_labels": train_labels,
        "validation_embeddings": validation_embeddings,
        "validation_labels": validation_labels,
        "model_ids": model_ids,
        "column_indices": list(range(len(model_ids))),
        "training_families": training_families,
        "split_ids": split_ids,
        "feature_audits": {
            str(item["task"]): dict(item)
            for item in data.prompt_degeneracy
            if str(item["task"]) in training_families
        },
        "source_hashes": {
            "routerbench": data.source_sha256,
            "embeddings": embedding_audit["source_sha256"],
            "embedding_manifest": embedding_audit["manifest_sha256"],
        },
        "embedding_audit": embedding_audit,
        "training_data_sha256": hash_arrays(
            hash_metadata,
            (train_embeddings, train_labels, validation_embeddings, validation_labels),
        ),
        "embedding_dim": int(train_embeddings.shape[1]),
    }


def round_trip_check(
    method: str,
    checkpoint: Path,
    validation_embeddings: np.ndarray,
    model_ids: Sequence[str],
    device: str,
) -> dict[str, Any]:
    sample = validation_embeddings[: min(2, len(validation_embeddings))]
    if method == "flexrouter":
        loaded = FlexRouter.load(checkpoint, device=device)
        values = loaded.predict_quality(sample).detach().cpu().numpy()
    else:
        loaded = BaRPBaseline.load(checkpoint, device=device)
        values = loaded.logits(sample, [1.0, 0.0]).detach().cpu().numpy()
        if tuple(loaded.model_ids) != tuple(model_ids):
            raise ValueError("BaRP checkpoint model IDs changed during round trip")
    expected_shape = (len(sample), len(model_ids))
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError("checkpoint round trip produced invalid predictions")
    return {
        "passed": True,
        "prediction_shape": list(values.shape),
        "prediction_sha256": hash_arrays({"method": method}, (values,)),
    }


def train_checkpoint(arguments: argparse.Namespace) -> Path:
    destination = Path(arguments.out_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable output directory {destination}")
    if arguments.training_seed not in TRAINING_SEEDS:
        raise ValueError(f"training seed must be one of {TRAINING_SEEDS}")
    if arguments.source == "routereval":
        if not arguments.leaderboard or not arguments.router_dir or arguments.pool_size is None:
            raise ValueError("RouterEval needs leaderboard, router-dir, and pool-size")
        bundle = routereval_training_data(
            Path(arguments.leaderboard),
            Path(arguments.router_dir),
            arguments.target_family,
            arguments.pool_size,
            arguments.pool_seed,
            arguments.smoke,
        )
    else:
        if not arguments.routerbench_pickle or not arguments.embedding_archive:
            raise ValueError("RouterBench needs routerbench-pickle and embedding-archive")
        bundle = routerbench_training_data(
            Path(arguments.routerbench_pickle),
            Path(arguments.embedding_archive),
            Path(arguments.embedding_manifest) if arguments.embedding_manifest else None,
            arguments.target_family,
            arguments.smoke,
        )
    max_epochs = 1 if arguments.smoke else 100
    patience = 1 if arguments.smoke else 15
    model = train_learned_model(
        arguments.method,
        bundle["train_embeddings"],
        bundle["train_labels"],
        bundle["validation_embeddings"],
        bundle["validation_labels"],
        bundle["model_ids"],
        training_seed=arguments.training_seed,
        max_epochs=max_epochs,
        patience=patience,
        device=arguments.device,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=destination.parent))
    try:
        checkpoint = temporary / "checkpoint.pt"
        model.save(checkpoint)
        round_trip = round_trip_check(
            arguments.method,
            checkpoint,
            bundle["validation_embeddings"],
            bundle["model_ids"],
            arguments.device,
        )
        identity = {
            "source": arguments.source,
            "target_family": arguments.target_family,
            "excluded_family": arguments.target_family,
            "method": arguments.method,
            "pool_size": len(bundle["model_ids"]),
            "pool_seed": arguments.pool_seed,
            "training_seed": arguments.training_seed,
            "run_stamp": arguments.run_stamp,
            "host": socket.gethostname(),
            "smoke": bool(arguments.smoke),
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "identity": identity,
            "configuration": {
                "target_family_excluded_from_fit": True,
                "target_family_excluded_from_hyperparameter_selection": True,
                "training_families": list(bundle["training_families"]),
                "fit_fraction_per_training_family": 0.80 if arguments.source == "routereval" else None,
                "validation_fraction_per_training_family": 0.20 if arguments.source == "routereval" else None,
                "split_seed": EXTERNAL_SPLIT_SEED,
                "max_epochs": max_epochs,
                "patience": patience,
                "smoke": bool(arguments.smoke),
            },
            "model_pool": {
                "model_ids": list(bundle["model_ids"]),
                "model_ids_sha256": _hash_json(list(bundle["model_ids"])),
                "column_indices": list(bundle["column_indices"]),
                "pool_size": len(bundle["model_ids"]),
            },
            "training_data_sha256": bundle["training_data_sha256"],
            "source_hashes": bundle["source_hashes"],
            "implementation_source_hashes": {
                **implementation_source_hashes(),
                "target_heldout_runner": sha256_file(Path(__file__)),
            },
            "split_ids": bundle["split_ids"],
            "feature_audits": bundle["feature_audits"],
            "embedding_audit": bundle.get("embedding_audit"),
            "training": {
                "fit_query_count": len(bundle["train_embeddings"]),
                "validation_query_count": len(bundle["validation_embeddings"]),
                "epochs_completed": len(model.training_history),
                "best_epoch": getattr(model, "best_epoch", None),
                "target_labels_used": False,
                "checkpoint_selected_without_target_labels": True,
            },
            "environment": {
                "created_at": utc_now(),
                "host": socket.gethostname(),
                "run_stamp": arguments.run_stamp,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": arguments.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "checkpoint_round_trip": round_trip,
        }
        atomic_json(temporary / "training_history.json", list(model.training_history))
        atomic_json(temporary / "training_metadata.json", metadata)
        files = {
            name: sha256_file(temporary / name)
            for name in ("checkpoint.pt", "training_history.json", "training_metadata.json")
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "immutable": True,
            "identity": identity,
            "excluded_family": arguments.target_family,
            "training_data_sha256": bundle["training_data_sha256"],
            "checkpoint_sha256": files["checkpoint.pt"],
            "files": files,
            "metadata_identity_sha256": _hash_json(identity),
            "host": socket.gethostname(),
            "run_stamp": arguments.run_stamp,
        }
        atomic_json(temporary / "manifest.json", manifest)
        (temporary / "DONE").touch(exist_ok=False)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(destination)
    return destination


def finalize_family(arguments: argparse.Namespace) -> Path:
    root = Path(arguments.out_dir)
    destination = root / "family_manifest.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    checkpoints = []
    training_hashes = set()
    for method in METHODS:
        for seed in TRAINING_SEEDS:
            directory = root / method / f"seed{seed}"
            manifest_path = directory / "manifest.json"
            done_path = directory / "DONE"
            if not manifest_path.is_file() or not done_path.is_file():
                raise ValueError(f"incomplete checkpoint directory {directory}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity = manifest.get("identity", {})
            expected = {
                "source": arguments.source,
                "target_family": arguments.target_family,
                "excluded_family": arguments.target_family,
                "method": method,
                "training_seed": seed,
            }
            for key, value in expected.items():
                if identity.get(key) != value:
                    raise ValueError(f"{manifest_path}: expected {key}={value!r}")
            for name, expected_hash in manifest["files"].items():
                path = directory / name
                if not path.is_file() or sha256_file(path) != expected_hash:
                    raise ValueError(f"{manifest_path}: hash mismatch for {name}")
            training_hashes.add(manifest["training_data_sha256"])
            checkpoints.append(
                {
                    "method": method,
                    "training_seed": seed,
                    "directory": str(directory),
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "manifest_sha256": sha256_file(manifest_path),
                }
            )
    if len(training_hashes) != 1:
        raise ValueError(f"family checkpoints used differing training data hashes: {training_hashes}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "immutable": True,
        "source": arguments.source,
        "target_family": arguments.target_family,
        "excluded_family": arguments.target_family,
        "pool_size": arguments.pool_size,
        "pool_seed": arguments.pool_seed,
        "methods": list(METHODS),
        "training_seeds": list(TRAINING_SEEDS),
        "checkpoint_count": len(checkpoints),
        "training_data_sha256": next(iter(training_hashes)),
        "checkpoints": checkpoints,
        "host": socket.gethostname(),
        "run_stamp": arguments.run_stamp,
        "frozen_at": utc_now(),
    }
    atomic_json(destination, payload)
    (root / "FROZEN").touch(exist_ok=False)
    print(destination)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "finalize"))
    parser.add_argument("--source", required=True, choices=("routereval", "routerbench"))
    parser.add_argument("--target-family", required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--pool-seed", type=int, default=7001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--leaderboard")
    parser.add_argument("--router-dir")
    parser.add_argument("--routerbench-pickle")
    parser.add_argument("--embedding-archive")
    parser.add_argument("--embedding-manifest")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.action == "train" and (arguments.method is None or arguments.training_seed is None):
        parser.error("train requires --method and --training-seed")
    if arguments.action == "finalize" and arguments.smoke:
        parser.error("a smoke group cannot be finalized")
    targets = ROUTEREVAL_TARGETS if arguments.source == "routereval" else ROUTERBENCH_TARGETS
    if arguments.target_family not in targets:
        parser.error(f"target family must be one of {targets}")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.action == "train":
        train_checkpoint(arguments)
    else:
        finalize_family(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
