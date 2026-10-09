"""Train target-held-out RouterDC checkpoints and seal family manifests."""
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

from components.data_protocol import (
    PRIMARY_COVERAGE_POLICY,
    SUPPORTED_TASKS,
    assign_racer_splits,
    derive_coverage_labels,
    load_routerbench,
)
from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS
from components.routerdc_baseline import RouterDCBaseline, RouterDCConfig
from scripts.run_learned_baselines import (
    EXTERNAL_SPLIT_SEED,
    TRAINING_SEEDS,
    _hash_json,
    load_aligned_routerbench_embeddings,
    load_routereval_task,
    sha256_file,
    stable_hash_split,
)

SCHEMA_VERSION = 1
PROTOCOL = "issue63_routerdc_target_heldout_v1"
ROUTEREVAL_FAMILIES = ("bbh", "gpqa", "math", "ifeval", "musr")
ROUTERBENCH_TARGETS = ("mmlu", "hellaswag", "winogrande", "arc-challenge")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def hash_training_data(metadata: Mapping[str, Any], arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(canonical_json(metadata))
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(canonical_json(list(array.shape)))
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def deterministic_kmeans(features: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """Deterministic cosine k-means reconstruction without an added dependency."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < n_clusters or not np.all(np.isfinite(values)):
        raise ValueError("k-means features are invalid")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("k-means features contain a zero row")
    values = values / norms
    rng = np.random.default_rng(seed)
    centers = [values[int(rng.integers(len(values)))]]
    for _ in range(1, n_clusters):
        similarity = values @ np.asarray(centers).T
        distance = 1.0 - similarity.max(axis=1)
        centers.append(values[int(np.argmax(distance))])
    centers_array = np.asarray(centers)
    assignments = np.zeros(len(values), dtype=np.int64)
    for _ in range(100):
        updated = np.argmax(values @ centers_array.T, axis=1).astype(np.int64)
        if np.array_equal(updated, assignments):
            break
        assignments = updated
        for cluster in range(n_clusters):
            rows = values[assignments == cluster]
            if len(rows):
                center = rows.mean(axis=0)
                centers_array[cluster] = center / np.linalg.norm(center)
    if len(np.unique(assignments)) != n_clusters:
        raise ValueError("deterministic k-means produced an empty cluster")
    return assignments


def repeat_rows(rows: np.ndarray, count: int) -> np.ndarray:
    if not len(rows):
        raise ValueError("cannot repeat an empty source family")
    repeats = int(np.ceil(count / len(rows)))
    return np.tile(rows, repeats)[:count]


def routereval_bundle(
    leaderboard: Path,
    router_dir: Path,
    target_family: str,
    samples_per_family: int,
) -> dict[str, Any]:
    if target_family not in ROUTEREVAL_FAMILIES:
        raise ValueError(f"unknown RouterEval target family {target_family}")
    training_families = tuple(value for value in ROUTEREVAL_FAMILIES if value != target_family)
    tasks = {
        family: load_routereval_task(
            leaderboard, router_dir / f"{family}_router_dataset.pkl", family
        )
        for family in training_families
    }
    first = tasks[training_families[0]]
    fit_prompts: list[str] = []
    fit_quality: list[np.ndarray] = []
    fit_embeddings: list[np.ndarray] = []
    validation_prompts: list[str] = []
    validation_quality: list[np.ndarray] = []
    split_ids: dict[str, Any] = {}
    for family in training_families:
        task = tasks[family]
        if task.model_ids != first.model_ids:
            raise ValueError("RouterEval model columns differ across source families")
        split = stable_hash_split(
            task.query_ids,
            {"fit": 0.8, "validation": 0.2},
            seed=EXTERNAL_SPLIT_SEED,
            namespace=f"routerdc_target_heldout:{target_family}:{family}",
        )
        fit_rows = repeat_rows(split["fit"], samples_per_family)
        validation_rows = split["validation"]
        fit_prompts.extend(task.prompts[int(row)] for row in fit_rows)
        fit_quality.append(np.asarray(task.correctness[fit_rows], dtype=np.float32))
        fit_embeddings.append(np.asarray(task.embeddings[fit_rows], dtype=np.float64))
        validation_prompts.extend(task.prompts[int(row)] for row in validation_rows)
        validation_quality.append(np.asarray(task.correctness[validation_rows], dtype=np.float32))
        split_ids[family] = {
            "fit": [task.query_ids[int(row)] for row in fit_rows],
            "validation": [task.query_ids[int(row)] for row in validation_rows],
        }
    embeddings = np.concatenate(fit_embeddings)
    quality = np.concatenate(fit_quality)
    validation = np.concatenate(validation_quality)
    clusters = deterministic_kmeans(embeddings, 5, EXTERNAL_SPLIT_SEED)
    identity = {
        "source": "routereval",
        "excluded_family": target_family,
        "training_families": list(training_families),
        "model_ids": list(first.model_ids),
        "split_ids": split_ids,
        "samples_per_family": samples_per_family,
    }
    return {
        "prompts": fit_prompts,
        "quality": quality,
        "cluster_ids": clusters,
        "validation_prompts": validation_prompts,
        "validation_quality": validation,
        "model_ids": tuple(first.model_ids),
        "training_families": training_families,
        "split_ids": split_ids,
        "source_hashes": {family: dict(tasks[family].source_hashes) for family in training_families},
        "feature_audits": {family: dict(tasks[family].feature_audit) for family in training_families},
        "training_data_sha256": hash_training_data(identity, (quality, embeddings, clusters, validation)),
        "cluster_provenance": {
            "method": "deterministic_cosine_kmeans_on_frozen_mdeberta_embeddings",
            "n_clusters": 5,
            "seed": EXTERNAL_SPLIT_SEED,
        },
    }


def routerbench_bundle(
    routerbench_pickle: Path,
    embedding_archive: Path,
    embedding_manifest: Path,
    target_family: str,
    samples_per_family: int,
) -> dict[str, Any]:
    if target_family not in ROUTERBENCH_TARGETS:
        raise ValueError(f"unknown RouterBench target family {target_family}")
    data = load_routerbench(
        routerbench_pickle, shot="0shot", model_columns=ROUTERBENCH_MODEL_COLUMNS
    )
    embeddings, embedding_audit = load_aligned_routerbench_embeddings(
        embedding_archive, data.query_ids, manifest_path=embedding_manifest
    )
    quality = derive_coverage_labels(data, policy=PRIMARY_COVERAGE_POLICY).standard
    assignment = assign_racer_splits(data, seed=EXTERNAL_SPLIT_SEED)
    row_of = {query_id: row for row, query_id in enumerate(data.query_ids)}
    training_families = tuple(value for value in SUPPORTED_TASKS if value != target_family)
    fit_prompts: list[str] = []
    fit_quality: list[np.ndarray] = []
    fit_embeddings: list[np.ndarray] = []
    validation_prompts: list[str] = []
    validation_quality: list[np.ndarray] = []
    split_ids: dict[str, Any] = {}
    for family in training_families:
        fit_ids = list(assignment.ids(family, "train")) + list(assignment.ids(family, "design"))
        validation_ids = list(assignment.ids(family, "validation"))
        fit_rows = repeat_rows(
            np.asarray([row_of[value] for value in fit_ids], dtype=int), samples_per_family
        )
        validation_rows = np.asarray([row_of[value] for value in validation_ids], dtype=int)
        fit_prompts.extend(str(data.frame.iloc[int(row)]["prompt"]) for row in fit_rows)
        fit_quality.append(np.asarray(quality[fit_rows], dtype=np.float32))
        fit_embeddings.append(np.asarray(embeddings[fit_rows], dtype=np.float64))
        validation_prompts.extend(str(data.frame.iloc[int(row)]["prompt"]) for row in validation_rows)
        validation_quality.append(np.asarray(quality[validation_rows], dtype=np.float32))
        split_ids[family] = {
            "fit": [data.query_ids[int(row)] for row in fit_rows],
            "validation": [data.query_ids[int(row)] for row in validation_rows],
        }
    fit_embeddings_array = np.concatenate(fit_embeddings)
    fit_quality_array = np.concatenate(fit_quality)
    validation_array = np.concatenate(validation_quality)
    clusters = deterministic_kmeans(fit_embeddings_array, 5, EXTERNAL_SPLIT_SEED)
    identity = {
        "source": "routerbench",
        "excluded_family": target_family,
        "training_families": list(training_families),
        "model_ids": list(data.model_columns),
        "split_ids": split_ids,
        "samples_per_family": samples_per_family,
    }
    return {
        "prompts": fit_prompts,
        "quality": fit_quality_array,
        "cluster_ids": clusters,
        "validation_prompts": validation_prompts,
        "validation_quality": validation_array,
        "model_ids": tuple(data.model_columns),
        "training_families": training_families,
        "split_ids": split_ids,
        "source_hashes": {
            "routerbench": data.source_sha256,
            "embeddings": embedding_audit["source_sha256"],
            "embedding_manifest": embedding_audit["manifest_sha256"],
        },
        "feature_audits": {
            str(item["task"]): dict(item)
            for item in data.prompt_degeneracy
            if str(item["task"]) in training_families
        },
        "training_data_sha256": hash_training_data(
            identity, (fit_quality_array, fit_embeddings_array, clusters, validation_array)
        ),
        "cluster_provenance": {
            "method": "deterministic_cosine_kmeans_on_frozen_mdeberta_embeddings",
            "n_clusters": 5,
            "seed": EXTERNAL_SPLIT_SEED,
            "embedding_audit": embedding_audit,
        },
    }


def train_one(arguments: argparse.Namespace) -> Path:
    destination = Path(arguments.out_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable checkpoint {destination}")
    if arguments.training_seed not in TRAINING_SEEDS:
        raise ValueError(f"training seed must be one of {TRAINING_SEEDS}")
    samples = 32 if arguments.smoke else 1000
    if arguments.source == "routereval":
        bundle = routereval_bundle(
            Path(arguments.leaderboard), Path(arguments.router_dir), arguments.target_family, samples
        )
    else:
        bundle = routerbench_bundle(
            Path(arguments.routerbench_pickle), Path(arguments.embedding_archive),
            Path(arguments.embedding_manifest), arguments.target_family, samples
        )
    config = RouterDCConfig(
        seed=arguments.training_seed,
        training_steps=(2 if arguments.smoke else 1000),
        evaluation_steps=(1 if arguments.smoke else 50),
        max_length=(64 if arguments.smoke else 512),
    )
    model = RouterDCBaseline(
        arguments.backbone_path, bundle["model_ids"], config=config, device=arguments.device
    )
    model.fit(
        bundle["prompts"], bundle["quality"], bundle["cluster_ids"],
        validation_prompts=bundle["validation_prompts"],
        validation_quality=bundle["validation_quality"],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=destination.parent))
    try:
        checkpoint = temporary / "checkpoint.pt"
        model.save(checkpoint)
        loaded = RouterDCBaseline.load(
            checkpoint, backbone_path=arguments.backbone_path, device=arguments.device
        )
        probe = loaded.predict_proba(bundle["validation_prompts"][:2], model_indices=range(min(11, loaded.n_models)))
        if probe.shape != (2, min(11, loaded.n_models)) or not np.allclose(probe.sum(axis=1), 1.0):
            raise RuntimeError("RouterDC checkpoint round trip failed")
        identity = {
            "method": "routerdc",
            "source": arguments.source,
            "target_family": arguments.target_family,
            "excluded_family": arguments.target_family,
            "training_seed": arguments.training_seed,
            "full_pool_size": len(bundle["model_ids"]),
            "model_ids_sha256": _hash_json(list(bundle["model_ids"])),
            "host": socket.gethostname(),
            "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "run_stamp": arguments.run_stamp,
            "smoke": bool(arguments.smoke),
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "identity": identity,
            "model_pool": {"model_ids": list(bundle["model_ids"]), "model_ids_sha256": identity["model_ids_sha256"]},
            "configuration": model.metadata()["config"],
            "reconstruction": {
                "kind": "clean_room_full_encoder_dual_contrastive",
                "official_code_reused": False,
                "official_repository_license": "unlicensed",
                "cluster_difference": "deterministic cosine k-means replaces paper t-SNE plus k-means",
                "checkpoint_selection": "source_validation_selected_quality",
            },
            "training_families": list(bundle["training_families"]),
            "target_labels_used": False,
            "training_data_sha256": bundle["training_data_sha256"],
            "split_ids": bundle["split_ids"],
            "cluster_provenance": bundle["cluster_provenance"],
            "source_hashes": bundle["source_hashes"],
            "feature_audits": bundle["feature_audits"],
            "training_history": model.training_history,
            "best_step": model.best_step,
            "best_validation_quality": model.best_validation_quality,
            "round_trip": {"passed": True, "shape": list(probe.shape)},
            "environment": {
                "created_at": utc_now(), "python": platform.python_version(),
                "numpy": np.__version__, "torch": torch.__version__,
                "device": arguments.device, "host": socket.gethostname(),
                "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"), "run_stamp": arguments.run_stamp,
            },
        }
        atomic_json(temporary / "training_metadata.json", metadata)
        atomic_json(temporary / "training_history.json", model.training_history)
        files = {
            name: sha256_file(temporary / name)
            for name in ("checkpoint.pt", "training_metadata.json", "training_history.json")
        }
        manifest = {
            "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "immutable": True,
            "identity": identity, "checkpoint_sha256": files["checkpoint.pt"], "files": files,
            "training_data_sha256": bundle["training_data_sha256"],
            "host": socket.gethostname(), "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "training_seed": arguments.training_seed, "run_stamp": arguments.run_stamp,
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
    records = []
    training_hashes = set()
    for seed in TRAINING_SEEDS:
        directory = root / f"seed{seed}"
        manifest_path = directory / "manifest.json"
        if not (directory / "DONE").is_file() or not manifest_path.is_file():
            raise ValueError(f"incomplete RouterDC checkpoint {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest["identity"]
        if identity["training_seed"] != seed or identity["target_family"] != arguments.target_family:
            raise ValueError(f"checkpoint identity mismatch in {manifest_path}")
        for name, expected in manifest["files"].items():
            if sha256_file(directory / name) != expected:
                raise ValueError(f"checkpoint hash mismatch in {directory / name}")
        training_hashes.add(manifest["training_data_sha256"])
        records.append({
            "training_seed": seed, "directory": str(directory),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "manifest_sha256": sha256_file(manifest_path), "gpu_index": identity["gpu_index"],
        })
    if len(training_hashes) != 1:
        raise ValueError("RouterDC family used differing training data")
    payload = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "immutable": True,
        "source": arguments.source, "target_family": arguments.target_family,
        "excluded_family": arguments.target_family, "training_seeds": list(TRAINING_SEEDS),
        "checkpoint_count": len(records), "training_data_sha256": next(iter(training_hashes)),
        "checkpoints": records, "host": socket.gethostname(), "run_stamp": arguments.run_stamp,
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
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--backbone-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--leaderboard")
    parser.add_argument("--router-dir")
    parser.add_argument("--routerbench-pickle")
    parser.add_argument("--embedding-archive")
    parser.add_argument("--embedding-manifest")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "train":
        required = ["training_seed", "backbone_path"]
        if args.source == "routereval":
            required += ["leaderboard", "router_dir"]
        else:
            required += ["routerbench_pickle", "embedding_archive", "embedding_manifest"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error("train is missing " + ", ".join(missing))
    targets = ROUTEREVAL_FAMILIES if args.source == "routereval" else ROUTERBENCH_TARGETS
    if args.target_family not in targets:
        parser.error(f"target family must be one of {targets}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.action == "train":
        train_one(arguments)
    else:
        finalize_family(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
