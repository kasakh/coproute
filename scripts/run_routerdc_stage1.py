"""Run one RouterDC-seeded repaired B-prime Stage 1 shard."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from components.routereval_protocol import subsample_pool
from scripts.run_learned_baselines import sha256_file
from scripts.run_paired_main_suite import (
    ALPHA_GRID,
    BASE_SEED,
    _load_task,
    atomic_write_json,
    hash_json,
    run_paired_suite,
)


def load_scores(
    directory: Path,
    task: Any,
    pool_size: int,
    pool_seed: int,
    expected_target_family: str,
    expected_training_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    metadata_path = directory / "metadata.json"
    archive_path = directory / "scores.npz"
    if not (directory / "DONE").is_file():
        raise ValueError(f"RouterDC score directory is incomplete: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"RouterDC score hash mismatch for {directory / name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    identity = metadata["identity"]
    if identity["target_family"] != expected_target_family:
        raise ValueError("RouterDC score target family mismatch")
    if identity["training_seed"] != expected_training_seed:
        raise ValueError("RouterDC score training seed mismatch")
    with np.load(archive_path, allow_pickle=False) as archive:
        logits = np.asarray(archive["logits"], dtype=np.float64)
        query_ids = [str(value) for value in archive["query_ids"].tolist()]
        model_ids = [str(value) for value in archive["model_ids"].tolist()]
    if query_ids != list(task.query_ids) or model_ids != list(task.model_ids):
        raise ValueError("RouterDC score IDs differ from the registered task")
    if hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest() != metadata["logits_sha256"]:
        raise ValueError("RouterDC logits content hash mismatch")
    columns = subsample_pool(len(model_ids), pool_size, seed=pool_seed)
    selected = logits[:, columns]
    shifted = selected - selected.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if probabilities.shape != (len(query_ids), pool_size):
        raise ValueError("RouterDC sampled score shape is invalid")
    provenance = {
        "score_directory": str(directory),
        "score_manifest_sha256": sha256_file(manifest_path),
        "score_metadata_sha256": sha256_file(metadata_path),
        "full_logits_sha256": metadata["logits_sha256"],
        "sampled_probability_sha256": hashlib.sha256(
            np.ascontiguousarray(probabilities).tobytes()
        ).hexdigest(),
        "checkpoint_identity": metadata["checkpoint_identity"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "model_ids_sha256": hash_json([model_ids[int(index)] for index in columns]),
        "query_ids_sha256": hash_json(query_ids),
    }
    return probabilities, provenance


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    task = _load_task(
        arguments.protocol,
        arguments.benchmark,
        leaderboard=arguments.leaderboard,
        router_dir=arguments.router_dir,
        routerbench_pickle=arguments.routerbench_pickle,
        embedding_archive=arguments.embedding_archive,
        embedding_manifest=arguments.embedding_manifest,
    )
    probabilities, provenance = load_scores(
        arguments.routerdc_score_dir,
        task,
        arguments.pool_size,
        arguments.pool_seed,
        arguments.target_family,
        arguments.training_seed,
    )
    call_hashes: list[str] = []

    def frozen_routerdc_scores(*unused: Any, **unused_kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        value = probabilities.copy()
        call_hashes.append(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        return value, np.zeros((len(value), 0), dtype=int)

    payload = run_paired_suite(
        task,
        pool_size=arguments.pool_size,
        pool_seed=arguments.pool_seed,
        redraw_start=arguments.redraw_start,
        redraw_end=arguments.redraw_end,
        knn_k=40,
        base_seed=BASE_SEED,
        alphas=ALPHA_GRID,
        include_chain_discount=True,
        learned_ensembles={},
        knn_function=frozen_routerdc_scores,
        protocol=arguments.protocol,
    )
    if len(call_hashes) != arguments.redraw_end - arguments.redraw_start:
        raise RuntimeError("RouterDC score function call count differs from redraw count")
    if len(set(call_hashes)) != 1 or call_hashes[0] != provenance["sampled_probability_sha256"]:
        raise RuntimeError("RouterDC score matrix changed across redraws")
    required = {
        "racer_g", "racer_p", "exact_independence",
        "fixed_design_marginal_factorized", "fixed_tally_factorized",
        "fixed_tally_chain_discounted",
    }
    for cell in payload["cells"]:
        for redraw in cell["redraws"]:
            if set(redraw["methods"]) != required:
                raise RuntimeError("RouterDC Stage 1 arm presence check failed")
    payload.update({
        "design": "ISSUE_63_RouterDC_seeded_Stage_1",
        "base_router": "routerdc",
        "routerdc_training_seed": arguments.training_seed,
        "routerdc_target_family": arguments.target_family,
        "routerdc_score_provenance": provenance,
        "routerdc_same_score_matrix_for_all_methods": True,
        "routerdc_score_call_count": len(call_hashes),
        "routerdc_score_hashes": call_hashes,
        "method_identities": sorted(required),
    })
    atomic_write_json(arguments.out, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, choices=("routereval", "routerbench"))
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--target-family", required=True)
    parser.add_argument("--pool-size", required=True, type=int)
    parser.add_argument("--pool-seed", default=7001, type=int)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--routerdc-score-dir", required=True, type=Path)
    parser.add_argument("--redraw-start", required=True, type=int)
    parser.add_argument("--redraw-end", required=True, type=int)
    parser.add_argument("--leaderboard", type=Path)
    parser.add_argument("--router-dir", type=Path)
    parser.add_argument("--routerbench-pickle", type=Path)
    parser.add_argument("--embedding-archive", type=Path)
    parser.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
