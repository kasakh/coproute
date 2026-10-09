"""Registered Stage 1 task adapters for the canonical paired runner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from components.data_protocol import (
    PRIMARY_COVERAGE_POLICY,
    derive_coverage_labels,
    load_routerbench,
)
from components.routerbench_protocol import (
    ROUTERBENCH_MODEL_COLUMNS,
    _ensure_pandas_pickle_compat,
)
from scripts.run_learned_baselines import (
    LearnedTaskData,
    ROUTEREVAL_KEYS,
    audit_features,
    l2_normalize_rows,
    load_aligned_routerbench_embeddings,
    load_routereval_task,
)

REGISTERED_ROUTEREVAL_TASKS = {
    "ifeval": "ifeval",
    "musr": "musr",
    "bbh_boolean_expressions": "bbh",
    "bbh_navigate": "bbh",
    "gpqa_diamond": "gpqa",
    "math_algebra_hard": "math",
    "math_intermediate_algebra_hard": "math",
}
REGISTERED_ROUTERBENCH_TASKS = {
    "mmlu", "hellaswag", "winogrande", "arc-challenge"
}


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def checkpoint_target_family(protocol: str, benchmark: str) -> str | None:
    if protocol == "routereval":
        family = REGISTERED_ROUTEREVAL_TASKS.get(benchmark)
        return family if family in {"bbh", "gpqa", "math"} else None
    if protocol == "routerbench":
        if benchmark not in REGISTERED_ROUTERBENCH_TASKS:
            raise ValueError(f"unregistered RouterBench Stage 1 task {benchmark}")
        return benchmark
    raise ValueError(f"unknown protocol {protocol}")


def load_routereval_registered_task(
    leaderboard: str | Path,
    router_dir: str | Path,
    benchmark: str,
    *,
    threshold: float = 0.5,
) -> LearnedTaskData:
    if benchmark not in REGISTERED_ROUTEREVAL_TASKS:
        raise ValueError(f"unregistered RouterEval Stage 1 task {benchmark}")
    family = REGISTERED_ROUTEREVAL_TASKS[benchmark]
    family_task = load_routereval_task(
        leaderboard,
        Path(router_dir) / f"{family}_router_dataset.pkl",
        family,
        threshold=threshold,
    )
    if benchmark == family:
        return family_task

    with Path(leaderboard).open("rb") as handle:
        payload = pickle.load(handle)
    ordered_keys = ROUTEREVAL_KEYS[family]
    missing = [key for key in ordered_keys if key not in payload["data"]]
    if missing:
        raise ValueError(f"{family}: leaderboard is missing registered keys {missing}")
    lengths = [len(np.asarray(payload["data"][key]["correctness"])) for key in ordered_keys]
    position = ordered_keys.index(benchmark)
    start = int(sum(lengths[:position]))
    stop = start + int(lengths[position])
    direct = (
        np.asarray(payload["data"][benchmark]["correctness"], dtype=float) >= threshold
    ).astype(np.int8)
    sliced = np.asarray(family_task.correctness[start:stop], dtype=np.int8)
    if direct.shape != sliced.shape or not np.array_equal(direct, sliced):
        raise RuntimeError(f"{benchmark}: family alignment slice does not reproduce direct labels")
    prompts = tuple(family_task.prompts[start:stop])
    feature_audit = audit_features(prompts)
    if feature_audit["degenerate"] or feature_audit["exceeds_threshold"]:
        raise ValueError(f"{benchmark}: failed feature audit: {feature_audit}")
    query_ids = tuple(f"routereval:{benchmark}:{index}" for index in range(len(sliced)))
    slice_identity = {
        "family": family,
        "benchmark": benchmark,
        "ordered_keys": list(ordered_keys),
        "lengths": lengths,
        "start": start,
        "stop": stop,
    }
    return LearnedTaskData(
        task=benchmark,
        query_ids=query_ids,
        embeddings=np.asarray(family_task.embeddings[start:stop], dtype=float),
        correctness=sliced,
        model_ids=tuple(family_task.model_ids),
        prompts=prompts,
        feature_audit={
            **feature_audit,
            "threshold": threshold,
            "alignment": "family_original_index_exact_then_registered_slice",
            "family": family,
            "slice_start": start,
            "slice_stop": stop,
            "slice_identity_sha256": _hash_json(slice_identity),
        },
        source_hashes={
            **dict(family_task.source_hashes),
            "registered_slice": _hash_json(slice_identity),
        },
    )


def load_routerbench_registered_task(
    routerbench_pickle: str | Path,
    embedding_archive: str | Path,
    benchmark: str,
    *,
    embedding_manifest: str | Path | None = None,
) -> LearnedTaskData:
    if benchmark not in REGISTERED_ROUTERBENCH_TASKS:
        raise ValueError(f"unregistered RouterBench Stage 1 task {benchmark}")
    _ensure_pandas_pickle_compat()
    data = load_routerbench(
        routerbench_pickle,
        shot="0shot",
        model_columns=ROUTERBENCH_MODEL_COLUMNS,
    )
    labels = derive_coverage_labels(data, policy=PRIMARY_COVERAGE_POLICY)
    embeddings, embedding_audit = load_aligned_routerbench_embeddings(
        embedding_archive,
        data.query_ids,
        manifest_path=embedding_manifest,
    )
    tasks = data.frame["task"].astype(str).to_numpy()
    rows = np.flatnonzero(tasks == benchmark)
    if not len(rows):
        raise ValueError(f"{benchmark}: no audited RouterBench queries")
    prompts = tuple(str(value) for value in data.frame.iloc[rows]["prompt"].tolist())
    feature_audit = audit_features(prompts)
    if feature_audit["degenerate"] or feature_audit["exceeds_threshold"]:
        raise ValueError(f"{benchmark}: failed feature audit: {feature_audit}")
    query_ids = tuple(data.query_ids[int(index)] for index in rows)
    return LearnedTaskData(
        task=benchmark,
        query_ids=query_ids,
        embeddings=l2_normalize_rows(np.asarray(embeddings[rows], dtype=float)),
        correctness=np.asarray(labels.standard[rows], dtype=np.int8),
        model_ids=tuple(data.model_columns),
        prompts=prompts,
        feature_audit={
            **feature_audit,
            "alignment": "exact_query_id_set_and_reorder_then_task_slice",
            "coverage_policy": PRIMARY_COVERAGE_POLICY.name,
            "embedding_audit": embedding_audit,
        },
        source_hashes={
            "routerbench": data.source_sha256,
            "embeddings": embedding_audit["source_sha256"],
            "embedding_manifest": embedding_audit["manifest_sha256"],
        },
    )
