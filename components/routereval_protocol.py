"""RouterEval loader for the E1 scale grid, with a mandatory feature audit.

RouterEval (arXiv:2503.10657, HuggingFace linggm/RouterEval) supplies per-model
correctness over a pool of 3,811 models across 38 benchmarks, plus per-benchmark
prompts, 768-dimensional embeddings and train/val/test index arrays.

Two sources are combined:
  leaderboard_score/leaderboard_new.pkl   {"model": array[3811], "data": {bench: {"correctness": array[n, 3811]}}}
  router_dataset/<bench>_router_dataset.pkl   {"split_index", "embedding", "prompt", "easy", "hard"}

The loader REFUSES a benchmark whose feature column collapses. That guard exists
because on 2026-09-03 RouterBench's GSM8K was found to carry 7,450 rows with one
distinct prompt, which silently voided every routing number computed on it. The
audit is not optional and not a warning.

Pools are fixed-seed subsamples of the 3,811 model columns. The prepackaged
easy[k]/hard[k] scenarios are deliberately NOT used for the primary grid because
they bundle a difficulty-selection choice that would confound pool size with pool
quality; they are available separately as a labelled secondary comparison.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

MAX_DUPLICATE_RATE = 0.10


class RouterEvalDegenerateFeatures(RuntimeError):
    """Raised when a benchmark's prompts cannot distinguish its queries."""


@dataclass(frozen=True)
class RouterEvalBenchmark:
    """One audited RouterEval benchmark."""

    name: str
    correctness: np.ndarray
    model_names: tuple[str, ...]
    prompts: tuple[str, ...]
    embeddings: np.ndarray
    split_indices: Mapping[str, np.ndarray]
    feature_audit: Mapping[str, Any]
    source_sha256: str

    @property
    def n_queries(self) -> int:
        return int(self.correctness.shape[0])

    @property
    def n_models(self) -> int:
        return int(self.correctness.shape[1])

    def complete_pool_all_miss_rate(self, threshold: float = 0.5) -> float:
        """Fraction of queries no model in the FULL pool answers correctly."""

        return float(np.mean(np.all(self.correctness < threshold, axis=1)))


def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def audit_features(prompts: Sequence[str]) -> dict:
    """Distinct-prompt census. A collapsed feature column is fatal."""

    values = [str(value) for value in prompts]
    total = len(values)
    unique = len(set(values))
    duplicate_rate = float(1.0 - unique / total) if total else 0.0
    return {
        "rows": total,
        "unique_prompts": unique,
        "duplicate_rate": duplicate_rate,
        "degenerate": bool(unique <= 1),
        "exceeds_threshold": bool(duplicate_rate > MAX_DUPLICATE_RATE),
    }


def load_benchmark(
    *,
    leaderboard_path: str | Path,
    router_dataset_path: str | Path,
    benchmark: str,
    leaderboard_key: str | None = None,
) -> RouterEvalBenchmark:
    leaderboard_path = Path(leaderboard_path)
    router_dataset_path = Path(router_dataset_path)

    with leaderboard_path.open("rb") as handle:
        leaderboard = pickle.load(handle)
    with router_dataset_path.open("rb") as handle:
        router_dataset = pickle.load(handle)

    key = leaderboard_key or benchmark
    if key not in leaderboard["data"]:
        available = sorted(leaderboard["data"])
        raise KeyError(f"{key} not in leaderboard; available: {available[:8]} ...")

    correctness = np.asarray(leaderboard["data"][key]["correctness"], dtype=float)
    if correctness.ndim != 2 or correctness.size == 0:
        raise ValueError(f"{key}: correctness must be a non-empty 2D array")
    if not np.all(np.isfinite(correctness)):
        raise ValueError(f"{key}: correctness contains non-finite values")

    model_names = tuple(str(name) for name in np.asarray(leaderboard["model"]).ravel())
    if len(model_names) != correctness.shape[1]:
        raise ValueError(
            f"{key}: {len(model_names)} model names for {correctness.shape[1]} columns"
        )

    prompt_block = router_dataset["prompt"]
    embed_block = router_dataset["embedding"]
    split_block = router_dataset["split_index"]
    order = ("train", "val", "test")
    prompts: list[str] = []
    embeddings: list[np.ndarray] = []
    offsets: dict[str, np.ndarray] = {}
    cursor = 0
    for split in order:
        split_prompts = [str(value) for value in prompt_block[f"{split}_prompt"]]
        split_embed = np.asarray(embed_block[f"{split}_embed"], dtype=float)
        if split_embed.shape[0] != len(split_prompts):
            raise ValueError(f"{key}/{split}: embedding rows do not match prompts")
        prompts.extend(split_prompts)
        embeddings.append(split_embed)
        offsets[split] = np.arange(cursor, cursor + len(split_prompts), dtype=int)
        cursor += len(split_prompts)

    embedding_matrix = np.concatenate(embeddings, axis=0)
    if not np.all(np.isfinite(embedding_matrix)):
        raise ValueError(f"{key}: embeddings contain non-finite values")

    audit = audit_features(prompts)
    if audit["degenerate"]:
        raise RouterEvalDegenerateFeatures(
            f"{key}: prompts collapse to {audit['unique_prompts']} distinct value(s) "
            f"over {audit['rows']} rows; a router has no query signal so every "
            f"routing number on this benchmark would be void"
        )
    if audit["exceeds_threshold"]:
        raise RouterEvalDegenerateFeatures(
            f"{key}: duplicate prompt rate {audit['duplicate_rate']:.4f} exceeds "
            f"the {MAX_DUPLICATE_RATE:.2f} threshold"
        )

    seen: set[int] = set()
    for split, indices in offsets.items():
        overlap = seen.intersection(indices.tolist())
        if overlap:
            raise ValueError(f"{key}/{split}: split indices overlap")
        seen.update(indices.tolist())
    if len(seen) != cursor:
        raise ValueError(f"{key}: split indices do not cover every row exactly once")

    if cursor != correctness.shape[0]:
        raise ValueError(
            f"{key}: {cursor} feature rows but {correctness.shape[0]} correctness rows. "
            "The leaderboard key and the router_dataset file describe different query sets."
        )

    original_splits = {
        split: np.asarray(split_block[f"{split}_indices"], dtype=int) for split in order
    }

    return RouterEvalBenchmark(
        name=str(benchmark),
        correctness=correctness,
        model_names=model_names,
        prompts=tuple(prompts),
        embeddings=embedding_matrix,
        split_indices={**offsets, **{f"original_{k}": v for k, v in original_splits.items()}},
        feature_audit={**audit, "leaderboard_key": key},
        source_sha256=_sha256_of_file(leaderboard_path),
    )


def load_correctness_only(
    *,
    leaderboard_path: str | Path,
    leaderboard_key: str,
) -> RouterEvalBenchmark:
    """Load one leaderboard benchmark's correctness matrix WITHOUT features.

    E1's arms are estimator families over the joint failure structure: marginal
    ordering, pairwise co-failure counting, shrinkage, a lineage heuristic, a
    low-rank factor model, a Gaussian copula and an oracle. Every one of those is
    computed from design LABELS, not from query features, so the primary grid
    needs no router and no embeddings.

    That matters because `router_dataset/<bench>.pkl` AGGREGATES sub-tasks: its
    bbh file has 5,761 rows against the 24 individual `bbh_*` leaderboard keys
    which sum to exactly 5,761, and likewise gpqa 1,192 over 3 keys, math 1,324
    over 7 and musr 756 over 3. Concatenating those matrices to match the feature
    rows would require assuming a sub-task ORDER that nothing in the release
    states. Rather than assume it, E1 avoids features entirely and the alignment
    is left as a precondition for E2, which does need a per-query router.

    The feature audit is recorded as not-applicable rather than passed, so no
    downstream reader can mistake this for an audited feature column.
    """

    leaderboard_path = Path(leaderboard_path)
    with leaderboard_path.open("rb") as handle:
        leaderboard = pickle.load(handle)
    if leaderboard_key not in leaderboard["data"]:
        available = sorted(leaderboard["data"])
        raise KeyError(
            f"{leaderboard_key} not in leaderboard; {len(available)} available, "
            f"first few: {available[:6]}"
        )
    correctness = np.asarray(
        leaderboard["data"][leaderboard_key]["correctness"], dtype=float
    )
    if correctness.ndim != 2 or correctness.size == 0:
        raise ValueError(f"{leaderboard_key}: correctness must be a non-empty 2D array")
    if not np.all(np.isfinite(correctness)):
        raise ValueError(f"{leaderboard_key}: correctness contains non-finite values")
    model_names = tuple(str(name) for name in np.asarray(leaderboard["model"]).ravel())
    if len(model_names) != correctness.shape[1]:
        raise ValueError(
            f"{leaderboard_key}: {len(model_names)} model names for "
            f"{correctness.shape[1]} columns"
        )
    return RouterEvalBenchmark(
        name=str(leaderboard_key),
        correctness=correctness,
        model_names=model_names,
        prompts=(),
        embeddings=np.zeros((correctness.shape[0], 0), dtype=float),
        split_indices={},
        feature_audit={
            "leaderboard_key": leaderboard_key,
            "mode": "correctness_only",
            "features_present": False,
            "degenerate": None,
            "duplicate_rate": None,
            "note": (
                "labels only; no feature column was loaded so no feature audit "
                "applies. Any per-query router use requires the aligned loader."
            ),
        },
        source_sha256=_sha256_of_file(leaderboard_path),
    )


def subsample_pool(n_models: int, pool_size: int, *, seed: int) -> np.ndarray:
    """Fixed-seed model-column subsample. Bit-reproducible for a given seed."""

    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if pool_size > n_models:
        raise ValueError(f"pool_size {pool_size} exceeds available models {n_models}")
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(n_models, size=pool_size, replace=False))


def headroom_census(
    benchmark: RouterEvalBenchmark,
    pool_sizes: Sequence[int],
    *,
    seed: int,
    threshold: float = 0.5,
) -> dict:
    """All-miss rate of the complete pool at each pool size.

    This is the E1-P1 abort gate. If the complete pool essentially never fails,
    there is headroom for small-subset selection at any target level. If it fails
    more often than the target, no estimator can create headroom and the cell is
    structurally infeasible, exactly as RouterBench pool 3 was.
    """

    records = []
    for pool_size in pool_sizes:
        columns = subsample_pool(benchmark.n_models, pool_size, seed=seed)
        sub = benchmark.correctness[:, columns]
        records.append(
            {
                "pool_size": int(pool_size),
                "seed": int(seed),
                "complete_pool_all_miss_rate": float(
                    np.mean(np.all(sub < threshold, axis=1))
                ),
                "best_single_model_accuracy": float((sub >= threshold).mean(axis=0).max()),
                "mean_model_accuracy": float((sub >= threshold).mean()),
            }
        )
    return {
        "benchmark": benchmark.name,
        "n_queries": benchmark.n_queries,
        "n_models_available": benchmark.n_models,
        "threshold": threshold,
        "feature_audit": dict(benchmark.feature_audit),
        "records": records,
    }


__all__ = [
    "MAX_DUPLICATE_RATE",
    "RouterEvalBenchmark",
    "RouterEvalDegenerateFeatures",
    "audit_features",
    "headroom_census",
    "load_benchmark",
    "load_correctness_only",
    "subsample_pool",
]
