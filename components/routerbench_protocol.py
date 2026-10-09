"""RouterBench correctness loader mirroring the RouterEval interface.

RouterBench (arXiv:2403.12031) supplies per-prompt correctness scores for 11
proprietary and open LLMs across ~86 sub-benchmarks, plus per-prompt cost. The
release ships as a pandas DataFrame keyed by `sample_id` with `prompt`,
`eval_name`, one correctness score column per model, plus `<model>|model_response`
and `<model>|total_cost` columns.

For the RACER-style routing analyses only the correctness matrix is needed. The
loader signature deliberately mirrors `routereval_protocol.load_correctness_only`
so `scripts/frontier_curves.py`, `scripts/frontier_generalization.py` and the
upcoming `scripts/frontier_fourway.py` can consume RouterBench through a single
`--protocol` switch.

Benchmark aggregation. RouterBench's `eval_name` column carries ~86 fine-grained
labels; the 6 task families named in
`PLAN_ROUTERBENCH_AND_VALIDATION_20260903.md` are formed by aggregating over
substrings. For MMLU, all 57 `mmlu-*` sub-tasks are pooled into one benchmark.
The mapping is fixed at load time and recorded in the feature audit.

Feature guard. On 2026-09-03 a truncated-prompt bug in a prior loader made all
7,450 RouterBench GSM8K rows appear to share one prompt, silently voiding every
routing number computed on it. This loader recomputes the distinct-prompt count
from the raw `prompt` column and fails if the duplicate rate exceeds
MAX_DUPLICATE_RATE.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MAX_DUPLICATE_RATE = 0.10

ROUTERBENCH_MODEL_COLUMNS: tuple[str, ...] = (
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
)

# Six task families the plan targets. Each entry names the plan label together
# with the predicate that selects RouterBench eval_names for the family.
BENCHMARK_FAMILIES: dict[str, dict[str, Any]] = {
    "mbpp": {"kind": "exact", "keys": ("mbpp",)},
    "gsm8k": {"kind": "exact", "keys": ("grade-school-math",)},
    "winogrande": {"kind": "exact", "keys": ("winogrande",)},
    "hellaswag": {"kind": "exact", "keys": ("hellaswag",)},
    "arc-challenge": {"kind": "exact", "keys": ("arc-challenge",)},
    "mmlu": {"kind": "prefix", "keys": ("mmlu-",)},
}


class RouterBenchDegenerateFeatures(RuntimeError):
    """Raised when a benchmark's prompts cannot distinguish its queries."""


@dataclass(frozen=True)
class RouterBenchBenchmark:
    """One audited RouterBench benchmark. Structurally compatible with
    `RouterEvalBenchmark` for the fields the frontier drivers consume."""

    name: str
    correctness: np.ndarray
    model_names: tuple[str, ...]
    prompts: tuple[str, ...]
    embeddings: np.ndarray
    split_indices: Mapping[str, np.ndarray]
    feature_audit: Mapping[str, Any]
    source_sha256: str


def _ensure_pandas_pickle_compat() -> None:
    """Provide pandas' moved block unpickler when reading a newer release file."""

    try:
        import pandas._libs.internals as internals
        import pandas.core.internals.blocks as blocks
    except ImportError:
        return
    if hasattr(internals, "_unpickle_block"):
        return
    factory = getattr(blocks, "new_block", None) or getattr(blocks, "make_block", None)
    if factory is None:
        return
    placement_type = getattr(internals, "BlockPlacement", None)

    def _unpickle_block(values, placement, ndim):
        if placement_type is not None and not isinstance(placement, placement_type):
            placement = placement_type(placement)
        return factory(values, placement=placement, ndim=ndim)

    internals._unpickle_block = _unpickle_block


def _sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _select_rows(frame, family_key: str):
    if family_key not in BENCHMARK_FAMILIES:
        available = sorted(BENCHMARK_FAMILIES) + ["<raw eval_name>"]
        raise KeyError(f"unknown family {family_key!r}; known families: {available}")
    family = BENCHMARK_FAMILIES[family_key]
    kind = family["kind"]
    keys = family["keys"]
    if kind == "exact":
        selector = frame["eval_name"].isin(list(keys))
    elif kind == "prefix":
        selector = frame["eval_name"].str.startswith(keys[0])
    else:
        raise ValueError(f"unknown family kind {kind!r}")
    return frame[selector]


def load_correctness_only(
    *,
    pickle_path: str | Path,
    benchmark: str,
    threshold: float = 0.5,
) -> RouterBenchBenchmark:
    """Load one RouterBench benchmark family's correctness matrix WITHOUT costs.

    Parameters
    ----------
    pickle_path: RouterBench 0-shot or 5-shot pickle.
    benchmark: one of the six family keys in ``BENCHMARK_FAMILIES``.
    threshold: correctness threshold; RouterBench scores are already 0 or 1 for
        most tasks but a few benchmarks (mbpp pass rate) may be graded values.
        The frontier drivers work on a binary correctness matrix, so the
        threshold is applied at load time to match `routereval_protocol`'s
        contract.

    Returns
    -------
    A `RouterBenchBenchmark` whose `correctness` field is shape (n_queries, 11)
    binary, `model_names` are the 11 RouterBench models in the fixed order of
    `ROUTERBENCH_MODEL_COLUMNS`, and `feature_audit` records the distinct-prompt
    duplication rate plus the raw eval_name breakdown that composed the family.
    """
    pickle_path = Path(pickle_path)
    _ensure_pandas_pickle_compat()
    with pickle_path.open("rb") as handle:
        frame = pickle.load(handle)
    required = ("prompt", "eval_name") + ROUTERBENCH_MODEL_COLUMNS
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(
            f"RouterBench pickle at {pickle_path} missing required columns: {missing}"
        )
    sub = _select_rows(frame, benchmark)
    if len(sub) == 0:
        raise KeyError(f"benchmark {benchmark!r} yielded zero rows in {pickle_path}")

    # Feature audit
    n_rows = len(sub)
    distinct = int(sub["prompt"].nunique())
    duplicate_rate = 1.0 - (distinct / n_rows) if n_rows else 0.0
    if duplicate_rate > MAX_DUPLICATE_RATE:
        raise RouterBenchDegenerateFeatures(
            f"{benchmark}: prompt duplicate rate {duplicate_rate:.4f} exceeds "
            f"{MAX_DUPLICATE_RATE:.4f}; loader refuses to proceed silently"
        )

    raw_correctness = np.asarray(sub[list(ROUTERBENCH_MODEL_COLUMNS)].values, dtype=float)
    if raw_correctness.ndim != 2 or raw_correctness.size == 0:
        raise ValueError(f"{benchmark}: correctness must be non-empty 2D")
    # RouterBench sometimes carries NaN for a model whose response is missing.
    # Treat NaN as failed rather than silently omitting rows: this is a per-model
    # coverage floor, consistent with how RACER treats an abstaining model.
    if not np.all(np.isfinite(raw_correctness)):
        raw_correctness = np.where(np.isfinite(raw_correctness), raw_correctness, 0.0)
    binary = (raw_correctness >= threshold).astype(int)

    # Compose the eval_name breakdown for the audit
    breakdown = (
        sub["eval_name"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    audit: dict[str, Any] = {
        "protocol": "routerbench",
        "family_key": benchmark,
        "n_rows": n_rows,
        "n_distinct_prompts": distinct,
        "duplicate_rate": duplicate_rate,
        "eval_name_breakdown": {str(k): int(v) for k, v in breakdown.items()},
        "threshold": threshold,
        "features_present": True,
        "note": (
            "labels-only load; correctness is threshold-binarized. Prompt column "
            "carried through in `prompts` for optional per-query audits but the "
            "frontier drivers only need the correctness matrix."
        ),
    }

    return RouterBenchBenchmark(
        name=str(benchmark),
        correctness=binary,
        model_names=ROUTERBENCH_MODEL_COLUMNS,
        prompts=tuple(str(p) for p in sub["prompt"].tolist()),
        embeddings=np.zeros((binary.shape[0], 0), dtype=float),
        split_indices={},
        feature_audit=audit,
        source_sha256=_sha256_of_file(pickle_path),
    )


def subsample_pool(n_models: int, pool_size: int, *, seed: int) -> np.ndarray:
    """Deterministic subsample of model columns; RouterBench's pool is 11 so a
    request for pool_size == 11 returns the identity permutation and any request
    for a larger pool errors."""
    if pool_size > n_models:
        raise ValueError(
            f"pool_size {pool_size} exceeds RouterBench model count {n_models}"
        )
    if pool_size == n_models:
        return np.arange(n_models, dtype=int)
    generator = np.random.default_rng(seed)
    return generator.choice(n_models, size=pool_size, replace=False).astype(int)
