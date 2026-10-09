"""Auditable RouterBench loading and RACER compatible split protocol.

This module separates five operations that legacy experiments mixed together:

1. preserve RouterBench's official numerical correctness scores;
2. account for every source row with an inclusion or exclusion reason;
3. merge only conflict-free duplicate fragments into stable query records;
4. derive binary model coverage under an explicitly named task policy; and
5. assign deterministic train, design, calibration, validation, and test sets.

RACER reserves 50 percent of GSM8K and 40 percent of other datasets for
router development. The remainder is split 50/10/40 into calibration,
validation, and test, except ARC, which uses 40/20/40. DACR needs a design
set independent of calibration. By default this module divides RACER's router
development budget equally into train and design. Thus train plus design still
equals RACER's published router-training fraction, and the evaluation pool
keeps RACER's published ratios.

Loading never creates binary labels. Call :func:`derive_coverage_labels` with a
named :class:`CoveragePolicy` to make that scientific choice explicit.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
PROTOCOL_NAME = "routerbench-racer-corrected-v1"
NULL_MODEL_NAME = "__racer_null__"
SPLIT_NAMES = ("train", "design", "calibration", "validation", "test")
REQUIRED_METADATA = ("sample_id", "prompt", "eval_name")
NON_MODEL_COLUMNS = {
    "sample_id",
    "prompt",
    "eval_name",
    "oracle_model_to_route_to",
    "query_id",
    "task",
    "shot",
    "source_rows",
    "split",
}

_TASK_ALIASES = {
    "grade-school-math": "gsm8k",
    "gsm8k": "gsm8k",
    "gsm-8k": "gsm8k",
    "arc": "arc-challenge",
    "arc_challenge": "arc-challenge",
    "arc-challenge": "arc-challenge",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "mbpp": "mbpp",
}
SUPPORTED_TASKS = (
    "mmlu",
    "hellaswag",
    "gsm8k",
    "arc-challenge",
    "winogrande",
    "mbpp",
)


@dataclass(frozen=True)
class RouterBenchData:
    """One audited RouterBench release.

    ``frame`` contains one row per included query and retains the official
    numerical score in every model column. ``row_audit`` contains one row per
    raw input row, including rows merged into a query and excluded rows.
    """

    frame: pd.DataFrame
    row_audit: pd.DataFrame
    model_columns: Tuple[str, ...]
    source_name: str
    source_sha256: str
    shot: str
    prompt_degeneracy: Tuple[dict, ...] = ()

    @property
    def query_ids(self) -> Tuple[str, ...]:
        return tuple(self.frame["query_id"].astype(str))

    @property
    def degenerate_tasks(self) -> Tuple[str, ...]:
        """Tasks whose prompt column collapses to a single distinct value.

        A router cannot distinguish queries inside such a task, so every routing
        result computed on it is void. Discovered 2026-09-03: RouterBench's
        `grade-school-math` has 7,450 rows and exactly ONE distinct prompt, and
        eleven further families are equally degenerate.
        """

        return tuple(
            str(record["task"])
            for record in self.prompt_degeneracy
            if int(record["unique_prompts"]) <= 1
        )

    @property
    def tasks_with_duplicate_prompts(self) -> Tuple[str, ...]:
        return tuple(
            str(record["task"])
            for record in self.prompt_degeneracy
            if float(record["duplicate_rate"]) > 0.0
        )

    @property
    def raw_row_count(self) -> int:
        return int(len(self.row_audit))

    @property
    def included_query_count(self) -> int:
        return int(len(self.frame))


@dataclass(frozen=True)
class CoveragePolicy:
    """Explicit task-specific conversion from official score to coverage.

    A comparator is either ``ge`` or ``gt``. Binary-only tasks reject any
    official score outside exactly zero or one rather than silently applying a
    threshold intended for a different task.
    """

    name: str
    task_thresholds: Mapping[str, float]
    task_comparators: Mapping[str, str]
    binary_only_tasks: Tuple[str, ...]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_thresholds": dict(sorted(self.task_thresholds.items())),
            "task_comparators": dict(sorted(self.task_comparators.items())),
            "binary_only_tasks": list(self.binary_only_tasks),
            "description": self.description,
        }


@dataclass(frozen=True)
class CoverageLabels:
    """Binary standard-model labels and RACER augmented null labels."""

    query_ids: Tuple[str, ...]
    model_columns: Tuple[str, ...]
    standard: np.ndarray
    augmented: np.ndarray
    policy: CoveragePolicy


PRIMARY_COVERAGE_POLICY = CoveragePolicy(
    name="task_primary_gsm8k_ge_0_5",
    task_thresholds={task: (0.5 if task == "gsm8k" else 1.0)
                     for task in SUPPORTED_TASKS},
    task_comparators={task: "ge" for task in SUPPORTED_TASKS},
    binary_only_tasks=tuple(task for task in SUPPORTED_TASKS if task != "gsm8k"),
    description=(
        "Multiple-choice and MBPP cells must be official binary scores. "
        "GSM8K coverage is score at least 0.5."
    ),
)

GSM8K_STRICT_COVERAGE_POLICY = CoveragePolicy(
    name="sensitivity_gsm8k_eq_1",
    task_thresholds={task: 1.0 for task in SUPPORTED_TASKS},
    task_comparators={task: "ge" for task in SUPPORTED_TASKS},
    binary_only_tasks=tuple(task for task in SUPPORTED_TASKS if task != "gsm8k"),
    description=(
        "Sensitivity variant. GSM8K counts coverage only for an official "
        "score of one; other supported tasks remain official-binary."
    ),
)

GSM8K_GE_0_75_COVERAGE_POLICY = CoveragePolicy(
    name="sensitivity_gsm8k_ge_0_75",
    task_thresholds={task: (0.75 if task == "gsm8k" else 1.0)
                     for task in SUPPORTED_TASKS},
    task_comparators={task: "ge" for task in SUPPORTED_TASKS},
    binary_only_tasks=tuple(task for task in SUPPORTED_TASKS if task != "gsm8k"),
    description=(
        "Sensitivity variant. GSM8K coverage requires an official score of "
        "at least 0.75; other supported tasks remain official-binary."
    ),
)

GSM8K_POSITIVE_COVERAGE_POLICY = CoveragePolicy(
    name="sensitivity_gsm8k_gt_0",
    task_thresholds={task: (0.0 if task == "gsm8k" else 1.0)
                     for task in SUPPORTED_TASKS},
    task_comparators={task: ("gt" if task == "gsm8k" else "ge")
                      for task in SUPPORTED_TASKS},
    binary_only_tasks=tuple(task for task in SUPPORTED_TASKS if task != "gsm8k"),
    description=(
        "Sensitivity variant. Any positive GSM8K official score counts as "
        "coverage; other supported tasks remain official-binary."
    ),
)

COVERAGE_POLICIES: Mapping[str, CoveragePolicy] = {
    policy.name: policy
    for policy in (
        PRIMARY_COVERAGE_POLICY,
        GSM8K_STRICT_COVERAGE_POLICY,
        GSM8K_GE_0_75_COVERAGE_POLICY,
        GSM8K_POSITIVE_COVERAGE_POLICY,
    )
}


@dataclass(frozen=True)
class SplitAssignment:
    """Per-query deterministic split assignments and fractions."""

    assignments: pd.DataFrame
    seed: int
    design_share_of_racer_train: float
    fractions_by_task: Mapping[str, Mapping[str, float]]

    def ids(self, task: str, split: str) -> Tuple[str, ...]:
        rows = self.assignments
        selected = rows[(rows["task"] == task) & (rows["split"] == split)]
        return tuple(selected["query_id"].astype(str))


Source = Union[str, Path, pd.DataFrame]
Policy = Union[str, CoveragePolicy]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def file_sha256(path: Union[str, Path], *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it fully into memory."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_task(eval_name: Any) -> Optional[str]:
    """Map a RouterBench evaluation name to a supported paper task."""

    if not isinstance(eval_name, str) or not eval_name.strip():
        return None
    normalized = eval_name.strip().lower()
    if normalized.startswith("mmlu"):
        return "mmlu"
    return _TASK_ALIASES.get(normalized)


def stable_query_id(eval_name: Any, sample_id: Any) -> str:
    """Return a stable shot-independent query identifier.

    Exact evaluation name and sample ID are used, so MMLU subjects cannot
    collide and the same query receives the same ID in 0-shot and 5-shot data.
    """

    if pd.isna(eval_name) or pd.isna(sample_id):
        raise ValueError("eval_name and sample_id are required for a query ID")
    identity = ["routerbench-query-v1", str(eval_name).strip(), str(sample_id).strip()]
    return "rbq_" + _hash_json(identity)[:24]


def infer_model_columns(frame: pd.DataFrame) -> Tuple[str, ...]:
    """Infer score columns and require matching response and cost columns."""

    candidates = tuple(
        str(column)
        for column in frame.columns
        if "|" not in str(column) and str(column) not in NON_MODEL_COLUMNS
    )
    if not candidates:
        raise ValueError("no RouterBench model score columns were found")
    missing = []
    for model in candidates:
        for suffix in ("|model_response", "|total_cost"):
            if model + suffix not in frame.columns:
                missing.append(model + suffix)
    if missing:
        raise ValueError("missing matching RouterBench columns: " + ", ".join(missing))
    return candidates


def _frame_content_hash(frame: pd.DataFrame) -> str:
    """Create a deterministic in-memory source hash for synthetic data."""

    rows = []
    for _, row in frame.iterrows():
        rows.append([str(column) + "=" + repr(row[column]) for column in frame.columns])
    return _hash_json({"columns": [str(c) for c in frame.columns], "rows": rows})


def _row_fingerprint(row: pd.Series, columns: Sequence[str]) -> str:
    return _hash_json([repr(row[column]) for column in columns])


def _valid_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _is_missing(value: Any) -> bool:
    """Return scalar missingness without treating list-like payloads as booleans."""

    result = pd.isna(value)
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _equivalent(left: Any, right: Any) -> bool:
    left_missing = _is_missing(left)
    right_missing = _is_missing(right)
    if left_missing and right_missing:
        return True
    if left_missing or right_missing:
        return False
    try:
        result = left == right
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except (TypeError, ValueError):
        pass
    return repr(left) == repr(right)


def _coalesce_query(
    group: pd.DataFrame,
    model_columns: Sequence[str],
) -> Tuple[Optional[pd.Series], Optional[str], Tuple[int, ...]]:
    """Merge complementary duplicate fragments or return an exclusion reason."""

    source_positions = tuple(group.index)
    if len(group) == 1:
        row = group.iloc[0].copy()
        if any(_is_missing(row[model]) for model in model_columns):
            return None, "excluded_missing_correctness", source_positions
        if any(not _valid_number(row[model]) for model in model_columns):
            return None, "excluded_invalid_correctness", source_positions
        return row, None, source_positions

    ordered_columns = [
        str(column) for column in group.columns
        if not str(column).startswith("_source_")
    ]
    ranked = []
    for position, row in group.iterrows():
        completeness = sum(not pd.isna(row[model]) for model in model_columns)
        ranked.append((-completeness, _row_fingerprint(row, ordered_columns), position))
    ranked.sort()
    primary_position = ranked[0][2]
    merged = group.loc[primary_position].copy()

    for model in model_columns:
        score_rows = group.loc[
            [not _is_missing(value) for value in group[model]]
        ]
        values = list(score_rows[model])
        if not values:
            return None, "excluded_missing_correctness", tuple(group.index)
        first = values[0]
        if any(not _equivalent(first, other) for other in values[1:]):
            return None, "excluded_conflicting_duplicate_scores", tuple(group.index)
        merged[model] = first

        # A fragmented query can carry stale payloads for models whose score is
        # absent on that fragment. Bind response and cost to the fragment that
        # actually supplied this model's official score.
        for suffix in ("|model_response", "|total_cost"):
            column = model + suffix
            values = [value for value in score_rows[column] if not _is_missing(value)]
            if values:
                first = values[0]
                if any(not _equivalent(first, other) for other in values[1:]):
                    return None, "excluded_conflicting_duplicate_payload", tuple(group.index)
                merged[column] = first

    if any(not _valid_number(merged[model]) for model in model_columns):
        return None, "excluded_invalid_correctness", tuple(group.index)
    return merged, None, tuple(group.index)


def load_routerbench(
    source: Source,
    *,
    shot: Optional[str] = None,
    source_name: Optional[str] = None,
    model_columns: Optional[Sequence[str]] = None,
) -> RouterBenchData:
    """Load and audit one RouterBench release without binarizing scores.

    Unsupported task rows remain in ``row_audit`` with an explicit reason.
    Complementary duplicate fragments are merged only when their overlapping
    values agree. Conflicting or incomplete supported queries are excluded and
    every contributing raw row receives the same explicit reason.
    """

    if isinstance(source, pd.DataFrame):
        raw = source.copy(deep=True)
        resolved_name = source_name or "in_memory_routerbench"
        source_hash = _frame_content_hash(raw)
    else:
        path = Path(source)
        raw = pd.read_pickle(path)
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("RouterBench pickle must contain a pandas DataFrame")
        raw = raw.copy(deep=True)
        resolved_name = source_name or path.name
        source_hash = file_sha256(path)

    missing_metadata = [column for column in REQUIRED_METADATA if column not in raw]
    if missing_metadata:
        raise ValueError("missing required metadata columns: " + ", ".join(missing_metadata))
    models = tuple(model_columns) if model_columns is not None else infer_model_columns(raw)
    for model in models:
        if model not in raw:
            raise ValueError("missing requested model column: " + str(model))
        for suffix in ("|model_response", "|total_cost"):
            if model + suffix not in raw:
                raise ValueError("missing matching column: " + model + suffix)

    if shot is None:
        lowered = resolved_name.lower()
        shot = "5shot" if "5shot" in lowered else "0shot" if "0shot" in lowered else "unspecified"
    shot = str(shot)

    work = raw.reset_index(drop=True)
    work["_source_index"] = list(raw.index)
    work["_source_row"] = np.arange(len(work), dtype=int)
    audit_records = []
    query_groups: Dict[str, list[int]] = {}

    for position, row in work.iterrows():
        eval_name = row["eval_name"]
        sample_id = row["sample_id"]
        task = normalize_task(eval_name)
        if pd.isna(eval_name) or pd.isna(sample_id) or pd.isna(row["prompt"]):
            fallback = "rbr_" + _row_fingerprint(row, list(work.columns))[:24]
            audit_records.append({
                "source_row": int(row["_source_row"]),
                "source_index": repr(row["_source_index"]),
                "query_id": fallback,
                "eval_name": None if pd.isna(eval_name) else str(eval_name),
                "sample_id": None if pd.isna(sample_id) else str(sample_id),
                "task": task,
                "included": False,
                "reason": "excluded_missing_required_metadata",
            })
            continue
        query_id = stable_query_id(eval_name, sample_id)
        reason = "pending_supported_query"
        included = False
        if task is None:
            reason = "excluded_unsupported_task:" + str(eval_name)
        else:
            query_groups.setdefault(query_id, []).append(position)
        audit_records.append({
            "source_row": int(row["_source_row"]),
            "source_index": repr(row["_source_index"]),
            "query_id": query_id,
            "eval_name": str(eval_name),
            "sample_id": str(sample_id),
            "task": task,
            "included": included,
            "reason": reason,
        })

    audit = pd.DataFrame(audit_records).set_index("source_row", drop=False)
    included_rows = []
    for query_id, positions in sorted(query_groups.items()):
        group = work.loc[positions]
        merged, exclusion, source_positions = _coalesce_query(group, models)
        audit_positions = tuple(int(work.loc[position, "_source_row"]) for position in source_positions)
        if exclusion is not None:
            audit.loc[list(audit_positions), "included"] = False
            audit.loc[list(audit_positions), "reason"] = exclusion
            continue
        assert merged is not None
        duplicate = len(group) > 1
        reason = (
            "included_merged_complementary_rows"
            if duplicate else "included_supported_task"
        )
        audit.loc[list(audit_positions), "included"] = True
        audit.loc[list(audit_positions), "reason"] = reason
        merged["query_id"] = query_id
        merged["task"] = normalize_task(merged["eval_name"])
        merged["shot"] = shot
        merged["source_rows"] = tuple(sorted(audit_positions))
        included_rows.append(merged)

    if included_rows:
        frame = pd.DataFrame(included_rows).reset_index(drop=True)
        for model in models:
            frame[model] = pd.to_numeric(frame[model], errors="raise").astype(float)
        if not frame["query_id"].is_unique:
            raise RuntimeError("internal error: included query IDs are not unique")
        frame = frame.sort_values("query_id", kind="stable").reset_index(drop=True)
    else:
        frame = pd.DataFrame(columns=list(raw.columns) + [
            "query_id", "task", "shot", "source_rows"
        ])

    if len(audit) != len(raw):
        raise RuntimeError("internal error: source row accounting is incomplete")
    if audit["reason"].str.startswith("pending").any():
        raise RuntimeError("internal error: a source row has no final disposition")

    degeneracy = _audit_prompt_degeneracy(frame)

    return RouterBenchData(
        frame=frame,
        row_audit=audit.reset_index(drop=True),
        model_columns=models,
        source_name=resolved_name,
        source_sha256=source_hash,
        shot=shot,
        prompt_degeneracy=degeneracy,
    )


def _audit_prompt_degeneracy(frame: pd.DataFrame) -> Tuple[dict, ...]:
    """Per-task distinct-prompt census.

    The router's only input is the prompt. If a task's prompts are not distinct
    the router has no query signal and every routing number computed on that task
    is void, so this audit must run on load rather than being discovered later.
    Found 2026-09-03 on RouterBench: `grade-school-math` has 7,450 rows and one
    distinct prompt because the released prompt field is truncated before the
    question, and twelve families in total collapse to a single value.
    """

    if len(frame) == 0 or "prompt" not in frame.columns or "task" not in frame.columns:
        return ()
    normalized = frame["prompt"].map(_unwrap_routerbench_prompt)
    records: list[dict] = []
    for task, group in normalized.groupby(frame["task"].astype(str)):
        size = int(len(group))
        unique = int(group.nunique())
        records.append(
            {
                "task": str(task),
                "rows": size,
                "unique_prompts": unique,
                "duplicate_rate": float(1.0 - unique / size) if size else 0.0,
                "degenerate": bool(unique <= 1),
            }
        )
    records.sort(key=lambda record: (-record["duplicate_rate"], record["task"]))
    return tuple(records)


def _unwrap_routerbench_prompt(value: Any) -> str:
    """Unwrap RouterBench's single-element-list string encoding for prompts."""

    current: Any = value
    for _ in range(3):
        if isinstance(current, bytes):
            current = current.decode("utf-8", errors="replace")
        if isinstance(current, (list, tuple)):
            if not current:
                return ""
            current = current[0]
            continue
        if not isinstance(current, str):
            return "" if current is None else str(current)
        stripped = current.strip()
        if not stripped or stripped[0] not in "['\"":
            return stripped
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped
        if parsed == current:
            return stripped
        current = parsed
    return str(current) if current is not None else ""


def resolve_coverage_policy(policy: Policy) -> CoveragePolicy:
    if isinstance(policy, CoveragePolicy):
        resolved = policy
    elif isinstance(policy, str) and policy in COVERAGE_POLICIES:
        resolved = COVERAGE_POLICIES[policy]
    else:
        names = ", ".join(sorted(COVERAGE_POLICIES))
        raise ValueError("coverage policy must be explicit; available policies: " + names)
    missing = set(SUPPORTED_TASKS) - set(resolved.task_thresholds)
    if missing:
        raise ValueError("coverage policy is missing tasks: " + ", ".join(sorted(missing)))
    for task, comparator in resolved.task_comparators.items():
        if comparator not in ("ge", "gt"):
            raise ValueError("coverage comparator must be 'ge' or 'gt'")
    return resolved


def augment_null_ground_truth(standard_labels: np.ndarray) -> np.ndarray:
    """Append the RACER null label, true exactly when every model misses."""

    labels = np.asarray(standard_labels)
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("standard_labels must be a non-empty two-dimensional array")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("standard_labels must be binary")
    boolean = labels.astype(bool, copy=False)
    null = ~np.any(boolean, axis=1, keepdims=True)
    return np.concatenate((boolean, null), axis=1)


def derive_coverage_labels(
    data: RouterBenchData,
    *,
    policy: Policy,
) -> CoverageLabels:
    """Derive binary coverage only under an explicit task-specific policy."""

    resolved = resolve_coverage_policy(policy)
    scores = data.frame.loc[:, data.model_columns].to_numpy(dtype=float, copy=True)
    if scores.shape[0] == 0:
        raise ValueError("data contains no included queries")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("official correctness scores must be finite and in [0, 1]")

    labels = np.empty(scores.shape, dtype=bool)
    tasks = data.frame["task"].astype(str).to_numpy()
    for task in sorted(set(tasks)):
        if task not in resolved.task_thresholds:
            raise ValueError("coverage policy does not define task: " + task)
        rows = tasks == task
        task_scores = scores[rows]
        if task in resolved.binary_only_tasks and not np.all(
            (task_scores == 0.0) | (task_scores == 1.0)
        ):
            raise ValueError(
                "non-binary official score encountered for binary-only task " + task
            )
        threshold = float(resolved.task_thresholds[task])
        comparator = resolved.task_comparators.get(task, "ge")
        labels[rows] = task_scores >= threshold if comparator == "ge" else task_scores > threshold

    augmented = augment_null_ground_truth(labels)
    return CoverageLabels(
        query_ids=data.query_ids,
        model_columns=data.model_columns,
        standard=labels,
        augmented=augmented,
        policy=resolved,
    )


def _validate_fraction(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(name + " must be strictly between zero and one")
    return value


def split_fractions_for_task(
    task: str,
    *,
    design_share_of_racer_train: float = 0.5,
) -> Dict[str, float]:
    """Return five disjoint fractions while preserving RACER's budgets."""

    normalized = normalize_task(task) or str(task)
    if normalized not in SUPPORTED_TASKS:
        raise ValueError("unsupported task: " + str(task))
    design_share = _validate_fraction(
        design_share_of_racer_train, "design_share_of_racer_train"
    )
    racer_train = 0.5 if normalized == "gsm8k" else 0.4
    evaluation = 1.0 - racer_train
    if normalized == "arc-challenge":
        cal, validation, test = 0.4, 0.2, 0.4
    else:
        cal, validation, test = 0.5, 0.1, 0.4
    fractions = {
        "train": racer_train * (1.0 - design_share),
        "design": racer_train * design_share,
        "calibration": evaluation * cal,
        "validation": evaluation * validation,
        "test": evaluation * test,
    }
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("internal error: split fractions do not sum to one")
    return fractions


def _largest_remainder_counts(total: int, fractions: Sequence[float]) -> np.ndarray:
    exact = np.asarray(fractions, dtype=float) * int(total)
    counts = np.floor(exact).astype(int)
    remainder = int(total) - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def _split_rank(seed: int, task: str, query_id: str) -> str:
    return sha256(
        (PROTOCOL_NAME + "\x1f" + str(seed) + "\x1f" + task + "\x1f" + query_id).encode("utf-8")
    ).hexdigest()


def assign_racer_splits(
    data: RouterBenchData,
    *,
    seed: int,
    design_share_of_racer_train: float = 0.5,
) -> SplitAssignment:
    """Assign deterministic per-task splits by stable query-ID hash rank."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    design_share = _validate_fraction(
        design_share_of_racer_train, "design_share_of_racer_train"
    )

    records = []
    fractions_by_task: Dict[str, Mapping[str, float]] = {}
    for task, group in data.frame.groupby("task", sort=True):
        task = str(task)
        fractions = split_fractions_for_task(
            task, design_share_of_racer_train=design_share
        )
        fractions_by_task[task] = fractions
        ordered = sorted(
            group["query_id"].astype(str),
            key=lambda query_id: (_split_rank(seed, task, query_id), query_id),
        )
        racer_train_fraction = fractions["train"] + fractions["design"]
        evaluation_fraction = 1.0 - racer_train_fraction
        budget_counts = _largest_remainder_counts(
            len(ordered), [racer_train_fraction, evaluation_fraction]
        )
        development_counts = _largest_remainder_counts(
            int(budget_counts[0]), [1.0 - design_share, design_share]
        )
        evaluation_counts = _largest_remainder_counts(
            int(budget_counts[1]),
            [
                fractions["calibration"] / evaluation_fraction,
                fractions["validation"] / evaluation_fraction,
                fractions["test"] / evaluation_fraction,
            ],
        )
        counts = np.concatenate((development_counts, evaluation_counts))
        offset = 0
        for split, count in zip(SPLIT_NAMES, counts):
            for query_id in ordered[offset:offset + int(count)]:
                records.append({"query_id": query_id, "task": task, "split": split})
            offset += int(count)
        if offset != len(ordered):
            raise RuntimeError("internal error: split allocation is incomplete")

    assignments = pd.DataFrame(records)
    assignments = assignments.sort_values(["task", "split", "query_id"], kind="stable").reset_index(drop=True)
    if len(assignments) != len(data.frame):
        raise RuntimeError("split assignment does not account for every included query")
    if not assignments["query_id"].is_unique:
        raise RuntimeError("a query was assigned to more than one split")
    if set(assignments["query_id"]) != set(data.query_ids):
        raise RuntimeError("split assignment query IDs do not match the dataset")
    return SplitAssignment(
        assignments=assignments,
        seed=seed,
        design_share_of_racer_train=design_share,
        fractions_by_task=fractions_by_task,
    )


def _overlap_section(left_ids: Iterable[str], right_ids: Iterable[str]) -> Dict[str, Any]:
    left = set(map(str, left_ids))
    right = set(map(str, right_ids))
    intersection = left & right
    union = left | right
    return {
        "left_unique": len(left),
        "right_unique": len(right),
        "overlap": len(intersection),
        "left_only": len(left - right),
        "right_only": len(right - left),
        "jaccard": (len(intersection) / len(union)) if union else 1.0,
        "left_only_ids": sorted(left - right),
        "right_only_ids": sorted(right - left),
    }


def shot_overlap_report(
    zero_shot: RouterBenchData,
    five_shot: RouterBenchData,
) -> Dict[str, Any]:
    """Report query overlap and prompt/score reuse between shot releases."""

    all_overlap = _overlap_section(
        zero_shot.row_audit["query_id"], five_shot.row_audit["query_id"]
    )
    included_overlap = _overlap_section(zero_shot.query_ids, five_shot.query_ids)
    left = zero_shot.frame.set_index("query_id", drop=False)
    right = five_shot.frame.set_index("query_id", drop=False)
    shared = sorted(set(left.index) & set(right.index))
    prompt_equal = 0
    score_equal = 0
    common_models = tuple(model for model in zero_shot.model_columns if model in five_shot.model_columns)
    per_task = {}
    for query_id in shared:
        if _equivalent(left.at[query_id, "prompt"], right.at[query_id, "prompt"]):
            prompt_equal += 1
        if np.array_equal(
            left.loc[query_id, list(common_models)].to_numpy(dtype=float),
            right.loc[query_id, list(common_models)].to_numpy(dtype=float),
        ):
            score_equal += 1
    for task in sorted(set(left["task"]) | set(right["task"])):
        left_ids = left.index[left["task"] == task]
        right_ids = right.index[right["task"] == task]
        per_task[str(task)] = _overlap_section(left_ids, right_ids)
    return {
        "left_shot": zero_shot.shot,
        "right_shot": five_shot.shot,
        "raw_rows": {
            "left": zero_shot.raw_row_count,
            "right": five_shot.raw_row_count,
        },
        "all_query_identities": all_overlap,
        "included_queries": included_overlap,
        "shared_included_prompt_equal": prompt_equal,
        "shared_included_prompt_different": len(shared) - prompt_equal,
        "shared_included_official_scores_equal": score_equal,
        "shared_included_official_scores_different": len(shared) - score_equal,
        "per_task": per_task,
    }


def _reason_counts(audit: pd.DataFrame) -> Dict[str, int]:
    counts = audit["reason"].value_counts(dropna=False).sort_index()
    return {str(reason): int(count) for reason, count in counts.items()}


def build_split_manifest(
    data: RouterBenchData,
    assignment: SplitAssignment,
    *,
    coverage_policy: Optional[Policy] = None,
    overlap_report: Optional[Mapping[str, Any]] = None,
    include_row_accounting: bool = True,
) -> Dict[str, Any]:
    """Build a JSON-ready manifest with source, split, and content hashes."""

    if set(assignment.assignments["query_id"]) != set(data.query_ids):
        raise ValueError("assignment and data query IDs differ")
    policy_dict = (
        resolve_coverage_policy(coverage_policy).to_dict()
        if coverage_policy is not None else None
    )
    split_ids: Dict[str, Dict[str, list[str]]] = {}
    split_hashes: Dict[str, Dict[str, str]] = {}
    for task in sorted(assignment.fractions_by_task):
        split_ids[task] = {}
        split_hashes[task] = {}
        for split in SPLIT_NAMES:
            ids = sorted(assignment.ids(task, split))
            split_ids[task][split] = ids
            split_hashes[task][split] = _hash_json(ids)

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "source": {
            "name": data.source_name,
            "sha256": data.source_sha256,
            "shot": data.shot,
            "raw_rows": data.raw_row_count,
            "included_queries": data.included_query_count,
            "included_source_rows": int(data.row_audit["included"].sum()),
            "excluded_source_rows": int((~data.row_audit["included"]).sum()),
            "reason_counts": _reason_counts(data.row_audit),
        },
        "model_columns": list(data.model_columns),
        "seed": assignment.seed,
        "split_names": list(SPLIT_NAMES),
        "design_share_of_racer_train": assignment.design_share_of_racer_train,
        "fractions_by_task": {
            task: dict(fractions)
            for task, fractions in sorted(assignment.fractions_by_task.items())
        },
        "coverage_policy": policy_dict,
        "splits": split_ids,
        "split_hashes": split_hashes,
        "all_included_query_ids_sha256": _hash_json(sorted(data.query_ids)),
    }
    if include_row_accounting:
        manifest["row_accounting"] = [
            {
                "source_row": int(row.source_row),
                "source_index": str(row.source_index),
                "query_id": str(row.query_id),
                "included": bool(row.included),
                "reason": str(row.reason),
            }
            for row in data.row_audit.itertuples(index=False)
        ]
    if overlap_report is not None:
        manifest["shot_overlap_report"] = dict(overlap_report)
    manifest["manifest_sha256"] = _hash_json(manifest)
    return manifest


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported split manifest schema version")
    if manifest.get("protocol") != PROTOCOL_NAME:
        raise ValueError("unexpected split manifest protocol")
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str):
        raise ValueError("split manifest is missing manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if _hash_json(unsigned) != expected:
        raise ValueError("split manifest content hash mismatch")

    all_ids = []
    for task, task_splits in manifest.get("splits", {}).items():
        for split in SPLIT_NAMES:
            ids = task_splits.get(split)
            if not isinstance(ids, list):
                raise ValueError("split manifest is missing " + task + "/" + split)
            expected_hash = manifest.get("split_hashes", {}).get(task, {}).get(split)
            if _hash_json(sorted(map(str, ids))) != expected_hash:
                raise ValueError("split hash mismatch for " + task + "/" + split)
            all_ids.extend(map(str, ids))
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split manifest assigns at least one query more than once")
    if _hash_json(sorted(all_ids)) != manifest.get("all_included_query_ids_sha256"):
        raise ValueError("all-query split hash mismatch")


def save_split_manifest(manifest: Mapping[str, Any], path: Union[str, Path]) -> str:
    """Validate and atomically save a split manifest. Return its file hash."""

    materialized = dict(manifest)
    unsigned = dict(materialized)
    unsigned.pop("manifest_sha256", None)
    materialized["manifest_sha256"] = _hash_json(unsigned)
    _validate_manifest(materialized)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        materialized, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return file_sha256(destination)


def load_split_manifest(
    path: Union[str, Path],
    *,
    source_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load a manifest and verify all internal hashes and optional source hash."""

    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("split manifest must contain a JSON object")
    _validate_manifest(manifest)
    if source_path is not None:
        actual = file_sha256(source_path)
        expected = manifest.get("source", {}).get("sha256")
        if actual != expected:
            raise ValueError("RouterBench source file hash mismatch")
    return manifest


__all__ = [
    "COVERAGE_POLICIES",
    "CoverageLabels",
    "CoveragePolicy",
    "GSM8K_GE_0_75_COVERAGE_POLICY",
    "GSM8K_POSITIVE_COVERAGE_POLICY",
    "GSM8K_STRICT_COVERAGE_POLICY",
    "NULL_MODEL_NAME",
    "PRIMARY_COVERAGE_POLICY",
    "PROTOCOL_NAME",
    "RouterBenchData",
    "SCHEMA_VERSION",
    "SPLIT_NAMES",
    "SUPPORTED_TASKS",
    "SplitAssignment",
    "assign_racer_splits",
    "augment_null_ground_truth",
    "build_split_manifest",
    "derive_coverage_labels",
    "file_sha256",
    "infer_model_columns",
    "load_routerbench",
    "load_split_manifest",
    "normalize_task",
    "resolve_coverage_policy",
    "save_split_manifest",
    "shot_overlap_report",
    "split_fractions_for_task",
    "stable_query_id",
]
