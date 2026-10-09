"""Repaired RouterBench E2 runner for the standard all-miss protocol.

This runner intentionally reuses the calibration and candidate-selection helpers
from ``design_bprime_e2_run.py``. Every redraw has disjoint design, router,
validation, final-calibration, and test partitions. Wrapper hyperparameters are
selected with temporary validation calibration, then frozen. The untouched
final-calibration labels are used once for the frozen candidate. A failed final
certificate emits no calls or test-risk operating point.

Router choice
-------------
RouterBench does not need an externally fitted RACER router here. For each task
and redraw, a KNN router is fitted only on that redraw's dedicated router
partition using the archived mDeBERTa embeddings and model-correctness labels.
It predicts per-model marginals for the remaining queries. This keeps the
router task-specific, prevents target labels from crossing split boundaries,
and makes every arm share exactly the same frozen marginal predictions.

The registered E2 tasks are MMLU, HellaSwag, and Winogrande. ARC-Challenge is
included in the separate feasibility census but is not an ISSUE-50 main E2 task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.data_protocol import (  # noqa: E402
    PRIMARY_COVERAGE_POLICY,
    derive_coverage_labels,
    load_routerbench,
    resolve_coverage_policy,
)
from components.racer_protocol import (  # noqa: E402
    augment_ground_truth,
    augment_router_scores,
    nonconformity_scores,
)
from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS  # noqa: E402
from components.scale_selectors import build_order  # noqa: E402
from scripts.build_routerbench_embedding_archive import (  # noqa: E402
    MANIFEST_FILENAME as EMBEDDING_MANIFEST_FILENAME,
    PROTOCOL_NAME as EMBEDDING_ARCHIVE_PROTOCOL,
    load_archive_manifest,
    sha256_file,
)
from scripts.design_bprime_e2_run import (  # noqa: E402
    RACER_CONTROLS,
    WRAPPER_ARMS,
    WRAPPER_ARM_METADATA,
    WRAPPER_HYPER_GRIDS,
    _final_calibrate_and_test,
    _knn_probabilities,
    _score_wrapper_grid,
    _select_and_finalize_wrapper_arm,
    build_summary,
)

CLEAN_TASKS = ("mmlu", "hellaswag", "winogrande")
DEFAULT_ALPHAS = (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
SELECTION_PROTOCOL = (
    "design_and_validation_then_freeze_then_one_final_calibration_then_test"
)
ROUTER_PROTOCOL = "within_task_knn_on_mdeberta_dedicated_router_split"


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_rows(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embeddings contain a zero-norm row")
    return matrix / norms


def load_aligned_embedding_archive(
    archive_path: str | Path,
    query_ids: Sequence[str],
    *,
    manifest_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Verify an immutable archive and align it to audited query IDs."""

    path = Path(archive_path)
    if path.suffix != ".npz":
        raise ValueError("embedding archive must be an npz file")
    resolved_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else path.parent / EMBEDDING_MANIFEST_FILENAME
    )
    if not resolved_manifest.is_file():
        raise ValueError("embedding archive requires its manifest.json")
    manifest = load_archive_manifest(resolved_manifest, archive_path=path)
    if manifest.get("protocol") != EMBEDDING_ARCHIVE_PROTOCOL:
        raise ValueError("unexpected RouterBench embedding archive protocol")

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"embeddings", "query_ids"}:
            raise ValueError(
                "embedding archive must contain exactly embeddings and query_ids"
            )
        embeddings = np.asarray(archive["embeddings"], dtype=float)
        embedded_ids = tuple(str(value) for value in archive["query_ids"].tolist())

    expected_ids = tuple(str(value) for value in query_ids)
    if len(embedded_ids) != len(set(embedded_ids)):
        raise ValueError("embedding archive contains duplicate query IDs")
    if set(embedded_ids) != set(expected_ids):
        raise ValueError(
            "embedding query IDs do not exactly match audited RouterBench queries"
        )
    if manifest["reconstruction"]["query_ids_sha256"] != _hash_json(
        list(embedded_ids)
    ):
        raise ValueError("embedding query IDs do not match the manifest")

    row_of = {query_id: row for row, query_id in enumerate(embedded_ids)}
    aligned = _normalize_rows(
        embeddings[np.asarray([row_of[query_id] for query_id in expected_ids])]
    )
    if aligned.shape != (
        int(manifest["output"]["n_rows"]),
        int(manifest["output"]["embedding_dim"]),
    ):
        raise ValueError("embedding matrix shape does not match the manifest")
    return aligned, {
        "archive_sha256": sha256_file(path),
        "manifest_file_sha256": sha256_file(resolved_manifest),
        "manifest_content_sha256": manifest["manifest_sha256"],
        "archive_protocol": manifest["protocol"],
        "row_alignment": "exact_query_id_set_then_audited_loader_order",
        "n_rows": int(aligned.shape[0]),
        "embedding_dim": int(aligned.shape[1]),
        "mdeberta_source": True,
    }


def partition_task_rows(
    n_queries: int,
    *,
    seed: int,
    redraw: int,
    design_fraction: float = 0.25,
    router_fraction: float = 0.25,
    calibration_fraction: float = 0.20,
    validation_fraction: float = 0.15,
    minimum_partition_size: int = 1,
) -> dict[str, np.ndarray]:
    """Create the same disjoint five-way redraw split as the standard E2 runner."""

    fractions = np.asarray(
        [
            design_fraction,
            router_fraction,
            calibration_fraction,
            validation_fraction,
        ],
        dtype=float,
    )
    if n_queries <= 0 or redraw < 0 or seed < 0:
        raise ValueError("query count, seed, and redraw must be non-negative")
    if np.any(fractions <= 0.0) or float(fractions.sum()) >= 1.0:
        raise ValueError("four positive fractions must leave a positive test fraction")
    if minimum_partition_size < 1:
        raise ValueError("minimum partition size must be positive")

    permutation = np.random.default_rng(seed + 13 * redraw).permutation(n_queries)
    n_design = int(round(n_queries * design_fraction))
    n_router = int(round(n_queries * router_fraction))
    n_pool = n_queries - n_design - n_router
    test_fraction = 1.0 - float(fractions.sum())
    evaluation_fraction = calibration_fraction + validation_fraction + test_fraction
    n_calibration = int(round(n_pool * calibration_fraction / evaluation_fraction))
    n_validation = int(round(n_pool * validation_fraction / evaluation_fraction))
    counts = (
        n_design,
        n_router,
        n_calibration,
        n_validation,
        n_pool - n_calibration - n_validation,
    )
    if min(counts) < minimum_partition_size:
        raise ValueError(
            "partition too small: "
            + ", ".join(
                f"{name}={count}"
                for name, count in zip(
                    ("design", "router", "final_calibration", "validation", "test"),
                    counts,
                )
            )
        )

    output: dict[str, np.ndarray] = {}
    offset = 0
    for name, count in zip(
        ("design", "router", "final_calibration", "validation", "test"),
        counts,
    ):
        output[name] = permutation[offset : offset + count]
        offset += count
    combined = np.concatenate(tuple(output.values()))
    if offset != n_queries or len(np.unique(combined)) != n_queries:
        raise RuntimeError("task split is incomplete or overlapping")
    return output


def evaluate_task_redraw(
    *,
    correctness: np.ndarray,
    embeddings: np.ndarray,
    query_ids: Sequence[str],
    alphas: Sequence[float],
    redraw: int,
    seed: int,
    knn_k: int,
    max_depth: int,
    wrapper_arms: Sequence[str] = WRAPPER_ARMS,
    minimum_partition_size: int = 20,
    design_fraction: float = 0.25,
    router_fraction: float = 0.25,
    calibration_fraction: float = 0.20,
    validation_fraction: float = 0.15,
) -> tuple[dict[str, dict[str, dict]], dict[str, Any]]:
    """Evaluate one task redraw with one frozen within-task KNN router."""

    labels = np.asarray(correctness)
    features = np.asarray(embeddings, dtype=float)
    ids = tuple(str(value) for value in query_ids)
    if labels.ndim != 2 or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("correctness must be a non-empty binary matrix")
    if labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("correctness must be a non-empty binary matrix")
    if features.shape[0] != labels.shape[0] or len(ids) != labels.shape[0]:
        raise ValueError("correctness, embeddings, and query IDs must align")
    if len(ids) != len(set(ids)):
        raise ValueError("query IDs must be unique")
    if knn_k < 1 or max_depth < 1:
        raise ValueError("knn-k and max-depth must be positive")
    arms = tuple(wrapper_arms)
    unknown = set(arms) - set(WRAPPER_ARMS)
    if unknown:
        raise ValueError("unknown wrapper arms: " + ", ".join(sorted(unknown)))

    split = partition_task_rows(
        len(ids),
        seed=seed,
        redraw=redraw,
        design_fraction=design_fraction,
        router_fraction=router_fraction,
        calibration_fraction=calibration_fraction,
        validation_fraction=validation_fraction,
        minimum_partition_size=minimum_partition_size,
    )
    design_rows = split["design"]
    router_rows = split["router"]
    pool_rows = np.concatenate(
        (
            split["final_calibration"],
            split["validation"],
            split["test"],
        )
    )
    n_calibration = len(split["final_calibration"])
    n_validation = len(split["validation"])
    cal_idx = np.arange(n_calibration)
    val_idx = np.arange(n_calibration, n_calibration + n_validation)
    test_idx = np.arange(n_calibration + n_validation, len(pool_rows))

    probabilities, neighbor_correctness = _knn_probabilities(
        features[router_rows],
        labels[router_rows],
        features[pool_rows],
        knn_k,
        return_support=True,
    )
    standard_labels = labels[pool_rows].astype(bool)
    augmented_labels = augment_ground_truth(standard_labels)
    augmented_probabilities = augment_router_scores(probabilities)
    design_labels = labels[design_rows].astype(int)
    effective_max_depth = min(int(max_depth), labels.shape[1])
    fixed_tally_order = build_order(
        "empirical_tally",
        design_labels,
        model_names=[str(index) for index in range(labels.shape[1])],
        max_depth=effective_max_depth,
    )

    controls: dict[str, np.ndarray] = {}
    for variant in ("G", "P"):
        _, scores = nonconformity_scores(
            probabilities,
            variant,
            seed=seed,
            split_seed=redraw + 1,
        )
        controls[f"racer_{variant}"] = scores
    fitted = _score_wrapper_grid(
        probabilities=probabilities,
        design_labels=design_labels,
        neighbor_correctness=neighbor_correctness,
        fixed_tally_order=fixed_tally_order,
        max_depth=effective_max_depth,
        seed=seed,
        split_seed=redraw + 1,
        arms=arms,
    )

    records: dict[str, dict[str, dict]] = {}
    for alpha_value in alphas:
        alpha = float(alpha_value)
        alpha_key = f"{alpha}"
        alpha_records: dict[str, dict] = {}
        for control_name, scores in controls.items():
            record = _final_calibrate_and_test(
                scores=scores,
                augmented_probabilities=augmented_probabilities,
                standard_labels=standard_labels,
                augmented_labels=augmented_labels,
                cal_idx=cal_idx,
                test_idx=test_idx,
                alpha=alpha,
            )
            record.update(
                {
                    "redraw": int(redraw),
                    "selected_arm": control_name,
                    "chosen_hyper": None,
                    "selection_source": "predeclared_control",
                    "neighbor_k": int(knn_k),
                }
            )
            alpha_records[control_name] = record
        for arm in arms:
            record = _select_and_finalize_wrapper_arm(
                arm=arm,
                fitted=fitted,
                augmented_probabilities=augmented_probabilities,
                standard_labels=standard_labels,
                augmented_labels=augmented_labels,
                val_idx=val_idx,
                cal_idx=cal_idx,
                test_idx=test_idx,
                alpha=alpha,
                neighbor_support_size=neighbor_correctness.shape[1],
            )
            record.update({"redraw": int(redraw), "neighbor_k": int(knn_k)})
            alpha_records[f"wrapper_{arm}_tuned"] = record
        records[alpha_key] = alpha_records

    split_record = {
        "redraw": int(redraw),
        "split_query_ids": {
            name: [ids[int(index)] for index in rows] for name, rows in split.items()
        },
        "split_counts": {name: int(len(rows)) for name, rows in split.items()},
        "router_fit_scope": "same_task_dedicated_router_split_only",
        "router_fit_query_ids_sha256": _hash_json(
            [ids[int(index)] for index in split["router"]]
        ),
        "knn_k_requested": int(knn_k),
        "knn_k_effective": int(neighbor_correctness.shape[1]),
    }
    return records, split_record


def _parse_alphas(value: str) -> tuple[float, ...]:
    parsed = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not parsed or any(not 0.0 < alpha < 1.0 for alpha in parsed):
        raise ValueError("alphas must be comma-separated values strictly between zero and one")
    if len(parsed) != len(set(parsed)):
        raise ValueError("alphas must be unique")
    return parsed


def _parse_tasks(value: str) -> tuple[str, ...]:
    parsed = tuple(token.strip() for token in value.split(",") if token.strip())
    if not parsed:
        raise ValueError("at least one task is required")
    unknown = set(parsed) - set(CLEAN_TASKS)
    if unknown:
        raise ValueError(
            "E2 supports only " + ", ".join(CLEAN_TASKS) + "; unknown: "
            + ", ".join(sorted(unknown))
        )
    if len(parsed) != len(set(parsed)):
        raise ValueError("tasks must be unique")
    return parsed


def _feature_audits(data: Any, tasks: Sequence[str]) -> dict[str, dict]:
    by_task = {str(record["task"]): dict(record) for record in data.prompt_degeneracy}
    output = {}
    for task in tasks:
        if task not in by_task:
            raise ValueError(f"{task}: loader did not produce a feature audit")
        record = by_task[task]
        if int(record["unique_prompts"]) <= 1:
            raise ValueError(f"{task}: prompt features are degenerate")
        if float(record["duplicate_rate"]) > 0.10:
            raise ValueError(f"{task}: prompt duplicate rate exceeds 0.10")
        output[task] = record
    return output


def _write_immutable_output(output_dir: Path, result: Mapping[str, Any]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output directory {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=output_dir.parent)
    )
    try:
        result_path = temporary / "e2_results.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "DONE").write_text("ok\n", encoding="utf-8")
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routerbench-pkl", required=True)
    parser.add_argument("--embedding-archive", required=True)
    parser.add_argument("--embedding-manifest", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tasks", default=",".join(CLEAN_TASKS))
    parser.add_argument(
        "--alphas", default=",".join(str(value) for value in DEFAULT_ALPHAS)
    )
    parser.add_argument("--redraws", type=int, default=40)
    parser.add_argument("--redraw-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--knn-k", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=11)
    parser.add_argument("--design-fraction", type=float, default=0.25)
    parser.add_argument("--router-fraction", type=float, default=0.25)
    parser.add_argument("--cal-fraction", type=float, default=0.20)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--coverage-policy", default=PRIMARY_COVERAGE_POLICY.name
    )
    arguments = parser.parse_args(argv)
    if arguments.redraws <= 0 or arguments.redraw_start < 0:
        parser.error("redraws must be positive and redraw-start must be non-negative")
    if arguments.knn_k <= 0 or arguments.max_depth <= 0:
        parser.error("knn-k and max-depth must be positive")
    try:
        tasks = _parse_tasks(arguments.tasks)
        alphas = _parse_alphas(arguments.alphas)
        policy = resolve_coverage_policy(arguments.coverage_policy)
    except ValueError as error:
        parser.error(str(error))

    data = load_routerbench(
        arguments.routerbench_pkl,
        shot="0shot",
        model_columns=ROUTERBENCH_MODEL_COLUMNS,
    )
    labels = derive_coverage_labels(data, policy=policy)
    embeddings, embedding_audit = load_aligned_embedding_archive(
        arguments.embedding_archive,
        data.query_ids,
        manifest_path=arguments.embedding_manifest,
    )
    audits = _feature_audits(data, tasks)

    task_results: dict[str, dict] = {}
    for task in tasks:
        task_rows = np.flatnonzero(data.frame["task"].astype(str).to_numpy() == task)
        task_correctness = labels.standard[task_rows].astype(np.int8)
        task_embeddings = embeddings[task_rows]
        task_query_ids = tuple(data.query_ids[int(index)] for index in task_rows)
        per_redraw = {
            f"{alpha}": {
                name: []
                for name in RACER_CONTROLS
                + tuple(f"wrapper_{arm}_tuned" for arm in WRAPPER_ARMS)
            }
            for alpha in alphas
        }
        split_assignments = []
        for local_redraw in range(arguments.redraws):
            redraw = arguments.redraw_start + local_redraw
            redraw_records, split_record = evaluate_task_redraw(
                correctness=task_correctness,
                embeddings=task_embeddings,
                query_ids=task_query_ids,
                alphas=alphas,
                redraw=redraw,
                seed=arguments.seed,
                knn_k=arguments.knn_k,
                max_depth=arguments.max_depth,
                minimum_partition_size=20,
                design_fraction=arguments.design_fraction,
                router_fraction=arguments.router_fraction,
                calibration_fraction=arguments.cal_fraction,
                validation_fraction=arguments.val_fraction,
            )
            for alpha_key, arm_records in redraw_records.items():
                for name, record in arm_records.items():
                    per_redraw[alpha_key][name].append(record)
            split_assignments.append(split_record)

        summary = build_summary(per_redraw, list(alphas))
        task_results[task] = {
            "n_queries": int(len(task_rows)),
            "n_models": int(task_correctness.shape[1]),
            "query_ids_sha256": _hash_json(list(task_query_ids)),
            "complete_pool_miss": float(
                np.mean(np.sum(task_correctness, axis=1) == 0)
            ),
            "best_single_model_miss": float(
                1.0 - np.max(np.mean(task_correctness, axis=0))
            ),
            "feature_audit": audits[task],
            "split_assignments": split_assignments,
            "per_redraw": per_redraw,
            "summary": summary,
        }
        print(task, flush=True)
        for alpha in alphas:
            print(f"  alpha {alpha}", flush=True)
            for name, info in summary[f"{alpha}"]["arms"].items():
                if info["calls_mean"] is None:
                    print(f"    {name}: NON-CERTIFIED", flush=True)
                else:
                    print(
                        f"    {name}: calls={info['calls_mean']:.3f} "
                        f"standard_miss={info['standard_all_miss_risk_mean']:.4f} "
                        f"certification_rate={info['certification_rate']:.3f}",
                        flush=True,
                    )

    result = {
        "design": "B_prime_E2_frozen_selection",
        "protocol": "routerbench_standard_all_miss_v2",
        "selection_protocol": SELECTION_PROTOCOL,
        "primary_risk_name": "standard_all_miss_risk",
        "published_racer_diagnostic_name": "published_augmented_null_loss",
        "router_protocol": ROUTER_PROTOCOL,
        "router_choice": (
            "A task-specific KNN router is fitted inside each redraw from only "
            "that task's dedicated router split and archived mDeBERTa embeddings. "
            "No external fitted router is loaded."
        ),
        "tasks": list(tasks),
        "excluded_from_e2": {
            "arc-challenge": "feasibility census only, not an ISSUE-50 main E2 task",
            "gsm8k": "loader reports degenerate prompt features",
            "mbpp": "not in the ISSUE-50 main E2 task set",
        },
        "alphas": list(alphas),
        "redraws": int(arguments.redraws),
        "redraw_start": int(arguments.redraw_start),
        "redraw_end": int(arguments.redraw_start + arguments.redraws),
        "seed": int(arguments.seed),
        "knn_k": int(arguments.knn_k),
        "max_depth": int(min(arguments.max_depth, len(data.model_columns))),
        "full_prefix_evaluated": bool(arguments.max_depth >= len(data.model_columns)),
        "model_ids": list(data.model_columns),
        "coverage_policy": policy.to_dict(),
        "source": {
            "routerbench_pickle": str(Path(arguments.routerbench_pkl).resolve()),
            "routerbench_pickle_sha256": sha256_file(arguments.routerbench_pkl),
            "embedding_archive": str(Path(arguments.embedding_archive).resolve()),
            **embedding_audit,
        },
        "wrapper_arms": list(WRAPPER_ARMS),
        "wrapper_arm_metadata": WRAPPER_ARM_METADATA,
        "hyperparameter_grids": WRAPPER_HYPER_GRIDS,
        "racer_controls": list(RACER_CONTROLS),
        "partition_fractions": {
            "design": float(arguments.design_fraction),
            "router": float(arguments.router_fraction),
            "final_calibration": float(arguments.cal_fraction),
            "validation": float(arguments.val_fraction),
            "test": float(
                1.0
                - arguments.design_fraction
                - arguments.router_fraction
                - arguments.cal_fraction
                - arguments.val_fraction
            ),
        },
        "task_results": task_results,
    }
    _write_immutable_output(Path(arguments.out_dir), result)
    print(f"wrote {Path(arguments.out_dir) / 'e2_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
