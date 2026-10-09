"""Report RouterBench structural feasibility and marginal prefix headroom.

This is a descriptive full-data census, not a calibration certificate. For each
requested task it reports the complete-pool all-miss floor, the best one-model
miss rate, and the all-miss rate of the deterministic marginal-accuracy ordering
at every depth from one through all eleven models.

The exact empirical target ranges are:

* alpha below the complete-pool floor: structurally infeasible;
* alpha from the complete-pool floor up to the best-single miss rate: feasible
  with room for set routing to improve over one call;
* alpha at or above the best-single miss rate: already met by one model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS  # noqa: E402

DEFAULT_TASKS = ("mmlu", "hellaswag", "winogrande", "arc-challenge")
DEFAULT_TARGETS = (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_target(
    alpha: float,
    *,
    complete_pool_all_miss: float,
    best_single_model_miss: float,
) -> str:
    """Classify one target from exact empirical structural boundaries."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if alpha < complete_pool_all_miss:
        return "infeasible_even_with_complete_pool"
    if alpha >= best_single_model_miss:
        return "target_already_met_with_one_call"
    return "feasible_with_multi_model_headroom"


def task_feasibility(
    correctness: Any,
    model_ids: Sequence[str],
    *,
    targets: Sequence[float] = DEFAULT_TARGETS,
) -> dict[str, Any]:
    """Compute exact counts and rates for one binary task matrix."""

    labels = np.asarray(correctness)
    names = tuple(str(value) for value in model_ids)
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("correctness must be a non-empty two-dimensional matrix")
    if labels.shape[1] != len(names) or len(names) != len(set(names)):
        raise ValueError("model IDs must uniquely match correctness columns")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("correctness must be binary")

    n_queries = int(labels.shape[0])
    marginal_correct = np.sum(labels, axis=0)
    ordering = tuple(
        sorted(
            range(labels.shape[1]),
            key=lambda index: (-int(marginal_correct[index]), names[index]),
        )
    )
    depth_records = []
    for depth in range(1, labels.shape[1] + 1):
        selected = ordering[:depth]
        miss_count = int(np.sum(np.sum(labels[:, selected], axis=1) == 0))
        depth_records.append(
            {
                "depth": depth,
                "miss_count": miss_count,
                "miss_rate": miss_count / n_queries,
                "added_model": names[ordering[depth - 1]],
                "prefix_models": [names[index] for index in selected],
            }
        )

    complete_miss_count = int(np.sum(np.sum(labels, axis=1) == 0))
    best_index = ordering[0]
    best_single_miss_count = int(n_queries - marginal_correct[best_index])
    complete_rate = complete_miss_count / n_queries
    best_single_rate = best_single_miss_count / n_queries
    if depth_records[0]["miss_count"] != best_single_miss_count:
        raise RuntimeError("marginal depth one does not equal the best single model")
    if depth_records[-1]["miss_count"] != complete_miss_count:
        raise RuntimeError("marginal full depth does not equal the complete-pool floor")

    classifications = {
        f"{float(alpha)}": classify_target(
            float(alpha),
            complete_pool_all_miss=complete_rate,
            best_single_model_miss=best_single_rate,
        )
        for alpha in targets
    }
    return {
        "n_queries": n_queries,
        "n_models": int(labels.shape[1]),
        "complete_pool_all_miss_count": complete_miss_count,
        "complete_pool_all_miss_rate": complete_rate,
        "best_single_model": names[best_index],
        "best_single_model_miss_count": best_single_miss_count,
        "best_single_model_miss_rate": best_single_rate,
        "marginal_model_order": [names[index] for index in ordering],
        "marginal_depths": depth_records,
        "structural_alpha_ranges": {
            "infeasible_even_with_complete_pool": {
                "lower": 0.0,
                "lower_inclusive": True,
                "upper": complete_rate,
                "upper_inclusive": False,
            },
            "feasible_with_multi_model_headroom": {
                "lower": complete_rate,
                "lower_inclusive": True,
                "upper": best_single_rate,
                "upper_inclusive": False,
            },
            "target_already_met_with_one_call": {
                "lower": best_single_rate,
                "lower_inclusive": True,
                "upper": 1.0,
                "upper_inclusive": True,
            },
        },
        "target_classification": classifications,
    }


def build_report(
    routerbench_pickle: str | Path,
    *,
    tasks: Sequence[str] = DEFAULT_TASKS,
    targets: Sequence[float] = DEFAULT_TARGETS,
    coverage_policy: str = PRIMARY_COVERAGE_POLICY.name,
) -> dict[str, Any]:
    """Load audited RouterBench rows and build the requested task census."""

    selected_tasks = tuple(str(value) for value in tasks)
    if not selected_tasks or len(selected_tasks) != len(set(selected_tasks)):
        raise ValueError("tasks must be a non-empty unique sequence")
    unknown = set(selected_tasks) - set(DEFAULT_TASKS)
    if unknown:
        raise ValueError("unsupported feasibility tasks: " + ", ".join(sorted(unknown)))
    parsed_targets = tuple(float(value) for value in targets)
    if any(not 0.0 <= value <= 1.0 for value in parsed_targets):
        raise ValueError("targets must be between zero and one")

    policy = resolve_coverage_policy(coverage_policy)
    data = load_routerbench(
        routerbench_pickle,
        shot="0shot",
        model_columns=ROUTERBENCH_MODEL_COLUMNS,
    )
    labels = derive_coverage_labels(data, policy=policy)
    task_array = data.frame["task"].astype(str).to_numpy()
    audits = {str(item["task"]): dict(item) for item in data.prompt_degeneracy}
    results = {}
    for task in selected_tasks:
        rows = np.flatnonzero(task_array == task)
        if len(rows) == 0:
            raise ValueError(f"{task}: audited loader returned zero included queries")
        results[task] = {
            **task_feasibility(
                labels.standard[rows], labels.model_columns, targets=parsed_targets
            ),
            "feature_audit": audits.get(task),
        }

    return {
        "schema_version": 1,
        "protocol": "routerbench_structural_feasibility_v1",
        "scope": "descriptive_complete_cleaned_query_pool_not_a_calibration_certificate",
        "source": {
            "path": str(Path(routerbench_pickle).resolve()),
            "sha256": sha256_file(routerbench_pickle),
            "raw_rows": data.raw_row_count,
            "included_cleaned_queries": data.included_query_count,
        },
        "coverage_policy": policy.to_dict(),
        "tasks": list(selected_tasks),
        "targets": list(parsed_targets),
        "results": results,
    }


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in value.split(",") if token.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(token.strip()) for token in value.split(",") if token.strip())


def write_immutable_json(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable report {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routerbench-pkl", required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument(
        "--targets", default=",".join(str(value) for value in DEFAULT_TARGETS)
    )
    parser.add_argument("--coverage-policy", default=PRIMARY_COVERAGE_POLICY.name)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args(argv)
    try:
        report = build_report(
            arguments.routerbench_pkl,
            tasks=_parse_csv(arguments.tasks),
            targets=_parse_floats(arguments.targets),
            coverage_policy=arguments.coverage_policy,
        )
    except ValueError as error:
        parser.error(str(error))

    print("task,n_queries,complete_pool_all_miss,best_single_model_miss")
    for task in report["tasks"]:
        record = report["results"][task]
        print(
            f"{task},{record['n_queries']},"
            f"{record['complete_pool_all_miss_rate']:.12f},"
            f"{record['best_single_model_miss_rate']:.12f}"
        )
        print(
            "  marginal_depth_miss_rates="
            + ",".join(
                f"{item['depth']}:{item['miss_rate']:.12f}"
                for item in record["marginal_depths"]
            )
        )
    if arguments.out:
        write_immutable_json(arguments.out, report)
        print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
