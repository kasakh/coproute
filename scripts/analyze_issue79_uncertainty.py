"""Build stratified paired uncertainty for the ISSUE-79 selector result."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

TARGETS = (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
SELECTOR = "validation_selected_factorized"
TALLY = "fixed_tally_factorized"
REPETITIONS = 10000
SEED = 2026097901


def atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp",
        delete=False, newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def bootstrap(groups: list[tuple[np.ndarray, int]], seed: int) -> dict[str, Any]:
    if not groups:
        raise ValueError("no paired groups")
    generator = np.random.default_rng(seed)
    numerator = np.zeros(REPETITIONS, dtype=float)
    denominator = 0
    observed_numerator = 0.0
    cluster_count = 0
    for values, query_count in groups:
        values = np.asarray(values, dtype=float)
        sample = generator.integers(0, len(values), size=(REPETITIONS, len(values)))
        numerator += values[sample].sum(axis=1) * query_count
        denominator += len(values) * query_count
        observed_numerator += float(values.sum()) * query_count
        cluster_count += len(values)
    draws = numerator / denominator
    return {
        "paired_redraw_clusters": cluster_count,
        "paired_queries": denominator,
        "mean_calls_difference": observed_numerator / denominator,
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_improvement": float(np.mean(draws < 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--csv-out", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.json_out, args.csv_out):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, int, float], list[tuple[float, int]]] = {}
    paths = sorted((args.root / "shards").glob(f"issue79_{args.host}_{args.stamp}_*.json"))
    if len(paths) != 640:
        raise RuntimeError(f"expected 640 shards, found {len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        redraw = payload["redraws"][0]
        for alpha_row in redraw["alpha_results"]:
            selected = alpha_row["methods"][SELECTOR]
            tally = alpha_row["methods"][TALLY]
            if not (selected["certified"] and tally["certified"]):
                continue
            left = np.asarray(selected["test_calls_by_query"], dtype=float)
            right = np.asarray(tally["test_calls_by_query"], dtype=float)
            if len(left) != len(right) or selected["test_query_ids"] != tally["test_query_ids"]:
                raise RuntimeError("paired test identity mismatch")
            key = (
                payload["protocol"], payload["benchmark"], int(payload["pool_size"]),
                float(alpha_row["alpha"]),
            )
            grouped.setdefault(key, []).append((float(np.mean(left - right)), len(left)))
    all_groups: list[tuple[np.ndarray, int]] = []
    alpha_rows: list[dict[str, Any]] = []
    for alpha_index, alpha in enumerate(TARGETS):
        selected_groups = []
        cell_means = []
        for key, records in grouped.items():
            if key[3] != alpha:
                continue
            query_counts = {record[1] for record in records}
            if len(query_counts) != 1:
                raise RuntimeError(f"query count varies within cell {key}")
            values = np.asarray([record[0] for record in records], dtype=float)
            query_count = query_counts.pop()
            selected_groups.append((values, query_count))
            all_groups.append((values, query_count))
            cell_means.append(float(np.mean(values)))
        result = bootstrap(selected_groups, SEED + alpha_index)
        result.update({
            "target_alpha": alpha,
            "certified_cell_targets": len(selected_groups),
            "cell_target_mean": float(np.mean(cell_means)),
            "cell_target_median": float(np.median(cell_means)),
            "cell_target_wins": int(np.count_nonzero(np.asarray(cell_means) < 0)),
            "cell_target_ties": int(np.count_nonzero(np.asarray(cell_means) == 0)),
            "cell_target_losses": int(np.count_nonzero(np.asarray(cell_means) > 0)),
        })
        alpha_rows.append(result)
    overall = bootstrap(all_groups, SEED + 100)
    cell_means = [float(np.mean([record[0] for record in records])) for records in grouped.values()]
    overall.update({
        "certified_cell_targets": len(grouped),
        "cell_target_mean": float(np.mean(cell_means)),
        "cell_target_median": float(np.median(cell_means)),
        "cell_target_wins": int(np.count_nonzero(np.asarray(cell_means) < 0)),
        "cell_target_ties": int(np.count_nonzero(np.asarray(cell_means) == 0)),
        "cell_target_losses": int(np.count_nonzero(np.asarray(cell_means) > 0)),
    })
    output = {
        "schema_version": 1,
        "issue": 79,
        "host": args.host,
        "stamp": args.stamp,
        "bootstrap_repetitions": REPETITIONS,
        "bootstrap_seed": SEED,
        "bootstrap_unit": "paired redraw within registered dataset_pool_target strata",
        "overall": overall,
        "by_target": alpha_rows,
    }
    atomic_json(args.json_out, output)
    atomic_csv(args.csv_out, alpha_rows)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
