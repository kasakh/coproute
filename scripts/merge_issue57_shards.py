"""Merge disjoint corrected ISSUE-57 seed shards for one registered cell."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.design_bprime_run import DEPENDENCE_ARMS, _assert_arm_presence, _standard_error


IDENTITY_FIELDS = (
    "schema_version",
    "design",
    "issue",
    "registered_arms",
    "protocol",
    "benchmark",
    "pool_size",
    "design_budget",
    "alphas",
    "split_ratios",
    "max_depth",
    "null_shuffle",
    "hyperparameter_grids",
    "feature_audit",
    "source_sha256",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _aggregate(seed_results: list[dict], alphas: list[float], null_shuffle: bool) -> dict:
    aggregate: dict[str, Any] = {}
    for alpha in alphas:
        key = f"{alpha}"
        baseline = [row["per_alpha"][key]["marginal"] for row in seed_results]
        base_calls = [row["test_calls"] for row in baseline if row["test_calls"] is not None]
        base_risks = [row["test_risk"] for row in baseline if row["test_risk"] is not None]
        cell: dict[str, Any] = {
            "marginal_test_calls_mean": float(np.mean(base_calls)) if base_calls else None,
            "marginal_test_calls_se": _standard_error(base_calls),
            "marginal_test_risk_mean": float(np.mean(base_risks)) if base_risks else None,
            "marginal_standard_all_miss_risk_mean": (
                float(np.mean(base_risks)) if base_risks else None
            ),
            "marginal_standard_all_miss_risk_se": _standard_error(base_risks),
            "marginal_certification_count": len(base_calls),
            "marginal_certification_rate": len(base_calls) / len(seed_results),
            "methods": {},
        }
        for arm in DEPENDENCE_ARMS:
            records = [row["per_alpha"][key]["tuned"][arm] for row in seed_results]
            calls = [row["test_calls"] for row in records if row["test_calls"] is not None]
            risks = [row["test_risk"] for row in records if row["test_risk"] is not None]
            info: dict[str, Any] = {
                "hyper_name": records[0]["hyper_name"],
                "grid_size": records[0]["grid_size"],
                "effective_k": max(row["effective_k"] for row in records),
                "chosen_hyper_by_seed": [row["chosen_hyper"] for row in records],
                "test_calls_mean": float(np.mean(calls)) if calls else None,
                "test_calls_se": _standard_error(calls),
                "test_risk_mean": float(np.mean(risks)) if risks else None,
                "standard_all_miss_risk_mean": float(np.mean(risks)) if risks else None,
                "standard_all_miss_risk_se": _standard_error(risks),
                "n_seeds_feasible": len(calls),
                "certification_count": len(calls),
                "certification_rate": len(calls) / len(seed_results),
                "risk_valid": float(np.mean(risks)) <= alpha + 1e-12 if risks else None,
                "beats_marginal_on_calls": (
                    float(np.mean(calls)) < float(np.mean(base_calls)) - 1e-12
                    if calls and base_calls
                    else None
                ),
            }
            if null_shuffle:
                shuffled = [
                    row["per_alpha"][key]["null_shuffled"][arm]
                    for row in seed_results
                ]
                shuffled_calls = [
                    row["test_calls"] for row in shuffled if row["test_calls"] is not None
                ]
                info["null_shuffled_test_calls_mean"] = (
                    float(np.mean(shuffled_calls)) if shuffled_calls else None
                )
            cell["methods"][arm] = info
        aggregate[key] = cell
    return aggregate


def merge(paths: Sequence[Path]) -> dict:
    if not paths:
        raise ValueError("at least one shard is required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = payloads[0]
    identity = {field: reference.get(field) for field in IDENTITY_FIELDS}
    seeds: set[int] = set()
    seed_results: list[dict] = []
    for path, payload in zip(paths, payloads):
        candidate = {field: payload.get(field) for field in IDENTITY_FIELDS}
        if _canonical(candidate) != _canonical(identity):
            raise ValueError(f"{path}: immutable cell identity differs")
        shard_seeds = [int(seed) for seed in payload.get("seeds", [])]
        overlap = sorted(seeds.intersection(shard_seeds))
        if overlap:
            raise ValueError(f"{path}: duplicate seeds {overlap}")
        if shard_seeds != [int(row["seed"]) for row in payload.get("seed_results", [])]:
            raise ValueError(f"{path}: seed metadata differs from seed rows")
        seeds.update(shard_seeds)
        seed_results.extend(copy.deepcopy(payload["seed_results"]))
    seed_results.sort(key=lambda row: int(row["seed"]))
    ordered_seeds = [int(row["seed"]) for row in seed_results]
    _assert_arm_presence(seed_results, [float(value) for value in reference["alphas"]])
    merged = copy.deepcopy(reference)
    merged["seeds"] = ordered_seeds
    merged["seed_results"] = seed_results
    merged["aggregate"] = _aggregate(
        seed_results,
        [float(value) for value in merged["alphas"]],
        bool(merged["null_shuffle"]),
    )
    merged["source_shards"] = [str(path) for path in paths]
    merged["run_started_utc"] = min(payload["run_started_utc"] for payload in payloads)
    merged["run_finished_utc"] = max(payload["run_finished_utc"] for payload in payloads)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite immutable result {args.out}")
    payload = merge(args.shards)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
