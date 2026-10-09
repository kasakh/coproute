"""Merge ISSUE-74 redraw shards and summarize paired differences from tally."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_issue74_ordering_factorial import REGISTERED_METHODS, SCHEMA_VERSION
from scripts.run_paired_main_suite import atomic_write_json

DYNAMIC_FIELDS = {"redraw_start", "redraw_end", "cells"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in payload.items() if key not in DYNAMIC_FIELDS}


def _calls(record: Mapping[str, Any]) -> float | None:
    value = record.get("test_calls", record.get("calls_test"))
    return None if value is None else float(value)


def _risk(record: Mapping[str, Any]) -> float | None:
    value = record.get("standard_all_miss_risk")
    return None if value is None else float(value)


def _standard_error(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _bootstrap_interval(values: Sequence[float], *, seed: int, repetitions: int = 10000) -> list[float] | None:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return None
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(repetitions, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def _validate(payload: Mapping[str, Any], source: Path) -> tuple[int, int]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{source}: wrong schema version")
    if payload.get("issue") != "ISSUE-74":
        raise ValueError(f"{source}: wrong issue")
    if payload.get("preserve_score_prefixes") is not True:
        raise ValueError(f"{source}: corrected prefix preservation is required")
    methods = payload.get("registered_methods")
    if methods != list(REGISTERED_METHODS):
        raise ValueError(f"{source}: registered methods differ")
    start = int(payload.get("redraw_start", -1))
    end = int(payload.get("redraw_end", -1))
    if start < 0 or end <= start:
        raise ValueError(f"{source}: invalid redraw range")
    cells = payload.get("cells")
    if not isinstance(cells, list) or [cell.get("alpha") for cell in cells] != payload.get("target_grid"):
        raise ValueError(f"{source}: target cells differ from target grid")
    for cell in cells:
        redraws = cell.get("redraws")
        if not isinstance(redraws, list) or [row.get("redraw_id") for row in redraws] != list(range(start, end)):
            raise ValueError(f"{source}: redraw rows differ from range")
        for redraw in redraws:
            records = redraw.get("methods")
            if not isinstance(records, dict) or set(records) != set(REGISTERED_METHODS):
                raise ValueError(f"{source}: registered arm presence mismatch")
            for method, record in records.items():
                if record.get("selection_source") != "validation_only":
                    raise ValueError(f"{source}: {method} was not validation selected")
                if record.get("final_calibration_count") != 1:
                    raise ValueError(f"{source}: {method} did not use one final calibration")
    return start, end


def summarize(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for cell_index, cell in enumerate(cells):
        alpha = str(cell["alpha"])
        method_summary = {}
        redraws = cell["redraws"]
        tally_records = [row["methods"]["fixed_tally_factorized"] for row in redraws]
        for method_index, method in enumerate(REGISTERED_METHODS):
            records = [row["methods"][method] for row in redraws]
            calls = [_calls(record) for record in records]
            risks = [_risk(record) for record in records]
            certified_calls = [value for value in calls if value is not None]
            certified_risks = [value for value in risks if value is not None]
            paired_differences = []
            paired_risk_differences = []
            for record, tally in zip(records, tally_records):
                call = _calls(record)
                tally_call = _calls(tally)
                risk = _risk(record)
                tally_risk = _risk(tally)
                if call is not None and tally_call is not None:
                    paired_differences.append(call - tally_call)
                if risk is not None and tally_risk is not None:
                    paired_risk_differences.append(risk - tally_risk)
            seed = 20260920 + 1000 * cell_index + method_index
            method_summary[method] = {
                "redraw_count": len(records),
                "certified_redraw_count": len(certified_calls),
                "certification_rate": len(certified_calls) / len(records),
                "mean_calls": float(np.mean(certified_calls)) if certified_calls else None,
                "mean_calls_se": _standard_error(certified_calls),
                "standard_all_miss_risk_mean": (
                    float(np.mean(certified_risks)) if certified_risks else None
                ),
                "standard_all_miss_risk_se": _standard_error(certified_risks),
                "paired_tally_redraw_count": len(paired_differences),
                "paired_calls_minus_tally_mean": (
                    float(np.mean(paired_differences)) if paired_differences else None
                ),
                "paired_calls_minus_tally_se": _standard_error(paired_differences),
                "paired_calls_minus_tally_bootstrap_ci95": _bootstrap_interval(
                    paired_differences, seed=seed
                ),
                "paired_risk_minus_tally_mean": (
                    float(np.mean(paired_risk_differences))
                    if paired_risk_differences else None
                ),
                "paired_risk_minus_tally_bootstrap_ci95": _bootstrap_interval(
                    paired_risk_differences, seed=seed + 500000
                ),
            }
        output[alpha] = {"methods": method_summary}
    return output


def merge(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one shard is required")
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        start, end = _validate(payload, path)
        rows.append((start, end, path, payload))
    rows.sort(key=lambda item: (item[0], item[1], str(item[2])))
    reference_identity = _canonical(_identity(rows[0][3]))
    expected = rows[0][0]
    for start, end, path, payload in rows:
        if start != expected:
            raise ValueError(f"{path}: redraw ranges overlap or leave a gap after {expected}")
        if _canonical(_identity(payload)) != reference_identity:
            raise ValueError(f"{path}: immutable identity differs")
        expected = end
    merged = copy.deepcopy(rows[0][3])
    merged["redraw_start"] = rows[0][0]
    merged["redraw_end"] = rows[-1][1]
    alpha_rows = {str(alpha): {"alpha": alpha, "redraws": []} for alpha in merged["target_grid"]}
    for _, _, _, payload in rows:
        for cell in payload["cells"]:
            alpha_rows[str(cell["alpha"])]["redraws"].extend(copy.deepcopy(cell["redraws"]))
    merged["cells"] = [alpha_rows[str(alpha)] for alpha in merged["target_grid"]]
    for cell in merged["cells"]:
        cell["redraws"].sort(key=lambda row: row["redraw_id"])
    _validate(merged, Path("<merged>"))
    merged["summary"] = summarize(merged["cells"])
    merged["source_shards"] = [str(path) for _, _, path, _ in rows]
    merged["artifact_manifest"] = {
        "issue": "ISSUE-74",
        "host": merged["host"],
        "utc_stamp": merged["utc_stamp"],
        "source_shard_count": len(paths),
        "redraw_count": merged["redraw_end"] - merged["redraw_start"],
        "registered_arm_count": len(REGISTERED_METHODS),
        "target_count": len(merged["target_grid"]),
    }
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    atomic_write_json(args.out, merge(args.shards))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
