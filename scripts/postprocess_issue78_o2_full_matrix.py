"""Merge and analyze all 32 ISSUE-78 Stage 1 cells."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures_src.analyze_paired_main_suite import analyze_records, write_csv, write_json
from scripts.merge_paired_suite_shards import merge_payloads, validate_payload
from scripts.run_issue78_o2_full_matrix import D1_METHODS, D2_METHODS
from scripts.run_paired_main_suite import atomic_write_json

ROUTEREVAL_TASKS = (
    "ifeval",
    "musr",
    "bbh_boolean_expressions",
    "bbh_navigate",
    "gpqa_diamond",
    "math_algebra_hard",
    "math_intermediate_algebra_hard",
)
ROUTERBENCH_TASKS = ("mmlu", "hellaswag", "winogrande", "arc-challenge")
POOLS = (100, 200, 300, 500)
REDRAWS = tuple(range(20))
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20260904


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def expected_cells() -> list[tuple[str, str, int]]:
    return [
        *(('routereval', task, pool) for task in ROUTEREVAL_TASKS for pool in POOLS),
        *(('routerbench', task, 11) for task in ROUTERBENCH_TASKS),
    ]


def comparison_pairs() -> list[tuple[str, str]]:
    new_d1 = [
        (method, "fixed_tally_factorized")
        for method in D1_METHODS
        if method not in {"fixed_tally_factorized", "fixed_design_marginal_factorized"}
    ]
    new_d2 = [
        (d2, d1)
        for d1, d2 in zip(D1_METHODS, D2_METHODS)
        if d1 != "fixed_tally_factorized"
    ]
    pairs = new_d1 + new_d2
    if len(pairs) != 25 or len(set(pairs)) != 25:
        raise RuntimeError("ISSUE-78 comparison pair registry must contain 25 unique pairs")
    return pairs


def shard_path(root: Path, host: str, stamp: str, cell: tuple[str, str, int], redraw: int) -> Path:
    protocol, benchmark, pool = cell
    return root / "shards" / (
        f"issue78_{host}_{stamp}_{protocol}_{benchmark}_pool{pool}_r{redraw:02d}.json"
    )


def hyper_frequencies(payload: dict[str, Any], host: str, stamp: str) -> dict[str, Any]:
    methods = list(payload["method_identities"])
    frequencies: dict[str, dict[str, dict[str, int]]] = {}
    for cell in payload["cells"]:
        alpha_key = str(cell["alpha"])
        frequencies[alpha_key] = {}
        for method in methods:
            counts: dict[str, int] = {}
            for redraw in cell["redraws"]:
                key = json.dumps(
                    redraw["methods"][method].get("chosen_hyper"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                counts[key] = counts.get(key, 0) + 1
            if sum(counts.values()) != 20:
                raise RuntimeError(f"{method} alpha {alpha_key} hyper count is not 20")
            frequencies[alpha_key][method] = counts
    return {
        "schema_version": 1,
        "issue": 78,
        "host": host,
        "stamp": stamp,
        "protocol": payload["protocol"],
        "benchmark": payload["benchmark"],
        "pool": payload["pool_size"],
        "redraws": 20,
        "frequencies": frequencies,
    }


def process_cell(
    root: Path,
    host: str,
    stamp: str,
    cell: tuple[str, str, int],
) -> dict[str, Any]:
    protocol, benchmark, pool = cell
    paths = [shard_path(root, host, stamp, cell, redraw) for redraw in REDRAWS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{cell}: missing {len(missing)} shards")
    merged_path = root / "merged" / (
        f"issue78_{host}_{stamp}_{protocol}_{benchmark}_pool{pool}.json"
    )
    if merged_path.is_file():
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
        validate_payload(merged, merged_path)
    else:
        shards = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
        merged = merge_payloads(shards)
        atomic_write_json(merged_path, merged)
    if merged["redraw_range"] != {
        "start_inclusive": 0,
        "end_exclusive": 20,
        "count": 20,
    }:
        raise RuntimeError(f"{cell}: merged redraw range differs")
    if len(merged["method_identities"]) != 28:
        raise RuntimeError(f"{cell}: merged method count differs")

    expected_rows = 25 * 7
    analysis_json = root / "analysis" / (
        f"issue78_{host}_{stamp}_{protocol}_{benchmark}_pool{pool}_paired.json"
    )
    analysis_csv = analysis_json.with_suffix(".csv")
    hyper_path = root / "analysis" / (
        f"issue78_{host}_{stamp}_{protocol}_{benchmark}_pool{pool}_hyperparameter_frequencies.json"
    )
    outputs = (analysis_json, analysis_csv, hyper_path)
    if all(path.is_file() for path in outputs):
        analysis = json.loads(analysis_json.read_text(encoding="utf-8"))
        if analysis.get("row_count") != expected_rows:
            raise RuntimeError(f"{cell}: existing analysis row count differs")
        reused = True
    elif any(path.exists() for path in outputs):
        raise FileExistsError(f"{cell}: incomplete existing analysis outputs")
    else:
        analysis = analyze_records(
            [(str(merged_path), sha256_file(merged_path), merged)],
            bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED,
            comparison_pairs=comparison_pairs(),
        )
        if analysis["row_count"] != expected_rows:
            raise RuntimeError(f"{cell}: expected {expected_rows} analysis rows")
        write_json(analysis, analysis_json)
        write_csv(analysis["rows"], analysis_csv)
        hyper = hyper_frequencies(merged, host, stamp)
        atomic_json(hyper_path, hyper)
        reused = False
    return {
        "protocol": protocol,
        "benchmark": benchmark,
        "pool": pool,
        "status": "complete",
        "reused_complete_outputs": reused,
        "merged": str(merged_path),
        "merged_sha256": sha256_file(merged_path),
        "analysis_json": str(analysis_json),
        "analysis_json_sha256": sha256_file(analysis_json),
        "analysis_csv": str(analysis_csv),
        "analysis_csv_sha256": sha256_file(analysis_csv),
        "hyper": str(hyper_path),
        "hyper_sha256": sha256_file(hyper_path),
        "analysis_rows": analysis["row_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root
    prefix = f"issue78_{args.host}_{args.stamp}"
    if root.name != prefix:
        raise SystemExit("root name differs from host-stamped prefix")
    for directory in (root / "merged", root / "analysis"):
        directory.mkdir(parents=True, exist_ok=True)
    progress = root / f"{prefix}_postprocess_v3_progress.json"
    done = root / f"{prefix}_POSTPROCESS_DONE"
    failed = root / f"{prefix}_POSTPROCESS_V3_FAILED"
    for path in (progress, done, failed):
        if path.exists():
            raise SystemExit(f"refusing existing path {path}")
    records: list[dict[str, Any]] = []
    started = time.monotonic()

    def publish() -> None:
        value = {
            "schema_version": 1,
            "issue": 78,
            "host": args.host,
            "stamp": args.stamp,
            "workers": args.workers,
            "executor": "process_pool",
            "resume_complete_outputs": True,
            "requested": 32,
            "complete": sum(row["status"] == "complete" for row in records),
            "failed": sum(row["status"] == "failed" for row in records),
            "elapsed_seconds": time.monotonic() - started,
            "records": sorted(
                records,
                key=lambda row: (row["protocol"], row["benchmark"], row["pool"]),
            ),
        }
        temporary = progress.with_name(progress.name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, progress)

    cells = expected_cells()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_cell, root, args.host, args.stamp, cell): cell
            for cell in cells
        }
        for future in concurrent.futures.as_completed(futures):
            cell = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {
                    "protocol": cell[0],
                    "benchmark": cell[1],
                    "pool": cell[2],
                    "status": "failed",
                    "reason": repr(error),
                }
            records.append(row)
            publish()
    failures = [row for row in records if row["status"] != "complete"]
    (failed if failures else done).write_text("\n")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
