"""Check the complete ISSUE-78 Stage 1 uncertainty rollup."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_issue78_o2_full_matrix import D1_METHODS, D2_METHODS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def expected_pairs() -> set[tuple[str, str]]:
    return {
        *((method, "fixed_tally_factorized") for method in D1_METHODS if method not in {"fixed_tally_factorized", "fixed_design_marginal_factorized"}),
        *((d2, d1) for d1, d2 in zip(D1_METHODS, D2_METHODS) if d1 != "fixed_tally_factorized"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--stamp", required=True)
    args = parser.parse_args()
    prefix = f"issue78_{args.host}_{args.stamp}"
    rollup = args.root / "rollup" / f"{prefix}_stage1_uncertainty.csv"
    manifest = args.root / "rollup" / f"{prefix}_stage1_uncertainty.json"
    report = args.root / "rollup" / f"{prefix}_stage1_uncertainty_check.json"
    done = args.root / "rollup" / f"{prefix}_CHECK_DONE"
    for path in (report, done):
        if path.exists():
            raise SystemExit(f"refusing existing path {path}")
    csv.field_size_limit(sys.maxsize)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    with rollup.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    if payload.get("row_count") != 5600 or len(rows) != 5600:
        errors.append("row count mismatch")
    if payload.get("cell_count") != 32:
        errors.append("cell count mismatch")
    if payload.get("bootstrap_repetitions") != 10000 or payload.get("bootstrap_seed") != 20260904:
        errors.append("bootstrap manifest mismatch")
    hashes = payload.get("source_analysis_sha256", {})
    if len(hashes) != 32:
        errors.append("source analysis count mismatch")
    for source, expected in hashes.items():
        path = Path(source)
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"source hash mismatch: {source}")
    keys = []
    by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for index, row in enumerate(rows):
        key = (
            row["protocol"], row["benchmark"], row["pool_size"],
            row["target_alpha"], row["method"], row["reference_method"],
        )
        keys.append(key)
        by_cell.setdefault((row["protocol"], row["benchmark"], row["pool_size"]), []).append(row)
        if any(value.strip().lower() in {"na", "n/a", "nan"} for value in row.values()):
            errors.append(f"row {index}: forbidden NA token")
        if int(row["bootstrap_repetitions"]) != 10000 or int(row["bootstrap_seed"]) != 20260904:
            errors.append(f"row {index}: bootstrap identity mismatch")
        cert_n = int(row["method_certification_numerator"])
        cert_d = int(row["method_certification_denominator"])
        if cert_d != 20 or not close(float(row["method_certification_rate"]), cert_n / cert_d):
            errors.append(f"row {index}: certification arithmetic mismatch")
        event_n = int(row["method_test_miss_numerator"])
        event_d = int(row["method_test_miss_denominator"])
        risk = optional_float(row["method_certified_standard_risk"])
        risk_se = optional_float(row["method_certified_standard_risk_se"])
        if event_d:
            if row["event_count_status"] != "exact" or risk is None or not close(risk, event_n / event_d):
                errors.append(f"row {index}: event arithmetic mismatch")
        elif row["event_count_status"] != "no_deployed_test_queries" or risk is not None:
            errors.append(f"row {index}: zero-event status mismatch")
        expected_flag = (
            "" if risk is None or risk_se is None
            else str(risk > float(row["target_alpha"]) + 2.0 * risk_se).lower()
        )
        if row["above_target_2se"] != expected_flag:
            errors.append(f"row {index}: above-target flag mismatch")
    if len(keys) != len(set(keys)):
        errors.append("duplicate row identity")
    if len(by_cell) != 32:
        errors.append(f"expected 32 cells, found {len(by_cell)}")
    pair_set = expected_pairs()
    targets = {"0.02", "0.03", "0.05", "0.075", "0.1", "0.15", "0.2"}
    for cell, cell_rows in by_cell.items():
        observed_targets = {row["target_alpha"] for row in cell_rows}
        if observed_targets != targets:
            errors.append(f"{cell}: target coverage mismatch")
        for alpha in targets:
            observed_pairs = {
                (row["method"], row["reference_method"])
                for row in cell_rows if row["target_alpha"] == alpha
            }
            if observed_pairs != pair_set:
                errors.append(f"{cell} alpha {alpha}: pair coverage mismatch")
    hyper = payload.get("hyperparameter_frequencies", [])
    if len(hyper) != 32:
        errors.append("hyperparameter source count mismatch")
    for record in hyper:
        frequencies = record.get("frequencies", {})
        if set(frequencies) != targets:
            errors.append(f"{record.get('benchmark')}: hyper target coverage mismatch")
            continue
        for alpha, methods in frequencies.items():
            if len(methods) != 28:
                errors.append(f"{record.get('benchmark')} alpha {alpha}: hyper method count mismatch")
            for method, counts in methods.items():
                if sum(int(value) for value in counts.values()) != 20:
                    errors.append(f"{record.get('benchmark')} alpha {alpha} {method}: hyper count mismatch")
    result = {
        "schema_version": 1,
        "issue": 78,
        "host": args.host,
        "stamp": args.stamp,
        "passed": not errors,
        "row_count": len(rows),
        "cell_count": len(by_cell),
        "source_count": len(hashes),
        "errors": errors,
    }
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(result, indent=2))
        return 2
    done.write_text("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
