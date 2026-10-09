"""Build an ISSUE-62-compatible compact rollup from ISSUE-78 analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

FIELDS = [
    "protocol", "benchmark", "pool_id", "pool_size", "target_alpha", "method", "reference_method",
    "redraw_count", "paired_certified_redraw_count", "method_certified_redraw_count",
    "reference_certified_redraw_count", "method_certified_mean_calls", "method_certified_calls_se",
    "reference_certified_mean_calls", "reference_certified_calls_se", "paired_mean_calls_difference",
    "paired_calls_difference_se", "paired_calls_redraw_ci95_low", "paired_calls_redraw_ci95_high",
    "paired_calls_query_cluster_ci95_low", "paired_calls_query_cluster_ci95_high",
    "method_certified_standard_risk", "method_certified_standard_risk_se",
    "reference_certified_standard_risk", "reference_certified_standard_risk_se",
    "paired_mean_standard_risk_difference", "paired_standard_risk_difference_se",
    "paired_standard_risk_redraw_ci95_low", "paired_standard_risk_redraw_ci95_high",
    "paired_standard_risk_query_cluster_ci95_low", "paired_standard_risk_query_cluster_ci95_high",
    "method_test_miss_numerator", "method_test_miss_denominator",
    "reference_test_miss_numerator", "reference_test_miss_denominator",
    "method_certification_numerator", "method_certification_denominator", "method_certification_rate",
    "method_certification_wilson_ci95_low", "method_certification_wilson_ci95_high",
    "reference_certification_numerator", "reference_certification_denominator", "reference_certification_rate",
    "reference_certification_wilson_ci95_low", "reference_certification_wilson_ci95_high",
    "method_full_pool_calibration_misses", "method_full_pool_calibration_denominator",
    "method_full_pool_test_misses", "method_full_pool_test_denominator",
    "shared_feasibility_equivalent", "certification_disagreement_count",
    "bootstrap_seed", "bootstrap_repetitions", "redraw_interval_method", "query_interval_method",
    "difference_definition", "pairing_rule", "test_retuning_used",
]
EXTRA = [
    "certification_status",
    "event_count_status",
    "above_target_2se",
    "source_analysis_csv",
    "source_analysis_sha256",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def project(row: dict[str, str], source: Path, source_hash: str) -> dict[str, str]:
    output = {field: row[field] for field in FIELDS}
    cert_n = int(row["method_certification_numerator"])
    cert_d = int(row["method_certification_denominator"])
    event_d = int(row["method_test_miss_denominator"])
    output["certification_status"] = (
        "certified_redraws_present" if cert_n else "zero_certification_explicit"
    )
    output["event_count_status"] = "exact" if event_d else "no_deployed_test_queries"
    risk = optional_float(row["method_certified_standard_risk"])
    risk_se = optional_float(row["method_certified_standard_risk_se"])
    alpha = float(row["target_alpha"])
    output["above_target_2se"] = (
        "" if risk is None or risk_se is None else str(risk > alpha + 2.0 * risk_se).lower()
    )
    output["source_analysis_csv"] = str(source)
    output["source_analysis_sha256"] = source_hash
    if cert_d != 20:
        raise ValueError(f"{source}: certification denominator is not 20")
    if int(row["bootstrap_repetitions"]) != 10000 or int(row["bootstrap_seed"]) != 20260904:
        raise ValueError(f"{source}: bootstrap identity differs")
    if row["test_retuning_used"].lower() != "false":
        raise ValueError(f"{source}: test retuning was used")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--worker-count", required=True, type=int)
    args = parser.parse_args()
    prefix = f"issue78_{args.host}_{args.stamp}"
    if args.root.name != prefix:
        raise SystemExit("root name differs from host-stamped prefix")
    output_dir = args.root / "rollup"
    if output_dir.exists():
        raise SystemExit(f"refusing existing output directory {output_dir}")
    output_dir.mkdir()
    csv.field_size_limit(sys.maxsize)
    sources = sorted((args.root / "analysis").glob("*_paired.csv"))
    hyper_sources = sorted((args.root / "analysis").glob("*_hyperparameter_frequencies.json"))
    if len(sources) != 32 or len(hyper_sources) != 32:
        raise SystemExit(
            f"expected 32 analysis and hyper files, found {len(sources)} and {len(hyper_sources)}"
        )
    rows = []
    hashes = {}
    for source in sources:
        digest = sha256(source)
        hashes[str(source)] = digest
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(set(FIELDS) - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"{source}: missing fields {missing}")
            rows.extend(project(row, source, digest) for row in reader)
    rows.sort(
        key=lambda row: (
            row["protocol"],
            row["benchmark"],
            int(row["pool_size"]),
            float(row["target_alpha"]),
            row["method"],
            row["reference_method"],
        )
    )
    keys = [
        (
            row["protocol"], row["benchmark"], row["pool_size"],
            row["target_alpha"], row["method"], row["reference_method"],
        )
        for row in rows
    ]
    if len(rows) != 5600 or len(keys) != len(set(keys)):
        raise ValueError(f"expected 5600 unique rows, found {len(rows)}")
    output_csv = output_dir / f"{prefix}_stage1_uncertainty.csv"
    with output_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS + EXTRA)
        writer.writeheader()
        writer.writerows(rows)
    hyper = [json.loads(path.read_text(encoding="utf-8")) for path in hyper_sources]
    output_json = output_dir / f"{prefix}_stage1_uncertainty.json"
    atomic_json(output_json, {
        "schema_version": 1,
        "issue": 78,
        "host": args.host,
        "stamp": args.stamp,
        "worker_count": args.worker_count,
        "row_count": len(rows),
        "cell_count": 32,
        "comparison_pairs_per_target": 25,
        "targets_per_cell": 7,
        "source_analysis_sha256": hashes,
        "hyperparameter_frequencies": hyper,
        "classification_rule": "mean standard risk > target alpha + 2 * standard error",
        "bootstrap_repetitions": 10000,
        "bootstrap_seed": 20260904,
        "rows": rows,
    })
    (output_dir / f"{prefix}_ROLLUP_DONE").write_text("\n")
    print(json.dumps({"rows": len(rows), "sources": len(sources), "hyper_sources": len(hyper_sources)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
