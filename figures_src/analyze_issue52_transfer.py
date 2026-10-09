"""Analyze merged ISSUE-52 calibration-transfer cells without test retuning."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import merge_issue52_shards as merger  # noqa: E402
from scripts import run_issue52_transfer as transfer  # noqa: E402
from scripts import run_paired_main_suite as paired  # noqa: E402

BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260920
WILSON_Z = 1.959963984540054

CSV_COLUMNS = (
    "protocol",
    "fit_mode",
    "source_benchmark",
    "target_benchmark",
    "pool_size",
    "pool_seed",
    "alpha",
    "method",
    "redraw_count",
    "source_calibrated_certification_count",
    "source_calibrated_certification_total",
    "source_calibrated_certification_rate",
    "source_calibrated_certification_wilson_ci95_low",
    "source_calibrated_certification_wilson_ci95_high",
    "source_calibrated_mean_calls",
    "source_calibrated_calls_se",
    "source_calibrated_target_risk",
    "source_calibrated_target_risk_se",
    "source_calibrated_target_event_count",
    "source_calibrated_target_trial_count",
    "source_calibrated_calibration_event_count",
    "source_calibrated_calibration_trial_count",
    "target_recalibrated_certification_count",
    "target_recalibrated_certification_total",
    "target_recalibrated_certification_rate",
    "target_recalibrated_certification_wilson_ci95_low",
    "target_recalibrated_certification_wilson_ci95_high",
    "target_recalibrated_mean_calls",
    "target_recalibrated_calls_se",
    "target_recalibrated_target_risk",
    "target_recalibrated_target_risk_se",
    "target_recalibrated_target_event_count",
    "target_recalibrated_target_trial_count",
    "target_recalibrated_calibration_event_count",
    "target_recalibrated_calibration_trial_count",
    "paired_risk_inflation_mean",
    "paired_risk_inflation_se",
    "paired_risk_inflation_redraw_count",
    "paired_risk_inflation_redraw_ci95_low",
    "paired_risk_inflation_redraw_ci95_high",
    "paired_risk_inflation_query_cluster_ci95_low",
    "paired_risk_inflation_query_cluster_ci95_high",
    "paired_risk_inflation_query_cluster_count",
    "paired_call_overhead_mean",
    "paired_call_overhead_se",
    "paired_call_overhead_redraw_count",
    "paired_call_overhead_redraw_ci95_low",
    "paired_call_overhead_redraw_ci95_high",
    "paired_call_overhead_query_cluster_ci95_low",
    "paired_call_overhead_query_cluster_ci95_high",
    "paired_call_overhead_query_cluster_count",
    "o2xd2_hyperparameter_frequencies",
    "source_artifact_sha256",
    "source_data_hashes",
    "target_data_hashes",
    "source_shard_hashes",
    "bootstrap_repetitions",
    "bootstrap_seed",
    "test_retuning_used",
)


class TransferAnalysisError(ValueError):
    """Raised when a transfer record violates the analysis contract."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(paired.canonical_json_bytes(payload)).hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _se(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values) / math.sqrt(len(values)))


def wilson_interval(successes: int, trials: int) -> tuple[float | None, float | None, float | None]:
    """Return the rate and two-sided 95 percent Wilson interval."""
    if successes < 0 or trials < 0 or successes > trials:
        raise TransferAnalysisError("invalid Wilson event counts")
    if trials == 0:
        return None, None, None
    proportion = successes / trials
    z2 = WILSON_Z * WILSON_Z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    radius = (
        WILSON_Z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z2 / (4.0 * trials * trials)
        )
        / denominator
    )
    return proportion, max(0.0, center - radius), min(1.0, center + radius)


def paired_redraw_interval(
    values: Sequence[float], *, seed: int, repetitions: int = BOOTSTRAP_REPETITIONS
) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval with redraw as the resampling unit."""
    if repetitions < 2:
        raise TransferAnalysisError("bootstrap repetitions must be at least two")
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        value = float(array[0])
        return value, value
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 1024):
        size = min(1024, repetitions - start)
        indices = generator.integers(0, len(array), size=(size, len(array)))
        estimates[start : start + size] = array[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def query_cluster_interval(
    observations: Sequence[tuple[str, float]],
    *,
    seed: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> tuple[float | None, float | None, int]:
    """Percentile bootstrap interval clustered by stable query ID."""
    if repetitions < 2:
        raise TransferAnalysisError("bootstrap repetitions must be at least two")
    if not observations:
        return None, None, 0
    sums: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for query_id, value in observations:
        sums[str(query_id)] += float(value)
        counts[str(query_id)] += 1
    query_ids = sorted(sums)
    cluster_sums = np.asarray([sums[query_id] for query_id in query_ids], dtype=float)
    cluster_counts = np.asarray([counts[query_id] for query_id in query_ids], dtype=float)
    if len(query_ids) == 1:
        value = float(cluster_sums[0] / cluster_counts[0])
        return value, value, 1
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 512):
        size = min(512, repetitions - start)
        indices = generator.integers(0, len(query_ids), size=(size, len(query_ids)))
        estimates[start : start + size] = (
            cluster_sums[indices].sum(axis=1)
            / cluster_counts[indices].sum(axis=1)
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high), len(query_ids)


def _calibration_record(
    record: Mapping[str, Any],
    *,
    source: str,
    alpha: float,
    method: str,
    mode: str,
    target_test_ids: Sequence[str],
    score_hash: str,
) -> dict[str, Any]:
    try:
        merger._validate_calibration(
            record,
            source=Path(source),
            alpha=alpha,
            method=method,
            mode=mode,
            target_test_ids=target_test_ids,
            score_hash=score_hash,
        )
    except ValueError as error:
        raise TransferAnalysisError(str(error)) from error
    return {
        "certified": bool(record["certified"]),
        "calls": record["target_test_mean_calls"],
        "risk": record["target_test_all_miss_risk"],
        "calls_by_query": record["target_test_calls_by_query"],
        "miss_by_query": record["target_test_miss_by_query"],
        "test_event_count": record["target_test_event_count"],
        "test_trial_count": record["target_test_trial_count"],
        "calibration_event_count": int(record["calibration_event_count"]),
        "calibration_trial_count": int(record["calibration_trial_count"]),
    }


def _arm_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    certified = [record for record in records if record["certified"]]
    calls = [float(record["calls"]) for record in certified]
    risks = [float(record["risk"]) for record in certified]
    count = len(certified)
    rate, low, high = wilson_interval(count, len(records))
    return {
        "certification_count": count,
        "certification_total": len(records),
        "certification_rate": rate,
        "certification_wilson_ci95_low": low,
        "certification_wilson_ci95_high": high,
        "mean_calls": _mean(calls),
        "calls_se": _se(calls),
        "target_risk": _mean(risks),
        "target_risk_se": _se(risks),
        "target_event_count": sum(int(record["test_event_count"]) for record in certified),
        "target_trial_count": sum(int(record["test_trial_count"]) for record in certified),
        "calibration_event_count": sum(
            int(record["calibration_event_count"]) for record in records
        ),
        "calibration_trial_count": sum(
            int(record["calibration_trial_count"]) for record in records
        ),
    }


def _frequency_table(records: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    if method != paired.CHAIN_DISCOUNT_METHOD:
        return []
    counts: Counter[str] = Counter()
    values: dict[str, Any] = {}
    for record in records:
        hyperparameter = record.get("chosen_hyperparameter")
        if not isinstance(hyperparameter, Mapping):
            raise TransferAnalysisError("O2xD2 record lacks its chosen hyperparameter")
        key = _canonical(hyperparameter)
        counts[key] += 1
        values[key] = dict(hyperparameter)
    total = len(records)
    return [
        {
            "hyperparameter": values[key],
            "count": counts[key],
            "frequency": counts[key] / total,
        }
        for key in sorted(counts, key=lambda item: _canonical(values[item]))
    ]


def _validate_no_test_retuning(payload: Mapping[str, Any]) -> None:
    fit_mode = payload["fit_mode"]
    for cell in payload["cells"]:
        for redraw in cell["redraws"]:
            access = redraw.get("fit_label_access")
            if not isinstance(access, Mapping):
                raise TransferAnalysisError("redraw lacks fit label access audit")
            if fit_mode == "source_fitted" and access.get("target_fit_label_roles") != []:
                raise TransferAnalysisError("source-fitted record used target fit labels")
            for method, record in redraw["methods"].items():
                origin = record.get("hyperparameter_selection_origin")
                if origin is None:
                    continue
                normalized = str(origin).lower()
                if "test" in normalized or not normalized.endswith(":validation"):
                    raise TransferAnalysisError(
                        f"method {method} hyperparameter was not fixed on source validation"
                    )


def _source_shard_hashes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = payload.get("source_shards", [])
    if not isinstance(values, list):
        raise TransferAnalysisError("source_shards must be a list")
    output = []
    for value in values:
        if not isinstance(value, Mapping):
            raise TransferAnalysisError("source_shards entries must be objects")
        output.append({"path": value.get("path"), "sha256": value.get("sha256")})
    return output


def analyze_payload(
    payload: Mapping[str, Any],
    *,
    source_id: str = "<memory>",
    source_sha256: str | None = None,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return deterministic per-alpha, per-method transfer summaries."""
    try:
        merger.validate_payload(payload, Path(source_id), require_one_redraw=False)
    except ValueError as error:
        raise TransferAnalysisError(str(error)) from error
    if payload["redraw_start"] != 0 or payload["redraw_end"] != transfer.REDRAW_COUNT:
        raise TransferAnalysisError("analysis requires all 20 registered redraws")
    if repetitions < 2:
        raise TransferAnalysisError("bootstrap repetitions must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TransferAnalysisError("bootstrap seed must be a nonnegative integer")
    _validate_no_test_retuning(payload)
    artifact_hash = source_sha256 or _payload_sha256(payload)
    shard_hashes = _source_shard_hashes(payload)
    rows: list[dict[str, Any]] = []

    for alpha_index, cell in enumerate(payload["cells"]):
        alpha = float(cell["alpha"])
        if len(cell["redraws"]) != transfer.REDRAW_COUNT:
            raise TransferAnalysisError(f"alpha={alpha} does not contain 20 redraws")
        redraw_ids = [redraw["redraw_id"] for redraw in cell["redraws"]]
        if redraw_ids != list(range(transfer.REDRAW_COUNT)):
            raise TransferAnalysisError(f"alpha={alpha} redraw pairing is not contiguous")
        for method_index, method in enumerate(payload["method_identities"]):
            source_records = []
            target_records = []
            method_records = []
            risk_inflation = []
            call_overhead = []
            risk_query_observations: list[tuple[str, float]] = []
            call_query_observations: list[tuple[str, float]] = []

            for redraw in cell["redraws"]:
                target_test_ids = redraw["target_split_ids"]["test"]
                method_record = redraw["methods"][method]
                method_records.append(method_record)
                score_hash = method_record["frozen_target_scores_sha256"]
                twins = method_record["calibrations"]
                source_record = _calibration_record(
                    twins["source_calibrated"],
                    source=source_id,
                    alpha=alpha,
                    method=method,
                    mode="source_calibrated",
                    target_test_ids=target_test_ids,
                    score_hash=score_hash,
                )
                target_record = _calibration_record(
                    twins["target_recalibrated"],
                    source=source_id,
                    alpha=alpha,
                    method=method,
                    mode="target_recalibrated",
                    target_test_ids=target_test_ids,
                    score_hash=score_hash,
                )
                source_records.append(source_record)
                target_records.append(target_record)

                if source_record["certified"]:
                    risk_inflation.append(float(source_record["risk"]) - alpha)
                    for query_id, miss in zip(
                        target_test_ids, source_record["miss_by_query"], strict=True
                    ):
                        risk_query_observations.append((query_id, float(miss) - alpha))
                if source_record["certified"] and target_record["certified"]:
                    call_overhead.append(
                        float(source_record["calls"]) - float(target_record["calls"])
                    )
                    for query_id, source_calls, target_calls in zip(
                        target_test_ids,
                        source_record["calls_by_query"],
                        target_record["calls_by_query"],
                        strict=True,
                    ):
                        call_query_observations.append(
                            (query_id, float(source_calls) - float(target_calls))
                        )

            source_summary = _arm_summary(source_records)
            target_summary = _arm_summary(target_records)
            row_seed = seed + alpha_index * 1000 + method_index * 10
            risk_redraw_low, risk_redraw_high = paired_redraw_interval(
                risk_inflation, seed=row_seed, repetitions=repetitions
            )
            risk_query_low, risk_query_high, risk_query_count = query_cluster_interval(
                risk_query_observations, seed=row_seed + 1, repetitions=repetitions
            )
            calls_redraw_low, calls_redraw_high = paired_redraw_interval(
                call_overhead, seed=row_seed + 2, repetitions=repetitions
            )
            calls_query_low, calls_query_high, calls_query_count = query_cluster_interval(
                call_query_observations, seed=row_seed + 3, repetitions=repetitions
            )
            row: dict[str, Any] = {
                "protocol": payload["protocol"],
                "fit_mode": payload["fit_mode"],
                "source_benchmark": payload["source_benchmark"],
                "target_benchmark": payload["target_benchmark"],
                "pool_size": payload["pool_size"],
                "pool_seed": payload["pool_seed"],
                "alpha": alpha,
                "method": method,
                "redraw_count": transfer.REDRAW_COUNT,
            }
            for mode, summary in (
                ("source_calibrated", source_summary),
                ("target_recalibrated", target_summary),
            ):
                for name, value in summary.items():
                    row[f"{mode}_{name}"] = value
            row.update({
                "paired_risk_inflation_mean": _mean(risk_inflation),
                "paired_risk_inflation_se": _se(risk_inflation),
                "paired_risk_inflation_redraw_count": len(risk_inflation),
                "paired_risk_inflation_redraw_ci95_low": risk_redraw_low,
                "paired_risk_inflation_redraw_ci95_high": risk_redraw_high,
                "paired_risk_inflation_query_cluster_ci95_low": risk_query_low,
                "paired_risk_inflation_query_cluster_ci95_high": risk_query_high,
                "paired_risk_inflation_query_cluster_count": risk_query_count,
                "paired_call_overhead_mean": _mean(call_overhead),
                "paired_call_overhead_se": _se(call_overhead),
                "paired_call_overhead_redraw_count": len(call_overhead),
                "paired_call_overhead_redraw_ci95_low": calls_redraw_low,
                "paired_call_overhead_redraw_ci95_high": calls_redraw_high,
                "paired_call_overhead_query_cluster_ci95_low": calls_query_low,
                "paired_call_overhead_query_cluster_ci95_high": calls_query_high,
                "paired_call_overhead_query_cluster_count": calls_query_count,
                "o2xd2_hyperparameter_frequencies": _frequency_table(
                    method_records, method
                ),
                "source_artifact_sha256": artifact_hash,
                "source_data_hashes": copy_json(payload["source_hashes"]),
                "target_data_hashes": copy_json(payload["target_hashes"]),
                "source_shard_hashes": copy_json(shard_hashes),
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": row_seed,
                "risk_inflation_definition": "source_calibrated_target_risk_minus_alpha",
                "call_overhead_definition": "source_calibrated_calls_minus_target_recalibrated_calls",
                "redraw_interval_method": "paired_redraw_percentile_bootstrap",
                "query_interval_method": "query_id_cluster_percentile_bootstrap",
                "test_retuning_used": False,
            })
            rows.append(row)

    return {
        "schema_version": "issue52_transfer_analysis_v1",
        "analysis_kind": "fit_and_calibration_transfer",
        "row_count": len(rows),
        "bootstrap": {
            "repetitions": repetitions,
            "seed": seed,
            "confidence_level": 0.95,
            "redraw_resampling_unit": "paired_redraw",
            "query_resampling_unit": "query_id_cluster",
        },
        "source_artifact": {"path": source_id, "sha256": artifact_hash},
        "source_data_hashes": copy_json(payload["source_hashes"]),
        "target_data_hashes": copy_json(payload["target_hashes"]),
        "source_shard_hashes": copy_json(shard_hashes),
        "risk_inflation_definition": "source_calibrated_target_risk_minus_alpha",
        "call_overhead_definition": "source_calibrated_calls_minus_target_recalibrated_calls",
        "test_retuning_used": False,
        "rows": rows,
    }


def copy_json(value: Any) -> Any:
    """Copy a JSON value while rejecting non-JSON floating-point values."""
    return json.loads(_canonical(value))


def write_json(payload: Mapping[str, Any], output: Path) -> None:
    paired.atomic_write_json(output, payload)


def write_csv(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for raw_row in rows:
                row = {}
                for column in CSV_COLUMNS:
                    value = raw_row.get(column)
                    row[column] = _canonical(value) if isinstance(value, (dict, list)) else value
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError(f"refusing to overwrite immutable result {output}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    arguments = parser.parse_args(argv)
    for output in (arguments.out_json, arguments.out_csv):
        if output.exists():
            raise SystemExit(f"refusing to overwrite immutable result {output}")
    try:
        payload = json.loads(arguments.input.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TransferAnalysisError("input must contain one JSON object")
        analysis = analyze_payload(
            payload,
            source_id=str(arguments.input),
            source_sha256=sha256_file(arguments.input),
        )
        write_json(analysis, arguments.out_json)
        write_csv(analysis["rows"], arguments.out_csv)
    except (FileExistsError, OSError, json.JSONDecodeError, TransferAnalysisError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote {arguments.out_json}")
    print(f"wrote {arguments.out_csv}")
    return 0


analyze = analyze_payload


if __name__ == "__main__":
    raise SystemExit(main())
