"""Validate and merge ISSUE-50 paired-suite redraw shards.

Shard order is irrelevant.  Immutable task, pool, method, endpoint, source, and
checkpoint identities must match exactly.  Redraw ranges must be disjoint and
contiguous.  No shard aggregate is trusted: every aggregate is checked against
its raw query vectors and rebuilt after concatenation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_paired_main_suite import (  # noqa: E402
    BASE_SEED,
    SCHEMA_VERSION,
    SPLIT_STEP,
    atomic_write_json,
    canonical_json_bytes,
    recompute_aggregates,
)

_DYNAMIC_FIELDS = {
    "redraw_start",
    "redraw_end",
    "redraw_range",
    "cells",
    "aggregates",
}


def _canonical(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _plain_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _immutable_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    for field in _DYNAMIC_FIELDS:
        result.pop(field, None)
    return result


def _range(payload: Mapping[str, Any], source: Path) -> tuple[int, int]:
    metadata = payload.get("redraw_range")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source}: missing redraw_range")
    start = _plain_int(metadata.get("start_inclusive"), f"{source}: redraw start")
    end = _plain_int(metadata.get("end_exclusive"), f"{source}: redraw end")
    count = _plain_int(metadata.get("count"), f"{source}: redraw count")
    if start < 0 or end <= start or count != end - start:
        raise ValueError(f"{source}: invalid redraw range [{start}, {end})")
    if payload.get("redraw_start") != start or payload.get("redraw_end") != end:
        raise ValueError(f"{source}: top-level redraw bounds disagree with redraw_range")
    return start, end


def _validate_method_record(
    record: Mapping[str, Any],
    *,
    source: Path,
    method: str,
    test_ids: Sequence[str],
    model_ids: Sequence[str],
    full_pool: Mapping[str, Any],
) -> None:
    prefix = f"{source}: {method}"
    if record.get("test_query_ids") != list(test_ids):
        raise ValueError(f"{prefix}: test query identities differ from the split")
    endpoint = record.get("endpoint")
    if not isinstance(endpoint, dict) or endpoint.get("model_ids") != list(model_ids):
        raise ValueError(f"{prefix}: endpoint model identity differs from the full pool")
    if endpoint.get("reaches_full_pool") is not True:
        raise ValueError(f"{prefix}: endpoint does not reach the full pool")
    if record.get("endpoint_calibration_count") != full_pool.get("calibration_count"):
        raise ValueError(f"{prefix}: endpoint calibration denominator differs from full pool")
    if record.get("endpoint_calibration_misses") != full_pool.get("calibration_misses"):
        raise ValueError(f"{prefix}: endpoint calibration misses differ from full pool")

    certified = record.get("certified")
    if not isinstance(certified, bool):
        raise ValueError(f"{prefix}: certified must be boolean")
    calls = record.get("test_calls_by_query")
    misses = record.get("test_miss_by_query")
    selected = record.get("selected_model_ids_by_query")
    if not certified:
        if calls is not None or misses is not None or selected is not None:
            raise ValueError(f"{prefix}: non-certified record contains deployed query vectors")
        if record.get("test_calls") is not None or record.get("standard_all_miss_risk") is not None:
            raise ValueError(f"{prefix}: non-certified record contains deployed aggregates")
        return
    if not isinstance(calls, list) or not isinstance(misses, list) or not isinstance(selected, list):
        raise ValueError(f"{prefix}: certified record lacks query vectors")
    if not (len(calls) == len(misses) == len(selected) == len(test_ids)):
        raise ValueError(f"{prefix}: query vector lengths differ")
    model_set = set(model_ids)
    for index, (call, miss, chosen) in enumerate(zip(calls, misses, selected)):
        if isinstance(call, bool) or not isinstance(call, int) or not 0 <= call <= len(model_ids):
            raise ValueError(f"{prefix}: invalid call count at query {index}")
        if miss not in (0, 1) or isinstance(miss, bool):
            raise ValueError(f"{prefix}: invalid miss indicator at query {index}")
        if not isinstance(chosen, list) or len(chosen) != call:
            raise ValueError(f"{prefix}: selected IDs disagree with calls at query {index}")
        if len(set(chosen)) != len(chosen) or not set(chosen) <= model_set:
            raise ValueError(f"{prefix}: invalid selected model IDs at query {index}")
    mean_calls = sum(calls) / len(calls)
    risk = sum(misses) / len(misses)
    if abs(float(record.get("test_calls")) - mean_calls) > 1e-12:
        raise ValueError(f"{prefix}: mean calls disagree with query vector")
    if abs(float(record.get("standard_all_miss_risk")) - risk) > 1e-12:
        raise ValueError(f"{prefix}: standard risk disagrees with query vector")


def validate_payload(payload: Mapping[str, Any], source: Path) -> tuple[int, int]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{source}: expected schema_version {SCHEMA_VERSION}")
    start, end = _range(payload, source)
    query_ids = payload.get("query_ids")
    model_ids = payload.get("model_ids")
    methods = payload.get("method_identities")
    target_grid = payload.get("target_grid")
    if not isinstance(query_ids, list) or not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError(f"{source}: invalid query_ids")
    if not isinstance(model_ids, list) or not model_ids or len(model_ids) != len(set(model_ids)):
        raise ValueError(f"{source}: invalid model_ids")
    if not isinstance(methods, list) or not methods or len(methods) != len(set(methods)):
        raise ValueError(f"{source}: invalid method_identities")
    if not isinstance(target_grid, list) or not target_grid:
        raise ValueError(f"{source}: invalid target_grid")
    cells = payload.get("cells")
    if not isinstance(cells, list) or [cell.get("alpha") for cell in cells] != target_grid:
        raise ValueError(f"{source}: cells do not match target_grid")

    expected_redraws = list(range(start, end))
    query_set = set(query_ids)
    roles = {"design", "router", "final_calibration", "validation", "test"}
    reference_splits: dict[int, Any] = {}
    for cell in cells:
        redraws = cell.get("redraws")
        if not isinstance(redraws, list) or [row.get("redraw_id") for row in redraws] != expected_redraws:
            raise ValueError(f"{source}: invalid redraw identities at alpha {cell.get('alpha')}")
        for redraw in redraws:
            redraw_id = redraw["redraw_id"]
            expected_seed = int(payload.get("base_seed", BASE_SEED)) + SPLIT_STEP * redraw_id
            if redraw.get("split_seed") != expected_seed:
                raise ValueError(f"{source}: redraw {redraw_id} has an invalid split seed")
            split_ids = redraw.get("split_ids")
            if not isinstance(split_ids, dict) or set(split_ids) != roles:
                raise ValueError(f"{source}: redraw {redraw_id} has invalid split roles")
            flattened = [query_id for role in split_ids.values() for query_id in role]
            if len(flattened) != len(query_ids) or set(flattened) != query_set:
                raise ValueError(f"{source}: redraw {redraw_id} split is overlapping or incomplete")
            if redraw_id in reference_splits and split_ids != reference_splits[redraw_id]:
                raise ValueError(f"{source}: redraw {redraw_id} split differs across targets")
            reference_splits[redraw_id] = split_ids

            full_pool = redraw.get("full_pool")
            if not isinstance(full_pool, dict) or full_pool.get("model_ids") != model_ids:
                raise ValueError(f"{source}: redraw {redraw_id} full-pool identity differs")
            if full_pool.get("calibration_query_ids") != split_ids["final_calibration"]:
                raise ValueError(f"{source}: redraw {redraw_id} calibration IDs differ")
            if full_pool.get("test_query_ids") != split_ids["test"]:
                raise ValueError(f"{source}: redraw {redraw_id} test IDs differ")
            cal_misses = full_pool.get("calibration_miss_by_query")
            test_misses = full_pool.get("test_miss_by_query")
            if not isinstance(cal_misses, list) or len(cal_misses) != len(split_ids["final_calibration"]):
                raise ValueError(f"{source}: redraw {redraw_id} invalid full-pool calibration vector")
            if not isinstance(test_misses, list) or len(test_misses) != len(split_ids["test"]):
                raise ValueError(f"{source}: redraw {redraw_id} invalid full-pool test vector")
            if any(value not in (0, 1) or isinstance(value, bool) for value in cal_misses + test_misses):
                raise ValueError(f"{source}: redraw {redraw_id} full-pool vectors are not binary")
            if full_pool.get("calibration_misses") != sum(cal_misses) or full_pool.get("calibration_count") != len(cal_misses):
                raise ValueError(f"{source}: redraw {redraw_id} full-pool calibration aggregate differs")
            if full_pool.get("test_misses") != sum(test_misses) or full_pool.get("test_count") != len(test_misses):
                raise ValueError(f"{source}: redraw {redraw_id} full-pool test aggregate differs")

            method_records = redraw.get("methods")
            if not isinstance(method_records, dict) or sorted(method_records) != sorted(methods):
                raise ValueError(f"{source}: redraw {redraw_id} method identities differ")
            for method, record in method_records.items():
                if not isinstance(record, dict):
                    raise ValueError(f"{source}: redraw {redraw_id} method record is not an object")
                _validate_method_record(
                    record,
                    source=source,
                    method=method,
                    test_ids=split_ids["test"],
                    model_ids=model_ids,
                    full_pool=full_pool,
                )

    recomputed = recompute_aggregates(cells)
    if _canonical(recomputed) != _canonical(payload.get("aggregates")):
        raise ValueError(f"{source}: stored aggregates disagree with raw query records")
    return start, end


def merge_payloads(shards: Sequence[tuple[Path, Mapping[str, Any]]]) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one shard is required")
    validated = []
    for source, payload in shards:
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: top-level JSON must be an object")
        start, end = validate_payload(payload, source)
        validated.append((start, end, source, payload))
    validated.sort(key=lambda item: (item[0], item[1], str(item[2])))

    identity = _canonical(_immutable_identity(validated[0][3]))
    for _, _, source, payload in validated[1:]:
        if _canonical(_immutable_identity(payload)) != identity:
            raise ValueError(f"{source}: immutable identities differ from other shards")

    previous_end = validated[0][0]
    seen: set[int] = set()
    for start, end, source, _ in validated:
        overlap = sorted(seen.intersection(range(start, end)))
        if overlap:
            raise ValueError(f"{source}: duplicate redraws {overlap}")
        if start != previous_end:
            raise ValueError(f"{source}: range [{start}, {end}) leaves a gap after {previous_end}")
        seen.update(range(start, end))
        previous_end = end

    merged = copy.deepcopy(validated[0][3])
    start = validated[0][0]
    end = validated[-1][1]
    merged["redraw_start"] = start
    merged["redraw_end"] = end
    merged["redraw_range"] = {
        "start_inclusive": start,
        "end_exclusive": end,
        "count": end - start,
    }
    alpha_to_cell = {
        str(cell["alpha"]): {"alpha": cell["alpha"], "redraws": []}
        for cell in merged["cells"]
    }
    for _, _, _, payload in validated:
        for cell in payload["cells"]:
            alpha_to_cell[str(cell["alpha"])]["redraws"].extend(copy.deepcopy(cell["redraws"]))
    merged["cells"] = [alpha_to_cell[str(alpha)] for alpha in merged["target_grid"]]
    for cell in merged["cells"]:
        cell["redraws"].sort(key=lambda record: record["redraw_id"])
    merged["aggregates"] = recompute_aggregates(merged["cells"])
    validate_payload(merged, Path("<merged>"))
    return merged


def load_and_merge(paths: Sequence[str | Path]) -> dict[str, Any]:
    shards = []
    for value in paths:
        path = Path(value)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: could not read valid JSON: {error}") from error
        shards.append((path, payload))
    return merge_payloads(shards)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.out.exists():
        raise SystemExit(f"refusing to overwrite immutable result {arguments.out}")
    try:
        merged = load_and_merge(arguments.shards)
        atomic_write_json(arguments.out, merged)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
