"""Merge the 20 immutable one-redraw ISSUE-52 transfer shards for one cell."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_issue52_transfer as transfer  # noqa: E402
from scripts import run_paired_main_suite as paired  # noqa: E402


DYNAMIC_FIELDS = {
    "redraw_start",
    "redraw_end",
    "redraw_range",
    "cells",
    "artifact_manifest",
    "source_shards",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key not in DYNAMIC_FIELDS
    }


def _validate_number(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _validate_calibration(
    calibration: Mapping[str, Any],
    *,
    source: Path,
    alpha: float,
    method: str,
    mode: str,
    target_test_ids: Sequence[str],
    score_hash: str,
) -> None:
    prefix = f"{source}: alpha={alpha}, method={method}, mode={mode}"
    if calibration.get("frozen_target_scores_sha256") != score_hash:
        raise ValueError(f"{prefix}: calibration twin score identity differs")
    if calibration.get("target_test_query_ids") != list(target_test_ids):
        raise ValueError(f"{prefix}: target test query identity differs")
    expected_id_hash = paired.hash_json(list(target_test_ids))
    if calibration.get("target_test_query_ids_sha256") != expected_id_hash:
        raise ValueError(f"{prefix}: target test query hash differs")
    trials = calibration.get("target_test_trial_count")
    if trials != len(target_test_ids):
        raise ValueError(f"{prefix}: target test trial count differs")
    certified = calibration.get("certified")
    if not isinstance(certified, bool):
        raise ValueError(f"{prefix}: certified must be boolean")
    calls = calibration.get("target_test_calls_by_query")
    misses = calibration.get("target_test_miss_by_query")
    events = calibration.get("target_test_event_count")
    mean_calls = calibration.get("target_test_mean_calls")
    risk = calibration.get("target_test_all_miss_risk")
    if not certified:
        if any(value is not None for value in (calls, misses, events, mean_calls, risk)):
            raise ValueError(f"{prefix}: non-certified result contains test estimates")
        return
    if not isinstance(calls, list) or not isinstance(misses, list):
        raise ValueError(f"{prefix}: certified result lacks query vectors")
    if len(calls) != trials or len(misses) != trials:
        raise ValueError(f"{prefix}: query vector length differs from test trials")
    if any(isinstance(value, bool) or int(value) != value or int(value) < 0 for value in calls):
        raise ValueError(f"{prefix}: calls must be nonnegative integers")
    if any(value not in (0, 1) or isinstance(value, bool) for value in misses):
        raise ValueError(f"{prefix}: misses must be binary integers")
    if events != sum(misses):
        raise ValueError(f"{prefix}: target event arithmetic differs")
    expected_calls = sum(calls) / trials
    expected_risk = sum(misses) / trials
    if not math.isclose(
        _validate_number(mean_calls, f"{prefix} mean calls"),
        expected_calls,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{prefix}: target mean calls arithmetic differs")
    if not math.isclose(
        _validate_number(risk, f"{prefix} risk"),
        expected_risk,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{prefix}: target risk arithmetic differs")


def validate_payload(
    payload: Mapping[str, Any], source: Path, *, require_one_redraw: bool = True
) -> tuple[int, int]:
    """Validate one shard or a fully merged ISSUE-52 payload."""
    if payload.get("schema_version") != transfer.SCHEMA_VERSION:
        raise ValueError(f"{source}: wrong ISSUE-52 schema version")
    if payload.get("design") != "ISSUE_52_fit_and_calibration_transfer":
        raise ValueError(f"{source}: wrong ISSUE-52 design")
    if payload.get("protocol") not in ("routereval", "routerbench"):
        raise ValueError(f"{source}: invalid protocol")
    if payload.get("fit_mode") not in transfer.FIT_MODES:
        raise ValueError(f"{source}: invalid fit mode")
    if not payload.get("source_benchmark") or not payload.get("target_benchmark"):
        raise ValueError(f"{source}: missing source or target benchmark")
    if payload.get("calibration_modes") != list(transfer.CALIBRATION_MODES):
        raise ValueError(f"{source}: calibration twin identities differ")
    if payload.get("method_identities") != list(transfer.registered_methods()):
        raise ValueError(f"{source}: registered method identities differ")
    if payload.get("preserve_prefixes") is not True:
        raise ValueError(f"{source}: prefix preservation is required")

    start = payload.get("redraw_start")
    end = payload.get("redraw_end")
    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError(f"{source}: invalid redraw range")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"{source}: invalid redraw range")
    if not 0 <= start < end <= transfer.REDRAW_COUNT:
        raise ValueError(f"{source}: redraw range lies outside the registered range")
    if require_one_redraw and end != start + 1:
        raise ValueError(f"{source}: every shard must contain exactly one redraw")
    redraw_range = payload.get("redraw_range")
    expected_range = {
        "start_inclusive": start,
        "end_exclusive": end,
        "count": end - start,
        "registered_total": transfer.REDRAW_COUNT,
    }
    if redraw_range != expected_range:
        raise ValueError(f"{source}: redraw range metadata differs")

    target_grid = payload.get("target_grid")
    if not isinstance(target_grid, list) or target_grid != list(paired.ALPHA_GRID):
        raise ValueError(f"{source}: target alpha grid differs")
    cells = payload.get("cells")
    if not isinstance(cells, list) or [cell.get("alpha") for cell in cells] != target_grid:
        raise ValueError(f"{source}: alpha cells differ from target grid")
    expected_redraws = list(range(start, end))
    expected_methods = set(payload["method_identities"])
    for cell in cells:
        alpha = float(cell["alpha"])
        redraws = cell.get("redraws")
        if not isinstance(redraws, list):
            raise ValueError(f"{source}: alpha={alpha} lacks redraw rows")
        if [row.get("redraw_id") for row in redraws] != expected_redraws:
            raise ValueError(f"{source}: alpha={alpha} redraw rows differ from range")
        for redraw in redraws:
            target_split = redraw.get("target_split_ids")
            if not isinstance(target_split, Mapping) or not isinstance(
                target_split.get("test"), list
            ):
                raise ValueError(f"{source}: alpha={alpha} lacks target test IDs")
            target_test_ids = target_split["test"]
            if len(target_test_ids) != len(set(target_test_ids)):
                raise ValueError(f"{source}: alpha={alpha} target test IDs are not unique")
            methods = redraw.get("methods")
            if not isinstance(methods, Mapping) or set(methods) != expected_methods:
                raise ValueError(f"{source}: alpha={alpha} registered method mismatch")
            for method in payload["method_identities"]:
                record = methods[method]
                if not isinstance(record, Mapping):
                    raise ValueError(f"{source}: alpha={alpha}, method={method} is invalid")
                score_hash = record.get("frozen_target_scores_sha256")
                if not isinstance(score_hash, str) or not score_hash:
                    raise ValueError(f"{source}: alpha={alpha}, method={method} lacks score hash")
                twins = record.get("calibrations")
                if not isinstance(twins, Mapping) or set(twins) != set(
                    transfer.CALIBRATION_MODES
                ):
                    raise ValueError(
                        f"{source}: alpha={alpha}, method={method} calibration twins differ"
                    )
                for mode in transfer.CALIBRATION_MODES:
                    calibration = twins[mode]
                    if not isinstance(calibration, Mapping):
                        raise ValueError(
                            f"{source}: alpha={alpha}, method={method}, mode={mode} is invalid"
                        )
                    _validate_calibration(
                        calibration,
                        source=source,
                        alpha=alpha,
                        method=method,
                        mode=mode,
                        target_test_ids=target_test_ids,
                        score_hash=score_hash,
                    )
    return start, end


def merge_payloads(
    shards: Sequence[tuple[Path, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Merge exactly 20 one-redraw shards in redraw order."""
    if len(shards) != transfer.REDRAW_COUNT:
        raise ValueError(
            f"ISSUE-52 requires exactly {transfer.REDRAW_COUNT} one-redraw shards"
        )
    validated: list[tuple[int, Path, Mapping[str, Any]]] = []
    for source, payload in shards:
        start, end = validate_payload(payload, source, require_one_redraw=True)
        validated.append((start, source, payload))
        if end != start + 1:
            raise AssertionError("one-redraw validation returned a wider range")
    validated.sort(key=lambda item: (item[0], str(item[1])))
    starts = [item[0] for item in validated]
    if starts != list(range(transfer.REDRAW_COUNT)):
        raise ValueError("ISSUE-52 redraw shards must be contiguous from 0 through 19")

    reference_identity = _canonical(_identity(validated[0][2]))
    for _, source, payload in validated[1:]:
        if _canonical(_identity(payload)) != reference_identity:
            raise ValueError(f"{source}: immutable ISSUE-52 identity differs")

    merged = copy.deepcopy(validated[0][2])
    merged["redraw_start"] = 0
    merged["redraw_end"] = transfer.REDRAW_COUNT
    merged["redraw_range"] = {
        "start_inclusive": 0,
        "end_exclusive": transfer.REDRAW_COUNT,
        "count": transfer.REDRAW_COUNT,
        "registered_total": transfer.REDRAW_COUNT,
    }
    by_alpha = {
        str(alpha): {"alpha": alpha, "redraws": []}
        for alpha in merged["target_grid"]
    }
    for _, _, payload in validated:
        for cell in payload["cells"]:
            by_alpha[str(cell["alpha"])]["redraws"].extend(
                copy.deepcopy(cell["redraws"])
            )
    merged["cells"] = [by_alpha[str(alpha)] for alpha in merged["target_grid"]]
    merged["source_shards"] = [
        {"path": str(source), "sha256": _sha256(source) if source.is_file() else None}
        for _, source, _ in validated
    ]
    merged["artifact_manifest"] = {
        "kind": "issue52_merged_transfer_cell",
        "source_shard_count": len(validated),
        "redraw_count": transfer.REDRAW_COUNT,
        "alpha_count": len(merged["target_grid"]),
        "method_count": len(merged["method_identities"]),
    }
    validate_payload(merged, Path("<merged>"), require_one_redraw=False)
    return merged


def load_and_merge(paths: Sequence[Path]) -> dict[str, Any]:
    shards = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: could not read valid JSON") from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}: shard must contain one JSON object")
        shards.append((path, payload))
    return merge_payloads(shards)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.out.exists():
        raise SystemExit(f"refusing to overwrite immutable result {arguments.out}")
    try:
        merged = load_and_merge(arguments.shards)
        paired.atomic_write_json(arguments.out, merged)
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
