"""Merge deterministic Design B prime E2 redraw shards.

Each input must be an immutable E2 result with a disjoint range. Ranges must
form one contiguous interval. Raw redraw records are concatenated by global
redraw identity, while all derived aggregates are recomputed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import design_bprime_e2_run as e2  # noqa: E402


_DYNAMIC_TOP_LEVEL = {
    "redraws",
    "redraw_start",
    "redraw_end",
    "redraw_range",
    "per_redraw",
    "summary",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _immutable_configuration(payload: dict[str, Any]) -> dict[str, Any]:
    """Return every field that must match across shards."""

    immutable = copy.deepcopy(payload)
    for key in _DYNAMIC_TOP_LEVEL:
        immutable.pop(key, None)
    information = immutable.get("information_destruction_control")
    if isinstance(information, dict):
        information.pop("per_redraw", None)
        information.pop("summary", None)
    return immutable


def _require_plain_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _range_for(payload: dict[str, Any], source: Path) -> tuple[int, int]:
    metadata = payload.get("redraw_range")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source}: missing redraw_range metadata")
    start = _require_plain_int(
        metadata.get("start_inclusive"), f"{source}: redraw range start"
    )
    end = _require_plain_int(
        metadata.get("end_exclusive"), f"{source}: redraw range end"
    )
    count = _require_plain_int(metadata.get("count"), f"{source}: redraw range count")
    if start < 0 or end <= start or count != end - start:
        raise ValueError(f"{source}: invalid redraw range [{start}, {end})")
    if payload.get("redraw_start") != start:
        raise ValueError(f"{source}: redraw_start does not match redraw_range")
    if payload.get("redraw_end") != end:
        raise ValueError(f"{source}: redraw_end does not match redraw_range")
    if payload.get("redraws") != count:
        raise ValueError(f"{source}: redraws does not match redraw_range count")
    return start, end


def _validate_redraw_records(
    payload: dict[str, Any], source: Path, start: int, end: int
) -> None:
    expected_redraws = list(range(start, end))
    alphas = payload.get("alphas")
    if not isinstance(alphas, list) or not alphas:
        raise ValueError(f"{source}: alphas must be a nonempty list")
    alpha_keys = [f"{alpha}" for alpha in alphas]
    if len(set(alpha_keys)) != len(alpha_keys):
        raise ValueError(f"{source}: duplicate alpha identities")

    controls = payload.get("racer_controls")
    wrapper_arms = payload.get("wrapper_arms")
    if not isinstance(controls, list) or not isinstance(wrapper_arms, list):
        raise ValueError(f"{source}: missing arm metadata")
    expected_arms = controls + [f"wrapper_{arm}_tuned" for arm in wrapper_arms]
    if len(set(expected_arms)) != len(expected_arms):
        raise ValueError(f"{source}: duplicate arm identities")

    per_redraw = payload.get("per_redraw")
    if not isinstance(per_redraw, dict) or set(per_redraw) != set(alpha_keys):
        raise ValueError(f"{source}: per_redraw alpha keys do not match alphas")
    for alpha_key in alpha_keys:
        by_arm = per_redraw[alpha_key]
        if not isinstance(by_arm, dict) or set(by_arm) != set(expected_arms):
            raise ValueError(f"{source}: arm keys do not match at alpha {alpha_key}")
        for arm in expected_arms:
            records = by_arm[arm]
            if not isinstance(records, list) or len(records) != len(expected_redraws):
                raise ValueError(
                    f"{source}: {arm} at alpha {alpha_key} has the wrong record count"
                )
            identities = [record.get("redraw") for record in records]
            if identities != expected_redraws:
                raise ValueError(
                    f"{source}: {arm} at alpha {alpha_key} has invalid redraw identities"
                )

    information = payload.get("information_destruction_control")
    if not isinstance(information, dict):
        raise ValueError(f"{source}: missing information_destruction_control")
    permutation_count = _require_plain_int(
        information.get("requested_permutations_per_redraw"),
        f"{source}: requested permutation count",
    )
    if permutation_count < 0:
        raise ValueError(f"{source}: requested permutation count must be nonnegative")
    information_records = information.get("per_redraw")
    if not isinstance(information_records, list):
        raise ValueError(f"{source}: information per_redraw must be a list")
    if [record.get("redraw") for record in information_records] != expected_redraws:
        raise ValueError(f"{source}: invalid information redraw identities")

    seed = _require_plain_int(payload.get("seed"), f"{source}: seed")
    for redraw_record in information_records:
        redraw = redraw_record["redraw"]
        permutations = redraw_record.get("permutations")
        if not isinstance(permutations, list) or len(permutations) != permutation_count:
            raise ValueError(f"{source}: redraw {redraw} has the wrong permutation count")
        for index, permutation in enumerate(permutations):
            if permutation.get("permutation_index") != index:
                raise ValueError(
                    f"{source}: redraw {redraw} has invalid permutation indices"
                )
            expected_seed = e2._design_permutation_seed(seed, redraw, index)
            if permutation.get("permutation_seed") != expected_seed:
                raise ValueError(
                    f"{source}: redraw {redraw} permutation {index} has an invalid seed"
                )
            per_alpha = permutation.get("per_alpha")
            if not isinstance(per_alpha, dict) or set(per_alpha) != set(alpha_keys):
                raise ValueError(
                    f"{source}: redraw {redraw} permutation {index} has invalid alphas"
                )


def merge_payloads(
    shard_payloads: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Validate and merge parsed shard payloads into one deterministic payload."""

    if not shard_payloads:
        raise ValueError("at least one shard is required")

    validated = []
    for source, payload in shard_payloads:
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: top-level JSON value must be an object")
        start, end = _range_for(payload, source)
        _validate_redraw_records(payload, source, start, end)
        validated.append((start, end, source, payload))
    validated.sort(key=lambda item: (item[0], item[1], str(item[2])))

    immutable_reference = _canonical_json(_immutable_configuration(validated[0][3]))
    for _, _, source, payload in validated[1:]:
        if _canonical_json(_immutable_configuration(payload)) != immutable_reference:
            raise ValueError(f"{source}: immutable configuration differs from other shards")

    seen: set[int] = set()
    previous_end = validated[0][0]
    for start, end, source, _ in validated:
        identities = set(range(start, end))
        duplicates = seen.intersection(identities)
        if duplicates:
            raise ValueError(f"{source}: duplicate redraws {sorted(duplicates)}")
        if start != previous_end:
            raise ValueError(
                f"{source}: range [{start}, {end}) leaves a gap after {previous_end}"
            )
        seen.update(identities)
        previous_end = end

    merged = copy.deepcopy(validated[0][3])
    start = validated[0][0]
    end = validated[-1][1]
    count = end - start
    merged["redraws"] = count
    merged["redraw_start"] = start
    merged["redraw_end"] = end
    merged["redraw_range"] = {
        "start_inclusive": start,
        "end_exclusive": end,
        "count": count,
    }

    alphas = merged["alphas"]
    alpha_keys = [f"{alpha}" for alpha in alphas]
    arm_names = list(merged["per_redraw"][alpha_keys[0]])
    merged_per_redraw = {
        alpha_key: {arm: [] for arm in arm_names} for alpha_key in alpha_keys
    }
    merged_information_records = []
    for _, _, _, payload in validated:
        for alpha_key in alpha_keys:
            for arm in arm_names:
                merged_per_redraw[alpha_key][arm].extend(
                    copy.deepcopy(payload["per_redraw"][alpha_key][arm])
                )
        merged_information_records.extend(
            copy.deepcopy(payload["information_destruction_control"]["per_redraw"])
        )

    for alpha_key in alpha_keys:
        for arm in arm_names:
            merged_per_redraw[alpha_key][arm].sort(key=lambda record: record["redraw"])
    merged_information_records.sort(key=lambda record: record["redraw"])

    merged["per_redraw"] = merged_per_redraw
    merged["summary"] = e2.build_summary(merged_per_redraw, alphas)
    merged["information_destruction_control"]["per_redraw"] = (
        merged_information_records
    )
    merged["information_destruction_control"]["summary"] = (
        e2.build_design_permutation_summary(merged_information_records, alphas)
    )
    return merged


def load_and_merge(paths: list[Path]) -> dict[str, Any]:
    shard_payloads = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: could not read valid JSON: {error}") from error
        shard_payloads.append((path, payload))
    return merge_payloads(shard_payloads)


def write_merged_output(output: Path, payload: dict[str, Any]) -> None:
    """Write one canonical JSON object atomically without overwriting."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.out.exists():
        raise SystemExit(
            f"refusing to overwrite immutable result {arguments.out}"
        )
    try:
        merged = load_and_merge(arguments.shards)
        write_merged_output(arguments.out, merged)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
