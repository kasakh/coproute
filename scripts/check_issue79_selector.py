"""Independently verify ISSUE-79 selector shards and aggregate tables."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

TARGETS = (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
SELECTOR = "validation_selected_factorized"
TALLY = "fixed_tally_factorized"
ERROR = "fixed_error_correlation_factorized"
FAMILY_MULTIPLIER = math.sqrt(2.0 * math.log(12.0))


def close(left: float | None, right: float | None, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def number(value: str) -> float | None:
    return None if value == "" else float(value)


def atomic(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if len(rows) != 224:
        raise RuntimeError(f"expected 224 CSV rows, found {len(rows)}")
    row_map = {
        (row["protocol"], row["benchmark"], int(row["pool_size"]), float(row["target_alpha"])): row
        for row in rows
    }
    if len(row_map) != 224:
        raise RuntimeError("duplicate CSV row identity")
    paths = sorted((args.root / "shards").glob(f"issue79_{args.host}_{args.stamp}_*.json"))
    if len(paths) != 640:
        raise RuntimeError(f"expected 640 shards, found {len(paths)}")

    grouped: dict[tuple[str, str, int, float], list[tuple[dict, dict, dict, dict]]] = defaultdict(list)
    picks: Counter[str] = Counter()
    split_checks = penalty_checks = target_checks = calibration_checks = 0
    seen_redraws: set[tuple[str, str, int, int]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["host"] != args.host or payload["stamp"] != args.stamp:
            raise RuntimeError(f"identity mismatch {path}")
        if payload["family_menu_size"] != 12 or not close(payload["family_bias_multiplier"], FAMILY_MULTIPLIER):
            raise RuntimeError(f"family correction mismatch {path}")
        redraw = payload["redraws"][0]
        identity = (
            payload["protocol"], payload["benchmark"], int(payload["pool_size"]),
            int(redraw["redraw_id"]),
        )
        if identity in seen_redraws:
            raise RuntimeError(f"duplicate shard identity {identity}")
        seen_redraws.add(identity)
        split_ids = {name: set(values) for name, values in redraw["split_ids"].items()}
        tune = set(redraw["validation_partition"]["validation_tune_query_ids"])
        select = set(redraw["validation_partition"]["validation_select_query_ids"])
        if tune & select or tune | select != split_ids["validation"]:
            raise RuntimeError(f"invalid validation partition {identity}")
        if (tune | select) & (split_ids["final_calibration"] | split_ids["test"] | split_ids["design"] | split_ids["router"]):
            raise RuntimeError(f"selection/query role overlap {identity}")
        split_checks += 1

        family_rows = redraw["family_selection_trajectory"]
        if len(family_rows) != 12 or len({row["method"] for row in family_rows}) != 12:
            raise RuntimeError(f"family trajectory mismatch {identity}")
        for row in family_rows:
            if not close(row["bias_multiplier"], FAMILY_MULTIPLIER):
                raise RuntimeError(f"family multiplier mismatch {identity}")
            expected = row["mean_discovery_calls"] + FAMILY_MULTIPLIER * row["discovery_calls_se"]
            if not close(row["penalized_discovery_calls"], expected):
                raise RuntimeError(f"family penalty mismatch {identity}")
            penalty_checks += 1
        expected_family = min(
            family_rows,
            key=lambda row: (
                row["penalized_discovery_calls"], row["mean_discovery_calls"],
                0 if row["method"] == TALLY else 1, row["registry_index"],
            ),
        )["method"]
        if redraw["selected_method"] != expected_family:
            raise RuntimeError(f"selected family is not penalized minimum {identity}")
        if redraw["selected_method"] in payload["random_control_methods"]:
            raise RuntimeError(f"random control selected {identity}")
        picks[redraw["selected_method"]] += 1

        for method, tuning in redraw["within_family_tuning"].items():
            effective = int(tuning["effective_candidate_count"])
            multiplier = math.sqrt(2.0 * math.log(effective)) if effective > 1 else 0.0
            if not close(tuning["bias_multiplier"], multiplier):
                raise RuntimeError(f"within-family multiplier mismatch {identity} {method}")
            trajectory = tuning["trajectory"]
            for row in trajectory:
                expected = row["mean_discovery_calls"] + multiplier * row["discovery_calls_se"]
                if not close(row["penalized_discovery_calls"], expected):
                    raise RuntimeError(f"within-family penalty mismatch {identity} {method}")
                penalty_checks += 1
            expected_index = min(
                trajectory,
                key=lambda row: (
                    row["penalized_discovery_calls"], row["mean_discovery_calls"], row["grid_index"],
                ),
            )["grid_index"]
            if int(tuning["selected_grid_index"]) != int(expected_index):
                raise RuntimeError(f"within-family selection mismatch {identity} {method}")

        sources = set()
        for target_index, alpha_row in enumerate(redraw["alpha_results"]):
            alpha = float(alpha_row["alpha"])
            if alpha != TARGETS[target_index]:
                raise RuntimeError(f"target order mismatch {identity}")
            methods = alpha_row["methods"]
            if set(methods) != {SELECTOR, TALLY, ERROR}:
                raise RuntimeError(f"output arms mismatch {identity}")
            for record in methods.values():
                if int(record["final_calibration_count"]) != 1:
                    raise RuntimeError(f"final calibration count mismatch {identity}")
                calibration_checks += 1
            sources.add(methods[SELECTOR]["source_order_method"])
            grouped[(identity[0], identity[1], identity[2], alpha)].append(
                (redraw, methods[SELECTOR], methods[TALLY], methods[ERROR])
            )
            target_checks += 1
        if sources != {redraw["selected_method"]}:
            raise RuntimeError(f"family changed across targets {identity}")

    if len(seen_redraws) != 640 or len(grouped) != 224:
        raise RuntimeError("coverage mismatch")

    selector_cert_total = tally_cert_total = 0
    weighted_n = 0
    weighted_sum = 0.0
    cell_wins = cell_ties = cell_losses = 0
    error_win_cells = selector_win_on_error_cells = error_blowup_cells = 0
    blowup_opportunities = blowups_avoided = 0
    compared_rows = 0
    for identity, records in grouped.items():
        if len(records) != 20:
            raise RuntimeError(f"expected 20 redraws for {identity}")
        csv_row = row_map[identity]
        selector_cert = sum(selected["certified"] for _, selected, _, _ in records)
        tally_cert = sum(tally["certified"] for _, _, tally, _ in records)
        error_cert = sum(error["certified"] for _, _, _, error in records)
        selector_cert_total += selector_cert
        tally_cert_total += tally_cert
        if selector_cert != int(csv_row["selector_certified_redraws"]):
            raise RuntimeError(f"selector certification mismatch {identity}")
        if tally_cert != int(csv_row["tally_certified_redraws"]):
            raise RuntimeError(f"tally certification mismatch {identity}")
        if error_cert != int(csv_row["error_correlation_certified_redraws"]):
            raise RuntimeError(f"error certification mismatch {identity}")
        diffs = []
        error_diffs = []
        query_count = 0
        recovery_opportunities = recovery_successes = 0
        local_blowups = local_avoided = 0
        for redraw, selected, tally, error in records:
            if selected["certified"] and tally["certified"]:
                if selected["test_query_ids"] != tally["test_query_ids"]:
                    raise RuntimeError(f"paired query mismatch {identity}")
                left = np.asarray(selected["test_calls_by_query"], dtype=float)
                right = np.asarray(tally["test_calls_by_query"], dtype=float)
                diffs.append(float(np.mean(left - right)))
                query_count += len(left)
            if error["certified"] and tally["certified"]:
                difference = float(error["test_calls"] - tally["test_calls"])
                error_diffs.append(difference)
                if difference < 0:
                    recovery_opportunities += 1
                    if redraw["selected_method"] == ERROR and selected["certified"] and selected["test_calls"] < tally["test_calls"]:
                        recovery_successes += 1
                if difference >= 2.0:
                    local_blowups += 1
                    if redraw["selected_method"] != ERROR:
                        local_avoided += 1
        mean_diff = float(np.mean(diffs)) if diffs else None
        mean_error = float(np.mean(error_diffs)) if error_diffs else None
        if not close(mean_diff, number(csv_row["selector_minus_tally_calls"])):
            raise RuntimeError(f"paired call difference mismatch {identity}")
        if not close(mean_error, number(csv_row["error_minus_tally_calls"])):
            raise RuntimeError(f"error call difference mismatch {identity}")
        if query_count != int(csv_row["paired_selector_tally_queries"]):
            raise RuntimeError(f"paired query count mismatch {identity}")
        if recovery_opportunities != int(csv_row["error_win_redraw_opportunities"]):
            raise RuntimeError(f"recovery opportunity mismatch {identity}")
        if recovery_successes != int(csv_row["error_win_recovered_by_selector"]):
            raise RuntimeError(f"recovery count mismatch {identity}")
        if local_blowups != int(csv_row["error_blowup_redraw_opportunities"]):
            raise RuntimeError(f"blowup opportunity mismatch {identity}")
        if local_avoided != int(csv_row["error_blowups_avoided"]):
            raise RuntimeError(f"blowup avoidance mismatch {identity}")
        blowup_opportunities += local_blowups
        blowups_avoided += local_avoided
        if mean_diff is not None:
            compared_rows += 1
            weighted_n += query_count
            weighted_sum += mean_diff * query_count
            if mean_diff < 0:
                cell_wins += 1
            elif mean_diff > 0:
                cell_losses += 1
            else:
                cell_ties += 1
        if mean_error is not None and mean_error < 0:
            error_win_cells += 1
            if mean_diff is not None and mean_diff < 0:
                selector_win_on_error_cells += 1
        if mean_error is not None and mean_error >= 2.0:
            error_blowup_cells += 1

    weighted = weighted_sum / weighted_n
    checks = {
        "selection_pick_counts": dict(sorted(picks.items())),
        "selection_pick_total": sum(picks.values()),
        "selector_certified_redraw_total": selector_cert_total,
        "tally_certified_redraw_total": tally_cert_total,
        "weighted_selector_minus_tally_calls": weighted,
        "selector_cell_target_wins_ties_losses": [cell_wins, cell_ties, cell_losses],
        "error_correlation_win_cell_targets": error_win_cells,
        "selector_win_among_error_win_cell_targets": selector_win_on_error_cells,
        "error_correlation_blowup_cell_targets": error_blowup_cells,
        "selector_error_blowup_opportunity_redraws": blowup_opportunities,
        "selector_error_blowup_avoidance_redraws": blowups_avoided,
    }
    for key, observed in checks.items():
        expected = summary[key]
        if isinstance(observed, float):
            valid = close(observed, float(expected))
        else:
            valid = observed == expected
        if not valid:
            raise RuntimeError(f"summary mismatch for {key}: {observed} != {expected}")
    output = {
        "schema_version": 1,
        "issue": 79,
        "host": args.host,
        "stamp": args.stamp,
        "status": "pass",
        "shards_checked": len(paths),
        "cell_target_rows_checked": compared_rows,
        "split_isolation_checks": split_checks,
        "penalty_arithmetic_checks": penalty_checks,
        "target_invariance_checks": target_checks,
        "final_calibration_checks": calibration_checks,
        "aggregate_checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic(args.out, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
