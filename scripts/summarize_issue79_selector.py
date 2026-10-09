"""Summarize immutable ISSUE-79 shards with paired selector diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

TARGETS = (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
ROUTEREVAL_TASKS = (
    "ifeval", "musr", "bbh_boolean_expressions", "bbh_navigate",
    "gpqa_diamond", "math_algebra_hard", "math_intermediate_algebra_hard",
)
ROUTERBENCH_TASKS = ("mmlu", "hellaswag", "winogrande", "arc-challenge")
POOLS = (100, 200, 300, 500)
SELECTOR = "validation_selected_factorized"
TALLY = "fixed_tally_factorized"
ERROR = "fixed_error_correlation_factorized"
EXPECTED_METHODS = {SELECTOR, TALLY, ERROR}
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20260979


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp",
        delete=False, newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def mean_or_none(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)) if len(array) else None


def se_or_none(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else None


def bootstrap_ci(values: list[float], seed: int) -> list[float] | None:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return None
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(BOOTSTRAP_REPETITIONS, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def expected_cells() -> set[tuple[str, str, int]]:
    return {
        *(("routerbench", task, 11) for task in ROUTERBENCH_TASKS),
        *(("routereval", task, pool) for task in ROUTEREVAL_TASKS for pool in POOLS),
    }


def selected_bias(row: dict[str, Any]) -> tuple[float, float]:
    selected = row["selected_method"]
    family_row = next(
        item for item in row["family_selection_trajectory"] if item["method"] == selected
    )
    tuning = row["within_family_tuning"][selected]
    tune_row = next(
        item for item in tuning["trajectory"]
        if int(item["grid_index"]) == int(tuning["selected_grid_index"])
    )
    return (
        float(family_row["winner_bias_budget_calls"]),
        float(tune_row["winner_bias_budget_calls"]),
    )


def summarize(root: Path, host: str, stamp: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prefix = f"issue79_{host}_{stamp}"
    paths = sorted((root / "shards").glob(f"{prefix}_*.json"))
    if len(paths) != 640:
        raise RuntimeError(f"expected 640 shards, found {len(paths)}")
    grouped: dict[tuple[str, str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    source_hashes: dict[str, str] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["host"] != host or payload["stamp"] != stamp or payload["issue"] != 79:
            raise RuntimeError(f"identity mismatch in {path}")
        if payload["family_menu_size"] != 12 or payload["selection_target_free"] is not True:
            raise RuntimeError(f"selector protocol mismatch in {path}")
        cell = (payload["protocol"], payload["benchmark"], int(payload["pool_size"]))
        redraw = payload["redraws"][0]
        redraw_id = int(redraw["redraw_id"])
        if redraw_id in grouped[cell]:
            raise RuntimeError(f"duplicate redraw {cell} {redraw_id}")
        grouped[cell][redraw_id] = redraw
        source_hashes[str(path.relative_to(root))] = sha256(path)
    if set(grouped) != expected_cells():
        raise RuntimeError("cell registry mismatch")
    for cell, redraws in grouped.items():
        if set(redraws) != set(range(20)):
            raise RuntimeError(f"redraw registry mismatch for {cell}")

    rows: list[dict[str, Any]] = []
    global_picks: Counter[str] = Counter()
    family_biases: list[float] = []
    tuning_biases: list[float] = []
    for cell in sorted(grouped):
        redraws = [grouped[cell][index] for index in range(20)]
        picks = Counter(row["selected_method"] for row in redraws)
        global_picks.update(picks)
        for redraw in redraws:
            family_bias, tuning_bias = selected_bias(redraw)
            family_biases.append(family_bias)
            tuning_biases.append(tuning_bias)
        for target_index, alpha in enumerate(TARGETS):
            selector_records = []
            tally_records = []
            error_records = []
            for redraw in redraws:
                alpha_row = redraw["alpha_results"][target_index]
                if float(alpha_row["alpha"]) != alpha:
                    raise RuntimeError("target order mismatch")
                methods = alpha_row["methods"]
                if set(methods) != EXPECTED_METHODS:
                    raise RuntimeError("output arm mismatch")
                selector_records.append(methods[SELECTOR])
                tally_records.append(methods[TALLY])
                error_records.append(methods[ERROR])
            selector_cert = sum(record["certified"] for record in selector_records)
            tally_cert = sum(record["certified"] for record in tally_records)
            error_cert = sum(record["certified"] for record in error_records)
            pair_diffs: list[float] = []
            pair_risk_diffs: list[float] = []
            paired_queries = 0
            pair_wins = pair_ties = pair_losses = 0
            error_diffs: list[float] = []
            recovery_opportunities = recovery_successes = 0
            blowup_opportunities = blowup_avoided = 0
            selector_regrets_in_blowups: list[float] = []
            for redraw, selected, tally, error in zip(
                redraws, selector_records, tally_records, error_records
            ):
                if selected["certified"] and tally["certified"]:
                    selected_calls = np.asarray(selected["test_calls_by_query"], dtype=float)
                    tally_calls = np.asarray(tally["test_calls_by_query"], dtype=float)
                    if selected["test_query_ids"] != tally["test_query_ids"]:
                        raise RuntimeError("selector/tally test identities differ")
                    difference = float(np.mean(selected_calls - tally_calls))
                    pair_diffs.append(difference)
                    paired_queries += len(selected_calls)
                    pair_risk_diffs.append(
                        float(selected["standard_all_miss_risk"] - tally["standard_all_miss_risk"])
                    )
                    if difference < 0:
                        pair_wins += 1
                    elif difference > 0:
                        pair_losses += 1
                    else:
                        pair_ties += 1
                if error["certified"] and tally["certified"]:
                    error_difference = float(error["test_calls"] - tally["test_calls"])
                    error_diffs.append(error_difference)
                    if error_difference < 0:
                        recovery_opportunities += 1
                        if (
                            redraw["selected_method"] == ERROR
                            and selected["certified"]
                            and selected["test_calls"] < tally["test_calls"]
                        ):
                            recovery_successes += 1
                    if error_difference >= 2.0:
                        blowup_opportunities += 1
                        if redraw["selected_method"] != ERROR:
                            blowup_avoided += 1
                        if selected["certified"]:
                            selector_regrets_in_blowups.append(
                                float(selected["test_calls"] - tally["test_calls"])
                            )
            seed = BOOTSTRAP_SEED + 1000 * len(rows)
            interval = bootstrap_ci(pair_diffs, seed)
            rows.append({
                "protocol": cell[0],
                "benchmark": cell[1],
                "pool_size": cell[2],
                "target_alpha": alpha,
                "redraw_count": 20,
                "selector_certified_redraws": selector_cert,
                "tally_certified_redraws": tally_cert,
                "error_correlation_certified_redraws": error_cert,
                "paired_selector_tally_redraws": len(pair_diffs),
                "paired_selector_tally_queries": paired_queries,
                "selector_minus_tally_calls": mean_or_none(pair_diffs),
                "selector_minus_tally_calls_se_redraw": se_or_none(pair_diffs),
                "selector_minus_tally_calls_ci_low": (interval or [None, None])[0],
                "selector_minus_tally_calls_ci_high": (interval or [None, None])[1],
                "selector_minus_tally_risk": mean_or_none(pair_risk_diffs),
                "selector_redraw_wins": pair_wins,
                "selector_redraw_ties": pair_ties,
                "selector_redraw_losses": pair_losses,
                "error_minus_tally_calls": mean_or_none(error_diffs),
                "error_win_redraw_opportunities": recovery_opportunities,
                "error_win_recovered_by_selector": recovery_successes,
                "error_blowup_redraw_opportunities": blowup_opportunities,
                "error_blowups_avoided": blowup_avoided,
                "selector_max_regret_in_error_blowups": (
                    max(selector_regrets_in_blowups) if selector_regrets_in_blowups else None
                ),
                "selector_pick_counts_json": json.dumps(dict(sorted(picks.items())), sort_keys=True),
            })

    available_rows = [row for row in rows if row["selector_minus_tally_calls"] is not None]
    weighted_n = sum(row["paired_selector_tally_queries"] for row in available_rows)
    weighted_diff = (
        sum(row["selector_minus_tally_calls"] * row["paired_selector_tally_queries"] for row in available_rows)
        / weighted_n if weighted_n else None
    )
    error_win_cells = [
        row for row in rows
        if row["error_minus_tally_calls"] is not None and row["error_minus_tally_calls"] < 0
    ]
    recovered_cells = [
        row for row in error_win_cells
        if row["selector_minus_tally_calls"] is not None and row["selector_minus_tally_calls"] < 0
    ]
    error_blowup_cells = [
        row for row in rows
        if row["error_minus_tally_calls"] is not None and row["error_minus_tally_calls"] >= 2.0
    ]
    selector_wins = sum(row["selector_minus_tally_calls"] < 0 for row in available_rows)
    selector_ties = sum(row["selector_minus_tally_calls"] == 0 for row in available_rows)
    selector_losses = sum(row["selector_minus_tally_calls"] > 0 for row in available_rows)
    summary = {
        "schema_version": 1,
        "issue": 79,
        "host": host,
        "stamp": stamp,
        "cells": 32,
        "redraws": 640,
        "targets": list(TARGETS),
        "cell_target_rows": len(rows),
        "family_menu_size": 12,
        "family_bias_multiplier": float(np.sqrt(2.0 * np.log(12.0))),
        "selection_target_free": True,
        "selection_pick_counts": dict(sorted(global_picks.items())),
        "selection_pick_total": int(sum(global_picks.values())),
        "mean_selected_family_bias_budget_calls": mean_or_none(family_biases),
        "mean_selected_within_family_bias_budget_calls": mean_or_none(tuning_biases),
        "paired_query_count": weighted_n,
        "weighted_selector_minus_tally_calls": weighted_diff,
        "selector_cell_target_wins_ties_losses": [selector_wins, selector_ties, selector_losses],
        "selector_certified_redraw_total": int(sum(row["selector_certified_redraws"] for row in rows)),
        "tally_certified_redraw_total": int(sum(row["tally_certified_redraws"] for row in rows)),
        "total_method_redraw_targets": 32 * 20 * len(TARGETS),
        "error_correlation_win_cell_targets": len(error_win_cells),
        "selector_win_among_error_win_cell_targets": len(recovered_cells),
        "error_correlation_blowup_cell_targets": len(error_blowup_cells),
        "selector_error_blowup_avoidance_redraws": int(sum(row["error_blowups_avoided"] for row in rows)),
        "selector_error_blowup_opportunity_redraws": int(sum(row["error_blowup_redraw_opportunities"] for row in rows)),
        "maximum_selector_regret_calls": max(
            row["selector_minus_tally_calls"] for row in available_rows
        ) if available_rows else None,
        "minimum_selector_gain_calls": min(
            row["selector_minus_tally_calls"] for row in available_rows
        ) if available_rows else None,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "source_shard_count": len(paths),
        "source_shard_sha256": source_hashes,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--csv-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.csv_out, args.json_out):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    rows, summary = summarize(args.root, args.host, args.stamp)
    atomic_csv(args.csv_out, rows)
    atomic_json(args.json_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
