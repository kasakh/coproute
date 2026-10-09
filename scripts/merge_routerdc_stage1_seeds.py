"""Merge five RouterDC seed records into one 100-redraw cell record."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_paired_main_suite import atomic_write_json, recompute_aggregates

SEEDS = (9101, 9102, 9103, 9104, 9105)


def merge_records(paths: Sequence[Path]) -> dict[str, Any]:
    if len(paths) != len(SEEDS):
        raise ValueError("exactly five RouterDC seed records are required")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_seed = {int(record["routerdc_training_seed"]): record for record in records}
    if tuple(sorted(by_seed)) != SEEDS:
        raise ValueError("RouterDC training seeds are incomplete")
    first = by_seed[SEEDS[0]]
    identity_keys = (
        "schema_version", "protocol", "benchmark", "pool_id", "pool_size", "pool_seed",
        "base_seed", "target_grid", "query_ids", "model_ids", "sampled_model_column_indices",
    )
    for seed, record in by_seed.items():
        for key in identity_keys:
            if record[key] != first[key]:
                raise ValueError(f"seed {seed} differs on {key}")
    output = {key: copy.deepcopy(value) for key, value in first.items() if key not in {
        "cells", "aggregates", "routerdc_training_seed", "routerdc_score_provenance",
        "routerdc_score_hashes", "routerdc_score_call_count",
    }}
    cells = []
    for cell_index, alpha in enumerate(first["target_grid"]):
        redraws = []
        for seed in SEEDS:
            record = by_seed[seed]
            cell = record["cells"][cell_index]
            if float(cell["alpha"]) != float(alpha):
                raise ValueError("RouterDC seed alpha order differs")
            for redraw in cell["redraws"]:
                copied = copy.deepcopy(redraw)
                original = copied["redraw_id"]
                copied["redraw_id"] = f"seed{seed}_redraw{original}"
                copied["routerdc_training_seed"] = seed
                for method in copied["methods"].values():
                    method["routerdc_training_seed"] = seed
                redraws.append(copied)
        cells.append({"alpha": float(alpha), "redraws": redraws})
    output.update({
        "design": "ISSUE_63_RouterDC_seed_merged_Stage_1",
        "routerdc_training_seeds": list(SEEDS),
        "routerdc_seed_count": len(SEEDS),
        "redraws_per_seed": 20,
        "redraw_start": None,
        "redraw_end": None,
        "redraw_range": {"per_seed": 20, "training_seed_count": 5, "total": 100},
        "total_seed_redraw_records": 100,
        "routerdc_score_provenance_by_seed": {
            str(seed): by_seed[seed]["routerdc_score_provenance"] for seed in SEEDS
        },
        "cells": cells,
    })
    output["aggregates"] = recompute_aggregates(cells)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs=5, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    atomic_write_json(args.out, merge_records(args.inputs))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
