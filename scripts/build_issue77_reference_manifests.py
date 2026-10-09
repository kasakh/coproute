"""Extract selected order and endpoint identities from merged ISSUE-74 cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_issue77_selected_orders import DEFAULT_METHODS, result_signature
from scripts.run_paired_main_suite import atomic_write_json


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def build(source: Path, methods: Sequence[str]) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    selected = tuple(methods)
    if payload.get("issue") != "ISSUE-74" or payload.get("preserve_score_prefixes") is not True:
        raise ValueError(f"{source}: not a corrected ISSUE-74 merged cell")
    if not set(selected) <= set(payload["registered_methods"]):
        raise ValueError(f"{source}: selected method is absent")
    redraws: dict[str, dict[str, dict[str, Any]]] = {}
    for cell in payload["cells"]:
        alpha = str(cell["alpha"])
        for redraw in cell["redraws"]:
            redraw_id = str(redraw["redraw_id"])
            redraws.setdefault(redraw_id, {})[alpha] = {}
            for method in selected:
                record = redraw["methods"][method]
                order = record.get("chosen_order")
                if not isinstance(order, list) or len(order) != payload["pool_size"]:
                    raise ValueError(f"{source}: {method} redraw {redraw_id} alpha {alpha} lacks full order")
                redraws[redraw_id][alpha][method] = {
                    "chosen_hyper": record.get("chosen_hyper"),
                    "chosen_order": order,
                    "result_signature": result_signature(record),
                }
    result = {
        "schema_version": "issue77_reference_manifest_v1",
        "issue": "ISSUE-77",
        "source_issue": "ISSUE-74",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "host": payload["host"],
        "utc_stamp": payload["utc_stamp"],
        "protocol": payload["protocol"],
        "benchmark": payload["benchmark"],
        "pool_size": payload["pool_size"],
        "pool_seed": payload["pool_seed"],
        "base_seed": payload["base_seed"],
        "knn_k": payload["knn_k"],
        "query_ids_sha256": payload["query_ids_sha256"],
        "model_ids_sha256": payload["model_ids_sha256"],
        "selected_methods": list(selected),
        "target_grid": payload["target_grid"],
        "redraw_start": payload["redraw_start"],
        "redraw_end": payload["redraw_end"],
        "redraws": redraws,
    }
    result["manifest_sha256"] = _hash_json(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing output directory {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    for source in args.sources:
        payload = build(source, methods)
        name = (
            f"issue77_{payload['host']}_{payload['utc_stamp']}_reference_"
            f"{payload['protocol']}_{payload['benchmark']}_p{payload['pool_size']}.json"
        )
        atomic_write_json(args.out_dir / name, payload)
    print(f"wrote {len(args.sources)} manifests to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
