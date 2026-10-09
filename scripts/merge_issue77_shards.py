"""Merge selected-order ISSUE-77 shards and summarize discounted gains."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_issue77_selected_orders import SCHEMA_VERSION
from scripts.run_paired_main_suite import atomic_write_json

DYNAMIC_FIELDS = {"redraw_start", "redraw_end", "cells"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in payload.items() if key not in DYNAMIC_FIELDS}


def _calls(record: Mapping[str, Any]) -> float | None:
    value = record.get("test_calls", record.get("calls_test"))
    return None if value is None else float(value)


def _risk(record: Mapping[str, Any]) -> float | None:
    value = record.get("standard_all_miss_risk")
    return None if value is None else float(value)


def _se(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None


def _ci(values: Sequence[float], seed: int) -> list[float] | None:
    array = np.asarray(values, dtype=float)
    if not len(array): return None
    generator = np.random.default_rng(seed)
    samples = array[generator.integers(0, len(array), size=(10000, len(array)))].mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def _validate(payload: Mapping[str, Any], source: Path) -> tuple[int, int]:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("issue") != "ISSUE-77":
        raise ValueError(f"{source}: wrong ISSUE-77 schema")
    if payload.get("preserve_score_prefixes") is not True:
        raise ValueError(f"{source}: prefix preservation missing")
    methods = payload.get("registered_methods")
    if not isinstance(methods, list) or len(methods) != 4:
        raise ValueError(f"{source}: expected four registered pair arms")
    start, end = int(payload.get("redraw_start", -1)), int(payload.get("redraw_end", -1))
    if start < 0 or end <= start:
        raise ValueError(f"{source}: invalid redraw range")
    for cell in payload.get("cells", []):
        rows = cell.get("redraws")
        if [row.get("redraw_id") for row in rows] != list(range(start, end)):
            raise ValueError(f"{source}: redraw identities differ")
        for row in rows:
            if set(row["methods"]) != set(methods):
                raise ValueError(f"{source}: registered arm presence mismatch")
            for method, record in row["methods"].items():
                if record.get("final_calibration_count") != 1:
                    raise ValueError(f"{source}: {method} final calibration count differs")
                if method.endswith("_chain_discounted"):
                    if record.get("endpoint_score_identity_verified") is not True:
                        raise ValueError(f"{source}: {method} score endpoint unverified")
                    if record.get("endpoint_result_identity_verified") is not True:
                        raise ValueError(f"{source}: {method} result endpoint unverified")
                elif record.get("endpoint_reference_manifest_verified") is not True:
                    raise ValueError(f"{source}: {method} ISSUE-74 reference unverified")
    return start, end


def summarize(cells: Sequence[Mapping[str, Any]], selected_methods: Sequence[str]) -> dict[str, Any]:
    output = {}
    for cell_index, cell in enumerate(cells):
        alpha = str(cell["alpha"]); pairs = {}
        for method_index, product in enumerate(selected_methods):
            chain = product.replace("_factorized", "_chain_discounted")
            call_diffs=[]; risk_diffs=[]; minima=[]; shrinkage=[]; certified=0
            for row in cell["redraws"]:
                a=row["methods"][chain]; b=row["methods"][product]
                ca,cb=_calls(a),_calls(b); ra,rb=_risk(a),_risk(b)
                if ca is not None and cb is not None:
                    call_diffs.append(ca-cb); certified+=1
                if ra is not None and rb is not None: risk_diffs.append(ra-rb)
                minima.append(int(a["chosen_hyper"]["minimum_survivor_count"]))
                shrinkage.append(int(a["chain_discount_profile"]["shrinkage_engaged_prefix_count"]))
            seed=20260920+1000*cell_index+method_index
            pairs[product]={
                "chain_method":chain,
                "redraw_count":len(cell["redraws"]),
                "paired_certified_redraw_count":certified,
                "paired_calls_chain_minus_product_mean":float(np.mean(call_diffs)) if call_diffs else None,
                "paired_calls_chain_minus_product_se":_se(call_diffs),
                "paired_calls_chain_minus_product_bootstrap_ci95":_ci(call_diffs,seed),
                "paired_risk_chain_minus_product_mean":float(np.mean(risk_diffs)) if risk_diffs else None,
                "paired_risk_chain_minus_product_bootstrap_ci95":_ci(risk_diffs,seed+500000),
                "chosen_minimum_survivor_count_distribution":{str(value):minima.count(value) for value in sorted(set(minima))},
                "mean_shrinkage_engaged_prefix_count":float(np.mean(shrinkage)),
            }
        output[alpha]={"pairs":pairs}
    return output


def merge(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths: raise ValueError("at least one shard is required")
    rows=[]
    for path in paths:
        payload=json.loads(path.read_text()); start,end=_validate(payload,path); rows.append((start,end,path,payload))
    rows.sort(key=lambda item:(item[0],item[1],str(item[2])))
    identity=_canonical(_identity(rows[0][3])); expected=rows[0][0]
    for start,end,path,payload in rows:
        if start!=expected: raise ValueError(f"{path}: redraw gap or overlap after {expected}")
        if _canonical(_identity(payload))!=identity: raise ValueError(f"{path}: immutable identity differs")
        expected=end
    merged=copy.deepcopy(rows[0][3]); merged["redraw_start"]=rows[0][0]; merged["redraw_end"]=rows[-1][1]
    by_alpha={str(alpha):{"alpha":alpha,"redraws":[]} for alpha in merged["target_grid"]}
    for _,_,_,payload in rows:
        for cell in payload["cells"]: by_alpha[str(cell["alpha"])]["redraws"].extend(copy.deepcopy(cell["redraws"]))
    merged["cells"]=[by_alpha[str(alpha)] for alpha in merged["target_grid"]]
    for cell in merged["cells"]: cell["redraws"].sort(key=lambda row:row["redraw_id"])
    _validate(merged,Path("<merged>"))
    merged["summary"]=summarize(merged["cells"],merged["selected_methods"])
    merged["source_shards"]=[str(path) for _,_,path,_ in rows]
    merged["artifact_manifest"]={"issue":"ISSUE-77","host":merged["host"],"utc_stamp":merged["utc_stamp"],"source_shard_count":len(paths),"redraw_count":merged["redraw_end"]-merged["redraw_start"],"registered_arm_count":len(merged["registered_methods"]),"target_count":len(merged["target_grid"])}
    return merged


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--out",required=True,type=Path); parser.add_argument("shards",nargs="+",type=Path); args=parser.parse_args()
    atomic_write_json(args.out,merge(args.shards)); print(f"wrote {args.out}"); return 0

if __name__=="__main__": raise SystemExit(main())
