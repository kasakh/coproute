"""Independently check ISSUE-71 higher-order crossing outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

TETRA_D1="fixed_tetrachoric_normal_score_factorized";TETRA_D2="fixed_tetrachoric_normal_score_chain_discounted"
VINE_D1="fixed_discrete_vine_tree_factorized";VINE_D2="fixed_discrete_vine_tree_chain_discounted";TALLY="fixed_tally_factorized"
METHODS={TETRA_D1,TETRA_D2,VINE_D1,VINE_D2}
PAIRS={"tetrachoric_D1_minus_tally_D1":(TETRA_D1,TALLY),"vine_D1_minus_tally_D1":(VINE_D1,TALLY),"tetrachoric_D2_minus_own_D1":(TETRA_D2,TETRA_D1),"vine_D2_minus_own_D1":(VINE_D2,VINE_D1)}


def atomic(path:Path,value:Any)->None:
    with tempfile.NamedTemporaryFile("w",dir=path.parent,prefix="."+path.name+".",suffix=".tmp",delete=False) as h:
        json.dump(value,h,indent=2,sort_keys=True,allow_nan=False);h.write("\n");h.flush();os.fsync(h.fileno());tmp=Path(h.name)
    os.replace(tmp,path)


def chosen_index(record:dict[str,Any])->int:
    trajectory=record["validation_trajectory"]
    certified=[row for row in trajectory if row["validation_certified"]]
    if certified:
        selected=min(certified,key=lambda row:(row["validation_calls"],row["validation_standard_all_miss_risk"],row["grid_index"]))
    else:
        selected=min(trajectory,key=lambda row:(row["validation_standard_all_miss_risk"],row["validation_calls"],row["grid_index"]))
    return int(selected["grid_index"])


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",required=True,type=Path);parser.add_argument("--reference",required=True,type=Path);parser.add_argument("--host",required=True);parser.add_argument("--stamp",required=True);parser.add_argument("--summary",required=True,type=Path);parser.add_argument("--out",required=True,type=Path);args=parser.parse_args()
    if args.out.exists():raise SystemExit(f"refusing to overwrite {args.out}")
    current=sorted((args.root/"shards").glob(f"issue71_{args.host}_{args.stamp}_*.json"));reference=sorted((args.reference/"shards").glob("issue78_reference_*.json"))
    if len(current)!=640 or len(reference)!=640:raise RuntimeError("shard count mismatch")
    refs={}
    for path in reference:
        p=json.loads(path.read_text());r=p["cells"][0]["redraws"][0]["redraw_id"];refs[(p["protocol"],p["benchmark"],p["pool_size"],r)]=p
    seen=set();endpoint_checks=selection_checks=identity_checks=0;groups=defaultdict(list)
    for path in current:
        p=json.loads(path.read_text());r=p["cells"][0]["redraws"][0]["redraw_id"];key=(p["protocol"],p["benchmark"],p["pool_size"],r)
        if key in seen:raise RuntimeError(f"duplicate {key}")
        seen.add(key);ref=refs[key]
        if p["host"]!=args.host or p["stamp"]!=args.stamp or p["model_ids_sha256"]!=ref["model_ids_sha256"] or p["query_ids_sha256"]!=ref["query_ids_sha256"]:raise RuntimeError(f"identity mismatch {key}")
        identity_checks+=1
        for index,cell in enumerate(p["cells"]):
            row=cell["redraws"][0];refrow=ref["cells"][index]["redraws"][0]
            if row["split_ids"]!=refrow["split_ids"] or set(row["methods"])!=METHODS:raise RuntimeError(f"split/arm mismatch {key}")
            records=row["methods"]
            for name,record in records.items():
                if record["final_calibration_count"]!=1 or record["chosen_hyper"]!=record["validation_trajectory"][chosen_index(record)]["hyper"]:raise RuntimeError(f"selection mismatch {key} {name}")
                selection_checks+=1
                if name.startswith("fixed_tetrachoric") and record["fit_diagnostics"]["psd_strategy"]!="low_rank_svd_plus_positive_diagonal":raise RuntimeError("tetrachoric PSD mismatch")
                if name.startswith("fixed_discrete_vine") and record["fit_diagnostics"]["higher_vine_levels_fit"]!=0:raise RuntimeError("unidentified vine level fit")
                if name.endswith("_chain_discounted"):
                    if record["endpoint_score_identity_verified"] is not True or record["endpoint_result_identity_verified"] is not True:raise RuntimeError("endpoint flag mismatch")
                    endpoint_checks+=1
            for label,(left_name,right_name) in PAIRS.items():
                left=records[left_name];right=records[right_name] if right_name in records else refrow["methods"][right_name]
                if left["certified"] and right["certified"]:
                    if left["test_query_ids"]!=right["test_query_ids"]:raise RuntimeError("paired ids differ")
                    lv=np.asarray(left["test_calls_by_query"],float);rv=np.asarray(right["test_calls_by_query"],float)
                    groups[label].append((float(np.sum(lv-rv)),len(lv)))
    if set(refs)!=seen:raise RuntimeError("shard identity coverage mismatch")
    summary=json.loads(args.summary.read_text())
    observed={}
    for label,records in groups.items():
        mean=sum(value for value,_ in records)/sum(count for _,count in records);expected=summary["aggregate"][label]["mean_calls_difference"]
        if not np.isclose(mean,expected,rtol=0,atol=1e-12):raise RuntimeError(f"aggregate mismatch {label}")
        observed[label]=mean
    output={"schema_version":1,"issue":71,"host":args.host,"stamp":args.stamp,"status":"pass","shards_checked":len(seen),"identity_checks":identity_checks,"selection_checks":selection_checks,"endpoint_checks":endpoint_checks,"aggregate_checks":observed}
    args.out.parent.mkdir(parents=True,exist_ok=True);atomic(args.out,output);print(json.dumps(output,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
