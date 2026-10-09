"""Independently verify ISSUE-65 optimizer outputs."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import tempfile
from typing import Any
import numpy as np

TALLY="fixed_tally_factorized";EXACT="fixed_exact_minsum_factorized";BEAM="fixed_beam_minsum_factorized";SWAP="fixed_local_swap_minsum_factorized";PAIRS={"exact_minus_tally_pool11":EXACT,"beam_minus_tally":BEAM,"local_swap_minus_tally":SWAP}

def atomic(path:Path,value:Any)->None:
    with tempfile.NamedTemporaryFile("w",dir=path.parent,prefix="."+path.name+".",suffix=".tmp",delete=False) as h:json.dump(value,h,indent=2,sort_keys=True,allow_nan=False);h.write("\n");h.flush();os.fsync(h.fileno());tmp=Path(h.name)
    os.replace(tmp,path)

def choose(record:dict)->int:
    t=record["validation_trajectory"];cert=[x for x in t if x["validation_certified"]]
    row=min(cert,key=lambda x:(x["validation_calls"],x["validation_standard_all_miss_risk"],x["grid_index"])) if cert else min(t,key=lambda x:(x["validation_standard_all_miss_risk"],x["validation_calls"],x["grid_index"]))
    return int(row["grid_index"])

def load(root:Path,pattern:str)->dict:
    out={}
    for path in sorted((root/"shards").glob(pattern)):
        p=json.loads(path.read_text());r=p["cells"][0]["redraws"][0]["redraw_id"];key=(p["protocol"],p["benchmark"],p["pool_size"],r)
        if key in out:raise RuntimeError(f"duplicate {key}")
        out[key]=p
    if len(out)!=640:raise RuntimeError(f"expected 640 shards, found {len(out)}")
    return out

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",required=True,type=Path);p.add_argument("--reference",required=True,type=Path);p.add_argument("--host",required=True);p.add_argument("--stamp",required=True);p.add_argument("--summary",required=True,type=Path);p.add_argument("--out",required=True,type=Path);a=p.parse_args()
    if a.out.exists():raise SystemExit(f"refusing to overwrite {a.out}")
    current=load(a.root,f"issue65_{a.host}_{a.stamp}_*.json");reference=load(a.reference,"issue78_reference_*.json")
    if set(current)!=set(reference):raise RuntimeError("shard identities differ")
    endpoints=selection_checks=exact_redraws=exact_depth_checks=identity_checks=0;sums={k:[0.0,0] for k in PAIRS}
    for key,payload in current.items():
        ref=reference[key]
        if payload["host"]!=a.host or payload["stamp"]!=a.stamp or payload["model_ids_sha256"]!=ref["model_ids_sha256"] or payload["query_ids_sha256"]!=ref["query_ids_sha256"]:raise RuntimeError(f"identity mismatch {key}")
        identity_checks+=1;first=payload["cells"][0]["redraws"][0];audit=first["optimization_audit"]
        if audit["beam_greedy_endpoint_verified"] is not True or audit["swap_greedy_endpoint_verified"] is not True:raise RuntimeError("endpoint mismatch")
        endpoints+=2
        if payload["include_exact"]:
            e=audit["exact"];exact_redraws+=1
            if e["state_count"]!=2048 or e["exact_design_objective"]>e["greedy_design_objective"] or e["nested_objective_gap"]!=e["greedy_design_objective"]-e["exact_design_objective"]:raise RuntimeError("exact DP invariant mismatch")
            for row in e["fixed_depth_evaluation"]:
                if row["design_exact_misses"]>row["design_greedy_misses"]:raise RuntimeError("exact depth design optimum mismatch")
                exact_depth_checks+=1
        for index,cell in enumerate(payload["cells"]):
            row=cell["redraws"][0];refrow=ref["cells"][index]["redraws"][0]
            if row["split_ids"]!=refrow["split_ids"]:raise RuntimeError("split mismatch")
            for method,record in row["methods"].items():
                if record["final_calibration_count"]!=1 or record["chosen_hyper"]!=record["validation_trajectory"][choose(record)]["hyper"]:raise RuntimeError("selection mismatch")
                if method in {BEAM,SWAP} and record["greedy_endpoint_score_identity_verified"] is not True:raise RuntimeError("endpoint flag missing")
                selection_checks+=1
            for label,method in PAIRS.items():
                if method not in row["methods"]:continue
                left=row["methods"][method];right=refrow["methods"][TALLY]
                if left["certified"] and right["certified"]:
                    if left["test_query_ids"]!=right["test_query_ids"]:raise RuntimeError("paired ids differ")
                    lv=np.asarray(left["test_calls_by_query"],float);rv=np.asarray(right["test_calls_by_query"],float);sums[label][0]+=float(np.sum(lv-rv));sums[label][1]+=len(lv)
    summary=json.loads(a.summary.read_text());observed={}
    for label,(total,count) in sums.items():
        mean=total/count;expected=summary["aggregate"][label]["mean_calls_difference"]
        if not np.isclose(mean,expected,rtol=0,atol=1e-12):raise RuntimeError(f"aggregate mismatch {label}")
        observed[label]=mean
    output={"schema_version":1,"issue":65,"host":a.host,"stamp":a.stamp,"status":"pass","shards_checked":640,"identity_checks":identity_checks,"greedy_endpoint_checks":endpoints,"selection_checks":selection_checks,"exact_redraws":exact_redraws,"exact_depth_checks":exact_depth_checks,"aggregate_checks":observed}
    a.out.parent.mkdir(parents=True,exist_ok=True);atomic(a.out,output);print(json.dumps(output,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
