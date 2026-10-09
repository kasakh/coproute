"""Summarize ISSUE-65 optimizer results against immutable ISSUE-78 tally."""

from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv,json,os
from pathlib import Path
import tempfile
from typing import Any
import numpy as np

TARGETS=(0.02,0.03,0.05,0.075,0.1,0.15,0.2);TALLY="fixed_tally_factorized";EXACT="fixed_exact_minsum_factorized";BEAM="fixed_beam_minsum_factorized";SWAP="fixed_local_swap_minsum_factorized";PAIRS=(("exact_minus_tally_pool11",EXACT),("beam_minus_tally",BEAM),("local_swap_minus_tally",SWAP));REPETITIONS=10000;SEED=20260965


def atomic_json(path:Path,value:Any)->None:
    with tempfile.NamedTemporaryFile("w",dir=path.parent,prefix="."+path.name+".",suffix=".tmp",delete=False) as h:json.dump(value,h,indent=2,sort_keys=True,allow_nan=False);h.write("\n");h.flush();os.fsync(h.fileno());tmp=Path(h.name)
    os.replace(tmp,path)


def atomic_csv(path:Path,rows:list[dict[str,Any]])->None:
    with tempfile.NamedTemporaryFile("w",dir=path.parent,prefix="."+path.name+".",suffix=".tmp",delete=False,newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);h.flush();os.fsync(h.fileno());tmp=Path(h.name)
    os.replace(tmp,path)


def load(root:Path,prefix:str,issue:int)->dict:
    paths=sorted((root/"shards").glob(prefix+"*.json"))
    if len(paths)!=640:raise RuntimeError(f"expected 640 issue {issue} shards, found {len(paths)}")
    out={}
    for path in paths:
        p=json.loads(path.read_text());r=p["cells"][0]["redraws"][0]["redraw_id"];key=(p["protocol"],p["benchmark"],p["pool_size"],r)
        if key in out:raise RuntimeError(f"duplicate {key}")
        out[key]=p
    return out


def bootstrap(groups:list[tuple[np.ndarray,int]],seed:int)->dict:
    g=np.random.default_rng(seed);draw=np.zeros(REPETITIONS);den=0;obs=0.0;clusters=0
    for values,n in groups:
        idx=g.integers(0,len(values),size=(REPETITIONS,len(values)));draw+=values[idx].sum(axis=1)*n;den+=len(values)*n;obs+=float(values.sum())*n;clusters+=len(values)
    draw/=den
    return {"mean_calls_difference":obs/den,"ci95":[float(np.quantile(draw,.025)),float(np.quantile(draw,.975))],"bootstrap_se":float(np.std(draw,ddof=1)),"paired_redraw_clusters":clusters,"paired_queries":den}


def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",required=True,type=Path);p.add_argument("--reference",required=True,type=Path);p.add_argument("--host",required=True);p.add_argument("--stamp",required=True);p.add_argument("--csv-out",required=True,type=Path);p.add_argument("--json-out",required=True,type=Path);args=p.parse_args()
    for path in (args.csv_out,args.json_out):
        if path.exists():raise SystemExit(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
    current=load(args.root,f"issue65_{args.host}_{args.stamp}_",65);reference=load(args.reference,"issue78_reference_",78)
    if set(current)!=set(reference):raise RuntimeError("ISSUE-65/78 identities differ")
    rows=[];groups=defaultdict(list);hyper=defaultdict(Counter);runtime=defaultdict(list);objective=defaultdict(list);exact_audits=[]
    for key,payload in current.items():
        ref=reference[key]
        if payload["model_ids_sha256"]!=ref["model_ids_sha256"] or payload["query_ids_sha256"]!=ref["query_ids_sha256"]:raise RuntimeError(f"hash mismatch {key}")
        first=payload["cells"][0]["redraws"][0]
        if payload["include_exact"]:
            exact_audits.append((key,first["optimization_audit"]["exact"]))
        for index,alpha in enumerate(TARGETS):
            row=payload["cells"][index]["redraws"][0];refrow=ref["cells"][index]["redraws"][0]
            if row["split_ids"]!=refrow["split_ids"]:raise RuntimeError(f"split mismatch {key}")
            for label,method in PAIRS:
                if method not in row["methods"]:continue
                left=row["methods"][method];right=refrow["methods"][TALLY]
                hyper[(method,alpha)][json.dumps(left["chosen_hyper"],sort_keys=True)]+=1
                runtime[method].append(float(left["optimizer_diagnostics"].get("elapsed_seconds",0.0)))
                if "objective" in left["optimizer_diagnostics"]:objective[method].append(float(left["optimizer_diagnostics"]["objective"]-row["optimization_audit"]["tally_design_objective"]))
                diffs=[];risk=[]
                if left["certified"] and right["certified"]:
                    if left["test_query_ids"]!=right["test_query_ids"]:raise RuntimeError("paired ids differ")
                    lv=np.asarray(left["test_calls_by_query"],float);rv=np.asarray(right["test_calls_by_query"],float);diffs=[float(np.mean(lv-rv))];risk=[float(left["standard_all_miss_risk"]-right["standard_all_miss_risk"])]
                    groups[(label,key[0],key[1],key[2],alpha)].append((diffs[0],len(lv)))
                rows.append({"protocol":key[0],"benchmark":key[1],"pool_size":key[2],"redraw":key[3],"target_alpha":alpha,"comparison":label,"left_certified":int(left["certified"]),"tally_certified":int(right["certified"]),"calls_difference":diffs[0] if diffs else None,"risk_difference":risk[0] if risk else None,"chosen_hyper_json":json.dumps(left["chosen_hyper"],sort_keys=True),"optimizer_elapsed_seconds":left["optimizer_diagnostics"].get("elapsed_seconds",0.0),"design_objective_difference":left["optimizer_diagnostics"].get("objective",row["optimization_audit"]["tally_design_objective"])-row["optimization_audit"]["tally_design_objective"]})
    aggregate={}
    for pair_index,(label,method) in enumerate(PAIRS):
        strata=[];cellmeans=[]
        bycell=defaultdict(list)
        for (lab,protocol,bench,pool,alpha),records in groups.items():
            if lab!=label:continue
            values=np.asarray([x[0] for x in records]);counts={x[1] for x in records}
            if len(counts)!=1:raise RuntimeError("query count varies")
            strata.append((values,counts.pop()));cellmeans.append(float(values.mean()));bycell[(protocol,bench,pool,alpha)]=values
        result=bootstrap(strata,SEED+pair_index);arr=np.asarray(cellmeans)
        result.update({"cell_target_count":len(arr),"cell_target_wins":int(np.sum(arr<0)),"cell_target_ties":int(np.sum(arr==0)),"cell_target_losses":int(np.sum(arr>0)),"mean_selected_optimizer_seconds":float(np.mean(runtime[method])) if runtime[method] else None,"median_selected_optimizer_seconds":float(np.median(runtime[method])) if runtime[method] else None,"mean_design_objective_difference":float(np.mean(objective[method])) if objective[method] else None})
        aggregate[label]=result
    depth=defaultdict(lambda:defaultdict(int));exact_gaps=[];exact_relative=[];exact_seconds=[];exact_order_changed=0
    for key,audit in exact_audits:
        exact_gaps.append(audit["nested_objective_gap"]);exact_relative.append(audit["nested_relative_gap"]);exact_seconds.append(audit["elapsed_seconds"]);exact_order_changed+=int(audit["nested_objective_gap"]>0)
        for record in audit["fixed_depth_evaluation"]:
            d=record["depth"]
            for split in ("design","validation","test"):
                depth[d][f"{split}_exact_misses"]+=record[f"{split}_exact_misses"]
                depth[d][f"{split}_greedy_misses"]+=record[f"{split}_greedy_misses"]
                depth[d][f"{split}_count"]+=record.get(f"{split}_count",record.get("test_count" if split=="test" else "validation_count",0)) if split!="design" else 1
    exact_summary={"redraws":len(exact_audits),"exact_order_strictly_improves_design_objective":exact_order_changed,"mean_nested_objective_gap":float(np.mean(exact_gaps)),"maximum_nested_objective_gap":int(max(exact_gaps)),"mean_relative_gap":float(np.mean(exact_relative)),"mean_exact_runtime_seconds":float(np.mean(exact_seconds)),"maximum_exact_runtime_seconds":float(np.max(exact_seconds)),"depth_rows":[{"depth":d,**values,"validation_miss_difference":values["validation_exact_misses"]-values["validation_greedy_misses"],"test_miss_difference":values["test_exact_misses"]-values["test_greedy_misses"]} for d,values in sorted(depth.items())]}
    output={"schema_version":1,"issue":65,"host":args.host,"stamp":args.stamp,"source_issue78":"issue78_reference","shards":640,"aggregate":aggregate,"pool11_exact":exact_summary,"hyperparameter_frequencies":{f"{m}|{a}":dict(sorted(c.items())) for (m,a),c in sorted(hyper.items())},"bootstrap_repetitions":REPETITIONS,"bootstrap_seed":SEED}
    atomic_csv(args.csv_out,rows);atomic_json(args.json_out,output);print(json.dumps(aggregate,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in exact_summary.items() if k!="depth_rows"},indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
