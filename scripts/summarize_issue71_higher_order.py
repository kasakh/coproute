"""Summarize ISSUE-71 higher-order crossings against immutable ISSUE-78 tally."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

TARGETS = (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
TALLY = "fixed_tally_factorized"
TETRA_D1 = "fixed_tetrachoric_normal_score_factorized"
TETRA_D2 = "fixed_tetrachoric_normal_score_chain_discounted"
VINE_D1 = "fixed_discrete_vine_tree_factorized"
VINE_D2 = "fixed_discrete_vine_tree_chain_discounted"
PAIRS = (
    ("tetrachoric_D1_minus_tally_D1", TETRA_D1, TALLY),
    ("vine_D1_minus_tally_D1", VINE_D1, TALLY),
    ("tetrachoric_D2_minus_own_D1", TETRA_D2, TETRA_D1),
    ("vine_D2_minus_own_D1", VINE_D2, VINE_D1),
)
REPETITIONS = 10000
SEED = 20260971


def atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n");handle.flush();os.fsync(handle.fileno());temporary=Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False, newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        handle.flush();os.fsync(handle.fileno());temporary=Path(handle.name)
    os.replace(temporary,path)


def load(root: Path, prefix: str, issue: int) -> dict[tuple[str,str,int,int],dict[str,Any]]:
    paths=sorted((root/"shards").glob(prefix+"*.json"))
    if len(paths)!=640:raise RuntimeError(f"expected 640 issue {issue} shards, found {len(paths)}")
    output={}
    for path in paths:
        payload=json.loads(path.read_text(encoding="utf-8"))
        if payload["issue"]!=issue:raise RuntimeError(f"issue mismatch {path}")
        if issue==71:
            redraw=payload["cells"][0]["redraws"][0]["redraw_id"]
        else:
            redraw=payload["cells"][0]["redraws"][0]["redraw_id"]
        key=(payload["protocol"],payload["benchmark"],int(payload["pool_size"]),int(redraw))
        if key in output:raise RuntimeError(f"duplicate shard {key}")
        output[key]=payload
    return output


def bootstrap(groups:list[tuple[np.ndarray,int]],seed:int)->dict[str,Any]:
    generator=np.random.default_rng(seed);draw=np.zeros(REPETITIONS);denominator=0;observed=0.0;clusters=0
    for values,n_queries in groups:
        index=generator.integers(0,len(values),size=(REPETITIONS,len(values)))
        draw+=values[index].sum(axis=1)*n_queries
        denominator+=len(values)*n_queries;observed+=float(values.sum())*n_queries;clusters+=len(values)
    draw/=denominator
    return {"mean_calls_difference":observed/denominator,"ci95":[float(np.quantile(draw,.025)),float(np.quantile(draw,.975))],"bootstrap_se":float(np.std(draw,ddof=1)),"paired_redraw_clusters":clusters,"paired_queries":denominator}


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue71-root",required=True,type=Path);parser.add_argument("--issue78-root",required=True,type=Path)
    parser.add_argument("--host",required=True);parser.add_argument("--stamp",required=True)
    parser.add_argument("--csv-out",required=True,type=Path);parser.add_argument("--json-out",required=True,type=Path)
    args=parser.parse_args()
    for path in (args.csv_out,args.json_out):
        if path.exists():raise SystemExit(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
    current=load(args.issue71_root,f"issue71_{args.host}_{args.stamp}_",71)
    reference=load(args.issue78_root,"issue78_reference_",78)
    if set(current)!=set(reference):raise RuntimeError("ISSUE-71/78 shard identities differ")
    cells:dict[tuple[str,str,int,float],dict[int,tuple[dict,dict]]]=defaultdict(dict)
    hyper=defaultdict(Counter);diagnostics=defaultdict(list)
    for key,payload in current.items():
        ref=reference[key]
        if payload["model_ids_sha256"]!=ref["model_ids_sha256"] or payload["query_ids_sha256"]!=ref["query_ids_sha256"]:
            raise RuntimeError(f"ISSUE-71/78 identity hash mismatch {key}")
        for index,alpha in enumerate(TARGETS):
            left=payload["cells"][index]["redraws"][0]
            right=ref["cells"][index]["redraws"][0]
            if left["split_ids"]!=right["split_ids"]:raise RuntimeError(f"split identity mismatch {key} {alpha}")
            cells[(key[0],key[1],key[2],alpha)][key[3]]=(left["methods"],right["methods"])
            for method in (TETRA_D1,VINE_D1):
                hyper[(method,alpha)][json.dumps(left["methods"][method]["chosen_hyper"],sort_keys=True)]+=1
                diagnostics[method].append(left["methods"][method]["fit_diagnostics"])
            for method in (TETRA_D2,VINE_D2):
                hyper[(method,alpha)][json.dumps(left["methods"][method]["chosen_hyper"],sort_keys=True)]+=1
    rows=[];pair_groups=defaultdict(list)
    for identity in sorted(cells):
        records=cells[identity]
        if set(records)!=set(range(20)):raise RuntimeError(f"redraw mismatch {identity}")
        for pair_index,(label,left_name,right_name) in enumerate(PAIRS):
            diffs=[];risk_diffs=[];n_queries=None;cert_left=cert_right=0
            for redraw in range(20):
                methods,ref_methods=records[redraw]
                left=methods[left_name]
                right=methods[right_name] if right_name in methods else ref_methods[right_name]
                cert_left+=int(left["certified"]);cert_right+=int(right["certified"])
                if left["certified"] and right["certified"]:
                    if left["test_query_ids"]!=right["test_query_ids"]:raise RuntimeError("paired query ids differ")
                    lv=np.asarray(left["test_calls_by_query"],dtype=float);rv=np.asarray(right["test_calls_by_query"],dtype=float)
                    if n_queries is None:n_queries=len(lv)
                    elif n_queries!=len(lv):raise RuntimeError("query count varies within cell")
                    diffs.append(float(np.mean(lv-rv)))
                    risk_diffs.append(float(left["standard_all_miss_risk"]-right["standard_all_miss_risk"]))
            values=np.asarray(diffs,dtype=float)
            mean=float(values.mean()) if len(values) else None
            row={"protocol":identity[0],"benchmark":identity[1],"pool_size":identity[2],"target_alpha":identity[3],"comparison":label,"left_certified":cert_left,"right_certified":cert_right,"paired_redraws":len(values),"paired_queries":len(values)*(n_queries or 0),"mean_calls_difference":mean,"mean_risk_difference":float(np.mean(risk_diffs)) if risk_diffs else None,"redraw_wins":int(np.count_nonzero(values<0)),"redraw_ties":int(np.count_nonzero(values==0)),"redraw_losses":int(np.count_nonzero(values>0))}
            rows.append(row)
            if len(values):pair_groups[label].append((values,int(n_queries)))
    aggregate={}
    for index,(label,_,_) in enumerate(PAIRS):
        result=bootstrap(pair_groups[label],SEED+index)
        matching=[row for row in rows if row["comparison"]==label and row["mean_calls_difference"] is not None]
        result.update({"cell_target_wins":sum(row["mean_calls_difference"]<0 for row in matching),"cell_target_ties":sum(row["mean_calls_difference"]==0 for row in matching),"cell_target_losses":sum(row["mean_calls_difference"]>0 for row in matching),"cell_target_count":len(matching)})
        aggregate[label]=result
    identifiability={}
    for method,values in diagnostics.items():
        unique={(d["n_design_rows"],d["n_models"],json.dumps({k:d[k] for k in d if k in {"ridge","laplace_smoothing"}},sort_keys=True)):d for d in values}.values()
        identifiability[method]={"records":len(values),"unique_fit_records":len(list(unique)),"minimum_rows_per_model":float(min(d["rows_per_model"] for d in values)),"maximum_rows_per_model":float(max(d["rows_per_model"] for d in values)),"mean_zero_contingency_cell_fraction":float(np.mean([d["zero_contingency_cell_fraction"] for d in values])) if method==VINE_D1 else None,"psd_strategies":sorted({d["psd_strategy"] for d in values}),"higher_vine_levels_fit":sorted({d.get("higher_vine_levels_fit") for d in values if "higher_vine_levels_fit" in d})}
    output={"schema_version":1,"issue":71,"host":args.host,"stamp":args.stamp,"source_issue78":"issue78_reference","cells":32,"redraws":640,"targets":list(TARGETS),"comparison_rows":len(rows),"aggregate":aggregate,"hyperparameter_frequencies":{f"{method}|{alpha}":dict(sorted(count.items())) for (method,alpha),count in sorted(hyper.items())},"identifiability":identifiability,"bootstrap_repetitions":REPETITIONS,"bootstrap_seed":SEED}
    atomic_csv(args.csv_out,rows);atomic_json(args.json_out,output);print(json.dumps(output["aggregate"],indent=2,sort_keys=True));print(json.dumps(identifiability,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
