"""Evaluate Cascade Routing's honest no-response-signal degeneration.

Without an observable response-derived signal, the official cascade's state does not
change after a call. The policy is therefore a pre-response router. This runner uses
frozen target-held-out RouterDC probabilities as quality estimates, orders models by
predicted quality, and matches each reference arm's total test calls exactly. Labels
are read only after selection to evaluate standard all-miss risk.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_paired_main_suite import _load_task,atomic_write_json
from scripts.run_routerdc_stage1 import load_scores
SEEDS=(9101,9102,9103,9104,9105)
REFERENCES=("fixed_tally_factorized","fixed_tally_chain_discounted")
def allocate_depths(total_calls:int,query_ids:list[str],salt:str,pool_size:int)->np.ndarray:
 n=len(query_ids)
 if total_calls<n or total_calls>n*pool_size: raise ValueError("call budget outside one-to-pool range")
 base,extra=divmod(total_calls,n); depths=np.full(n,base,dtype=int)
 order=sorted(range(n),key=lambda i:hashlib.sha256(f"{salt}|{query_ids[i]}".encode()).digest())
 depths[np.asarray(order[:extra],dtype=int)]+=1
 if int(depths.sum())!=total_calls or np.any(depths<1) or np.any(depths>pool_size): raise RuntimeError("depth allocation failed")
 return depths
def evaluate(arguments:argparse.Namespace)->dict[str,Any]:
 merged=json.loads(arguments.merged.read_text())
 task=_load_task(arguments.protocol,arguments.benchmark,leaderboard=arguments.leaderboard,router_dir=arguments.router_dir,routerbench_pickle=arguments.routerbench_pickle,embedding_archive=arguments.embedding_archive,embedding_manifest=arguments.embedding_manifest)
 if merged['protocol']!=arguments.protocol or merged['benchmark']!=arguments.benchmark or int(merged['pool_size'])!=arguments.pool_size: raise ValueError("merged identity mismatch")
 columns=np.asarray(merged['sampled_model_column_indices'],dtype=int); correctness=np.asarray(task.correctness)[:,columns].astype(bool)
 if [task.model_ids[int(i)] for i in columns]!=merged['model_ids']: raise ValueError("model pool mismatch")
 qindex={str(q):i for i,q in enumerate(task.query_ids)}; probabilities={}; provenance={}
 for seed in SEEDS:
  directory=Path(merged['routerdc_score_provenance_by_seed'][str(seed)]['score_directory'])
  p,meta=load_scores(directory,task,arguments.pool_size,int(merged['pool_seed']),str(merged['routerdc_target_family']),seed)
  probabilities[seed]=p; provenance[str(seed)]=meta
 rows=[]; selection_label_reads=0
 for cell in merged['cells']:
  alpha=float(cell['alpha'])
  for redraw in cell['redraws']:
   match=re.fullmatch(r'seed(\d+)_redraw(\d+)',str(redraw['redraw_id']))
   if not match: raise ValueError("unexpected redraw identity")
   seed,redraw_id=map(int,match.groups()); test_ids=list(redraw['split_ids']['test']); idx=np.asarray([qindex[x] for x in test_ids],dtype=int)
   scores=probabilities[seed][idx]; tie=np.broadcast_to(np.arange(arguments.pool_size),(len(idx),arguments.pool_size)); orders=np.empty_like(tie)
   for i in range(len(idx)): orders[i]=np.lexsort((tie[i],-scores[i]))
   for reference in REFERENCES:
    ref=redraw['methods'][reference]
    if not ref['certified']:
     rows.append({'target_alpha':alpha,'training_seed':seed,'redraw_id':redraw_id,'reference_method':reference,'reference_certified':False,'matched_total_calls':None,'mean_calls':None,'standard_all_miss_risk':None,'standard_all_miss_events':None,'test_query_count':len(test_ids),'native_distribution_free_certificate':False,'policy_kind':'pre_response_router_degeneration','unsupported_native_cascade_reason':'no honest post_response signal in correctness release'})
     continue
    total=int(sum(int(x) for x in ref['test_calls_by_query'])); depths=allocate_depths(total,test_ids,f"{arguments.protocol}|{arguments.benchmark}|{arguments.pool_size}|{alpha}|{seed}|{redraw_id}|{reference}",arguments.pool_size)
    misses=[]
    for i,(row,depth) in enumerate(zip(idx,depths)):
     chosen=orders[i,:int(depth)]; misses.append(not bool(correctness[row,chosen].any()))
    selection_label_reads+=0
    if int(depths.sum())!=total: raise RuntimeError("matched call identity failed")
    rows.append({'target_alpha':alpha,'training_seed':seed,'redraw_id':redraw_id,'reference_method':reference,'reference_certified':True,'matched_total_calls':total,'mean_calls':float(depths.mean()),'reference_mean_calls':float(ref['test_calls']),'call_identity_difference':float(depths.mean()-float(ref['test_calls'])),'standard_all_miss_risk':float(np.mean(misses)),'reference_standard_all_miss_risk':float(ref['standard_all_miss_risk']),'risk_difference_cascade_minus_reference':float(np.mean(misses)-float(ref['standard_all_miss_risk'])),'standard_all_miss_events':int(sum(misses)),'test_query_count':len(test_ids),'native_distribution_free_certificate':False,'policy_kind':'pre_response_router_degeneration','quality_signal':'frozen_target_heldout_routerdc_probability','post_response_signal':None,'selection_uses_target_labels':False})
 aggregates=[]
 for alpha in sorted({x['target_alpha'] for x in rows}):
  for ref in REFERENCES:
   values=[x for x in rows if x['target_alpha']==alpha and x['reference_method']==ref]; cert=[x for x in values if x['reference_certified']]
   den=sum(x['test_query_count'] for x in cert); events=sum(x['standard_all_miss_events'] for x in cert)
   ref_events=sum(round(x['reference_standard_all_miss_risk']*x['test_query_count']) for x in cert)
   aggregates.append({'target_alpha':alpha,'reference_method':ref,'seed_redraw_count':len(values),'certified_seed_redraw_count':len(cert),'certification_rate':len(cert)/len(values),'mean_calls':sum(x['matched_total_calls'] for x in cert)/den if den else None,'standard_all_miss_risk':events/den if den else None,'standard_all_miss_events':events,'reference_standard_all_miss_risk':ref_events/den if den else None,'reference_standard_all_miss_events':ref_events,'test_query_count':den,'native_distribution_free_certificate':False,'policy_kind':'pre_response_router_degeneration'})
 return {'schema_version':1,'analysis_kind':'issue58_honest_pre_response_cascade_merit','protocol':arguments.protocol,'benchmark':arguments.benchmark,'pool_size':arguments.pool_size,'training_seeds':list(SEEDS),'redraws_per_seed':20,'target_grid':list(merged['target_grid']),'reference_methods':list(REFERENCES),'native_sequential_cascade_supported':False,'native_distribution_free_certificate':False,'unsupported_native_cascade_reason':'public correctness releases lack honestly observable response-level confidence, agreement, execution feedback, or answer representations','selection_uses_target_labels':False,'selection_target_label_read_count':selection_label_reads,'quality_signal':'frozen target-held-out RouterDC probability available before any response','cost_signal':'unit call cost with exact matched total-call budgets','score_provenance_by_seed':provenance,'rows':rows,'aggregates':aggregates}
def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--protocol',required=True,choices=('routereval','routerbench')); p.add_argument('--benchmark',required=True); p.add_argument('--pool-size',required=True,type=int); p.add_argument('--merged',required=True,type=Path); p.add_argument('--leaderboard',type=Path); p.add_argument('--router-dir',type=Path); p.add_argument('--routerbench-pickle',type=Path); p.add_argument('--embedding-archive',type=Path); p.add_argument('--embedding-manifest',type=Path); p.add_argument('--out',required=True,type=Path); return p.parse_args()
def main():
 a=parse_args()
 if a.out.exists(): raise SystemExit(f'refusing to overwrite {a.out}')
 atomic_write_json(a.out,evaluate(a)); print(a.out)
if __name__=='__main__': main()
