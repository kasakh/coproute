"""Replay RACER-G/P Stage 1 routing to audit empty and tied-block events."""
from __future__ import annotations
import argparse,json,os
from collections import Counter
from pathlib import Path
import sys
from typing import Sequence
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from components.racer_protocol import nonconformity_scores,raw_nonconformity_scores,select_standard_crc_threshold
from components.routereval_protocol import subsample_pool
from scripts.run_paired_main_suite import ALPHA_GRID,BASE_SEED,SPLIT_STEP,_knn,_load_task,_target_split,hash_json

def main(argv:Sequence[str]|None=None)->int:
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',choices=('routereval','routerbench'),required=True);ap.add_argument('--leaderboard',type=Path);ap.add_argument('--router-dir',type=Path);ap.add_argument('--routerbench-pickle',type=Path);ap.add_argument('--embedding-archive',type=Path);ap.add_argument('--embedding-manifest',type=Path);ap.add_argument('--benchmark',required=True);ap.add_argument('--pool-size',type=int,required=True);ap.add_argument('--host',required=True);ap.add_argument('--stamp',required=True);ap.add_argument('--worker-count',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise SystemExit(f'collision {a.out}')
 task=_load_task(a.protocol,a.benchmark,leaderboard=a.leaderboard,router_dir=a.router_dir,routerbench_pickle=a.routerbench_pickle,embedding_archive=a.embedding_archive,embedding_manifest=a.embedding_manifest)
 columns=subsample_pool(len(task.model_ids),a.pool_size,seed=7001);labels=np.asarray(task.correctness[:,columns],dtype=np.int8);emb=np.asarray(task.embeddings,float)
 result={variant:{str(alpha):{'certified_redraws':0,'noncertified_redraws':0,'noncertification_reasons':Counter(),'deployed_test_queries':0,'empty_real_sets':0,'null_selected_diagnostic':0,'real_calls':0,'tied_blocks':0,'fully_admitted_tied_blocks':0,'partially_admitted_tied_blocks':0,'unadmitted_tied_blocks':0,'queries_with_full_tied_block':0,'queries_with_partial_tied_block':0,'full_admitted_tied_block_sizes':[]} for alpha in ALPHA_GRID} for variant in ('racer_g','racer_p')}
 for redraw in range(20):
  split=_target_split(len(labels),redraw,BASE_SEED);prob,_=_knn(emb[split['router']],labels[split['router']],emb,40);split_seed=BASE_SEED+SPLIT_STEP*redraw
  for variant,code in (('racer_g','G'),('racer_p','P')):
   augmented,scores=nonconformity_scores(prob,code,seed=BASE_SEED,split_seed=split_seed);real=scores[:,:-1];raw=raw_nonconformity_scores(augmented,code)[:,:-1]
   for alpha in ALPHA_GRID:
    row=result[variant][str(alpha)];cal=select_standard_crc_threshold(real[split['final_calibration']],labels[split['final_calibration']],alpha)
    if not cal.feasible:
     row['noncertified_redraws']+=1;row['noncertification_reasons'][cal.non_certification_reason]+=1;continue
    row['certified_redraws']+=1;test=split['test'];mask=real[test]<=cal.threshold;null=scores[test,-1]<=cal.threshold;row['deployed_test_queries']+=len(test);row['empty_real_sets']+=int(np.count_nonzero(mask.sum(axis=1)==0));row['null_selected_diagnostic']+=int(np.count_nonzero(null));row['real_calls']+=int(mask.sum())
    for local,query in enumerate(test):
     values,counts=np.unique(raw[query],return_counts=True);had_full=False;had_partial=False
     for value,count in zip(values,counts):
      if count<=1:continue
      members=raw[query]==value;selected=int(np.count_nonzero(mask[local]&members));row['tied_blocks']+=1
      if selected==count:
       row['fully_admitted_tied_blocks']+=1;row['full_admitted_tied_block_sizes'].append(int(count));had_full=True
      elif selected:
       row['partially_admitted_tied_blocks']+=1;had_partial=True
      else:row['unadmitted_tied_blocks']+=1
     row['queries_with_full_tied_block']+=int(had_full);row['queries_with_partial_tied_block']+=int(had_partial)
 for variant in result:
  for alpha,row in result[variant].items():
   row['noncertification_reasons']=dict(row['noncertification_reasons']);n=row['deployed_test_queries'];row['empty_set_rate']=row['empty_real_sets']/n if n else None;row['null_selection_diagnostic_rate']=row['null_selected_diagnostic']/n if n else None;row['mean_real_calls']=row['real_calls']/n if n else None;sizes=row.pop('full_admitted_tied_block_sizes');row['mean_full_admitted_tied_block_size']=float(np.mean(sizes)) if sizes else None;row['max_full_admitted_tied_block_size']=max(sizes) if sizes else None
 payload={'schema_version':1,'issue':73,'host':a.host,'stamp':a.stamp,'worker_count':a.worker_count,'protocol':a.protocol,'benchmark':a.benchmark,'pool_size':a.pool_size,'pool_id':hash_json({'pool_seed':7001,'model_ids':[task.model_ids[int(i)] for i in columns]}),'redraws':20,'target_grid':list(ALPHA_GRID),'methods':result,'source_hashes':dict(task.source_hashes),'diagnostic_null_only':True,'standard_deployment_uses_real_scores_only':True}
 a.out.parent.mkdir(parents=True,exist_ok=True);tmp=a.out.with_name(a.out.name+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(tmp,a.out);print(a.out);return 0
if __name__=='__main__':raise SystemExit(main())
