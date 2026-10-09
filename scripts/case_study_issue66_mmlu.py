"""Scoped ISSUE-66 MMLU pool-11 answer selection and cost case study."""
from __future__ import annotations
import argparse,csv,json,os,time
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any,Sequence
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from components.data_protocol import PRIMARY_COVERAGE_POLICY,derive_coverage_labels,load_routerbench
from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS,_ensure_pandas_pickle_compat
from scripts.run_learned_baselines import _routerbench_costs
from scripts.run_paired_main_suite import BASE_SEED,SPLIT_STEP,_knn,_load_task,atomic_write_json,run_paired_suite
METHODS=('fixed_tally_factorized','fixed_tally_chain_discounted','racer_g','racer_p','exact_independence','fixed_design_marginal_factorized')
ALPHAS=(0.05,0.10)
def se(v):return float(np.std(v,ddof=1)/np.sqrt(len(v))) if len(v)>1 else None
def main(argv:Sequence[str]|None=None)->int:
 ap=argparse.ArgumentParser();ap.add_argument('--routerbench-pickle',type=Path,required=True);ap.add_argument('--embedding-archive',type=Path,required=True);ap.add_argument('--embedding-manifest',type=Path,required=True);ap.add_argument('--host',required=True);ap.add_argument('--stamp',required=True);ap.add_argument('--worker-count',type=int,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
 if a.out_dir.exists():raise SystemExit(f'collision {a.out_dir}')
 a.out_dir.mkdir(parents=True);_ensure_pandas_pickle_compat();task=_load_task('routerbench','mmlu',routerbench_pickle=a.routerbench_pickle,embedding_archive=a.embedding_archive,embedding_manifest=a.embedding_manifest)
 payload=run_paired_suite(task,pool_size=11,pool_seed=7001,redraw_start=0,redraw_end=20,knn_k=40,include_chain_discount=True,learned_ensembles={},protocol='routerbench')
 if set(payload['method_identities'])!=set(METHODS):raise RuntimeError(payload['method_identities'])
 raw=load_routerbench(a.routerbench_pickle,shot='0shot',model_columns=ROUTERBENCH_MODEL_COLUMNS);all_costs=_routerbench_costs(raw);rows=np.flatnonzero(raw.frame['task'].astype(str).to_numpy()=='mmlu');costs=all_costs[rows];raw_ids=[raw.query_ids[int(i)] for i in rows]
 if list(task.query_ids)!=raw_ids:raise RuntimeError('task and price query identities differ')
 labels=np.asarray(task.correctness,dtype=np.int8);model_ids=list(task.model_ids);model_of={m:i for i,m in enumerate(model_ids)};row_of={q:i for i,q in enumerate(task.query_ids)}
 records=defaultdict(list);selector_ns=[]
 cells={str(c['alpha']):c for c in payload['cells']}
 for alpha in ALPHAS:
  cell=cells[str(alpha)]
  for redraw in cell['redraws']:
   split=redraw['split_ids'];router_idx=np.asarray([row_of[q] for q in split['router']]);test_idx=np.asarray([row_of[q] for q in split['test']]);prob,_=_knn(task.embeddings[router_idx],labels[router_idx],task.embeddings,40);methods=redraw['methods']
   for method in METHODS:
    r=methods[method]
    if not r['certified']:
     records[(alpha,method)].append({'certified':False});continue
    query_rows=[]
    for local,(qid,selected_ids) in enumerate(zip(r['test_query_ids'],r['selected_model_ids_by_query'])):
     q=row_of[qid];selected=[model_of[m] for m in selected_ids];begin=time.perf_counter_ns()
     chosen=min(selected,key=lambda m:(-prob[q,m],m)) if selected else None
     selector_ns.append(time.perf_counter_ns()-begin);contain=bool(selected and np.any(labels[q,selected]));correct=bool(chosen is not None and labels[q,chosen]);cost=float(costs[q,selected].sum()) if selected else 0.0
     query_rows.append((len(selected),contain,correct,cost))
    records[(alpha,method)].append({'certified':True,'rows':query_rows})
   design_idx=np.asarray([row_of[q] for q in split['design']]);best=int(np.argmax(labels[design_idx].mean(axis=0)))
   for name,selector in (('best_single_design',lambda q:[best]),('all_models_router_top',lambda q:list(range(11)))):
    query_rows=[]
    for q in test_idx:
     selected=selector(q);chosen=min(selected,key=lambda m:(-prob[q,m],m));query_rows.append((len(selected),bool(np.any(labels[q,selected])),bool(labels[q,chosen]),float(costs[q,selected].sum())))
    records[(alpha,name)].append({'certified':True,'rows':query_rows})
 summary=[]
 for (alpha,method),redraws in sorted(records.items()):
  certified=[r for r in redraws if r['certified']];flat=[x for r in certified for x in r['rows']];redraw_calls=[float(np.mean([x[0] for x in r['rows']])) for r in certified];redraw_cost=[float(np.mean([x[3] for x in r['rows']])) for r in certified];redraw_acc=[float(np.mean([x[2] for x in r['rows']])) for r in certified];redraw_cov=[float(np.mean([x[1] for x in r['rows']])) for r in certified]
  summary.append({'target_alpha':alpha,'method':method,'redraws':20,'certified_redraws':len(certified),'certification_rate':len(certified)/20,'test_queries':len(flat),'mean_calls':float(np.mean([x[0] for x in flat])) if flat else None,'mean_calls_se_redraw':se(redraw_calls),'containment_accuracy':float(np.mean([x[1] for x in flat])) if flat else None,'containment_accuracy_se_redraw':se(redraw_cov),'final_answer_accuracy':float(np.mean([x[2] for x in flat])) if flat else None,'final_answer_accuracy_se_redraw':se(redraw_acc),'mean_public_price_cost_usd':float(np.mean([x[3] for x in flat])) if flat else None,'mean_cost_se_redraw':se(redraw_cost),'answer_selector':'highest_router_probability_within_called_set','selector_mean_microseconds':float(np.mean(selector_ns))/1000 if selector_ns else None})
 by={(r['target_alpha'],r['method']):r for r in summary}
 for row in summary:
  ref=by.get((row['target_alpha'],'racer_g'));row['cost_reduction_vs_racer_g']=None if not ref or row['mean_public_price_cost_usd'] is None or ref['mean_public_price_cost_usd'] is None else ref['mean_public_price_cost_usd']-row['mean_public_price_cost_usd'];row['calls_reduction_vs_racer_g']=None if not ref or row['mean_calls'] is None or ref['mean_calls'] is None else ref['mean_calls']-row['mean_calls']
 fields=list(summary[0]);
 with (a.out_dir/f'issue66_{a.host}_{a.stamp}_case_study.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
 price=[]
 for i,m in enumerate(model_ids):price.append({'model':m,'mmlu_queries':len(costs),'mean_response_cost_usd':float(costs[:,i].mean()),'median_response_cost_usd':float(np.median(costs[:,i])),'p90_response_cost_usd':float(np.quantile(costs[:,i],0.9)),'source_field':m+'|total_cost'})
 with (a.out_dir/f'issue66_{a.host}_{a.stamp}_public_price_sheet.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(price[0]));w.writeheader();w.writerows(price)
 output={'schema_version':1,'issue':66,'host':a.host,'stamp':a.stamp,'worker_count':a.worker_count,'scope':'MMLU pool 11, targets 0.05 and 0.10','methods':list(METHODS),'answer_selector':'choose the called model with highest frozen KNN router probability; ties use model index','cost_source':'RouterBench public-price total_cost fields for each archived model response','cost_includes':'all called model responses; selector CPU cost separately measured','generated_tokens':'not separately available in the public archive; token-dependent charges are encoded in total_cost','summary':summary,'price_sheet':price,'source_hashes':dict(task.source_hashes),'repeated_sampling_scope_note':'not estimated because the archive contains one fixed response per model/query and no new inference is in scope','raw_replay_method_identities':payload['method_identities']}
 atomic_write_json(a.out_dir/f'issue66_{a.host}_{a.stamp}_case_study.json',output);atomic_write_json(a.out_dir/f'issue66_{a.host}_{a.stamp}_nonlearned_replay.json',payload);(a.out_dir/f'issue66_{a.host}_{a.stamp}_DONE').write_text('\n');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
