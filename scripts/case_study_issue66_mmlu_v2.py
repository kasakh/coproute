"""Optimized scoped ISSUE-66 MMLU pool-11 answer selection and cost study."""
from __future__ import annotations
import argparse,csv,json,os,time
from collections import defaultdict
from pathlib import Path
import sys
from typing import Sequence
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from components.data_protocol import load_routerbench
from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS,_ensure_pandas_pickle_compat
from scripts.run_learned_baselines import _routerbench_costs
from scripts.run_issue78_o2_full_matrix import _select_candidate
from scripts.run_paired_main_suite import BASE_SEED,SPLIT_STEP,_knn,_load_task,_score_chain_discount_candidates,_score_nonlearned,_split_query_ids,_target_split,atomic_write_json,evaluate_score_family
METHODS=('fixed_tally_factorized','fixed_tally_chain_discounted','racer_g','racer_p','exact_independence','fixed_design_marginal_factorized');ALPHAS=(0.05,0.10)
def se(v):return float(np.std(v,ddof=1)/np.sqrt(len(v))) if len(v)>1 else None
def main(argv:Sequence[str]|None=None)->int:
 ap=argparse.ArgumentParser();ap.add_argument('--routerbench-pickle',type=Path,required=True);ap.add_argument('--embedding-archive',type=Path,required=True);ap.add_argument('--embedding-manifest',type=Path,required=True);ap.add_argument('--host',required=True);ap.add_argument('--stamp',required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
 if a.out_dir.exists():raise SystemExit(f'collision {a.out_dir}')
 a.out_dir.mkdir(parents=True);_ensure_pandas_pickle_compat();task=_load_task('routerbench','mmlu',routerbench_pickle=a.routerbench_pickle,embedding_archive=a.embedding_archive,embedding_manifest=a.embedding_manifest);labels=np.asarray(task.correctness,dtype=np.int8);model_ids=list(task.model_ids)
 cells={str(alpha):[] for alpha in ALPHAS}
 for redraw in range(20):
  split=_target_split(len(labels),redraw,BASE_SEED);split_ids=_split_query_ids(task,split);prob,_=_knn(task.embeddings[split['router']],labels[split['router']],task.embeddings,40);split_seed=BASE_SEED+SPLIT_STEP*redraw;scores=_score_nonlearned(prob,labels[split['design']],model_ids,base_seed=BASE_SEED,split_seed=split_seed,preserve_score_prefixes=True);chain,profile=_score_chain_discount_candidates(prob,labels[split['design']],model_ids,scores['fixed_tally_factorized'],base_seed=BASE_SEED,split_seed=split_seed)
  chain_list=[{'grid_index':i,'hyper':{'minimum_survivor_count':m},'scores':chain[m]} for i,m in enumerate((0,5,20,50))]
  for alpha in ALPHAS:
   methods={name:evaluate_score_family(value,labels,split['final_calibration'],split['test'],task.query_ids,model_ids,alpha,method_family=name) for name,value in scores.items()}
   chosen,trajectory,fallback=_select_candidate(chain_list,labels,split['validation'],alpha);endpoint=evaluate_score_family(chain[0],labels,split['final_calibration'],split['test'],task.query_ids,model_ids,alpha,method_family='fixed_tally_factorized')
   if endpoint!=methods['fixed_tally_factorized']:raise RuntimeError('chain off endpoint differs')
   chain_result=evaluate_score_family(chosen['scores'],labels,split['final_calibration'],split['test'],task.query_ids,model_ids,alpha,method_family='fixed_tally_chain_discounted');chain_result.update({'chosen_hyper':chosen['hyper'],'selection_source':'validation_only','selection_fallback_used':fallback,'validation_trajectory':trajectory,'endpoint_result_identity_verified':True});methods['fixed_tally_chain_discounted']=chain_result
   if set(methods)!=set(METHODS):raise RuntimeError(set(methods))
   cells[str(alpha)].append({'redraw_id':redraw,'split_ids':split_ids,'methods':methods,'probabilities':prob[split['test']].tolist()})
 raw=load_routerbench(a.routerbench_pickle,shot='0shot',model_columns=ROUTERBENCH_MODEL_COLUMNS);all_costs=_routerbench_costs(raw);task_rows=np.flatnonzero(raw.frame['task'].astype(str).to_numpy()=='mmlu');costs=all_costs[task_rows];raw_ids=[raw.query_ids[int(i)] for i in task_rows]
 if list(task.query_ids)!=raw_ids:raise RuntimeError('price identity differs')
 model_of={m:i for i,m in enumerate(model_ids)};records=defaultdict(list);selector_ns=[]
 for alpha in ALPHAS:
  for redraw in cells[str(alpha)]:
   test_ids=redraw['split_ids']['test'];test_rows=[task.query_ids.index(q) for q in test_ids];prob=np.asarray(redraw.pop('probabilities'))
   for method in METHODS:
    r=redraw['methods'][method]
    if not r['certified']:records[(alpha,method)].append({'certified':False});continue
    values=[]
    for i,(qid,selected_ids) in enumerate(zip(test_ids,r['selected_model_ids_by_query'])):
     q=test_rows[i];selected=[model_of[m] for m in selected_ids];begin=time.perf_counter_ns();chosen=min(selected,key=lambda m:(-prob[i,m],m)) if selected else None;selector_ns.append(time.perf_counter_ns()-begin);values.append((len(selected),bool(selected and np.any(labels[q,selected])),bool(chosen is not None and labels[q,chosen]),float(costs[q,selected].sum()) if selected else 0.0))
    records[(alpha,method)].append({'certified':True,'rows':values})
   design=[task.query_ids.index(q) for q in redraw['split_ids']['design']];best=int(np.argmax(labels[design].mean(axis=0)))
   for name,selected in (('best_single_design',[best]),('all_models_router_top',list(range(11)))):
    values=[]
    for i,q in enumerate(test_rows):chosen=min(selected,key=lambda m:(-prob[i,m],m));values.append((len(selected),bool(np.any(labels[q,selected])),bool(labels[q,chosen]),float(costs[q,selected].sum())))
    records[(alpha,name)].append({'certified':True,'rows':values})
 summary=[]
 for (alpha,method),redraws in sorted(records.items()):
  good=[r for r in redraws if r['certified']];flat=[x for r in good for x in r['rows']];calls=[np.mean([x[0] for x in r['rows']]) for r in good];cost=[np.mean([x[3] for x in r['rows']]) for r in good];acc=[np.mean([x[2] for x in r['rows']]) for r in good];cov=[np.mean([x[1] for x in r['rows']]) for r in good]
  summary.append({'target_alpha':alpha,'method':method,'certified_redraws':len(good),'certification_rate':len(good)/20,'test_queries':len(flat),'mean_calls':float(np.mean([x[0] for x in flat])) if flat else None,'mean_calls_se_redraw':se(calls),'containment_accuracy':float(np.mean([x[1] for x in flat])) if flat else None,'containment_accuracy_se_redraw':se(cov),'final_answer_accuracy':float(np.mean([x[2] for x in flat])) if flat else None,'final_answer_accuracy_se_redraw':se(acc),'mean_public_price_cost_usd':float(np.mean([x[3] for x in flat])) if flat else None,'mean_cost_se_redraw':se(cost),'selector_mean_microseconds':float(np.mean(selector_ns))/1000})
 by={(r['target_alpha'],r['method']):r for r in summary}
 for r in summary:
  ref=by[(r['target_alpha'],'racer_g')];r['calls_reduction_vs_racer_g']=None if r['mean_calls'] is None or ref['mean_calls'] is None else ref['mean_calls']-r['mean_calls'];r['cost_reduction_vs_racer_g_usd']=None if r['mean_public_price_cost_usd'] is None or ref['mean_public_price_cost_usd'] is None else ref['mean_public_price_cost_usd']-r['mean_public_price_cost_usd']
 with (a.out_dir/f'issue66_{a.host}_{a.stamp}_case_study.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 price=[{'model':m,'mmlu_queries':len(costs),'mean_response_cost_usd':float(costs[:,i].mean()),'median_response_cost_usd':float(np.median(costs[:,i])),'p90_response_cost_usd':float(np.quantile(costs[:,i],0.9)),'source_field':m+'|total_cost'} for i,m in enumerate(model_ids)]
 with (a.out_dir/f'issue66_{a.host}_{a.stamp}_public_price_sheet.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(price[0]));w.writeheader();w.writerows(price)
 out={'schema_version':2,'issue':66,'host':a.host,'stamp':a.stamp,'scope':'MMLU pool11, alpha 0.05 and 0.10','answer_selector':'highest frozen KNN probability among called models, model-index tie break','cost_source':'RouterBench archived public-price total_cost per generated response','cost_includes':'all called model generations; selector CPU measured separately','generated_tokens':'not separately available; token-dependent charges are encoded in total_cost','summary':summary,'price_sheet':price,'source_hashes':dict(task.source_hashes),'repeated_sampling_scope_note':'not estimated because archive has one fixed response per model/query and no new inference is in scope'};atomic_write_json(a.out_dir/f'issue66_{a.host}_{a.stamp}_case_study.json',out);(a.out_dir/f'issue66_{a.host}_{a.stamp}_DONE').write_text('\n');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
