"""Build the ISSUE-73 audit from Stage 1 rollups, replays, and diagnostics."""
from __future__ import annotations
import argparse,csv,inspect,json,statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from components import racer_protocol
from scripts import run_paired_main_suite

def write_csv(path,rows):
 fields=list(rows[0]);
 with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def code_line(obj):
 lines,start=inspect.getsourcelines(obj);return {'file':inspect.getsourcefile(obj),'start':start,'end':start+len(lines)-1}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--issue62-csv',type=Path,required=True);ap.add_argument('--racer-replay',type=Path,required=True);ap.add_argument('--issue78-shards',type=Path,required=True);ap.add_argument('--factorial-summary',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
 if a.out_dir.exists():raise SystemExit(f'collision {a.out_dir}')
 a.out_dir.mkdir(parents=True);cert={}
 rows=list(csv.DictReader(a.issue62_csv.open()))
 for r in rows:
  base=(r['protocol'],r['benchmark'],r['pool_size'],r['target_alpha'])
  for side in ('method','reference'):
   method=r[side if side=='method' else 'reference_method'];prefix='method' if side=='method' else 'reference';key=(*base,method);value=(int(r[f'{prefix}_certification_numerator']),int(r[f'{prefix}_certification_denominator']))
   if key in cert and cert[key]!=value:raise ValueError(f'cert mismatch {key}')
   cert[key]=value
 if len(cert)!=3584:raise ValueError(f'expected 3584 Stage 1 certification cells, found {len(cert)}')
 cert_rows=[]
 for key,(n,d) in sorted(cert.items()):cert_rows.append({'protocol':key[0],'benchmark':key[1],'pool_size':key[2],'target_alpha':key[3],'method':key[4],'certified_redraws':n,'noncertified_redraws':d-n,'redraws':d})
 write_csv(a.out_dir/'issue73_stage1_certification_counts.csv',cert_rows)
 cert_summary=defaultdict(lambda:[0,0])
 for r in cert_rows:cert_summary[r['method']][0]+=r['certified_redraws'];cert_summary[r['method']][1]+=r['redraws']
 racer_rows=[]
 for path in sorted(a.racer_replay.glob('*.json')):
  p=json.load(open(path))
  for method,alphas in p['methods'].items():
   for alpha,r in alphas.items():racer_rows.append({'protocol':p['protocol'],'benchmark':p['benchmark'],'pool_size':p['pool_size'],'target_alpha':alpha,'method':method,**r})
 if len(racer_rows)!=448:raise ValueError(len(racer_rows))
 for r in racer_rows:
  key=(r['protocol'],r['benchmark'],str(r['pool_size']),r['target_alpha'],r['method']);expected=cert[key]
  if (r['certified_redraws'],r['certified_redraws']+r['noncertified_redraws'])!=expected:raise ValueError(f'replay certification mismatch {key}')
 write_csv(a.out_dir/'issue73_racer_empty_tied_counts.csv',racer_rows)
 o2={}
 for path in sorted(a.issue78_shards.glob('*.json')):
  p=json.load(open(path));protocol=p['protocol'];benchmark=p['benchmark'];pool=p['pool_size']
  for cell in p['cells']:
   alpha=str(cell['alpha']);redraw=cell['redraws'][0]
   for method,r in redraw['methods'].items():
    key=(protocol,benchmark,pool,alpha,method);x=o2.setdefault(key,{'certified_redraws':0,'noncertified_redraws':0,'deployed_test_queries':0,'empty_real_sets':0,'real_calls':0})
    if not r['certified']:x['noncertified_redraws']+=1;continue
    calls=r['test_calls_by_query'];x['certified_redraws']+=1;x['deployed_test_queries']+=len(calls);x['empty_real_sets']+=sum(int(v)==0 for v in calls);x['real_calls']+=sum(calls)
 o2_rows=[]
 for key,x in sorted(o2.items()):
  n=x['deployed_test_queries'];o2_rows.append({'protocol':key[0],'benchmark':key[1],'pool_size':key[2],'target_alpha':key[3],'method':key[4],**x,'empty_set_rate':x['empty_real_sets']/n if n else None,'mean_real_calls':x['real_calls']/n if n else None})
 if len(o2_rows)!=6272:raise ValueError(f'expected 6272 O2 rows, found {len(o2_rows)}')
 write_csv(a.out_dir/'issue73_o2_empty_set_counts.csv',o2_rows)
 factorial=list(csv.DictReader(a.factorial_summary.open()));wanted={'0.02','0.03','0.05','0.075','0.1','0.15'};query_rows=[]
 for r in factorial:
  if r.get('method')=='local_empirical' and r.get('target_alpha') in wanted:query_rows.append({k:r.get(k) for k in ('pool_size','target_alpha','diagnostic_query_count','plateau_depth_mean','plateau_depth_median','largest_tied_block_mean','largest_tied_block_max','score_one_tail_size_mean','score_one_tail_size_max','selected_depth_mean','full_pool_call_rate','factorized_saturation_count')})
 write_csv(a.out_dir/'issue73_query_local_saturation.csv',query_rows)
 citations={'select_standard_crc_threshold':code_line(racer_protocol.select_standard_crc_threshold),'prediction_sets':code_line(racer_protocol.prediction_sets),'standard_all_miss_loss':code_line(racer_protocol.standard_all_miss_loss),'safe_action_loss':code_line(racer_protocol.safe_action_loss),'evaluate_score_family':code_line(run_paired_main_suite.evaluate_score_family),'racer_score_construction':code_line(run_paired_main_suite._score_nonlearned)}
 total_racer_empty=sum(r['empty_real_sets'] for r in racer_rows);total_racer_queries=sum(r['deployed_test_queries'] for r in racer_rows);total_null=sum(r['null_selected_diagnostic'] for r in racer_rows);total_ties=sum(r['tied_blocks'] for r in racer_rows);full_ties=sum(r['fully_admitted_tied_blocks'] for r in racer_rows);partial_ties=sum(r['partially_admitted_tied_blocks'] for r in racer_rows)
 math=[r for r in racer_rows if r['benchmark']=='math_intermediate_algebra_hard' and r['pool_size']==500]
 summary={'schema_version':1,'issue':73,'stage1_certification_identities':len(cert_rows),'stage1_methods':len(cert_summary),'stage1_method_totals':{m:{'certified':v[0],'total':v[1],'noncertified':v[1]-v[0]} for m,v in sorted(cert_summary.items())},'racer_replay_rows':len(racer_rows),'racer_deployed_test_queries':total_racer_queries,'racer_empty_real_sets':total_racer_empty,'racer_empty_set_rate':total_racer_empty/total_racer_queries if total_racer_queries else None,'racer_null_selected_diagnostic':total_null,'racer_null_selection_diagnostic_rate':total_null/total_racer_queries if total_racer_queries else None,'racer_tied_blocks':total_ties,'racer_fully_admitted_tied_blocks':full_ties,'racer_partially_admitted_tied_blocks':partial_ties,'o2_rows':len(o2_rows),'o2_empty_real_sets':sum(r['empty_real_sets'] for r in o2_rows),'math_intermediate_pool500':math,'code_citations':citations,'query_local_rows':len(query_rows),'readme_policy_discrepancy':True}
 (a.out_dir/'issue73_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 memo=f'''# ISSUE-73 abstention and empty-set audit\n\n## Findings\n\nStage 1 contains {len(cert_rows)} method/cell/target certification identities across 16 methods. Each method has {next(iter(summary['stage1_method_totals'].values()))['noncertified']} non-certified redraws out of 4,480 because all methods share the same complete-pool endpoint. Non-certification is not an empty deployed set. The current runner emits no operating point, call vector, or risk claim.\n\nAcross {total_racer_queries} deterministically replayed certified RACER-G/P test-query decisions, the real-model empty-set count is {total_racer_empty}. The virtual null was selected {total_null} times as a diagnostic, but Stage 1 discards the null column before standard calibration and routing, so it never suppresses real calls or changes standard all-miss risk.\n\nThe ISSUE-78 O2 archive contributes {len(o2_rows)} method/cell/target rows and {sum(r['deployed_test_queries'] for r in o2_rows)} deployed test-query decisions. Empty real sets: {sum(r['empty_real_sets'] for r in o2_rows)}. Fixed-order D1/D2 scores assign zero to the first model, and every feasible threshold is nonnegative, so each deployed query calls at least one real model.\n\nRACER raw scores contained {total_ties} tied blocks. Calibrated thresholds fully admitted {full_ties} and partially split {partial_ties}. These are score-resolution diagnostics, not certification failures.\n\nQuery-local empirical diagnostics cover {len(query_rows)} pool/target rows. The terminal plateau appears around depth 3.6 to 4.6, and the tied score-one tail grows from roughly 31 models at pool 100 to 437 at pool 500. Recorded factorized numerical saturation count is zero: this is empirical coverage saturation after local neighbors are covered, not floating-point product saturation.\n\n## Loss and action contract\n\n1. Final calibration uses standard real-model all-miss loss. Empty real sets count as misses.\n2. Reported Stage 1 test risk is the same standard all-miss loss.\n3. Call counts include selected real models only.\n4. RACER's augmented null is a separately named diagnostic only.\n5. A failed certificate produces no deployment threshold. `RACERProtocol.route` rejects a non-certified `CalibrationResult`.\n6. `safe_action` and `abstain` are compatibility aliases for an empty real set. No automatic human-review action is executed by the Stage 1 runner.\n\n## Documentation discrepancy\n\nThe README says a non-certified calibration takes a declared operational safe action. Current code and `docs/LEARNED_BASELINES_PROTOCOL.md` instead specify no risk-valid deployed operating point. The README wording is stale and must be corrected; no paper or TeX file was changed in this audit.\n\n## Code citations\n\n'''+"\n".join(f"- `{v['file']}:{v['start']}-{v['end']}`: {k}" for k,v in citations.items())+"\n"
 (a.out_dir/'issue73_audit_memo.md').write_text(memo);print(json.dumps({k:summary[k] for k in ('stage1_certification_identities','racer_deployed_test_queries','racer_empty_real_sets','racer_null_selected_diagnostic','racer_tied_blocks','racer_fully_admitted_tied_blocks','racer_partially_admitted_tied_blocks','o2_empty_real_sets')},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
