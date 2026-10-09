#!/usr/bin/env python3
import argparse, csv, hashlib, json, math, os, statistics, sys, tempfile
from collections import defaultdict
from pathlib import Path

METHODS=['fixed_tally_factorized','fixed_tally_chain_discounted','racer_g','racer_p']+[f'{family}_seed_{seed}' for family in ('flexrouter','barp') for seed in (9101,9102,9103,9104,9105)]
ALPHAS=[0.02,0.03,0.05,0.075,0.1,0.15,0.2]
ROUTEREVAL_FAMILIES={'ifeval':'ifeval','musr':'musr','bbh_boolean_expressions':'bbh','bbh_navigate':'bbh','gpqa_diamond':'gpqa','math_algebra_hard':'math','math_intermediate_algebra_hard':'math'}

def family(method):
 if method.startswith('flexrouter_seed_'): return 'flexrouter'
 if method.startswith('barp_seed_'): return 'barp'
 return method

def number(value):
 return None if value in ('',None) else float(value)

def weighted(rows,value_key,weight_key):
 pairs=[(number(r[value_key]),int(r[weight_key])) for r in rows if number(r[value_key]) is not None and int(r[weight_key])>0]
 total=sum(w for _,w in pairs)
 return None if not total else sum(v*w for v,w in pairs)/total

def mean(values):
 values=[v for v in values if v is not None]
 return None if not values else statistics.fmean(values)

def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as h:
  for b in iter(lambda:h.read(1048576),b''): d.update(b)
 return d.hexdigest()

def atomic(path,value):
 with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as h:
  json.dump(value,h,indent=2,sort_keys=True,allow_nan=False); h.write('\n'); h.flush(); os.fsync(h.fileno()); tmp=Path(h.name)
 os.replace(tmp,path)

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--root',required=True,type=Path); parser.add_argument('--host',required=True); parser.add_argument('--stamp',required=True); parser.add_argument('--out',required=True,type=Path); args=parser.parse_args(); root=args.root.resolve(); prefix=f'issue52_{args.host}_{args.stamp}'
 if root.name!=prefix or prefix not in args.out.name: raise SystemExit('host/stamp identity mismatch')
 if args.out.exists(): raise SystemExit(f'refusing to overwrite {args.out}')
 for marker in ('DONE','POSTPROCESS_DONE'):
  if not (root/f'{prefix}_{marker}').is_file(): raise SystemExit(f'missing {marker}')
 for marker in ('FAILED','POSTPROCESS_FAILED'):
  if (root/f'{prefix}_{marker}').exists(): raise SystemExit(f'failure marker present: {marker}')
 manifest=json.load(open(root/f'{prefix}_manifest.json')); protocol=manifest['protocol']; tasks=list(manifest['tasks'])
 if protocol=='routereval':
  pairs=[(t,t,'in_domain') for t in tasks]+[(s,t,'source_fitted') for s in tasks for t in tasks if ROUTEREVAL_FAMILIES[s]!=ROUTEREVAL_FAMILIES[t]]; pools=list(manifest['pools']); expected_shards=3600
 elif protocol=='routerbench':
  pairs=[(t,t,'in_domain') for t in tasks]+[(s,t,'source_fitted') for s in tasks for t in tasks if s!=t]; pools=[int(manifest['pool'])]; expected_shards=320
 else: raise SystemExit('unknown protocol')
 expected_cells={(s,t,f,int(p)) for s,t,f in pairs for p in pools}; errors=[]
 shards=list((root/'shards').glob('*.json')); merged=list((root/'merged').glob('*.json')); analyses=list((root/'analysis').glob('*_analysis.json')); csvs=list((root/'analysis').glob('*_analysis.csv'))
 if len(shards)!=expected_shards: errors.append(f'shard count {len(shards)} != {expected_shards}')
 if len(merged)!=len(expected_cells): errors.append(f'merged count {len(merged)} != {len(expected_cells)}')
 if len(analyses)!=len(expected_cells) or len(csvs)!=len(expected_cells): errors.append('analysis file count mismatch')
 observed_cells=set(); rows=[]; source_hashes={}
 csv.field_size_limit(sys.maxsize)
 for path in sorted(csvs):
  source_hashes[path.name]=sha(path)
  with path.open(newline='') as h: file_rows=list(csv.DictReader(h))
  if len(file_rows)!=98: errors.append(f'{path.name}: row count {len(file_rows)} != 98'); continue
  identity={(r['source_benchmark'],r['target_benchmark'],r['fit_mode'],int(r['pool_size'])) for r in file_rows}
  if len(identity)!=1: errors.append(f'{path.name}: mixed identity'); continue
  cell=next(iter(identity)); observed_cells.add(cell)
  if {r['method'] for r in file_rows}!=set(METHODS): errors.append(f'{path.name}: method coverage mismatch')
  if {float(r['alpha']) for r in file_rows}!=set(ALPHAS): errors.append(f'{path.name}: alpha coverage mismatch')
  if any(int(r['bootstrap_repetitions'])!=10000 or r['test_retuning_used'].lower()!='false' for r in file_rows): errors.append(f'{path.name}: bootstrap or retuning mismatch')
  if any(any(str(v).lower() in ('na','n/a','nan') for v in r.values()) for r in file_rows): errors.append(f'{path.name}: forbidden NA token')
  rows.extend(file_rows)
 if observed_cells!=expected_cells: errors.append(f'analysis identity coverage differs: missing={len(expected_cells-observed_cells)} extra={len(observed_cells-expected_cells)}')
 in_domain={(r['target_benchmark'],int(r['pool_size']),float(r['alpha']),r['method']):r for r in rows if r['fit_mode']=='in_domain'}
 ood=[r for r in rows if r['fit_mode']=='source_fitted']
 if not in_domain or not ood: errors.append('both fit modes are not covered')
 family_rows=defaultdict(list)
 for r in ood: family_rows[family(r['method'])].append(r)
 method_summary={}
 for name,group in sorted(family_rows.items()):
  source_cert=sum(int(r['source_calibrated_certification_count']) for r in group); source_total=sum(int(r['source_calibrated_certification_total']) for r in group); target_cert=sum(int(r['target_recalibrated_certification_count']) for r in group); target_total=sum(int(r['target_recalibrated_certification_total']) for r in group)
  premiums=[]
  for r in group:
   baseline=in_domain.get((r['target_benchmark'],int(r['pool_size']),float(r['alpha']),r['method']))
   out_calls=number(r['target_recalibrated_mean_calls']); in_calls=number(baseline['target_recalibrated_mean_calls']) if baseline else None
   if out_calls is not None and in_calls is not None: premiums.append(out_calls-in_calls)
  method_summary[name]={
   'source_calibrated_certification_rate':source_cert/source_total,
   'target_recalibrated_certification_rate':target_cert/target_total,
   'source_calibrated_risk_inflation_mean':weighted(group,'paired_risk_inflation_mean','paired_risk_inflation_redraw_count'),
   'source_calibrated_call_overhead_mean':weighted(group,'paired_call_overhead_mean','paired_call_overhead_redraw_count'),
   'target_recalibrated_risk_minus_alpha_mean':weighted([{**r,'target_inflation':str(number(r['target_recalibrated_target_risk'])-float(r['alpha'])) if number(r['target_recalibrated_target_risk']) is not None else ''} for r in group],'target_inflation','target_recalibrated_certification_count'),
   'target_recalibrated_fit_call_premium_vs_in_domain_mean':mean(premiums),
   'fit_premium_comparison_count':len(premiums),
   'source_risk_inflation_positive_count':sum(number(r['paired_risk_inflation_mean']) is not None and number(r['paired_risk_inflation_mean'])>0 for r in group),
   'source_risk_inflation_available_count':sum(number(r['paired_risk_inflation_mean']) is not None for r in group),
   'source_risk_inflation_ci_excludes_zero_count':sum(number(r['paired_risk_inflation_redraw_ci95_low']) is not None and number(r['paired_risk_inflation_redraw_ci95_low'])>0 for r in group),
  }
 def global_calls(fit_mode,name):
  selected=[r for r in rows if r['fit_mode']==fit_mode and family(r['method'])==name]
  return weighted(selected,'target_recalibrated_mean_calls','target_recalibrated_certification_count')
 families=['fixed_tally_factorized','fixed_tally_chain_discounted','racer_g','racer_p','flexrouter','barp']
 in_rank=sorted(families,key=lambda n:(math.inf if global_calls('in_domain',n) is None else global_calls('in_domain',n),n)); out_rank=sorted(families,key=lambda n:(math.inf if global_calls('source_fitted',n) is None else global_calls('source_fitted',n),n))
 progress_path=root/f'{prefix}_progress.json'
 progress_data=json.load(open(progress_path)) if progress_path.is_file() else None
 progress_compact=None if progress_data is None else {key:progress_data.get(key) for key in ('requested','complete','failed','elapsed_seconds','completed_utc')}
 progress_compact=None if progress_compact is None else {**progress_compact,'unexpected_zero_certification_count':len(progress_data.get('unexpected_zero_certification_cells',[]))}
 payload={'schema_version':1,'issue':52,'host':args.host,'stamp':args.stamp,'protocol':protocol,'status':'passed' if not errors else 'failed','errors':errors,'expected':{'shards':expected_shards,'cells':len(expected_cells),'analysis_rows':len(expected_cells)*98,'fit_modes':['in_domain','source_fitted'],'method_families':families,'calibration_modes':['source_calibrated','target_recalibrated'],'bootstrap_repetitions':10000},'observed':{'shards':len(shards),'merged':len(merged),'analysis_json':len(analyses),'analysis_csv':len(csvs),'analysis_rows':len(rows),'cells':len(observed_cells)},'method_summary':method_summary,'target_recalibrated_global_call_ranking':{'in_domain':in_rank,'source_fitted':out_rank,'same_order':in_rank==out_rank},'analysis_csv_sha256':source_hashes,'progress_snapshot':progress_compact}
 atomic(args.out,payload); print(json.dumps({k:payload[k] for k in ('status','errors','observed','target_recalibrated_global_call_ranking')},indent=2)); return 0 if not errors else 2
if __name__=='__main__': raise SystemExit(main())
