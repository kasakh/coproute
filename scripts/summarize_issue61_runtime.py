"""Summarize ISSUE-61 phase timing records and lazy-greedy scaling."""
from __future__ import annotations
import argparse,csv,json,math,pickle
from pathlib import Path
from typing import Any
import numpy as np

def load(roots:list[Path])->list[dict[str,Any]]:
 rows=[]
 for root in roots:
  for path in sorted((root/'raw').glob('*.json')):
   row=json.loads(path.read_text());row['source_file']=str(path);rows.append(row)
 by={}
 for row in rows:
  key=(row['pool_size'],row['design_size'],row['phase'])
  if key in by:raise ValueError(f'duplicate timing key {key}')
  by[key]=row
 if len(by)!=320:raise ValueError(f'expected 320 records, found {len(by)}')
 return [by[key] for key in sorted(by)]

def slope(rows,phase,design):
 points=[r for r in rows if r['phase']==phase and r['design_size']==design and r['pool_size'] in (100,200,300,500)]
 if len(points)!=4:return None
 x=np.log([r['pool_size'] for r in points]);y=np.log([r['median_seconds'] for r in points]);coef=np.polyfit(x,y,1)
 return {'phase':phase,'design_size':design,'measured_pool_exponent':float(coef[0]),'intercept':float(coef[1]),'predicted_seconds_pool1000':float(np.exp(coef[1])*1000**coef[0]),'predicted_seconds_pool3811':float(np.exp(coef[1])*3811**coef[0])}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',action='append',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args();rows=load(a.root)
 if a.out_dir.exists():raise SystemExit(f'refusing existing {a.out_dir}')
 a.out_dir.mkdir()
 fields=['pool_size','design_size','phase','complexity','repetitions','cold_seconds_per_unit','median_seconds','p90_seconds','median_ci95_low','median_ci95_high','p90_ci95_low','p90_ci95_high','timing_unit','query_batch_size','router_rows','calibration_rows','max_order_depth','serialized_result_bytes','peak_rss_kib','lazy_order_exact','source_file']
 with (a.out_dir/'issue61_runtime_memory_table.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in rows)
 lazy=[]
 lookup={(r['pool_size'],r['design_size'],r['phase']):r for r in rows}
 for pool in (11,100,200,300,500,1000,3811):
  designs=(25,50,100,250) if pool!=11 or True else (100,)
  for design in designs:
   for eager_phase,lazy_phase,kind in [('fit_tally_eager','fit_tally_lazy','fixed_tally_fit'),('infer_query_local_eager','infer_query_local_lazy','query_local_order')]:
    eager=lookup.get((pool,design,eager_phase));lazy_row=lookup.get((pool,design,lazy_phase))
    if not eager or not lazy_row:continue
    diag=lazy_row.get('lazy_diagnostics',{})
    lazy.append({'kind':kind,'pool_size':pool,'design_size':design,'eager_median_seconds':eager['median_seconds'],'lazy_median_seconds':lazy_row['median_seconds'],'eager_over_lazy_speedup':eager['median_seconds']/lazy_row['median_seconds'],'lazy_order_exact':lazy_row.get('lazy_order_exact'),'lazy_candidate_gain_evaluations':diag.get('candidate_gain_evaluations') or lazy_row.get('query_local_lazy_candidate_gain_evaluations'),'eager_candidate_gain_evaluations':diag.get('eager_candidate_gain_evaluations'),'evaluation_reduction_fraction':(1-diag['candidate_gain_evaluations']/diag['eager_candidate_gain_evaluations']) if diag.get('eager_candidate_gain_evaluations') else None})
 lfields=list(lazy[0])
 with (a.out_dir/'issue61_lazy_greedy_table.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=lfields);w.writeheader();w.writerows(lazy)
 scaling=[]
 for phase in sorted({r['phase'] for r in rows}):
  for design in (25,50,100,250):
   value=slope(rows,phase,design)
   if value:scaling.append(value)
 sfields=list(scaling[0])
 with (a.out_dir/'issue61_scaling_exponents.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=sfields);w.writeheader();w.writerows(scaling)
 complexity=[]
 for phase in sorted({r['phase'] for r in rows}):
  subset=[r for r in rows if r['phase']==phase]
  exponents=[r['measured_pool_exponent'] for r in scaling if r['phase']==phase]
  complexity.append({'phase':phase,'theoretical_complexity':subset[0]['complexity'],'timing_unit':subset[0]['timing_unit'],'measured_pool_exponent_median':float(np.median(exponents)) if exponents else None,'peak_rss_mib_max':max(r['peak_rss_kib'] for r in subset)/1024,'serialized_result_bytes_max':max((r['serialized_result_bytes'] or 0) for r in subset)})
 cfields=list(complexity[0])
 with (a.out_dir/'issue61_complexity_scaling_table.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=cfields);w.writeheader();w.writerows(complexity)
 state_rows=[]
 for pool in (11,100,200,300,500):
  chain=next(r for r in rows if r['pool_size']==pool and r['design_size']==100 and r['phase']=='fit_chain')
  state_rows.extend([
   {'method':'fixed_tally_D1','pool_size':pool,'serialized_state_bytes':len(pickle.dumps(np.arange(pool,dtype=np.int64),protocol=5)),'state_description':'one full fixed order'},
   {'method':'fixed_tally_D2','pool_size':pool,'serialized_state_bytes':chain['serialized_result_bytes'],'state_description':'fixed order plus chain counts and discounts'},
   {'method':'RACER_G','pool_size':pool,'serialized_state_bytes':len(pickle.dumps({'variant':'G','seed':7001,'smoothing':1e-6},protocol=5)),'state_description':'constant scorer configuration'},
   {'method':'RACER_P','pool_size':pool,'serialized_state_bytes':len(pickle.dumps({'variant':'P','seed':7001,'smoothing':1e-6},protocol=5)),'state_description':'constant scorer configuration'},
  ])
 sfields=list(state_rows[0])
 with (a.out_dir/'issue61_serialized_state_table.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=sfields);w.writeheader();w.writerows(state_rows)
 fixed=[r for r in lazy if r['kind']=='fixed_tally_fit'];local=[r for r in lazy if r['kind']=='query_local_order']
 payload={'schema_version':1,'issue':61,'record_count':len(rows),'base_grid_records':300,'extended_scaling_records':20,'all_lazy_orders_exact':all(r['lazy_order_exact'] for r in lazy),'lazy_fixed_median_speedup':float(np.median([r['eager_over_lazy_speedup'] for r in fixed])),'lazy_fixed_speedup_pool3811_design250':next(r['eager_over_lazy_speedup'] for r in fixed if r['pool_size']==3811 and r['design_size']==250),'lazy_fixed_evaluation_reduction_median':float(np.median([r['evaluation_reduction_fraction'] for r in fixed if r['evaluation_reduction_fraction'] is not None])),'lazy_query_local_median_speedup':float(np.median([r['eager_over_lazy_speedup'] for r in local])),'max_peak_rss_mib':max(r['peak_rss_kib'] for r in rows)/1024,'tables':['issue61_runtime_memory_table.csv','issue61_lazy_greedy_table.csv','issue61_scaling_exponents.csv','issue61_complexity_scaling_table.csv','issue61_serialized_state_table.csv']}
 (a.out_dir/'issue61_summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
