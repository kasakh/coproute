#!/usr/bin/env python3
"""Relative call efficiency radar, paper version.

Axes are the ten main-table datasets (RouterBench at pool 11, RouterEval at
pool 100). Value = best certified mean calls on that dataset divided by the
policy's certified mean calls, so the per-dataset winner touches the rim at
1.0 and needing twice the calls scores 0.5. One panel per target risk.
Source: issue62 canonical rollup, same cells as Table 1.
"""
import csv, os, math, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pathlib
REPO_ROOT=pathlib.Path(__file__).resolve().parents[2]
ROLL=str(REPO_ROOT/'results'/'paper_rollups')
OUTP=str(REPO_ROOT/'figures_out'/'paper')
pathlib.Path(OUTP).mkdir(parents=True,exist_ok=True)
SRC=os.path.join(ROLL,'certified_calls_uncertainty.csv')
OUT=OUTP
FIGDIR=OUTP
BENCH=[('arc-challenge',11,'ARC'),('hellaswag',11,'HSwag'),('mmlu',11,'MMLU'),
       ('winogrande',11,'WinoG'),('bbh_boolean_expressions',100,'BBH-b'),
       ('bbh_navigate',100,'BBH-n'),('gpqa_diamond',100,'GPQA'),('ifeval',100,'IFEval'),
       ('math_algebra_hard',100,'MATH-a'),('musr',100,'MuSR')]
COLORS={'CopRoute-Ind':'#4477AA','CopRoute-Dep':'#66CCEE','RACER-G':'#EE6677',
        'RACER-P':'#AA3377','FlexRouter':'#228833','BaRP':'#CCBB44'}
STYLES={'CopRoute-Ind':'-','CopRoute-Dep':'--','RACER-G':'-','RACER-P':'--','FlexRouter':'-','BaRP':'--'}
rows=list(csv.DictReader(open(SRC)))

def calls(bench,pool,alpha):
    sub=[r for r in rows if r['benchmark']==bench and int(r['pool_size'])==pool and r['target_alpha']==alpha]
    out={}
    for r in sub:
        if r['method']=='fixed_tally_chain_discounted':
            if r['reference_method']=='fixed_tally_factorized':
                out['CopRoute-Dep']=float(r['method_certified_mean_calls'])
                out['CopRoute-Ind']=float(r['reference_certified_mean_calls'])
            if r['reference_method']=='racer_g': out['RACER-G']=float(r['reference_certified_mean_calls'])
            if r['reference_method']=='racer_p': out['RACER-P']=float(r['reference_certified_mean_calls'])
    for fam,name in [('flexrouter','FlexRouter'),('barp','BaRP')]:
        vals=[float(r['method_certified_mean_calls']) for s in range(9101,9106) for r in sub
              if r['method']==f'{fam}_seed_{s}' and r['reference_method']=='fixed_tally_factorized' and r['method_certified_mean_calls']]
        if vals: out[name]=st.mean(vals)
    return out

plt.rcParams.update({'font.size':7.5})
fig,axes=plt.subplots(1,2,figsize=(6.6,3.15),subplot_kw=dict(projection='polar'))
data_rows=[]
for ax,alpha,title in [(axes[0],'0.05',r'target risk 0.05'),(axes[1],'0.1',r'target risk 0.10')]:
    labels=[l for _,_,l in BENCH]
    ang=np.linspace(0,2*np.pi,len(labels),endpoint=False)
    eff={p:[] for p in COLORS}
    for bench,pool,_ in BENCH:
        c=calls(bench,pool,alpha)
        best=min(c.values())
        for p in COLORS:
            e=best/c[p]
            eff[p].append(e)
            data_rows.append({'alpha':alpha,'dataset':bench,'policy':p,'certified_mean_calls':c[p],'efficiency':e})
    for p in COLORS:
        v=eff[p]+eff[p][:1]; a=np.concatenate([ang,ang[:1]])
        lw=1.9 if p.startswith('CopRoute') else 1.0
        ax.plot(a,v,STYLES[p],color=COLORS[p],lw=lw,zorder=3 if p.startswith('CopRoute') else 2)
        if p.startswith('CopRoute'):
            ax.fill(a,v,color=COLORS[p],alpha=0.06,zorder=1)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels,fontsize=7)
    ax.set_ylim(0,1.02)
    ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(['.25','.5','.75','1'],fontsize=6)
    ax.set_title(title,fontsize=8.5,pad=12)
    ax.grid(alpha=0.3,lw=0.5)
handles=[plt.Line2D([],[],color=COLORS[p],ls=STYLES[p],lw=1.6,label=p) for p in COLORS]
fig.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(0.5,-0.16),frameon=False,
           fontsize=7,columnspacing=1.1,handletextpad=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR,'radar_call_efficiency.pdf'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(OUT,'radar_call_efficiency.png'),dpi=200,bbox_inches='tight')
with open(os.path.join(OUT,'radar_call_efficiency_data.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(data_rows[0])); w.writeheader(); w.writerows(data_rows)
print('wrote radar,',len(data_rows),'points')
