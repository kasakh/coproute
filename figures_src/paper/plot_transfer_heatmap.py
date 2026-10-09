#!/usr/bin/env python3
"""Out-of-domain transfer heatmaps for CopRoute-Ind.

Cell (source s, target t): extra certified mean calls on target t when the
ordering is fit on source s and the cutoff is recalibrated on t, relative to
fitting on t itself (in-domain). Diagonal is zero by construction. Two
panels: RouterBench pool 11 and RouterEval pool 100, alpha 0.10 (all cells
certify). Source: the per-pair transfer analyses.
"""
import csv, glob, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pathlib
REPO_ROOT=pathlib.Path(__file__).resolve().parents[2]
ROLL=str(REPO_ROOT/'results'/'paper_rollups')
OUTP=str(REPO_ROOT/'figures_out'/'paper')
pathlib.Path(OUTP).mkdir(parents=True,exist_ok=True)
OUT_DIR=OUTP
FIGDIR=OUTP
RB_DIR=os.path.join(ROLL,'transfer','routerbench')
RE_DIR=os.path.join(ROLL,'transfer','routereval')
ALPHA='0.1'
METHOD='fixed_tally_factorized'
RB=[('arc-challenge','ARC'),('hellaswag','HellaSwag'),('mmlu','MMLU'),('winogrande','WinoGrande')]
REV=[('bbh_boolean_expressions','BBH-bool'),('bbh_navigate','BBH-nav'),('gpqa_diamond','GPQA'),
     ('ifeval','IFEval'),('math_algebra_hard','MATH-alg'),('musr','MuSR')]

def load_calls(directory,src,tgt,pool,mode):
    pat=os.path.join(directory,f'{src}_to_{tgt}_{mode}_pool{pool}_analysis.csv')
    fs=glob.glob(pat)
    if not fs: return None,None
    for r in csv.DictReader(open(fs[0])):
        if r['alpha']==ALPHA and r['method']==METHOD:
            c=r['target_recalibrated_mean_calls']
            cr=r['target_recalibrated_certification_rate']
            return (float(c) if c else None),(float(cr) if cr else None)
    return None,None

def grid(directory,names,pool):
    n=len(names)
    M=np.full((n,n),np.nan)
    rows=[]
    for i,(s,_) in enumerate(names):
        for j,(t,_) in enumerate(names):
            mode='in_domain' if s==t else 'source_fitted'
            calls,cert=load_calls(directory,s,t,pool,mode)
            indom,_=load_calls(directory,t,t,pool,'in_domain')
            if calls is not None and indom is not None:
                M[i,j]=calls-indom
                rows.append({'source':s,'target':t,'pool':pool,'alpha':ALPHA,
                             'source_fitted_calls':calls,'in_domain_calls':indom,
                             'premium':calls-indom,'certification_rate':cert})
    return M,rows

fig,axes=plt.subplots(1,2,figsize=(8.6,3.0),gridspec_kw={'width_ratios':[4,6]})
all_rows=[]
for ax,(directory,names,pool,title,vcap) in zip(axes,[(RB_DIR,RB,11,'RouterBench (pool 11)',None),
                                                 (RE_DIR,REV,100,'RouterEval (pool 100)',12.0)]):
    M,rows=grid(directory,names,pool)
    all_rows+=rows
    vmax=np.nanmax(M) if vcap is None else vcap
    vmax=max(vmax,0.5)
    cmap=plt.get_cmap('Reds').copy()
    im=ax.imshow(np.clip(M,0,vmax),cmap=cmap,vmin=0,vmax=vmax)
    labs=[l for _,l in names]
    ax.set_xticks(range(len(labs)),labs,rotation=45,ha='right',fontsize=7.5)
    ax.set_yticks(range(len(labs)),labs,fontsize=7.5)
    ax.set_xlabel('evaluated on',fontsize=8.5)
    ax.set_title(title,fontsize=9.5)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if np.isnan(M[i,j]):
                ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,color='0.92'))
                ax.text(j,i,'n/r',ha='center',va='center',fontsize=6.8,color='0.4')
            else:
                shade=M[i,j]>0.65*vmax
                ax.text(j,i,f"{M[i,j]:+.1f}" if i!=j else "0",ha='center',va='center',fontsize=6.8,
                        color='white' if shade else 'black')
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03)
    cb.ax.tick_params(labelsize=7)
    if vcap is not None:
        cb.set_label('extra calls (capped)',fontsize=7)
axes[0].set_ylabel('order fit on',fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR,'ood_call_premium.pdf'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR,'ood_call_premium.png'),dpi=200,bbox_inches='tight')
with open(os.path.join(OUT_DIR,'ood_call_premium_data.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
print('wrote OOD heatmap,',len(all_rows),'cells')
import numpy as _np
