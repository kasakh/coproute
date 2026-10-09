#!/usr/bin/env python3
"""Sweep figure: certified mean calls against target risk alpha.

Four pool-100 panels chosen for the paper: MuSR, BBH-navigate, IFEval,
MATH-algebra. Six policies with reader-facing names only. Filled markers
mean at least 90 percent of redraws certify; open markers mean fewer.
Source: issue62 canonical rollup. Writes PDF (300 dpi) + PNG (200 dpi)
into figures_out/paper.
"""
import csv, statistics as st, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pathlib
REPO_ROOT=pathlib.Path(__file__).resolve().parents[2]
ROLL=str(REPO_ROOT/'results'/'paper_rollups')
OUTP=str(REPO_ROOT/'figures_out'/'paper')
pathlib.Path(OUTP).mkdir(parents=True,exist_ok=True)
SRC=os.path.join(ROLL,'certified_calls_uncertainty.csv')
OUT_DIR=OUTP
FIGDIR=OUTP
ALPHAS=[0.02,0.03,0.05,0.075,0.1,0.15,0.2]
PANELS=[('musr','MuSR'),('bbh_navigate','BBH-navigate'),('ifeval','IFEval'),('math_algebra_hard','MATH-algebra')]
POOL=100
# Tol bright
COLORS={'CopRoute-Ind':'#4477AA','CopRoute-Dep':'#66CCEE','RACER-G':'#EE6677',
        'RACER-P':'#AA3377','FlexRouter':'#228833','BaRP':'#CCBB44'}
STYLES={'CopRoute-Ind':'-','CopRoute-Dep':'--','RACER-G':'-','RACER-P':'--','FlexRouter':'-','BaRP':'--'}
MARKERS={'CopRoute-Ind':'o','CopRoute-Dep':'s','RACER-G':'^','RACER-P':'v','FlexRouter':'D','BaRP':'P'}

rows=list(csv.DictReader(open(SRC)))

def series(bench,policy):
    xs,ys,certs=[],[],[]
    for a in ALPHAS:
        sa=str(a)
        sub=[r for r in rows if r['benchmark']==bench and int(r['pool_size'])==POOL and r['target_alpha']==sa]
        val=cert=None
        if policy in ('CopRoute-Ind','CopRoute-Dep','RACER-G','RACER-P'):
            for r in sub:
                if r['method']!='fixed_tally_chain_discounted': continue
                if policy=='CopRoute-Dep' and r['reference_method']=='fixed_tally_factorized':
                    val=r['method_certified_mean_calls']; cert=r['method_certification_rate']
                elif policy=='CopRoute-Ind' and r['reference_method']=='fixed_tally_factorized':
                    val=r['reference_certified_mean_calls']; cert=r['reference_certification_rate']
                elif policy=='RACER-G' and r['reference_method']=='racer_g':
                    val=r['reference_certified_mean_calls']; cert=r['reference_certification_rate']
                elif policy=='RACER-P' and r['reference_method']=='racer_p':
                    val=r['reference_certified_mean_calls']; cert=r['reference_certification_rate']
        else:
            fam='flexrouter' if policy=='FlexRouter' else 'barp'
            vals=[];cs=[]
            for s in range(9101,9106):
                for r in sub:
                    if r['method']==f'{fam}_seed_{s}' and r['reference_method']=='fixed_tally_factorized' and r['method_certified_mean_calls']:
                        vals.append(float(r['method_certified_mean_calls']));cs.append(float(r['method_certification_rate']))
            if vals: val,cert=st.mean(vals),st.mean(cs)
        if val not in (None,''):
            xs.append(a);ys.append(float(val));certs.append(float(cert))
    return xs,ys,certs

plt.rcParams.update({'font.size':8.5,'axes.labelsize':9,'axes.titlesize':9.5,
                     'legend.fontsize':8,'mathtext.default':'regular'})
fig,axes=plt.subplots(1,4,figsize=(11.0,2.55))
data_rows=[]
for ax,(bench,title) in zip(axes,PANELS):
    for pol in COLORS:
        xs,ys,certs=series(bench,pol)
        if not xs: continue
        lw=2.0 if pol.startswith('CopRoute') else 1.3
        ax.plot(xs,ys,STYLES[pol],color=COLORS[pol],lw=lw,zorder=3 if pol.startswith('CopRoute') else 2)
        for x,y,c in zip(xs,ys,certs):
            filled=c>=0.9
            ax.plot([x],[y],MARKERS[pol],color=COLORS[pol],ms=4.5,
                    mfc=COLORS[pol] if filled else 'white',mew=1.0,zorder=4)
            data_rows.append({'dataset':title,'policy':pol,'alpha':x,'certified_mean_calls':y,'certification_rate':c})
    ax.set_title(title)
    ax.set_xlabel(r'target risk $\alpha$')
    ax.set_yscale('log')
    ax.set_xticks([0.02,0.05,0.1,0.15,0.2])
    ax.grid(True,alpha=0.25,lw=0.5)
axes[0].set_ylabel('certified mean calls')
handles=[plt.Line2D([],[],color=COLORS[p],ls=STYLES[p],marker=MARKERS[p],ms=4.5,lw=1.6,label=p) for p in COLORS]
handles.append(plt.Line2D([],[],color='grey',marker='o',ls='',ms=4.5,mfc='white',label='below 90% certification'))
fig.legend(handles=handles,ncol=7,loc='upper center',bbox_to_anchor=(0.5,1.14),frameon=False,columnspacing=1.0,handletextpad=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR,'sweep_calls_vs_target.pdf'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR,'sweep_calls_vs_target.png'),dpi=200,bbox_inches='tight')
with open(os.path.join(OUT_DIR,'sweep_calls_vs_target_data.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(data_rows[0])); w.writeheader(); w.writerows(data_rows)
print('wrote sweep figure,',len(data_rows),'points')
