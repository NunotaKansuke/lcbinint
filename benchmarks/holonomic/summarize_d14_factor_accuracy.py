"""Prepare saved-candidate probes or summarize correction errors (not certificates)."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('mode',choices=['prepare','summary'])
p.add_argument('--input',required=True)
p.add_argument('--roots')
p.add_argument('--output',required=True)
a=p.parse_args()
if a.mode=='prepare':
 cols='case_id configuration_id profile d_bin_index epoch_index s q rho x y time u X reference'.split()
 x=pd.read_csv(a.input,sep=r'\s+',comment='#',names=cols,float_precision='round_trip');x['sample']=range(len(x))
 roots=pd.read_csv(a.roots,sep=r'\s+',float_precision='round_trip');roots=roots[roots.lane=='cold']
 roots=roots.merge(x,on=cols[:5],validate='many_to_one')
 assert len(roots)==14*len(x)
 with open(a.output,'w') as f:
  for r in roots.itertuples():
   for stage,re,im in [(0,r.double_seed_v_re,r.double_seed_v_im),(1,r.final_v_re,r.final_v_im)]:
    print(r.sample,r.root_index,stage,r.time,r.y,r.rho,r.q,r.s,re,im,r.qf_nearest_sep,file=f)
else:
 x=pd.read_csv(a.input,sep=r'\s+',float_precision='round_trip');out=[]
 assert len(x)==7216*14*2*3
 for (stage,method),g in x.groupby(['stage','method']):
  finite=np.isfinite(g.correction_error)
  good=finite & (g.scaled_error<1e-12) & (g.separation_ratio<1e-6)
  out.append({'stage':int(stage),'method':int(method),'rows':len(g),'nonfinite':int((~finite).sum()),
   'finite_only_quantiles':{k:{str(t):float(g.loc[finite,k].quantile(t))for t in [.5,.9,.99,1]}for k in ['scaled_error','separation_ratio']},
   'below_both_diagnostic_thresholds':int(good.sum()),
   'all_14_below_both_cases':int(good.groupby(g['sample']).all().sum())})
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
