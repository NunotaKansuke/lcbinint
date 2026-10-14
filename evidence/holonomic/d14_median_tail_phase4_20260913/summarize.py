#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd

here=Path(__file__).resolve().parent
keys=['case_id','configuration_id','profile','d_bin_index','epoch_index','target']
def read(name): return pd.read_csv(here/name,sep=r'\s+',comment='#')
b=read('whole_baseline.tsv.gz')
out={'rows':len(b),'variants':{}}
for name in ['active','combined']:
 c=read(f'whole_{name}.tsv.gz');x=b.merge(c,on=keys,suffixes=('_baseline','_candidate'),validate='one_to_one')
 result={}
 for target in sorted(x.target.unique(),reverse=True):
  y=x[x.target==target];result[f'{target:g}']={}
  for lane in ['cold','warm','radial']:
   old=y[f'{lane}_ms_baseline'];new=y[f'{lane}_ms_candidate']
   stats={'coverage':int(y[f'{lane}_value_converged_candidate'].sum()),
          'status_mismatches':int((y[f'{lane}_status_baseline']!=y[f'{lane}_status_candidate']).sum()),
          'mu_max_absdiff':float(np.max(np.abs(y[f'{lane}_mu_baseline']-y[f'{lane}_mu_candidate'])))}
   for q,label in [(.5,'p50'),(.9,'p90'),(.95,'p95'),(.99,'p99'),(1,'max')]:
    a=float(old.quantile(q));d=float(new.quantile(q));stats[label]={'baseline_ms':a,'candidate_ms':d,'delta_percent':100*(d/a-1)}
   result[f'{target:g}'][lane]=stats
 out['variants'][name]=result
(here/'summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
