"""Matched whole-epoch reciprocal A/B; timings include every attempted row."""
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
root=Path('evidence/holonomic/adaptive_extreme_phase11')
parser=argparse.ArgumentParser()
parser.add_argument('--baseline', default='whole_base.tsv')
parser.add_argument('--candidate', default='whole_recip.tsv')
parser.add_argument('--output', default='whole_summary.json')
args=parser.parse_args()
a=pd.read_csv(root/args.baseline,sep=r'\s+',comment='#')
b=pd.read_csv(root/args.candidate,sep=r'\s+',comment='#')
keys=['case_id','profile','d_bin_index','epoch_index','target']
assert a[keys].equals(b[keys]) and len(a)==14432
out={'rows':len(a),'groups':{}}
for tol in [1e-3,1e-4]:
 out['groups'][str(tol)]={}
 for lane in ['cold','warm','radial']:
  mask=a.target==tol
  x,y=a[mask],b[mask]
  delta=abs(x[lane+'_mu']-y[lane+'_mu'])/abs(x.reference)
  out['groups'][str(tol)][lane]={
   'baseline_ms':dict(zip(['p50','p90','p95','p99','max'],map(float,np.quantile(x[lane+'_ms'],[.5,.9,.95,.99,1])))),
   'candidate_ms':dict(zip(['p50','p90','p95','p99','max'],map(float,np.quantile(y[lane+'_ms'],[.5,.9,.95,.99,1])))),
   'baseline_converged':int(x[lane+'_value_converged'].sum()),
   'candidate_converged':int(y[lane+'_value_converged'].sum()),
   'status_mismatches':int((x[lane+'_status']!=y[lane+'_status']).sum()),
   'nodes_mismatches':int((x[lane+'_nodes']!=y[lane+'_nodes']).sum()),
   'max_relative_mu_difference':float(delta.max()),
   'baseline_reference_over_target':int((abs(x[lane+'_mu']-x.reference)>tol*abs(x.reference)).sum()),
   'candidate_reference_over_target':int((abs(y[lane+'_mu']-y.reference)>tol*abs(y.reference)).sum())}
(root/args.output).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
