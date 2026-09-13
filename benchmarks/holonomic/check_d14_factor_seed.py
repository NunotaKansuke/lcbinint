"""Compare discovery candidates with saved all-root oracle, without accepting them."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--oracle',required=True);p.add_argument('--candidate',required=True);p.add_argument('--timing');p.add_argument('--output',required=True);a=p.parse_args()
cols='case_id configuration_id profile d_bin_index epoch_index s q rho x y time u X reference'.split()
x=pd.read_csv(a.input,sep=r'\s+',comment='#',names=cols,float_precision='round_trip');x['sample']=range(len(x))
gold=pd.read_csv(a.oracle,sep=r'\s+',float_precision='round_trip');gold=gold[gold.lane=='cold'].merge(x[cols[:5]+['sample']],on=cols[:5],validate='many_to_one')
golds={int(i):g.final_v_re.to_numpy()+1j*g.final_v_im.to_numpy() for i,g in gold.groupby('sample')}
c=pd.read_csv(a.candidate,sep=r'\s+',float_precision='round_trip');rows=[]
for (sample,stage),g in c.groupby(['sample','stage']):
 z=g.re.to_numpy()+1j*g.im.to_numpy();ref=golds[int(sample)];finite=np.isfinite(z).all()
 error=float('inf');dups=14;real_delta=99
 if finite:
  cost=np.abs(z[:,None]-ref[None,:])/(1+np.abs(ref[None,:]));i,j=linear_sum_assignment(cost);error=float(cost[i,j].max())
  dups=14-len(np.unique(np.argmin(cost,axis=1)))
  real_delta=int(np.sum((z.real>0)&(np.abs(z.imag)<=1e-8*(1+np.abs(z.real))))-np.sum((ref.real>0)&(np.abs(ref.imag)<=1e-8*(1+np.abs(ref.real)))))
 rows.append({'sample':int(sample),'stage':int(stage),'finite':bool(finite),'max_scaled_root_error':error,'nearest_assignment_duplicates':dups,'positive_real_candidate_count_delta':real_delta})
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);r=pd.DataFrame(rows);r.to_csv(out/'parity.tsv',sep='\t',index=False)
summary=[]
for stage,g in r.groupby('stage'):
 summary.append({'stage':int(stage),'rows':len(g),'finite':int(g.finite.sum()),'all_roots_error_below_1e8':int((g.max_scaled_root_error<1e-8).sum()),'all_roots_error_below_1e10':int((g.max_scaled_root_error<1e-10).sum()),'nearest_assignment_duplicate_cases':int((g.nearest_assignment_duplicates>0).sum()),'positive_real_count_mismatch':int((g.positive_real_candidate_count_delta!=0).sum())})
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if a.timing:
 t=pd.read_csv(a.timing,sep=r'\s+',float_precision='round_trip')
 counts=(gold.physical_real==1).groupby(gold['sample']).sum()
 t['physical_oracle']=t['sample'].map(counts)
 physical={str(stage):{'physical_count_mismatch':int((t[f'physical{stage}']!=t.physical_oracle).sum()),'missing_physical_cases':int((t[f'physical{stage}']<t.physical_oracle).sum()),'extra_physical_cases':int((t[f'physical{stage}']>t.physical_oracle).sum())}for stage in [0,2,4]}
 (out/'physical_summary.json').write_text(json.dumps(physical,indent=2)+'\n')
 t[['sample','physical_oracle','physical0','physical2','physical4']].to_csv(out/'physical_counts.tsv',sep='\t',index=False)
 times={k:{str(q):float(t[k].quantile(q))for q in [.5,.9,.99]}for k in ['factor_ms','split_ms','refine2_ms','refine4_ms','classify_ms','low_sweeps']}
 (out/'timing_summary.json').write_text(json.dumps(times,indent=2)+'\n')
