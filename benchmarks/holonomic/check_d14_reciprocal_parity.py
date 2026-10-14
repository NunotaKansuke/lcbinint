"""Compare final root sets by bijection, not incidental solver index."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='evidence/holonomic/adaptive_extreme_phase11')
parser.add_argument('--root-base',default='evidence/holonomic/adaptive_extreme_phase11/p11_roots_base.tsv.gz')
parser.add_argument('--root-candidate',default='evidence/holonomic/adaptive_extreme_phase11/p11_roots_recip.tsv.gz')
parser.add_argument('--event-base',default='evidence/holonomic/adaptive_extreme_phase11/p11_events_base.tsv.gz')
parser.add_argument('--event-candidate',default='evidence/holonomic/adaptive_extreme_phase11/p11_events_recip.tsv.gz')
args=parser.parse_args()
out=Path(args.output)
a=pd.read_csv(args.root_base,sep=r'\s+')
b=pd.read_csv(args.root_candidate,sep=r'\s+')
keys=['case_id','configuration_id','profile','d_bin_index','epoch_index','rep','lane']
assert len(a)==len(b)==14432*14
assert a[keys].drop_duplicates().sort_values(keys).reset_index(drop=True).equals(b[keys].drop_duplicates().sort_values(keys).reset_index(drop=True))
records=[]
for (key,x),(key2,y) in zip(a.groupby(keys,sort=True),b.groupby(keys,sort=True)):
 assert key==key2 and len(x)==len(y)==14
 z=x.final_v_re.to_numpy()+1j*x.final_v_im.to_numpy();w=y.final_v_re.to_numpy()+1j*y.final_v_im.to_numpy()
 cost=abs(z[:,None]-w[None,:]);i,j=linear_sum_assignment(cost)
 error=cost[i,j]/(1+abs(z[i]))
 records.append(dict(zip(keys,key))|{'max_scaled_root_delta':float(max(error)), 'role_mismatches':int(np.count_nonzero(x.role.to_numpy()[i]!=y.role.to_numpy()[j])),'physical_mismatches':int(np.count_nonzero(x.physical_real.to_numpy()[i]!=y.physical_real.to_numpy()[j]))})
assert len(records)==14432
pd.DataFrame(records).to_csv(out/'root_parity.tsv',sep='\t',index=False)
x=pd.read_csv(args.event_base,sep=r'\s+');y=pd.read_csv(args.event_candidate,sep=r'\s+')
assert x[keys].equals(y[keys])
fields=['topology_status','cells','events','physical_real_events','soft_events','qf_cold_calls','completeness_fails']
summary={'rows':len(records),'max_scaled_root_delta':max(r['max_scaled_root_delta'] for r in records),'role_mismatches':sum(r['role_mismatches'] for r in records),'physical_mismatches':sum(r['physical_mismatches'] for r in records),'event_counter_mismatches':{c:int((x[c]!=y[c]).sum()) for c in fields},'qf_cold_total':{'baseline':int(x.qf_cold_calls.sum()),'candidate':int(y.qf_cold_calls.sum())},'limitation':'Final root coordinates captured as binary64; compares roles and event counts, not interval enclosures or every cell endpoint.'}
(out/'root_parity_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
