import argparse,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--probe',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=lambda f:pd.read_csv(f,sep=r'\s+',float_precision='round_trip')
b,c=r(a.baseline),r(a.probe)
assert len(b)==len(c)==7216 and b[['case','profile','db','epoch']].equals(c[['case','profile','db','epoch']])
diff={k:int((b[k]!=c[k]).sum()) for k in ['mu','ok','nodes']};assert not any(diff.values())
out={'rows':len(c),'changed':diff,'groups':{}}
fields=['quartic_cold','quartic_warm','quartic_unseeded','quartic_disabled','quartic_after_warm','quartic_degree_change','quartic_cold_sweeps','quartic_warm_sweeps']
for profile,g in c[c.trajectory_pos>0].groupby('profile'):
 d={k:int(g[k].sum()) for k in fields}
 assert d['quartic_cold']==sum(d[k] for k in ['quartic_unseeded','quartic_disabled','quartic_after_warm','quartic_degree_change'])
 d['cold_sweeps_per_call']=d['quartic_cold_sweeps']/d['quartic_cold']
 d['warm_sweeps_per_call']=d['quartic_warm_sweeps']/d['quartic_warm']
 out['groups'][profile]=d
Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
