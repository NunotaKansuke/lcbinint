import json
from pathlib import Path
import pandas as pd
p=Path(__file__).resolve().parent
read=lambda n:pd.read_csv(p/(n+'.tsv'),sep=r'\s+',float_precision='round_trip')
b=read('base1');keys=['case','profile','db','epoch']
out={'rows':7216,'runs':{},'timing_scope':'warm whole with V2Profile timers, first trajectory epoch excluded from timing summary'}
for n in ['base1','cache1','cache2','base2']:
 c=read(n)
 assert len(c)==len(b)==7216 and c[keys].equals(b[keys])
 diff={k:int((b[k]!=c[k]).sum()) for k in ['mu','ok','nodes']}
 assert not any(diff.values())
 d={}
 for profile,g in c[c.trajectory_pos>0].groupby('profile'):
  d[profile]={k:{str(q):float(g[k].quantile(q)) for q in [.5,.9,.95,.99,1]} for k in ['whole','physics','endpoint','arc','rootpair_ms']}
 out['runs'][n]={'changed':diff,'timing_ms':d, 'newton_iterations':{str(k):int(v) for k,v in c[c.trajectory_pos>0].groupby('profile').newton_iterations.sum().items()}}
(p/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
