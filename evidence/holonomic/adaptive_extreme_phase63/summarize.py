import json
from pathlib import Path
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parent
out={}
for folder,expected in [(p,256),(p/'guarded',256)]:
 r=lambda n:pd.read_csv(folder/(n+'.tsv'),sep=r'\s+',float_precision='round_trip')
 b=r('base1');d={}
 for n in ['base1','cache1','cache2','base2']:
  c=r(n);assert len(c)==len(b)==expected
  assert c[['case','profile','db','epoch']].equals(b[['case','profile','db','epoch']])
  assert np.isfinite(c.mu).all()
  q={k:int((c[k]!=b[k]).sum()) for k in ['ok','nodes']}
  q['max_scaled_mu_difference']=float((abs(c.mu-b.mu)/np.maximum(1,abs(b.mu))).max())
  q['timing']={profile:{k:{str(v):float(g[k].quantile(v)) for v in [.5,.9,.95,.99,1]} for k in ['whole','arc','rootpair_ms']} for profile,g in c[c.trajectory_pos>0].groupby('profile')}
  d[n]=q
 out[folder.name]=d
(p/'profile_summary.json').write_text(json.dumps(out,indent=2)+'\n')
