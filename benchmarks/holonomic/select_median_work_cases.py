"""Select diagnostic trajectories by observed ordinary-epoch runtime, not routing."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--whole',required=True);p.add_argument('--profile',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=lambda f:pd.read_csv(f,sep=r'\s+',comment='#',float_precision='round_trip')
w=r(a.whole);pr=r(a.profile).rename(columns={'case':'case_id','db':'d_bin_index','epoch':'epoch_index'})
g=w[(w.target==1e-3)&(w.profile=='uniform')&(w.trajectory_pos>0)].copy()
assert len(g)==2706
lo,hi=g.warm_ms.quantile([.4,.6]);g=g[g.warm_ms.between(lo,hi)].sort_values('warm_ms')
keys=['case_id','profile','d_bin_index','epoch_index']
assert not pr.duplicated(keys).any()
m=g.merge(pr,on=keys,validate='one_to_one',suffixes=('','_profile'))
assert len(m)==len(g)
assert np.array_equal(m.warm_mu,m.mu) and np.array_equal(m.warm_nodes,m.nodes)
# Draw from throughout the central band rather than just one median neighbor.
traj=['case_id','configuration_id','d_bin_index']
u=m.drop_duplicates(traj)
chosen=u.iloc[np.unique(np.linspace(0,len(u)-1,min(32,len(u))).astype(int))]
selected=set(map(tuple,chosen[traj].to_numpy()))
lines=[]
for line in Path(a.input).read_text().splitlines():
 t=line.split()
 if not t or t[0].startswith('#'):continue
 try:k=(int(t[0]),int(t[1]),int(t[3]))
 except ValueError:continue
 if k in selected:lines.append(line)
assert len(lines)==len(selected)*8, (len(lines),len(selected))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
(out/'input_snapshot.tsv').write_text('\n'.join(lines)+'\n')
chosen[keys+['warm_ms','warm_nodes','warm_panels','topology','physics','arc','endpoint','presearch','real','residual','metadata']].to_csv(out/'selected.tsv',sep='\t',index=False)
cols=['topology','setup','physics','estimator','scheduler','arc','endpoint','presearch','real','residual','metadata','classification','soft','probes']
s={'sources':{k:{'path':getattr(a,k),'sha256':hashlib.sha256(Path(getattr(a,k)).read_bytes()).hexdigest()} for k in ['whole','profile','input']},'central_band_rows':len(m),'band_warm_ms':[float(lo),float(hi)],'selected_trajectories_per_profile':len(selected),'selected_input_rows':len(lines),'central_band_profile_medians_ms':{c:float(m[c].median()) for c in cols},'scope':'Diagnostic subset only. Stages include parent/child overlap; do not sum. Historical profile time is not the uninstrumented whole time.'}
(out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
