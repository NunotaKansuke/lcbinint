"""Counterfactual cost scaling on matched rows; not achievable speed claims."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--matched',required=True);p.add_argument('--whole',required=True);p.add_argument('--output',required=True);a=p.parse_args()
k=['case_id','profile','d_bin_index','epoch_index','target']
r=lambda f:pd.read_csv(f,sep=r'\s+',comment='#',float_precision='round_trip')
m=r(a.matched);w=r(a.whole)
assert not m.duplicated(k).any() and not w.duplicated(k).any()
w=w[k+['warm_nodes','warm_panels','warm_splits','trajectory_pos','warm_value_converged','warm_error','warm_mu']]
x=m.merge(w,on=k,validate='one_to_one');assert len(x)==len(m)==len(w)==14432
assert (x.warm_value_converged==1).all()
x['other_ms']=x.warm_ms-x.warm_topology_ms-x.warm_physical_ms
assert np.isfinite(x.other_ms).all() and (x.other_ms>=0).all()
out={'sources':{f:{'path':getattr(a,f),'sha256':hashlib.sha256(Path(getattr(a,f)).read_bytes()).hexdigest()} for f in ['matched','whole']},'groups':{},'caveat':'Counterfactual scales keep mesh and other costs fixed; neither measured optimization nor certified runtime lower bound.'}
front=[]
for (tol,profile),g in x.groupby(['target','profile']):
 d={'rows':len(g),'zero_split_rows':int((g.warm_splits==0).sum()),'nodes_p50':float(g.warm_nodes.median()),'panels_p50':float(g.warm_panels.median()),'nodes_per_panel_p50':float((g.warm_nodes/g.warm_panels).median()),'unclassified_remainder_p50_ms':float(g.other_ms.median())}
 # Hypothetical minimum-level mesh, not a quadrature accuracy claim.
 initial_fraction=np.minimum(1.,7*g.warm_panels/g.warm_nodes)
 d['initial_level_only_physics_fraction_p50']=float(initial_fraction.median())
 d['initial_level_only_with_current_topology_speedup']=float((g.vbm_ms/(g.other_ms+g.warm_topology_ms+initial_fraction*g.warm_physical_ms)).median())
 d['initial_level_only_with_zero_topology_speedup']=float((g.vbm_ms/(g.other_ms+initial_fraction*g.warm_physical_ms)).median())
 # More conservative margin than mere ratio>1, to avoid calling a tie a win.
 for threshold in [1.,1.15]:
  # Largest common retained fraction of topology+physics reaching paired median.
  lo,hi=0.,1.
  if np.median(g.vbm_ms/g.other_ms)<threshold:
   answer=None
  else:
   for _ in range(60):
    mid=(lo+hi)/2
    ratio=np.median(g.vbm_ms/(g.other_ms+mid*(g.warm_topology_ms+g.warm_physical_ms)))
    if ratio>=threshold:lo=mid
    else:hi=mid
   answer=lo
  d['retained_joint_fraction_for_'+str(threshold)]=answer
 for top in [0.,.25,.5,.75,1.]:
  for phys in [0.,.25,.5,.75,1.]:
   t=g.other_ms+top*g.warm_topology_ms+phys*g.warm_physical_ms
   front.append({'target':tol,'profile':profile,'topology_fraction':top,'physics_fraction':phys,'p50_ms':float(t.median()),'paired_speedup_median':float((g.vbm_ms/t).median()),'faster_rows':int((t<g.vbm_ms).sum())})
 out['groups'][f'{tol:g}/{profile}']=d
path=Path(a.output);path.mkdir(parents=True,exist_ok=True)
(path/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
pd.DataFrame(front).to_csv(path/'frontier.tsv',sep='\t',index=False)
x[k+['warm_nodes','warm_panels','warm_splits','other_ms']].to_csv(path/'mesh.tsv',sep='\t',index=False)
print(json.dumps(out['groups'],indent=2))
