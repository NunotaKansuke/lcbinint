import sys,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;sys.meta_path=[x for x in sys.meta_path if '_lcbinint_editable'not in str(x)];sys.path.insert(0,str(P/'guarded-package'));import lcbinint as lc
c=json.load(open(P/'inputs/0483.json'));p=c['params'];ts=np.array(c['epochs']);m=lc.LightCurve(lens='triple',param_type='vbm',ld_c=c['limb'],options=lc.Options(finite_source_tol=0.,finite_source_reltol=c['rtol']))
def clean(x):
 if hasattr(x,'tolist'):return clean(x.tolist())
 if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v)for v in x]
 if isinstance(x,float)and not math.isfinite(x):return None
 return x
rows=[]
for name,t,par in [('base',ts,p),('period',ts,dict(p,alpha=p['alpha']+2*math.pi,ang=p['ang']+2*math.pi)),('shift',ts+7,dict(p,t0=p['t0']+7)),('reverse',np.r_[ts[::-1],ts[0]],p)]:
 r=m._native._evaluate_algebraic_boundary(t,par,0.,c['rtol'],True,True);rows.append(dict(name=name,params=par,times=t.tolist(),raw=clean(r)));print(name,r['mu'],r['stop'],flush=True)
(P/'transform483.json').write_text(json.dumps(rows,indent=2))
