import sys,json,math,os
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable'not in str(f)];sys.path.insert(0,str(P/'adopted-package'));import lcbinint as lc
c=json.load(open(P/'followup/inputs/0009.json'));p=c['params'];times=np.array(c['epochs']);m=lc.LightCurve(lens='triple',param_type='vbm',ld_c=c['limb'],options=lc.Options(finite_source_tol=0.,finite_source_reltol=c['rtol']))
def clean(x):
 if hasattr(x,'tolist'):return clean(x.tolist())
 if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v)for v in x]
 if isinstance(x,float)and not math.isfinite(x):return None
 return x
rows=[]
for n in range(3):
 r=m._native._evaluate_algebraic_boundary(times,p,0.,c['rtol'],True,True);rows.append(clean(r));print(n,r['mu'],r['stop'],flush=True)
(P/'adopted-cold-probe.json').write_text(json.dumps(rows,indent=2))
