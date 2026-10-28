import json,sys,math,os
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
s=(P/'audit.py').read_text().split("if '--worker'in sys.argv:")[0];exec(compile(s,str(P/'audit.py'),'exec'))
os.sched_setaffinity(0,{0})
import VBMicrolensing as V
cases=json.loads((P/'cases.json').read_text());report=dict(status='RUNNING',rows=[],checks=[],protocol='Independent VBM Singlepoly/Multipoly scalar reference at Tol=0/RelTol=1e-7. Per trajectory/LD, 9 regular grid indices plus the 3 largest coarse inter-engine disagreements and first/last native integral indices (union). Reference agreement is diagnostic, not a rigorous error enclosure. Native scalar and tangent scalar values both checked on the selected points; derivative precision is not certified by finite flags.')
def save():(P/'references.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for c in cases:
 for limb in [0.,.5]:
  data=[json.loads((P/'rows'/f"{c['name']}-u{limb}-r{tol}.json").read_text())for tol in [1e-3,1e-4]]
  inds=set(np.linspace(0,c['n']-1,9,dtype=int).tolist())
  for d in data:
   e=d['engines'];mu=np.array(e['value']['diagnostic']['output']['mu']);sp=np.array(e['SinglePoly']['diagnostic']['output']['mu']);inds.update(np.argsort(np.abs(mu/sp-1))[-3:].tolist())
   routes=np.array(e['value']['diagnostic']['output']['routes']);j=np.flatnonzero(routes==7)
   if len(j):inds.update([int(j[0]),int(j[-1])])
  inds=sorted(inds);p=parameters(c);xs=np.linspace(c['xmin'],c['xmax'],c['n']);all_times=p['q']/(1+p['q'])*p['s']-xs;times=all_times[inds].tolist();refs={}
  for method in ['Singlepoly','Multipoly']:
   v=V.VBMicrolensing();v.SetMethod(getattr(v.Method,method));v.Tol=0.;v.RelTol=1e-7;v.a1=limb;v.a2=0.;v.SetLDprofile(v.LDlinear)
   z=v.TripleLightCurve(vparams(p),times);refs[method]=list(z[0])
  reference_gap=np.abs(np.array(refs['Singlepoly'])/refs['Multipoly']-1)
  row=dict(name=c['name'],limb=limb,indices=inds,params=p,epochs=times,references=refs,max_reference_disagreement=float(reference_gap.max()),checks=[])
  for d in data:
   for mode in ['value','gradient']:
    out=d['engines'][mode]['diagnostic']['output'];mu=np.array(out['mu'])[inds];errs={k:np.abs(mu/z-1).tolist()for k,z in refs.items()}
    r=dict(name=c['name'],limb=limb,rtol=d['condition']['rtol'],mode=mode,relative_errors=errs,max_relative_error=max(max(v)for v in errs.values()),above_requested={k:sum(x>d['condition']['rtol']for x in v)for k,v in errs.items()})
    row['checks'].append(r);report['checks'].append(r)
  report['rows'].append(row);save();print(c['name'],limb,'n',len(inds),'reference gap',float(reference_gap.max()),'native max',max(r['max_relative_error']for r in row['checks']),flush=True)
report['status']='COMPLETE';save()
