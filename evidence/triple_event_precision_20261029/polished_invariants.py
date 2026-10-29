import sys,json,math,hashlib,resource,os
os.sched_setaffinity(0,{0});resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent
sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable'not in str(f)];sys.path.insert(0,str(P/'polished-package'))
import lcbinint as lc
cases=json.loads((P/'cases.json').read_text());rows=[]
selected=set(range(0,400,3))|{r['case']for r in json.loads((P.parent/'triple_final_polish_20261029/pilot-contact.json').read_text())}|{135,189,314,337,361,431,444,483,504,514,940,972,1508}
for i in sorted(selected):
 c=cases[i]
 f=P/'polished-rows'/f'{i:04d}-gradient.json'
 if not f.exists()or json.loads(f.read_text())['classification']!='ok':continue
 p=c['params'];ts=np.asarray(c['epochs']);r={'case':i,'native_sha256':hashlib.sha256(Path(lc._native.__file__).read_bytes()).hexdigest()}
 try:
  m=lc.LightCurve(lens='triple',param_type='vbm',options=lc.Options(finite_source_tol=0.,finite_source_reltol=c['rtol']),ld_c=c['limb'])
  a=m.magnification(ts,p,with_gradient=True);mu=np.asarray(a.magnifications);g=np.asarray(a.gradients)
  checks={}
  r['base']={'mu':mu.tolist(),'gradient':g.tolist(),'parameters':list(a.gradient_parameters)}
  for name,t,par,idx in [('period',ts,dict(p,alpha=p['alpha']+2*math.pi,ang=p['ang']+2*math.pi),np.arange(len(ts))),('epoch_shift',ts+7,dict(p,t0=p['t0']+7),np.arange(len(ts))),('reverse_duplicate',np.r_[ts[::-1],ts[0]],p,np.r_[np.arange(len(ts))[::-1],0])]:
   b=m.magnification(t,par,with_gradient=True);mb=np.asarray(b.magnifications);gb=np.asarray(b.gradients)
   checks[name]={'mu':mb.tolist(),'gradient':gb.tolist(),'value_relative':float(np.max(abs(mb/mu[idx]-1))),'gradient_scaled':float(np.max(abs(gb-g[idx])/(1+abs(g[idx])))),'status':b.status.tolist(),'valid':b.gradient_valid.tolist()}
  r['checks']=checks
 except Exception as e:r['error']=repr(e)
 rows.append(r);(P/'polished-invariants.json').write_text(json.dumps(rows,indent=2));print(i,r.get('error',r.get('checks')),flush=True)
