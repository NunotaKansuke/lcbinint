from pathlib import Path
import os,sys,json,time,subprocess
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
def worker(c):
 sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
 np,l,VB=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});p=c['params'];tol=1e-3
 times=np.linspace(-c['span'],c['span'],33)*p['tE'];opts=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary')
 curve=l.LightCurve(options=opts,limb_darkening=l.LimbDarkening.linear(c['limb']))
 t=time.perf_counter();info=curve.info(times,p);ms=(time.perf_counter()-t)*1000;a=np.asarray(info['mu'])
 reverse=np.asarray(curve(times[::-1],p))[::-1];order_diff=float(np.nanmax(abs(a-reverse)/np.maximum(abs(a),1e-12))) if np.isfinite(a).any() else None
 v=VB.VBMicrolensing();v.Tol=v.RelTol=tol;v.a1=c['limb'];v.a2=0
 vp=[np.log(p['s']),np.log(p['q']),p['u0'],p['alpha'],np.log(p['rho']),np.log(p['tE']),p['t0']]
 print('native value completed',c['id'],flush=True)
 np.savez_compressed(OUT/(c['id']+'_partial.npz'),times=times,mu=a,reverse=reverse,converged=info['value_converged'])
 t=time.perf_counter();b=np.asarray(v.BinaryLightCurve(vp,times.tolist())[0]);vbms=(time.perf_counter()-t)*1000
 rel=abs(a-b)/np.maximum(abs(b),1e-12);g=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol);grad=np.asarray(g['grad_mu'])
 np.savez_compressed(OUT/(c['id']+'.npz'),times=times,mu=a,vbm=b,reverse=reverse,gradient=grad,converged=info['value_converged'],nodes=info['node_evaluations'],support=info['support_certified'])
 r={**c,'status':'ok','ms':ms,'vbm_ms':vbms,'finite':int(np.isfinite(a).sum()),'converged':int(np.asarray(info['value_converged']).sum()),'finite_grad':int(np.isfinite(grad).sum()),'max_rel_vbm':float(np.nanmax(rel)),'worst':int(np.nanargmax(rel)) if np.isfinite(rel).any() else None,'order_diff':order_diff,'nodes':int(np.sum(info['node_evaluations']))}
 (OUT/(c['id']+'.json')).write_text(json.dumps(r,indent=2)+'\n')
def run():
 import numpy as np
 rng=np.random.default_rng(20261015);cases=[]
 # Stratified logarithmic random matrix plus exact/near degeneracies.
 for i in range(120):
  s=10**rng.uniform(-1,1);q=10**rng.uniform(-7,0);rho=10**rng.uniform(-5,.3)
  if i<24:s=[.1,.6,.999999,1.,1.000001,2.,10.,1.5][i%8];q=[1.,1e-7,1e-3][i//8]
  u0=float(rng.choice([0.,1e-9,-1e-5,float(rng.uniform(-.5,.5))]));alpha=float(rng.uniform(-np.pi,np.pi))
  cases.append(dict(id=f'case_{i:03}',params=dict(s=float(s),q=float(q),rho=float(rho),t0=0.,tE=10.,u0=u0,alpha=alpha),limb=float([0.,.5,.9,.999999][i%4]),span=float(max(.03,min(2.,rho*2)))))
 (OUT/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
 for c in cases:
  f=OUT/(c['id']+'.json')
  if f.exists():continue
  try:
   r=subprocess.run([sys.executable,str(Path(__file__)),'worker',json.dumps(c)],cwd=ROOT,env=os.environ,capture_output=True,text=True,timeout=20)
   if r.returncode:raise RuntimeError(r.stderr[-2500:])
   d=json.loads(f.read_text());print(c['id'],d['finite'],d['converged'],round(d['ms'],2),d['max_rel_vbm'],d['order_diff'],flush=True)
  except Exception as e:
   f.write_text(json.dumps({**c,'status':'failed','error':str(e)},indent=2)+'\n');print(c['id'],'FAILED',str(e)[-200:],flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(json.loads(sys.argv[2]))
 else:run()
