from pathlib import Path
import sys,os,json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
if sys.argv[1]=='orders':
 cases=json.loads((ROOT/'evidence/holonomic/lightcurve_matrix_20261015/cases.json').read_text());rng=np.random.default_rng(414)
 for c in cases:
  if c['case'] not in ['large_source','broad_source','equal_mass','close_planet'] or c['limb'] not in [0,.5] or c['tol']!=1e-3:continue
  base=c['params']['tE']*np.linspace(c['tau_min'],c['tau_max'],41)
  for order,times in [('reverse',base[::-1]),('shuffle_repeat',np.r_[rng.permutation(base),base[::3]])]:
   config={**c,'id':c['id']+'_'+order,'times':times.tolist(),'n_times':len(times)}
   for variant in ['before','final']:
    f=OUT/f"{config['id']}_{variant}.json"
    if not f.exists():subprocess.run([sys.executable,str(OUT/'bench.py'),'worker',json.dumps(config),variant],cwd=ROOT,env=os.environ,check=True,timeout=40)
elif sys.argv[1]=='gradients':
 sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
 np,l,VB=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});rows=[]
 for c in json.loads((ROOT/'evidence/holonomic/lightcurve_matrix_20261015/cases.json').read_text()):
  old=np.load(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'/(c['id']+'.npz'));times=old['times'];p=c['params'];tol=c['tol'];curve=l.LightCurve(options=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(c['limb']))
  g=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol);a=np.asarray(g['grad_mu']);b=old['gradient'];delta=abs(a-b);scaled=delta/np.maximum(abs(b),1e-8)
  rows.append(dict(id=c['id'],finite=int(np.isfinite(a).sum()),maximum_absolute_difference=float(np.nanmax(delta)),maximum_scaled_difference=float(np.nanmax(scaled)),value_converged=int(np.asarray(g['value_converged']).sum()),baseline_value_converged=int(old['gradient_value_converged'].sum())))
  np.savez_compressed(OUT/(c['id']+'_gradient.npz'),gradient=a,valid=g['gradient_valid'],value_converged=g['value_converged']);print(c['id'],rows[-1]['maximum_scaled_difference'],flush=True)
 (OUT/'gradient_checks.json').write_text(json.dumps(rows,indent=2)+'\n')
