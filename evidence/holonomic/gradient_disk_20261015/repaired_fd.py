from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,_=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});rows=[]
for i,index in [(4,16),(49,16),(58,16),(81,12),(109,24)]:
 c=json.loads((ROOT/f'evidence/holonomic/stress_20261015/case_{i:03}.json').read_text());p=c['params'];t=np.linspace(-c['span'],c['span'],33)[index]*p['tE']
 def curve(tol):return l.LightCurve(options=l.Options(coordinates='vbm',finite_source_backend='algebraic_boundary',finite_source_tol=tol,finite_source_reltol=tol),limb_darkening=l.LimbDarkening.linear(c['limb']))
 g=curve(1e-3).holonomic_gradient([t],p,mu_atol=1e-3,mu_rtol=1e-3);a=np.asarray(g['grad_mu'])[0];strict=curve(1e-7)
 for key in ['u0','rho','s']:
  col=list(g['param_order']).index(key);scale=p['rho'] if key=='u0' else p[key];fd=[];conv=[]
  for fraction in [.01,.003,.001]:
   h=scale*fraction;plus=dict(p);minus=dict(p);plus[key]+=h;minus[key]-=h;x=strict.info([t],plus);y=strict.info([t],minus);fd.append(float((x['mu'][0]-y['mu'][0])/(2*h)));conv.append(bool(x['value_converged'][0] and y['value_converged'][0]))
  r=dict(case=i,parameter=key,gradient=float(a[col]),fd=fd,converged=conv);rows.append(r);print(r,flush=True)
 (OUT/'repaired_fd.json').write_text(json.dumps(rows,indent=2)+'\n')
