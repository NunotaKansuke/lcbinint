from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,_=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});rows=[]
for i in [0,2,8,16,24,30,33,45,51,64,70,89]:
 c=json.loads((OUT/f'case_{i:03}.json').read_text());p=c['params'];old=np.load(OUT/(c['id']+'.npz'));times=old['times'][[8,16,24]]
 def curve(tol):return l.LightCurve(options=l.Options(coordinates='vbm',finite_source_backend='algebraic_boundary',finite_source_tol=tol,finite_source_reltol=tol),limb_darkening=l.LimbDarkening.linear(c['limb']))
 g=curve(1e-3).holonomic_gradient(times,p,mu_atol=1e-3,mu_rtol=1e-3);strict=curve(1e-7)
 for key in ['u0','rho','q','s']:
  col=list(g['param_order']).index(key);scale=max(p['rho'],1e-4) if key=='u0' else p[key];fds=[];conv=[]
  for frac in [1e-2,3e-3,1e-3]:
   h=scale*frac;plus=dict(p);minus=dict(p);plus[key]+=h;minus[key]-=h
   a=strict.info(times,plus);b=strict.info(times,minus);fds.append((np.asarray(a['mu'])-b['mu'])/(2*h));conv.append(bool(np.all(a['value_converged']) and np.all(b['value_converged'])))
  fd=np.asarray(fds);gg=np.asarray(g['grad_mu'])[:,col];den=np.maximum(np.abs(fd[-1]),1e-5)
  r=dict(case=c['id'],parameter=key,gradient=gg.tolist(),fd=fd.tolist(),fd_converged=conv,step_change=(abs(fd[-1]-fd[-2])/den).tolist(),gradient_relative_difference=(abs(gg-fd[-1])/den).tolist());rows.append(r);print(c['id'],key,'fdchange',r['step_change'],'gdiff',r['gradient_relative_difference'],conv,flush=True)
 (OUT/'fd.json').write_text(json.dumps(rows,indent=2)+'\n')
