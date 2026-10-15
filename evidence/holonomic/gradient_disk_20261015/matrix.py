from pathlib import Path
import os,sys,json,time
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,_=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});rows=[]
for c in json.loads((ROOT/'evidence/holonomic/lightcurve_matrix_20261015/cases.json').read_text()):
 p=c['params'];tol=c['tol'];times=p['tE']*np.linspace(c['tau_min'],c['tau_max'],401);curve=l.LightCurve(options=l.Options(coordinates='vbm',finite_source_backend='algebraic_boundary',finite_source_tol=tol,finite_source_reltol=tol),limb_darkening=l.LimbDarkening.linear(c['limb']));info=curve.info(times,p);g=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol)
 old=np.load(ROOT/'evidence/holonomic/large_source_speed_20261015'/(c['id']+'_final.npz'));a=np.asarray(info['mu']);b=old['mu'];fields=['value_converged','fully_converged','seed_support_proven','support_certified','center_image_count','augmented_seed_count','node_evaluations','integrator_used'];r=dict(id=c['id'],finite=int(np.isfinite(g['grad_mu']).sum()),gradient_converged=int(np.asarray(g['value_converged']).sum()),mu_relative_change=float(np.max(abs(a-b)/np.maximum(abs(b),1e-12))),changed_value_fields=[k for k in fields if not np.array_equal(np.asarray(info[k]),old[k])],gradient_mu_value_relative_difference=float(np.max(abs(np.asarray(g['mu'])-a)/np.maximum(abs(a),1e-12))));rows.append(r)
 np.savez_compressed(OUT/(c['id']+'_matrix.npz'),times=times,gradient=g['grad_mu'],gradient_mu=g['mu'],mu=a,valid=g['gradient_valid'],gradient_converged=g['value_converged']);print(r,flush=True)
 (OUT/'matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
