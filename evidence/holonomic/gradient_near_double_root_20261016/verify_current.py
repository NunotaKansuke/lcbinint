"""Reproduce the default close-root gradient check on the historical -0.7 d case."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'))
from benchmark import load
np,lc,_=load()
p=dict(t0=0.,tE=10.,u0=-1e-5,alpha=.5,s=.6,q=1e-4,rho=.05)
times=np.array([-.704,-.700,-.696,-.680,.780])
keys=('u0','alpha','s','rho','q')
analytic=[]
for tol in (1e-3,1e-5,1e-6):
    curve=lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(.5))
    result=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol);order=list(result['param_order'])
    analytic.append({'tol':tol,'gradient':{key:[float(result['grad_mu'][i,order.index(key)]) for i in range(len(times))] for key in keys},'value_converged':[bool(x) for x in result['value_converged']],'support_certified':[bool(x) for x in result['support_certified']],'gradient_valid':[bool(all(x)) for x in result['gradient_valid']]})

# Dense gradient scan matching the previous geometry experiment.
scan=np.unique(np.r_[np.linspace(-.706,-.690,65),np.linspace(.775,.785,21),[-.72,-.68,.1]])
curve=lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=1e-3,finite_source_reltol=1e-3,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(.5))
r=curve.holonomic_gradient(scan,p,mu_atol=1e-3,mu_rtol=1e-3);order=list(r['param_order']);u=np.asarray(r['grad_mu'])[:,order.index('u0')]
scan_rows=[]
for lo,hi in ((-.706,-.690),(.775,.785)):
    m=(scan>=lo)&(scan<=hi);ix=np.flatnonzero(m);imin=ix[np.argmin(u[m])];imax=ix[np.argmax(u[m])]
    scan_rows.append({'range':[lo,hi],'epochs':int(m.sum()),'u0_min':float(u[imin]),'t_at_min':float(scan[imin]),'u0_max':float(u[imax]),'t_at_max':float(scan[imax]),'max_adjacent_abs_change':float(np.max(np.abs(np.diff(u[m])))),'finite_all':bool(np.isfinite(np.asarray(r['grad_mu'])[m]).all()),'value_converged_all':bool(np.all(r['value_converged'][m])),'support_certified_all':bool(np.all(r['support_certified'][m]))})

ref={}
with (ROOT/'evidence/holonomic/gradient_geometry_20261015/comparison.csv').open() as f:
    for row in csv.DictReader(f):
        if row['variant']=='candidate':ref[(float(row['t']),row['param'])]={'fd_median':float(row['vbm_fd_median']),'fd_min':float(row['vbm_fd_min']),'fd_max':float(row['vbm_fd_max'])}
alpha=json.loads((OUT/'alpha_vbm_fd.json').read_text())
for row in alpha['vbm_central_differences']:
    for i,t in enumerate(alpha['times']):
        ref[(float(t),'alpha')]=ref.get((float(t),'alpha'),{})
        ref[(float(t),'alpha')].setdefault('fd_samples',[]).append(float(row['fd'][i]))
for (t,key),record in list(ref.items()):
    if 'fd_samples' in record:
        record['fd_median']=float(np.median(record['fd_samples']));record['fd_min']=float(np.min(record['fd_samples']));record['fd_max']=float(np.max(record['fd_samples']))
comparisons=[]
for tolrow in analytic:
    for i,t in enumerate(times):
        for key in keys:
            reference=ref.get((float(t),key))
            if not reference: continue
            value=tolrow['gradient'][key][i];fd=reference['fd_median']
            comparisons.append({'tol':tolrow['tol'],'time':float(t),'parameter':key,'gradient':value,**reference,'relative_difference_percent':100*(value/fd-1)})
module=Path(__import__('_lcbinint_holonomic_fused').__file__).resolve();native=Path(lc._native.__file__).resolve()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
result={'claim':'The previously reported -0.7 d gradient spike is absent in the current default close-pair build for the tested near-double-root window.','git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'params':p,'limb_darkening_linear':.5,'times':times.tolist(),'analytic_tolerance_sweep':analytic,'vbm_finite_difference_comparisons':comparisons,'vbm_reference_sources':{'u0_rho_q_s':str(ROOT/'evidence/holonomic/gradient_geometry_20261015/comparison.csv'),'alpha':str(OUT/'alpha_vbm_fd.json')},'scan_89_epochs':scan_rows,'scan_epochs':len(scan),'scan_finite_gradient_components':int(np.isfinite(r['grad_mu']).sum()),'scan_gradient_components':int(r['grad_mu'].size),'scan_invalid_epochs':[float(scan[i]) for i,x in enumerate(r['gradient_valid']) if not all(x)],'module':str(module),'module_sha256':digest(module),'native':str(native),'native_sha256':digest(native),'build_api_overlay_sha256':digest(ROOT/'benchmarks/holonomic/fused_gradient/build_api_overlay.py'),'build_geometry_overlay_sha256':digest(ROOT/'benchmarks/holonomic/fused_gradient/build_geometry_overlay.py')}
(OUT/'current_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
