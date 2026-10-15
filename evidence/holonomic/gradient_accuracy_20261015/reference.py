from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import sys, json, csv
import numpy as np
import VBMicrolensing
root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
source = root/'python/lcbinint'; build = root/'build-algebraic-boundary/python-api/lcbinint'
for f in tuple(sys.meta_path):
    if type(f).__module__ == '_lcbinint_editable': sys.meta_path.remove(f)
for n in tuple(sys.modules):
    if n == 'lcbinint' or n.startswith('lcbinint.'): del sys.modules[n]
spec=spec_from_file_location('lcbinint',source/'__init__.py',submodule_search_locations=[str(source),str(build)])
l=module_from_spec(spec);sys.modules['lcbinint']=l;spec.loader.exec_module(l)
p=dict(t0=0.,tE=10.,u0=-1e-5,alpha=.5,s=.6,q=1e-4,rho=.05)
times=np.array([-.72,-.704,-.7,-.696,-.68,.1,.78])
c=l.LightCurve(options=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=1e-10,finite_source_reltol=1e-9,finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(.5))
geometry=c.finite_source_geometry(times,p)
with (out/'inputs.tsv').open('w') as f:
    for i,t in enumerate(times):
        f.write('\t'.join(map(str,[t,geometry.source_x[i],geometry.source_y[i],p['rho'],geometry.mass_ratio[i],p['s'],.5]))+'\n')
def vbm_value(params):
    v=VBMicrolensing.VBMicrolensing();v.Tol=1e-10;v.RelTol=1e-10;v.a1=.5;v.a2=0.
    args=[np.log(params['s']),np.log(params['q']),params['u0'],params['alpha'],np.log(params['rho']),np.log(params['tE']),params['t0']]
    return np.asarray(v.BinaryLightCurve(args,times.tolist())[0])
rows=[]
for name,h0 in [('t0',1e-3),('u0',1e-5),('rho',1e-5),('q',1e-7),('s',1e-4)]:
    for factor in [1.,.3,.1]:
        h=h0*factor;plus=p.copy();minus=p.copy();plus[name]+=h;minus[name]-=h
        vp,vm=vbm_value(plus),vbm_value(minus)
        ip,im=c.info(times,plus),c.info(times,minus)
        npv,nmv=np.asarray(ip['mu']),np.asarray(im['mu'])
        for i,t in enumerate(times):
            rows.append(dict(t=t,param=name,h=h,vbm_fd=(vp[i]-vm[i])/(2*h),native_fd=(npv[i]-nmv[i])/(2*h),vp=vp[i],vm=vm[i],np=npv[i],nm=nmv[i],native_plus_converged=bool(ip['value_converged'][i]),native_minus_converged=bool(im['value_converged'][i])))
    print('FD done:',name,flush=True)
with (out/'reference.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
(out/'config.json').write_text(json.dumps(dict(params=p,times=times.tolist(),vbm_atol=1e-10,vbm_rtol=1e-10,native_atol=1e-10,native_rtol=1e-9,native_file=l._native.__file__),indent=2)+'\n')
