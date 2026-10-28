"""Default routed triple API versus tighter VBM roots/contours on six lenses."""
import sys,os,math,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent
sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable'not in str(f)]
sys.path.insert(0,str(R/'build-triple-native'))
import lcbinint as lc
import VBMicrolensing as V
assert Path(lc._native.__file__).resolve().is_relative_to(R/'build-triple-native')
report=dict(status='RUNNING',rows=[],checks=[],protocol='Ordinary default LightCurve.info, seven physical lens configurations, two radii, uniform/linear LD, two tolerances; both native modes. VBM SinglePoly and MultiPoly scalar reference at relative1e-7, absolute0. Reference agreement is diagnostic, not a rigorous enclosure. No derivative-error refinement or finite differences in this matrix.')
configs=[('baseline',1.,-.4+.6j,[.72,.2,.08]),('equal',1.,-.5+.6j,[1,1,1]),('close',.4,-.3+.2j,[.6,.25,.15]),('wide',2.,-1.+.7j,[.7,.2,.1]),('planetary',1.2,.5+.8j,[1.,.01,.002]),('resonant',.9,.8+.2j,[.45,.35,.2]),('secondary_dominant',1.1,-.7+.8j,[.15,.6,.25])]
def save():(P/'matrix.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for name,s,b,m in configs:
 q=m[1]/m[0];q2=m[2]/m[0]
 for rho in [.005,.03]:
  for limb in [0.,.5]:
   xs=np.linspace(-1.5,2.5,9);times=(q/(1+q)*s-xs).tolist()
   p=dict(t0=0.,u0=-.15,tE=1.,alpha=0.,s=s,q=q,q2=q2,sep2=abs(b),ang=math.atan2(b.imag,b.real),rho=rho)
   vp=[math.log(s),math.log(q),p['u0'],0.,math.log(rho),0.,0.,math.log(abs(b)),math.log(q2),p['ang']]
   refs={}
   for method in ['Singlepoly','Multipoly']:
    v=V.VBMicrolensing();v.SetMethod(getattr(v.Method,method));v.Tol=0.;v.RelTol=1e-7;v.a1=limb;v.a2=0.;v.SetLDprofile(v.LDlinear)
    refs[method]=v.TripleLightCurve(vp,times)[0]
   row=dict(name=name,params=p,epochs=times,physical_x=xs.tolist(),limb=limb,references=refs,native=[]);report['rows'].append(row)
   for tol in [1e-3,1e-4]:
    model=lc.LightCurve(lens='triple',param_type='vbm',ld_c=limb,options=lc.Options(finite_source_tol=0.,finite_source_reltol=tol))
    assert model.options.finite_source_backend=='algebraic_boundary'
    for gradient in [False,True]:
     result=model.info(times,p,with_gradient=gradient)
     errors={k:np.abs(result.magnifications/np.asarray(z)-1).tolist()for k,z in refs.items()}
     z=dict(rtol=tol,gradient=gradient,mu=result.magnifications.tolist(),status=result.status.tolist(),routes=result.diagnostics['finite_source_method'].tolist(),gradient_valid=result.gradient_valid.tolist()if gradient else None,relative_errors=errors)
     row['native'].append(z)
     report['checks'].append(dict(name=name,rho=rho,limb=limb,rtol=tol,gradient=gradient,nonzero_status=int(np.count_nonzero(result.status)),invalid_gradient=int(np.count_nonzero(~result.gradient_valid))if gradient else 0,max_relative_error=max(max(v)for v in errors.values()),above_requested=sum(e>tol for e in errors['Singlepoly'])))
   save();print(name,rho,limb,max(z['max_relative_error'] for z in report['checks'][-4:]),flush=True)
report['status']='COMPLETE';save()
