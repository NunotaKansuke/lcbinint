"""New changed-geometry cases and intermediate LD strengths; independent VBM radial reference."""
import argparse,os,json,math,time,hashlib,warnings
from pathlib import Path
import VBMicrolensing
from scipy.integrate import quad,IntegrationWarning
from .probe_build import activate
ROOT=Path('tests/diagnostics/results/recal2026')
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='adaptive-defect-fresh-v2');a=p.parse_args();out=ROOT/a.output;out.mkdir(exist_ok=False)
 for k in list(os.environ):
  if k.startswith('LCBININT_DIAGNOSTIC_'):os.environ.pop(k)
 for k in ('DIRECTIONAL_ADAPTIVE','LOCAL_ROW_INTERPOLATION','LOCAL_BOUNDARY_STRIPS','LOCAL_VERTICAL_CAPS'):os.environ['LCBININT_DIAGNOSTIC_'+k]='1'
 n=activate()._lcbinint;src=ROOT/'adaptive-image-area-holdout-v4/summary.json';saved=json.loads(src.read_text())['rows'];rows=[];refs=[];notes=[]
 def checkpoint():
  result=dict(rows=rows,references=refs,warnings=notes,count=len(rows),actual_pass=sum(r['actual_error']<=r['tolerance'] for r in rows),baseline_pass=sum(r['baseline_error']<=r['tolerance'] for r in rows),false_accept=sum(r['estimated_target_met'] and r['actual_error']>r['tolerance'] for r in rows),source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),native_sha256=hashlib.sha256(Path(n.__file__).read_bytes()).hexdigest(),scope='New deterministic perturbations of six held-out geometries, both signs, c=0/.2/.5/.8/1, absolute1e-3/1e-4. Independent VBM uniform-area radial quadrature, two reference accuracy settings. No runtime reference access.',certified=False,complete_error_model=False)
  (out/'summary.json').write_text(json.dumps(result,indent=2))
 for idx in (3,6,10,14,17,19):
  for sign in (-1,1):
   base=saved[idx];r={k:base[k] for k in ('s','q','x','y','rho')}
   r.update(s=base['s']*(1+sign*.0033),q=base['q']*(1-sign*.0091),x=base['x']+sign*.11*base['rho'],y=base['y']-sign*.13*base['rho'],rho=base['rho']*(1-sign*.021))
   v=VBMicrolensing.VBMicrolensing();v.a1=0;v.Tol=1e-11;v.RelTol=1e-11;cache={};A=math.pi*r['rho']**2;start=time.perf_counter()
   def area(radius):
    if radius not in cache:cache[radius]=math.pi*radius**2*v.BinaryMagDark(r['s'],r['q'],-r['x'],r['y'],radius,1e-11)
    return cache[radius]
   outer=area(r['rho'])
   def integrand(u):
    t=1-u*u;return 4*u*t*area(r['rho']*u*math.sqrt((2-u*u)*(1+t*t)))
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',IntegrationWarning)
    first,err1=quad(integrand,0,1,epsabs=A*1e-9,epsrel=1e-10,limit=150)
    second,err2=quad(integrand,0,1,epsabs=A*1e-10,epsrel=1e-11,limit=150)
   notes.extend(dict(index=idx,sign=sign,message=str(w.message)) for w in caught)
   reference=dict(index=idx,sign=sign,input=r,uniform_area=outer,ld_area=second,first_ld_area=first,quadrature_error=err2,reference_radii=len(cache),seconds=time.perf_counter()-start);refs.append(reference)
   for c in (0.,.2,.5,.8,1.):
    norm=A*(1-c/3);ref=((1-c)*outer+c*second)/norm;stability=c*abs(second-first)/norm;assert stability<1e-8,(idx,sign,c,stability)
    for tol in (1e-3,1e-4):
     kw=dict(s=r['s'],q=1/r['q'],rho=r['rho'],limb_darkening=n.LimbDarkening(c,0),initial_bins=16,max_bins=512,absolute_tolerance=tol,max_lattice_maps=1000000)
     os.environ.pop('LCBININT_DIAGNOSTIC_DEFECT_CORRECTION',None);old=n.binary_adaptive_flux(r['x'],r['y'],**kw)
     os.environ['LCBININT_DIAGNOSTIC_DEFECT_CORRECTION']='1';g=n.binary_adaptive_flux(r['x'],r['y'],**kw)
     rows.append(dict(index=idx,sign=sign,input=r,c=c,tolerance=tol,reference=ref,reference_stability=stability,reference_quad_indicator=c*err2/norm,actual_error=abs(g['magnification']-ref),baseline_error=abs(old['magnification']-ref),baseline_estimated_met=old['estimated_target_met'],baseline_steps=old['component_refinements'],baseline_maps=old['new_lattice_maps'],**g))
   checkpoint();print(json.dumps(dict(index=idx,sign=sign,count=len(rows),failures=sum(r['actual_error']>r['tolerance'] for r in rows),reference_seconds=reference['seconds'])),flush=True)
 checkpoint()
if __name__=='__main__':main()
