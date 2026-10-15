"""Independent LD annulus identity using only VBM uniform-disk calls."""
from pathlib import Path
import sys,os,json,subprocess,time
from benchmark import ROOT,OUT,load

def evaluate(c):
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});np,l,VB=load();p=c['params'];t=[c['worst_time']];limb=c['limb'];result={**c,'native':[],'vbm_direct':[],'annulus':[]};base=[np.log(p['s']),np.log(p['q']),p['u0'],p['alpha'],np.log(p['rho']),np.log(p['tE']),p['t0']]
 for tol in [1e-5,1e-7]:
  curve=l.LightCurve(options=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(limb));r=curve.info(t,p)
  result['native'].append(dict(tol=tol,mu=float(r['mu'][0]),error=float(r['error_estimate'][0]),value_converged=bool(r['value_converged'][0]),fully_converged=bool(r['fully_converged'][0]),nodes=int(r['node_evaluations'][0])))
 for tol in [1e-3,1e-4,1e-6,1e-8]:
  v=VB.VBMicrolensing();v.Tol=v.RelTol=tol;v.a1=limb;v.a2=0.;mu=float(v.BinaryLightCurve(base,t)[0][0]);result['vbm_direct'].append(dict(tol=tol,mu=mu))
 v=VB.VBMicrolensing();v.Tol=v.RelTol=1e-9;v.a1=v.a2=0.;outer=float(v.BinaryLightCurve(base,t)[0][0]);result['vbm_uniform_outer']=outer
 curve=l.LightCurve(options=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=1e-6,finite_source_reltol=1e-6,finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(0.));r=curve.info(t,p);result['native_uniform_outer']={'mu':float(r['mu'][0]),'converged':bool(r['value_converged'][0])}
 # Integration by parts in the disk radius, z=sqrt(1-(r/rho)^2):
 # A_LD=((1-c) A_U(rho)+c*int_0^1 (1-z^2) A_U(rho*sqrt(1-z^2)) dz)/(1-c/3).
 for n in [32,64,128,256]:
  x,w=np.polynomial.legendre.leggauss(n);z=(x+1)/2;weights=w/2;values=[];started=time.perf_counter()
  for zi in z:
   args=base.copy();args[4]=np.log(p['rho']*np.sqrt(1-zi*zi));values.append(float(v.BinaryLightCurve(args,t)[0][0]))
  integral=float(np.dot(weights*(1-z*z),values));mu=((1-limb)*outer+limb*integral)/(1-limb/3)
  result['annulus'].append(dict(n=n,mu=mu,seconds=time.perf_counter()-started))
 (OUT/(c['id']+'_reference.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(c['id'],result['native'][-1],result['vbm_direct'][-1],result['annulus'][-1],flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:evaluate(json.loads(sys.argv[1]))
 else:
  for c in json.loads((OUT/'reference_cases.json').read_text()):
   try:
    r=subprocess.run([sys.executable,str(Path(__file__)),json.dumps(c)],cwd=ROOT,env=os.environ,capture_output=True,text=True,timeout=40)
    if r.returncode:raise RuntimeError(r.stderr[-2000:])
    print(r.stdout.strip(),flush=True)
   except Exception as e:
    (OUT/(c['id']+'_reference.json')).write_text(json.dumps({**c,'status':'failed','error':str(e)},indent=2)+'\n');print('REFERENCE FAILED',c['id'],str(e)[-200:],flush=True)
