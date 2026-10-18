from pathlib import Path
import argparse,os,sys,json,time,hashlib
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--build',default='build-route-unified');p.add_argument('--label',required=True);a=p.parse_args()
sys.meta_path[:]=[f for f in sys.meta_path if 'editable' not in getattr(f,'__module__','')];sys.path.insert(0,str(ROOT/a.build))
import numpy as np,lcbinint as lc
assert Path(lc.__file__).is_relative_to(ROOT/a.build)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
nb=dict(t0=0.,tE=10.,u0=-.01,alpha=180.,s=1.5,q=.01,rho=.001)
cases=[('notebook',nb,np.linspace(-.8,.8,401),.5,1e-4),('ordinary',dict(t0=0.,tE=10.,u0=.2,alpha=.5,s=1.2,q=.01,rho=.001),np.linspace(-10,10,401),.5,1e-4),('crossing',dict(t0=0.,tE=1.,u0=.03,alpha=-.08,s=1.2,q=.01,rho=.003),np.linspace(-.2,.8,101),.5,1e-4),('point',dict(t0=0.,tE=1.,u0=.3,alpha=0.,s=1.2,q=.01,rho=.003),np.linspace(-1,1,201),.5,1e-4)]
summary=[]
for name,p,t,limb,tol in cases:
 c=lc.LightCurve(options=lc.Options(param_type='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol),limb_darkening=lc.LimbDarkening.linear(limb))
 c([1.],dict(p,s=1.3),with_gradient=True)
 epochs={False:[],True:[]};elapsed={False:[],True:[]};result={}
 for k in range(9):
  for g in ((False,True) if k%2==0 else (True,False)):
   start=time.perf_counter();v=c(t,p,with_gradient=g,with_diagnostics=True);elapsed[g].append((time.perf_counter()-start)*1000)
   epochs[g].append(np.asarray(v.diagnostics['seconds'])*1000);result[g]=v
 out=O/a.label;out.mkdir(exist_ok=True)
 np.savez(out/(name+'.npz'),times=t,value_seconds=np.asarray(epochs[False]),gradient_seconds=np.asarray(epochs[True]),methods=result[True].diagnostics['finite_source_method'],values=result[False].magnifications,fused=result[True].magnifications,gradients=result[True].gradients,valid=result[True].gradient_valid,nodes=result[True].diagnostics['node_evaluations'])
 geo=c._native.finite_source_geometry(t,**p)
 data=np.column_stack([geo.separation,geo.mass_ratio,geo.source_x,geo.source_y,geo.source_radius,result[False].magnifications])
 with (O/(name+'-geometry.txt')).open('w') as f:f.write(str(len(data))+'\n');np.savetxt(f,data,fmt='%.17g')
 row=dict(case=name,n=len(t),params=p,limb=limb,tolerance=tol,value_ms=float(np.median(elapsed[False])),gradient_ms=float(np.median(elapsed[True])),method_counts={str(int(m)):int(np.sum(np.asarray(result[True].diagnostics['finite_source_method'])==m)) for m in np.unique(result[True].diagnostics['finite_source_method'])})
 summary.append(row);print(row,flush=True)
(out/'summary.json').write_text(json.dumps(dict(native_sha256=hashlib.sha256(Path(lc._native.__file__).read_bytes()).hexdigest(),cases=summary),indent=2)+'\n')
