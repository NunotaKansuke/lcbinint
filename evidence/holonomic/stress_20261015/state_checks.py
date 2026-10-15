from pathlib import Path
import sys,json,os,time,subprocess
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,VB=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
opts=l.Options(coordinates='vbm',finite_source_backend='algebraic_boundary',finite_source_tol=1e-3,finite_source_reltol=1e-3)
curve=l.LightCurve(options=opts,limb_darkening=l.LimbDarkening.linear(.5));rows=[]
for i in range(0,120,4):
 c=json.loads((OUT/f'case_{i:03}.json').read_text());p=c['params'];times=np.linspace(-.2,.2,7)
 a=np.asarray(curve(times,p));b=np.asarray(l.LightCurve(options=opts,limb_darkening=l.LimbDarkening.linear(.5))(times,p));difference=float(np.max(abs(a-b)/np.maximum(abs(b),1e-12)));rows.append(dict(case=c['id'],relative_difference=difference));print(c['id'],difference,flush=True)
(OUT/'state_checks.json').write_text(json.dumps(rows,indent=2)+'\n')
# Isolate the previous whole-worker timeout by recording each stage before VBM.
c=json.loads((OUT/'case_054.json').read_text());p=c['params'];times=np.linspace(-.3,.3,33);curve=l.LightCurve(options=opts,limb_darkening=l.LimbDarkening.linear(c['limb']))
for name,fn in [('native_value',lambda:curve.info(times,p)),('native_gradient',lambda:curve.holonomic_gradient(times,p,mu_atol=1e-3,mu_rtol=1e-3))]:
 t=time.perf_counter();r=fn();print('case054',name,'seconds',time.perf_counter()-t,flush=True)
