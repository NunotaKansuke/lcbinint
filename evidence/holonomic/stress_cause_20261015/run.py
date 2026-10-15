from pathlib import Path
import sys,os,json,subprocess
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,_=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for i,t,limbs,tols in [(33,-.15,[0.,.5,.9],[1e-3,1e-5,1e-7]),(4,0.,[0.],[1e-3]),(49,0.,[.5],[1e-3])]:
 c=json.loads((ROOT/f'evidence/holonomic/stress_20261015/case_{i:03}.json').read_text());p=c['params'];curve=l.LightCurve(options=l.Options(coordinates='vbm'),limb_darkening=l.LimbDarkening.linear(c['limb']));geo=curve.finite_source_geometry([t],p)
 for u in limbs:
  for tol in tols:
   tag=f'case{i}_u{u}_tol{tol}';args=[float(geo.separation[0]),float(geo.mass_ratio[0]),float(geo.source_radius[0]),float(geo.source_x[0]),float(geo.source_y[0]),u,tol]
   result=subprocess.run([str(ROOT/'build-algebraic-boundary/stress-cause/probe'),*[str(v) for v in args]],capture_output=True,text=True,env={**os.environ,'LCBININT_CAUSE_DEBUG':'1'},check=True,timeout=15)
   (OUT/(tag+'.log')).write_text(result.stderr);(OUT/(tag+'.json')).write_text(json.dumps(dict(args=args,result=result.stdout),indent=2)+'\n');print(tag,result.stdout.strip(),flush=True)
