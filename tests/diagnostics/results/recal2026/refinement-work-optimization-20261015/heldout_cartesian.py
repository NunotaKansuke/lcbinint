import os,sys,json,importlib.util,time,argparse
from pathlib import Path
ROOT=Path('/rogue1_8/nunota/lcbinint');OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--baseline',action='store_true');p.add_argument('--cohort',type=Path,default=OUT/'cohort.json');p.add_argument('--ids',nargs='*');p.add_argument('--trace',action='store_true');p.add_argument('--disable-outer',action='store_true');p.add_argument('--bins',type=int,default=16);p.add_argument('--row-ratio',type=int,choices=[1,2],default=1);a=p.parse_args()
for k in list(os.environ):
 if k.startswith('LCBININT_'):os.environ.pop(k)
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LCBININT_DIAGNOSTIC_ADAPTIVE_COST_TRACE='1')
os.environ['LCBININT_DIAGNOSTIC_CARTESIAN_INITIAL_ROW_RATIO']=str(a.row_ratio)
if a.trace:os.environ['LCBININT_DIAGNOSTIC_ROW_PANEL_TRACE']='1'
if a.disable_outer:os.environ['LCBININT_DISABLE_BENNETT_OUTER']='1'
os.sched_setaffinity(0,{0});native=next((OUT/'baseline'if a.baseline else ROOT/'build-adaptive-production').glob('_lcbinint*.so'))
spec=importlib.util.spec_from_file_location('_lcbinint',native);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
inputs=json.loads(a.cohort.read_text());records=[]
for i,r in enumerate(inputs):
 if a.ids and r['geom_id']not in a.ids:continue
 for tol in [1e-3,1e-4]:
  row=dict(id=r['geom_id'],target=tol,profile=r['profile'],reference=float(r['reference']));start=time.perf_counter()
  try:
   g=n.binary_adaptive_flux(float(r['time']),float(r['y']),s=float(r['s']),q=1/float(r['q']),rho=float(r['rho']),limb_darkening=n.LimbDarkening(float(r['c']),0),initial_bins=a.bins,max_bins=512,relative_tolerance=tol,max_lattice_maps=1000000)
   row.update(result=g,error=abs(g['magnification']/row['reference']-1))
  except Exception as e:row['exception']=str(e)
  row['seconds']=time.perf_counter()-start;records.append(row)
 print(i+1,len(inputs),flush=True)
import hashlib
(OUT/(a.name+'-identity.json')).write_text(json.dumps(dict(native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(a.cohort.read_bytes()).hexdigest(),initial_bins=a.bins,row_ratio=a.row_ratio,disable_outer=a.disable_outer,trace=a.trace),indent=2)+'\n')
(OUT/(a.name+'.json')).write_text(json.dumps(records)+'\n')
