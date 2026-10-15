"""Paired numerical parity and retrospective refinement-work diagnosis."""
import argparse,csv,hashlib,importlib.util,json,math,os,random,time,statistics
from pathlib import Path
ROOT=Path('/rogue1_8/nunota/lcbinint');OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--baseline',action='store_true');p.add_argument('--trace',action='store_true');p.add_argument('--count',type=int,default=128);p.add_argument('--policy',type=int,default=0);p.add_argument('--cost',action='store_true');p.add_argument('--blocks',action='store_true');p.add_argument('--audit-blocks',action='store_true');p.add_argument('--batch',type=int,choices=[0,4,8],default=0);a=p.parse_args()
for k in list(os.environ):
 if k.startswith('LCBININT_'):os.environ.pop(k)
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.environ['LCBININT_DIAGNOSTIC_WORK_SCHEDULER']=str(a.policy)
if a.batch:os.environ['LCBININT_DIAGNOSTIC_CARTESIAN_BATCH']=str(a.batch)
if a.audit_blocks:os.environ['LCBININT_DIAGNOSTIC_AUDIT_INTERIOR_BLOCKS']='1'
if a.blocks:os.environ['LCBININT_DIAGNOSTIC_INTERIOR_DISCRETE_BLOCKS']='1'
if a.cost:os.environ['LCBININT_DIAGNOSTIC_ADAPTIVE_COST_TRACE']='1'
if a.trace:os.environ['LCBININT_DIAGNOSTIC_REFINEMENT_WORK_TRACE']='1'
os.sched_setaffinity(0,{0})
path=next((OUT/'baseline'if a.baseline else ROOT/'build-adaptive-production').glob('_lcbinint*.so'));s=importlib.util.spec_from_file_location('_lcbinint',path);n=importlib.util.module_from_spec(s);s.loader.exec_module(n)
source=ROOT/'figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv';inputs=list(csv.DictReader(source.open(),delimiter='\t'));rng=random.Random(202610150824);selected=[]
for profile in ['uniform','linear']:selected+=rng.sample([r for r in inputs if r['profile']==profile],a.count//2)
selected.sort(key=lambda r:int(r['geom_id']));refs={r['geom_id']:float(r['reference'])for r in inputs}
for r in csv.DictReader((ROOT/'figures/recal2026/polar-relative-matched-six-panel-20261015/reference-adjudications.tsv').open(),delimiter='\t'):refs[r['geom_id']]=float(r['adjudicated_reference'])
records=[]
for index,r in enumerate(selected):
 for method in ['cartesian','polar']:
  for tol in [1e-3,1e-4]:
   row=dict(id=r['geom_id'],input=r,profile=r['profile'],method=method,target=tol,reference=refs[r['geom_id']]);t=time.perf_counter()
   try:
    fn=n.binary_adaptive_flux if method=='cartesian'else n.binary_adaptive_polar_flux
    got=fn(float(r['time']),float(r['y']),s=float(r['s']),q=1/float(r['q']),rho=float(r['rho']),limb_darkening=n.LimbDarkening(float(r['c']),0),relative_tolerance=tol,initial_bins=16 if method=='cartesian'else 2)
    row.update(result=got,error=abs(got['magnification']/row['reference']-1))
   except Exception as e:row['exception']=str(e)
   row['seconds']=time.perf_counter()-t;records.append(row)
 if (index+1)%16==0:print(index+1,a.count,flush=True)
(OUT/(a.name+'.json')).write_text(json.dumps(dict(native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),trace=a.trace,records=records))+'\n')
