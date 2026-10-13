import json,subprocess,os,csv,importlib.util,math
from pathlib import Path
p=Path(__file__).resolve().parent;base=p.parent/'initial8-local-repair-20261014';root=p.parents[4]
c=json.loads((base/'compile.json').read_text());c[c.index('-o')+1]=str(p/'runner');subprocess.run(c,check=True)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LCBININT_BENCH_INITIAL_BINS='16')
comparisons=[]
for cohort in ['development','holdout','prospective']:
 out=p/f'{cohort}-control16.tsv'
 subprocess.run(['taskset','-c','0',str(p/'runner'),str(base/f'{cohort}-input.tsv'),str(out),'adaptive'],env=env,check=True)
 a=list(csv.DictReader(out.open(),delimiter='\t'));b=list(csv.DictReader((base/f'{cohort}-validated16.tsv').open(),delimiter='\t'))
 assert len(a)==len(b)==120
 for x,y in zip(a,b):
  assert all(x[k]==y[k] for k in ['sample_id','target','magnification','converged','maps','indicator','stop_reason']),(cohort,x,y)
 comparisons.append(dict(cohort=cohort,unchanged=120))
 print('control',cohort,flush=True)
# Inspect modeled rows against complete scans without using this audit in runtime.
env['LCBININT_BENCH_INITIAL_BINS']='8';env['LCBININT_DIAGNOSTIC_AUDIT_INTERPOLATED_ROWS']='1';env['LCBININT_DIAGNOSTIC_AUDIT_BOUNDARY_STRIPS']='1'
subprocess.run(['taskset','-c','0',str(p/'runner'),str(p/'tips-input.tsv'),str(p/'audit-tips.tsv'),'adaptive'],env=env,check=True)
# Public API budget checks and scope restoration, preserving the earlier output.
s=(base/'budget_smoke.py').read_text().replace("p=Path(__file__).resolve().parent","p=Path('"+str(base)+"')").replace("p/'budget-smoke.json'","Path('"+str(p/'budget-smoke.json')+"')")
exec(compile(s,'budget_smoke','exec'),{})
(p/'verification.json').write_text(json.dumps(comparisons,indent=2)+'\n')
