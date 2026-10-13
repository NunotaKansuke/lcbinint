import os,time,json,statistics
from pathlib import Path
import VBMicrolensing
R=Path(__file__).resolve().parents[3];P=R/'tests/diagnostics/results/recal2026/normal-seed-index-20261014'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
rows=[]
for line in (P.parent/'deferred-cache-production-20261013/input.tsv').read_text().splitlines():
 key,s,q,rho,x,y,c=map(float,line.split());key=int(key)
 for target in [1e-3,1e-4]:
  v=VBMicrolensing.VBMicrolensing();v.Tol=1e-12;v.RelTol=target
  if c:v.a1=c;v.SetLDprofile(v.LDlinear)
  f=v.BinaryMagDark if c else v.BinaryMag
  f(s,q,-x,y,rho,1e-12);ts=[]
  for _ in range(5):
   start=time.perf_counter();mu=float(f(s,q,-x,y,rho,1e-12));ts.append((time.perf_counter()-start)*1000)
  rows.append(dict(sample_id=key,target=target,magnification=mu,ms=statistics.median(ts)))
 (P/'vbm.json').write_text(json.dumps({'module':VBMicrolensing.__file__,'rows':rows},indent=2))
print('VBM completed',len(rows))
