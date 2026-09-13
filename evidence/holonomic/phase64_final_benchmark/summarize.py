import json, hashlib, subprocess
from pathlib import Path
import numpy as np
import make_figures as p
rows=p.join_rows()
keys=[(r['case_id'],r['profile'],r['d_bin_index'],r['epoch_index'],r['target']) for r in rows]
assert len(set(keys))==14432
summary=[]
for profile in p.PROFILES:
 for target in p.TARGETS:
  group=[r for r in rows if r['profile']==profile and r['target']==target]
  total,ok=p.all_counts(group,'q','rho','warm')
  assert (total>=8).sum()==144
  for lane in ('cold','warm','radial'):
   times=np.array([r[lane+'_ms'] for r in group])
   assert np.isfinite(times).all() and (times>0).all()
   errors=np.array([abs(r[lane+'_mu']/r['reference']-1) for r in group])
   ratios=np.array([r['warm_vbm_ms']/r[lane+'_ms'] for r in group])
   summary.append(dict(profile=profile,target=target,lane=lane,rows=len(group),
    converged=sum(r[lane+'_value_converged']==1 for r in group),
    time_ms=dict(zip(('p50','p90','p95','p99','max'),map(float,np.percentile(times,[50,90,95,99,100])))),
    vbm_ms_median=float(np.median([r['warm_vbm_ms'] for r in group])),
    paired_speedup_median=float(np.median(ratios)), faster_fraction=float(np.mean(ratios>1)),
    error=dict(zip(('p50','p95','p99','max'),map(float,np.percentile(errors,[50,95,99,100])))),
    reference_excess=int(np.sum(errors>target)),qrho_populated=144))
(p.ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import csv
with (p.ROOT/'joined.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
paths=['evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv',
'evidence/holonomic/v2_vbm_pure_kernel_20260911/input_snapshot.tsv',
'evidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp',
'/tmp/p64_final_whole',str(p.RESULTS),str(p.VBM_RESULTS)]
provenance={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in paths}
provenance['head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
provenance['compiler']=subprocess.check_output(['c++','--version'],text=True).splitlines()[0]
(p.ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(summary,indent=2))
