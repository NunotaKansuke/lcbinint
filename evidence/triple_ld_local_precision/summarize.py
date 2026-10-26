import json,statistics as S,math,hashlib,subprocess,platform
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent.parent
runs=[json.loads((P/f'final-matched-{i}.json').read_text())['rows'] for i in range(3)]
assert all(len(r)==48 for r in runs)
extra=json.loads((P/'validation-final.json').read_text())['rows']
assert len(extra)==56 and all(r['passed'] for r in extra)
groups=[]
for u in [0.,.5]:
 for tol in [.001,.0001]:
  subsets=[[x for x in run if x['limb']==u and x['rtol']==tol]for run in runs]
  reductions=[100*(1-math.exp(S.mean(math.log(x['median']['after']/x['median']['before'])for x in sub)))for sub in subsets]
  times={k:S.median(S.median(subsets[j][i]['median'][k]for j in range(3))for i in range(12))for k in ['before','after','Singlepoly','Multipoly','Nopoly']}
  robust_wins=sum(all(subsets[j][i]['median']['after']<subsets[j][i]['median']['Singlepoly']for j in range(3))for i in range(12))
  g=dict(limb=u,rtol=tol,reduction_percent=reductions,median_ms=times,robust_singlepoly_wins=robust_wins)
  groups.append(g);print(g)
checks=dict(main_pass=all(x['pass']for run in runs for x in run),extra_pass=all(x['passed']for x in extra),extra_conditions=len(extra),extra_estimate_contains_native=all(x['estimate_contains_native']for x in extra),groups=groups)
# Check all saved independent reference methods, including tighter supplemental native results.
d=json.loads((R/'evidence/triple_natural_integral_benchmark/results.json').read_text())
fail=[]
for row,orig in zip(runs[0],d['rows']):
 mu=row['samples']['after'][-1]['mu']
 for method,vals in d['references'][orig['reference_id']]['results'].items():
  for v in vals:
   if not math.isfinite(v['mu']):continue
   if abs(mu-v['mu'])/abs(v['mu'])>orig['rtol']:fail.append(dict(condition=orig,method=method,reference=v))
checks['saved_reference_failures']=fail
assert checks['main_pass']and not fail
(P/'checks.json').write_text(json.dumps(checks,indent=2))
paths=[P/'final.so',P/'matched.py',P/'validate_extra.py',P/'counts.cpp',P/'counts.so',*list((R/'src/lcbinint/magnification/algebraic_boundary/triple').glob('*.hpp')),R/'tests/triple_algebraic/test_panel.cpp',R/'evidence/triple_fixed_ld_breakdown/final.so',R/'evidence/triple_natural_integral_benchmark/kernel.so']
hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
(P/'provenance.json').write_text(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),host=platform.node(),hashes=hashes,protocol='CPU0 thread CPU time; immutable VBM wrapper; three complete repetitions; 4 warmup plus 8 measured; separate state per method/condition; rotated/reversed order'),indent=2))
(P/'SHA256SUMS').write_text(''.join(f'{h}  {p}\n'for p,h in hashes.items()))
patch=''
for name,target in [('radial_panel.hpp','src/lcbinint/magnification/algebraic_boundary/triple/radial_panel.hpp'),('test_panel.cpp','tests/triple_algebraic/test_panel.cpp')]:
 p=subprocess.run(['diff','-u','--label','a/'+target,'--label','b/'+target,str(P/('before-'+name)),str(R/target)],text=True,capture_output=True);assert p.returncode in [0,1];patch+=p.stdout
(P/'incremental.patch').write_text(patch)
