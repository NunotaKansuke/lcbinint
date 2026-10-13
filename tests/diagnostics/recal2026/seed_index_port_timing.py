"""Alternate native seed provider measurements; every source has a fresh seed cache."""
import os,subprocess,csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'tests/diagnostics/results/recal2026/seed-index-port-20261013'
B=P.parent/'correction-accuracy-cost-20261013'
def main():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 inputs=json.loads((B/'inputs.json').read_text())
 # Bounded timing cohort; entire7216 is independently checked for seed identity.
 for rep in range(5):
  for mode in (['before','after'] if rep%2==0 else ['after','before']):
   subprocess.run([str(P/f'seed-{mode}'),str(B/'input.tsv'),str(P/f'timing-{mode}-{rep}.tsv')],check=True)
  print('repeat',rep,flush=True)
 by={}
 for mode in ['before','after']:
  for rep in range(5):
   for r in csv.DictReader((P/f'timing-{mode}-{rep}.tsv').open(),delimiter='\t'):by.setdefault((int(r['id']),mode),[]).append(r)
 groups=[]
 for profile in ['uniform','linear']:
  pairs=[];exact=True
  for i in inputs:
   if i['profile']!=profile:continue
   a=by[(i['sample_id'],'before')];b=by[(i['sample_id'],'after')]
   for x,y in zip(a,b):
    for f in ['support','nseeds','coordinates']:exact=exact and x[f]==y[f]
   pairs.append((statistics.median(float(x['ms']) for x in a),statistics.median(float(x['ms']) for x in b)))
  groups.append(dict(profile=profile,count=len(pairs),median_before_us=1000*statistics.median(a for a,b in pairs),median_after_us=1000*statistics.median(b for a,b in pairs),median_paired_ratio=statistics.median(b/a for a,b in pairs),sum_ratio=sum(b for a,b in pairs)/sum(a for a,b in pairs),seed_identity=exact))
 (P/'timing-summary.json').write_text(json.dumps({'groups':groups,'repetitions':5,'scope':'fresh magnifier and seed cache per call; centre roots/caustic construction excluded; no earlier epoch seed reuse'},indent=2)+'\n');print(groups)
if __name__=='__main__':main()
