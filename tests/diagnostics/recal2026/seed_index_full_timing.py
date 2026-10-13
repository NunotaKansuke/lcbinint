"""Fresh-epoch paired integration timing, without caustic construction."""
import os,subprocess,csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'tests/diagnostics/results/recal2026/seed-index-port-20261013';B=P.parent/'correction-accuracy-cost-20261013'
def main():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 for batch in range(5):
  for mode in (['before','after'] if batch%2==0 else ['after','before']):
   exe=B/'runner-phase-profile' if mode=='before' else P/'phase-after'
   subprocess.run([str(exe),str(B/'input.tsv'),str(P/f'full-{mode}-{batch}.tsv')],check=True)
  print('batch',batch,flush=True)
 inputs=json.loads((B/'inputs.json').read_text());allrows={}
 for mode in ['before','after']:
  for batch in range(5):
   for r in csv.DictReader((P/f'full-{mode}-{batch}.tsv').open(),delimiter='\t'):
    if r['trace']=='0':allrows.setdefault((mode,int(r['sample_id']),float(r['target'])),[]).append(r)
 groups=[]
 for profile in ['uniform','linear']:
  for target in [1e-3,1e-4]:
   pairs=[];exact=True;accepted=0;bad=0
   for i in inputs:
    if i['profile']!=profile:continue
    a=allrows[('before',i['sample_id'],target)];b=allrows[('after',i['sample_id'],target)]
    for x,y in zip(a,b):
     for f in ['magnification','met','maps','refinements']:exact=exact and x[f]==y[f]
    accepted+=int(b[0]['met']);bad+=int(b[0]['met']) and abs(float(b[0]['magnification'])/i['reference']-1)>target
    pairs.append((statistics.median(float(x['centre_ms'])+float(x['kernel_ms']) for x in a),statistics.median(float(x['centre_ms'])+float(x['kernel_ms']) for x in b)))
   groups.append(dict(profile=profile,target=target,count=len(pairs),median_before_ms=statistics.median(a for a,b in pairs),median_after_ms=statistics.median(b for a,b in pairs),median_paired_ratio=statistics.median(b/a for a,b in pairs),sum_ratio=sum(b for a,b in pairs)/sum(a for a,b in pairs),numerical_fields_exact=exact,accepted=accepted,false_accept=bad))
 (P/'full-summary.json').write_text(json.dumps({'groups':groups,'scope':'fresh seed cache per call; centre roots + adaptive kernel included; caustic/critical curve construction outside timer; initial16, relative target; five alternating batch orders,3 untraced calls per batch per input'},indent=2)+'\n');print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
