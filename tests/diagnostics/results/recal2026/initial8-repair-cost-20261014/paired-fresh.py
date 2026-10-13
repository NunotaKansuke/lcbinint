import csv,os,subprocess,json,math,statistics
from pathlib import Path
p=Path(__file__).resolve().parent;base=p.parent/'initial8-local-repair-20261014'
data={r['geom_id']:r for r in json.loads((p/'fresh.json').read_text())}
input_lines=(p/'fresh-input.tsv').read_text().splitlines();rows=[]
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for i,line in enumerate(input_lines):
 (p/'paired-single-input.tsv').write_text(line+'\n')
 jobs=[('previous8',base/'runner',8),('final8',p/'runner',8),('control16',base/'runner',16)]
 jobs=jobs[i%3:]+jobs[:i%3]
 for name,exe,bins in jobs:
  env['LCBININT_BENCH_INITIAL_BINS']=str(bins)
  subprocess.run(['taskset','-c','0',str(exe),str(p/'paired-single-input.tsv'),str(p/'paired-single-output.tsv'),'adaptive'],env=env,check=True,stderr=subprocess.DEVNULL)
  for r in csv.DictReader((p/'paired-single-output.tsv').open(),delimiter='\t'):r['variant']=name;rows.append(r)
 if i%10==9:print(i+1,flush=True)
keys=list(rows[0]);
with (p/'fresh-paired.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys,delimiter='\t');w.writeheader();w.writerows(rows)
lookup={(r['sample_id'],r['target'],r['variant']):r for r in rows};summary=[]
for ld in [False,True]:
 for tol in ['0.001','0.0001']:
  ratios=[];improvements=[]
  for key,d in data.items():
   if (float(d['c'])!=0)!=ld:continue
   a,b,c=[lookup[key,tol,v] for v in ['final8','previous8','control16']]
   if all(r['converged']=='1' and abs(float(r['magnification'])/float(d['reference'])-1)<=float(tol) for r in [a,b,c]):ratios.append(float(c['ms'])/float(a['ms']));improvements.append(float(b['ms'])/float(a['ms']))
  summary.append(dict(profile='ld' if ld else 'uniform',target=float(tol),pairs=len(ratios),speed16_over_final8=math.exp(statistics.mean(map(math.log,ratios))),previous8_over_final8=math.exp(statistics.mean(map(math.log,improvements)))))
(p/'fresh-paired-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
