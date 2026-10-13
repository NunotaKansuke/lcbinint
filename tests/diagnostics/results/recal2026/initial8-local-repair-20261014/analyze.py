import csv,json,math,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
def read(name):
 return {(r['sample_id'],r['target']):r for r in csv.DictReader((p/name).open(),delimiter='\t')}
def error(r,d):
 return abs(float(r['magnification'])/float(d['reference'])-1)
def ok(r,d):
 return math.isfinite(error(r,d)) and error(r,d)<=float(r['target'])
def operational(r,d):
 return r['converged']=='1' and ok(r,d)
def geo(v):
 return math.exp(sum(math.log(x) for x in v)/len(v)) if v else None
summary={}
for cohort in ['development','holdout','prospective']:
 inputs={r['geom_id']:r for r in json.loads((p/f'{cohort}.json').read_text())}
 eight=read(f'{cohort}-validated8.tsv');sixteen=read(f'{cohort}-validated16.tsv')
 old=read(f'{cohort}-'+('old8.tsv' if cohort=='prospective' else 'original8.tsv'))
 reference16=read(f'{cohort}-'+('final16.tsv' if cohort=='prospective' else '16.tsv'))
 changed16=[k for k in sixteen if any(sixteen[k][v]!=reference16[k][v] for v in ['magnification','converged','maps','stop_reason','indicator'])]
 groups=[]
 for profile in ['uniform','ld']:
  for target in ['0.001','0.0001']:
   keys=[k for k in eight if k[1]==target and ('uniform' if float(inputs[k[0]]['c'])==0 else 'ld')==profile]
   speed=[float(sixteen[k]['ms'])/float(eight[k]['ms']) for k in keys if operational(eight[k],inputs[k[0]]) and operational(sixteen[k],inputs[k[0]])]
   groups.append(dict(profile=profile,target=float(target),n=len(keys),actual_pass8=sum(ok(eight[k],inputs[k[0]]) for k in keys),actual_pass16=sum(ok(sixteen[k],inputs[k[0]]) for k in keys),actual_pass_old8=sum(ok(old[k],inputs[k[0]]) for k in keys),accepted8=sum(eight[k]['converged']=='1' for k in keys),false_accept8=[k[0] for k in keys if eight[k]['converged']=='1' and not ok(eight[k],inputs[k[0]])],false_accept16=[k[0] for k in keys if sixteen[k]['converged']=='1' and not ok(sixteen[k],inputs[k[0]])],new_accuracy_regressions=[k[0] for k in keys if ok(sixteen[k],inputs[k[0]]) and not ok(eight[k],inputs[k[0]])],valid_timing_pairs=len(speed),speed16_over8_geomean=geo(speed)))
 summary[cohort]=dict(groups=groups,changed_default16=changed16)
eager=read('prospective-eager8.tsv');eight=read('prospective-validated8.tsv')
summary['deferred_eager_numeric_differences']=[k for k in eager if any(eager[k][v]!=eight[k][v] for v in ['magnification','converged','stop_reason','indicator'])]
summary['cohort_roles']={'development':'tuned','holdout':'reviewed and tuned; not final holdout','prospective':'lens cases absent from other cohorts; selected before final mechanical cache/row reuse changes'}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

paired={(r['sample_id'],r['target'],r['nbin']):r for r in csv.DictReader((p/'prospective-paired.tsv').open(),delimiter='\t')}
inputs={r['geom_id']:r for r in json.loads((p/'prospective.json').read_text())}
groups=[]
for profile in ['uniform','ld']:
 for target in ['0.001','0.0001']:
  ratios=[]
  for key,d in inputs.items():
   if ('uniform' if float(d['c'])==0 else 'ld')!=profile:continue
   a,b=paired[key,target,'8'],paired[key,target,'16']
   if operational(a,d) and operational(b,d):ratios.append(float(b['ms'])/float(a['ms']))
  groups.append(dict(profile=profile,target=float(target),pairs=len(ratios),speed16_over8_geomean=geo(ratios)))
(p/'paired-summary.json').write_text(json.dumps(groups,indent=2)+'\n')
