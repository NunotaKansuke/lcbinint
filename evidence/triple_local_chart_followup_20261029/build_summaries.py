import json,collections,math,statistics
from pathlib import Path
P=Path(__file__).resolve().parent;O=P.parent/'triple_event_precision_20261029';s=json.load(open(P/'SUMMARY.json'));refs=json.load(open(P/'qualified-comparisons.json'));native=json.load(open(P/'accepted-controlled-timing.json'));vbm=json.load(open(P/'controlled-timing.json'));rows=[]
assert len(native)==52 and all('warm'in r and 'error'not in r for r in native)
for i in [21,400,403,433,460,467,484,515,483,940,972,1508,514]:
 c=json.load(open(O/'inputs'/f'{i:04d}.json'));r=dict(case=i,epochs=len(c['epochs']),requested_rtol=c['rtol'],native={},vbm={},reference_points=len({v['index']for v in refs if v['case']==i}))
 for record in native:
  if record['case']!=i:continue
  z=dict(cold_ms=record['first']*1000,warm_ms=record['warm']*1000,sha256=record['sha256']);qualified=[v for v in refs if v['case']==i and v['engine']=='value'];z['maximum_reference_ratio']=max((max(abs(record['mu'][v['index']]/t-1)for t in v['reference_values'])/c['rtol']for v in qualified),default=None);r['native'][record['engine']]=z
 for record in vbm:
  if record['case']!=i or record['engine'].startswith('native'):continue
  z=dict(record);qualified=[v for v in refs if v['case']==i and v['engine']=='value']
  if 'warm'in z:
   z['warm_ms']=z['warm']*1000;z['maximum_reference_ratio']=max((max(abs(z['mu'][v['index']]/t-1)for t in v['reference_values'])/c['rtol']for v in qualified),default=None)
  r['vbm'][z['engine']]=z
 rows.append(r)
(P/'timing-summary.json').write_text(json.dumps(rows,indent=2))
profile=[]
for line in (P/'stage-profile.jsonl').read_text().splitlines():
 r=json.loads(line);r['exclusive_seconds']=dict(zip(['topology','endpoint','chart','pair','angular','witness'],r['exclusive_seconds']));r['other_seconds']=r['integral_seconds']-sum(r['exclusive_seconds'].values());profile.append(r)
assert len(profile)==6;(P/'stage-profile-summary.json').write_text(json.dumps(profile,indent=2))
remaining=[]
for i in s['remaining']:
 for e in ['value','gradient']:
  r=json.load(open(P/'accepted-rows'/f'main-{i:04d}-{e}.json'));remaining.append(dict(case=i,engine=e,input=json.load(open(O/'inputs'/f'{i:04d}.json')),error=r['error'],raw_native=r['raw_native']))
(P/'remaining-failures.json').write_text(json.dumps(remaining,indent=2))
c=json.load(open(P/'holdout/inputs/0032.json'));counter=dict(case=32,input=c,reference=json.load(open(P/'holdout/0032-mp-reference.json')),observations=[json.load(open(P/f'vbm-replay/0032-MultiPoly-{k}.json'))for k in [1,2]],native=json.load(open(P/'accepted-rows/holdout-0032-value.json')));(P/'VBM_COUNTEREXAMPLE.json').write_text(json.dumps(counter,indent=2))
(P/'test-summary.json').write_text(json.dumps(s['tests'],indent=2))
comparison=json.load(open(P/'comparison.json'));changes=collections.Counter(r['engine']for r in comparison['classification_changes']);assert not comparison['regressions'] and changes=={'value':40,'gradient':41};assert not s['reference_validation']['exceedances']
print('references',s['reference_validation'],'timing curves',len(rows),'classification recoveries',dict(changes))
