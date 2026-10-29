import json,collections,sys,math
from pathlib import Path
P=Path(__file__).resolve().parent/'followup';variant='polished';D=P/'polished-rows';cases=json.loads((P/'cases.json').read_text());rows=[];excluded=[];vbm=[]
for f in sorted((P/'rows').glob('*-MultiPoly-tight.json')):
 i=int(f.name.split('-')[0]);c=cases[i];rr=[json.loads((P/'rows'/f'{i:04d}-{e}-tight.json').read_text())for e in ['MultiPoly','NoPoly']]
 if any(r['classification']!='ok'for r in rr):excluded.append(dict(case=i,reason='reference_curve_failure'));continue
 for j in range(len(c['epochs'])):
  vs=[r['mu'][j]for r in rr]
  if any(not x or not math.isfinite(x)or x<=0 for x in vs):excluded.append(dict(case=i,index=j,reason='invalid_reference'));continue
  if all(abs(v-1)<=8*sys.float_info.epsilon for v in vs):excluded.append(dict(case=i,index=j,reason='unverified_unlensed_reference'));continue
  if c['params']['rho']>0 and any(r['ref_diagnostics'][j]['therr']>1e-7*abs(r['mu'][j])for r in rr):excluded.append(dict(case=i,index=j,reason='reference_error_estimate_above_requested'));continue
  if abs(vs[0]/vs[1]-1)>c['rtol']*.1:excluded.append(dict(case=i,index=j,reason='reference_engine_disagreement',reference_values=vs));continue
  for e in ['value','gradient']:
   a=json.loads((D/f'{i:04d}-{e}.json').read_text())
   if a['classification']!='ok':continue
   mu=a['mu'][j];error=max(abs(mu/v-1)for v in vs);rows.append(dict(case=i,index=j,engine=e,reference_values=vs,native=mu,relative_error=error,requested=c['rtol'],ratio=error/c['rtol']))
  for e in ['SinglePoly','MultiPoly','NoPoly']:
   a=json.loads((P/'rows'/f'{i:04d}-{e}.json').read_text())
   if a['classification']=='ok':
    error=max(abs(a['mu'][j]/v-1)for v in vs)
    if error>c['rtol']:vbm.append(dict(case=i,index=j,engine=e,error=error))
s=dict(comparisons=len(rows),points=len({(r['case'],r['index'])for r in rows}),conditions=len({r['case']for r in rows}),maximum_ratio=max((r['ratio']for r in rows),default=None),exceedances=[r for r in rows if r['ratio']>1],exclusions=dict(collections.Counter(x['reason']for x in excluded)),vbm_exceedances=vbm);(P/(variant+'-new-reference-validation.json')).write_text(json.dumps(s,indent=2));(P/(variant+'-new-reference-comparisons.json')).write_text(json.dumps(rows,indent=2));(P/(variant+'-new-reference-exclusions.json')).write_text(json.dumps(excluded,indent=2));print(s)

counts={};regressions=[];recoveries=[];failures=[]
for e in ['value','gradient']:
 a=collections.Counter();b=collections.Counter()
 for i in range(len(cases)):
  old=json.loads((P/'baseline-rows'/f'{i:04d}-{e}.json').read_text());new=json.loads((D/f'{i:04d}-{e}.json').read_text());a[old['classification']]+=1;b[new['classification']]+=1
  if old['classification']=='ok'and new['classification']!='ok':regressions.append(dict(case=i,engine=e))
  if old['classification']!='ok'and new['classification']=='ok':recoveries.append(dict(case=i,engine=e))
  if new['classification']!='ok':failures.append(dict(case=i,engine=e,stop=new.get('raw_native',{}).get('stop')))
 counts[e]=dict(before=dict(a),after=dict(b))
s=dict(counts=counts,regressions=regressions,recoveries=recoveries,remaining=failures);(P/'polished-comparison.json').write_text(json.dumps(s,indent=2));print(s)
