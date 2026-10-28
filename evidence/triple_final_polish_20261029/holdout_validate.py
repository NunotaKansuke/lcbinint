from pathlib import Path
import json,collections,math
P=Path(__file__).resolve().parent;D=P/'holdout';cases=json.loads((D/'cases.json').read_text());counts={};rows=[];excluded=[];failures=[];vbm_discrepancies=[]
engines=['value','gradient','SinglePoly','MultiPoly','NoPoly','MultiPoly-tight','NoPoly-tight']
for e in engines:
 counts[e]=collections.Counter()
for i,c in enumerate(cases):
 z={e:json.loads((D/'rows'/f'{i:04d}-{e}.json').read_text())for e in engines}
 for e,r in z.items():
  counts[e][r['classification']]+=1
  if e in ['value','gradient']and r['classification']!='ok':failures.append(dict(case=i,engine=e,classification=r['classification'],family=c['family']))
 refs=[z[e]for e in ['MultiPoly-tight','NoPoly-tight']]
 if not all(r['classification']=='ok'and len(r.get('mu',[]))==len(c['epochs'])for r in refs):excluded.append(dict(case=i,reason='reference_curve_failure'));continue
 for j in range(len(c['epochs'])):
  vs=[r['mu'][j]for r in refs]
  if any(not x or not math.isfinite(x)or x<=0 for x in vs):excluded.append(dict(case=i,index=j,reason='invalid_reference'));continue
  if c['params']['rho']>0:
   diagnostics=[r['ref_diagnostics'][j]for r in refs]
   if any(v['therr']>1e-7*abs(v['mu'])for v in diagnostics):excluded.append(dict(case=i,index=j,reason='reference_error_estimate_above_requested'));continue
  spread=abs(vs[0]/vs[1]-1)
  if spread>c['rtol']*.1:excluded.append(dict(case=i,index=j,reason='reference_engine_disagreement',values=vs));continue
  for e in ['value','gradient']:
   r=z[e]
   if r['classification']!='ok':continue
   native=r['mu'][j];error=max(abs(native/x-1)for x in vs);rows.append(dict(case=i,index=j,engine=e,native=native,reference_values=vs,reference_spread=spread,relative_error=error,requested=c['rtol'],ratio=error/c['rtol']))
  for e in ['SinglePoly','MultiPoly','NoPoly']:
   r=z[e]
   if r['classification']!='ok':continue
   value=r['mu'][j];error=max(abs(value/x-1)for x in vs)
   if error>c['rtol']:vbm_discrepancies.append(dict(case=i,index=j,engine=e,value=value,reference_values=vs,relative_error=error,requested=c['rtol']))
summary=dict(cases=len(cases),epochs=sum(len(c['epochs'])for c in cases),counts={e:dict(v)for e,v in counts.items()},native_failures=failures,comparisons=len(rows),points=len({(r['case'],r['index'])for r in rows}),conditions=len({r['case']for r in rows}),maximum_ratio=max((r['ratio']for r in rows),default=None),exceedances=[r for r in rows if r['ratio']>1],excluded=dict(collections.Counter(r['reason']for r in excluded)),vbm_discrepancies=len(vbm_discrepancies))
(D/'validation-summary.json').write_text(json.dumps(summary,indent=2));(D/'validated-comparisons.json').write_text(json.dumps(rows,indent=2));(D/'excluded-references.json').write_text(json.dumps(excluded,indent=2));(D/'vbm-discrepancies.json').write_text(json.dumps(vbm_discrepancies,indent=2));print(summary)
