"""Reference gates include the LD annulus-cap audit, not just engine agreement."""
from pathlib import Path
import json,collections
P=Path(__file__).resolve().parent
cases=json.loads((P/'cases.json').read_text());refs={};exclusions=[]
old=json.loads((P.parent/'triple_hardening_20261028/validated-reference-comparisons.json').read_text())
for r in old:
 i,j=r['case'],r['index']
 if cases[i]['limb']>0 and 'independent'not in r['reference_source']:
  audits=[P/'ld-cap-audit'/f'{i:04d}-{j}-{k}.json'for k in [0,1]]
  if not all(f.exists()for f in audits):exclusions.append(dict(case=i,index=j,reason='missing_ld_audit'));continue
  zz=[json.loads(f.read_text())for f in audits]
  if any(z.get('cap_exhausted')or'error'in z or'result'not in z for z in zz):exclusions.append(dict(case=i,index=j,reason='ld_cap_or_audit_failure'));continue
 refs[i,j]=dict(reference_values=r['reference_values'],reference_source=r['reference_source'])
for i in json.loads((P/'comparison.json').read_text())['recovered_conditions']:
 if cases[i]['limb']>0:continue
 fs=[P/'recovered-tight'/f'{i:04d}-{e}.json'for e in ['MultiPoly','NoPoly']]
 zz=[json.loads(f.read_text())for f in fs]
 if not all(z.get('classification')=='ok'for z in zz):continue
 for j in range(len(cases[i]['epochs'])):
  vs=[z['mu'][j]for z in zz]
  if abs(vs[0]/vs[1]-1)<=cases[i]['rtol']*.1:refs[i,j]=dict(reference_values=vs,reference_source='recovered VBM uniform RelTol=1e-7')
wide=json.loads((P/'ld-independent-wide-comparison.json').read_text())
for r in wide['rows']:
 refs[r['case'],r['index']]=dict(reference_values=r['reference'],reference_source='independent LD uniform disks RelTol=1e-6, nested GL convergence and MP/NP agreement')
# Further references for the six previously budget-limited curves.
for i in [135,189]:
 zz=[json.loads((P/'annulus-tight'/f'{i:04d}-{e}.json').read_text())for e in ['MultiPoly','NoPoly']]
 if all(z.get('classification')=='ok'for z in zz):
  for j in range(len(cases[i]['epochs'])):
   vs=[z['mu'][j]for z in zz]
   if abs(vs[0]/vs[1]-1)<=cases[i]['rtol']*.1:refs[i,j]=dict(reference_values=vs,reference_source='VBM uniform strict references after annulus recovery')
for i in [135,189,314,337,361,431]:
 for j in range(len(cases[i]['epochs'])):
  accepted=[]
  for method,directory in [('Multipoly','point-disk-vbm-frame'),('Nopoly','point-disk-vbm-frame'),('Singlepoly','point-disk-singlepoly')]:
   f=P/directory/f'{i:04d}-{j}-{method}.json'
   if not f.exists():continue
   z=json.loads(f.read_text());v=[r['mu']for r in z['rows']]
   if len(v)==3 and all(x>0 for x in v) and max(abs(v[2]-v[1]),abs(v[1]-v[0]))<=cases[i]['rtol']*.02*abs(v[2]):accepted.append((method,v[2]))
  if len(accepted)<2:continue
  # Prefer the two stable methods agreeing most closely; preserve the full
  # per-method raw evidence rather than silently substituting an outlier.
  pairs=[(abs(a[1]/b[1]-1),a,b)for n,a in enumerate(accepted)for b in accepted[n+1:]]
  spread,a,b=min(pairs)
  if spread<=cases[i]['rtol']*.02:refs[i,j]=dict(reference_values=[a[1],b[1]],reference_source='independent point-source disk average '+a[0]+'/'+b[0]+', three quadrature meshes')

rows=[]
for i,c in enumerate(cases):
 for e in ['value','gradient']:
  a=json.loads((P/'mass-bound-rows'/f'{i:04d}-{e}.json').read_text())
  for j,mu in enumerate(a.get('mu',[])):
   if not mu or a['status'][j] or (i,j)not in refs:continue
   if e=='gradient'and not a['gradient_valid'][j]:continue
   r=dict(refs[i,j],case=i,index=j,engine=e,native=mu,requested=c['rtol'])
   r['relative_error']=max(abs(mu/b-1)for b in r['reference_values']);r['ratio']=r['relative_error']/c['rtol'];rows.append(r)
summary=dict(comparisons=len(rows),conditions=len({r['case']for r in rows}),points=len({(r['case'],r['index'])for r in rows}),exceedances=[r for r in rows if r['ratio']>1],maximum_ratio=max(r['ratio']for r in rows),sources=dict(collections.Counter(r['reference_source']for r in rows)),excluded_prior_comparisons=dict(collections.Counter(r['reason']for r in exclusions)))
(P/'final-validated-reference-summary.json').write_text(json.dumps(summary,indent=2));(P/'final-validated-reference-comparisons.json').write_text(json.dumps(rows,indent=2));(P/'excluded-prior-references.json').write_text(json.dumps(exclusions,indent=2));print(summary)
