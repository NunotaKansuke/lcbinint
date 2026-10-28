from pathlib import Path
import json,collections,sys
P=Path(__file__).resolve().parent;B=P.parent/'triple_final_polish_20261029';cases=json.loads((P/'cases.json').read_text());variant=sys.argv[1];D=P/(variant+'-rows');counts={e:collections.Counter()for e in ['value','gradient']};changes=[];changed=[];failures=[]
for i,c in enumerate(cases):
 for e in counts:
  f=D/f'{i:04d}-{e}.json'
  if not f.exists():continue
  a=json.loads(f.read_text());b=json.loads(((B/'mass-bound-rows'/f'{i:04d}-{e}.json')if i<1528 else(B/'holdout/rows'/f'{i-1528:04d}-{e}.json')).read_text());counts[e][a['classification']]+=1
  if a['classification']!=b['classification']:changes.append(dict(case=i,engine=e,before=b['classification'],after=a['classification']))
  if a['classification']!='ok':failures.append(dict(case=i,engine=e,classification=a['classification'],family=c['family'],stop=a.get('raw_native',{}).get('stop')))
  if a['classification']==b['classification']=='ok':
   delta=max(abs(x/y-1)for x,y in zip(a['mu'],b['mu']))
   if delta>c['rtol']*.1:changed.append(dict(case=i,engine=e,relative_change=delta,requested=c['rtol']))
s=dict(counts={k:dict(v)for k,v in counts.items()},changes=changes,regressions=[x for x in changes if x['before']=='ok'],changed=changed,failures=failures)
(P/(variant+'-comparison.json')).write_text(json.dumps(s,indent=2));print({k:v for k,v in s.items()if k not in ['changes','failures','changed']});print('recoveries',len([x for x in changes if x['after']=='ok']),'changed',changed)
refs=json.loads((B/'final-validated-reference-comparisons.json').read_text())
for r in json.loads((B/'holdout/validated-comparisons.json').read_text()):r['case']+=1528;refs.append(r)
rows=[]
for r in refs:
 i,e,j=r['case'],r['engine'],r['index'];f=D/f'{i:04d}-{e}.json'
 if not f.exists():continue
 a=json.loads(f.read_text())
 if a['classification']!='ok':continue
 r=dict(r);r['native']=a['mu'][j];r['relative_error']=max(abs(r['native']/x-1)for x in r['reference_values']);r['ratio']=r['relative_error']/r['requested'];rows.append(r)
z=dict(comparisons=len(rows),points=len({(r['case'],r['index'])for r in rows}),maximum_ratio=max((r['ratio']for r in rows),default=None),exceedances=[r for r in rows if r['ratio']>1]);(P/(variant+'-validation.json')).write_text(json.dumps(z,indent=2));print(z)
