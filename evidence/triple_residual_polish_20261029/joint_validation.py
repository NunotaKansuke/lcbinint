import json,sys,collections
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent/'triple_final_polish_20261029';variant=sys.argv[1];D=P/(variant+'-rows');refs={}
old=json.loads((B/'final-validated-reference-comparisons.json').read_text())
for r in json.loads((B/'holdout/validated-comparisons.json').read_text()):r['case']+=1528;old.append(r)
old+=json.loads((P/(variant+'-new-reference-comparisons.json')).read_text())
for r in old:
 key=(r['case'],r['index'],r['engine'])
 if key not in refs:refs[key]=dict(r,reference_values=list(r['reference_values']))
 else:refs[key]['reference_values']+=r['reference_values']
rows=[]
for (i,j,e),r in refs.items():
 a=json.loads((D/f'{i:04d}-{e}.json').read_text())
 if a['classification']!='ok':continue
 r=dict(r,native=a['mu'][j]);r['relative_error']=max(abs(r['native']/x-1)for x in r['reference_values']);r['ratio']=r['relative_error']/r['requested'];rows.append(r)
s=dict(comparisons=len(rows),points=len({(r['case'],r['index'])for r in rows}),conditions=len({r['case']for r in rows}),maximum_ratio=max(r['ratio']for r in rows),exceedances=[r for r in rows if r['ratio']>1]);(P/(variant+'-joint-validation.json')).write_text(json.dumps(s,indent=2));(P/(variant+'-joint-comparisons.json')).write_text(json.dumps(rows,indent=2));print(s)
