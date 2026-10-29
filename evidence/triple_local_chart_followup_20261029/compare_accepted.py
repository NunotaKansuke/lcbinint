import json,math,collections
from pathlib import Path
P=Path(__file__).resolve().parent;O=P.parent/'triple_event_precision_20261029';changes=[];regressions=[];deltas=[];counts=collections.defaultdict(collections.Counter)
for f in sorted((P/'accepted-rows').glob('*.json')):
 a=json.load(open(f));i=a['case'];e=a['engine'];g=a['group'];before=(O/('polished-rows'if g=='main'else'followup/polished-rows')if g!='holdout'else P/'holdout/before-rows')/f'{i:04d}-{e}.json';b=json.load(open(before));counts[e][a['classification']]+=1
 if a['classification']!=b['classification']:
  row=dict(case=i,group=g,engine=e,before=b['classification'],after=a['classification']);changes.append(row)
  if b['classification']=='ok':regressions.append(row)
 if a['classification']==b['classification']=='ok':
  delta=max(abs(x/y-1)for x,y in zip(a['mu'],b['mu']));deltas.append(dict(case=i,group=g,engine=e,relative=delta))
new_refs=[];gradient_refs=[]
for i in [400,403,433,460,467,484,515]:
 c=json.load(open(O/'inputs'/f'{i:04d}.json'));ref=json.load(open(P/f'{i:04d}-mp-reference-v2.json'))
 for e in ['value','gradient']:
  a=json.load(open(P/'accepted-rows'/f'main-{i:04d}-{e}.json'));assert a['classification']=='ok'
  for j in sorted({r['index']for r in ref['rows']}):
   vs=[float(r['mu'])for r in ref['rows']if r['index']==j and r['n']>=8];assert max(vs)/min(vs)-1<c['rtol']*.01
   error=max(abs(a['mu'][j]/v-1)for v in vs);new_refs.append(dict(case=i,index=j,engine=e,native=a['mu'][j],references=vs,ratio=error/c['rtol']))
 f=P/f'{i:04d}-mp-gradient-v2.json'
 if f.exists():
  a=json.load(open(P/'accepted-rows'/f'main-{i:04d}-gradient.json'));fd=json.load(open(f))
  for r in fd['rows']:
   name=r['parameter'];col=a['gradient_parameters'].index(name);native=a['gradients'][2][col];ref=float(r['richardson']);gradient_refs.append(dict(case=i,parameter=name,method=a['info']['stop'][2],native=native,finite_source_reference=ref,scaled_difference=abs(native-ref)/(1+abs(ref)),fd_scaled_scatter=abs(float(r['estimates'][1])-float(r['estimates'][0]))/(1+abs(ref))))
(P/'independent-gradient-comparison.json').write_text(json.dumps(gradient_refs,indent=2));(P/'independent-value-comparison.json').write_text(json.dumps(new_refs,indent=2))
s=dict(counts=counts,classification_changes=changes,regressions=regressions,largest_value_changes=sorted(deltas,key=lambda z:z['relative'],reverse=True)[:10],independent_value_comparisons=len(new_refs),independent_value_maximum_ratio=max(r['ratio']for r in new_refs),independent_value_exceedances=[r for r in new_refs if r['ratio']>1],finite_source_gradient_exceedances=[r for r in gradient_refs if r['scaled_difference']>1e-4]);(P/'comparison.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
