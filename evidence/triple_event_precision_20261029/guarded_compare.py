from pathlib import Path
import json,collections,math
P=Path(__file__).resolve().parent;B=P.parent/'triple_residual_polish_20261029';cases=json.loads((P/'cases.json').read_text());counts={e:collections.Counter()for e in ['value','gradient']};changes=[];regressions=[];grad_changes=[];value_changes=[];selected=[];cpu={e:[0.,0.]for e in counts}
for i,c in enumerate(cases):
 for e in counts:
  a=json.loads((P/'guarded-rows'/f'{i:04d}-{e}.json').read_text());b=json.loads((B/'deferred-rows'/f'{i:04d}-{e}.json').read_text());counts[e][a['classification']]+=1
  if a['classification']!=b['classification']:
   r=dict(case=i,engine=e,before=b['classification'],after=a['classification']);changes.append(r)
   if b['classification']=='ok':regressions.append(r)
  if a['classification']!=b['classification']or a['classification']!='ok':continue
  cpu[e][0]+=b['cpu_seconds'];cpu[e][1]+=a['cpu_seconds'];delta=max(abs(x/y-1)for x,y in zip(a['mu'],b['mu']))
  if delta:value_changes.append(dict(case=i,engine=e,relative=delta,ratio=delta/c['rtol']))
  if e=='gradient':
   if any('primal_work_primary_'in s for s in a['info']['stop']):selected.append(i)
   error=max(abs(x-y)/(1+abs(y))for aa,bb in zip(a['gradients'],b['gradients'])for x,y in zip(aa,bb))
   if error>1e-4:grad_changes.append(dict(case=i,scaled=error,cpu_before=b['cpu_seconds'],cpu_after=a['cpu_seconds']))
refs=json.loads((B/'deferred-joint-comparisons.json').read_text());rows=[]
for r in refs:
 a=json.loads((P/'guarded-rows'/f"{r['case']:04d}-{r['engine']}.json").read_text())
 if a['classification']!='ok':continue
 r=dict(r,native=a['mu'][r['index']]);r['relative_error']=max(abs(r['native']/x-1)for x in r['reference_values']);r['ratio']=r['relative_error']/r['requested'];rows.append(r)
summary=dict(counts={e:dict(x)for e,x in counts.items()},regressions=regressions,changes=changes,selected=selected,selected_curves=len(selected),gradient_changes=sorted(grad_changes,key=lambda x:x['scaled'],reverse=True),cpu_sums_uncontrolled=cpu,reference_comparisons=len(rows),reference_points=len({(r['case'],r['index'])for r in rows}),maximum_ratio=max(r['ratio']for r in rows),exceedances=[r for r in rows if r['ratio']>1],value_changes=sorted(value_changes,key=lambda x:x['relative'],reverse=True));(P/'guarded-comparison.json').write_text(json.dumps(summary,indent=2));(P/'guarded-validated-comparisons.json').write_text(json.dumps(rows,indent=2));print({k:v for k,v in summary.items()if k not in ['gradient_changes','value_changes','selected']});print('largest gradient changes',summary['gradient_changes'][:8]);print('largest values',summary['value_changes'][:3])
