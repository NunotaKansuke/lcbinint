import json,math,collections,hashlib,re
from pathlib import Path
P=Path(__file__).resolve().parent;O=P.parent/'triple_event_precision_20261029';cases={};native={};counts={e:collections.Counter()for e in ['value','gradient']};baseline_regressions=[]
for group,inp,rows,offset,n in [('main',O/'inputs',P/'accepted-rows',0,1768),('followup',O/'followup/inputs',P/'accepted-rows',1768,32),('holdout',P/'holdout/inputs',P/'accepted-rows',1800,56)]:
 for i in range(n):
  cases[offset+i]=json.load(open(inp/f'{i:04d}.json'))
  for e in counts:
   f=rows/f'{group}-{i:04d}-{e}.json';a=json.load(open(f));native[offset+i,e]=a;counts[e][a['classification']]+=1
   assert a['input_sha256']==hashlib.sha256((inp/f'{i:04d}.json').read_bytes()).hexdigest()
   assert a['native_sha256']=='8fc87192444d742e1dfa48bad43800b03b638f09f476d1f8638020bce36a6534'
qualified={};exclusions=[]
def add(i,j,refs,source):
 assert all(math.isfinite(v)and v>0 for v in refs)
 for e in counts:
  a=native[i,e]
  if a['classification']!='ok':continue
  error=max(abs(a['mu'][j]/r-1)for r in refs);qualified[i,j,e]=dict(case=i,index=j,engine=e,native=a['mu'][j],reference_values=refs,reference_source=source,ratio=error/cases[i]['rtol'])
status=json.load(open(P/'accepted-status.json'));assert status['completed']==status['total']==3712 and status['state'].startswith('COMPLETE')
old=json.load(open(O/'polished-joint-comparisons.json'));assert len(old)==4502
for r in old:add(r['case'],r['index'],r['reference_values'],r['reference_source'])
for i in [400,403,433,460,467,484,515]:
 ref=json.load(open(P/(f'{i:04d}-mp-full-reference.json'if (P/f'{i:04d}-mp-full-reference.json').exists()else f'{i:04d}-mp-reference-v2.json')))
 for j in sorted({r['index']for r in ref['rows']}):
  values=[float(r['mu'])for r in ref['rows']if r['index']==j and r['n']>=8];assert max(values)/min(values)-1<=cases[i]['rtol']*.01
  add(i,j,values,'independent 70-digit original equation; 4 distinct images per sample; GL8/16, angles32/64')
for i in range(56):
 c=cases[1800+i];refs=[json.load(open(P/'holdout/rows'/f'{i:04d}-{e}-tight.json'))for e in ['MultiPoly','NoPoly']]
 if any(r['classification']!='ok'for r in refs):exclusions.append(dict(case=1800+i,reason='reference_curve_failure'))
 else:
  for j in range(5):
   values=[r['mu'][j]for r in refs]
   if not all(math.isfinite(v)and v>0 for v in values):reason='nonfinite_reference'
   elif all(abs(v-1)<=8*2.220446049250313e-16 for v in values):reason='unverified_unlensed_reference'
   elif any(not math.isfinite(r['ref_diagnostics'][j]['therr'])or r['ref_diagnostics'][j]['therr']<0 or r['ref_diagnostics'][j]['therr']>1e-7*abs(v)for r,v in zip(refs,values)):reason='reference_error_estimate_above_requested'
   elif max(values)/min(values)-1>2e-7:reason='reference_disagreement'
   else:reason=None
   if reason:exclusions.append(dict(case=1800+i,index=j,reason=reason))
   else:add(1800+i,j,values,'tight MP/NP mutual and reported-error gate at 1e-7; not an independent proof')
 ref=json.load(open(P/'holdout'/f'{i:04d}-mp-reference.json'));vs=[float(r['mu'])for r in ref['rows']if r['n']>=8];assert len(vs)==2 and max(vs)/min(vs)-1<c['rtol']*.01
 add(1800+i,2,vs,f"independent 70-digit original equation; {len(ref['rows'][0]['images'])} distinct images per sample; GL8/16, angles32/64")
q=list(qualified.values());(P/'qualified-comparisons.json').write_text(json.dumps(q,indent=2));(P/'reference-exclusions.json').write_text(json.dumps(exclusions,indent=2))
# Normal VBM accuracy uses independent references, including where NoPoly failed.
vbm=[];vbmcounters={};repair_ids=[400,403,433,460,467,484,515]
for e in ['SinglePoly','MultiPoly','NoPoly']:
 counter=collections.Counter()
 for i in range(63):
  r=json.load(open(P/'holdout/rows'/f'{i:04d}-{e}.json'));counter[r['classification']+(':'+str(r['returncode'])if 'returncode'in r else'')]+=1
  if r['classification']!='ok':continue
  globalid=1800+i if i<56 else repair_ids[i-56];truth=qualified[globalid,2,'value']['reference_values'];error=max(abs(r['mu'][2]/v-1)for v in truth);vbm.append(dict(case=i,engine=e,requested=cases[globalid]['rtol'],relative_error=error,ratio=error/cases[globalid]['rtol']))
 vbmcounters[e]=counter
(P/'vbm-independent-comparisons.json').write_text(json.dumps(vbm,indent=2))
invariants=json.load(open(P/'accepted-invariants.json'));invchecks=[dict(case=r['case'],name=k,**v)for r in invariants for k,v in r.get('checks',{}).items()];errors=[r for r in invariants if 'error'in r];exceed=[r for r in invchecks if r['value_relative']>cases[r['case']]['rtol']or r['gradient_scaled']>1e-4]
value_grad=[];warm=[]
for (i,e),r in native.items():
 if r['classification']!='ok':continue
 if 'info_mu'in r:warm.append(dict(case=i,engine=e,ratio=max(abs(x/y-1)for x,y in zip(r['mu'],r['info_mu']))/cases[i]['rtol']))
 if e=='value'and native[i,'gradient']['classification']=='ok':value_grad.append(dict(case=i,ratio=max(abs(x/y-1)for x,y in zip(r['mu'],native[i,'gradient']['mu']))/cases[i]['rtol']))
legacy=lambda f:set(re.findall(r'^FAILED (\S+)',(P/f).read_text(),re.M))
assert legacy('accepted-regression.log')==legacy('baseline-legacy.log')and len(legacy('accepted-regression.log'))==9
summary=dict(conditions=len(cases),epochs=sum(len(c['epochs'])for c in cases.values()),native_sha256='8fc87192444d742e1dfa48bad43800b03b638f09f476d1f8638020bce36a6534',counts=counts,remaining=sorted({i for (i,e),r in native.items()if r['classification']!='ok'}),reference_validation=dict(comparisons=len(q),points=len({(r['case'],r['index'])for r in q}),conditions=len({r['case']for r in q}),maximum_ratio=max(r['ratio']for r in q),exceedances=[r for r in q if r['ratio']>1],new_reference_exclusions=collections.Counter(r['reason']for r in exclusions)),invariants=dict(curves=len(invariants),checks=len(invchecks),errors=errors,exceedances=exceed,maximum_value=max(r['value_relative']for r in invchecks),maximum_scaled_gradient=max(r['gradient_scaled']for r in invchecks)),vbm_63_conditions=vbmcounters,vbm_independent_exceedances=[r for r in vbm if r['ratio']>1],maximum_worker_rss_kib=max(r.get('max_rss_kib',0)for r in native.values()),value_gradient_largest=sorted(value_grad,key=lambda r:r['ratio'],reverse=True)[:5],warm_largest=sorted(warm,key=lambda r:r['ratio'],reverse=True)[:5],tests=dict(cpp_passed=33,target_python_passed=171,legacy_python_passed=18,legacy_baseline_failures=sorted(legacy('accepted-regression.log'))))
(P/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
