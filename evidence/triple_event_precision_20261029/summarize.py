import json,collections,hashlib,re
from pathlib import Path
P=Path(__file__).resolve().parent;variant='polished';cases=json.load(open(P/'cases.json'));new=json.load(open(P/'followup/cases.json'));counts={};remaining=[];mode_checks=[];warm_checks=[];rss=[]
for group,cc,offset in [('',cases,0),('followup/',new,len(cases))]:
 for i,c in enumerate(cc):
  pair={}
  for e in ['value','gradient']:
   z=json.load(open(P/group/(variant+'-rows')/f'{i:04d}-{e}.json'));counts.setdefault(e,collections.Counter())[z['classification']]+=1;pair[e]=z;rss.append(z.get('max_rss_kib',0))
   if z['classification']!='ok':remaining.append(dict(case=offset+i,group=group or'main',engine=e,stop=z.get('raw_native',{}).get('stop')))
   if z['classification']=='ok'and'info_mu'in z:warm_checks.append(dict(case=offset+i,engine=e,ratio=max(abs(x/y-1)for x,y in zip(z['mu'],z['info_mu']))/c['rtol']))
  if all(z['classification']=='ok'for z in pair.values()):mode_checks.append(dict(case=offset+i,ratio=max(abs(x/y-1)for x,y in zip(pair['value']['mu'],pair['gradient']['mu']))/c['rtol']))
inv=json.load(open(P/(variant+'-invariants.json')));rr=[dict(case=r['case'],name=n,**{k:c[k]for k in ['value_relative','gradient_scaled','status','valid']})for r in inv for n,c in r.get('checks',{}).items()];bad=[r for r in rr if any(r['status'])or not all(r['valid'])or r['value_relative']>cases[r['case']]['rtol']or r['gradient_scaled']>1e-4];isummary=dict(curves=len(inv),checks=len(rr),errors=[r for r in inv if'error'in r],exceedances=bad,maximum_value=max(r['value_relative']for r in rr),maximum_gradient=max(r['gradient_scaled']for r in rr));(P/'invariant-summary-final.json').write_text(json.dumps(isummary,indent=2));(P/'remaining-failures-final.json').write_text(json.dumps(remaining,indent=2))
s=dict(conditions=len(cases)+len(new),epochs=sum(len(c['epochs'])for c in cases+new),counts={e:dict(n)for e,n in counts.items()},native_sha256=hashlib.sha256((P/(variant+'-package/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so')).read_bytes()).hexdigest(),maximum_worker_rss_kib=max(rss),largest_value_gradient_differences=sorted(mode_checks,key=lambda r:r['ratio'],reverse=True)[:5],largest_warm_differences=sorted(warm_checks,key=lambda r:r['ratio'],reverse=True)[:5],invariants=isummary,reference_validation=json.load(open(P/(variant+'-joint-validation.json'))));(P/'SUMMARY.json').write_text(json.dumps(s,indent=2));print(s)
