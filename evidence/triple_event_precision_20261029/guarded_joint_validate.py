import json,collections,math
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent/'triple_residual_polish_20261029';cases=json.load(open(P/'cases.json'));rows={};excluded=[]
def add(i,j,refs,source):
 c=cases[i]
 if all(abs(v-1)<=8*2.220446049250313e-16 for v in refs)and '80-digit'not in source:
  if not any(r.get('case')==i and r.get('index')==j and r.get('reason')=='unverified_unlensed_reference'for r in excluded):excluded.append(dict(case=i,index=j,reason='unverified_unlensed_reference'))
  return
 for e in ['value','gradient']:
  a=json.load(open(P/'guarded-rows'/f'{i:04d}-{e}.json'))
  if a['classification']!='ok':continue
  value=a['mu'][j];error=max(abs(value/v-1)for v in refs);key=(i,j,e)
  rows[key]=dict(case=i,index=j,engine=e,native=value,reference_values=refs,reference_source=source,requested=c['rtol'],relative_error=error,ratio=error/c['rtol'])
for r in json.load(open(B/'deferred-joint-comparisons.json')):add(r['case'],r['index'],r['reference_values'],r.get('reference_source','inherited tight MP/NP gate'))
for i in [444,483,504,514]:
 folder='final-references'if i in [444,483]else'new-references';refs=[json.load(open(P/folder/f'{i:04d}-{e}-tight.json'))for e in ['MultiPoly','NoPoly']]
 if any(r['classification']!='ok'for r in refs):excluded.append(dict(case=i,reason='reference_curve_failure'));continue
 for j in range(len(cases[i]['epochs'])):
  vs=[r['mu'][j]for r in refs]
  if any(r['ref_diagnostics'][j]['therr']>1e-7*abs(v)for r,v in zip(refs,vs)):excluded.append(dict(case=i,index=j,reason='reference_error_estimate_above_requested'));continue
  if abs(vs[0]/vs[1]-1)>cases[i]['rtol']*.1:excluded.append(dict(case=i,index=j,reason='reference_engine_disagreement'));continue
  add(i,j,vs,'tight MP/NP, estimates <= 1e-7 mu and mutual agreement <= 0.1 native rtol')
# Independent LD integral bypasses the shared VBM LD annulus cap. Require
# all three nested meshes for both polynomial routes and a 0.1-budget envelope.
ind=[json.load(open(P/f'independent-values/0483-{e}.json'))for e in ['Multipoly','Nopoly']]
for j in range(5):
 refs=[next(r['mu']for r in a['rows']if r['n']==n and r['index']==j)for a in ind for n in [128,256,512]]
 assert all(math.isfinite(x)and x>0 for x in refs)
 assert max(refs)/min(refs)-1<=cases[483]['rtol']*.1
 add(483,j,refs,'independent uniform disks RelTol=1e-9; LD GL 128/256/512 with MP/NP; envelope <= 0.1 native rtol')
# Both tight VBM MP/NP methods returned exactly 1 at case 514, with reported error 0.
# Independent 80-digit original-equation solves retain all four images and
# supply a source-disk reference; never promote VBM mutual agreement alone.
far=json.load(open(P/'independent-far-disk.json'))
for j in range(5):
 refs=[float(r['mu'])for r in far['rows']if r['index']==j]
 assert len(refs)==3 and max(refs)/min(refs)-1<=1e-12
 add(514,j,refs,'independent 80-digit original equation; 4-image source-disk GL2/4/8 and 8/16/32 angles; validation only')
main=list(rows.values());extra=json.load(open(P/'followup/guarded-new-reference-comparisons.json'));joint=main+[dict(r,case=r['case']+len(cases),reference_source='fresh seed tight MP/NP')for r in extra];summary=dict(comparisons=len(joint),main_comparisons=len(main),followup_comparisons=len(extra),points=len({(r['case'],r['index'])for r in joint}),conditions=len({r['case']for r in joint}),maximum_ratio=max(r['ratio']for r in joint),exceedances=[r for r in joint if r['ratio']>1],additional_reference_exclusions=collections.Counter(x['reason']for x in excluded));(P/'guarded-joint-validation.json').write_text(json.dumps(summary,indent=2));(P/'guarded-joint-comparisons.json').write_text(json.dumps(joint,indent=2));(P/'guarded-additional-reference-exclusions.json').write_text(json.dumps(excluded,indent=2));print(summary)
