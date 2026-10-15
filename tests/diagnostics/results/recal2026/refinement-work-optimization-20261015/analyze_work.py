import json,statistics,sys
from pathlib import Path
P=Path(__file__).resolve().parent;a=json.loads((P/'trace128.json').read_text())['records'];b=json.loads((P/'baseline128.json').read_text())['records'];changed=[];groups=[]
for x,y in zip(a,b):
 gx=x.get('result',{});gy=y.get('result',{})
 if {k:v for k,v in gx.items()if k!='refinement_work_trace'}!=gy or x.get('exception')!=y.get('exception'):changed.append([x['id'],x['method'],x['target']])
for method in ['cartesian','polar']:
 for profile in ['uniform','linear']:
  for tol in [1e-3,1e-4]:
   rows=[r for r in a if r['method']==method and r['profile']==profile and r['target']==tol];last=[];excess=[];steps=[];initialpass=0;tt=0;et=0;kind={};per=[]
   for r in rows:
    tr=r.get('result',{}).get('refinement_work_trace',[])
    if not tr:continue
    ref=r['reference'];errors=[abs(v[7]/ref-1)for v in tr];stable=next((j for j in range(len(tr))if all(e<=tol for e in errors[j:])),None);total=sum(v[2]for v in tr[1:]);late=sum(v[2]for v in tr[stable+1:])if stable is not None else 0
    tt+=total;et+=late;initialpass+=stable==0;steps.append(len(tr)-1)
    if len(tr)>1 and total>0:last.append(tr[-1][2]/total)
    if total>0:excess.append(late/total)
    for v in tr[1:]:kind.setdefault(str(v[1]),[0,0,0]);kind[str(v[1])][0]+=1;kind[str(v[1])][1]+=v[2];kind[str(v[1])][2]+=v[3]
    per.append(dict(id=r['id'],steps=len(tr)-1,stable=stable,seconds=total,after_pass=late,last_error_over_target=errors[-1]/tol))
   entry=dict(method=method,profile=profile,target=tol,inputs=len(rows),finite_traced=len(steps),initial_actual_pass=initialpass,median_steps=statistics.median(steps),median_last_trial_fraction=statistics.median(last)if last else None,seconds=tt,seconds_after_actual_stable_pass=et,fraction_after_actual_stable_pass=et/tt if tt else None,kinds=kind,per_case=per);groups.append(entry);print({k:v for k,v in entry.items()if k not in ['per_case','kinds']});print('kinds',kind)
(P/'trace-analysis.json').write_text(json.dumps(dict(changed_vs_baseline=changed,groups=groups,note='Retrospective reference-informed pass is diagnostic, never operational stopping. Trace time not a latency benchmark.'),indent=2)+'\n');print('numeric changes',len(changed),changed[:5]);assert not changed
