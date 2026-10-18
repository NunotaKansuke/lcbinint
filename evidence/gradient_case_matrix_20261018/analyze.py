from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
for tol in ['1e-3','1e-4']:
 p=P/('tol_'+tol)
 if not (p/'summary.json').exists():continue
 s=json.loads((p/'summary.json').read_text());rows=[]
 for b,c in zip(s['before'],s['current']):
  assert b['name']==c['name'];v=c['measurements']['False'];g=c['measurements']['True'];old=b['measurements']['True']['ms'];routes=g['method_counts']
  assert routes==b['measurements']['True']['method_counts']==v['method_counts']
  assert v['finite'] and g['finite']
  rows.append(dict(name=c['name'],n=c['n'],value_ms=v['ms'],gradient_ms=g['ms'],before_gradient_ms=old,change_percent=100*(g['ms']/old-1),ratio=c['ratio'],routes=routes,converged=g['converged'],gradient_valid=g['gradient_valid'],max_value_fused_difference=c['max_value_fused_difference']))
 (p/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
 if tol=='1e-4':
  for r in rows:print(r['name'],r['n'],round(r['value_ms'],3),round(r['gradient_ms'],3),round(r['ratio'],2),round(r['change_percent'],1),r['routes'],r['converged'],r['gradient_valid'])
