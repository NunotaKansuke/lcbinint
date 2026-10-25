from pathlib import Path
import json,os,subprocess
P=Path(__file__).resolve().parent;os.sched_setaffinity(0,{0});audit=json.loads((P/'audit.json').read_text());rows=[]
for f in sorted((P/'conditions').glob('*-algebraic-*.json')):
 c=json.loads(f.read_text());g=c['case'];data=' '.join(map(str,g['source']+[g['rho']]+[v for z,m in zip(g['lenses'],g['masses']) for v in (*z,m)]))
 proc=subprocess.run([str(P/'point-before'),'algebraic','0',str(c['limb']),str(c['rtol']),'5'],input=data,text=True,capture_output=True,check=True)
 old=[json.loads(s) for s in proc.stdout.splitlines()];new=[r['value'] for r in audit if (r['case'],r['limb'],r['rtol'],r['index'])==(c['name'],c['limb'],c['rtol'],0)]
 assert len(old)==len(new)==5
 for a,b in zip(old,new):
  keys=['resultant_nodes','resultant_solves','resultant_events','events','boxes','nodes','quad','event_quad_refinements','event_double_refinements','component_domains']
  rows.append(dict(case=c['name'],limb=c['limb'],rtol=c['rtol'],repeat=a['repeat'],before={k:a[k] for k in keys},after={k:b[k] for k in keys}))
(P/'work-counters.json').write_text(json.dumps(dict(status='PASS',note='Work counts only; executable I/O timings are not used',rows=rows),indent=2)+'\n')
