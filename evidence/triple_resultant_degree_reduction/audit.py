import json,pathlib,subprocess,os
P=pathlib.Path(__file__).resolve().parent
os.sched_setaffinity(0,{0});out=[]
for f in sorted((P/'conditions').glob('*-algebraic-*.json')):
 c=json.loads(f.read_text());g=c['case']
 for j,source in enumerate([g['source']]+c['sources'][9:]):
  inp=' '.join(map(str,source+[g['rho']]+[v for z,m in zip(g['lenses'],g['masses']) for v in (*z,m)]))
  proc=subprocess.run([str(P/'point-final'),'algebraic','0',str(c['limb']),str(c['rtol']),'5'],input=inp,text=True,capture_output=True,check=True)
  ref=json.loads((P/'references'/f"{c['name']}-u{c['limb']}-i{j}.json").read_text())['result']
  for line in proc.stdout.splitlines():
   r=json.loads(line);err=abs(r['mu']-ref['mu']);row=dict(case=c['name'],limb=c['limb'],rtol=c['rtol'],index=j,value=r,relative_error=err/abs(ref['mu']));out.append(row)
   assert r['converged'] and err<=c['rtol']*abs(ref['mu']) and err<=r['error_estimate']+ref['error_estimate'],row
  (P/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
 print(c['name'],c['limb'],c['rtol'],'PASS',flush=True)
