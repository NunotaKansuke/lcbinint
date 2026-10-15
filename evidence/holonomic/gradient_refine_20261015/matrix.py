from pathlib import Path
import os,subprocess,json,math,csv
import numpy as np
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});cases=[]
for s,q in [(.6,1e-4),(.6,1e-3),(1.2,.01),(1.,1.),(2.,.3)]:
 for rho in [.0005,.05]:
  for u0 in [-1e-5,.2]:
   for t in [-.7,.1,.78]:
    cases.append(dict(index=len(cases),s=s,q_input=q,q_solver=max(q,1/q),rho=rho,u0=u0,t=t,x=t/10*math.cos(.5)-u0*math.sin(.5),y=t/10*math.sin(.5)+u0*math.cos(.5)))
with (out/'matrix_inputs.tsv').open('w') as f:
 for c in cases:f.write('\t'.join(map(str,[c['index'],c['x'],c['y'],c['rho'],c['q_solver'],c['s'],.5]))+'\n')
(out/'matrix_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
for variant,mode in [('on','1'),('off','0')]:
 env={**os.environ,'LCBININT_PROBE_MU_TOL':'1e-3','LCBININT_GEOMETRY_SCREEN':mode,'LD_LIBRARY_PATH':'/rogue1_8/nunota/local/gsl/lib:'+os.environ.get('LD_LIBRARY_PATH','')}
 with (out/f'matrix_{variant}.log').open('w') as log:
  subprocess.run([str(root/'build-algebraic-boundary/gradient-probe/probe_refine_final'),str(out/'matrix_inputs.tsv'),str(out/f'matrix_{variant}.tsv')],cwd=root,env=env,stderr=log,check=True)
a=list(csv.DictReader((out/'matrix_on.tsv').open(),delimiter='\t'));b=list(csv.DictReader((out/'matrix_off.tsv').open(),delimiter='\t'));different=[]
for x,y in zip(a,b):
 for col in ['mu','g']:
  p,q=float(x[col]),float(y[col])
  if not(p==q or math.isnan(p) and math.isnan(q)):different.append(dict(index=int(float(x['t'])),component=int(x['j']),field=col,on=p,off=q))
report=dict(epochs=len(cases),gradient_components=len(a),finite_on=sum(math.isfinite(float(r['g'])) for r in a),finite_off=sum(math.isfinite(float(r['g'])) for r in b),value_converged_on=sum(r['value_converged']=='1' for r in a if r['j']=='0'),value_converged_off=sum(r['value_converged']=='1' for r in b if r['j']=='0'),differences=different)
(out/'matrix_comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
