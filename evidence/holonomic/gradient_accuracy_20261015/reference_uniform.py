from pathlib import Path
import json,csv,numpy as np,VBMicrolensing
out=Path(__file__).resolve().parent
cfg=json.loads((out/'config.json').read_text());p=cfg['params'];times=cfg['times'];rows=[]
for name,h0 in [('u0',1e-5),('s',1e-4),('q',1e-7)]:
 for factor in [1,.3,.1]:
  h=h0*factor;values=[]
  for sign in [1,-1]:
   z=p.copy();z[name]+=sign*h
   v=VBMicrolensing.VBMicrolensing();v.Tol=v.RelTol=1e-10;v.a1=0
   args=[np.log(z['s']),np.log(z['q']),z['u0'],z['alpha'],np.log(z['rho']),np.log(z['tE']),z['t0']]
   values.append(np.asarray(v.BinaryLightCurve(args,times)[0]))
  rows.extend(dict(t=t,param=name,h=h,vbm_fd=(values[0][i]-values[1][i])/(2*h)) for i,t in enumerate(times))
with (out/'reference_uniform.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
data=np.loadtxt(out/'inputs.tsv');data[:,-1]=0
np.savetxt(out/'inputs_uniform.tsv',data,fmt='%.17g',delimiter='\t')
