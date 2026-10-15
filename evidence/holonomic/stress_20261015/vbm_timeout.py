import os,sys,json,time,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
if len(sys.argv)>1:
 import VBMicrolensing as VB,numpy as np
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});c=json.loads((p/'case_054.json').read_text());x=c['params'];v=VB.VBMicrolensing();v.Tol=v.RelTol=1e-3;v.a1=c['limb'];v.a2=0;args=[np.log(x['s']),np.log(x['q']),x['u0'],x['alpha'],np.log(x['rho']),np.log(x['tE']),x['t0']];v.BinaryLightCurve(args,np.linspace(-.3,.3,33).tolist())
else:
 t=time.perf_counter()
 try:subprocess.run([sys.executable,str(Path(__file__)),'worker'],check=True,timeout=5);r=dict(status='completed')
 except subprocess.TimeoutExpired:r=dict(status='timeout',limit_seconds=5)
 r['elapsed_seconds']=time.perf_counter()-t;(p/'vbm_timeout.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
