from pathlib import Path
import os,subprocess,math
import numpy as np
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
with (out/'ordinary_inputs.tsv').open('w') as f:
 for t in np.linspace(-10,10,401):
  x=t/10*math.cos(.5)-.2*math.sin(.5);y=t/10*math.sin(.5)+.2*math.cos(.5);f.write(f"{t}\t{x}\t{y}\t.001\t100\t1.2\t.5\n")
for case in ['current','ordinary','historical']:
 for variant,binary in [('original','probe_selected'),('screen','probe_screen')]:
  env={**os.environ,'LCBININT_PROBE_MU_TOL':'1e-3','LD_LIBRARY_PATH':'/rogue1_8/nunota/local/gsl/lib:'+os.environ.get('LD_LIBRARY_PATH','')}
  with (out/f'profile_{case}_{variant}.log').open('w') as log:
   subprocess.run([str(root/'build-algebraic-boundary/gradient-probe'/binary),str(out/f'{case}_inputs.tsv'),str(out/f'profile_{case}_{variant}.tsv')],cwd=root,env=env,stderr=log,check=True)
