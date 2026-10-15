import json,subprocess,sys,time
from pathlib import Path
R=Path('/rogue1_8/nunota/lcbinint');P=Path(__file__).resolve().parent;D=R/'tests/diagnostics/recal2026'
commands=[
 [P/'audit_full.py','--method','cartesian','--initial-bins','16','--output',P/'final-cartesian-full'],
 [P/'audit_full.py','--method','polar','--initial-bins','2','--output',P/'final-polar-full'],
 [D/'cartesian_internal_contract_audit.py','--output',P/'final-cartesian-contracts.json'],
 [D/'polar_internal_contract_audit.py','--output',P/'final-polar-contracts.json'],
 [P/'heldout_cartesian.py','--name','final-heldout-cartesian','--cohort',R/'tests/diagnostics/results/recal2026/cartesian-final-redesign-20261015/heldout-inputs.json'],
 [P/'independent_polar.py','--name','final-independent-polar']]
results=[]
for index,cmd in enumerate(commands):
 start=time.monotonic();log=P/f'final-validation-{index}.log'
 with log.open('w')as f:ret=subprocess.run([sys.executable,*map(str,cmd)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(command=list(map(str,cmd)),returncode=ret.returncode,seconds=time.monotonic()-start,log=str(log)));(P/'final-validation-processes.json').write_text(json.dumps(results,indent=2)+'\n');print(index,ret.returncode,round(results[-1]['seconds'],2),flush=True)
