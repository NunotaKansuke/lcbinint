import os,sys,json,subprocess,statistics
from pathlib import Path
P=Path(__file__).resolve().parent;rows=[]
for i in [21,940,1508]:
 for engine in ['native-after-value','native-before-value','native-before-value','native-after-value']:
  r=subprocess.run([sys.executable,str(P/'accepted_timing.py'),'--worker',str(i),engine],capture_output=True,text=True,timeout=90,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'));assert r.returncode==0,r.stderr
  z=json.loads(r.stdout);rows.append(z);(P/'repeat-mainstream-timing.json').write_text(json.dumps(rows,indent=2));print(i,engine,z['warm'],flush=True)
summary={str(i):{e:statistics.median([s for r in rows if r['case']==i and r['engine']==e for s in r['samples']])for e in ['native-before-value','native-after-value']}for i in [21,940,1508]};(P/'repeat-mainstream-summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
