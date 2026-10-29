import os,sys,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'repeat-heavy-timing';D.mkdir(exist_ok=True);rows=[]
for k,mode in enumerate(['after','before','before','after']):
 engine='native-'+mode+'-gradient';f=D/f'1508-{k}-{mode}.json';a=time.time();r=subprocess.run([sys.executable,str(P/'polished_timing.py'),'--worker','1508',engine],text=True,capture_output=True,timeout=75,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'));z=json.loads(r.stdout)if r.returncode==0 else dict(returncode=r.returncode,stderr=r.stderr);z['block']=k;z['started']=a;z['finished']=time.time();f.write_text(json.dumps(z,indent=2));rows.append(z);print(k,mode,z.get('warm',z.get('returncode')),flush=True)
(P/'repeat-heavy-timing.json').write_text(json.dumps(rows,indent=2))
