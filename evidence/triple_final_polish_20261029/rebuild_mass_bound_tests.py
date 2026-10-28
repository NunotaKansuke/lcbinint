import os,subprocess,json,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent.parent;env=dict(os.environ,TMPDIR=str(P));state=dict(state='BUILDING',pid=os.getpid(),started=time.time());(P/'mass-bound-test-build-status.json').write_text(json.dumps(state))
with open(P/'mass-bound-test-build.log','w')as f:r=subprocess.run(['cmake','--build','build-triple-native','-j2'],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
state.update(build_returncode=r.returncode,state='TESTING'if r.returncode==0 else'BUILD_FAILED');(P/'mass-bound-test-build-status.json').write_text(json.dumps(state))
if r.returncode==0:
 with open(P/'mass-bound-ctest.log','w')as f:r=subprocess.run(['ctest','--test-dir','build-triple-native','--output-on-failure','-j1'],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 state.update(test_returncode=r.returncode,state='PASS'if r.returncode==0 else'TEST_FAILED',finished=time.time());(P/'mass-bound-test-build-status.json').write_text(json.dumps(state,indent=2))
