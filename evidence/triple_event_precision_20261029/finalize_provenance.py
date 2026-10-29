import json,hashlib,subprocess,sys,socket,importlib.metadata
from pathlib import Path
import numpy,mpmath,VBMicrolensing
P=Path(__file__).resolve().parent
sha=lambda data:hashlib.sha256(data).hexdigest()
old=json.load(open(P.parent/'triple_residual_polish_20261029/deferred-provenance.json'))
new_header='src/lcbinint/magnification/algebraic_boundary/triple/exterior_caustic_bound.hpp';files=sorted(set(old['source_hashes'])|{new_header})
source_hashes={f:sha(Path(f).read_bytes())for f in files}
for f in files:
 d=P/'source-snapshots/accepted'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(Path(f).read_bytes())
# The controlled timing baseline is the correctness-fixed intermediate DSO,
# not the original commit where case 514 refused to produce a value.
baseline={}
for f in old['source_hashes']:
 if f=='src/lcbinint/magnification/finite_source_magnifier.cpp':data=(P/'finite_source_magnifier.pre-pole-frame.cpp').read_bytes()
 elif f=='tests/triple_algebraic/CMakeLists.txt':data=subprocess.check_output(['git','show','43e1f083:'+f])
 else:data=Path(f).read_bytes()
 baseline[f]=sha(data);d=P/'source-snapshots/pre-analytic'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(data)
tests=['tests/triple_algebraic/test_events.cpp','tests/triple_algebraic/test_regular.cpp','tests/triple_algebraic/test_exterior_clearance.cpp','tests/triple_algebraic/CMakeLists.txt','tests/regression/test_triple_event_precision.py']
for f in tests:
 d=P/'test-snapshots'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(Path(f).read_bytes())
vbm_native=next(Path(VBMicrolensing.__file__).parent.glob('*.so'))
z=dict(base_commit='43e1f083ac6b682b46cd01a146c62397102d14da',branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),host=socket.gethostname(),accepted_variant='polished',native_sha256=sha((P/'polished-package/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so').read_bytes()),timing_baseline_sha256=sha((P/'adopted-package/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so').read_bytes()),source_hashes=source_hashes,timing_baseline_source_hashes=baseline,test_hashes={f:sha(Path(f).read_bytes())for f in tests},python=sys.version,numpy=numpy.__version__,mpmath=mpmath.__version__,vbm_version=importlib.metadata.version('VBMicrolensing'),vbm_native=str(vbm_native),vbm_sha256=sha(vbm_native.read_bytes()),compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],cpu_limit=8,ram_gib=16,worker_threads=1,native_workers=6,worker_address_space_gib=2,build_workers=2,scratch=str(P/'compiler-tmp'),native_flags=Path('build-triple-native/CMakeFiles/lcbinint_magnification.dir/flags.make').read_text())
assert z['native_sha256']==sha(Path('build-triple-native/_lcbinint.cpython-310-x86_64-linux-gnu.so').read_bytes())
(P/'PROVENANCE.json').write_text(json.dumps(z,indent=2));(P/'accepted-source.patch').write_bytes(subprocess.check_output(['git','diff','--','src/lcbinint/magnification/finite_source_magnifier.cpp','src/lcbinint/magnification/algebraic_boundary/triple/radial_events.hpp','src/lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp',*tests]));print({k:z[k]for k in ['branch','native_sha256','timing_baseline_sha256','compiler','vbm_version','vbm_sha256']})
