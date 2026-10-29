import os,sys,json,hashlib,subprocess,platform,socket,shlex,shutil,importlib.metadata
from pathlib import Path
import numpy,mpmath,VBMicrolensing
P=Path(__file__).resolve().parent;R=Path.cwd().resolve();old=json.load(open(P.parent/'triple_event_precision_20261029/PROVENANCE.json'));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=set(old['source_hashes'])|{'CMakeLists.txt'}
for dependency in Path('build-triple-native/CMakeFiles').rglob('*.o.d'):
 for token in shlex.split(dependency.read_text().replace('\\\n',' ')):
  q=Path(token)
  if not q.is_absolute():q=R/q
  try:rel=q.resolve().relative_to(R)
  except ValueError:continue
  if q.is_file()and not str(rel).startswith(('build-','evidence/')):files.add(str(rel))
source_hashes={f:sha(f)for f in sorted(files)}
for f in files:
 q=P/'source-snapshots/accepted'/f;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,q)
# Timing baseline source: the correctness-fixed terminal-only intermediate.
point=Path('src/lcbinint/magnification/point_source_magnifier.cpp').read_text()
point=point.replace('''    if(distinct_positive_lenses &&
       ((expected_images>=4 && physical_count>expected_images) ||
        physical_count!=2*positive_count+2)) {''','''    if(distinct_positive_lenses && expected_images>=4 &&
       (physical_count>expected_images || physical_count!=2*positive_count+2)) {''').replace('''        // An independent count or the necessary physical index triggers
        // repair; neither selects a prefix of the requested size or licenses
        // dropping a distinct valid root.''','''        // The independent count triggers repair; it never selects a prefix
        // of the requested size or licenses dropping a distinct valid root.''')
overrides={'src/lcbinint/magnification/finite_source_magnifier.cpp':(P/'finite_source_magnifier.terminal-only.cpp').read_bytes(),'src/lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp':(P/'regular_support.pre-witness-cache.hpp').read_bytes(),'src/lcbinint/magnification/point_source_magnifier.cpp':point.encode()}
for f,data in overrides.items():q=P/'source-snapshots/terminal-only'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
wrappers={}
for f in sorted((P/'accepted-package').rglob('*.py')):
 rel=f.relative_to(P/'accepted-package');wrappers[str(rel)]=sha(f);q=P/'python-wrapper-snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,q)
tests=['tests/triple_algebraic/test_regular.cpp','tests/triple_algebraic/test_local_images.cpp','tests/regression/test_triple_caustic_progress.py']
for f in tests:q=P/'test-snapshots'/f;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,q)
probe=P.parent/'triple_stress_expanded_20261028/vbm-audit/probe';reference={}
for name in ['VBMicrolensingLibrary.cpp','VBMicrolensingLibrary.h','main.cpp']:
 f=probe.parent/name;reference[name]=sha(f);q=P/'vbm-reference-source'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,q)
vbm_native=next(Path(VBMicrolensing.__file__).parent.glob('*.so'));baseline0=P.parent/'triple_event_precision_20261029/polished-package/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so';accepted=P/'accepted-package/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so'
assert sha(accepted)==sha('build-triple-native/_lcbinint.cpython-310-x86_64-linux-gnu.so')=='8fc87192444d742e1dfa48bad43800b03b638f09f476d1f8638020bce36a6534'
model=next((line.split(':',1)[1].strip()for line in Path('/proc/cpuinfo').read_text().splitlines()if line.startswith('model name')),'unknown')
prov=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),host=socket.gethostname(),platform=platform.platform(),cpu_model=model,python=sys.version,numpy=numpy.__version__,mpmath=mpmath.__version__,vbm_version=importlib.metadata.version('VBMicrolensing'),vbm_native=str(vbm_native),vbm_sha256=sha(vbm_native),vbm_reference_source_hashes=reference,vbm_probe_sha256=sha(probe),compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],native_sha256=sha(accepted),baseline_committed_sha256=sha(baseline0),timing_terminal_only_sha256=sha(P/'repolish-package/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so'),source_hashes=source_hashes,source_closure_note='Prior 69-file manifest extended using native target dependency files to include Python bindings. Terminal-only baseline reconstructs three overrides atop this closure; old dirty binding object bytes cannot be independently reconstructed from the prior manifest alone.',wrapper_hashes=wrappers,test_hashes={f:sha(f)for f in tests},build_flags={p.parent.name:p.read_text()for p in [Path('build-triple-native/CMakeFiles/lcbinint_magnification.dir/flags.make'),Path('build-triple-native/CMakeFiles/lcbinint_python.dir/flags.make')]},resource_budget=dict(cpu=8,ram_gib=16,native_workers=4,reference_workers=3,threads_per_library=1,worker_address_space_gib=2),geometry_limits=dict(default_new_phase_boxes=256,continued_new_phase_boxes=65536,maximum_tree_nodes=131072,maximum_depth=24,winding_witness_balls=16),compiler_scratch=str(P/'compiler-tmp'),unchanged_prior_hashes=sum(source_hashes[f]==h for f,h in old['source_hashes'].items()),changed_prior_source_paths=[f for f,h in old['source_hashes'].items()if source_hashes[f]!=h])
(P/'PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n')
paths=['src/lcbinint/magnification/finite_source_magnifier.cpp','src/lcbinint/magnification/point_source_magnifier.cpp','src/lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp',*tests[:2]]
patch=subprocess.check_output(['git','diff','--',*paths]);extra=subprocess.run(['git','diff','--no-index','--','/dev/null',tests[2]],capture_output=True);assert extra.returncode==1
(P/'accepted-source.patch').write_bytes(patch+extra.stdout);print(json.dumps({k:prov[k]for k in ['base_commit','native_sha256','baseline_committed_sha256','timing_terminal_only_sha256','changed_prior_source_paths']}));print('project dependency files',len(source_hashes),'frozen Python wrappers',len(wrappers))
