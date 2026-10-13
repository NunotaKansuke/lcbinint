"""Ordinary-build seed port and bounded matched accuracy/cost checkpoint."""
import csv,json,os,subprocess,shutil,statistics,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[3]; P=R/'tests/diagnostics/results/recal2026/normal-seed-index-20261014'; B=R/'build-adaptive-production'
def run(cmd):
 print('run',str(cmd)[:180],flush=True); subprocess.run(list(map(str,cmd)),check=True,cwd=R)
def compile_runner(source,name,lib,adaptive=False):
 cmd=['/usr/bin/c++','-O3','-DNDEBUG','-std=gnu++17','-fopenmp','-I'+str(R/'include'),'-I'+str(R/'src')]
 if adaptive:cmd+=['-DADAPTIVE','-DRELATIVE_SINGLE_PASS']
 cmd += [str(source),'-o',str(P/name),'-Wl,--start-group',str(B/'liblcbinint_lightcurve.a'),str(lib),'-Wl,--end-group','-Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib','/home/nunota/.miniconda3/envs/myenv/lib/libgsl.so','/home/nunota/.miniconda3/envs/myenv/lib/libgslcblas.so','-lm'];run(cmd)
 return cmd
if __name__=='__main__':
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 shutil.copy2(B/'liblcbinint_magnification.a',P/'before.a');run(['ar','r',P/'before.a',P/'finite_source_magnifier.cpp.o'])
 cmds=[]
 for mode,lib in [('before',P/'before.a'),('after',B/'liblcbinint_magnification.a')]:
  cmds.append(compile_runner(P/'seed_runner.cpp','seed-'+mode,lib))
  cmds.append(compile_runner(R/'tests/diagnostics/recal2026/correction_accuracy_cost_runner.cpp','kernel-'+mode,lib,True))
 inputs=R/'tests/diagnostics/results/recal2026/deferred-cache-production-20261013/input.tsv'
 for rep in range(3):
  for mode in (['before','after'] if rep%2==0 else ['after','before']):
   run([P/('seed-'+mode),inputs,P/f'seed-{mode}-{rep}.tsv'])
   run([P/('kernel-'+mode),inputs,P/f'kernel-{mode}-{rep}.tsv','adaptive'])
  print('batch completed',rep,flush=True)
 # Fixed outputs are an oracle-selected diagnostic, never a production selector.
 run([P/'kernel-after',inputs,P/'fixed.tsv','fixed'])
 (P/'provenance.json').write_text(json.dumps({'commands':cmds,'input_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((R/'src/lcbinint/magnification/finite_source_magnifier.cpp').read_bytes()).hexdigest(),'timing':'center roots/caustic construction excluded; seed warmed for kernel; no duplicated scale pass; native median3; pinned one CPU','installed':False},indent=2))
