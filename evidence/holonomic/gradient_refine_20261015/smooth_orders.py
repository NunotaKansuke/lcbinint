from pathlib import Path
import os,subprocess
root=Path.cwd();out=root/'evidence/holonomic/gradient_refine_20261015';build=root/'build-algebraic-boundary/gradient-probe'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
env={**os.environ,'TMPDIR':str(build/'scratch'),'LCBININT_PROBE_MU_TOL':'1e-3','LD_LIBRARY_PATH':'/rogue1_8/nunota/local/gsl/lib:'+os.environ.get('LD_LIBRARY_PATH','')}
s=(out/'baseline_gaussian.hpp').read_text().replace(' case 64:', ' case 32:return fixed_gaussian_rule<32>();\n case 96:return fixed_gaussian_rule<96>();\n case 256:return fixed_gaussian_rule<256>();\n case 64:').replace('gaussian_rule(hot?64:', 'gaussian_rule(hot?(std::getenv("LCBININT_TRIAL_ORDER")?std::atoi(std::getenv("LCBININT_TRIAL_ORDER")):64):')
s=s.replace(' case 64:', ' case 7:return fixed_gaussian_rule<7>();\n case 10:return fixed_gaussian_rule<10>();\n case 64:').replace('gaussian_rule(hot?', 'gaussian_rule(!hot&&panel.depth==0?(std::getenv("LCBININT_SMOOTH_ORDER")?std::atoi(std::getenv("LCBININT_SMOOTH_ORDER")):14):hot?')
s=s.replace('if(panel.depth==0&&!hot){','if(panel.depth==0&&!hot&&!(geometry_experiment::active&&!geometry_experiment::active->empty())){')
(build/'geometry-include/fused_gaussian.hpp').write_text(s)
cmd=['g++','-std=gnu++17','-O2','-DNDEBUG','-fext-numeric-literals','-DONLY_VALUE_MESH','-I'+str(build/'geometry-include'),'-Isrc','-Iinclude','-Ithird_party/skowron_gould','-I/rogue1_8/nunota/local/gsl/include',str(root/'evidence/holonomic/gradient_screen_20261015/probe.cpp'),'-Wl,--start-group',str(root/'build-algebraic-boundary/python-api/liblcbinint_magnification.a'),str(root/'build-algebraic-boundary/python-api/liblcbinint_lightcurve.a'),'-Wl,--end-group','-L/rogue1_8/nunota/local/gsl/lib','-lgsl','-lgslcblas','-lquadmath','-o',str(build/'probe_refine')]
subprocess.run(cmd,check=True,env=env)
for n in [7,10,14,30]:
 with (out/f'smooth{n}.log').open('w') as log:subprocess.run([str(build/'probe_refine'),str(root/'evidence/holonomic/gradient_accuracy_20261015/inputs.tsv'),str(out/f'smooth{n}.tsv')],env={**env,'LCBININT_TRIAL_ORDER':'64','LCBININT_SMOOTH_ORDER':str(n)},stderr=log,check=True)
