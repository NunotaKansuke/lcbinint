"""Bounded gradient diagnosis and compiled-API regression/timing report."""
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[3]
old=root/'evidence/holonomic/fused_gradient_20261014'
out=root/'evidence/holonomic/fused_gradient_followup'
def read(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def key(r):return tuple(r[k] for k in ('case','profile','dbin','epoch','mode'))
before={key(r):r for r in read(old/'full_fused.tsv')}
after=read(out/'full_api.tsv')
assert before.keys()=={key(r) for r in after}
fields=('mu','value_converged','stop','nodes','support')
changes={mode:sum(any(r[f]!=before[key(r)][f] for f in fields)
                 for r in after if r['mode']==mode) for mode in ('0','3')}
assert not any(changes.values()),changes
value={key(r)[:4]:r for r in after if r['mode']=='0'}
gradient={key(r)[:4]:r for r in after if r['mode']=='3'}
timing={}
for profile in ('all','uniform','linear'):
    keys=[k for k in value if profile=='all' or k[1]==profile]
    ratios=[float(gradient[k]['total_ms'])/float(value[k]['total_ms']) for k in keys]
    timing[profile]={
        'inputs':len(keys),'median_ratio':statistics.median(ratios),
        'sum_ratio':sum(float(gradient[k]['total_ms']) for k in keys)/sum(float(value[k]['total_ms']) for k in keys),
        'frontier_sum_ratio':sum(float(gradient[k]['frontier_ms']) for k in keys)/sum(float(value[k]['frontier_ms']) for k in keys),
        'inputs_over_1_3':sum(r>1.3 for r in ratios),
    }
fd={}
for r in read(old/'reference_fd.tsv'):
    fd.setdefault(tuple(r[k] for k in ('case','profile','dbin','epoch','param')),[]).append(r)
case=('155','uniform','0','7')
diagnosis={}
for param in (1,4):
    reference=fd[case+(str(param),)]
    medium,small=map(lambda r:float(r['fd']),reference[-2:])
    extrapolated=(4*small-medium)/3 # Diagnostic only; no new solver rule.
    values={'1e-4':float(before[case+('3',)][f'g{param}'])}
    for tol in ('1e-5','1e-6'):
        r=next(r for r in read(out/f'resolution_{tol}.tsv') if key(r)==case+('3',))
        values[tol]=float(r[f'g{param}'])
    diagnosis[str(param)]={'fd_steps':[float(r['step']) for r in reference],
        'fd_values':[float(r['fd']) for r in reference],
        'fd_plus_minus_converged':all(r['converged_plus']=='1' and r['converged_minus']=='1' for r in reference),
        'h_squared_extrapolation':extrapolated,'fused_by_value_tolerance':values}
report={'inputs':len(value),'primal_changes_vs_previous':changes,'timing':timing,
    'gradient_precision_checked':False,
    'max_scaled_gradient_difference_vs_previous':max(
        abs(float(r[f'g{j}'])-float(before[key(r)][f'g{j}']))/max(1,abs(float(before[key(r)][f'g{j}'])))
        for r in after if r['mode']=='3' for j in range(5)),
    'case_155_uniform_resolution_diagnosis':diagnosis}
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
paths=list((root/'benchmarks/holonomic/fused_gradient').glob('*'))+[
    root/'src/lcbinint/magnification/holonomic/fused_gradient.hpp',
    root/'src/lcbinint/magnification/holonomic/fused_gradient.cpp',
    root/'build-seeded-events/fused-gradient/api_bench',
    root/'build-seeded-events/fused-gradient/liblcbinint_fused_gradient.a',
    root/'build-seeded-events/fused-gradient/api-include/manifest.json',
    root/'evidence/holonomic/seeded_paper_qrho_20261014/input_snapshot.tsv',
    root/'build-seeded-events/seed_indexed_v30.o',
    root/'build-standard-current/liblcbinint_magnification.a',
    root/'build-standard-current/liblcbinint_lightcurve.a']
provenance={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    'command':'build-seeded-events/fused-gradient/api_bench evidence/holonomic/seeded_paper_qrho_20261014/input_snapshot.tsv evidence/holonomic/fused_gradient_followup/full_api.tsv 3 1',
    'build_command':'bash benchmarks/holonomic/fused_gradient/build.sh',
    'mu_atol':1e-8,'mu_rtol':1e-4,'repetitions':3,
    'timing':'fresh seed/index preparation included; caustic prebuild excluded; alternating value/API order; same seeds; independent workspaces',
    'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(report,indent=2))
