"""Summarize paired adaptive runs; failures remain in attempt-time statistics."""
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'evidence/holonomic/adaptive_radial_20260911')

def rows(name):
    with (root/name).open() as f:
        return list(csv.DictReader(f))

def percentile(values, p):
    a = sorted(values)
    if not a:
        return None
    x = (len(a)-1)*p
    lo = int(x)
    return a[lo] + (a[min(lo+1, len(a)-1)]-a[lo])*(x-lo)

def distribution(values):
    return {key: percentile(values, p) for key, p in [('p50', .5), ('p90', .9), ('p95', .95), ('p99', .99), ('max', 1)]}

refs = {}
ref_summary = {}
for warm, name in [(0, 'reference.csv'), (1, 'reference_warm.csv')]:
    data = rows(name)
    for r in data:
        refs[(warm,r['name'],float(r['u']),float(r['rtol']))] = r
    ref_summary['warm' if warm else 'cold'] = {
        'rows': len(data), 'stops': dict(Counter(r['stop'] for r in data)),
        'reference_unusable': sum(r['reference_usable'] != '1' for r in data),
        'observed_violations': sum(r['violation'] == '1' for r in data),
        'note': '128/256 radial + 256/512 angular self-difference estimates reference uncertainty; not a verified enclosure.'}

# Repeats reduce timing noise without counting one physical case multiple times.
by_case = defaultdict(list)
for r in rows('paired.csv'):
    key = (r['name'], float(r['u']), int(r['jac']), int(r['warm']), float(r['rtol']), int(r['cache']))
    by_case[key].append(r)
groups = defaultdict(list)
for key, repeats in by_case.items():
    name,u,jac,warm,tol,cache = key
    r = dict(repeats[0])
    for column in ['whole_ms','fixed64_ms','topology_ms','setup_ms','physics_ms','estimator_ms','scheduler_ms']:
        r[column] = statistics.median(float(x[column]) for x in repeats)
    r['repeat_status_consistent'] = len({x['stop'] for x in repeats}) == 1
    ref = refs[(warm,name,u,tol)]
    budget = max(1e-8,tol*abs(float(r['mu'])))
    usable = ref['reference_usable'] == '1'
    uncertainty = float(ref['reference_uncertainty'])
    truth = float(ref['reference256'])
    r['value_accuracy_qualified_pair'] = (not jac and usable and r['stop'] == 'Converged'
        and int(r['fixed64_status']) == 0
        and abs(float(r['mu'])-truth)+uncertainty <= budget
        and abs(float(r['fixed64_mu'])-truth)+uncertainty <= max(1e-8,tol*abs(float(r['fixed64_mu']))))
    groups[(u,jac,warm,tol,cache)].append(r)

summary = []
for key, data in sorted(groups.items()):
    u,jac,warm,tol,cache=key
    qualified = [r for r in data if r['value_accuracy_qualified_pair']]
    summary.append({
        'u':u, 'jac':jac, 'warm':warm, 'rtol':tol, 'cache':cache,
        'cases':len(data), 'stops':dict(Counter(r['stop'] for r in data)),
        'repeat_status_consistent':all(r['repeat_status_consistent'] for r in data),
        'whole_attempt_ms':distribution([r['whole_ms'] for r in data]),
        'fixed64_attempt_ms':distribution([r['fixed64_ms'] for r in data]),
        'stage_medians_ms':{c:statistics.median(r[c] for r in data) for c in ['topology_ms','setup_ms','physics_ms','estimator_ms','scheduler_ms']},
        'median_nodes':statistics.median(int(r['nodes']) for r in data),
        'median_evaluations':statistics.median(int(r['evaluations']) for r in data),
        'value_accuracy_qualified_pairs':len(qualified),
        'qualified_speedup_fixed_over_adaptive':distribution([r['fixed64_ms']/r['whole_ms'] for r in qualified]),
    })

control_cases = defaultdict(list)
for r in rows('controls.csv'):
    control_cases[(r['name'],float(r['u']),int(r['jac']),int(r['warm']),r['mode'])].append(r)
control_groups = defaultdict(list)
for (name,u,jac,warm,mode), data in control_cases.items():
    control_groups[(u,jac,warm,mode)].append({'ms':statistics.median(float(r['whole_ms']) for r in data), 'stop':data[0]['stop'], 'nodes':int(data[0]['evaluations'])})
controls = [{'u':key[0], 'jac':key[1], 'warm':key[2], 'mode':key[3], 'cases':len(data),
             'whole_attempt_ms':distribution([r['ms'] for r in data]),
             'stops':dict(Counter(r['stop'] for r in data)),
             'median_evaluations':statistics.median(r['nodes'] for r in data)}
            for key,data in sorted(control_groups.items())]
jac_ref = rows('jacobian_reference.csv')
jac_summary = {name:{'rows':len(data), 'finite_reference_rows':sum(math.isfinite(float(r['finite_difference'])) for r in data),
                    'stops':dict(Counter(r['stop'] for r in data))}
               for name in sorted({r['name'] for r in jac_ref})
               for data in [[r for r in jac_ref if r['name']==name]]}

out = {'reference':ref_summary,'paired':summary,'controls':controls,'jacobian_reference':jac_summary,
    'precision':'binary64 nodes and observables, qf local event correction',
    'error_budget':'Eabs <= max(Tol, RelTol * abs(Q)) independently for every requested output',
    'limitations':['Estimated, not Bounded. require_bound returns BoundUnavailable.',
                   'Jacobian whole-case times include failures and do not establish same-accuracy speedup.',
                   'Fixed64 is not an accuracy reference.',
                   'initial/unmatched and one-repeat ladder files are diagnostics, not final timing evidence.']}
(root/'summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(ref_summary,indent=2))
for r in summary:
    if r['cache']==1 and r['rtol'] in [1e-3,1e-6]:
        print(r['u'],r['jac'],r['warm'],r['rtol'],r['stops'],r['whole_attempt_ms'], 'qualified',r['value_accuracy_qualified_pairs'])
