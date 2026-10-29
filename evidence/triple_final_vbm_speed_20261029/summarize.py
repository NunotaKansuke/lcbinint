"""Speedup is VBM-value time/native time; admit only accuracy-qualified pairs."""
import json
from pathlib import Path
import statistics

D = Path(__file__).resolve().parent
refs = json.loads((D / 'references.json').read_text())
reference = {}
for r in refs:
    key = r['case'], r['index']
    if key in reference:
        assert reference[key] == r['references']
    reference[key] = r['references']

cases = sorted(int(p.stem) for p in (D/'inputs').glob('*.json'))
rows = []
methods = ['SinglePoly', 'MultiPoly', 'NoPoly']
for i in cases:
    c = json.loads((D/'inputs'/f'{i:04d}.json').read_text())
    native = {e:json.loads((D/'rows'/f'{i:04d}-native-{e}.json').read_text()) for e in ['value','gradient']}
    assert all('warm' in r for r in native.values()), ('native failure', i)
    covered = [j for j in range(len(c['epochs'])) if (i,j) in reference]
    full = len(covered) == len(c['epochs'])
    def error(r):
        return max((abs(o['mu'][j]/v-1)/c['rtol'] for o in r['outputs'] if o['phase']=='warm'
                    for j in covered for v in reference[i,j]), default=None)
    nerror = max((error(r) for r in native.values()), default=None) if covered else None
    for method in methods:
        b = json.loads((D/'rows'/f'{i:04d}-{method}.json').read_text())
        r = dict(case=i, method=method, epochs=len(c['epochs']), limb=c['limb'], rtol=c['rtol'],
                 reference_epochs=covered, full_reference_coverage=full,
                 native_value_ms=1000*native['value']['warm'], native_gradient_ms=1000*native['gradient']['warm'],
                 native_error_over_requested=nerror)
        if 'warm' not in b:
            r.update(classification='failed', returncode=b.get('returncode'), error=b.get('error'), stderr=b.get('stderr'))
        else:
            berr = error(b)
            agreement = max(abs(o['mu'][j]/native['value']['outputs'][-1]['mu'][j]-1)/c['rtol']
                            for o in b['outputs'] if o['phase']=='warm' for j in range(len(c['epochs'])))
            if covered and (nerror > 1 or berr > 1):
                classification = 'reference_error'
            elif not full:
                classification = 'incomplete_reference'
            else:
                classification = 'qualified'
            r.update(classification=classification, vbm_ms=1000*b['warm'],
                     vbm_error_over_requested=berr, agreement_over_requested=agreement,
                     value_speedup=b['warm']/native['value']['warm'],
                     gradient_vs_vbm_value_ratio=b['warm']/native['gradient']['warm'])
        rows.append(r)

def stats(selected):
    if not selected:
        return dict(count=0)
    v=[r['value_speedup'] for r in selected]
    return dict(count=len(v), median_value_speedup=statistics.median(v), min_value_speedup=min(v), max_value_speedup=max(v),
                native_faster=sum(x>1 for x in v),
                total_value_time_speedup=sum(r['vbm_ms']for r in selected)/sum(r['native_value_ms']for r in selected),
                median_gradient_vs_vbm_value_ratio=statistics.median(r['gradient_vs_vbm_value_ratio']for r in selected),
                cases=[r['case']for r in selected])

common = set.intersection(*[{r['case'] for r in rows if r['method']==m and r['classification']=='qualified'} for m in methods])
summary = dict(definition='VBM value warm CPU median / native value warm CPU median; greater than one means native is faster',
               cases=len(cases), fully_referenced_cases=sum(all((i,j)in reference for j in range(len(json.loads((D/'inputs'/f'{i:04d}.json').read_text())['epochs'])))for i in cases),
               methods={m:dict(stats([r for r in rows if r['method']==m and r['classification']=='qualified']),
                   exclusions={k:sum(r['method']==m and r['classification']==k for r in rows)for k in ['failed','reference_error','incomplete_reference']})for m in methods},
               agreement_only_methods={m:stats([r for r in rows if r['method']==m and r.get('agreement_over_requested', float('inf'))<=1 and r['classification']!='reference_error'])for m in methods},
               common_qualified_cases=sorted(common),
               common_methods={m:stats([r for r in rows if r['method']==m and r['case']in common])for m in methods})
(D/'comparisons.json').write_text(json.dumps(rows,indent=2))
(D/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
