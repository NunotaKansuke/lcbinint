"""Summarize stage timings and preserve reference-unresolved comparisons."""
import csv
import json
import statistics
import sys
from pathlib import Path

folder = Path(sys.argv[1])
def read(name):
    return list(csv.DictReader((folder / name).open(), delimiter=' '))

stage = read('rootpair_final.tsv')
value = read('adaptive_final.tsv')
a = [float(row['trace_us']) for row in stage]
b = [float(row['d14_us']) for row in stage]
matched = [r for r in value if r['d14_converged'] == '1']
unresolved = [r for r in value if r['d14_converged'] != '1']
regressions = [r for r in matched if r['seeded_converged'] != '1'
               or float(r['relative_difference']) > float(r['tol'])]
def percentile(xs, p):
    xs = sorted(xs)
    x = (len(xs)-1)*p
    lo = int(x)
    hi = min(lo+1, len(xs)-1)
    return xs[lo] + (xs[hi]-xs[lo])*(x-lo)

result = {
    'timing_boundary': 'events plus cell plan; supplied seeds; no integration or audit',
    'geometry_count': len(stage),
    'topology_failures': sum(r['status'] != '1' for r in stage),
    'seeded_median_us': statistics.median(a),
    'd14_median_us': statistics.median(b),
    'seeded_p90_us': percentile(a, .9),
    'd14_p90_us': percentile(b, .9),
    'sum_time_ratio': sum(a)/sum(b),
    'seeded_faster_cases': sum(x < y for x,y in zip(a,b)),
    'adaptive_comparisons': len(value),
    'reference_converged': len(matched),
    'regressions_vs_converged_reference': len(regressions),
    'max_relative_difference_over_tolerance': max(
        float(r['relative_difference'])/float(r['tol']) for r in matched),
    'reference_unresolved': len(unresolved),
    'both_unresolved': sum(r['seeded_converged'] != '1' for r in unresolved),
    'seed_cold_snapshot_median_us': statistics.median(float(r['seed_cold_ms'])*1000 for r in stage),
    'seed_cold_snapshot_in_stage_timing': False,
    'global_coverage_certified': False,
    'production_default_changed': False,
}
(folder/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
