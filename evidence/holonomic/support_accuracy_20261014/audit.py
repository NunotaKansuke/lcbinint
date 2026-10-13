import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEY = ('case_id', 'profile', 'd_bin_index', 'epoch_index', 'target')

def load(name):
    with (ROOT / name).open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    result = {tuple(r[k] for k in KEY): r for r in rows}
    assert len(result) == len(rows) == 14432
    return result

def compare(before, after):
    a, b = load(before), load(after)
    assert a.keys() == b.keys()
    pairs = [(a[k], b[k]) for k in a]
    timing = {}
    for field in ('seed_ms', 'planning_ms', 'integration_ms', 'ms'):
        timing[field] = {
            'baseline_sum': sum(float(x[field]) for x, _ in pairs),
            'current_sum': sum(float(y[field]) for _, y in pairs),
            'sum_ratio': sum(float(y[field]) for _, y in pairs) / sum(float(x[field]) for x, _ in pairs),
            'median_paired_ratio': statistics.median(float(y[field]) / float(x[field]) for x, y in pairs),
        }
    return {
        'baseline_file': before, 'current_file': after,
        'rows': len(pairs),
        'baseline_status': dict(Counter(x['status'] for x, _ in pairs)),
        'current_status': dict(Counter(y['status'] for _, y in pairs)),
        'transitions': dict(Counter(x['status'] + ' -> ' + y['status'] for x, y in pairs)),
        'changed_fields': {field: sum(x[field] != y[field] for x, y in pairs)
                           for field in ('v2_mu', 'value_error', 'nodes', 'bands', 'coverage_certified')},
        'new_non_ok': sum(x['status'] == 'OK' and y['status'] != 'OK' for x, y in pairs),
        'reference_discrepancies_over_target': sum(abs(float(y['v2_mu']) / float(y['reference_1e-6']) - 1) > float(y['target']) for _, y in pairs),
        'remaining_support_geometries': sorted({(y['case_id'], y['d_bin_index'], y['epoch_index']) for _, y in pairs if y['status'] == 'SEED_SUPPORT_UNPROVEN'}),
        'timing': timing,
    }

report = {
    'source_support': compare('baseline_full.tsv', 'final_full.tsv'),
    'rejected_global_rescaling': compare('baseline_full.tsv', 'candidate_full.tsv'),
}
if (ROOT / 'baseline_standard.tsv').exists():
    report['matched_standard_timing'] = compare('baseline_standard.tsv', 'final_repeat.tsv')
(ROOT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
