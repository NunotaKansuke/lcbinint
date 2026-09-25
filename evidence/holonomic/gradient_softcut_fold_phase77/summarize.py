#!/usr/bin/env python3
import csv
import json
import sys

base_path, cand_path, base_focus_path, cand_focus_path, grad_ref_path, mu_ref_path = sys.argv[1:7]

def table(path, delimiter='\t'):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))

def indexed(rows, fields):
    return {tuple(r[k] for k in fields): r for r in rows}

base_rows, cand_rows = table(base_path), table(cand_path)
base = indexed(base_rows, ('row', 'u', 'rtol', 'policy'))
cand = indexed(cand_rows, ('row', 'u', 'rtol', 'policy'))
if set(base) != set(cand) or len(base) != 440:
    raise SystemExit(f'paired corpus mismatch: baseline={len(base)} candidate={len(cand)}')

out = {'input_rows': 110, 'paired_rows': len(base), 'groups': {}, 'pairwise': {},
       'affected_rows': {}, 'targeted_rand035': {}}
for tol in ('0.001', '0.0001'):
    for policy in ('None', 'ValueFirst'):
        keys = [k for k in base if k[2] == tol and k[3] == policy]
        group = {}
        for arm, rows in (('baseline', base), ('candidate', cand)):
            group[arm] = {
                'value_converged': sum(int(rows[k]['value_ok']) for k in keys),
                'rows': len(keys),
                'stop_counts': {stop: sum(rows[k]['value_stop'] == stop for k in keys)
                                for stop in sorted({rows[k]['value_stop'] for k in keys})},
                'node_sum': sum(int(rows[k]['nodes']) for k in keys),
            }
        out['groups'][f'{tol}/{policy}'] = group

status_changes = [k for k in base if any(base[k][field] != cand[k][field]
                  for field in ('value_ok', 'value_stop', 'status'))]
mu_changes = [k for k in base if base[k]['mu'] != cand[k]['mu']]
non_target_node_changes = [k for k in base if k[0] not in ('99', '100') and
                           base[k]['nodes'] != cand[k]['nodes']]
value_policy_mismatches = []
for rows, arm in ((base, 'baseline'), (cand, 'candidate')):
    by = {(k[0], k[1], k[2]): r for k, r in rows.items() if k[3] == 'None'}
    vf = {(k[0], k[1], k[2]): r for k, r in rows.items() if k[3] == 'ValueFirst'}
    mismatch_keys = [k for k in by if by[k]['mu'] != vf[k]['mu']]
    value_policy_mismatches += [(arm, k) for k in mismatch_keys]
    out.setdefault('value_only_valuefirst_parity', {})[arm] = {
        'bitwise_mu_mismatches': len(mismatch_keys),
        'ulp_or_roundoff_only': all(abs(float(by[k]['mu'])-float(vf[k]['mu'])) <=
                                    2e-14*max(1.0, abs(float(by[k]['mu'])))
                                    for k in mismatch_keys),
    }
quality_downgrades = []
quality_upgrades = []
for k in base:
    for j in range(5):
        old, new = base[k][f'grad_quality{j}'], cand[k][f'grad_quality{j}']
        if old == 'ToleranceMet' and new != 'ToleranceMet':
            quality_downgrades.append((k, j, old, new))
        if old != 'ToleranceMet' and new == 'ToleranceMet':
            quality_upgrades.append((k, j, old, new))
out['pairwise'] = {
    'value_or_status_changes': len(status_changes),
    'mu_bitwise_mismatches': len(mu_changes),
    'mu_mismatch_rows': sorted({k[0] for k in mu_changes}, key=int),
    'value_only_vs_valuefirst_mu_mismatches': len(value_policy_mismatches),
    'candidate_merges': sum(int(r['softcut_fold_merges']) for r in cand_rows),
    'baseline_node_sum': sum(int(r['nodes']) for r in base_rows),
    'candidate_node_sum': sum(int(r['nodes']) for r in cand_rows),
    'node_change_percent': 100*(sum(int(r['nodes']) for r in cand_rows) /
                                 sum(int(r['nodes']) for r in base_rows) - 1),
    'non_target_node_mismatches': len(non_target_node_changes),
    'gradient_quality_downgrades': len(quality_downgrades),
    'gradient_quality_upgrades': len(quality_upgrades),
}
for row in ('99', '100'):
    keys = [k for k in base if k[0] == row]
    out['affected_rows'][row] = {
        'name': cand[keys[0]]['name'], 'u': cand[keys[0]]['u'],
        'baseline_nodes': sum(int(base[k]['nodes']) for k in keys),
        'candidate_nodes': sum(int(cand[k]['nodes']) for k in keys),
        'candidate_merge_count': sum(int(cand[k]['softcut_fold_merges']) for k in keys),
    }

grad_refs = {(r['u'], r['parameter']): r for r in table(grad_ref_path)}
mu_refs = {r['u']: r for r in table(mu_ref_path)}
base_focus = indexed(table(base_focus_path), ('u', 'rtol', 'policy'))
cand_focus = indexed(table(cand_focus_path), ('u', 'rtol', 'policy'))
for key, r in cand_focus.items():
    if r['policy'] != 'ValueFirst':
        continue
    u, tol, _ = key
    rx, ry = grad_refs[(u, 'X')], grad_refs[(u, 'Y')]
    rm = float(mu_refs[u]['mu_reference_order256'])
    b = base_focus[key]
    out['targeted_rand035'][f'u={u},rtol={tol}'] = {
        'baseline_nodes': int(b['nodes']), 'candidate_nodes': int(r['nodes']),
        'baseline_abs_mu_error': abs(float(b['mu']) - rm),
        'candidate_abs_mu_error': abs(float(r['mu']) - rm),
        'baseline_grad_xy_abs_error': [abs(float(b['grad0']) - float(rx['gradient_reference'])),
                                       abs(float(b['grad1']) - float(ry['gradient_reference']))],
        'candidate_grad_xy_abs_error': [abs(float(r['grad0']) - float(rx['gradient_reference'])),
                                        abs(float(r['grad1']) - float(ry['gradient_reference']))],
        'candidate_quality_xy': [r['grad_quality0'], r['grad_quality1']],
        'candidate_merge_count': int(r['merges']),
    }

print(json.dumps(out, indent=2, sort_keys=True))
