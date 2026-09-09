"""Summarize paired, same-input cold/warm research measurements (ms)."""
import csv
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[2] / 'evidence' / 'holonomic'

def rows(name):
    with (root / name).open() as f:
        return list(csv.DictReader(f))

def median(items, key):
    return statistics.median(float(x[key]) for x in items)

print('NOTE: analytic5Jac rows are the pre-certification local-gate experiment.')
print('Final LD analytic status is GRADIENT_UNRELIABLE; see gm_jac_fail_closed_phase3.csv.')
print((root / 'gm_coverage64_jac_refined_phase3.txt').read_text().splitlines()[-1])
value = rows('gm_cold_warm_value_phase3.csv')
jac = rows('gm_cold_warm_jac_local_gate_phase3.csv')
for lane, data in [('value', value), ('analytic5Jac-local-gate-only', jac)]:
    for mode in ['cold', 'warm']:
        for u in ['0.0', '0.5']:
            batch = [r for r in data if r['mode'] == mode and r['u'] == u]
            if not batch:
                continue
            print(lane, mode, 'u='+u, 'epochs='+str(len(batch)),
                  'GM/V2 median ms', median(batch, 'gm_ms'), median(batch, 'v2_ms'),
                  'status OK', sum(x['gm_status'] == '0' for x in batch),
                  sum(x['v2_status'] == '0' for x in batch),
                  'max mu/Jac err', max(float(x['mu_error']) for x in batch),
                  max(float(x['jac_error']) for x in batch))
            print('  median stage ms:', {k: median(batch, k) for k in
                  ['topology_ms', 'seed_ms', 'connection_ms', 'transport_ms', 'arc_ms', 'reseed_ms']})
            print('  connection successes/attempts:',
                  sum(int(x['connection_attempts']) - int(x['connection_failed']) for x in batch),
                  sum(int(x['connection_attempts']) for x in batch),
                  'transport nodes:', sum(int(x['transported']) for x in batch),
                  '/', sum(int(x['nodes']) for x in batch))

def key(row):
    return tuple(row[k] for k in ['mode', 'name', 'u', 'epoch'])

vmap = {key(r): r for r in value}
for mode in ['cold', 'warm']:
    batch = [r for r in jac if r['mode'] == mode and r['u'] == '0.5']
    if batch:
        marginal = [float(r['gm_ms']) - float(vmap[key(r)]['gm_ms']) for r in batch]
        ratio = [float(r['gm_ms']) / float(vmap[key(r)]['gm_ms']) for r in batch]
        print('analytic Jacobian marginal', mode, 'median ms', statistics.median(marginal),
              'median paired ratio', statistics.median(ratio))

for lane, data in [('value', value), ('analytic5Jac-local-gate-only', jac)]:
    initial = [r for r in data if r['mode'] == 'cold' and r['epoch'] == '0']
    ld = [r for r in initial if r['u'] == '0.5']
    print(lane, 'original 108 cases: status OK',
          sum(x['gm_status'] == '0' for x in initial), '/', len(initial),
          'LD all-node transport', sum(x['nodes'] == x['transported'] and int(x['nodes']) > 0 for x in ld), '/', len(ld))

coarse = rows('gm_angular_jac512_phase3.csv')
fine = rows('gm_angular_jac2048_phase3.csv')
conv = []
for a, b in zip(coarse, fine):
    assert a['name'] == b['name']
    err = max(abs(float(a[k])-float(b[k]))/(1+abs(float(b[k])))
              for k in ['ref_dx', 'ref_dy', 'ref_drho', 'ref_dq', 'ref_da'])
    if err < 1e-7 and float(a['jac_error']) < 1e100 and float(b['jac_error']) < 1e100:
        conv.append(b)
print('angular derivative reference: 512/2048 agreement <1e-7:', len(conv), '/', len(fine))
print('Unconverged angular derivatives are not independent scientific acceptance.')
