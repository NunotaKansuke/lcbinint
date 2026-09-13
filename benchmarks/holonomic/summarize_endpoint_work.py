"""Summarize diagnostic endpoint work; never infer speedup from probe timing."""
import argparse
import json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--baseline', required=True)
p.add_argument('--probe', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
read = lambda name: pd.read_csv(name, sep=r'\s+', float_precision='round_trip')
b, c = read(a.baseline), read(a.probe)
keys = ['case', 'profile', 'db', 'epoch']
assert not b.duplicated(keys).any() and not c.duplicated(keys).any()
m = b.merge(c, on=keys, suffixes=('_b', '_c'), validate='one_to_one')
assert len(m) == len(b) == len(c) == 7216
parity = {k: int((m[k+'_b'] != m[k+'_c']).sum()) for k in ['mu', 'ok', 'nodes']}
assert not any(parity.values()), parity
out = {'rows': len(m), 'changed': parity, 'profiles': {},
       'timing_warning': 'Probe includes diagnostic argument scans; not a speed A/B.'}
for name, g in c[c.trajectory_pos > 0].groupby('profile'):
    fields = ['endpoint_calls', 'endpoint_evaluations', 'endpoint_repeated',
              'endpoint_stationary', 'endpoint_small_step', 'endpoint_limit']
    d = {k: int(g[k].sum()) for k in fields}
    assert d['endpoint_calls'] == sum(d[k] for k in fields[3:])
    d['limit_fraction'] = d['endpoint_limit'] / d['endpoint_calls']
    d['repeated_evaluation_fraction'] = d['endpoint_repeated'] / d['endpoint_evaluations']
    d['evaluations_per_call'] = d['endpoint_evaluations'] / d['endpoint_calls']
    d['rows'] = len(g)
    out['profiles'][name] = d
Path(a.output).write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
