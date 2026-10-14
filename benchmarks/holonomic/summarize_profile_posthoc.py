"""Paired diagnostic-on/off profile audit; never a production speed claim."""
import argparse
import json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--diagnostic', required=True)
p.add_argument('--timing', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
x, y = [pd.read_csv(f, sep=r'\s+', float_precision='round_trip')
        for f in (a.diagnostic, a.timing)]
keys = ['case', 'profile', 'db', 'epoch', 'trajectory_pos']
assert len(x) == 7216 and x[keys].equals(y[keys])
assert (y.diagnostic == 0).all()
out = {'rows': len(x), 'ok_mismatches': int((x.ok != y.ok).sum()),
       'nodes_mismatches': int((x.nodes != y.nodes).sum()),
       'max_mu_difference': float((x.mu-y.mu).abs().max()), 'profiles': {}}
columns = ['whole','topology','physics','arc','endpoint','setup','real',
           'residual','metadata','presearch','diagnostic','conjugate',
           'classification','probes','k']
for name in ['uniform','linear']:
    mask = (x.profile == name) & (x.trajectory_pos > 0)
    out['profiles'][name] = {'rows': int(mask.sum()), 'stages': {}}
    for col in columns:
        before, after = x.loc[mask,col], y.loc[mask,col]
        out['profiles'][name]['stages'][col] = {
            'diagnostic_median_ms': float(before.median()),
            'timing_median_ms': float(after.median()),
            'paired_median_difference_ms': float((before-after).median())}
Path(a.output).write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
