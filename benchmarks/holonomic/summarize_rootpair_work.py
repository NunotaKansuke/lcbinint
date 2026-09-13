"""Summarize actual corrector work, keeping overlapping rejection tags separate."""
import argparse
import json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--baseline',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
x,b=[pd.read_csv(f,sep=r'\s+',float_precision='round_trip') for f in [a.input,a.baseline]]
keys=['case','profile','db','epoch','trajectory_pos']
assert len(x)==7216 and x[keys].equals(b[keys])
assert (x.newton_iterations>=x.newton_pairs_accepted).all()
assert (x.rootpair_calls==x.rootpair_success+x.rootpair_cold).all()
out={'rows':len(x),'mu_max_difference':float((x.mu-b.mu).abs().max()),
 'ok_mismatches':int((x.ok!=b.ok).sum()),'nodes_mismatches':int((x.nodes!=b.nodes).sum()),'profiles':{}}
counts=['rootpair_calls','rootpair_success','rootpair_cold','predictor_reject','newton_reject',
 'branch_reject','tmax_reject','vfloor_reject','newton_iterations','newton_pairs_accepted','quartic_cold','quartic_warm']
for name,g in x[x.trajectory_pos>0].groupby('profile'):
 sums={c:int(g[c].sum()) for c in counts}
 out['profiles'][name]={'rows':len(g),'counts':sums,'transport_success_fraction':sums['rootpair_success']/sums['rootpair_calls'],
 'arc_median_ms':float(g.arc.median()),'rootpair_median_ms':float(g.rootpair_ms.median())}
Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
