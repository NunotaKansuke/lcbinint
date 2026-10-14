import csv,json
from pathlib import Path
D=Path(__file__).resolve().parent
def load(p):
 with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
b=load(D.parent/'radial_hp_v34/full_after.tsv');a=load(D/'full_after.tsv')
keys=['case_id','profile','d_bin_index','epoch_index','rtol']
assert len(a)==len(b)==14432
assert [[r[k] for k in keys] for r in a]==[[r[k] for k in keys] for r in b]
fields=['mu','estimated_abs_error','nodes','value_converged','coverage_certified','stop']
s={k:sum(x[k]!=y[k] for x,y in zip(b,a)) for k in fields}
assert not any(s.values()),s
(D/'audit.json').write_text(json.dumps(s,indent=2)+'\n')
print(s)
