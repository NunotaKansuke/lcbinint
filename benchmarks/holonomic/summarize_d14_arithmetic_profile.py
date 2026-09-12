"""Profile diagnostic A/B (not uninstrumented whole-epoch acceptance)."""
import json
from pathlib import Path
import pandas as pd
root=Path('evidence/holonomic/adaptive_extreme_phase12')
a=pd.read_csv(root/'profile_base.tsv',sep=r'\s+')
keys=['case','profile','db','epoch']
out={'rows':len(a),'candidates':{}}
for name in ['interleaved','maxnorm']:
 b=pd.read_csv(root/f'profile_{name}.tsv',sep=r'\s+')
 assert len(a)==7216 and a[keys].equals(b[keys])
 stages=['whole','presearch','real','qf','residual','physics']
 out['candidates'][name]={
  'baseline_p50_ms':a[stages].median().to_dict(),
  'candidate_p50_ms':b[stages].median().to_dict(),
  'max_mu_absolute_difference':float(abs(a.mu-b.mu).max()),
  'convergence_mismatches':int((a.ok!=b.ok).sum()),
  'node_mismatches':int((a.nodes!=b.nodes).sum()),
  'baseline_converged':int(a.ok.sum()),'candidate_converged':int(b.ok.sum())}
(root/'profile_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
