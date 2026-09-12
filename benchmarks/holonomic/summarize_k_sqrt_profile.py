"""Compare K-rule profile by LD profile; never use this as whole acceptance."""
from pathlib import Path
import json
import pandas as pd
r=Path('evidence/holonomic/adaptive_extreme_phase15')
a=pd.read_csv('evidence/holonomic/adaptive_extreme_phase14/profile_warm_count_only.tsv',sep=r'\s+')
b=pd.read_csv(r/'profile_single_sqrt.tsv',sep=r'\s+')
assert len(a)==len(b)==7216
assert a[['case','profile','db','epoch']].equals(b[['case','profile','db','epoch']])
out={}
for profile in ['uniform','linear']:
 m=(a.profile==profile)&(a.trajectory_pos>0)
 cols=['whole','topology','physics','k','rescue']
 out[profile]={'base_ms':a[m][cols].median().to_dict(),
 'candidate_ms':b[m][cols].median().to_dict(),
 'max_mu_relative_difference':float((abs(a[m].mu-b[m].mu)/abs(a[m].mu)).max()),
 'node_mismatch':int((a[m].nodes!=b[m].nodes).sum()),
 'ok_mismatch':int((a[m].ok!=b[m].ok).sum())}
(r/'profile_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
