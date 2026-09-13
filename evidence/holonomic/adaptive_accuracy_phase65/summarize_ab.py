import pandas as pd,numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent
key=['case_id','profile','d_bin_index','epoch_index','target']
b=pd.read_csv(p/'base.tsv',sep=r'\s+',comment='#').set_index(key).sort_index()
out=[]
for name in ['safety1','initial15','k12','initial3']:
 path=p/(name+'.tsv')
 if not path.exists():continue
 a=pd.read_csv(path,sep=r'\s+',comment='#').set_index(key).sort_index()
 if not a.index.equals(b.index):continue
 for pro in ['uniform','linear']:
  for tol in [1e-3,1e-4]:
   ix=(a.index.get_level_values('profile')==pro)&(a.index.get_level_values('target')==tol)
   x=a[ix];y=b[ix]
   er=abs(x.warm_mu/x.reference-1);br=abs(y.warm_mu/y.reference-1)
   out.append(dict(candidate=name,profile=pro,target=tol,
    paired_speedup=float(np.median(y.warm_ms/x.warm_ms)),
    baseline_p50=float(y.warm_ms.median()),p50=float(x.warm_ms.median()),
    baseline_p99=float(y.warm_ms.quantile(.99)),p99=float(x.warm_ms.quantile(.99)),
    radial_speedup=float(np.median(y.radial_ms/x.radial_ms)),
    baseline_nodes=int(y.warm_nodes.sum()),nodes=int(x.warm_nodes.sum()),
    converged=int(x.warm_value_converged.sum()),status_changes=int((x.warm_status!=y.warm_status).sum()),
    reference_excess=int((er>tol).sum()),new_excess=int(((er>tol)&(br<=tol)).sum()),
    error_p95=float(er.quantile(.95)),max_mu_change=float((abs(x.warm_mu/y.warm_mu-1)).max())))
(p/'ab_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
