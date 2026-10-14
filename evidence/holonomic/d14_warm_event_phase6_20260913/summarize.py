#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd

H=Path(__file__).resolve().parent
KW=dict(sep=r"\s+",comment="#",low_memory=False)
keys=["case_id","configuration_id","profile","d_bin_index","epoch_index","target"]
b=pd.read_csv(H/"whole_eager_dd.tsv.gz",**KW)
c=pd.read_csv(H/"whole_lazy_dd.tsv.gz",**KW)
x=b.merge(c,on=keys,suffixes=("_eager","_lazy"),validate="one_to_one")
out={"rows":len(x),"whole":{},"presearch_case0":{}}
for target in sorted(x.target.unique(),reverse=True):
 y=x[x.target==target];t={}
 for lane in ("cold","warm","radial"):
  a,d=y[lane+"_ms_eager"],y[lane+"_ms_lazy"]
  z={"coverage":int(y[lane+"_value_converged_lazy"].sum()),
     "status_mismatches":int((y[lane+"_status_eager"]!=y[lane+"_status_lazy"]).sum()),
     "mu_max_absdiff":float(np.max(np.abs(y[lane+"_mu_eager"]-y[lane+"_mu_lazy"]))) }
  for q,name in ((.5,"p50"),(.9,"p90"),(.95,"p95"),(.99,"p99"),(1,"max")):
   av,dv=float(a.quantile(q)),float(d.quantile(q));z[name]={"eager_ms":av,"lazy_ms":dv,"delta_percent":100*(dv/av-1)}
  z["paired_median_delta_us"]=1000*float((d-a).median());t[lane]=z
 out["whole"][f"{target:g}"]=t

p=pd.read_csv(H/"presearch_newton_case0.tsv.gz",**KW)
p=p[p.stage.isin(["presearch","presearch_newton1","presearch_newton2","presearch_newton4"])]
for lane in ("cold","warm"):
 out["presearch_case0"][lane]={}
 for stage in sorted(p.stage.unique()):
  y=p[(p.lane==lane)&(p.stage==stage)]
  out["presearch_case0"][lane][stage]={
   "rows":len(y),"scalar_certificates":int(y.scalar_certificate.sum()),
   "event_count_mismatches":int(((y.candidate_positive!=y.oracle_positive)|(y.candidate_physical!=y.oracle_physical)|(y.candidate_complex_soft!=y.oracle_complex_soft)).sum()),
   "cell_count_mismatches":int((y.candidate_cells!=y.oracle_cells).sum()),
   "lost_convergence":int(((y.oracle_value_converged==1)&(y.candidate_value_converged==0)).sum()),
   "mu_max_absdiff":float(y.mu_absdiff.max())}
(H/"summary.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=2,sort_keys=True))
