#!/usr/bin/env python3
import gzip,json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
def read(name): return pd.read_csv(HERE/name,sep=r"\s+")
summary={}
g=read("local_continuation_guard.tsv.gz")
summary["local_event_continuation"]={
 "rows":len(g),"cell_plan_match":int(g.cell_plan_match.sum()),
 "cell_plan_match_rate":float(g.cell_plan_match.mean()),
 "p50_ms":float(g.continuation_ms.median()),
 "p90_ms":float(g.continuation_ms.quantile(.9)),
 "previous_plan_guard_accepts":int(g.previous_plan_match.sum()),
 "previous_plan_guard_false_accepts":int(((g.previous_plan_match==1)&(g.cell_plan_match==0)).sum()),
 "shift_guard_01_accepts":int(g.guard_01.sum()),
 "shift_guard_01_false_accepts":int(((g.guard_01==1)&(g.cell_plan_match==0)).sum()),
}
n=read("neumaier_3_8.tsv.gz")
summary["neumaier_point_qf"]={
 str(s):{"rows":len(x),"pairwise_disjoint":int(x.disk_disjoint.sum()),
  "classifiable":int(((x.disk_disjoint==1)&(x.disk_classifiable==1)).sum()),
  "false_topology_matches":int(((x.disk_disjoint==1)&(x.disk_classifiable==1)&(x.cell_plan_match==0)).sum()),
  "p50_ms":float(x.method_ms.median()),"p90_ms":float(x.method_ms.quantile(.9)),
  "p99_ms":float(x.method_ms.quantile(.99))}
 for s,x in n.groupby("sweeps")}
p=read("profile_nocapture_timing.tsv.gz")
fields=["whole_classify_ms","radial_events_ms","cell_topology_remainder_ms",
 "d14_struct_build_ms","d14_expand_ms","d14_solve_ms","presearch_ms",
 "d14real_ms","qf_polish_ms","residual_eval_ms","completeness_check_ms",
 "physical_classify_ms","soft_event_ms"]
summary["exclusive_profile"]={lane:{f:{"p50_ms":float(x[f].median()),
 "p90_ms":float(x[f].quantile(.9))} for f in fields}
 for lane,x in p.groupby("lane")}
(HERE/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2,sort_keys=True))
