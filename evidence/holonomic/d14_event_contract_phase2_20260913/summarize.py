#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
KEYS=["case_id","configuration_id","profile","d_bin_index","epoch_index","target"]
ROOT_KEYS=["case_id","configuration_id","profile","d_bin_index","epoch_index","rep","lane"]

def read(name,comment=None):
    path=HERE/name
    if not path.exists(): path=HERE/(name+".gz")
    return pd.read_csv(path,sep=r"\s+",comment=comment)

baseline=read("whole_baseline_rep3.tsv",comment="#")
candidate=read("whole_rouche_rep3.tsv",comment="#")
paired=baseline.merge(candidate,on=KEYS,suffixes=("_baseline","_candidate"),validate="one_to_one")
timing_base=read("rootwork_baseline_timing.tsv")
timing_new=read("rootwork_rouche_timing.tsv")
root_paired=timing_base.merge(timing_new,on=ROOT_KEYS,suffixes=("_baseline","_candidate"),validate="one_to_one")

summary={"rows":len(paired),"whole":{},"root_work":{},"correctness":{}}
for target in sorted(paired.target.unique(),reverse=True):
    t=paired[paired.target==target]
    target_out={}
    for lane in ["cold","warm","radial"]:
        b=t[f"{lane}_ms_baseline"]
        c=t[f"{lane}_ms_candidate"]
        stats={}
        for q,name in [(0.5,"p50"),(0.9,"p90"),(0.95,"p95"),(0.99,"p99"),(1.0,"max")]:
            bv=float(b.quantile(q));cv=float(c.quantile(q))
            stats[name]={"baseline_ms":bv,"candidate_ms":cv,
                         "delta_percent":100*(cv/bv-1)}
        ref=np.abs(t[f"{lane}_mu_candidate"]-t.reference_candidate)
        limit=np.maximum(1e-12,target*np.abs(t.reference_candidate))
        stats["coverage"]=int(t[f"{lane}_value_converged_candidate"].sum())
        stats["observed_reference_violations"]=int((ref>limit).sum())
        target_out[lane]=stats
    summary["whole"][f"{target:g}"]=target_out

discrete=["topology_status","topology_cells","topology_events"]
for lane in ["cold","warm","radial"]:
    discrete += [f"{lane}_value_converged",f"{lane}_stop",f"{lane}_status"]
summary["correctness"]["discrete_mismatches"]={
    name:int((paired[f"{name}_baseline"].astype(str)!=
              paired[f"{name}_candidate"].astype(str)).sum()) for name in discrete}
for lane in ["cold","warm","radial"]:
    delta=np.abs(paired[f"{lane}_mu_baseline"]-paired[f"{lane}_mu_candidate"])
    summary["correctness"][f"{lane}_mu"]={
        "changed_rows":int(np.count_nonzero(delta)),"max_abs":float(delta.max()),
        "max_scaled":float(np.max(delta/np.maximum(1,np.abs(paired[f"{lane}_mu_baseline"]))))}

for lane in ["cold","warm"]:
    x=root_paired[root_paired.lane==lane]
    summary["root_work"][lane]={
        "rows":len(x),
        "contract_attempts":int(x.event_contract_attempts.sum()),
        "contract_accepts":int(x.event_contract_accepts.sum()),
        "qf_cold_calls_baseline":int(x.qf_cold_calls_baseline.sum()),
        "qf_cold_calls_candidate":int(x.qf_cold_calls_candidate.sum()),
        "qf_cold_sweeps_baseline":int(x.qf_cold_sweeps_baseline.sum()),
        "qf_cold_sweeps_candidate":int(x.qf_cold_sweeps_candidate.sum()),
        "event_count_mismatches":int((x.events_baseline!=x.events_candidate).sum()),
        "physical_count_mismatches":int((x.physical_real_events_baseline!=x.physical_real_events_candidate).sum()),
        "soft_count_mismatches":int((x.soft_events_baseline!=x.soft_events_candidate).sum()),
    }

(HERE/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2,sort_keys=True))
