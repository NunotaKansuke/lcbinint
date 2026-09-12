#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
d=pd.read_csv(HERE/'event_contract.tsv.gz',sep=' ')
base=d[~d.stage.str.endswith(('_local','_nosoft'))].copy()
local=d[d.stage.str.endswith('_local')].copy()
local['stage']=local.stage.str.removesuffix('_local')
keys=['case_id','configuration_id','profile','d_bin_index','epoch_index','lane','stage','rtol']
x=base.merge(local,on=keys,suffixes=('_raw','_local'))

def event_mismatch(g,suffix):
    tail=f'_{suffix}' if suffix else ''
    return ((g[f'candidate_positive{tail}']!=g[f'oracle_positive{tail}'])|
            (g[f'candidate_physical{tail}']!=g[f'oracle_physical{tail}'])|
            (g[f'candidate_complex_soft{tail}']!=g[f'oracle_complex_soft{tail}'])|
            (g[f'candidate_cells{tail}']!=g[f'oracle_cells{tail}'])|
            (g[f'candidate_topology_status{tail}']!=g[f'oracle_topology_status{tail}']))

summary={'rows':int(len(d)),'input_geometries':int(d[
    ['case_id','configuration_id','profile','d_bin_index','epoch_index']
    ].drop_duplicates().shape[0]),
         'stages':{},'event_gate':{},'certified_event_gate':{},
         'complex_soft_omission':{},'timing':{}}
for (lane,stage,converged,scalar),g in base.groupby(
        ['lane','stage','candidate_converged','scalar_certificate']):
    k=f'{lane}/{stage}/converged={converged}/scalar={scalar}'
    summary['stages'][k]={
        'rows':int(len(g)),
        'event_mismatch':int(event_mismatch(g,'').sum()),
        'value_convergence_loss':int(((g.oracle_value_converged==1)&
                                      (g.candidate_value_converged==0)).sum()),
        'stop_mismatch':int((g.candidate_stop!=g.oracle_stop).sum()),
        'max_mu_absdiff':float(g.mu_absdiff.max())}

# Research gate: it is evaluated only for finite non-converged candidates.
# The 5% factor mirrors the existing adaptive event-budget allocation.  The
# gate is evidence only and has no authority in solve_d14().
for (lane,stage,rtol),g in x.groupby(['lane','stage','rtol']):
    g=g[g.candidate_converged_raw==0].copy()
    gate=((g.scalar_certificate_raw==1)&(g.ambiguous_roots_raw==0)&
          (g.candidate_positive_raw==g.candidate_positive_local)&
          (g.candidate_complex_soft_raw==g.candidate_complex_soft_local)&
          (g.local_max_shift_local<=0.05*g.rtol))
    a=g[gate]
    mismatch=event_mismatch(a,'local') if len(a) else pd.Series(dtype=bool)
    violation=(a.mu_absdiff_local>
               a.rtol*a.oracle_mu_local.abs().clip(lower=1.0)) if len(a) else pd.Series(dtype=bool)
    k=f'{lane}/{stage}/rtol={rtol:g}'
    summary['event_gate'][k]={
        'nonconverged_candidates':int(len(g)),'gate_pass':int(gate.sum()),
        'gate_reject':int((~gate).sum()),'event_mismatch':int(mismatch.sum()),
        'value_convergence_loss':int(((a.oracle_value_converged_local==1)&
                                      (a.candidate_value_converged_local==0)).sum()),
        'stop_mismatch':int((a.candidate_stop_local!=a.oracle_stop_local).sum()),
        'value_contract_violation':int(violation.sum()),
        'max_mu_absdiff':float(a.mu_absdiff_local.max()) if len(a) else 0.0,
        'max_positive_radius_diff':float(a.positive_radius_maxdiff_local.max()) if len(a) else 0.0,
        'max_soft_radius_diff':float(a.soft_radius_maxdiff_local.max()) if len(a) else 0.0}
    certified=a[a.positive_certificate_raw==1]
    cm=event_mismatch(certified,'local') if len(certified) else pd.Series(dtype=bool)
    summary['certified_event_gate'][k]={
        'gate_pass':int(len(certified)),
        'event_mismatch':int(cm.sum()),
        'value_convergence_loss':int(((certified.oracle_value_converged_local==1)&
            (certified.candidate_value_converged_local==0)).sum()),
        'value_contract_violation':int((certified.mu_absdiff_local>
            certified.rtol*certified.oracle_mu_local.abs().clip(lower=1.0)).sum()),
        'positive_certificate_ms_p50':float(certified.positive_certificate_ms_raw.median()) if len(certified) else 0.0,
        'positive_certificate_ms_max':float(certified.positive_certificate_ms_raw.max()) if len(certified) else 0.0,
        'local_event_ms_p50':float(certified.event_build_ms_local.median()) if len(certified) else 0.0,
        'local_event_ms_max':float(certified.event_build_ms_local.max()) if len(certified) else 0.0}

nosoft=d[d.stage.str.endswith('_nosoft')].copy()
for (lane,stage,rtol),g in nosoft.groupby(['lane','stage','rtol']):
    k=f'{lane}/{stage}/rtol={rtol:g}'
    summary['complex_soft_omission'][k]={
        'rows':int(len(g)),
        'cell_mismatch':int((g.candidate_cells!=g.oracle_cells).sum()),
        'value_convergence_loss':int(((g.oracle_value_converged==1)&
                                      (g.candidate_value_converged==0)).sum()),
        'stop_mismatch':int((g.candidate_stop!=g.oracle_stop).sum()),
        'value_contract_violation':int((g.mu_absdiff>
            g.rtol*g.oracle_mu.abs().clip(lower=1.0)).sum()),
        'max_mu_absdiff':float(g.mu_absdiff.max())}

for (lane,stage),g in d[d.rtol==1e-3].groupby(['lane','stage']):
    summary['timing'][f'{lane}/{stage}']={
        'event_build_ms_p50':float(g.event_build_ms.median()),
        'event_build_ms_p90':float(g.event_build_ms.quantile(.9)),
        'event_build_ms_p99':float(g.event_build_ms.quantile(.99)),
        'event_build_ms_max':float(g.event_build_ms.max())}

bad=base[event_mismatch(base,'') | (base.candidate_stop!=base.oracle_stop) |
         (base.mu_absdiff>base.rtol*base.oracle_mu.abs().clip(lower=1.0))]
bad.to_csv(HERE/'mismatch_rows.tsv',sep=' ',index=False)
(HERE/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,indent=2,sort_keys=True))
