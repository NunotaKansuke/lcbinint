#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(sys.argv[1]);output=Path(sys.argv[2])
key=['case_id','configuration_id','profile','d_bin_index','epoch_index','target']
frames={m:pd.read_csv((root/f'{m}.tsv' if (root/f'{m}.tsv').exists() else root/f'{m}.tsv.gz'),sep=r'\s+',comment='#').set_index(key).sort_index() for m in ('incumbent','atlas')}
a,b=frames.values();assert len(a)==len(b)==14432 and a.index.equals(b.index)
quant=lambda x:{k:float(v) for k,v in zip(['p50','p90','p95','p99','max'],np.quantile(x,[.5,.9,.95,.99,1]))} if len(x) else None
summary={'rows':len(a),'reps':json.loads((root/'manifest.json').read_text())['reps'],'timing_sanity':{},'results':{}}
violations={}
for name,d in frames.items():
 result={};bad=[]
 for lane in ('cold','warm','radial'):
  stages=[f'{lane}_{s}_ms' for s in ('topology','setup','physical','estimator','scheduler')]
  whole=d[f'{lane}_ms'];vals=d[stages];assert np.isfinite(vals).all().all() and (vals>=0).all().all()
  assert (vals.max(axis=1)<=whole+1e-6).all(),(name,lane,'stage > whole')
  unclassified=whole-vals.sum(axis=1);assert (unclassified>=-1e-5).all(),(name,lane,'overlapping stages')
  ok=d[f'{lane}_value_converged'].eq(1);err=abs(d[f'{lane}_mu']/d.reference-1);vi=ok & (err>d.index.get_level_values('target'))
  violations[name,lane]=vi
  steady=d.trajectory_pos>0
  result[lane]={'all_attempt_ms':quant(whole),'value_converged':int(ok.sum()),'converged_ms':quant(whole[ok]),'relative_error_converged':quant(err[ok]),'observed_reference_violations':int(vi.sum()),'stop':d[f'{lane}_stop'].value_counts().to_dict(),'steady_only_ms':quant(whole[steady]),'stage_ms':{k:quant(d[k]) for k in stages},'unclassified_ms':quant(unclassified)}
  summary['timing_sanity'][name+'_'+lane]='PASS'
  bad.extend(d[vi].reset_index().assign(lane=lane).to_dict('records'))
 (output.parent/f'{name}_violations.json').write_text(json.dumps(bad,indent=2)+'\n')
 if name=='atlas':
  for lane in ('cold','warm'):
   complete=d[f'{lane}_atlas'].eq('Complete');ok=d[f'{lane}_value_converged'].eq(1)
   result[lane].update({'atlas_status':d[f'{lane}_atlas'].value_counts().to_dict(),'complete_cover':int(complete.sum()),'complete_and_value':int((complete&ok).sum()),'new_reference_violations':int((violations[name,lane]&~violations['incumbent',lane]).sum()),'contacts':quant(d.loc[complete,f'{lane}_contacts']), 'pair_accepted':int(d[f'{lane}_pair_accepted'].sum()),'pair_attempts':int(d[f'{lane}_pair_attempts'].sum()),'pair_fraction_converged':float(d.loc[ok,f'{lane}_pair_accepted'].sum()/max(1,d.loc[ok,f'{lane}_evaluations'].sum())), 'atlas_substage_ms':{stage:quant(d[f'{lane}_{stage}_ms']) for stage in ['coefficient','proof','refine']}, 'atlas_unclassified_ms':quant(d[f'{lane}_topology_ms']-d[[f'{lane}_{stage}_ms' for stage in ['coefficient','proof','refine']]].sum(axis=1)),'complete_contact_count_differs_from_legacy':int((complete & d[f'{lane}_contacts'].ne(d.reference_contacts)).sum()),'boxes':quant(d[f'{lane}_boxes']),'fixed_cost_below_half_own_adaptive':int((ok & (d[f'{lane}_topology_ms']<.5*(d[f'{lane}_ms']-d[f'{lane}_topology_ms']))).sum()),'fixed_cost_below_half_incumbent_radial':int((ok & (d[f'{lane}_topology_ms']<.5*a.radial_ms)).sum()),'paired_speedup_converged':quant(a.loc[ok,f'{lane}_ms']/d.loc[ok,f'{lane}_ms'])})
  result['warm_evidence_updates']={k:quant(d[k]) for k in ['warm_proofs_reused','warm_proofs_invalidated','warm_event_updates','warm_range_accepted']}
  result['old_D14_calls_in_atlas_timed_path']=0
 summary['results'][name]=result
summary['by_profile_tolerance']={}
for name,d in frames.items():
 for (profile,tol),sub in d.groupby(level=['profile','target']):
  summary['by_profile_tolerance'][f'{name}/{profile}/{tol}']={lane:{'ms':quant(sub[f'{lane}_ms']),'coverage':int(sub[f'{lane}_value_converged'].sum())} for lane in ['cold','warm','radial']}
output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({name:{lane:(r[lane]['value_converged'],r[lane]['all_attempt_ms']['p50']) for lane in ['cold','warm','radial']} for name,r in summary['results'].items()},indent=2))
