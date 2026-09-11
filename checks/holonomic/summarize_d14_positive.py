#!/usr/bin/env python3
"""Matched A/B/C trajectory summary. Failures remain in timing distributions."""
import hashlib,json,sys,gzip
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(sys.argv[1]);modes=['all-soft','no-soft','positive'];lanes=['cold','warm','radial']
key=['case_id','configuration_id','profile','d_bin_index','epoch_index','target']
def dist(v):
 a=np.asarray(v,dtype=float);f=a[np.isfinite(a)]
 return {'n':len(a),'finite':len(f),**{k:float(np.quantile(f,q)) if len(f) else None for k,q in [('p50',.5),('p90',.9),('p95',.95),('p99',.99),('max',1)]}}
def errors(d,l):return abs(d[l+'_mu']-d.reference)
def violations(d,l):return d[l+'_value_converged'].astype(bool)&(errors(d,l)>np.maximum(1e-16,d.target*abs(d[l+'_mu'])))
def stats(d):
 out={'rows':len(d),'lanes':{}}
 for l in lanes:
  whole=d[l+'_ms'];stage=d[[l+'_'+n+'_ms' for n in ['topology','setup','physical','estimator','scheduler']]]
  out['lanes'][l]={'whole_ms':dist(whole),'value_converged':int(d[l+'_value_converged'].sum()),'stop':d[l+'_stop'].value_counts().to_dict(),'status':d[l+'_status'].value_counts().to_dict(),
   'error_to_stored_VBM_1e6':dist(errors(d,l)/abs(d.reference)), 'observed_violation_on_converged':int(violations(d,l).sum()),
   'nodes':dist(d[l+'_nodes']),'splits':dist(d[l+'_splits']),
   'stage_ms':{n:dist(d[l+'_'+n+'_ms']) for n in ['topology','setup','physical','estimator','scheduler']},'whole_minus_stages_ms':dist(whole-stage.sum(axis=1))}
  assert np.isfinite(whole).all() and (whole>=0).all()
  assert np.isfinite(stage).all().all() and (stage>=0).all().all()
  assert (stage.max(axis=1)<=whole+0.001).all(),(l,'stage larger than enclosing timer')
  assert (stage.sum(axis=1)<=whole+0.001).all(),(l,'overlapping stage ledger')
  out['lanes'][l]['minimum_unclassified_ms']=float((whole-stage.sum(axis=1)).min())
 for pref in ['prebuilt','warm']:
  out[pref+'_backend']={'assurance':d[pref+'_assurance'].value_counts().to_dict(),'reason':d[pref+'_reason'].value_counts().to_dict(),'tier':d[pref+'_tier'].value_counts().to_dict(),'legacy_calls':int(d[pref+'_legacy'].sum()),'certified_count':dist(d.loc[d[pref+'_assurance']==2,pref+'_count'])}
 out['warm_direct_epochs']=int((d.warm_direct>0).sum());out['warm_attempted_epochs']=int((d.warm_attempts>0).sum());out['repaired_intervals']=int(d.warm_repaired.sum())
 out['positive_stage_ms']={n:dist(d[n]) for n in ['prebuilt_coefficient_ms','prebuilt_chain_ms','prebuilt_isolation_ms','prebuilt_refine_ms','prebuilt_variation_ms','prebuilt_event_ms']}
 return out
data={};out={'timing_sanity':'PASS','repeats':3,'rows_expected':14432,'timing_note':'same binary, serial modes, includes unsuccessful rows; variation is a nested subset of isolation/refine, not additive','modes':{},'comparisons':{}}
for mode in modes:
 path=root/(mode+'-final.tsv');path=path if path.exists() else path.with_suffix('.tsv.gz');d=pd.read_csv(path,sep=r'\s+',comment='#');assert len(d)==14432;assert not d.duplicated(key).any();data[mode]=d.set_index(key).sort_index()
 out['modes'][mode]={'sha256':hashlib.sha256(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()).hexdigest(),'all':stats(d),'steady':stats(d[d.trajectory_pos>0]),'by_profile_target':{f'{p}/{t}':stats(g) for (p,t),g in d.groupby(['profile','target'])}}
for left,right in [('all-soft','no-soft'),('no-soft','positive'),('all-soft','positive')]:
 a,b=data[left],data[right];assert a.index.equals(b.index);a=a.reset_index();b=b.reset_index();c={}
 for l in lanes:
  va,vb=violations(a,l),violations(b,l)
  lost=(a[l+'_value_converged']==1)&(b[l+'_value_converged']==0)
  new=vb&~va
  c[l]={'lost_convergence':int(lost.sum()),'gained_convergence':int(((a[l+'_value_converged']==0)&(b[l+'_value_converged']==1)).sum()),'new_observed_reference_violations':int(new.sum()),'relative_mu_change':dist(abs(a[l+'_mu']-b[l+'_mu'])/abs(a.reference)),'paired_speedup_left_over_right':dist(a[l+'_ms']/b[l+'_ms'])}
  b.loc[lost|new].to_csv(root/f'{left}-to-{right}-{l}-regressions.tsv',sep='\t',index=False)
 out['comparisons'][left+' -> '+right]=c
(root/'summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({m:{l:out['modes'][m]['all']['lanes'][l]['whole_ms']['p50'] for l in lanes} for m in modes},indent=2))
