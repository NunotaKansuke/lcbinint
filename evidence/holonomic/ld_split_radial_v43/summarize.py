import csv,json,statistics as s
from pathlib import Path
D=Path(__file__).resolve().parent
KEY=['case_id','profile','d_bin_index','epoch_index','rtol']
def load(p,tab=True):
 with p.open() as f:
  return list(csv.DictReader(f,delimiter='\t')) if tab else [dict(zip(h,r.split())) for h in [f.readline().split()] for r in f]
def group(rows):
 d={}
 for r in rows:d.setdefault(tuple(r[k] for k in KEY),[]).append(r)
 return d
def median(rows,f):return s.median(float(r[f]) for r in rows)
out={'timing_contract':'fresh seed included; caustic construction excluded; one thread; 5 reps; both orders','timing':{}}
for direction in ['forward','reverse']:
 b,a=[group(load(D/f'{direction}_{stage}.tsv')) for stage in ['before','after']]
 assert b.keys()==a.keys()
 out['timing'][direction]={}
 for profile in ['uniform','linear']:
  for tol in [1e-3,1e-4]:
   keys=[k for k in b if k[1]==profile and float(k[4])==tol and float(b[k][0]['reference_1e-6'])<1.1]
   out['timing'][direction][f'{profile}_{tol}']={'inputs':len(keys),'us':{f:{'before':1000*s.median(median(b[k],f) for k in keys),'after':1000*s.median(median(a[k],f) for k in keys),'paired_reduction_percent':100*(1-s.median(median(a[k],f)/median(b[k],f) for k in keys))} for f in ['seed_ms','planning_ms','integration_ms','full_ms']},'nodes':{'before':s.median(median(b[k],'nodes') for k in keys),'after':s.median(median(a[k],'nodes') for k in keys)}}
b=load(D.parent/'ld_fold_handoff_v42/full_after.tsv');a=load(D/'full_after.tsv')
assert len(b)==len(a)==14432
assert [tuple(r[k] for k in KEY) for r in b]==[tuple(r[k] for k in KEY) for r in a]
out['full_audit']={'rows':len(a),'changed_values':sum(x['mu']!=y['mu'] for x,y in zip(b,a)),'changed_nodes':sum(x['nodes']!=y['nodes'] for x,y in zip(b,a)),'new_failures':sum(x['value_converged']=='1' and y['value_converged']!='1' for x,y in zip(b,a)),'failed_before':sum(r['value_converged']!='1' for r in b),'failed_after':sum(r['value_converged']!='1' for r in a),'new_tolerance_violations':sum(y['value_converged']=='1' and float(y['relative_error_reference'])>float(y['rtol']) and not(x['value_converged']=='1' and float(x['relative_error_reference'])>float(x['rtol'])) for x,y in zip(b,a)),'by_tolerance':{}}
for tol in [1e-3,1e-4]:
 rows=[r for r in a if float(r['rtol'])==tol and r['value_converged']=='1']
 out['full_audit']['by_tolerance'][str(tol)]={'converged':len(rows),'violations':sum(float(r['relative_error_reference'])>tol for r in rows),'maximum_relative_error':max(float(r['relative_error_reference']) for r in rows),'median_relative_error':s.median(float(r['relative_error_reference']) for r in rows)}
with (D.parent/'seeded_cartesian_cost_split_v8/selected.tsv').open() as f:sel={(r['id'],r['u']) for r in csv.DictReader(f,delimiter=' ')}
out['normal']={}
for direction in ['forward','reverse']:
 out['normal'][direction]={}
 for stage in ['before','after']:
  rows=load(D/f'normal_{direction}_{stage}.tsv',False);g={}
  for r in rows:
   if (r['id'],r['u']) in sel and float(r['tol'])==1e-4:g.setdefault((r['id'],r['u']),[]).append(r)
  out['normal'][direction][stage]={'pairs':len(g),'mean_pair_median_ms':{f:s.mean(median(v,f) for v in g.values()) for f in ['seed_ms','planning_ms','candidate_integration_ms','candidate_no_caustic_ms']}}
vbm_rows=load(D.parent/'seeded_paper_qrho_20261013/vbm_timing_reference.tsv')
vbm={tuple(r[k] for k in KEY[:-1])+(str(float(r['target'])),):float(r['vbm_ms']) for r in vbm_rows}
after=group(load(D/'reverse_after.tsv'));before=group(load(D/'reverse_before.tsv'))
out['saved_vbm_comparison']={}
for profile in ['uniform','linear']:
 for tol in [1e-3,1e-4]:
  keys=[k for k in after if k[1]==profile and float(k[4])==tol and float(after[k][0]['reference_1e-6'])<1.1]
  pairs=[(k,k[:-1]+(str(float(k[-1])),)) for k in keys]
  assert all(vk in vbm for k,vk in pairs)
  out['saved_vbm_comparison'][f'{profile}_{tol}']={'inputs':len(pairs),'vbm_median_us':1000*s.median(vbm[vk] for k,vk in pairs),'before_faster_count':sum(median(before[k],'full_ms')<vbm[vk] for k,vk in pairs),'after_faster_count':sum(median(after[k],'full_ms')<vbm[vk] for k,vk in pairs),'median_vbm_over_after_ratio':s.median(vbm[vk]/median(after[k],'full_ms') for k,vk in pairs),'provenance':'saved matched inputs/tolerance VBM timings; VBM was not rerun this turn'}
(D/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
