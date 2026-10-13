import csv,json,statistics as s
from pathlib import Path
D=Path(__file__).resolve().parent

def load(name,tab=True):
 with (D/name).open() as f:
  return list(csv.DictReader(f,delimiter='\t')) if tab else [dict(zip(h,r.split())) for h in [f.readline().split()] for r in f]
def med(rows,field):return s.median(float(r[field]) for r in rows)
def groups(rows):
 d={}
 for r in rows:d.setdefault((r['case_id'],r['profile'],r['d_bin_index'],r['epoch_index'],r['rtol']),[]).append(r)
 return d
out={}
for direction in ['forward','reverse']:
 b,a=[groups(load(f'{direction}_{stage}.tsv')) for stage in ['before','after']]
 out[direction]={}
 for profile in ['uniform','linear']:
  keys=[k for k in b if k[1]==profile and float(k[4])==1e-3 and float(b[k][0]['reference_1e-6'])<1.1]
  out[direction][profile]={'inputs':len(keys),'timing_us':{f:{'before':1000*s.median(med(b[k],f) for k in keys),'after':1000*s.median(med(a[k],f) for k in keys),'paired_reduction_percent':100*(1-s.median(med(a[k],f)/med(b[k],f) for k in keys))} for f in ['seed_ms','planning_ms','integration_ms','full_ms']},'fast_count_fraction':sum(int(r['fast_counts']) for k in keys for r in a[k])/sum(int(r['radius_probes']) for k in keys for r in a[k])}
b,a=[load(f'full_{stage}.tsv') for stage in ['before','after']]
assert len(b)==len(a)==14432
assert [(r['case_id'],r['profile'],r['d_bin_index'],r['epoch_index'],r['rtol']) for r in b]==[(r['case_id'],r['profile'],r['d_bin_index'],r['epoch_index'],r['rtol']) for r in a]
fields=['mu','nodes','evaluations','value_converged','stop','numerical_status','radius_probes','support_proven']
out['full_audit']={'rows':len(a),'changed_fields':{f:sum(x[f]!=y[f] for x,y in zip(b,a)) for f in fields},'failed_before':sum(r['value_converged']!='1' for r in b),'failed_after':sum(r['value_converged']!='1' for r in a),'previously_successful_changed':sum(x['value_converged']=='1' and any(x[f]!=y[f] for f in ['mu','nodes','evaluations']) for x,y in zip(b,a)),'new_tolerance_violations':sum(y['value_converged']=='1' and float(y['relative_error_reference'])>float(y['rtol']) and not(x['value_converged']=='1' and float(x['relative_error_reference'])>float(x['rtol'])) for x,y in zip(b,a))}
selected=Path('evidence/holonomic/seeded_cartesian_cost_split_v8/selected.tsv')
with selected.open() as f:sel={(r['id'],r['u']) for r in csv.DictReader(f,delimiter=' ')}
out['normal']={}
for prefix in ['normal','normal_reverse']:
 out['normal'][prefix]={}
 for stage in ['before','after']:
  rows=load(f'{prefix}_{stage}.tsv',False);g={}
  for r in rows:
   if (r['id'],r['u']) in sel and float(r['tol'])==1e-4:g.setdefault((r['id'],r['u']),[]).append(r)
  out['normal'][prefix][stage]={'pairs':len(g), 'mean_pair_median_ms':{f:s.mean(med(v,f) for v in g.values()) for f in ['seed_ms','planning_ms','candidate_integration_ms','candidate_no_caustic_ms']}}
out['exact_count_audit']=json.loads((D/'exact_count_audit.json').read_text())
(D/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
