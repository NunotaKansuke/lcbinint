import csv,json,math,statistics as s
from pathlib import Path
D=Path(__file__).resolve().parent
def load(p):
 with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
b=load(D.parent/'ld_fold_handoff_v42/full_after.tsv');a=load(D/'full_after.tsv')
K=['case_id','profile','d_bin_index','epoch_index','rtol']
assert len(a)==len(b)==14432
assert [tuple(r[k] for k in K) for r in a]==[tuple(r[k] for k in K) for r in b]
bad=lambda r:r['value_converged']=='1' and (not math.isfinite(float(r['relative_error_reference'])) or float(r['relative_error_reference'])>float(r['rtol']))
o={'rows':len(a),'new_violations':[{k:y[k] for k in K+['relative_error_reference']} for x,y in zip(b,a) if bad(y) and not bad(x)],'new_failures':[{k:y[k] for k in K+['stop','frontier_reason']} for x,y in zip(b,a) if x['value_converged']=='1' and y['value_converged']!='1'],'failures':sum(y['value_converged']!='1' for y in a),'violations':sum(bad(y) for y in a),'changed_values':sum(x['mu']!=y['mu'] for x,y in zip(b,a))}
for profile in ['uniform','linear']:
 sel=[i for i,r in enumerate(a) if r['profile']==profile and float(r['rtol'])==1e-3 and float(r['reference_1e-6'])<1.1]
 o[profile]={'probes_before':s.median(float(b[i]['radius_probes']) for i in sel),'probes_after':s.median(float(a[i]['radius_probes']) for i in sel),'nodes_before':s.median(float(b[i]['nodes']) for i in sel),'nodes_after':s.median(float(a[i]['nodes']) for i in sel)}
o['changed_uniform_values']=sum(x['mu']!=y['mu'] for x,y in zip(b,a) if y['profile']=='uniform')
o['newly_converged']=[{k:y[k] for k in K+['relative_error_reference']} for x,y in zip(b,a) if x['value_converged']!='1' and y['value_converged']=='1']
o['certified_rows']=sum(y['coverage_certified']=='1' for y in a)
o['checks']=sum(int((D/name).read_text().split()[0]) for name in ['test_split.log','test_adaptive_radial.log','test_weighted_radial.log','test_seeded_radial_frontier.log'])
(D/'audit.json').write_text(json.dumps(o,indent=2)+'\n')
print({k:len(v) if isinstance(v,list) else v for k,v in o.items()})
