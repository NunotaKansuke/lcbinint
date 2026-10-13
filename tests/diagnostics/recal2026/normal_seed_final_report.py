import csv,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=R/'tests/diagnostics/results/recal2026/normal-seed-index-20261014'
def read(path):return list(csv.DictReader(path.open(),delimiter='\t'))
meta={str(r['sample_id']):r for r in json.loads((P.parent/'deferred-cache-production-20261013/inputs.json').read_text())}
vbm={(str(r['sample_id']),r['target']):r for r in json.loads((P/'vbm.json').read_text())['rows']}
seeds={mode:[read(P/f'seed-{mode}-{rep}.tsv') for rep in range(3)] for mode in ['before','after']}
kernels={mode:[{(r['sample_id'],float(r['target'])):r for r in read(P/f'kernel-{mode}-{rep}.tsv')} for rep in range(3)] for mode in ['before','after']}
groups=[]
fixed=read(P/'fixed.tsv');fixed_by={}
for row in fixed:
 if row['stop_reason']=='completed':fixed_by.setdefault(row['sample_id'],[]).append(row)
for profile in ['uniform','linear']:
 ids=[k for k,v in meta.items() if v['profile']==profile]
 seed_old=[statistics.median(float(next(r for r in seeds['before'][rep] if r['id']==k)['ms']) for rep in range(3)) for k in ids]
 seed_new=[statistics.median(float(next(r for r in seeds['after'][rep] if r['id']==k)['ms']) for rep in range(3)) for k in ids]
 for target in [.001,.0001]:
  pairs=[];bad=[];accepted=0;vbad=[];vr=[];fx=[];fc=[];winner_bins=[];exceptions=[];mismatches=[]
  for k in ids:
   key=(k,target)
   if key not in kernels['after'][0]:exceptions.append(k);continue
   r=kernels['after'][0][key];ref=meta[k]['reference'];err=abs(float(r['magnification'])/ref-1)
   for rep in range(3):
    for f in ['magnification','indicator','refinements','converged','stop_reason','maps']:
     if kernels['before'][rep][key][f]!=kernels['after'][rep][key][f]:mismatches.append([k,rep,f])
   old=statistics.median(float(kernels['before'][rep][key]['ms']) for rep in range(3));new=statistics.median(float(kernels['after'][rep][key]['ms']) for rep in range(3));pairs.append((old,new))
   accepted+=r['converged']=='1'
   if r['converged']=='1' and err>target:bad.append(k)
   v=vbm[key]
   if abs(v['magnification']/ref-1)>target:vbad.append(k)
   if r['converged']=='1' and err<=target and abs(v['magnification']/ref-1)<=target:vr.append(v['ms']/new)
   eligible=[float(f['ms']) for f in fixed_by.get(k,[]) if abs(float(f['magnification'])/ref-1)<=target]
   if eligible and r['converged']=='1' and err<=target:
    fx.append(min(eligible)/new)
    winner=min((f for f in fixed_by[k] if abs(float(f['magnification'])/ref-1)<=target),key=lambda f:float(f['ms']));winner_bins.append(int(winner['nbin']))
   cart=[float(f['ms']) for f in fixed_by.get(k,[]) if f['method']=='cartesian' and abs(float(f['magnification'])/ref-1)<=target]
   if cart and r['converged']=='1' and err<=target:fc.append(min(cart)/new)
  groups.append(dict(profile=profile,target=target,seed_median_before_us=statistics.median(seed_old)*1000,seed_median_after_us=statistics.median(seed_new)*1000,seed_sum_ratio=sum(seed_new)/sum(seed_old),kernel_count=len(pairs),kernel_median_new_over_old=statistics.median(b/a for a,b in pairs),accepted=accepted,false_accept_ids=bad,exceptions=exceptions,numerical_mismatches=mismatches,vbm_actual_over_target_ids=vbad,matched_vbm_count=len(vr),median_vbm_over_adaptive=statistics.median(vr) if vr else None,oracle_fixed_count=len(fx),oracle_cartesian_count=len(fc),median_fastest_accurate_cartesian_over_adaptive=statistics.median(fc) if fc else None,oracle_winner_nbin_median=statistics.median(winner_bins) if winner_bins else None,median_fastest_accurate_fixed_over_adaptive=statistics.median(fx) if fx else None))
a=read(P.parent/'seed-index-port-20261013/seeds-before.tsv');b=read(P/'all-seeds-after.tsv');mis=[x['id'] for x,y in zip(a,b) if any(x[f]!=y[f] for f in ['id','support','nseeds','coordinates'])]
result=dict(groups=groups,all_seed_identity={'count':len(b),'mismatch_ids':mis},fixed_rows=len(fixed),fixed_scope='fastest actually accurate measured Cartesian or polar resolution chosen using reference; oracle lower bound, not production selection',kernel_scope='warm seed geometry cache, single-pass initial16; excludes center roots and caustic construction',vbm_scope='retimed same source and targets, Python-call overhead included; independent reference saved RelTol1e-6',certified=False)
(P/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
