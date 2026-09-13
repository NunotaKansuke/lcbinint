"""Oracle-only trajectory geometry diagnostic, NOT a reuse certificate."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
p=argparse.ArgumentParser();p.add_argument('--roots',required=True);p.add_argument('--output',required=True);a=p.parse_args()
x=pd.read_csv(a.roots,sep=r'\s+',float_precision='round_trip');x=x[x.lane=='warm'];records=[]
for key,trajectory in x.groupby(['case_id','configuration_id','profile','d_bin_index'],sort=False):
 previous=None
 for epoch,g in trajectory.groupby('epoch_index',sort=True):
  z=g.final_v_re.to_numpy()+1j*g.final_v_im.to_numpy();role=g.physical_real.to_numpy()==1
  assert len(z)==14 and np.isfinite(z).all()
  if previous is not None:
   old,oldrole=previous
   cost=np.abs(old[:,None]-z[None,:])/(1+np.abs(old[:,None]));i,j=linear_sum_assignment(cost)
   new=z[j];newrole=role[j];motion=np.abs(new-old)
   separation=np.abs(old[:,None]-old[None,:]);np.fill_diagonal(separation,np.inf)
   nearest=separation.min(axis=1)
   # An oracle-informed geometric opportunity, not a necessary certificate.
   # Includes a binary64 export floor; no polynomial containment is proved.
   radius=motion+32*np.finfo(float).eps*(1+np.abs(new))
   overlap=radius[:,None]+radius[None,:]>=separation
   np.fill_diagonal(overlap,False)
   physical_mask=oldrole|newrole
   records.append(dict(zip(['case_id','configuration_id','profile','d_bin_index'],key))|{
    'epoch':int(epoch),'physical_count_old':int(oldrole.sum()),'physical_count_new':int(newrole.sum()),
    'physical_identity_changes':int((oldrole!=newrole).sum()),
    'max_motion_over_separation':float((motion/nearest).max()),
    'median_motion_over_separation':float(np.median(motion/nearest)),
    'overlap_pairs':int(np.triu(overlap,1).sum()),
    'physical_overlap_roots':int((overlap.any(axis=1)&physical_mask).sum()),
    'small_motion_roots':int((radius<nearest/4).sum())})
  previous=z,role
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);r=pd.DataFrame(records);assert len(r)==5412;r.to_csv(out/'motion.tsv',sep='\t',index=False)
s={'transitions':len(r),'unchanged_physical_count':int((r.physical_count_old==r.physical_count_new).sum()),'unchanged_matched_physical_identity':int((r.physical_identity_changes==0).sum()),'all_old_center_disks_disjoint':int((r.overlap_pairs==0).sum()),'no_physical_disk_overlap':int((r.physical_overlap_roots==0).sum()),'median_small_motion_roots':float(r.small_motion_roots.median()),'median_root_motion_over_separation_quantiles':{str(q):float(r.median_motion_over_separation.quantile(q))for q in [.5,.9,.99]},'limitation':'Uses future oracle roots and binary64 exports. No event completeness or interval certificate; not a production routing rule.'}
(out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
