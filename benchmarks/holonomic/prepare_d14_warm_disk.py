"""Export previous-epoch oracle centers for the standalone certificate probe."""
import argparse
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--roots',required=True);p.add_argument('--output',required=True);a=p.parse_args()
cols='case_id configuration_id profile d_bin_index epoch_index s q rho x y time u X reference'.split()
x=pd.read_csv(a.input,sep=r'\s+',comment='#',names=cols,float_precision='round_trip');x['sample']=range(len(x))
r=pd.read_csv(a.roots,sep=r'\s+',float_precision='round_trip');r=r[r.lane=='warm'].merge(x,on=cols[:5],validate='many_to_one')
count=0
with open(a.output,'w') as f:
 for key,tr in r.groupby(cols[:4],sort=False):
  previous=None
  for epoch,g in tr.groupby('epoch_index',sort=True):
   assert len(g)==14
   if previous is not None:
    row=g.iloc[0];print(int(row['sample']),row.time,row.y,row.rho,row.q,row.s,*previous,file=f);count+=1
   previous=g[['final_v_re','final_v_im']].to_numpy().flatten()
assert count==5412
