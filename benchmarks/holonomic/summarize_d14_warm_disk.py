import argparse,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
x=pd.read_csv(a.input,sep=r'\s+',float_precision='round_trip');out=[]
for stage,g in x.groupby('stage'):
 assert len(g)==5412
 total=g.prepare_ms+g.correction_ms+g.screen_ms+g.interval_ms
 out.append({'stage':int(stage),'rows':len(g),'finite':int(g.finite.sum()),'screened':int(g.screened.sum()),'certified':int(g.certified.sum()),'p50_ms':{k:float(g[k].median())for k in ['prepare_ms','correction_ms','screen_ms','interval_ms']},'total_p50_ms':float(total.median()),'total_mean_ms':float(total.mean()),'interval_mean_when_screened_ms':float(g.loc[g.screened==1,'interval_ms'].mean()) if g.screened.sum() else None})
Path(a.output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,indent=2,allow_nan=False))
