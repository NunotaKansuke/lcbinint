from pathlib import Path
import json
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parent
r=pd.read_csv(p/'reference.csv')
read=lambda f:pd.read_csv(f,sep=r'\s+',header=None,float_precision='round_trip')
g=read(p/'gradient.tsv');b=read(p.parent/'adaptive_extreme_phase59/gradient.tsv')
assert len(g)==len(b)==220 and g[[0,1,2]].equals(b[[0,1,2]])
q=[8,10,12,14,16];j=[7,9,11,13,15]
mask=(g[q]!=b[q]).any(axis=1)
out={'reference_rows':len(r),'reference_usable':int((r.reference_usable==1).sum()),'reference_violations':int(((r.reference_usable==1)&(r.violation==1)).sum()),'unit':(p/'unit.txt').read_text().strip(),'jac_rows':len(g),'jac_quality_changed':int(mask.sum()),'jac_stop_changed':int((g[5]!=b[5]).sum()),'jac_max_scaled_difference':float((abs(g[j]-b[j])/np.maximum(1,abs(b[j]))).to_numpy().max()),'quality_changes':[{'baseline':b.loc[i].tolist(),'candidate':g.loc[i].tolist()} for i in g.index[mask]]}
(p/'validation.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item())+'\n')
