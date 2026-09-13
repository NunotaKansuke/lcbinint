import json,hashlib,subprocess
from pathlib import Path
import pandas as pd,numpy as np
p=Path(__file__).resolve().parent
ref=pd.read_csv(p/'reference_safety.csv')
grad0=pd.read_csv(p/'gradient_base.tsv',sep=r'\s+',header=None)
grad=pd.read_csv(p/'gradient_safety.tsv',sep=r'\s+',header=None)
gcols=[7,9,11,13,15];qcols=[8,10,12,14,16]
res=dict(reference_rows=len(ref),reference_usable=int((ref.reference_usable==1).sum()),reference_violations=int(((ref.reference_usable==1)&(ref.violation==1)).sum()),
 gradient_rows=len(grad),gradient_nonfinite=int((~np.isfinite(grad[gcols])).sum().sum()),
 gradient_quality_changes=int((grad[qcols]!=grad0[qcols]).sum().sum()),
 gradient_invalid_before=int((grad0[qcols]==3).sum().sum()),gradient_invalid_after=int((grad[qcols]==3).sum().sum()),
 gradient_max_scaled_delta=float((abs(grad[gcols]-grad0[gcols])/np.maximum(1,abs(grad0[gcols]))).max().max()),
 gradient_value_nonconverged=int((grad[4]!=1).sum()))
res.update(gradient_baseline_value_nonconverged=int((grad0[4]!=1).sum()),
 gradient_new_value_nonconverged=int(((grad[4]!=1)&(grad0[4]==1)).sum()),
 gradient_quality_downgrades=int(((grad0[qcols]==1)&(grad[qcols]==2)).sum().sum()),
 gradient_quality_upgrades=int(((grad0[qcols]==2)&(grad[qcols]==1)).sum().sum()))
aa=pd.read_csv(p/'rounded_v2.tsv',sep=r'\s+',comment='#',float_precision='round_trip')
bb=pd.read_csv(p/'tolerance_probe.tsv',sep=r'\s+',comment='#',float_precision='round_trip')
mm=aa.merge(bb,on=['case_id','profile','d_bin_index','epoch_index','target'],suffixes=('_rounded','_exact'))
res['rounded_input_v2_max_relative_delta']=float(abs(mm.warm_mu_rounded/mm.warm_mu_exact-1).max())
(p/'validation.json').write_text(json.dumps(res,indent=2)+'\n')
paths=[p/'ab.cpp',p/'base.tsv',p/'safety1.tsv',p/'initial15.tsv',p/'initial3.tsv',p/'k12.tsv',Path('evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv')]
meta={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
meta['head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
meta['compiler']=subprocess.check_output(['c++','--version'],text=True).splitlines()[0]
(p/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(res,indent=2))
