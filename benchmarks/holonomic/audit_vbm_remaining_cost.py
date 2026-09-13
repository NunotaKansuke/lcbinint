"""Match saved VBM kernel rows and audit remaining cost, not new timings."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--vbm',required=True);p.add_argument('--v2',required=True);p.add_argument('--output',required=True)
a=p.parse_args();source=json.loads(Path(a.vbm).read_text());rows=[]
for r in source['results']:
    for i,(sec,status) in enumerate(zip(r['vbm']['selected_seconds'],r['vbm']['selected_status'])):
        rows.append({**{k:r[k] for k in ['case_id','profile','d_bin_index','target','s','q','rho','x','y']},
                     'epoch_index':source['reference_indices'][i],'vbm_ms':None if sec is None else sec*1000,'vbm_status':status,
                     'vbm_reltol':r['vbm']['selected_reltol'][i]})
v=pd.DataFrame(rows);d=pd.read_csv(a.v2,sep=r'\s+',comment='#',float_precision='round_trip')
keys=['case_id','profile','d_bin_index','epoch_index','target']
assert not v.duplicated(keys).any() and not d.duplicated(keys).any()
m=d.merge(v,on=keys,how='left',validate='one_to_one',suffixes=('','_vbm'),indicator=True)
assert len(m)==14432 and (m['_merge']=='both').all()
for k in ['s','q','rho','x','y']:
    assert np.allclose(m[k],m[k+'_vbm'],rtol=2e-15,atol=0), k
assert (m.target==m.vbm_reltol).all()
assert (m.vbm_status=='completed').all() and np.isfinite(m.vbm_ms).all() and (m.vbm_ms>0).all()
assert (m.warm_ms>=m.warm_topology_ms).all()
m['warm_without_topology_ms']=m.warm_ms-m.warm_topology_ms
out={'rows':len(m),'sources':{'vbm':a.vbm,'v2':a.v2},'groups':{}}
for (tol,profile),g in m.groupby(['target','profile']):
    label=f'{tol:g}/{profile}';o={'rows':len(g),'vbm_p50_ms':float(g.vbm_ms.median()),'lanes':{}}
    for lane in ['cold','warm','radial','warm_without_topology','warm_physical']:
        t=g[lane+'_ms'];ratio=g.vbm_ms/t
        o['lanes'][lane]={'p50_ms':float(t.median()),'p90_ms':float(t.quantile(.9)),
          'paired_speedup_median':float(ratio.median()),'faster_rows':int((t<g.vbm_ms).sum())}
    out['groups'][label]=o
outdir=Path(a.output);outdir.mkdir(parents=True,exist_ok=True)
m[keys+['vbm_ms','cold_ms','warm_ms','radial_ms','warm_topology_ms','warm_without_topology_ms','warm_physical_ms']].to_csv(outdir/'matched.tsv',sep='\t',index=False)
(outdir/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
