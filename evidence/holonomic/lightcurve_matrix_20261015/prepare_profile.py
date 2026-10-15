from benchmark import *
import numpy as np
np,l,VB=load();selected=[]
for name in ['broad_source','large_source','resonant','wide_caustic']:
 for limb in [0.,.5]:
  cid=f'{name}_ld{limb:g}_tol0.001';d=json.loads((OUT/(cid+'.json')).read_text());a=np.load(OUT/(cid+'.npz'));ix=np.where(a['integral'])[0];order=np.argsort(a['epoch_seconds'][ix]) if 'epoch_seconds' in a else np.argsort(a['nodes'][ix]);chosen=np.unique(np.r_[ix[np.linspace(0,len(ix)-1,min(5,len(ix))).astype(int)],ix[order[-5:]]]);p=d['params']
  c=l.LightCurve(options=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=.001,finite_source_reltol=.001,finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(limb));g=c.finite_source_geometry(a['times'],p)
  for k in chosen:selected.append([cid,int(k),float(a['times'][k]),float(g.source_x[k]),float(g.source_y[k]),p['rho'],float(g.mass_ratio[k]),p['s'],limb,.001,float(a['lc'][k])])
with (OUT/'profile_inputs.tsv').open('w') as f:
 for row in selected:f.write('\t'.join(map(str,row))+'\n')
print('profile epochs',len(selected))
