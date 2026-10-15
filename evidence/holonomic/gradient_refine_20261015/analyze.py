from pathlib import Path
from collections import defaultdict
import numpy as np,csv,json,hashlib,subprocess,sys
root=Path.cwd();out=root/'evidence/holonomic/gradient_refine_20261015';old=root/'evidence/holonomic/gradient_screen_20261015'
report={'tol':1e-3,'reltol':1e-3,'comparison':{},'timings':json.loads((out/'timing_screen.json').read_text()),'matrix':json.loads((out/'matrix_comparison.json').read_text())}
for case in ['historical','current','ordinary','scan']:
 prefix='scan' if case=='scan' else 'curve_'+case
 a=np.load(out/(prefix+'_screen.npz'));b=np.load(out/(prefix+'_screen_off.npz'));prev=np.load(old/(prefix+'_screen.npz'))
 report['comparison'][case]={'mu_on_off_bitwise':bool(np.array_equal(a['mu'],b['mu'],equal_nan=True)),'grad_on_off_bitwise':bool(np.array_equal(a['grad_mu'],b['grad_mu'],equal_nan=True)),'mu_previous_bitwise':bool(np.array_equal(a['mu'],prev['mu'],equal_nan=True)),'changed_gradient_epochs':int(np.sum(np.any(a['grad_mu']!=prev['grad_mu'],axis=1))),'finite':int(np.isfinite(a['grad_mu']).sum()),'components':a['grad_mu'].size,'value_converged':int(a['value_converged'].sum())}
refs=defaultdict(list)
for r in csv.DictReader((root/'evidence/holonomic/gradient_accuracy_20261015/reference.csv').open()):refs[float(r['t']),r['param']].append(float(r['vbm_fd']))
rows=[]
for variant,directory in [('previous',old),('final',out)]:
 data=np.load(directory/'scan_screen.npz')
 for t in [-.72,-.704,-.7,-.696,-.68,.1,.78]:
  k=np.argmin(abs(data['times']-t))
  for j,name in [(0,'t0'),(1,'u0'),(4,'rho'),(5,'q'),(6,'s')]:
   rs=refs[t,name];ref=float(np.median(rs));v=float(data['grad_mu'][k,j]);rows.append(dict(variant=variant,t=t,param=name,value=v,reference=ref,reference_min=min(rs),reference_max=max(rs),relative_error=abs(v-ref)/abs(ref)))
with (out/'accuracy.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
report['checks']={n:(out/n).read_text().strip() for n in ['bounds_check.txt','radius_check.txt']}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
paths=[root/'benchmarks/holonomic/fused_gradient'/n for n in ['fused_geometry.hpp','fused_geometry_screen.hpp','build_geometry_overlay.py','fused_geometry_pair.inc','fused_geometry_exterior.inc','build_geometry.sh']]+list(out.glob('*.py'))+list(out.glob('*.cpp'))+list(out.glob('*.sh'))+list(out.glob('*.npz'))+list((root/'build-algebraic-boundary/gradient-probe/geometry-python').glob('*.so'))
(out/'provenance.json').write_text(json.dumps({'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),'python':sys.version,'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')
print(json.dumps(report,indent=2))
assert all(r['mu_on_off_bitwise'] and r['grad_on_off_bitwise'] and r['mu_previous_bitwise'] for r in report['comparison'].values())
assert not report['matrix']['differences']
