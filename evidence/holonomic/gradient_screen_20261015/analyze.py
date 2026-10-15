from pathlib import Path
from collections import defaultdict
import csv,json,hashlib,platform,subprocess,sys
import numpy as np
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
report={};timings={}
for variant in ['original','full','screen_off','screen']:
 timings[variant]=json.loads((out/f'timing_{variant}.json').read_text())
report['api_timings']=timings
invariants={}
for case in ['historical','current','ordinary']:
 a=np.load(out/f'curve_{case}_screen.npz');b=np.load(out/f'curve_{case}_screen_off.npz')
 invariants[case]=dict(mu_bitwise_equal=bool(np.array_equal(a['mu'],b['mu'],equal_nan=True)),gradient_bitwise_equal=bool(np.array_equal(a['grad_mu'],b['grad_mu'],equal_nan=True)),finite_gradient_components=int(np.isfinite(a['grad_mu']).sum()),value_converged=int(np.sum(a['value_converged'])))
a=np.load(out/'scan_screen.npz');b=np.load(out/'scan_screen_off.npz')
invariants['scan89']=dict(mu_bitwise_equal=bool(np.array_equal(a['mu'],b['mu'],equal_nan=True)),gradient_bitwise_equal=bool(np.array_equal(a['grad_mu'],b['grad_mu'],equal_nan=True)),finite_gradient_components=int(np.isfinite(a['grad_mu']).sum()),invalid_epochs=a['times'][~np.all(np.isfinite(a['grad_mu']),axis=1)].tolist())
report['screen_on_off']=invariants
report['matrix60']=json.loads((out/'matrix_comparison.json').read_text())
profiles={}
for case in ['historical','current','ordinary']:
 rows=[r for r in csv.DictReader((out/f'profile_{case}_screen.tsv').open(),delimiter='\t') if r['j']=='0']
 profiles[case]=dict(epochs=len(rows),ordinary_epochs=sum(r['fine']=='0' and r['special']=='0' for r in rows),detailed_detector_epochs=sum(r['fine']=='1' for r in rows),local_replay_epochs=sum(r['special']=='1' for r in rows),counters=(out/f'profile_{case}_screen.log').read_text().splitlines()[-1])
report['profile']=profiles
refs=defaultdict(list)
for r in csv.DictReader((root/'evidence/holonomic/gradient_accuracy_20261015/reference.csv').open()):refs[float(r['t']),r['param']].append(float(r['vbm_fd']))
rows=[]
for variant in ['original','full','screen']:
 data=np.load(out/f'scan_{variant}.npz')
 for t in [-.72,-.704,-.7,-.696,-.68,.1,.78]:
  k=np.argmin(abs(data['times']-t))
  for j,name in [(0,'t0'),(1,'u0'),(4,'rho'),(5,'q'),(6,'s')]:
   reference=refs[t,name];median=float(np.median(reference));value=float(data['grad_mu'][k,j])
   rows.append(dict(variant=variant,t=t,param=name,value=value,reference=median,reference_min=min(reference),reference_max=max(reference),abs_error=abs(value-median),relative_error=abs(value-median)/abs(median) if median else None))
with (out/'accuracy.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
report['bounds_check']=(out/'bounds_check.txt').read_text().strip()
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
paths=[root/'benchmarks/holonomic/fused_gradient'/n for n in ['fused_geometry.hpp','fused_geometry_screen.hpp','fused_geometry_exterior.inc','fused_geometry_pair.inc','build_geometry_overlay.py','build_geometry.sh']]
paths += list(out.glob('*.py'))+list(out.glob('*.npz'))+list(out.glob('*.tsv'))+list(out.glob('*.json'))
paths += list((root/'build-algebraic-boundary/gradient-probe/geometry-python').glob('*.so'))+list((root/'build-algebraic-boundary/gradient-probe/geometry-before-screen-python').glob('*.so'))
paths += [root/'build-algebraic-boundary/python-api/liblcbinint_magnification.a',root/'build-algebraic-boundary/python-api/liblcbinint_lightcurve.a']
provenance=dict(checkout=str(root),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),python=sys.version,platform=platform.platform(),tol=1e-3,reltol=1e-3,timing='one pinned allowed CPU; -O2; API seed preparation and mapping included; warmup then five repeats; fresh seeder per curve; no persistent certificate cache',sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file() and p.name!='provenance.json'})
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(dict(on_off=invariants,profile=profiles,bounds=report['bounds_check']),indent=2))
assert all(r['mu_bitwise_equal'] and r['gradient_bitwise_equal'] for r in invariants.values())
assert not report['matrix60']['differences']
