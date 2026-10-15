from pathlib import Path
import json,hashlib,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
matrix=json.loads((OUT/'matrix.json').read_text());sweep=json.loads((OUT/'sweep_gradients.json').read_text());timings=[]
for f in sorted(OUT.glob('*_after.json')):
 after=json.loads(f.read_text());before=json.loads(Path(str(f).replace('_after.json','_before.json')).read_text());timings.append(dict(id=after['id'],before_ms=before['median_ms'],after_ms=after['median_ms'],ratio=after['median_ms']/before['median_ms']))
c=np.load(OUT/'case_033_gradient.npz');initial=np.load(ROOT/'evidence/holonomic/stress_20261015/case_033.npz')
summary=dict(matrix_configurations=len(matrix),matrix_epochs=len(matrix)*401,matrix_finite_gradients=sum(r['finite'] for r in matrix),max_value_api_change=max(r['mu_relative_change'] for r in matrix),changed_value_fields=[r for r in matrix if r['changed_value_fields']],stress_configurations=len(sweep),stress_finite_gradients=sum(r['finite'] for r in sweep),stress_invalid_cases=[r for r in sweep if r['finite']!=231],stress_value_converged=sum(int(np.asarray(np.load(OUT/(r['id']+'_gradient.npz'))['converged']).sum()) for r in sweep),case033_rho_before=initial['gradient'][[8,24],4].tolist(),case033_rho_after=c['gradient'][[8,24],4].tolist(),timing=timings)
assert len(matrix)==60 and summary['matrix_finite_gradients']==168420 and summary['max_value_api_change']==0 and not summary['changed_value_fields']
assert len(sweep)==120 and summary['stress_finite_gradients']==27720 and not summary['stress_invalid_cases'] and summary['stress_value_converged']==3960
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
files=['python/bind_holonomic_fused.cpp','src/lcbinint/magnification/binary_caustic_enclosure.hpp','src/lcbinint/magnification/holonomic/source_disk_gradient.hpp','src/lcbinint/magnification/holonomic/fused_gradient.cpp','src/lcbinint/magnification/holonomic/fused_gradient.hpp','benchmarks/holonomic/fused_gradient/fused_geometry.hpp','benchmarks/holonomic/fused_gradient/fused_geometry_exact.hpp','benchmarks/holonomic/fused_gradient/fused_phi.inc','benchmarks/holonomic/fused_gradient/fused_gaussian.hpp','benchmarks/holonomic/fused_gradient/fused_motion.hpp','benchmarks/holonomic/fused_gradient/build_geometry_overlay.py','benchmarks/holonomic/fused_gradient/build_material_overlay.py','benchmarks/holonomic/fused_gradient/fused_k.hpp','src/lcbinint/magnification/finite_source_magnifier.cpp','src/lcbinint/magnification/finite_source_magnifier.hpp','src/lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp']
for name in files:
 target=OUT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
mods=list((ROOT/'build-algebraic-boundary/python-api/lcbinint').glob('*.so'))+list((ROOT/'build-algebraic-boundary/fused-gradient/python').glob('*.so'))
(OUT/'provenance.json').write_text(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},module_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in mods},timing='one pinned allowed CPU, warmup then median of three 401-epoch gradient calls, BLAS/OpenMP one thread, constructors excluded',baseline_module_sha256=hashlib.sha256((OUT/'before.so').read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='timing'},indent=2))
