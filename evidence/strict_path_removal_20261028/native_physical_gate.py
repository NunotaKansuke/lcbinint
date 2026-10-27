import sys,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'triple_coupled_optimization';root=P.parent.parent;build=root/'build-triple-native';sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable' not in str(f)];sys.path.insert(0,str(build));import lcbinint as native
assert Path(native._native.__file__).resolve().is_relative_to(build)
refs=json.loads((P.parent/'triple_ld_local_precision/gradient-recheck/finite-difference.json').read_text())['rows'];cache={};rows=[]
for ref in refs:
 for tol in [1e-3,1e-4]:
  key=ref['case'],ref['limb'],tol
  if key not in cache:
   X,Y,rho=(.1,.15,.15)if ref['case']=='contact'else(.1258549239677236,.06379371886791989,.3)
   out=native.triple_algebraic([0j,1+0j,-.4+.6j],[.72,.2,.08],complex(X,Y),rho,limb=ref['limb'],with_gradient=True,atol=0,rtol=tol)
   assert out['value_converged']and out['gradient_valid']and out['support_complete']and not out['smooth_source_used'],out
   cache[key]=dict(zip(out['gradient_parameters'],out['gradient']))
  value=cache[key][ref['parameter']];fd=ref['finite_difference_step_sweep'][-1]['fd'];budget=1e-4+2e-4*abs(fd);assert abs(value-fd)<=budget,(key,ref['parameter'],value,fd);rows.append(dict(case=ref['case'],limb=ref['limb'],rtol=tol,parameter=ref['parameter'],analytic=value,reference=fd,budget=budget))
module=Path(native._native.__file__);(P/(sys.argv[1] if len(sys.argv)>1 else 'native-first-order-physical.json')).write_text(json.dumps(dict(module=str(module),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),rows=rows),indent=2));print('PASS',len(rows),'physical components through rebuilt public API; maximum ratio',max(abs(r['analytic']-r['reference'])/r['budget']for r in rows))
