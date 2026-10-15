"""Cartesian state, tracing, budget recovery, and admission contracts."""
import argparse,hashlib,importlib.util,json,math,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 for key in list(os.environ):
  if key.startswith('LCBININT_'):os.environ.pop(key)
 native=next((ROOT/'build-adaptive-production').glob('_lcbinint*.so'));spec=importlib.util.spec_from_file_location('_lcbinint',native);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
 rows=json.loads((ROOT/'tests/diagnostics/results/recal2026/cartesian-final-redesign-20261015/cohort.json').read_text());checks=[]
 fields=['magnification','error_indicator','new_lattice_maps','component_grid_bins','components','history','estimated_target_met','stop_reason']
 def call(r,bins=16,budget=1000000,target=1e-3):
  return n.binary_adaptive_flux(float(r['time']),float(r['y']),s=float(r['s']),q=1/float(r['q']),rho=float(r['rho']),limb_darkening=n.LimbDarkening(float(r['c']),0),initial_bins=bins,max_lattice_maps=budget,relative_tolerance=target)
 def check(label,ok):
  checks.append(dict(check=label,passed=bool(ok)))
 selected=[next(r for r in rows if r['geom_id']==i)for i in ['408','777','1604']]
 prior=json.loads((ROOT/'tests/diagnostics/results/recal2026/cartesian-final-redesign-20261015/baseline-probe.json').read_text())
 eligible={r['id']for r in prior if r['target']==1e-4 and r.get('result',{}).get('estimated_target_met')and r['result']['new_lattice_maps']<20000}
 selected += [r for r in rows if r['profile']=='linear'and r['geom_id']not in ['1604']and r['geom_id']in eligible][:3]
 for r in selected:
  key=r['geom_id']
  for target in [1e-3,1e-4]:
   base=call(r,target=target);again=call(r,target=target);check(f'{key}:{target}:repeat',all(base[k]==again[k]for k in fields))
   check(f'{key}:{target}:flux-ledger',abs(sum(c[1]for c in base['components'])-base['magnification'])<=1e-11*max(1,abs(base['magnification'])))
   check(f'{key}:{target}:admission',not base['estimated_target_met']or(base['stop_reason']=='estimated_target_met'and math.isfinite(base['error_indicator'])and base['error_indicator']<=target*base['initial_magnification_scale']))
   for flag in ['LCBININT_DIAGNOSTIC_ROW_PANEL_TRACE','LCBININT_DIAGNOSTIC_ADAPTIVE_COST_TRACE']:
    os.environ[flag]='1'
    try:traced=call(r,target=target)
    finally:os.environ.pop(flag)
    check(f'{key}:{target}:{flag}',all(traced[k]==base[k]for k in fields))
   for budget in [1,256,1024]:
    try:
     limited=call(r,budget=budget,target=target);check(f'{key}:{target}:budget{budget}',limited['new_lattice_maps']<=budget and(not limited['estimated_target_met']or limited['error_indicator']<=target*limited['initial_magnification_scale']))
    except RuntimeError as exc:check(f'{key}:{target}:budget{budget}','budget' in str(exc))
    restored=call(r,target=target);check(f'{key}:{target}:restore{budget}',all(restored[k]==base[k]for k in fields))
 failures=[c for c in checks if not c['passed']];report=dict(native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),row_ratio=1,checks=checks,failures=failures,certified=False)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print('checks',len(checks),'failures',len(failures));return int(bool(failures))
if __name__=='__main__':raise SystemExit(main())
