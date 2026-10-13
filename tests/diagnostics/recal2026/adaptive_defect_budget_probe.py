"""Budget exhaustion and reported component accounting for corrected flux."""
import argparse,json,os,math
from pathlib import Path
from .probe_build import activate
ROOT=Path('tests/diagnostics/results/recal2026')
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='adaptive-defect-correction-v8');a=p.parse_args();out=ROOT/a.output
 source=json.loads((out/'accuracy.json').read_text());n=activate()._lcbinint
 for k in ('DIRECTIONAL_ADAPTIVE','LOCAL_ROW_INTERPOLATION','LOCAL_BOUNDARY_STRIPS','LOCAL_VERTICAL_CAPS','DEFECT_CORRECTION','ADAPTIVE_COST_TRACE'):os.environ['LCBININT_DIAGNOSTIC_'+k]='1'
 checks=[]
 for c in (0,.2,.5,.8,1):
  candidates=[r for r in source if r['c']==c and r['tolerance']==1e-4];r=max(candidates,key=lambda r:r['component_refinements']);i=r['input']
  kw=dict(s=i['s'],q=1/i['q'],rho=i['rho'],limb_darkening=n.LimbDarkening(c,0),initial_bins=16,max_bins=512,absolute_tolerance=1e-4)
  full=n.binary_adaptive_flux(i['x'],i['y'],**kw,max_lattice_maps=1000000)
  for k in ('magnification','error_indicator','components','history','new_lattice_maps','applied_flux_correction'):assert full[k]==r[k]
  assert math.isclose(sum(v[1] for v in full['components']),full['magnification'],rel_tol=5e-15)
  assert math.isclose(sum(abs(v[2])+v[3] for v in full['components']),full['error_indicator'],rel_tol=5e-15)
  for budget in sorted({int(v[2])+1 for v in full['history'][:-1]}):
   g=n.binary_adaptive_flux(i['x'],i['y'],**kw,max_lattice_maps=budget)
   assert not g['estimated_target_met'] and g['stop_reason']=='evaluation_budget'
   assert g['new_lattice_maps']<=budget and g['magnification']>0
   assert math.isclose(g['magnification'],g['history_magnification'][-1],rel_tol=5e-16)
   assert math.isclose(sum(v[1] for v in g['components']),g['magnification'],rel_tol=5e-15)
   assert math.isclose(g['raw_magnification']+g['applied_flux_correction'],g['magnification'],rel_tol=5e-15)
   checks.append(dict(c=c,budget=budget,stop_reason=g['stop_reason'],magnification=g['magnification'],applied_flux_correction=g['applied_flux_correction'],cap_trials=g['cap_trials'],cap_adopted=g['cap_adopted']))
  repeat=n.binary_adaptive_flux(i['x'],i['y'],**kw,max_lattice_maps=1000000)
  for k in ('magnification','error_indicator','components','history','new_lattice_maps','applied_flux_correction'):assert full[k]==repeat[k]
 # Nonlinear LD follows the previous route even when correction was requested.
 r=source[0];i=r['input'];kw=dict(s=i['s'],q=1/i['q'],rho=i['rho'],limb_darkening=n.LimbDarkening(.3,.2),initial_bins=16,max_bins=512,absolute_tolerance=1e-3)
 def rejected():
  try:n.binary_adaptive_flux(i['x'],i['y'],**kw)
  except ValueError as exc:return str(exc)
  raise AssertionError('nonlinear LD must retain existing API rejection')
 before=rejected();os.environ.pop('LCBININT_DIAGNOSTIC_DEFECT_CORRECTION');assert rejected()==before
 (out/'correction-budget-regression.json').write_text(json.dumps(dict(checks=checks,count=len(checks),component_accounting=True,nonlinear_ld_rejection_preserved=True),indent=2));print(len(checks),'budget/accounting/repeat checks PASS; nonlinear LD rejection preserved')
if __name__=='__main__':main()
