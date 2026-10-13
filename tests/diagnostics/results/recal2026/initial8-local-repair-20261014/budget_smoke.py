import importlib.util,json,math,os
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[4]
module=next((root/'build-adaptive-production').glob('_lcbinint*.so'))
spec=importlib.util.spec_from_file_location('_lcbinint',module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
lc=m
inputs={r['geom_id']:r for c in ['development','holdout','prospective'] for r in json.loads((p/f'{c}.json').read_text())}
results=[]
for key in ['2297','5230','5312','3994']:
 r=inputs[key]
 for budget in [64,256,4096,1000000]:
  for bins in [8,16]:
   params=dict(s=float(r['s']),q=1/float(r['q']),rho=float(r['rho']),limb_darkening=lc.LimbDarkening(float(r['c']),0),initial_bins=bins,max_bins=512,absolute_tolerance=1e-4,relative_tolerance=1e-4,max_lattice_maps=budget)
   try:
    v=lc.binary_adaptive_flux(float(r['time']),float(r['y']),**params)
    assert v['new_lattice_maps']<=budget
    assert not v['certified'] and not v['complete_error_model']
    if v['stop_reason']=='evaluation_budget':assert not v['estimated_target_met']
    results.append(dict(id=key,bins=bins,budget=budget,mu=v['magnification'],met=v['estimated_target_met'],stop=v['stop_reason'],maps=v['new_lattice_maps']))
   except RuntimeError as e:
    assert 'budget exhausted' in str(e),str(e)
    results.append(dict(id=key,bins=bins,budget=budget,exception=str(e)))
 # A full call after exhausted budgets also checks restoration of request state.
 params['initial_bins']=8;params['max_lattice_maps']=1000000
 v=lc.binary_adaptive_flux(float(r['time']),float(r['y']),**params)
 assert math.isfinite(v['magnification'])
 results.append(dict(id=key,post_exception_restored=True,mu=v['magnification']))
(p/'budget-smoke.json').write_text(json.dumps(results,indent=2)+'\n');print('budget/scope checks passed',len(results))
