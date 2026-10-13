import importlib.util,json,os,math
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[4];base=p.parent/'initial8-local-repair-20261014'
spec=importlib.util.spec_from_file_location('_lcbinint',next((root/'build-adaptive-production').glob('_lcbinint*.so')));lc=importlib.util.module_from_spec(spec);spec.loader.exec_module(lc)
data={r['geom_id']:r for r in json.loads((base/'development.json').read_text())};out=[]
for key in ['1509','2078','3847','2895','4229','4844']:
 r=data[key]
 for tol in [1e-3,1e-4]:
  kwargs=dict(s=float(r['s']),q=1/float(r['q']),rho=float(r['rho']),limb_darkening=lc.LimbDarkening(float(r['c']),0),initial_bins=8,max_bins=512,absolute_tolerance=1e-4,relative_tolerance=tol,max_lattice_maps=10000000)
  a=lc.binary_adaptive_flux(float(r['time']),float(r['y']),**kwargs)
  os.environ['LCBININT_DIAGNOSTIC_AUDIT_INTERPOLATED_ROWS']='1';os.environ['LCBININT_DIAGNOSTIC_AUDIT_BOUNDARY_STRIPS']='1'
  b=lc.binary_adaptive_flux(float(r['time']),float(r['y']),**kwargs)
  del os.environ['LCBININT_DIAGNOSTIC_AUDIT_INTERPOLATED_ROWS'];del os.environ['LCBININT_DIAGNOSTIC_AUDIT_BOUNDARY_STRIPS']
  assert a['magnification']==b['magnification'] and a['estimated_target_met']==b['estimated_target_met']
  assert b['strip_audit_gap_rows']==0 and b['interpolated_audit_gap_rows']==0,(key,tol)
  out.append(dict(id=key,tolerance=tol,actual_relative_error=abs(b['magnification']/float(r['reference'])-1),**{k:b[k] for k in ['strip_audit_rows','strip_audit_max_difference','interpolated_audit_rows','interpolated_audit_max_difference','estimated_target_met','interpolated_rows','retained_boundary_strip_rows']}))
(p/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print('audited calls',len(out),'row scans',sum(r['strip_audit_rows']+r['interpolated_audit_rows'] for r in out))
