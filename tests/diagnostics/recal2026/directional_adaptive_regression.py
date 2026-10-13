"""Check unchanged square paths and incremental anisotropic grid history."""
import os,json,hashlib,argparse
from pathlib import Path
from .probe_build import activate

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output",default="directional-adaptive-v1");parser.add_argument("--audit",default="directional-adaptive-audit-v2");args=parser.parse_args()
 lc=activate();n=lc._lcbinint;root=Path('tests/diagnostics/results/recal2026');out=root/args.output
 os.environ.pop('LCBININT_DIAGNOSTIC_DIRECTIONAL_ADAPTIVE',None);os.environ.pop('LCBININT_DIAGNOSTIC_INTERIOR_BLOCKS',None)
 adaptive=json.loads((root/'adaptive-shared-flux-audit-v5/summary.json').read_text())['rows'];count=0
 for r in adaptive:
  i=r['input'];got=n.binary_adaptive_flux(i['x'],i['y'],s=i['s'],q=1/i['q'],rho=i['rho'],limb_darkening=n.LimbDarkening(r['c'],0),initial_bins=16,max_bins=512,absolute_tolerance=r['tolerance'])
  for key in ['magnification','error_indicator','new_lattice_maps','history','components','estimated_target_met']:
   assert got[key]==r[key],(key,i,r['c'])
  assert all(x==y for x,y in got['component_grid_bins']);count+=1
 fixed=json.loads((root/'adaptive-shared-fixed-final-check-v1/summary.json').read_text())['rows'];fixed_count=0
 for r in fixed:
  i=r['input'];opts=lc.Options(source_bins=r['nbin'],max_source_bins=r['nbin'],mode=1,point_source_threshold=0.,hexadecapole_threshold=0.,adaptive_hex_threshold=0.)
  got=n.binary_flux_estimate(i['x'],i['y'],s=i['s'],q=1/i['q'],rho=i['rho'],limb_darkening=n.LimbDarkening(r['c'],0),options=opts)
  assert got['magnification']==r['magnification'] and got['error_indicator']==r['error_indicator'],(i,r['c'],r['nbin'])
  fixed_count+=1
 os.environ['LCBININT_DIAGNOSTIC_DIRECTIONAL_ADAPTIVE']='1'
 directional=json.loads((root/args.audit/'summary.json').read_text())['rows']
 for r in directional:
  for _,bx,by,ax,ay,axis in r['refinement_grid_bins']:
   assert ax/bx in [1,2] and ay/by in [1,2] and (ax>bx or ay>by)
   assert max(ax,ay)<=512
 # Budget exhaustion retains prior flux; repeat with ample budget does not carry stale state.
 r=max(directional,key=lambda r:r['new_lattice_maps']);i=r['input'];kw=dict(s=i['s'],q=1/i['q'],rho=i['rho'],limb_darkening=n.LimbDarkening(r['c'],0),initial_bins=16,max_bins=512,absolute_tolerance=r['tolerance'])
 small=n.binary_adaptive_flux(i['x'],i['y'],**kw,max_lattice_maps=int(r['history'][0][2])+1)
 assert not small['estimated_target_met'] and small['stop_reason']=='evaluation_budget'
 assert small['magnification']>0
 full=n.binary_adaptive_flux(i['x'],i['y'],**kw)
 assert full['magnification']==r['magnification'] and full['estimated_target_met']
 result=dict(square_adaptive_exact_matches=count,fixed_exact_matches=fixed_count,directional_history_validated=len(directional),budget_fallback=small,repeat_full=full,native_sha256=hashlib.sha256(Path(n.__file__).read_bytes()).hexdigest())
 (out/'regression.json').write_text(json.dumps(result,indent=2));print(count,fixed_count,len(directional),'PASS',flush=True)
if __name__=='__main__':main()
