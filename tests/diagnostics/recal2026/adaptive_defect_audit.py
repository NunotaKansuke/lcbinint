"""Same-lattice defect correction; unchanged legacy route and independent errors."""
import os,json,time,statistics,hashlib,argparse
from pathlib import Path
from .probe_build import activate
ROOT=Path('tests/diagnostics/results/recal2026')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='adaptive-defect-correction-v1');p.add_argument('--timing',action='store_true');p.add_argument('--fresh',type=Path,action='append',default=[]);args=p.parse_args();out=ROOT/args.output;out.mkdir(exist_ok=True)
 if (out/'accuracy.json').exists():raise RuntimeError('choose a new output')
 for k in list(os.environ):
  if k.startswith('LCBININT_DIAGNOSTIC_'):os.environ.pop(k)
 for k in ('DIRECTIONAL_ADAPTIVE','LOCAL_ROW_INTERPOLATION','LOCAL_BOUNDARY_STRIPS','LOCAL_VERTICAL_CAPS'):os.environ['LCBININT_DIAGNOSTIC_'+k]='1'
 lc=activate();n=lc._lcbinint
 source=json.loads((ROOT/'adaptive-cache-and-transfer-v3/accuracy.json').read_text())
 oracle=json.loads((ROOT/'directional-adaptive-cost-v2/summary.json').read_text())['rows']
 def identity(r):return tuple(r['input'][k] for k in ('s','q','x','y','rho'))+(r['c'],r['tolerance'])
 bins={identity(r):r['selected_bins'] for r in oracle}
 options={k:lc.Options(source_bins=k,max_source_bins=k,mode=1,point_source_threshold=0.,hexadecapole_threshold=0.,adaptive_hex_threshold=0.) for k in (16,32,64,128,256,512)}
 def fixed(r):
  i=r['input'];return n.binary_ray_shooting(i['x'],i['y'],s=i['s'],q=1/i['q'],rho=i['rho'],limb_darkening=n.LimbDarkening(r['c'],0),options=options[bins[identity(r)]])
 for fresh_path in args.fresh:
  for row in json.loads(fresh_path.read_text())['rows']:
   source.append(dict(source='new_geometry',input=row['input'],c=row['c'],tolerance=row['tolerance'],reference=row['reference'],expected_legacy_error=row['baseline_error']))
 def call(r,corrected):
  if corrected:os.environ['LCBININT_DIAGNOSTIC_DEFECT_CORRECTION']='1'
  else:os.environ.pop('LCBININT_DIAGNOSTIC_DEFECT_CORRECTION',None)
  i=r['input'];return n.binary_adaptive_flux(i['x'],i['y'],s=i['s'],q=1/i['q'],rho=i['rho'],limb_darkening=n.LimbDarkening(r['c'],0),initial_bins=16,max_bins=512,absolute_tolerance=r['tolerance'],max_lattice_maps=1000000)
 rows=[]
 for r in source:
  old=call(r,False)
  if r['source']=='new_geometry':assert abs(old['magnification']-r['reference'])==r['expected_legacy_error']
  else:
   for k in ('magnification','error_indicator','history','components','new_lattice_maps','reused_lattice_maps','cap_trials','cap_adopted'):assert old[k]==r[k],(k,r['input'])
  new=call(r,True)
  if r['source']=='main':assert abs(fixed(r)-r['reference'])<=r['tolerance']
  rows.append(dict(source=r['source'],input=r['input'],c=r['c'],tolerance=r['tolerance'],reference=r['reference'],actual_error=abs(new['magnification']-r['reference']),baseline_actual_error=abs(old['magnification']-r['reference']),baseline_steps=old['component_refinements'],baseline_maps=old['new_lattice_maps'],**new))
 (out/'accuracy.json').write_text(json.dumps(rows,indent=2))
 groups=[]
 for tol in (1e-3,1e-4):
  for c in sorted({r['c'] for r in rows}):
   rs=[r for r in rows if r['c']==c and r['tolerance']==tol];g=dict(tolerance=tol,c=c,count=len(rs),actual_pass=sum(r['actual_error']<=tol for r in rs),estimated_met=sum(r['estimated_target_met'] for r in rs),false_accept=sum(r['estimated_target_met'] and r['actual_error']>tol for r in rs),max_actual_error=max(r['actual_error'] for r in rs),steps=sum(r['component_refinements'] for r in rs),baseline_steps=sum(r['baseline_steps'] for r in rs),maps=sum(r['new_lattice_maps'] for r in rs),baseline_maps=sum(r['baseline_maps'] for r in rs))
   if args.timing and any(r['source']=='main' and r['c']==c and r['tolerance']==tol for r in source):
    selected=[r for r in source if r['source']=='main' and r['c']==c and r['tolerance']==tol];samples={'old':[],'new':[],'fixed':[]}
    geometry=[(r['input']['s'],r['input']['q']) for r in selected];assert all(a!=b for a,b in zip(geometry,geometry[1:]+geometry[:1]))
    modes=['old','new','fixed']
    for repeat in range(21):
     order=modes[repeat%3:]+modes[:repeat%3]
     if repeat%2:order=order[::-1]
     for mode in order:
      start=time.perf_counter()
      for r in selected:
       if mode=='fixed':fixed(r)
       else:call(r,mode=='new')
      samples[mode].append(time.perf_counter()-start)
    med={k:statistics.median(v) for k,v in samples.items()};g.update(samples=samples,medians=med,new_over_old=med['new']/med['old'],new_over_fixed=med['new']/med['fixed'],old_over_fixed=med['old']/med['fixed'])
   groups.append(g);print(json.dumps(g),flush=True)
 summary=dict(groups=groups,count=len(rows),actual_pass=sum(r['actual_error']<=r['tolerance'] for r in rows),false_accept=sum(r['estimated_target_met'] and r['actual_error']>r['tolerance'] for r in rows),native_sha256=hashlib.sha256(Path(n.__file__).read_bytes()).hexdigest(),reference_sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.fresh],certified=False,complete_error_model=False)
 (out/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
