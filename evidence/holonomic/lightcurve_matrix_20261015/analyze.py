from pathlib import Path
import json,csv,hashlib,subprocess,sys,platform,os
import numpy as np
from benchmark import ROOT,OUT
rows=[];flat=[]
for c in json.loads((OUT/'cases.json').read_text()):
 r=json.loads((OUT/(c['id']+'.json')).read_text());rows.append(r)
 flat.append({k:r.get(k) for k in ['id','case','limb','tol','status','lc_ms','vbm_ms','speed_vbm_over_lc','relative_difference_max','relative_difference_p99','relative_difference_median','scaled_difference_max','scaled_difference_over_one','value_converged','fully_converged','integral_epochs','integral_support_certified','finite_values','nodes_total','nodes_max','gradient_ms_one_call','finite_gradient_components','plain_matches_info','peak_rss_kib']})
with (OUT/'results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=flat[0].keys(),lineterminator='\n');w.writeheader();w.writerows(flat)
refs=[]
for f in sorted(OUT.glob('*_reference.json')):
 r=json.loads(f.read_text())
 if r.get('status')=='failed':continue
 ref=r['annulus'][-1]['mu'];previous=r['annulus'][-2]['mu'];refs.append(dict(id=r['id'],time=r['worst_time'],annulus_reference=ref,reference_relative_change=abs(ref-previous)/abs(ref),lc_relative_difference_from_annulus=abs(r['lc']-ref)/abs(ref),vbm_relative_difference_from_annulus=abs(r['vbm']-ref)/abs(ref),vbm_tight_relative_difference_from_annulus=abs(r['vbm_direct'][-1]['mu']-ref)/abs(ref),native_uniform_outer_relative_difference=abs(r['native_uniform_outer']['mu']-r['vbm_uniform_outer'])/abs(r['vbm_uniform_outer'])))
with (OUT/'reference_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=refs[0].keys(),lineterminator='\n');w.writeheader();w.writerows(refs)
summary=dict(cases=len(rows),epochs=sum(r['n_times'] for r in rows),failed=[r['id'] for r in rows if r['status']!='ok'],finite_values=sum(r['finite_values'] for r in rows),value_converged=sum(r['value_converged'] for r in rows),fully_converged=sum(r['fully_converged'] for r in rows),integral_epochs=sum(r['integral_epochs'] for r in rows),integral_support_certified=sum(r['integral_support_certified'] for r in rows),finite_gradient_components=sum(r['finite_gradient_components'] for r in rows),gradient_components_expected=len(rows)*401*7,lc_faster_cases=sum(r['speed_vbm_over_lc']>1 for r in rows),median_speed_ratio=float(np.median([r['speed_vbm_over_lc'] for r in rows])),pairwise_difference_over_request_cases=[r['id'] for r in rows if r['scaled_difference_max']>1],plain_call_matches_info=all(r['plain_matches_info'] for r in rows),peak_worker_rss_mib=max(r['peak_rss_kib'] for r in rows)/1024,annulus_checks=refs)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
module=Path(rows[0]['native_file']);grad=next((ROOT/'build-algebraic-boundary/fused-gradient/python').glob('*.so'))
import VBMicrolensing,importlib.metadata
vbmodule=next(Path(VBMicrolensing.__file__).parent.glob('*.so'))
files=[module,grad,vbmodule,ROOT/'benchmarks/holonomic/fused_gradient/fused_k.hpp']+list(OUT.glob('*.py'))+list(OUT.glob('*.cpp'))+list(OUT.glob('*.npz'))
(OUT/'provenance.json').write_text(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),python=sys.version,platform=platform.platform(),cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),None),vbmicrolensing_version=importlib.metadata.version('VBMicrolensing'),worker_cpu=min(os.sched_getaffinity(0)),threads=1,repeats=3,worker_timeout_seconds=40,definition='median complete public curve.info()/BinaryLightCurve calls, constructor and parameter conversion excluded, one cold call then alternating order; gradients separately one call; ordinary curve() checked separately',sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='annulus_checks'},indent=2))
