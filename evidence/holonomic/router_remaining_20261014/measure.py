"""Full-curve native epoch timing, plus VBM subsets labelled by lcbinint route."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import sys, json, time, statistics, math, subprocess
from pathlib import Path
import numpy as np
import VBMicrolensing
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root/'build-seeded-events/python-api'))
import _lcbinint as native
rows=[]
for label,u0,alpha,lo,hi in [('ordinary',.3,0.,-1.,1.),('wide_planetary',.03,-.08,-.2,.8)]:
 t=np.linspace(lo,hi,201); p=dict(t0=0.,tE=1.,u0=u0,alpha=alpha,s=1.2,q=.01,rho=.003)
 lc=native.LightCurve(lens='binary',param_type='vbm',nbin='auto',ld_c=0.,finite_source_tol=.001,finite_source_reltol=.001)
 vb=VBMicrolensing.VBMicrolensing(); vb.Tol=vb.RelTol=.001; vb.a1=0.
 vp=[math.log(1.2),math.log(.01),u0,alpha,math.log(.003),0.,0.]; vt=t.tolist()
 d=lc._evaluate_holonomic(t,p,.001,.001); vb.BinaryLightCurve(vp,vt)
 methods=np.array(d['finite_source_method_name']); groups={m:np.flatnonzero(methods==m) for m in sorted(set(methods))}
 samples=[]
 for k in range(41):
  entry={}
  for route in (['lc','vb'] if k%2 else ['vb','lc']):
   start=time.perf_counter()
   if route=='lc': d=lc._evaluate_holonomic(t,p,.001,.001)
   else: vb.BinaryLightCurve(vp,vt)
   entry[route+'_ms']=1000*(time.perf_counter()-start)
  sec=np.array(d['seconds']); entry['epoch_sum_ms']=1000*sum(sec)
  entry['by_method_ms']={m:float(1000*sum(sec[ix])) for m,ix in groups.items()}
  entry['first_epoch_ms']=float(1000*sec[0])
  warm=lc._evaluate_preplanned(t,p,[8]*201,[1]*201,False)
  np.testing.assert_allclose(warm['magnification'],d['mu'],rtol=2e-12,atol=2e-12)
  np.testing.assert_array_equal(warm['method'],d['finite_source_method'])
  ws=np.array(warm['seconds']); entry['warm_epoch_sum_ms']=float(1000*sum(ws))
  entry['warm_first_epoch_ms']=float(1000*ws[0])
  entry['warm_by_method_ms']={m:float(1000*sum(ws[ix])) for m,ix in groups.items()}
  samples.append(entry)
 subset={}
 for m,ix in groups.items():
  ts=t[ix].tolist(); vals=[]; vb.BinaryLightCurve(vp,ts)
  for k in range(41):
   start=time.perf_counter(); vb.BinaryLightCurve(vp,ts); vals.append(1000*(time.perf_counter()-start))
  subset[m]=dict(median_ms=statistics.median(vals),samples_ms=vals)
 med=lambda key:statistics.median([x[key] for x in samples])
 row=dict(case=label,params=p,ld=0.,atol=.001,rtol=.001,counts={m:len(ix) for m,ix in groups.items()},lc_ms=med('lc_ms'),vbm_ms=med('vb_ms'),epoch_sum_ms=med('epoch_sum_ms'),first_epoch_ms=med('first_epoch_ms'),by_method_ms={m:statistics.median([x['by_method_ms'][m] for x in samples]) for m in groups},vbm_route_subsets=subset,samples=samples)
 row.update(warm_epoch_sum_ms=med('warm_epoch_sum_ms'),warm_first_epoch_ms=med('warm_first_epoch_ms'),warm_by_method_ms={m:statistics.median([x['warm_by_method_ms'][m] for x in samples]) for m in groups})
 rows.append(row)
 print(json.dumps({k:v for k,v in row.items() if k not in ['samples','vbm_route_subsets']}),flush=True)
 print('VBM subsets:', {m:v['median_ms'] for m,v in subset.items()},flush=True)
out=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),native=native.__file__,vbm_version=VBMicrolensing.__version__,repetitions=41,rows=rows,notes='Native groups timed within one uninterrupted full curve, including preparation in the epoch that incurs it. VBM subsets are separate BinaryLightCurve calls at epochs classified by lcbinint, not VBM internal routes; they do not sum to a full-curve decomposition.')
Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2)+'\n')
