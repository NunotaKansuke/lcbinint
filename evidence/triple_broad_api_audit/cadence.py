import os,sys,json,time,statistics,subprocess,collections
from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'audit.py').read_text().split("if '--worker'in sys.argv:")[0];exec(compile(s,str(P/'audit.py'),'exec'))
os.sched_setaffinity(0,{0});cases={c['name']:c for c in json.loads((P/'cases.json').read_text())}
if '--child'in sys.argv:
 c=json.loads(sys.argv[-1]);engine=c['engine'];model,p,times,call,_=setup(c,engine);samples=[]
 for i in range(8):
  a=time.thread_time_ns();z=call(p);ms=(time.thread_time_ns()-a)/1e6
  if i>=2:samples.append(ms)
 info=model.info(times,p,with_gradient=engine=='gradient')
 print(json.dumps(dict(condition=c,engine=engine,median_ms=statistics.median(samples),samples_ms=samples,routes=info.diagnostics['finite_source_method'].tolist(),status=info.status.tolist(),gradient_valid=info.gradient_valid.tolist()if engine=='gradient'else None)),flush=True);sys.exit()
report=dict(protocol='Supplementary native-only cadence diagnostic, separate fresh process per size/mode; two warmups plus six uninterrupted thread CPU samples. LD=.5, relative1e-3. No VBM speedup claim from this diagnostic. CPU0, one thread, no other numerical work concurrent; CPU frequency uncontrolled. 0.75rho is an existing source-code eligibility heuristic for contact event proposals, not an accuracy tolerance. Different N also changes sampled epochs, route mix and fixed cost amortization, so timing alone does not causally isolate that heuristic.',rows=[])
for name in ['crossing','grazing','close']:
 for n in [17,65,129,257]:
  for mode in ['value','gradient']:
   c=dict(cases[name],n=n,engine=mode,limb=.5,rtol=1e-3)
   z=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',json.dumps(c)],capture_output=True,text=True,check=True,timeout=45,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
   row=json.loads(z.stdout);row['step_over_rho']=(c['xmax']-c['xmin'])/(n-1)/c['rho'];row['milliseconds_per_epoch']=row['median_ms']/n;report['rows'].append(row);(P/'cadence.json').write_text(json.dumps(report,indent=2)+'\n');print(name,n,mode,round(row['median_ms'],3),flush=True)
