import ctypes as C,pathlib,json,sys,os,subprocess,statistics,math
P=pathlib.Path(__file__).resolve().parent
class Out(C.Structure):_fields_=[('mu',C.c_double),('error',C.c_double),('seconds',C.c_double),('samples',C.c_int),('annuli',C.c_int),('converged',C.c_int)]
if len(sys.argv)>1:
 import VBMicrolensing
 e=sys.argv[1];init=json.loads(sys.stdin.readline());lib=C.CDLL(str(P/('kernel-before.so' if e=='before' else 'kernel.so')));prefix='algebraic' if e in ['before','algebraic'] else 'vbm';c=init['case'];g=(C.c_double*9)(*[v for z,m in zip(c['lenses'],c['masses']) for v in (*z,m)])
 create=getattr(lib,prefix+'_create');create.restype=C.c_void_p;create.argtypes=[C.POINTER(C.c_double)] if prefix=='algebraic' else [C.c_int,C.POINTER(C.c_double)];handle=create(g) if prefix=='algebraic' else create({'Singlepoly':0,'Multipoly':1,'Nopoly':2}[e],g)
 fn=getattr(lib,'algebraic_stages' if prefix=='algebraic' else 'vbm_point');fn.argtypes=([C.c_void_p]+[C.c_double]*5+[C.POINTER(Out),C.POINTER(C.c_double)]) if prefix=='algebraic' else ([C.c_void_p]+[C.c_double]*5+[C.c_int,C.POINTER(Out)]);destroy=getattr(lib,prefix+'_destroy');destroy.argtypes=[C.c_void_p]
 for line in sys.stdin:
  source=json.loads(line);o=Out();cost=(C.c_double*6)()
  if prefix=='algebraic':fn(handle,*source,c['rho'],init['limb'],init['rtol'],C.byref(o),cost)
  else:fn(handle,*source,c['rho'],init['limb'],init['rtol'],int(init['limb']!=0),C.byref(o))
  r={k:getattr(o,k) for k,_ in Out._fields_}
  if prefix=='algebraic':r['stages']=list(cost)
  print(json.dumps({k:None if isinstance(v,float) and not math.isfinite(v) else v for k,v in r.items()}),flush=True)
 destroy(handle);sys.exit()
os.sched_setaffinity(0,{0})
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
engines=['before','algebraic','Singlepoly','Multipoly','Nopoly'];report=dict(status='RUNNING',protocol='CPU0, one active worker, persistent geometry, 5 original-point warmups per engine, 2 cycles of 8 identical fresh coordinates, rotated and reversed engine order',rows=[],measurements=[],warmups=[])
def save():(P/'gap-stages.json').write_text(json.dumps(report,indent=2)+'\n')
for f in sorted((P/'conditions').glob('*-algebraic-*.json')):
 c=json.loads(f.read_text())
 if (c['name'],c['limb'],c['rtol'])!=('contact',0.,.001):continue
 workers={e:subprocess.Popen([sys.executable,str(__file__),e],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1) for e in engines}
 for w in workers.values():w.stdin.write(json.dumps(c)+'\n');w.stdin.flush()
 def call(e,source):
  w=workers[e];w.stdin.write(json.dumps(source)+'\n');w.stdin.flush();line=w.stdout.readline();assert line,w.stderr.read();return json.loads(line)
 for i in range(5):
  for e in engines:report['warmups'].append(dict(case=c['name'],limb=c['limb'],rtol=c['rtol'],engine=e,result=call(e,c['case']['source'])))
 measured=[]
 for cycle in range(8):
  for i,source in enumerate(c['sources'][9:]):
   ref=json.loads((P/'references'/f"{c['name']}-u{c['limb']}-i{i+1}.json").read_text())['result'];offset=(cycle*8+i)%len(engines);order=engines[offset:]+engines[:offset]
   if cycle%2:order=list(reversed(order))
   row=dict(case=c['name'],limb=c['limb'],rtol=c['rtol'],cycle=cycle,index=i,source=source,order=order,results={})
   for e in order:
    r=call(e,source);r['relative_error']=None if r['mu'] is None else abs(r['mu']-ref['mu'])/abs(ref['mu']);row['results'][e]=r
    if e in ['before','algebraic']:assert r['converged']==1 and r['relative_error']<=c['rtol'] and abs(r['mu']-ref['mu'])<=r['error']+ref['error_estimate'],row
   measured.append(row);report['measurements'].append(row)
 for e,w in workers.items():w.stdin.close();assert w.wait(timeout=10)==0,(e,w.stderr.read())
 for e in engines:
  values=[r['results'][e] for r in measured];summary=dict(case=c['name'],limb=c['limb'],rtol=c['rtol'],engine=e,count=len(values),median_ms=1000*statistics.median(r['seconds'] for r in values),nonfinite=sum(r['mu'] is None for r in values),above_requested=sum(r['relative_error'] is not None and r['relative_error']>c['rtol'] for r in values),annuli=sorted(set(r['annuli'] for r in values)));report['rows'].append(summary)
 save();print(c['name'],c['limb'],c['rtol'],[(r['engine'],round(r['median_ms'],4),r['nonfinite']) for r in report['rows'][-5:]],flush=True)
report['status']='PASS';save()
