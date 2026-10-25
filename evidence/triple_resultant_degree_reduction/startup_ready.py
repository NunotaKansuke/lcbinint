import pathlib,json,subprocess,sys,statistics,os
P=pathlib.Path(__file__).resolve().parent
if len(sys.argv)>1:
 import ctypes as C
 import VBMicrolensing
 class Out(C.Structure):_fields_=[('mu',C.c_double),('error',C.c_double),('seconds',C.c_double),('samples',C.c_int),('annuli',C.c_int),('converged',C.c_int)]
 init=json.loads(sys.stdin.readline());lib=C.CDLL(str(P/('kernel-before.so' if sys.argv[1]=='before' else 'kernel.so')))
 lib.algebraic_create.argtypes=[C.POINTER(C.c_double)];lib.algebraic_create.restype=C.c_void_p
 lib.algebraic_point.argtypes=[C.c_void_p]+[C.c_double]*5+[C.c_int,C.POINTER(Out)]
 lib.algebraic_destroy.argtypes=[C.c_void_p]
 c=init['case'];g=(C.c_double*9)(*[v for z,m in zip(c['lenses'],c['masses']) for v in (*z,m)])
 handle=lib.algebraic_create(g)
 print('READY',flush=True)
 for line in sys.stdin:
  source=json.loads(line);out=Out();lib.algebraic_point(handle,*source,c['rho'],init['limb'],init['rtol'],int(init['limb']!=0),C.byref(out));print(json.dumps({k:getattr(out,k) for k,_ in Out._fields_}),flush=True)
 lib.algebraic_destroy(handle);sys.exit()
os.sched_setaffinity(0,{0})
report={'status':'RUNNING','protocol':'8 fresh-process pairs per condition, explicit READY barrier for both engines; first eight distinct source queries, zero warmups; before/after alternating; only C++ integration timer reported','rows':[],'measurements':[]}
for f in sorted((P/'conditions').glob('*-algebraic-*.json')):
 c=json.loads(f.read_text());runs=[]
 for trial in range(8):
  workers={e:subprocess.Popen([sys.executable,str(__file__),e],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1) for e in ['before','after']}
  for w in workers.values():w.stdin.write(json.dumps(c)+'\n');w.stdin.flush()
  for w in workers.values():assert w.stdout.readline().strip()=='READY',w.stderr.read()
  for i,s in enumerate(c['sources'][9:]):
   r=dict(case=c['name'],limb=c['limb'],rtol=c['rtol'],trial=trial,query=i+1)
   ref=json.loads((P/'references'/f"{c['name']}-u{c['limb']}-i{i+1}.json").read_text())['result']
   for e in (['before','after'] if (trial+i)%2 else ['after','before']):
    w=workers[e];w.stdin.write(json.dumps(s)+'\n');w.stdin.flush();line=w.stdout.readline();assert line,w.stderr.read();v=json.loads(line)
    err=abs(v['mu']-ref['mu']);assert v['converged'] and err<=c['rtol']*abs(ref['mu']) and err<=v['error']+ref['error_estimate'],(r,e,v)
    r[e]=v
   runs.append(r);report['measurements'].append(r)
  for w in workers.values():w.stdin.close();assert w.wait()==0,w.stderr.read()
 row=dict(case=c['name'],limb=c['limb'],rtol=c['rtol'],queries=[])
 for i in range(1,9):
  a=[r for r in runs if r['query']==i];row['queries'].append(dict(query=i,before_ms=1000*statistics.median(r['before']['seconds'] for r in a),after_ms=1000*statistics.median(r['after']['seconds'] for r in a)))
 row['first8_pair_speedup']=statistics.median(sum(r['before']['seconds'] for r in runs if r['trial']==j)/sum(r['after']['seconds'] for r in runs if r['trial']==j) for j in range(8))
 report['rows'].append(row);(P/'startup-ready.json').write_text(json.dumps(report,indent=2)+'\n');print(row,flush=True)
report['status']='PASS';(P/'startup-ready.json').write_text(json.dumps(report,indent=2)+'\n')
