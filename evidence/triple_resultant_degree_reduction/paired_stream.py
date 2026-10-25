import ctypes as C,json,pathlib,os,sys,subprocess,statistics,math
P=pathlib.Path(__file__).resolve().parent
class Out(C.Structure):_fields_=[('mu',C.c_double),('error',C.c_double),('seconds',C.c_double),('samples',C.c_int),('annuli',C.c_int),('converged',C.c_int)]
if len(sys.argv)>1:
 import VBMicrolensing
 init=json.loads(sys.stdin.readline());lib=C.CDLL(str(P/('kernel-before.so' if sys.argv[1]=='before' else 'kernel.so')));lib.algebraic_create.argtypes=[C.POINTER(C.c_double)];lib.algebraic_create.restype=C.c_void_p;lib.algebraic_point.argtypes=[C.c_void_p]+[C.c_double]*5+[C.c_int,C.POINTER(Out)];lib.algebraic_destroy.argtypes=[C.c_void_p]
 c=init['case'];g=(C.c_double*9)(*[a for xy,m in zip(c['lenses'],c['masses']) for a in (*xy,m)]);b=lib.algebraic_create(g)
 for line in sys.stdin:
  s=json.loads(line);r=Out();lib.algebraic_point(b,*s,c['rho'],init['limb'],init['rtol'],int(init['limb']!=0),C.byref(r));print(json.dumps({k:getattr(r,k) for k,_ in Out._fields_}),flush=True)
 lib.algebraic_destroy(b);sys.exit()
os.sched_setaffinity(0,{0});conditions=[]
for f in sorted((P/'conditions').glob('*-algebraic-*.json')):
 d=json.loads(f.read_text());conditions.append(d)
report=dict(rows=[],pairs=[])
for c in conditions:
 workers={engine:subprocess.Popen([sys.executable,str(__file__),engine],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1) for engine in ['before','after']}
 for w in workers.values():w.stdin.write(json.dumps(c)+'\n');w.stdin.flush()
 def call(engine,source):
  w=workers[engine];w.stdin.write(json.dumps(source)+'\n');w.stdin.flush();return json.loads(w.stdout.readline())
 for _ in range(9):
  for e in ['before','after']:call(e,c['case']['source'])
 pairs=[]
 for cycle in range(4):
  for j,s in enumerate(c['sources'][9:]):
   row=dict(case=c['name'],limb=c['limb'],rtol=c['rtol'],index=j,cycle=cycle,source=s)
   ref=json.loads((P/'references'/f"{c['name']}-u{c['limb']}-i{j+1}.json").read_text())['result']
   for e in (['before','after'] if (cycle+j)%2==0 else ['after','before']):
    v=call(e,s);row[e]=v;err=abs(v['mu']-ref['mu']);assert v['converged']==1 and err<=c['rtol']*abs(ref['mu']) and err<=v['error']+ref['error_estimate'],row
   pairs.append(row);report['pairs'].append(row)
 for w in workers.values():w.stdin.close();assert w.wait(timeout=10)==0,w.stderr.read()
 summary=dict(case=c['name'],limb=c['limb'],rtol=c['rtol'],pairs=len(pairs),before_ms=1000*statistics.median(p['before']['seconds'] for p in pairs),after_ms=1000*statistics.median(p['after']['seconds'] for p in pairs),median_pair_speedup=statistics.median(p['before']['seconds']/p['after']['seconds'] for p in pairs));report['rows'].append(summary);(P/'paired-stream.json').write_text(json.dumps(report,indent=2)+'\n');print(summary,flush=True)
report['status']='PASS';(P/'paired-stream.json').write_text(json.dumps(report,indent=2)+'\n')
