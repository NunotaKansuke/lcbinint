import ctypes as C,json,statistics as S,time,os
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent.parent;os.sched_setaffinity(0,{0});dl=C.CDLL('libdl.so.2');dl.dlmopen.argtypes=[C.c_long,C.c_char_p,C.c_int];dl.dlmopen.restype=C.c_void_p;libs=[]
for name in['chain-before.so','chain-gcd.so']:
 path=D/name;h=dl.dlmopen(-1,os.fsencode(path),os.RTLD_NOW|os.RTLD_LOCAL);l=C.CDLL(str(path),handle=h);l.build.argtypes=[C.POINTER(C.c_double)];l.build.restype=C.c_void_p;l.destroy.argtypes=[C.c_void_p];l.size.argtypes=[C.c_void_p,C.c_int];l.value.argtypes=[C.c_void_p,C.c_int,C.c_int];l.value.restype=C.c_char_p;libs.append(l)
rows=[]
for c in json.loads((R/'evidence/binary_triple_broad_cost/inputs.json').read_text()):
 if c['kind']!='binary':continue
 p=(C.c_double*5)(*c['p']);samples=[[],[]];eq=True
 for k in range(5):
  handles=[]
  for i in([0,1]if k%2==0 else[1,0]):
   l=libs[i];start=time.thread_time_ns();h=l.build(p);dt=(time.thread_time_ns()-start)/1e6;handles.append((i,h))
   if k:samples[i].append(dt)
  h=dict(handles);n=libs[0].size(h[0],-1);eq=eq and n==libs[1].size(h[1],-1)
  for a in range(n):
   degree=libs[0].size(h[0],a);eq=eq and degree==libs[1].size(h[1],a)
   for b in range(degree):eq=eq and libs[0].value(h[0],a,b)==libs[1].value(h[1],a,b)
  for i,h in handles:libs[i].destroy(h)
 row=dict(case=c,exact_coefficient_parity=eq,samples=samples,medians=[S.median(v)for v in samples]);rows.append(row);(D/'chain-gcd-results.json').write_text(json.dumps(rows,indent=2));print(c['id'],eq,row['medians'],flush=True)
