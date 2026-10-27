import ctypes as C,json,time,statistics as S,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent.parent;os.sched_setaffinity(0,{0});dl=C.CDLL('libdl.so.2');dl.dlmopen.argtypes=[C.c_long,C.c_char_p,C.c_int];dl.dlmopen.restype=C.c_void_p
libs={}
for name,path in [('before',R/'evidence/triple_fejer_boundary_shape/binary-current.so'),('after',D/sys.argv[1])]:
 h=dl.dlmopen(-1,os.fsencode(path),os.RTLD_NOW|os.RTLD_LOCAL);assert h;l=C.CDLL(str(path),handle=h);l.binary_create.argtypes=[C.POINTER(C.c_double)];l.binary_create.restype=C.c_void_p;l.binary_destroy.argtypes=[C.c_void_p];l.binary_evaluate.argtypes=[C.c_void_p,C.c_double,C.c_double,C.c_int,C.POINTER(C.c_double)];libs[name]=l
rows=[];engines=[('before',0),('after',0),('before',1),('after',1)]
for case in json.loads((R/'evidence/binary_triple_broad_cost/inputs.json').read_text()):
 if case['kind']!='binary':continue
 p=(C.c_double*5)(*case['p'])
 for u in[0.,.5]:
  for tol in[1e-3,1e-4]:
   states={(k,m):libs[k].binary_create(p)for k,m in engines};samples={f'{k}-{m}':[]for k,m in engines};outputs={}
   for j in range(16):
    order=engines[j%4:]+engines[:j%4]
    if j%2:order=order[::-1]
    for k,m in order:
     out=(C.c_double*16)();start=time.thread_time_ns();libs[k].binary_evaluate(states[k,m],u,tol,m,out);dt=(time.thread_time_ns()-start)/1e6;key=f'{k}-{m}';outputs[key]=list(out)
     if j>=4:samples[key].append(dt)
   for(k,m),h in states.items():libs[k].binary_destroy(h)
   med={k:S.median(v)for k,v in samples.items()};rows.append(dict(case=case,limb=u,rtol=tol,samples=samples,outputs=outputs,medians=med,gradient_speed_ratio=med['after-1']/med['before-1'],before_overhead=med['before-1']/med['before-0'],after_overhead=med['after-1']/med['after-0']))
   (D/sys.argv[2]).write_text(json.dumps(dict(protocol='Same binary physical primary-frame disks, natural router screen corrected separately. CPU0 threadCPU,4warmups+12 rotating/reversed samples. Separate ELF namespaces and prepared state per mode/condition. Exact seeds/support outside both timings. No concurrent builds or validation. Failures retained.',rows=rows),indent=2))
 print(case['id'],'done',flush=True)
