import ctypes as C
import json
import os
from pathlib import Path
import statistics
import time
import sys
import VBMicrolensing

P=Path(__file__).resolve().parent
R=P.parent.parent
os.sched_setaffinity(0,{0})
class Out(C.Structure):
    _fields_=[('mu',C.c_double),('error',C.c_double),('seconds',C.c_double),('samples',C.c_int),('annuli',C.c_int),('converged',C.c_int)]
libs={k:C.CDLL(str(p)) for k,p in {'before':R/'evidence/triple_fixed_ld_breakdown/final.so','after':P/(sys.argv[1] if len(sys.argv)>1 else 'candidate.so')}.items()}
for k in ['Singlepoly','Multipoly','Nopoly']:
    libs[k]=C.CDLL(str(R/'evidence/triple_natural_integral_benchmark/kernel.so'))
for l in libs.values():
    l.vbm_create.argtypes=[C.c_int,C.POINTER(C.c_double)];l.vbm_create.restype=C.c_void_p
    l.vbm_destroy.argtypes=[C.c_void_p]
    l.vbm_point.argtypes=[C.c_void_p]+[C.c_double]*5+[C.c_int,C.POINTER(Out)]
    l.algebraic_create.argtypes=[C.POINTER(C.c_double)];l.algebraic_create.restype=C.c_void_p
    l.algebraic_destroy.argtypes=[C.c_void_p]
    l.algebraic_point.argtypes=[C.c_void_p]+[C.c_double]*5+[C.c_int,C.POINTER(Out)]
base=json.loads((R/'evidence/triple_resultant_degree_reduction/conditions/contact-algebraic-r0.001-u0.0-route0.json').read_text())['case']
g=(C.c_double*9)(*[v for z,m in zip(base['lenses'],base['masses']) for v in (*z,m)])
data=json.loads((R/'evidence/triple_natural_integral_benchmark/results.json').read_text());rows=[]
repeat=int(os.environ.get('BENCH_REPEAT','0'))
for condition in data['rows']:
    frequency_before=int(Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq').read_text())
    states={k:(l.algebraic_create(g) if k in ['before','after'] else l.vbm_create(['Singlepoly','Multipoly','Nopoly'].index(k),g)) for k,l in libs.items()};samples={k:[] for k in libs}
    for i in range(12):
        order=list(libs);order=order[(i+repeat)%5:]+order[:(i+repeat)%5]
        for k in (order if i%2 else order[::-1]):
            out=Out();t=time.thread_time_ns();getattr(libs[k], 'algebraic_point' if k in ['before','after'] else 'vbm_point')(states[k],*condition['source'],condition['rho'],condition['limb'],condition['rtol'],int(condition['limb']!=0),C.byref(out));dt=(time.thread_time_ns()-t)/1e6
            if i>=4:samples[k].append(dict(cpu_ms=dt,**{f:getattr(out,f) for f,_ in Out._fields_}))
    for k,l in libs.items():getattr(l, 'algebraic_destroy' if k in ['before','after'] else 'vbm_destroy')(states[k])
    ref=data['references'][condition['reference_id']]['results']['algebraic'][-1]['mu']
    row={k:condition[k] for k in ['source','rho','limb','rtol']};row['samples']=samples;row['frequency_khz']=[frequency_before,int(Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq').read_text())];row['repeat']=repeat
    row['median']={k:statistics.median(x['cpu_ms'] for x in v) for k,v in samples.items()}
    row['speedup']=row['median']['before']/row['median']['after']
    row['relative_difference']=abs(samples['after'][-1]['mu']-ref)/abs(ref)
    row['pass']=all(x['converged']==1 for k in ['before','after'] for x in samples[k]) and row['relative_difference']<condition['rtol']
    rows.append(row);(P/(sys.argv[2] if len(sys.argv)>2 else 'matched-vbm.json')).write_text(json.dumps(dict(rows=rows),indent=2));print(row['source'],row['rho'],row['limb'],row['rtol'],row['speedup'],row['pass'],flush=True)
