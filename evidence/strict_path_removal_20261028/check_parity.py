import ctypes as C,json,os,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent.parent;B=D.parent/'source_disk_removal_20261028'
dl=C.CDLL('libdl.so.2');dl.dlmopen.argtypes=[C.c_long,C.c_char_p,C.c_int];dl.dlmopen.restype=C.c_void_p
libs={}
for name,path in [('before',B/'triple-final.so'),('after',D/'triple-final.so')]:
 h=dl.dlmopen(-1,os.fsencode(path),os.RTLD_NOW|os.RTLD_LOCAL);assert h
 l=C.CDLL(str(path),handle=h);l.create.restype=C.c_void_p;l.destroy.argtypes=[C.c_void_p];l.evaluate.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_double,C.c_double,C.c_int,C.POINTER(C.c_double)];libs[name]=l
rows=[]
for r in json.loads((R/'evidence/triple_speed_comparison_ld_20261028/results.json').read_text())['rows']:
 for mode in [0,2]:
  out={};p=[*r['source'],r['rho'],.2,.08,1,-.4,.6]
  for name,l in libs.items():
   h=l.create();o=(C.c_double*24)();l.evaluate(h,(C.c_double*8)(*p),r['limb'],r['rtol'],mode,o);l.destroy(h);out[name]=list(o)
  same=all(a==b for j,(a,b)in enumerate(zip(out['before'],out['after']))if j not in [20,21])
  rows.append(dict(p=p,u=r['limb'],tol=r['rtol'],mode=mode,out=out,exact=same))
(D/'triple-parity.json').write_text(json.dumps(dict(rows=rows),indent=2))
print('TRIPLE exact',sum(r['exact']for r in rows),'/',len(rows));assert all(all(a==b if j in [2,3,4,5,6,7,8,9,10,22,23] else abs(a-b)<=1e-12*(1+abs(a)) for j,(a,b)in enumerate(zip(r['out']['before'],r['out']['after'])) if j not in [20,21])for r in rows)
pairs=[]
for n in ['random','archived','exact']:
 a=json.loads((B/(n+'-final.json')).read_text());b=json.loads((D/(n+'-final.json')).read_text());pairs.extend(zip(a,b))
a=json.loads((B/'binary-final.json').read_text());b=json.loads((D/'binary-final.json').read_text());pairs.extend((x,y)for x,y in zip(a,b)if x['mode']in[0,1])
rows=[]
for a,b in pairs:
 assert (a['case'],a['p'],a['u'],a['tol'],a['mode'])==(b['case'],b['p'],b['u'],b['tol'],b['mode'])
 rows.append(dict(case=a['case'],p=a['p'],u=a['u'],tol=a['tol'],mode=a['mode'],before=a['out'],after=b['out'],exact=a['out']==b['out'],flags_equal=a['out'][2:5]==b['out'][2:5]))
(D/'binary-parity.json').write_text(json.dumps(dict(rows=rows),indent=2))
print('BINARY exact',sum(r['exact']for r in rows),'/',len(rows),'same flags',sum(r['flags_equal']for r in rows))
regular=[r for r in rows if r['before'][2]]
assert all(r['exact']for r in regular)
changed=[r for r in rows if not r['exact']]
assert len(changed)==1
r=changed[0];assert not r['before'][2] and r['after'][2] and r['after'][4]
scalar=next(x for x in rows if x['p']==r['p'] and x['tol']==r['tol'] and x['u']==r['u'] and x['mode']==0)
assert abs(r['after'][0]-scalar['after'][0])<=r['tol']*abs(scalar['after'][0])
print('PASS: all previously converged outputs unchanged; one former failure converges and matches scalar')
