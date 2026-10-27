import ctypes as C,json,sys,os
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent.parent;os.sched_setaffinity(0,{0});l=C.CDLL(str(D/sys.argv[1]));l.binary_create.argtypes=[C.POINTER(C.c_double)];l.binary_create.restype=C.c_void_p;l.binary_destroy.argtypes=[C.c_void_p];l.binary_evaluate.argtypes=[C.c_void_p,C.c_double,C.c_double,C.c_int,C.POINTER(C.c_double)];cache={}
def call(p,u,tol,mode=1):
 key=(*p,u,tol,mode)
 if key not in cache:
  h=l.binary_create((C.c_double*5)(*p));assert h
  o=(C.c_double*16)();l.binary_evaluate(h,u,tol,mode,o);l.binary_destroy(h);cache[key]=list(o)
 return cache[key]
fd=[]
for f in sorted((R/'evidence/binary_triple_broad_cost').glob('fd-[0-9]*.json')):
 a=json.loads(f.read_text())
 if a['case']['kind']!='binary' or a['case']['id']==25:continue
 for ref in a['rows']:
  if not ref['resolved']:continue
  o=call(a['case']['p'],ref['limb'],ref['rtol']);g=o[10+ref['parameter']];ratio=(abs(g-ref['reference'])+ref['uncertainty'])/ref['budget'];fd.append(dict(case=a['case']['id'],parameter=ref['parameter'],limb=ref['limb'],rtol=ref['rtol'],ratio=ratio,passed=bool(o[2]and o[5]and ratio<=1)))
a=json.loads((R/'evidence/binary_triple_broad_cost/binary-planetary-refined-fd.json').read_text())
for ref in a['rows']:
 o=call(a['case']['p'],ref['limb'],ref['rtol']);g=o[10+ref['parameter']];ratio=(abs(g-ref['reference'])+ref['uncertainty'])/ref['budget'];fd.append(dict(case=25,parameter=ref['parameter'],limb=ref['limb'],rtol=ref['rtol'],ratio=ratio,passed=bool(o[2]and o[5]and ratio<=1)))
# Previous wider independent scalar FD references, stored in COM coordinates.
lookup={r['case']:r['p']for r in json.loads((R/'evidence/strict_path_removal_20261028/binary-final.json').read_text())if r['mode']==1 and r['out'][5]==7}
for f in ['binary-fd.json','binary-used-fd.json','binary-contact-fd.json']:
 for ref in json.loads((R/'evidence/strict_path_removal_20261028'/f).read_text()):
  if not ref.get('pass_gate',False):continue
  p=lookup.get(ref['case'])
  if p is None:continue
  p=p.copy();p[0]+=p[3]/(1+p[3])*p[4];o=call(p,ref['u'],ref['tol']);g=o[10:15];g[3]+=p[4]/(1+p[3])**2*g[0];g[4]+=p[3]/(1+p[3])*g[0]
  ratio=(abs(g[ref['parameter']]-ref['reference'])+ref['uncertainty'])/ref['gate'];fd.append(dict(source=f,case=ref['case'],parameter=ref['parameter'],limb=ref['u'],rtol=ref['tol'],ratio=ratio,passed=ratio<=1))
rows=[]
for c in json.loads((R/'evidence/binary_triple_broad_cost/inputs.json').read_text()):
 if c['kind']!='binary':continue
 for u in[0.,.5]:
  for tol in[1e-3,1e-4]:
   rows.append(dict(case=c,limb=u,rtol=tol,outputs={mode:call(c['p'],u,tol,mode)for mode in[0,1]}))
(D/sys.argv[2]).write_text(json.dumps(dict(fd=fd,rows=rows),indent=2));print('FD',sum(r['passed']for r in fd),'/',len(fd));print('failures',[r for r in fd if not r['passed']])
