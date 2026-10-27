import ctypes as C,json,sys,hashlib,os
from pathlib import Path
P=Path(__file__).resolve().parent;os.sched_setaffinity(0,{0});file=P/sys.argv[1];lib=C.CDLL(str(file));lib.create.restype=C.c_void_p;lib.destroy.argtypes=[C.c_void_p];lib.evaluate.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_double,C.c_double,C.c_int,C.POINTER(C.c_double)]
sources=['selected-new-geometry-fd.json','hybrid-budget-half-angle-independent.json']+[f'hybrid-budget-independent-{i}.json'for i in range(3)]
rows=[];outputs={}
for source in sources:
 data=json.loads((P/source).read_text());assert len(data['rows'])==32,(source,len(data['rows']))
 for ref in data['rows']:
  assert ref['reference_resolved'],(source,ref)
  p=data['geometry'];key=(*p,ref['limb'],ref['rtol'])
  if key not in outputs:
   h=lib.create();o=(C.c_double*24)();lib.evaluate(h,(C.c_double*8)(*p),ref['limb'],ref['rtol'],2,o);lib.destroy(h);outputs[key]=list(o)
  o=outputs[key];g=o[12+ref['parameter']];ratio=(abs(g-ref['reference'])+ref['reference_uncertainty'])/ref['budget'];row=dict(source=source,geometry=p,limb=ref['limb'],rtol=ref['rtol'],parameter=ref['parameter'],gradient=g,reference=ref['reference'],uncertainty=ref['reference_uncertainty'],budget=ref['budget'],ratio=ratio,passed=bool(o[2]and o[3]and o[4]and ratio<=1));rows.append(row)
 for tol in [1e-3,1e-4]:
  key=(*data['geometry'],1.,tol)
  if key not in outputs:
   h=lib.create();o=(C.c_double*24)();lib.evaluate(h,(C.c_double*8)(*data['geometry']),1.,tol,2,o);lib.destroy(h);outputs[key]=list(o)
  o=outputs[key]
  for j in range(8):
   a=next(r for r in data['rows'] if r['limb']==0 and r['rtol']==tol and r['parameter']==j)
   b=next(r for r in data['rows'] if r['limb']==.5 and r['rtol']==tol and r['parameter']==j)
   reference=2.5*b['reference']-1.5*a['reference'];uncertainty=2.5*b['reference_uncertainty']+1.5*a['reference_uncertainty'];budget=1e-4+2e-4*abs(reference);g=o[12+j];ratio=(abs(g-reference)+uncertainty)/budget
   rows.append(dict(source=source,geometry=data['geometry'],limb=1.,rtol=tol,parameter=j,gradient=g,reference=reference,uncertainty=uncertainty,budget=budget,ratio=ratio,passed=bool(o[2]and o[3]and o[4]and ratio<=1),reference_method='exact linear LD identity: g(1)=2.5*g(.5)-1.5*g(0)'))
 (P/sys.argv[2]).write_text(json.dumps(dict(kernel_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),rows=rows),indent=2))
 print(source,sum(r['passed']for r in rows if r['source']==source),'/48',max(r['ratio']for r in rows if r['source']==source),flush=True)
assert all(r['passed']for r in rows),[r for r in rows if not r['passed']]
print('PASS',len(rows),'independent FD components',flush=True)
