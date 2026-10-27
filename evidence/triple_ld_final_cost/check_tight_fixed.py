import ctypes as C,json
from pathlib import Path
P=Path(__file__).resolve().parent;a=C.CDLL(str(P/'final.so'));a.create.restype=C.c_void_p;a.destroy.argtypes=[C.c_void_p];a.evaluate.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_double,C.c_double,C.c_int,C.POINTER(C.c_double)];refs=[r for r in json.loads((P/'tight.json').read_text())if r['engine']=='before'and r['out'][2]];rows=[]
z=.9+.1j;source=z-sum(m/(z-l).conjugate()for m,l in zip([.72,.2,.08],[0j,1+0j,-.4+.6j]))
for ref in refs:
 h=a.create();o=(C.c_double*24)();a.evaluate(h,(C.c_double*8)(source.real,source.imag,ref['rho'],.2,.08,1,-.4,.6),.5,ref['rtol'],0,o);a.destroy(h);assert o[2]and o[3],list(o);assert abs(o[0]-ref['out'][0])<=o[1]+ref['out'][1];rows.append(dict(rho=ref['rho'],rtol=ref['rtol'],out=list(o),reference=ref['out']));print('PASS',ref['rho'],ref['rtol'],'quad',o[7],flush=True)
(P/'tight-fixed.json').write_text(json.dumps(rows,indent=2))
