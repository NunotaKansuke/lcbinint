import ctypes as C,json,os,math,sys
from pathlib import Path
import VBMicrolensing
P=Path(__file__).resolve().parent;R=P.parent.parent;os.sched_setaffinity(0,{0})
class Out(C.Structure):_fields_=[('mu',C.c_double),('error',C.c_double),('seconds',C.c_double),('samples',C.c_int),('annuli',C.c_int),('converged',C.c_int)]
a=C.CDLL(str(P/sys.argv[1]));b=C.CDLL(str(R/'evidence/triple_fixed_ld_breakdown/final.so'));q=C.CDLL(str(R/'evidence/triple_value_contract/stages.so'))
for l in [a,b]:
 l.algebraic_create.argtypes=[C.POINTER(C.c_double)];l.algebraic_create.restype=C.c_void_p;l.algebraic_destroy.argtypes=[C.c_void_p];l.algebraic_point.argtypes=[C.c_void_p]+[C.c_double]*5+[C.c_int,C.POINTER(Out)]
b.vbm_create.argtypes=[C.c_int,C.POINTER(C.c_double)];b.vbm_create.restype=C.c_void_p;b.vbm_destroy.argtypes=[C.c_void_p];b.vbm_point.argtypes=[C.c_void_p]+[C.c_double]*5+[C.c_int,C.POINTER(Out)]
q.create.argtypes=[C.POINTER(C.c_double)];q.create.restype=C.c_void_p;q.destroy.argtypes=[C.c_void_p];q.reference.argtypes=[C.c_void_p]+[C.c_double]*4+[C.POINTER(C.c_double)]
base=dict(lenses=[[0.,0.],[1.,0.],[-.4,.6]],masses=[.72,.2,.08]);cases=[]
for source,rho in [([.06,.12],.08),([.12,.17],.12),([.08,.18],.25),([.15,.10],.32)]:cases.append(dict(**base,source=source,rho=rho))
for rho in [.12,.28]:cases.append(dict(lenses=[[0.,0.],[1.1,0.],[-.3,.8]],masses=[.5,.3,.2],source=[.1,.15],rho=rho))
for rho in [.1,.25]:cases.append(dict(lenses=[[0.,0.],[1.25,0.],[-.6,.4]],masses=[.90,.08,.02],source=[.12,.08],rho=rho))
cache_path=P/('validation-final.json' if (P/'validation-final.json').exists() else ('validation-joint.json' if (P/'validation-joint.json').exists() else 'validation-extra.json'))
cached=json.loads(cache_path.read_text())['rows'] if cache_path.exists() else []
for source in [[-.087243824247718094,-.52604606110908503],[-.087211699298757425,-.5259953857930596]]:
 cases.append(dict(**base,source=source,rho=.03))
for rho in [.1,.3]:cases.append(dict(lenses=[[0.,0.],[1.,0.],[-.2,.9]],masses=[.4,.3,.3],source=[.1,.15],rho=rho))
for rho in [.1,.25]:cases.append(dict(lenses=[[0.,0.],[.6,0.],[-.3,.4]],masses=[.65,.25,.1],source=[-.05,.1],rho=rho))
rows=[]
for i,c in enumerate(cases):
 g=(C.c_double*9)(*[v for z,m in zip(c['lenses'],c['masses'])for v in (*z,m)])
 for u in [0.,.5]:
  saved=next((r for r in cached if r['case']==c and r['limb']==u),None)
  if saved:
   ref=(C.c_double*3)(*saved['native_reference']);v=Out(**saved['vbm_reference'])
  else:
   h=q.create(g);ref=(C.c_double*3)();q.reference(h,*c['source'],c['rho'],u,ref);q.destroy(h)
   h=b.vbm_create(0,g);v=Out();b.vbm_point(h,*c['source'],c['rho'],u,1e-6,int(u!=0),C.byref(v));b.vbm_destroy(h)
  for tol in [.001,.0001]:
   h=a.algebraic_create(g);out=Out();a.algebraic_point(h,*c['source'],c['rho'],u,tol,int(u!=0),C.byref(out));a.algebraic_destroy(h)
   native_diff=abs(out.mu-ref[0]);external_diff=abs(out.mu-v.mu)/abs(v.mu)
   row=dict(case=c,limb=u,rtol=tol,native_reference=list(ref),vbm_reference={k:getattr(v,k)for k,_ in Out._fields_},result={k:getattr(out,k)for k,_ in Out._fields_},relative_native_difference=native_diff/abs(ref[0]),relative_vbm_difference=external_diff,estimate_contains_native=native_diff<=out.error+ref[1],passed=bool(ref[2])and out.converged==1 and native_diff/abs(ref[0])<=tol and external_diff<=tol)
   rows.append(row);(P/(sys.argv[2]if len(sys.argv)>2 else 'validation-extra.json')).write_text(json.dumps(dict(rows=rows),indent=2));print(i,u,tol,row['passed'],row['estimate_contains_native'],row['relative_native_difference'],external_diff,flush=True)
