"""Independent continuous horizontal slices of failed correction candidates."""
import json,math,os,argparse
from pathlib import Path
from scipy.optimize import brentq
from scipy.integrate import quad
from .probe_build import activate
ROOT=Path('tests/diagnostics/results/recal2026')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='adaptive-defect-row-cause-v2');args=parser.parse_args();out=ROOT/args.output;out.mkdir(exist_ok=False)
 for k in ('DIRECTIONAL_ADAPTIVE','LOCAL_ROW_INTERPOLATION','LOCAL_BOUNDARY_STRIPS','LOCAL_VERTICAL_CAPS','DEFECT_CORRECTION','ROW_PANEL_TRACE','ADAPTIVE_COST_TRACE'):os.environ['LCBININT_DIAGNOSTIC_'+k]='1'
 n=activate()._lcbinint;rs=json.load(open(ROOT/'adaptive-defect-fresh-v1/summary.json'))['rows'];results=[]
 for idx,c in ((4,.2),(12,1),(18,1)):
  r=next(r for r in rs if r['index']==idx and r['sign']==1 and r['c']==c and r['tolerance']==1e-4);i=r['input'];rho=i['rho'];m1=1/(1+i['q']);m2=i['q']*m1;s=i['s'];source=complex(i['x']+s*m1,i['y']);norm=math.pi*rho*rho*(1-c/3)
  g=n.binary_adaptive_flux(i['x'],i['y'],s=s,q=1/i['q'],rho=rho,limb_darkening=n.LimbDarkening(c,0),initial_bins=16,max_bins=512,absolute_tolerance=1e-4)
  stage=len(g['history'])-1;rows=[]
  for row in g['row_panel_rows']:
   if row[0]!=stage:continue
   _,comp,bx,by,iy,lo,hi,raw_integral,S,R,unresolved=row;hx=rho/bx;hy=rho/by;y=iy*hy
   def depth(x):
    z=complex(x,y)
    try:return 1-abs(z-m1/(z.conjugate()-s)-m2/z.conjugate()-source)**2/rho**2
    except ZeroDivisionError:return -float('inf')
   left=brentq(depth,(lo-1)*hx,lo*hx,xtol=1e-14);right=brentq(depth,hi*hx,(hi+1)*hx,xtol=1e-14)
   mid=(left+right)/2;half=(right-left)/2
   def f(theta):return ((1-c)+c*math.sqrt(max(0,depth(mid+half*math.cos(theta)))))*half*math.sin(theta)
   exact,err=quad(f,0,math.pi,epsabs=norm/hy*1e-11,epsrel=1e-11,limit=100)
   defect=(exact-raw_integral)*hy
   if hi==lo:assert abs(defect-S)/norm<=R/norm+1e-9,(idx,c,row,defect/norm,S/norm,R/norm)
   rows.append(dict(component=int(comp),iy=int(iy),nodes=int(hi-lo+1),true_signed=defect/norm,estimated_signed=S/norm,remainder=R/norm,miss=(defect-S)/norm,quadrature_error=err*hy/norm))
  components=[]
  for comp in sorted({v['component'] for v in rows}):
   cr=[v for v in rows if v['component']==comp];components.append(dict(component=comp,true_horizontal=sum(v['true_signed'] for v in cr),estimated_horizontal=sum(v['estimated_signed'] for v in cr),model_miss=sum(v['miss'] for v in cr),remainder=sum(v['remainder'] for v in cr),short_miss=sum(v['miss'] for v in cr if v['nodes']<=8),long_miss=sum(v['miss'] for v in cr if v['nodes']>8),worst_rows=sorted(cr,key=lambda v:abs(v['miss']),reverse=True)[:8]))
  rec=dict(index=idx,c=c,input=i,actual_error=abs(g['magnification']-r['reference']),components=components,rows=rows);results.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='rows'}),flush=True)
 (out/'summary.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
