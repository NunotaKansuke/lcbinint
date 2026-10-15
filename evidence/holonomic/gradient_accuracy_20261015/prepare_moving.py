from pathlib import Path
import shutil,hashlib,json
root=Path(__file__).resolve().parents[3];build=root/'build-algebraic-boundary/gradient-probe'
target=build/'moving-include';shutil.copytree(build/'api-include',target,dirs_exist_ok=True)
p=target/'lcbinint/magnification/holonomic/adaptive_epoch.hpp';s=p.read_text()
a=s.index("            // Retain V2's angular rescue.");b=s.index('\n    }\n#ifdef HOLO_FRONTIER_COST_PROFILE',a)
section=s[a:b]
section=section.replace('auto g=phi_val_dP(R,th,pf);','auto g=phi_grad(R,th,pf);')
for rule,sum_name in [('high','dh'),('low','dd')]:
    old=f'{sum_name}[j]+={rule}.w[k]*(jac*g.dP[j])/(2*sq);'
    new=f'''const double theta_motion=0.5*((1-{rule}.x[k])*dte[j]+(1+{rule}.x[k])*dtl[j]);{sum_name}[j]+={rule}.w[k]*(jac*g.dP[j]+g.dphi_dtheta*theta_motion)/(2*sq);'''
    assert section.count(old)==1
    section=section.replace(old,'{'+new+'}')
old='if(with_jac)for(int j=0;j<5;++j){dfh[j]+=R*half*dh[j];edfh[j]+=R*half*std::fabs(dh[j]-dd[j]);}'
new='''if(with_jac)for(int j=0;j<5;++j){
                const double half_motion=0.5*(dtl[j]-dte[j]);
                const double mapped_hi=half*dh[j]+half_motion*vh;
                const double mapped_lo=half*dd[j]+half_motion*vl;
                dfh[j]+=R*mapped_hi;edfh[j]+=R*std::fabs(mapped_hi-mapped_lo);
            }'''
assert section.count(old)==1;section=section.replace(old,new)
p.write_text(s[:a]+section+s[b:])
(build/'moving_manifest.json').write_text(json.dumps({'base':hashlib.sha256((build/'api-include/lcbinint/magnification/holonomic/adaptive_epoch.hpp').read_bytes()).hexdigest(),'candidate':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2)+'\n')
