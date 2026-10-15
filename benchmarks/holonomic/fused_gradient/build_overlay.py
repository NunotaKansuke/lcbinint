import os
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'src/lcbinint/magnification/holonomic'
OUT=Path(os.environ.get('LCBININT_FUSED_BUILD', str(ROOT/'build-seeded-events/fused-gradient')))/'include/lcbinint/magnification/holonomic'
OUT.mkdir(parents=True,exist_ok=True)
# Quote-relative includes must resolve within one consistent experimental
# header tree; mixing original and overlaid structs would violate the ABI.
shutil.copytree(BASE,OUT,dirs_exist_ok=True)
manifest={}
def replace(source,old,new):
    assert source.count(old)==1, ('unexpected source',old,source.count(old))
    return source.replace(old,new)
s=(BASE/'seeded_radial_frontier.hpp').read_text()
for option,value in [('reuse_frontier_count_kernel','true'),('prepared_fold_pair_seed','true'),('warm_root_seed_tolerance','1e-10')]:
    old=f'    if(cfg.effective_gradient_policy()==GradientPolicy::None && !cfg.with_jacobian)\n        exploration_cfg.{option}={value};'
    s=replace(s,old,f'    exploration_cfg.{option}={value};')
(OUT/'seeded_radial_frontier.hpp').write_text(s)
s=(BASE/'adaptive_epoch.hpp').read_text()
s=replace(s,'#include <optional>','#include <optional>\n#include "fused_k.hpp"')
s=replace(s,'if(cfg.prepared_fold_pair_seed && plain_value_lane && !force_cold &&','if(cfg.prepared_fold_pair_seed && !force_cold &&')
s=replace(s,'if(cfg.prepared_fold_pair_seed && plain_value_lane) {','if(cfg.prepared_fold_pair_seed) {')
old='''                auto hi=v_times_K_jac(ap.m,ap.v,ap.dm,ap.dv,R,pf,cp.p,dp.dp,reciprocal);
                if(hi.ok){auto lo=v_times_K_jac(ap.m,ap.v,ap.dm,ap.dv,R,pf,cp.p,dp.dp,reciprocal,8);
                    if(lo.ok){'''
new='''                const double vg=adaptive_cfg?0.05*adaptive_cfg->tol.mu_rtol:kHoloKRelTol;
                double gg=vg;
                if(adaptive_cfg)gg=0.05*(*std::min_element(adaptive_cfg->tol.grad_rtol.begin(),adaptive_cfg->tol.grad_rtol.end()));
                auto coupled=fused_gradient::angular_pair(ap.m,ap.v,ap.dm,ap.dv,R,pf,cp.p,dp.dp,reciprocal,vg,gg);
                auto hi=coupled.high,lo=coupled.low;
                if(coupled.ok){
                    if(lo.ok){'''
s=replace(s,old,new)
s=replace(s,'efh+=std::fabs(hi.vK-lo.vK);\n                        for(int j=0;j<5;++j){','efh+=2*std::fabs(hi.vK-lo.vK);\n                        for(int j=0;j<5;++j){')
s=replace(s,'edfh[j]+=jac*std::fabs(hi.dvK[j]-lo.dvK[j]);','edfh[j]+=2*jac*std::fabs(hi.dvK[j]-lo.dvK[j]);')
(OUT/'adaptive_epoch.hpp').write_text(s)
for name in ['seeded_radial_frontier.hpp','adaptive_epoch.hpp']:
    manifest[name]={'base_sha256':hashlib.sha256((BASE/name).read_bytes()).hexdigest(),'overlay_sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
