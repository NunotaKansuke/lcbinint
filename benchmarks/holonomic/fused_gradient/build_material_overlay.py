from pathlib import Path
import runpy,json,hashlib
runpy.run_path(str(Path(__file__).with_name('build_overlay.py')))
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'src/lcbinint/magnification/holonomic'
OUT=ROOT/'build-seeded-events/fused-gradient/include/lcbinint/magnification/holonomic'
def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
s=(BASE/'phi.hpp').read_text()
s=replace(s,'#include <cmath>','#include <cmath>\n#include "fused_motion.hpp"')
point='inline PhiGrad phi_grad('
pos=s.index(point)
s=s[:pos]+'''inline std::array<double,5> fused_material_phi_dp(const phi_detail::Core& c,
    double R,const PrimaryFrame& pf) {
    auto result=phi_detail::dP_from(c,pf);
    if(const auto* motion=fused_gradient::active_motion) {
        using namespace phi_detail;
        const C radial=(c.z+C{pf.m0,0}*c.inv_zb+
            C{c.m1,0}*(c.inv_zba*c.inv_zba)*c.zb)/C{R,0};
        const double slope=-2.0*(c.g.re*radial.re+c.g.im*radial.im)*c.inv_r2;
        for(int j=0;j<5;++j)result[j]+=slope*motion->radius[j];
    }
    return result;
}

'''+s[pos:]
s=replace(s,'out.dP = phi_detail::dP_from(c, pf);','out.dP = fused_material_phi_dp(c, R, pf);')
s=replace(s,'return {c.phi, phi_detail::dP_from(c, pf)};','return {c.phi, fused_material_phi_dp(c, R, pf)};')
(OUT/'phi.hpp').write_text(s)
s=(BASE/'holonomic_transport.hpp').read_text()
s=replace(s,'#pragma once','#pragma once\n#include "fused_motion.hpp"')
# This shared point routine is used by the experimental coupled K ladder.
s=replace(s,'        const double dw=wv*(c_S2*dS2+c_A*dA+c_B*dB);','''        if(const auto* motion=fused_gradient::active_motion)
            dB+=2.0*(reciprocal ? Rpa+Rma*t*t : Rma+Rpa*t*t)*motion->radius[j];
        const double dw=wv*(c_S2*dS2+c_A*dA+c_B*dB);''')
(OUT/'holonomic_transport.hpp').write_text(s)
s=(OUT/'seeded_radial_frontier.hpp').read_text()
s=replace(s,'''    if(cfg.effective_gradient_policy()==GradientPolicy::None && !cfg.with_jacobian) {
        exploration_cfg.weighted_radial=true;''','''    {
        exploration_cfg.weighted_radial=true;''')
s=replace(s,'if(u>0 && cfg.effective_gradient_policy()==GradientPolicy::None && !cfg.with_jacobian)', 'if(u>0 && ((cfg.effective_gradient_policy()==GradientPolicy::None && !cfg.with_jacobian) || cfg.fused_value_mesh))')
s=replace(s,'if(cfg.effective_gradient_policy()==GradientPolicy::None && !cfg.with_jacobian)\n        exploration_cfg.uniform_endpoint_budget_fraction=0.01;', 'if((cfg.effective_gradient_policy()==GradientPolicy::None && !cfg.with_jacobian) || cfg.fused_value_mesh)\n        exploration_cfg.uniform_endpoint_budget_fraction=0.01;')
(OUT/'seeded_radial_frontier.hpp').write_text(s)
s=(BASE/'adaptive_radial.hpp').read_text()
s=replace(s,'    bool with_jacobian=false,require_bound=false,fold_maps=true;', '    bool fused_value_mesh=false;\n    bool with_jacobian=false,require_bound=false,fold_maps=true;')
s=replace(s,'    double ld_radial_value=0,uniform_radial_value=0;','''    double ld_radial_value=0,uniform_radial_value=0;
    std::array<double,5> ld_radial_grad{},uniform_radial_grad{};''')
s=replace(s,'''        if(channel==1 && j==0)return s.uniform_radial_value;
        if(channel==2 && j==0)return s.ld_radial_value;''','''        if(channel==1)return j==0?s.uniform_radial_value:s.uniform_radial_grad[j-1];
        if(channel==2)return j==0?s.ld_radial_value:s.ld_radial_grad[j-1];''')
s=replace(s,'''            if(p.split_ld)s.ld_radial_value/=std::sqrt(std::max(0.0,1-x*x));''','''            if(p.split_ld) {
                const double weight=std::sqrt(std::max(0.0,1-x*x));
                s.ld_radial_value/=weight;
                for(auto& derivative:s.ld_radial_grad)derivative/=weight;
            }''')
a=s.index('        const double linear_q=p.q[0];');b=s.index('        return;\n    }\n    auto channel_value',a)
s=s[:a]+'''        const int nc=Components?Components:input_components;
        const auto linear_q=p.q;
        p.event=uniform.event;
        const auto& linear_rule=linear_radial_rule(p.level);
        const auto& uniform_rule=radial_rule(p.level,true);
        for(int j=0;j<nc;++j) {
            p.q[j]+=uniform.q[j];p.radial[j]+=uniform.radial[j];
            Sum inner,geometry,roundoff;
            for(int k=1;k<(1<<p.level);++k) {
                const auto& sample=w.samples[p.samples[k*(256/(1<<p.level))]];
                const double x=uniform_rule.x[k-1];
                const double wt=std::max(uniform_rule.w[k-1],
                    linear_rule.w[k-1]/std::sqrt(std::max(0.0,1-x*x)));
                const double uv=j?sample.uniform_radial_grad[j-1]:sample.uniform_radial_value;
                const double lv=j?sample.ld_radial_grad[j-1]:sample.ld_radial_value;
                inner.add(wt*sample.inner[j]);geometry.add(wt*sample.geometry[j]);
                roundoff.add(wt*sample.roundoff[j]+eps*(
                    std::fabs(uniform_rule.w[k-1]*uv)+std::fabs(linear_rule.w[k-1]*lv)));
            }
            p.inner[j]=inner.get();p.geometry[j]=geometry.get();
            p.roundoff[j]=roundoff.get()+eps*(std::fabs(linear_q[j])+std::fabs(p.q[j]));
            p.error[j]=p.radial[j]+p.inner[j]+p.geometry[j]+p.roundoff[j]+p.event[j];
            if(j)p.gradient_resolved[j-1]=p.gradient_resolved[j-1]&&uniform.gradient_resolved[j-1];
        }
        p.value_resolved=p.value_resolved&&uniform.value_resolved;
        p.resolved=p.value_resolved;
        for(int j=1;j<nc;++j)p.resolved=p.resolved&&p.gradient_resolved[j-1];
'''+s[b:]
s=replace(s,'policy==GradientPolicy::None && p.level>=4 && !p.value_resolved;', '(policy==GradientPolicy::None || !gradient_phase) && p.level>=4 && !p.value_resolved;')
# Value-mesh mode integrates all tangents but estimates errors only for mu.
# A finite unresolved sentinel preserves the existing result quality contract.
a=s.index('    auto channel_value=');b=s.index('    if(cfg.same_node_hermite_shadow',a)
section=s[a:b]
section=replace(section,'    const int nc=Components?Components:input_components;',
    '    const int nc=Components?Components:input_components;\n    const int error_components=cfg.fused_value_mesh?1:nc;')
section=replace(section,'q[j].add(wt*channel_value(s,j)); inn[j].add(wt*s.inner[j]);',
    'q[j].add(wt*channel_value(s,j));\n            if(j>=error_components)continue;\n            inn[j].add(wt*s.inner[j]);')
pos=section.index('    AdaptiveVector detail')
first,rest=section[:pos],section[pos:]
rest=rest.replace('j<nc','j<error_components').replace('nc==1','error_components==1')
# Finish all integral channels even though only the value error was estimated.
rest=replace(rest,'    for(int j=0;j<error_components;++j) {\n        p.q[j]=q[j].get();',
    '    for(int j=0;j<nc;++j) {\n        if(j>=error_components) {\n            p.q[j]=q[j].get();p.error[j]=kAdaptiveUnresolvedError;\n            p.gradient_resolved[j-1]=false;continue;\n        }\n        p.q[j]=q[j].get();')
s=s[:a]+first+rest+s[b:]
s=replace(s,'            p.q[j]+=uniform.q[j];p.radial[j]+=uniform.radial[j];',
    '            p.q[j]+=uniform.q[j];\n            if(cfg.fused_value_mesh && j>0) {\n                p.error[j]=kAdaptiveUnresolvedError;p.gradient_resolved[j-1]=false;continue;\n            }\n            p.radial[j]+=uniform.radial[j];')
(OUT/'adaptive_radial.hpp').write_text(s)
s=(OUT/'adaptive_epoch.hpp').read_text()
s=replace(s,'weighted_candidate && policy==GradientPolicy::None &&','weighted_candidate &&')
s=replace(s,'                auto coupled=fused_gradient::angular_pair', '                if(adaptive_cfg && adaptive_cfg->fused_value_mesh)gg=std::numeric_limits<double>::infinity();\n                auto coupled=fused_gradient::angular_pair')
s=replace(s,'    const std::size_t previous_panels=retain_matching_panels?', '''    std::vector<fused_gradient::MaterialMap> material_maps(integration_plan.spans.size());
    const std::size_t previous_panels=retain_matching_panels?''')
s=replace(s,'''            panel.split_ld=u!=0 && cfg.split_ld_radial;
            panel.weighted=true;panel.map.left=panel.map.right=false;''','''            bool material_ok=true;
            if(policy!=GradientPolicy::None) {
                const auto left=uncertainty(c.r_lo),right=uncertainty(c.r_hi);
                auto& material=material_maps[i];
                material=fused_gradient::make_material_map(left,right,pf);
                material_ok=material.valid;
            }
            if(material_ok) {
                panel.split_ld=u!=0 && cfg.split_ld_radial;
                panel.weighted=true;panel.map.left=panel.map.right=false;
            }''')
s=replace(s,'''        const CellPlan& cell=integration_plan.spans[static_cast<std::size_t>(i)].cell;''','''        const CellPlan& cell=integration_plan.spans[static_cast<std::size_t>(i)].cell;
        fused_gradient::Motion material_motion;
        const bool material_active=with_jacobian&&material_maps[i].valid;
        if(material_active)material_motion=material_maps[i].at(R,radius_lo,jac);
        fused_gradient::MotionScope motion_scope(material_active?&material_motion:nullptr);''')
s=replace(s,'QuarticParamJac dpc{};if(with_jac)dpc=boundary_quartic_dp(R,pf);','''QuarticParamJac dpc{};if(with_jac) {
        dpc=boundary_quartic_dp(R,pf);
        if(const auto* motion=fused_gradient::active_motion) {
            const auto radial=boundary_quartic_dR(R,pf);
            for(int j=0;j<5;++j)for(int k=0;k<5;++k)dpc.dp[j][k]+=radial.p[k]*motion->radius[j];
        }
    }''')
s=replace(s,'            df0[j]+=R*(dtl[j]-dte[j]);','''            df0[j]+=R*(dtl[j]-dte[j]);
            if(const auto* motion=fused_gradient::active_motion)
                df0[j]+=jac*motion->radius[j]*width;''')
s=replace(s,'            fh+=R*half*vh;efh+=R*half*std::fabs(vh-vl);','''            fh+=R*half*vh;efh+=R*half*std::fabs(vh-vl);
            if(with_jac)if(const auto* motion=fused_gradient::active_motion)
                for(int j=0;j<5;++j)dfh[j]+=jac*motion->radius[j]*half*vh;''')
s=replace(s,'''        auto d0=internal_to_user_jac(df0,p),dh=internal_to_user_jac(dfh,p);''','''        if(const auto* motion=fused_gradient::active_motion)for(int j=0;j<5;++j) {
            df0[j]+=motion->jacobian[j]*f0;
            dfh[j]+=motion->jacobian[j]*fh;
        }
        auto d0=internal_to_user_jac(df0,p),dh=internal_to_user_jac(dfh,p);
        if(adaptive_cfg && adaptive_cfg->split_ld_radial)for(int j=0;j<5;++j) {
            out.uniform_radial_grad[j]=(1-u)*d0[j]/D;
            out.ld_radial_grad[j]=u*dh[j]/D;
        }
        if(adaptive_cfg && adaptive_cfg->split_ld_radial) {
            out.uniform_radial_grad[2]-=2*out.uniform_radial_value/p.rho;
            out.ld_radial_grad[2]-=2*out.ld_radial_value/p.rho;
        }''')
s=replace(s,'adaptive_cfg && u>=0.0 && u<=1.0 && !with_jac', 'adaptive_cfg && u>=0.0 && u<=1.0 && (!with_jac || adaptive_cfg->fused_value_mesh)')
# The value-mesh result carries no derivative error bound. Avoid mapping
# five derivative-error columns after the actual tangents are already mapped.
s=replace(s,'        // Propagate absolute errors through the same linear map without cancellation.',
    """        if(adaptive_cfg && adaptive_cfg->fused_value_mesh) {
            for(int j=0;j<5;++j)out.value[j+1]=((1-u)*d0[j]+u*dh[j])/D;
            out.value[3]-=2*out.value[0]/p.rho;
            return out;
        }
        // Propagate absolute errors through the same linear map without cancellation.""")
s=replace(s,"""                bool regular=true;
                for(int j=0;j<5;++j) {
                    auto d=root_pair_dR(RootPair{ap.m,ap.v},cp.p,dp.dp[j]);""",
    """                bool regular=true;
                const auto tangents=fused_gradient::root_pair_tangents(RootPair{ap.m,ap.v},cp.p,dp.dp);
                for(int j=0;j<5;++j) {
                    const auto& d=tangents[j];""")
(OUT/'adaptive_epoch.hpp').write_text(s)
manifest={name:{'base_sha256':hashlib.sha256((BASE/name).read_bytes()).hexdigest(),
    'overlay_sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest()} for name in
    ['adaptive_radial.hpp','adaptive_epoch.hpp','phi.hpp','holonomic_transport.hpp','seeded_radial_frontier.hpp']}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
