#pragma once
#include "lcbinint/magnification/holonomic/adaptive_radial.hpp"
#include <memory>

namespace lcbinint::holonomic::adaptive_detail {
// Estimate the uncertainty of a physical event from P=Pt=0 using the
// original-frame qf polynomial family. qf stationary-root correction and
// nonzero Ptt/PR are local ordinary-fold checks, not D14 completeness proof.
inline bool valid_epoch_config(const LensParams& p,double u,const AdaptiveConfig& cfg) {
    return valid_config(cfg)&&std::isfinite(u)&&u>=0&&u<=1&&
        std::isfinite(p.xs)&&std::isfinite(p.ys)&&std::isfinite(p.rho)&&p.rho>0&&
        std::isfinite(p.q)&&p.q>0&&std::isfinite(p.a)&&p.a>0;
}
struct EventLocation {
    double radius,uncertainty,radius_lo=0;
    bool needs_qf=true;
    bool qf_refined=false;
    int precision_tier=0; // 0=double, 1=DD, 2=__float128
    EventDecisionReason double_reason=EventDecisionReason::NotAttempted;
    EventDecisionReason dd_reason=EventDecisionReason::NotAttempted;
    EventDecisionReason qf_reason=EventDecisionReason::NotAttempted;
    double double_residual=std::numeric_limits<double>::infinity();
    double dd_residual=std::numeric_limits<double>::infinity();
};
inline double eval_poly5(const std::array<double,5>& c,double x) {
    return c[4]*x+c[3]*x*x+c[2]*x*x*x+c[1]*x*x*x*x+c[0];
}
inline double eval_poly5_derivative(const std::array<double,5>& c,double x) {
    return c[1]+x*(2*c[2]+x*(3*c[3]+x*4*c[4]));
}
inline EventLocation double_event_estimate(double R,const PrimaryFrame& pf,double value_budget) {
    const auto q=boundary_quartic(R,pf);
    const auto qr=boundary_quartic_dR(R,pf);
    auto roots=aberth<double>(q.p.data(),4,32);
    EventLocation best{R,std::numeric_limits<double>::infinity(),0,true,false,0,
                       EventDecisionReason::NoRealCandidate,
                       EventDecisionReason::NotAttempted,
                       EventDecisionReason::NotAttempted};
    double best_res=std::numeric_limits<double>::infinity();
    double best_t=0,best_ap=1,best_at=1,best_ar=1;
    auto consider=[&](double t) {
        double ap=0,at=0,ar=0,att=0,pow=1,dpow=1;
        for(int k=0;k<5;++k) {
            ap+=std::fabs(q.p[k])*pow;
            ar+=std::fabs(qr.p[k])*pow;
            if(k>0)at+=k*std::fabs(q.p[k])*dpow;
            if(k>1)att+=k*(k-1)*std::fabs(q.p[k])*((k==2)?1:std::pow(std::fabs(t),k-2));
            pow*=std::fabs(t);
            if(k>0)dpow*=std::fabs(t);
        }
        ap=std::max(1.0,ap);at=std::max(1.0,at);ar=std::max(1.0,ar);att=std::max(1.0,att);
        const double P=eval_poly5(q.p,t),Pt=eval_poly5_derivative(q.p,t),PR=eval_poly5(qr.p,t);
        const double p_res=std::fabs(P)/ap,t_res=std::fabs(Pt)/at,res=std::max(p_res,t_res);
        if(res>=best_res)return;
        best_res=res;
        best_t=t;best_ap=ap;best_at=at;best_ar=ar;
        best.double_residual=res;
        const double ulp=std::fabs(std::nextafter(R,std::numeric_limits<double>::infinity())-R);
        const double radius_shift=std::fabs(PR)>64*eps*ar?std::fabs(P)/std::fabs(PR):std::numeric_limits<double>::infinity();
        best.uncertainty=std::max(8*ulp,64*eps*(1+std::fabs(R))+radius_shift);
        const bool numerically_clean=res<=4096*eps&&std::fabs(PR)>64*eps*ar&&
            std::fabs(eval_poly5(q.p,t+std::sqrt(eps))-2*P+eval_poly5(q.p,t-std::sqrt(eps)))>
            64*eps*att;
        // A physical event is used as the endpoint of a squared fold map.
        // Its binary64 anchor therefore needs a representation-level
        // certificate in addition to the value ledger: allowing a coarse
        // event merely because a loose value tolerance can absorb it makes
        // the controller non-monotone (the first mapped Fejer node can lie
        // inside the reported event uncertainty).  The minimum is a local
        // ULP/conditioning test, not a physical-parameter route.
        const double budget_radius=0.05*value_budget*std::max(1.0,std::fabs(R));
        const double ulp_budget=std::max(256*ulp,64*eps*(1+std::fabs(R)));
        const double anchor_budget=std::min(ulp_budget,budget_radius);
        if(numerically_clean && best.uncertainty<=anchor_budget) {
            best.double_reason=EventDecisionReason::CleanDouble;
            best.qf_reason=EventDecisionReason::NotAttempted;
            best.needs_qf=false;
        } else if(best.uncertainty<=anchor_budget) {
            best.double_reason=EventDecisionReason::DoubleBudgetAccepted;
            best.qf_reason=EventDecisionReason::NotAttempted;
            best.needs_qf=false;
        } else {
            best.double_reason=EventDecisionReason::DoubleAmbiguous;
            best.qf_reason=EventDecisionReason::QfRequired;
            best.needs_qf=true;
        }
    };
    for(const auto& z:roots) {
        if(std::fabs(z.im)>1e-7*(1+std::fabs(z.re)))continue;
        consider(z.re);
    }
    // At an ordinary fold the quartic has a double real root, which a
    // binary64 all-root solve may report as a complex pair. The stationary
    // root of P' is simple and is a cheaper, branch-independent local probe.
    if(!std::isfinite(best_res)) {
        double dc[4]={4*q.p[4],3*q.p[3],2*q.p[2],q.p[1]};
        int deg=3;while(deg&&dc[0]==0){for(int j=0;j<deg;++j)dc[j]=dc[j+1];--deg;}
        if(deg>0)for(const auto& z:aberth<double>(dc,deg,32))
            if(std::fabs(z.im)<=1e-7*(1+std::fabs(z.re)))consider(z.re);
    }
    if(std::isfinite(best_res)&&best.needs_qf) {
        const DD tt(best_t);
        auto dd_eval=[&](const std::array<double,5>& c) {
            DD v(c[4]);
            for(int k=3;k>=0;--k)v=v*tt+DD(c[k]);
            return v;
        };
        auto dd_deriv=[&](const std::array<double,5>& c) {
            DD v(c[4]*4.0);
            for(int k=3;k>=1;--k)v=v*tt+DD(c[k]*k);
            return v;
        };
        const __float128 pdd=fabsq(qf_from_dd(dd_eval(q.p)));
        const __float128 tdd=fabsq(qf_from_dd(dd_deriv(q.p)));
        const __float128 rdd=fabsq(qf_from_dd(dd_eval(qr.p)));
        const double dd_res=std::max((double)(pdd/(__float128)best_ap),
                                     (double)(tdd/(__float128)best_at));
        best.dd_residual=dd_res;
        const double ulp=std::fabs(std::nextafter(R,std::numeric_limits<double>::infinity())-R);
        const double dd_shift=(double)(rdd>0?pdd/rdd:1e300L);
        const double budget_radius=0.05*value_budget*std::max(1.0,std::fabs(R));
        const double anchor_budget=std::min(std::max(256*ulp,64*eps*(1+std::fabs(R))),budget_radius);
        const bool dd_residual_ok=dd_res<=1e-14;
        const bool dd_derivative_ok=rdd>64*(__float128)eps*(__float128)best_ar;
        const bool dd_budget_ok=std::max(8*ulp,64*eps*(1+std::fabs(R))+dd_shift)<=anchor_budget;
        if(dd_residual_ok&&dd_derivative_ok&&dd_budget_ok) {
            best.needs_qf=false;best.precision_tier=1;
            best.dd_reason=EventDecisionReason::DDAccepted;
            best.qf_reason=EventDecisionReason::NotAttempted;
            best.uncertainty=std::max(8*ulp,64*eps*(1+std::fabs(R))+dd_shift);
        } else {
            best.dd_reason=!dd_residual_ok ? EventDecisionReason::DDResidualRejected :
                (!dd_derivative_ok ? EventDecisionReason::DDDerivativeRejected :
                 EventDecisionReason::DDBudgetRejected);
        }
    }
    if(best.needs_qf && best.double_reason==EventDecisionReason::NoRealCandidate)
        best.qf_reason=EventDecisionReason::QfFailed;
    return best;
}
inline EventLocation refine_event(double R,const PrimaryFrame& pf,const re_detail::PolyFamilyR& fam) {
    using Q=__float128;
    auto q=boundary_quartic(R,pf);
    double dc[4]={4*q.p[4],3*q.p[3],2*q.p[2],q.p[1]};
    int deg=3;while(deg && dc[0]==0){for(int j=0;j<deg;++j)dc[j]=dc[j+1];--deg;}
    EventLocation best{R,std::numeric_limits<double>::infinity(),0,true,true,2,
                       EventDecisionReason::NotAttempted,
                       EventDecisionReason::NotAttempted,
                       EventDecisionReason::QfRequired};
    if(!deg)return best;
    auto roots=aberth<double>(dc,deg,24);
    Q nearest=HUGE_VALQ;
    for(auto z:roots)if(std::fabs(z.im)<1e-7*(1+std::fabs(z.re))){
        Q t=z.re,r=R,last=HUGE_VALQ;
        bool regular=true;
        for(int it=0;it<16;++it){
            Q pc[5]{},pr[5]{};
            for(int j=0;j<5;++j){auto& f=fam.p[j];for(int k=f.deg;k>=0;--k)pc[j]=pc[j]*r+f.c[k];for(int k=f.deg;k>0;--k)pr[j]=pr[j]*r+Q(k)*f.c[k];}
            Q P=pc[4],PR=pr[4];for(int j=3;j>=0;--j){P=P*t+pc[j];PR=PR*t+pr[j];}
            Q Pt=pc[1]+t*(2*pc[2]+t*(3*pc[3]+t*4*pc[4]));
            Q Ptt=2*pc[2]+t*(6*pc[3]+t*12*pc[4]);
            Q Ptr=pr[1]+t*(2*pr[2]+t*(3*pr[3]+t*4*pr[4]));
            Q det=PR*Ptt-Pt*Ptr;
            // Numerical separation screen in qf; this remains Estimated.
            if(fabsq(det)<=64*FLT128_EPSILON*(fabsq(PR*Ptt)+fabsq(Pt*Ptr))||PR==0||Ptt==0){regular=false;break;}
            Q dr=(P*Ptt-Pt*Pt)/det,dt=(PR*Pt-Ptr*P)/det;
            r-=dr;t-=dt;last=fabsq(dr);
            if(last<1e-30Q*(1+fabsq(r))&&fabsq(dt)<1e-30Q*(1+fabsq(t)))break;
        }
        if(!regular||!finiteq(r)||last>1e-25Q*(1+fabsq(r)))continue;
        Q shift=fabsq(r-Q(R));
        if(shift<nearest){nearest=shift;double rounded=double(r);
            best={rounded,double(fabsq(Q(rounded)-r)+last)+std::fabs(std::nextafter(rounded,INFINITY)-rounded),double(r-Q(rounded)),false,true,2,
                  EventDecisionReason::NotAttempted,EventDecisionReason::NotAttempted,
                  EventDecisionReason::QfRefined};}
    }
    return best;
}
// The fixed-resolution planner merges close representation / physical
// events. Restore retained physical cuts for adaptive integration; otherwise
// a fold lies inside a nominally smooth panel. No angular sampling authority.
inline bool restore_physical_cuts(const TopologyResult& topo,const PrimaryFrame& pf,
                                  std::vector<CellPlan>& cells) {
    double end=0;
    for(const auto& c:topo.cells) {
        // Skipped intervals have no certified geometry here. Do not silently
        // assign zero contribution or call it a radial tolerance success.
        if(c.r_lo!=end)return false;
        end=c.r_hi;
        std::vector<double> cuts{c.r_lo,c.r_hi};
        for(const auto& e:topo.events)if(e.physically_real&&e.kind=="physical_real"&&e.radius>c.r_lo&&e.radius<c.r_hi)cuts.push_back(e.radius);
        std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
        if(cuts.size()==2){cells.push_back(c);continue;}
        for(size_t j=1;j<cuts.size();++j) {
            double mid=cuts[j-1]+.5*(cuts[j]-cuts[j-1]);
            if(!(mid>cuts[j-1]&&mid<cuts[j]))return false;
            auto g=quartic_topology(mid,pf);
            if(!g.certified||g.n_crossings<0||(g.n_crossings&1))return false;
            cells.push_back(CellPlan{int(cells.size()),cuts[j-1],cuts[j],mid,g.kind,g.n_crossings,Status::OK});
        }
    }
    return end==topo.r_max;
}
// Reuses V2 arc/root continuation, endpoint IFT and vK. The adapter carries
// numerical estimates separately from the radial interpolation detail.
inline AdaptiveSample mapped_radius(double R,double jac,const LensParams& p,
    double u,const PrimaryFrame& pf,const CellPlan& cell,bool with_jac,
    bool warm,const AdaptiveSample* seed) {
    AdaptiveSample out;
    if(seed){out.quartic=seed->quartic;out.roots=seed->roots;}
    out.roots.certify=warm;
    auto arcs=arc_intervals(R,pf,&out.quartic,&out.roots);
    if(arcs.kind!=cell.kind || (arcs.kind==ArcKind::kArcs && int(arcs.arcs.size()*2)!=cell.n_crossings)) {out.reliable=false;return out;}
    if(arcs.kind==ArcKind::kFull || arcs.kind==ArcKind::kDegenerate){out.reliable=false;return out;}
    const double D=kPi*p.rho*p.rho*(1-u/3),scale=jac/D;
    double f0=0,fh=0,ef0=0,efh=0;
    std::array<double,5> df0{},dfh{},edf0{},edfh{};
    const auto pc=u!=0?boundary_quartic(R,pf):QuarticCoeffs{};
    QuarticParamJac dpc{};if(with_jac)dpc=boundary_quartic_dp(R,pf);
    for(auto a:arcs.arcs) {
        auto pe=polish_endpoint(R,a[0],pf),pl=polish_endpoint(R,a[1],pf);
        double te=pe.theta,tl=pl.theta;if(tl<=te)tl+=kTwoPi;
        // No flag override: transformed arithmetic still requires resolvable endpoints.
        if(!pe.reliable||!pl.reliable){out.reliable=false;return out;}
        PhiGrad ge{},gl{};
        if(with_jac){ge=phi_grad(R,te,pf);gl=phi_grad(R,tl,pf);}
        else{auto e=phi_val_dtheta(R,te,pf),l=phi_val_dtheta(R,tl,pf);ge.phi=e.phi;ge.dphi_dtheta=e.dphi_dtheta;gl.phi=l.phi;gl.dphi_dtheta=l.dphi_dtheta;}
        double de=std::fabs(ge.phi/ge.dphi_dtheta),dl=std::fabs(gl.phi/gl.dphi_dtheta);
        // Floating point lens-map cancellation grows as 1/rho. This is an
        // arithmetic sensitivity estimate, NOT an interval enclosure.
        double map_noise=eps*(std::fabs(R)+std::fabs(pf.X)+std::fabs(pf.Y)+1)/pf.rho;
        de+=(map_noise+eps*std::fabs(te)*std::fabs(ge.dphi_dtheta))/std::fabs(ge.dphi_dtheta);
        dl+=(map_noise+eps*std::fabs(tl)*std::fabs(gl.dphi_dtheta))/std::fabs(gl.dphi_dtheta);
        const double width=tl-te;
        if(!(width>de+dl) || !std::isfinite(de+dl)){out.reliable=false;return out;}
        f0+=R*width;ef0+=R*(de+dl);
        std::array<double,5> dte{},dtl{};
        if(with_jac)for(int j=0;j<5;++j){
            // Form the mapped derivative before division by the small fold
            // derivative: never construct the divergent fixed-R dtheta first.
            dte[j]=-(jac*ge.dP[j])/ge.dphi_dtheta;
            dtl[j]=-(jac*gl.dP[j])/gl.dphi_dtheta;
            df0[j]+=R*(dtl[j]-dte[j]);
            edf0[j]+=R*(std::fabs(dte[j])+std::fabs(dtl[j]))*((de+dl)/width+eps);
        }
        if(u==0)continue;
        bool success=false;
        // Same exact t / reciprocal choices and gates as V2. Parameter
        // derivatives come from regular E=O=0, not singular endpoint IFTs.
        const std::array<double,5> zero{};
        for(bool reciprocal:{false,true}) {
            auto ap=reciprocal?arc_pair_jac_reciprocal(te,tl,zero,zero):arc_pair_jac(te,tl,zero,zero);
            if(!ap.ok)continue;
            auto cp=reciprocal?boundary_quartic_reciprocal(pc):pc;
            auto dp=reciprocal?boundary_quartic_reciprocal_dp(dpc):dpc;
            if(with_jac) {
                bool regular=true;
                for(int j=0;j<5;++j) {
                    auto d=root_pair_dR(RootPair{ap.m,ap.v},cp.p,dp.dp[j]);
                    regular=regular&&d.ok&&std::isfinite(d.dm_dR)&&std::isfinite(d.dv_dR);
                    ap.dm[j]=d.dm_dR;ap.dv[j]=d.dv_dR;
                }
                if(!regular)continue;
                auto hi=v_times_K_jac(ap.m,ap.v,ap.dm,ap.dv,R,pf,cp.p,dp.dp,reciprocal);
                if(hi.ok){auto lo=v_times_K_jac(ap.m,ap.v,ap.dm,ap.dv,R,pf,cp.p,dp.dp,reciprocal,8);
                    if(lo.ok){fh+=hi.vK;efh+=std::fabs(hi.vK-lo.vK);for(int j=0;j<5;++j){dfh[j]+=jac*hi.dvK[j];edfh[j]+=jac*std::fabs(hi.dvK[j]-lo.dvK[j]);}success=true;}}
            }else{auto vk=v_times_K(ap.m,ap.v,R,pf,cp.p,reciprocal);if(vk.ok){fh+=vk.vK;efh+=vk.estimated_error;success=true;}}
            if(success)break;
        }
        if(!success){
            // Retain V2's angular rescue. A second, cheap 32-point rule supplies
            // the missing inner-error estimate; radial refinement cannot fix it.
            static const Cheb1<32> low;
            const auto& high=ang_rule();
            double vh=0,vl=0;std::array<double,5> dh{},dd{};
            double half=0.5*width,mid=te+half;
            for(int k=0;k<64;++k){double th=mid+half*high.x[k];
                if(with_jac){auto g=phi_val_dP(R,th,pf);if(!(g.phi>0)){out.reliable=false;return out;}double sq=std::sqrt(g.phi);vh+=high.w[k]*sq;for(int j=0;j<5;++j)dh[j]+=high.w[k]*(jac*g.dP[j])/(2*sq);}
                else{double ph=phi_val(R,th,pf);if(!(ph>0)){out.reliable=false;return out;}vh+=high.w[k]*std::sqrt(ph);}}
            for(int k=0;k<32;++k){double th=mid+half*low.x[k];
                if(with_jac){auto g=phi_val_dP(R,th,pf);if(!(g.phi>0)){out.reliable=false;return out;}double sq=std::sqrt(g.phi);vl+=low.w[k]*sq;for(int j=0;j<5;++j)dd[j]+=low.w[k]*(jac*g.dP[j])/(2*sq);}
                else{double ph=phi_val(R,th,pf);if(!(ph>0)){out.reliable=false;return out;}vl+=low.w[k]*std::sqrt(ph);}}
            fh+=R*half*vh;efh+=R*half*std::fabs(vh-vl);
            if(with_jac)for(int j=0;j<5;++j){dfh[j]+=R*half*dh[j];edfh[j]+=R*half*std::fabs(dh[j]-dd[j]);}
        }
    }
    out.value[0]=scale*((1-u)*f0+u*fh);
    out.inner[0]=std::fabs(scale*u)*efh;
    out.geometry[0]=std::fabs(scale*(1-u))*ef0+std::fabs(scale*u)*ef0;
    out.roundoff[0]=eps*std::fabs(scale)*(std::fabs((1-u)*f0)+std::fabs(u*fh));
    if(with_jac){
        auto d0=internal_to_user_jac(df0,p),dh=internal_to_user_jac(dfh,p);
        // Propagate absolute errors through the same linear map without cancellation.
        auto abs_map=[&](const std::array<double,5>& v){std::array<double,5> e{};
            for(int k=0;k<5;++k){std::array<double,5> unit{};unit[k]=1;auto col=internal_to_user_jac(unit,p);for(int j=0;j<5;++j)e[j]+=std::fabs(col[j])*v[k];}return e;};
        auto e0=abs_map(edf0),eh=abs_map(edfh);
        for(int j=0;j<5;++j){out.value[j+1]=((1-u)*d0[j]+u*dh[j])/D;out.inner[j+1]=std::fabs(u/D)*eh[j];out.geometry[j+1]=std::fabs((1-u)/D)*e0[j];
            out.roundoff[j+1]=eps/std::fabs(D)*(std::fabs((1-u)*d0[j])+std::fabs(u*dh[j]));}
        out.value[3]-=2*out.value[0]/p.rho;
        out.inner[3]+=2*out.inner[0]/p.rho;out.geometry[3]+=2*out.geometry[0]/p.rho;out.roundoff[3]+=2*(out.roundoff[0]+eps*std::fabs(out.value[0]))/p.rho;
    }
    return out;
}
}
namespace lcbinint::holonomic {
inline AdaptiveResult flux_adaptive_integrate(const LensParams& p,double u,
    const TopologyResult& topo,const AdaptiveConfig& cfg,AdaptiveWorkspace& workspace) {
    const ScopedFlushDenormals fp_guard;
    workspace.reset();
    const auto policy=cfg.effective_gradient_policy();
    if(!adaptive_detail::valid_epoch_config(p,u,cfg)) return adaptive_detail::failure_result(AdaptiveStop::InvalidConfig,policy);
    if(cfg.require_bound) return adaptive_detail::failure_result(AdaptiveStop::BoundUnavailable,policy);
    if(topo.status!=Status::OK) return adaptive_detail::failure_result(AdaptiveStop::TopologyUnresolved,policy);
    auto setup_start=adaptive_detail::Clock::now();
    auto frame_start=adaptive_detail::Clock::now();
    const auto pf=PrimaryFrame::from(p);
    const double setup_frame_ms=adaptive_detail::ms(frame_start);
    if(near_origin_source(pf)) return adaptive_detail::failure_result(AdaptiveStop::TopologyUnresolved,policy);
    auto& cells=workspace.cells;
    auto cuts_start=adaptive_detail::Clock::now();
    if(!adaptive_detail::restore_physical_cuts(topo,pf,cells)) return adaptive_detail::failure_result(AdaptiveStop::TopologyUnresolved,policy);
    const double setup_cuts_ms=adaptive_detail::ms(cuts_start);
    // A squared map is a valid substitution even if a physical event is a
    // higher contact: we make no smooth-fold guarantee from its label alone.
    auto physical=[&](double R){for(const auto& e:topo.events)if(e.radius==R && e.physically_real && e.kind=="physical_real")return true;return false;};
    std::vector<std::pair<double,adaptive_detail::EventLocation>> event_errors;
    std::unique_ptr<re_detail::PolyFamilyR> fam;
    bool fam_ready=false;
    size_t event_double_checks=0,event_dd_checks=0,event_dd_accepts=0,event_qf_refinements=0,qf_family_constructions=0;
    std::vector<AdaptiveEventDiagnostic> event_diagnostics;
    double setup_event_ms=0;
    double event_budget=cfg.tol.budget(0,1.0);
    if(policy==GradientPolicy::Strict)for(int j=1;j<6;++j)event_budget=std::min(event_budget,cfg.tol.budget(j,1.0));
    auto uncertainty=[&](double R){
        auto event_call_start=adaptive_detail::Clock::now();
        for(const auto& e:event_errors)if(e.first==R){setup_event_ms+=adaptive_detail::ms(event_call_start);return e.second;}
        ++event_double_checks;
        auto d=adaptive_detail::double_event_estimate(R,pf,event_budget);
        if(d.precision_tier>=1)++event_dd_checks;
        if(d.precision_tier==1)++event_dd_accepts;
        if(d.needs_qf) {
            if(!fam_ready) {
                fam=std::make_unique<re_detail::PolyFamilyR>(re_detail::p_coeffs_in_R(pf.a,pf.m0,pf.X,pf.Y,pf.rho));
                fam_ready=true;
                ++qf_family_constructions;
            }
            const auto double_reason=d.double_reason;
            const auto dd_reason=d.dd_reason;
            const double double_residual=d.double_residual;
            const double dd_residual=d.dd_residual;
            d=adaptive_detail::refine_event(R,pf,*fam);
            d.double_reason=double_reason;d.dd_reason=dd_reason;
            d.double_residual=double_residual;d.dd_residual=dd_residual;
            d.qf_reason=d.needs_qf?EventDecisionReason::QfFailed:EventDecisionReason::QfRefined;
            ++event_qf_refinements;
        }
        if(cfg.collect_diagnostics) {
            AdaptiveEventDiagnostic record;
            record.input_radius=R;record.selected_radius=d.radius;
            record.radius_lo=d.radius_lo;record.uncertainty=d.uncertainty;
            record.precision_tier=d.precision_tier;
            record.double_reason=d.double_reason;record.dd_reason=d.dd_reason;
            record.qf_reason=d.qf_reason;
            record.double_residual=d.double_residual;record.dd_residual=d.dd_residual;
            event_diagnostics.push_back(record);
        }
        event_errors.emplace_back(R,d);
        setup_event_ms+=adaptive_detail::ms(event_call_start);
        return d;};
    auto panel_start=adaptive_detail::Clock::now();
    for(size_t i=0;i<cells.size();++i){const auto& c=cells[i];if(c.kind==ArcKind::kEmpty)continue;
        if(workspace.panels.size()>=cfg.max_panels) return adaptive_detail::failure_result(AdaptiveStop::BudgetExceeded,policy);
        AdaptivePanel panel;panel.cell=i;panel.map={c.r_lo,c.r_hi,cfg.fold_maps&&physical(c.r_lo),cfg.fold_maps&&physical(c.r_hi)};if(physical(c.r_lo)){auto e=uncertainty(c.r_lo);panel.map.a=e.radius;panel.left_radius_lo=e.radius_lo;panel.left_uncertainty=e.uncertainty;}
        if(physical(c.r_hi)){auto e=uncertainty(c.r_hi);panel.map.b=e.radius;panel.right_radius_lo=e.radius_lo;panel.right_uncertainty=e.uncertainty;}
        // A local correction may not jump across another cell/event.
        if(std::fabs(panel.map.a-c.r_lo)>.25*(c.r_hi-c.r_lo)||std::fabs(panel.map.b-c.r_hi)>.25*(c.r_hi-c.r_lo)||!(panel.map.a<panel.map.b)) return adaptive_detail::failure_result(AdaptiveStop::EventLocationLimited,policy);
        if(!std::isfinite(panel.left_uncertainty+panel.right_uncertainty)) return adaptive_detail::failure_result(AdaptiveStop::EventLocationLimited,policy);
        workspace.panels.push_back(panel);}
    const bool with_jacobian=policy!=GradientPolicy::None;
    const double panel_total_ms=adaptive_detail::ms(panel_start);
    const double setup_panel_ms=std::max(0.0,panel_total_ms-setup_event_ms);
    auto result=adaptive_detail::integrate(workspace,cfg,[&](double R,double jac,int i,const AdaptiveSample* seed){return adaptive_detail::mapped_radius(R,jac,p,u,pf,cells[i],with_jacobian,topo.from_warm_d14,seed);});
    result.stats.setup_ms=adaptive_detail::ms(setup_start);
    result.stats.setup_frame_ms=setup_frame_ms;
    result.stats.setup_cuts_ms=setup_cuts_ms;
    result.stats.setup_event_ms=setup_event_ms;
    result.stats.setup_panel_ms=setup_panel_ms;
    result.stats.event_double_checks=event_double_checks;
    result.stats.event_dd_checks=event_dd_checks;
    result.stats.event_dd_accepts=event_dd_accepts;
    result.stats.event_qf_refinements=event_qf_refinements;
    result.stats.qf_family_constructions=qf_family_constructions;
    if(cfg.collect_diagnostics) result.stats.event_diagnostics=std::move(event_diagnostics);
    return result;
}
inline AdaptiveResult epoch_adaptive(const LensParams& p,double u,const AdaptiveConfig& cfg,AdaptiveWorkspace& w) {
    if(!adaptive_detail::valid_epoch_config(p,u,cfg)){w.reset();return adaptive_detail::failure_result(AdaptiveStop::InvalidConfig,cfg.effective_gradient_policy());}
    if(cfg.require_bound){w.reset();return adaptive_detail::failure_result(AdaptiveStop::BoundUnavailable,cfg.effective_gradient_policy());}
    auto start=adaptive_detail::Clock::now();auto topo=classify_cells(PrimaryFrame::from(p));double t=adaptive_detail::ms(start);
    auto out=flux_adaptive_integrate(p,u,topo,cfg,w);out.stats.topology_ms=t;return out;
}
inline AdaptiveResult epoch_value_adaptive(const LensParams& p,double u,AdaptiveConfig cfg,AdaptiveWorkspace& w) {cfg.gradient_policy=GradientPolicy::None;cfg.with_jacobian=false;return epoch_adaptive(p,u,cfg,w);}
inline AdaptiveResult epoch_jacobian_adaptive(const LensParams& p,double u,AdaptiveConfig cfg,AdaptiveWorkspace& w) {cfg.with_jacobian=true;return epoch_adaptive(p,u,cfg,w);}
inline AdaptiveResult epoch_adaptive_prepared(const LensParams& p,double u,const AdaptiveConfig& cfg,AdaptiveWorkspace& w,PreparedEpochGeometry& state,const PreparedReuseConfig& reuse=PreparedReuseConfig{}) {
    if(!adaptive_detail::valid_epoch_config(p,u,cfg)){w.reset();return adaptive_detail::failure_result(AdaptiveStop::InvalidConfig,cfg.effective_gradient_policy());}
    if(cfg.require_bound){w.reset();return adaptive_detail::failure_result(AdaptiveStop::BoundUnavailable,cfg.effective_gradient_policy());}
    auto start=adaptive_detail::Clock::now();auto topo=prepared_topology(PrimaryFrame::from(p),state,reuse,nullptr);double t=adaptive_detail::ms(start);
    auto out=flux_adaptive_integrate(p,u,topo,cfg,w);out.stats.topology_ms=t;return out;
}
}
