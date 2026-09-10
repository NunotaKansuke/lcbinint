#pragma once
#include "lcbinint/magnification/holonomic/adaptive_radial.hpp"

namespace lcbinint::holonomic::adaptive_detail {
// Estimate the uncertainty of a physical event from P=Pt=0 using the
// original-frame qf polynomial family. qf stationary-root correction and
// nonzero Ptt/PR are local ordinary-fold checks, not D14 completeness proof.
inline bool valid_epoch_config(const LensParams& p,double u,const AdaptiveConfig& cfg) {
    return valid_config(cfg)&&std::isfinite(u)&&u>=0&&u<=1&&
        std::isfinite(p.xs)&&std::isfinite(p.ys)&&std::isfinite(p.rho)&&p.rho>0&&
        std::isfinite(p.q)&&p.q>0&&std::isfinite(p.a)&&p.a>0;
}
struct EventLocation { double radius,uncertainty; };
inline EventLocation refine_event(double R,const PrimaryFrame& pf,const re_detail::PolyFamilyR& fam) {
    using Q=__float128;
    auto q=boundary_quartic(R,pf);
    double dc[4]={4*q.p[4],3*q.p[3],2*q.p[2],q.p[1]};
    int deg=3;while(deg && dc[0]==0){for(int j=0;j<deg;++j)dc[j]=dc[j+1];--deg;}
    EventLocation best{R,std::numeric_limits<double>::infinity()};
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
            best={rounded,double(fabsq(Q(rounded)-r)+last)+std::fabs(std::nextafter(rounded,INFINITY)-rounded)};}
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
    if(!adaptive_detail::valid_epoch_config(p,u,cfg)) {AdaptiveResult r;r.stop=AdaptiveStop::InvalidConfig;return r;}
    if(cfg.require_bound){AdaptiveResult r;r.stop=AdaptiveStop::BoundUnavailable;return r;}
    if(topo.status!=Status::OK){AdaptiveResult r;r.stop=AdaptiveStop::TopologyUnresolved;return r;}
    auto setup_start=adaptive_detail::Clock::now();
    const auto pf=PrimaryFrame::from(p);
    if(near_origin_source(pf)){AdaptiveResult r;r.stop=AdaptiveStop::TopologyUnresolved;return r;}
    const auto fam=re_detail::p_coeffs_in_R(pf.a,pf.m0,pf.X,pf.Y,pf.rho);
    std::vector<CellPlan> cells;
    if(!adaptive_detail::restore_physical_cuts(topo,pf,cells)){AdaptiveResult r;r.stop=AdaptiveStop::TopologyUnresolved;return r;}
    // A squared map is a valid substitution even if a physical event is a
    // higher contact: we make no smooth-fold guarantee from its label alone.
    auto physical=[&](double R){for(const auto& e:topo.events)if(e.radius==R && e.physically_real && e.kind=="physical_real")return true;return false;};
    std::vector<std::pair<double,adaptive_detail::EventLocation>> event_errors;
    auto uncertainty=[&](double R){for(auto e:event_errors)if(e.first==R)return e.second;
        auto d=adaptive_detail::refine_event(R,pf,fam);event_errors.emplace_back(R,d);return d;};
    for(size_t i=0;i<cells.size();++i){const auto& c=cells[i];if(c.kind==ArcKind::kEmpty)continue;
        if(workspace.panels.size()>=cfg.max_panels){AdaptiveResult r;r.stop=AdaptiveStop::BudgetExceeded;return r;}
        AdaptivePanel panel;panel.cell=i;panel.map={c.r_lo,c.r_hi,cfg.fold_maps&&physical(c.r_lo),cfg.fold_maps&&physical(c.r_hi)};if(physical(c.r_lo)){auto e=uncertainty(c.r_lo);panel.map.a=e.radius;panel.left_uncertainty=e.uncertainty;}
        if(physical(c.r_hi)){auto e=uncertainty(c.r_hi);panel.map.b=e.radius;panel.right_uncertainty=e.uncertainty;}
        // A local correction may not jump across another cell/event.
        if(std::fabs(panel.map.a-c.r_lo)>.25*(c.r_hi-c.r_lo)||std::fabs(panel.map.b-c.r_hi)>.25*(c.r_hi-c.r_lo)||!(panel.map.a<panel.map.b)){AdaptiveResult r;r.stop=AdaptiveStop::EventLocationLimited;return r;}
        if(!std::isfinite(panel.left_uncertainty+panel.right_uncertainty)){AdaptiveResult r;r.stop=AdaptiveStop::EventLocationLimited;return r;}
        workspace.panels.push_back(panel);}
    double setup_ms=adaptive_detail::ms(setup_start);
    auto result=adaptive_detail::integrate(workspace,cfg,[&](double R,double jac,int i,const AdaptiveSample* seed){return adaptive_detail::mapped_radius(R,jac,p,u,pf,cells[i],cfg.with_jacobian,topo.from_warm_d14,seed);});
    result.stats.setup_ms=setup_ms;return result;
}
inline AdaptiveResult epoch_adaptive(const LensParams& p,double u,const AdaptiveConfig& cfg,AdaptiveWorkspace& w) {
    if(!adaptive_detail::valid_epoch_config(p,u,cfg)){w.reset();AdaptiveResult r;r.stop=AdaptiveStop::InvalidConfig;return r;}
    if(cfg.require_bound){w.reset();AdaptiveResult r;r.stop=AdaptiveStop::BoundUnavailable;return r;}
    auto start=adaptive_detail::Clock::now();auto topo=classify_cells(PrimaryFrame::from(p));double t=adaptive_detail::ms(start);
    auto out=flux_adaptive_integrate(p,u,topo,cfg,w);out.stats.topology_ms=t;return out;
}
inline AdaptiveResult epoch_value_adaptive(const LensParams& p,double u,AdaptiveConfig cfg,AdaptiveWorkspace& w) {cfg.with_jacobian=false;return epoch_adaptive(p,u,cfg,w);}
inline AdaptiveResult epoch_jacobian_adaptive(const LensParams& p,double u,AdaptiveConfig cfg,AdaptiveWorkspace& w) {cfg.with_jacobian=true;return epoch_adaptive(p,u,cfg,w);}
inline AdaptiveResult epoch_adaptive_prepared(const LensParams& p,double u,const AdaptiveConfig& cfg,AdaptiveWorkspace& w,PreparedEpochGeometry& state,const PreparedReuseConfig& reuse=PreparedReuseConfig{}) {
    if(!adaptive_detail::valid_epoch_config(p,u,cfg)){w.reset();AdaptiveResult r;r.stop=AdaptiveStop::InvalidConfig;return r;}
    if(cfg.require_bound){w.reset();AdaptiveResult r;r.stop=AdaptiveStop::BoundUnavailable;return r;}
    auto start=adaptive_detail::Clock::now();auto topo=prepared_topology(PrimaryFrame::from(p),state,reuse,nullptr);double t=adaptive_detail::ms(start);
    auto out=flux_adaptive_integrate(p,u,topo,cfg,w);out.stats.topology_ms=t;return out;
}
}
