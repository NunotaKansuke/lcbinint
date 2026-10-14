#pragma once
// Experimental all-state true GM epoch. No node re-seeds and no packet.
#include "lcbinint/magnification/holonomic/gm_adaptive_transport.hpp"
#include "lcbinint/magnification/holonomic/gm_physical_transport.hpp"
#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"

namespace lcbinint::holonomic {
struct GmCoverageEpoch {
    double F0=0,F_half=0,mu=0;
    std::array<double,5> grad_mu{};
    Status status=Status::OK;
    Status local_status=Status::OK;
    GmAdaptiveCost transport;
    double topology_ms=0,seed_ms=0,arc_ms=0;
    int seeds=0,nodes=0,transported=0;
};

template<class S>
GmCoverageEpoch gm_coverage_epoch_from_topology(const LensParams& p,double u,int nr,
    const TopologyResult& topo){
    using Clock=std::chrono::steady_clock;
    constexpr bool jac=std::is_same_v<S,GmDDDual5>;
    GmCoverageEpoch out;out.status=topo.status;auto pf=PrimaryFrame::from(p);
    std::array<double,5> df0{},dfh{};
    Cheb1Dyn rr(nr);
    for(const auto& cell:topo.cells){
        if(cell.kind==ArcKind::kEmpty)continue;
        if(cell.kind!=ArcKind::kArcs){out.status=Status::BASIS_DEGENERATE;continue;}
        double rc=(cell.r_lo+cell.r_hi)*0.5,rh=(cell.r_hi-cell.r_lo)*0.5*(1-2e-9);
        auto t0=Clock::now();auto center=arc_intervals(rc,pf);
        out.arc_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        if(center.kind!=ArcKind::kArcs){out.status=Status::ARC_TOPOLOGY_INVALID;continue;}
        // F0 and its endpoint IFT derivatives are algebraic arc geometry.
        QuarticWarm qw;RootPairWarm rpw;rpw.certify=topo.from_warm_d14;
        for(int n=0;n<nr;++n){
            double r=rc+rh*rr.x[n],w=rh*rr.w[n];t0=Clock::now();
            auto as=arc_intervals(r,pf,&qw,&rpw);
            if(as.kind!=ArcKind::kArcs){out.status=Status::ARC_TOPOLOGY_INVALID;continue;}
            for(auto arc:as.arcs){
                auto pe=polish_endpoint(r,arc[0],pf),pl=polish_endpoint(r,arc[1],pf);
                double hi=pl.theta;if(hi<=pe.theta)hi+=2*M_PI;
                out.F0+=w*r*(hi-pe.theta);
                if(!pe.reliable||!pl.reliable)out.status=Status::GRADIENT_UNRELIABLE;
                if constexpr(jac){
                    auto ge=phi_val_dP(r,pe.theta,pf),gl=phi_val_dP(r,pl.theta,pf);
                    if(pe.dphi_dtheta==0||pl.dphi_dtheta==0){out.status=Status::GRADIENT_UNRELIABLE;continue;}
                    for(int j=0;j<5;++j)df0[j]+=w*r*(ge.dP[j]/pe.dphi_dtheta-gl.dP[j]/pl.dphi_dtheta);
                }
            }
            out.arc_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        }
        if(u==0)continue;
        for(auto arc:center.arcs){
            t0=Clock::now();auto geometry=gm_physical_period_seed(rc,pf,arc,8);
            std::array<S,7> seed{};bool seeded=gm_seed_sensitivity(rc,geometry,seed,512);
            out.seed_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();++out.seeds;
            if(!seeded){out.status=Status::BASIS_DEGENERATE;out.nodes+=nr;continue;}
            auto params=gm_sensitivity_params<S>(geometry.chart_pf);
            for(int dir:{1,-1}){
                double r=rc;auto state=seed;bool alive=true;
                for(int j=0;j<nr;++j){int n=dir>0?nr-1-j:j;
                    if(rr.x[n]*dir<=0)continue;
                    double target=rc+rh*rr.x[n],w=rh*rr.w[n];++out.nodes;
                    alive=alive&&gm_adaptive_advance(r,target,cell.r_lo,cell.r_hi,geometry.chart_pf,state,out.transport);
                    if(!alive){out.status=Status::TRANSPORT_TOLERANCE_FAILED;continue;}
                    ++out.transported;auto h=gm_physical_h(target,params);S value(0);
                    for(int k=0;k<7;++k)value=value+h[k]*state[k];value=value/params.rho;
                    if(!gm_finite(value)){out.status=Status::TRANSPORT_TOLERANCE_FAILED;continue;}
                    out.F_half+=w*(double)value;
                    if constexpr(jac)for(int k=0;k<5;++k){
                        double sign=geometry.chart2&&(k==0||k==1||k==4)?-1:1;
                        dfh[k]+=w*sign*(double)value.deriv[k];}
                }
            }
        }
    }
    double denom=M_PI*p.rho*p.rho*(1-u/3);
    out.mu=((1-u)*out.F0+u*out.F_half)/denom;
    if constexpr(jac){
        std::array<double,5> di{};for(int j=0;j<5;++j)di[j]=((1-u)*df0[j]+u*dfh[j])/denom;
        out.grad_mu=internal_to_user_jac(di,p);out.grad_mu[2]-=2*out.mu/p.rho;
    }
    // No reference fallback can mask a missing transported contribution.
    if(out.transported!=out.nodes){out.mu=std::numeric_limits<double>::quiet_NaN();out.grad_mu.fill(out.mu);}
    out.local_status=out.status;
    // The all-node sensitivity audit found forward errors at small mass
    // ratios which the local embedded estimator does not detect. Until a
    // conditioning/error certificate covers them, expose the diagnostic
    // derivatives but never certify the LD Jacobian as reliable. This is a
    // representation-wide gate, not a parameter-specific exception list.
    if constexpr(jac)if(u!=0 && out.status==Status::OK)
        out.status=Status::GRADIENT_UNRELIABLE;
    return out;
}

template<class S> GmCoverageEpoch gm_coverage_epoch(const LensParams& p,double u,int nr,
    std::vector<Cplx<__float128>>* warm=nullptr){
    using Clock=std::chrono::steady_clock;
    auto t0=Clock::now();std::vector<Cplx<__float128>> roots;
    auto topo=classify_cells(PrimaryFrame::from(p),warm&&!warm->empty()?warm:nullptr,&roots);
    topo.from_warm_d14=warm&&!warm->empty();
    double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    if(warm)*warm=std::move(roots);
    auto result=gm_coverage_epoch_from_topology<S>(p,u,nr,topo);result.topology_ms=ms;return result;
}
} // namespace lcbinint::holonomic
