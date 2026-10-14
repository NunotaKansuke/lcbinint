#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;
using Clock=std::chrono::steady_clock;

struct Capture {
    AdaptiveResult result;
    double wall_ms=0,topology_ms=0;
    std::size_t events=0,cells=0;
    std::size_t charts=0,folds=0;
    Status topology_status=Status::OK;
};

static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}

static AdaptiveConfig config(double tol,bool jac) {
    AdaptiveConfig c;
    c.gradient_policy=jac?GradientPolicy::ValueFirst:GradientPolicy::None;
    c.with_jacobian=jac;c.fold_maps=true;c.reuse_samples=true;
    c.tol.mu_atol=1e-16;c.tol.mu_rtol=tol;
    if(jac)c.tol.grad_rtol.fill(1e-3);
    return c;
}

static Capture cold(const LensParams& p,double u,const AdaptiveConfig& cfg,
                    AdaptiveWorkspace& ws,std::size_t events,std::size_t cells,
                    Status topo_status,std::size_t charts,std::size_t folds) {
    const auto t=Clock::now();Capture c;
    c.result=epoch_adaptive(p,u,cfg,ws);c.wall_ms=elapsed(t);
    c.topology_ms=c.result.stats.topology_ms;
    c.events=events;c.cells=cells;c.charts=charts;c.folds=folds;
    c.topology_status=topo_status;
    return c;
}

static Capture warm(const LensParams& p,double u,const AdaptiveConfig& cfg,
                    AdaptiveWorkspace& ws,PreparedEpochGeometry& state,
                    const PreparedReuseConfig& reuse,std::size_t charts,
                    std::size_t folds) {
    const auto whole=Clock::now();Capture c;
    const auto topology=Clock::now();
    const TopologyResult topo=prepared_topology(PrimaryFrame::from(p),state,reuse,nullptr);
    c.topology_ms=elapsed(topology);
    c.events=topo.events.size();c.cells=topo.cells.size();
    c.topology_status=topo.status;c.charts=charts;c.folds=folds;
    c.result=flux_adaptive_integrate(p,u,topo,cfg,ws);
    c.result.stats.topology_ms=c.topology_ms;
    c.wall_ms=elapsed(whole);
    return c;
}

static Capture radial(const LensParams& p,double u,const AdaptiveConfig& cfg,
                      const TopologyResult& topo,AdaptiveWorkspace& ws,
                      std::size_t charts,std::size_t folds) {
    const auto t=Clock::now();Capture c;
    c.result=flux_adaptive_integrate(p,u,topo,cfg,ws);c.wall_ms=elapsed(t);
    c.events=topo.events.size();c.cells=topo.cells.size();
    c.topology_status=topo.status;c.charts=charts;c.folds=folds;
    return c;
}

static void emit(std::ofstream& out,int rep,bool jac,double target,double u,
                 int epoch,const char* mode,const LensParams& p,const Capture& c) {
    const auto& r=c.result;
    out<<rep<<' '<<(jac?"ValueFirst":"None")<<' '<<target<<' '<<u<<' '
       <<epoch<<' '<<mode<<' '<<p.xs<<' '<<p.ys<<' '<<p.rho<<' '<<p.q<<' '
       <<p.a<<' '<<c.wall_ms<<' '<<c.topology_ms<<' '<<r.mu<<' '
       <<r.estimated_abs_error_mu<<' '<<r.value_converged<<' '
       <<adaptive_stop_name(r.value_stop_reason)<<' '<<adaptive_stop_name(r.stop)<<' '
       <<to_string(r.numerical_status)<<' '<<r.stats.unique_nodes<<' '
       <<r.stats.node_evaluations<<' '<<r.stats.panels<<' '<<r.stats.splits<<' '
       <<c.events<<' '<<c.cells<<' '<<to_string(c.topology_status)<<' '
       <<c.charts<<' '<<c.folds;
    for(int j=0;j<5;++j)out<<' '<<r.grad_mu[j]<<' '
       <<r.estimated_abs_error_grad[j]<<' '
       <<gradient_quality_name(r.grad_quality[j])<<' '
       <<gradient_reason_name(r.grad_reason[j]);
    out<<'\n';
}

int main(int argc,char**argv) {
    if(argc!=4){std::cerr<<"usage: projective_trajectory_ab OUTPUT value|jac REP_ID\n";return 2;}
    const bool jac=std::string(argv[2])=="jac";
    if(!jac&&std::string(argv[2])!="value")return 2;
    const int rep=std::atoi(argv[3]);
    std::ofstream out(argv[1]);if(!out)return 2;
    out<<std::setprecision(17)
       <<"rep policy target u epoch mode xs ys rho q a wall_ms topology_ms mu "
         "value_error value_converged value_stop stop status nodes evaluations "
         "panels splits events cells topology_status chart_events projective_folds";
    for(int j=0;j<5;++j) {
        out<<' '<<"g"<<j<<" g"<<j<<"_error g"<<j<<"_quality g"<<j<<"_reason";
    }
    out<<'\n';

    const std::array<double,4> ys{{-1e-6,0.0,1e-6,0.0}};
    const std::array<double,2> us{{0.0,0.5}};
    const std::array<double,2> targets{{1e-3,1e-4}};
    constexpr int measured_trajectories=48;
    volatile double sink=0;
    holo_mv_transport_override()=1;holo_holonomic_transport_override()=1;
    holo_ode_transport_override()=0;
    for(double u:us)for(double target:targets) {
        const AdaptiveConfig cfg=config(target,jac);
        std::array<LensParams,4> params{};
        std::array<TopologyResult,4> topologies{};
        std::array<std::size_t,4> charts{},folds{};
        for(std::size_t k=0;k<4;++k) {
            params[k]=LensParams{0.12,ys[k],0.02,0.5,1.0,false};
            topologies[k]=classify_cells(PrimaryFrame::from(params[k]),nullptr,nullptr,true);
            TopologyResult shadow=topologies[k];
            adaptive_detail::certify_projective_p4_events(shadow,PrimaryFrame::from(params[k]));
            for(const auto&e:shadow.events) {
                charts[k]+=e.kind=="chart_p4";
                folds[k]+=e.projective_fold_certified;
            }
        }
        // Exclude one whole four-epoch trajectory from all reported timings.
        {AdaptiveWorkspace ws;for(std::size_t k=0;k<4;++k)
            sink+=epoch_adaptive(params[k],u,cfg,ws).mu;}
        {AdaptiveWorkspace ws;PreparedEpochGeometry state;PreparedReuseConfig reuse;
         reuse.allow_topology_reuse=false;reuse.allow_warm_d14=true;reuse.l2_drift=1e18;
         for(std::size_t k=0;k<4;++k) {
             const auto topo=prepared_topology(PrimaryFrame::from(params[k]),state,reuse,nullptr);
             sink+=flux_adaptive_integrate(params[k],u,topo,cfg,ws).mu;
         }}
        {AdaptiveWorkspace ws;for(std::size_t k=0;k<4;++k)
            sink+=flux_adaptive_integrate(params[k],u,topologies[k],cfg,ws).mu;}

        for(int trajectory=0;trajectory<measured_trajectories;++trajectory) {
            std::array<Capture,4> c{},w{},r{};
            {AdaptiveWorkspace ws;for(std::size_t k=0;k<4;++k)
                c[k]=cold(params[k],u,cfg,ws,topologies[k].events.size(),
                          topologies[k].cells.size(),topologies[k].status,
                          charts[k],folds[k]);}
            {AdaptiveWorkspace ws;PreparedEpochGeometry state;PreparedReuseConfig reuse;
             reuse.allow_topology_reuse=false;reuse.allow_warm_d14=true;reuse.l2_drift=1e18;
             for(std::size_t k=0;k<4;++k)
                w[k]=warm(params[k],u,cfg,ws,state,reuse,charts[k],folds[k]);}
            {AdaptiveWorkspace ws;for(std::size_t k=0;k<4;++k)
                r[k]=radial(params[k],u,cfg,topologies[k],ws,charts[k],folds[k]);}
            for(std::size_t k=0;k<4;++k) {
                emit(out,rep,jac,target,u,(int)k,"full-cold",params[k],c[k]);
                emit(out,rep,jac,target,u,(int)k,"full-warm",params[k],w[k]);
                emit(out,rep,jac,target,u,(int)k,"radial-only",params[k],r[k]);
                sink+=c[k].result.mu+w[k].result.mu+r[k].result.mu;
            }
        }
    }
    std::cerr<<"rep="<<rep<<" policy="<<(jac?"ValueFirst":"None")
             <<" trajectories="<<measured_trajectories<<" sink="<<sink<<" variant="
#if defined(HOLO_ADAPTIVE_PROJECTIVE_P4_FAST_SCREEN)
             <<"interval_screen"
#elif defined(HOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH)
             <<"qf_probe"
#else
             <<"baseline"
#endif
             <<'\n';
}
