#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;
using Clock = std::chrono::steady_clock;

struct InputRow {
    int case_id=0, configuration_id=0, d_bin_index=0, epoch_index=0;
    std::string profile;
    double s=0, q=0, rho=0, x=0, y=0, time=0, u=0, X=0, reference=0;
};

struct EpochPlan { InputRow input; LensParams params; TopologyResult radial_topology; };
struct RunCapture {
    AdaptiveResult result;
    PreparedReuseStats reuse{};
    double whole_ms=0, topology_ms=0;
    std::uint64_t topology_hash=0;
    std::size_t topology_cells=0, topology_events=0;
    Status topology_status=Status::OK;
};

static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}

static void hash_bytes(std::uint64_t& h,const void* p,std::size_t n) {
    const auto* b=static_cast<const unsigned char*>(p);
    for(std::size_t i=0;i<n;++i){h^=b[i];h*=UINT64_C(1099511628211);}
}
static void hash_string(std::uint64_t& h,const std::string& s) {
    hash_bytes(h,s.data(),s.size());const unsigned char sep=0xff;hash_bytes(h,&sep,1);
}
static void hash_double(std::uint64_t& h,double x) {
    std::uint64_t bits=0;static_assert(sizeof(bits)==sizeof(x),"binary64 expected");
    std::memcpy(&bits,&x,sizeof(bits));hash_bytes(h,&bits,sizeof(bits));
}
static std::uint64_t topology_signature(const TopologyResult& topo) {
    std::uint64_t h=UINT64_C(1469598103934665603);
    const std::string status=to_string(topo.status);hash_string(h,status);
    for(const auto& e:topo.events){
        hash_string(h,e.kind);hash_string(h,e.detail);hash_double(h,e.radius);
        hash_bytes(h,&e.physically_real,sizeof(e.physically_real));
        hash_bytes(h,&e.positive_certified,sizeof(e.positive_certified));
    }
    for(const auto& c:topo.cells){
        const int kind=static_cast<int>(c.kind);
        hash_bytes(h,&kind,sizeof(kind));hash_double(h,c.r_lo);hash_double(h,c.r_hi);
        hash_bytes(h,&c.n_crossings,sizeof(c.n_crossings));
    }
    return h;
}

static AdaptiveConfig make_config(double rtol,bool with_jacobian) {
    AdaptiveConfig cfg;
    cfg.gradient_policy=with_jacobian?GradientPolicy::ValueFirst:GradientPolicy::None;
    cfg.with_jacobian=with_jacobian;
    cfg.fold_maps=true;cfg.reuse_samples=true;
    cfg.tol.mu_atol=1e-16;cfg.tol.mu_rtol=rtol;
    if(with_jacobian)cfg.tol.grad_rtol.fill(1e-3);
    return cfg;
}
static bool read_row(const std::string& line,InputRow& r) {
    if(line.empty()||line[0]=='#')return false;
    std::istringstream in(line);
    return bool(in>>r.case_id>>r.configuration_id>>r.profile>>r.d_bin_index>>r.epoch_index
        >>r.s>>r.q>>r.rho>>r.x>>r.y>>r.time>>r.u>>r.X>>r.reference);
}
static bool same_trajectory(const InputRow& a,const InputRow& b) {
    return a.case_id==b.case_id&&a.configuration_id==b.configuration_id&&
        a.profile==b.profile&&a.d_bin_index==b.d_bin_index;
}

static RunCapture run_full_cold(const EpochPlan& e,const AdaptiveConfig& cfg,
                                AdaptiveWorkspace& workspace) {
    const auto start=Clock::now();RunCapture c;
    c.result=epoch_adaptive(e.params,e.input.u,cfg,workspace);
    c.whole_ms=elapsed_ms(start);c.topology_ms=c.result.stats.topology_ms;
    c.topology_hash=topology_signature(e.radial_topology);
    c.topology_cells=e.radial_topology.cells.size();
    c.topology_events=e.radial_topology.events.size();
    c.topology_status=e.radial_topology.status;
    return c;
}
static RunCapture run_full_warm(const EpochPlan& e,const AdaptiveConfig& cfg,
                                AdaptiveWorkspace& workspace,PreparedEpochGeometry& state,
                                const PreparedReuseConfig& reuse) {
    const auto start=Clock::now();RunCapture c;
    const auto topology_start=Clock::now();
    TopologyResult topo=prepared_topology(PrimaryFrame::from(e.params),state,reuse,&c.reuse);
    c.topology_ms=elapsed_ms(topology_start);
    c.result=flux_adaptive_integrate(e.params,e.input.u,topo,cfg,workspace);
    c.result.stats.topology_ms=c.topology_ms;
    c.whole_ms=elapsed_ms(start);
    c.topology_hash=topology_signature(topo);c.topology_cells=topo.cells.size();
    c.topology_events=topo.events.size();c.topology_status=topo.status;
    return c;
}
static RunCapture run_radial_only(const EpochPlan& e,const AdaptiveConfig& cfg,
                                  AdaptiveWorkspace& workspace) {
    const auto start=Clock::now();RunCapture c;
    c.result=flux_adaptive_integrate(e.params,e.input.u,e.radial_topology,cfg,workspace);
    c.whole_ms=elapsed_ms(start);c.topology_hash=topology_signature(e.radial_topology);
    c.topology_cells=e.radial_topology.cells.size();
    c.topology_events=e.radial_topology.events.size();
    c.topology_status=e.radial_topology.status;return c;
}

static void write_header(std::ofstream& out) {
    out<<"# Projective-fold final A/B; one process repetition; whole timers include cold D14/topology or warm L2 D14; adaptive n_r\n";
    out<<"case_id configuration_id profile d_bin_index epoch_index trajectory_id trajectory_pos policy target "
          "s q rho x y time u X reference ";
    for(const char* lane:{"cold","warm","radial"}){
        out<<lane<<"_ms "<<lane<<"_topology_ms "<<lane<<"_adaptive_ms "
           <<lane<<"_mu "<<lane<<"_value_error "<<lane<<"_value_converged "
           <<lane<<"_value_stop "<<lane<<"_stop "<<lane<<"_status "
           <<lane<<"_nodes "<<lane<<"_evaluations "<<lane<<"_panels "<<lane<<"_splits "
           <<lane<<"_topology_status "<<lane<<"_topology_cells "<<lane<<"_topology_events "
           <<lane<<"_topology_hash ";
    }
    out<<"warm_l1 warm_l2 warm_l3 warm_rescreen_fail warm_seed_used ";
    for(const char* lane:{"cold","warm","radial"})for(int j=0;j<5;++j)
        out<<lane<<"_g"<<j<<' '<<lane<<"_g"<<j<<"_error "
           <<lane<<"_g"<<j<<"_quality "<<lane<<"_g"<<j<<"_reason ";
    out<<'\n';
}
static void write_lane(std::ofstream& out,const char* name,const RunCapture& c) {
    const auto& r=c.result;
    const double adaptive_ms=name[0]=='r'?c.whole_ms:std::max(0.0,c.whole_ms-c.topology_ms);
    out<<c.whole_ms<<' '<<c.topology_ms<<' '<<adaptive_ms<<' '
       <<r.mu<<' '<<r.estimated_abs_error_mu<<' '<<int(r.value_converged)<<' '
       <<adaptive_stop_name(r.value_stop_reason)<<' '<<adaptive_stop_name(r.stop)<<' '
       <<to_string(r.numerical_status)<<' '<<r.stats.unique_nodes<<' '
       <<r.stats.node_evaluations<<' '<<r.stats.panels<<' '<<r.stats.splits<<' '
       <<to_string(c.topology_status)<<' '<<c.topology_cells<<' '
       <<c.topology_events<<' '<<c.topology_hash<<' ';
}
static void write_gradient(std::ofstream& out,const AdaptiveResult& r) {
    for(int j=0;j<5;++j)out<<r.grad_mu[j]<<' '<<r.estimated_abs_error_grad[j]<<' '
        <<gradient_quality_name(r.grad_quality[j])<<' '
        <<gradient_reason_name(r.grad_reason[j])<<' ';
}

int main(int argc,char** argv) {
    if(argc!=5){std::cerr<<"usage: projective_fold_trajectory_ab INPUT OUTPUT value|jac REP_ID\n";return 2;}
    const bool with_jacobian=std::string(argv[3])=="jac";
    if(!with_jacobian&&std::string(argv[3])!="value"){std::cerr<<"policy must be value or jac\n";return 2;}
    const int rep_id=std::atoi(argv[4]);
    std::ifstream in(argv[1]);std::ofstream out(argv[2]);
    if(!in||!out){std::cerr<<"cannot open input/output\n";return 2;}
    holo_mv_transport_override()=1;holo_holonomic_transport_override()=1;holo_ode_transport_override()=0;
    std::vector<InputRow> rows;std::string line;
    while(std::getline(in,line)){InputRow r;if(read_row(line,r))rows.push_back(std::move(r));}
    if(rows.empty()||rows.size()%4!=0){std::cerr<<"input must contain complete 4-epoch trajectories\n";return 3;}
    std::stable_sort(rows.begin(),rows.end(),[](const InputRow& a,const InputRow& b){
        if(a.case_id!=b.case_id)return a.case_id<b.case_id;
        if(a.configuration_id!=b.configuration_id)return a.configuration_id<b.configuration_id;
        if(a.profile!=b.profile)return a.profile<b.profile;
        if(a.d_bin_index!=b.d_bin_index)return a.d_bin_index<b.d_bin_index;
        return a.epoch_index<b.epoch_index;
    });
    write_header(out);out<<std::setprecision(17);
    const std::array<double,2> targets{{1e-3,1e-4}};
    const std::size_t expected_trajectories=rows.size()/4;
    std::size_t trajectory_id=0,emitted=0;volatile double sink=0;
    for(std::size_t begin=0;begin<rows.size();){
        std::size_t end=begin+1;
        while(end<rows.size()&&same_trajectory(rows[begin],rows[end]))++end;
        if(end-begin!=4){std::cerr<<"expected 4 epochs in trajectory; got "<<end-begin<<'\n';return 4;}
        std::vector<EpochPlan> plan;plan.reserve(4);
        for(std::size_t i=begin;i<end;++i){
            EpochPlan e;e.input=rows[i];
            e.params=LensParams{e.input.time,e.input.y,e.input.rho,1.0/e.input.q,e.input.s,true};
            e.radial_topology=classify_cells(PrimaryFrame::from(e.params),nullptr,nullptr,true);
            plan.push_back(std::move(e));
        }
        for(double target:targets){
            const AdaptiveConfig cfg=make_config(target,with_jacobian);
            // Match the established harness: one unmeasured full trajectory per
            // lane; warm root state is discarded before measured rows begin.
            {AdaptiveWorkspace ws;for(const auto& e:plan)sink+=run_radial_only(e,cfg,ws).result.mu;}
            {AdaptiveWorkspace ws;for(const auto& e:plan)sink+=run_full_cold(e,cfg,ws).result.mu;}
            {AdaptiveWorkspace ws;PreparedEpochGeometry state;PreparedReuseConfig reuse;
             reuse.allow_topology_reuse=false;reuse.allow_warm_d14=true;reuse.l2_drift=1e18;
             for(const auto& e:plan)sink+=run_full_warm(e,cfg,ws,state,reuse).result.mu;}
            std::array<RunCapture,4> cold{},warm{},radial{};
            {AdaptiveWorkspace ws;for(std::size_t i=0;i<4;++i)cold[i]=run_full_cold(plan[i],cfg,ws);}
            {AdaptiveWorkspace ws;PreparedEpochGeometry state;PreparedReuseConfig reuse;
             reuse.allow_topology_reuse=false;reuse.allow_warm_d14=true;reuse.l2_drift=1e18;
             for(std::size_t i=0;i<4;++i)warm[i]=run_full_warm(plan[i],cfg,ws,state,reuse);}
            {AdaptiveWorkspace ws;for(std::size_t i=0;i<4;++i)radial[i]=run_radial_only(plan[i],cfg,ws);}
            for(std::size_t i=0;i<4;++i){
                const auto& e=plan[i];
                out<<e.input.case_id<<' '<<e.input.configuration_id<<' '<<e.input.profile<<' '
                   <<e.input.d_bin_index<<' '<<e.input.epoch_index<<' '<<trajectory_id<<' '<<i<<' '
                   <<(with_jacobian?"ValueFirst":"None")<<' '<<target<<' '
                   <<e.input.s<<' '<<e.input.q<<' '<<e.input.rho<<' '<<e.input.x<<' '
                   <<e.input.y<<' '<<e.input.time<<' '<<e.input.u<<' '<<e.input.X<<' '
                   <<e.input.reference<<' ';
                write_lane(out,"cold",cold[i]);write_lane(out,"warm",warm[i]);write_lane(out,"radial",radial[i]);
                out<<warm[i].reuse.l1_topology_reuse<<' '<<warm[i].reuse.l2_warm_recompute<<' '
                   <<warm[i].reuse.l3_cold_recompute<<' '<<warm[i].reuse.rescreen_fail<<' '
                   <<warm[i].reuse.warm_solve_used<<' ';
                write_gradient(out,cold[i].result);write_gradient(out,warm[i].result);write_gradient(out,radial[i].result);
                out<<'\n';sink+=cold[i].result.mu+warm[i].result.mu+radial[i].result.mu;++emitted;
            }
        }
        ++trajectory_id;begin=end;
        if((trajectory_id%128)==0)std::cerr<<"progress trajectories "<<trajectory_id<<'/'<<expected_trajectories<<" rows="<<emitted<<'\n';
    }
    std::cerr<<"variant="
#if defined(HOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH)
             <<"projective"
#else
             <<"baseline"
#endif
             <<" policy="<<(with_jacobian?"ValueFirst":"None")<<" rep="<<rep_id
             <<" wrote="<<emitted<<" trajectories="<<trajectory_id<<" sink="<<sink<<'\n';
    return emitted==rows.size()*2?0:5;
}
