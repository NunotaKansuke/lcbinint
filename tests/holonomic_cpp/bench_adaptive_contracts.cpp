#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace lcbinint::holonomic;

struct ContractCase { LensParams p; double u; std::string name; };

int main(int argc,char** argv) {
    if(argc<2) return 2;
    std::ifstream input(argv[1]);
    if(!input) return 3;
    const int repeats=argc>2?std::atoi(argv[2]):3;
    std::vector<ContractCase> cases;
    std::string line;
    while(std::getline(input,line)) {
        ContractCase c; int bary=0; double old=0;
        std::istringstream s(line);
        if(s>>c.p.xs>>c.p.ys>>c.p.rho>>c.p.q>>c.p.a>>bary>>c.u>>old>>c.name) {
            c.p.barycentric=bary;cases.push_back(std::move(c));
        }
    }
    std::printf("name,u,policy,warm,rtol,repeat,stop,value_stop,gradient_stop,value_converged,mu,value_error,whole_ms,topology_ms,setup_ms,setup_frame_ms,setup_cuts_ms,setup_event_ms,setup_panel_ms,nodes,evaluations,event_double_checks,event_dd_checks,event_dd_accepts,qf_family_constructions,event_qf_refinements,event_topology_reuses,event_radius_reuses,event_direct_qf_failures,gq0,gq1,gq2,gq3,gq4,gr0,gr1,gr2,gr3,gr4\n");
    const GradientPolicy policies[]={GradientPolicy::None,GradientPolicy::Strict,GradientPolicy::ValueFirst};
    for(const auto& c:cases) for(const auto policy:policies) for(bool warm:{false,true}) {
        AdaptiveWorkspace workspace; PreparedEpochGeometry prepared;
        for(double tol:{1e-3,1e-4}) for(int rep=0;rep<repeats;++rep) {
            AdaptiveConfig cfg; cfg.gradient_policy=policy; cfg.with_jacobian=policy!=GradientPolicy::None;
            cfg.tol.mu_rtol=tol; cfg.tol.grad_rtol.fill(1e-4); cfg.tol.grad_atol.fill(1e-6);
            cfg.value_first_gradient_node_budget=1024; cfg.value_first_gradient_round_budget=3;
            LensParams p=c.p;
            if(warm) {
                (void)epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared);
                p.xs+=.01*p.rho;
            }
            auto start=adaptive_detail::Clock::now();
            auto r=warm?epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared)
                       :epoch_adaptive(p,c.u,cfg,workspace);
            const double elapsed=adaptive_detail::ms(start);
            std::printf("%s,%.1f,%s,%d,%.1e,%d,%s,%s,%s,%d,%.17g,%.8g,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu",
                c.name.c_str(),c.u,gradient_policy_name(policy),warm,tol,rep,
                adaptive_stop_name(r.stop),adaptive_stop_name(r.value_stop_reason),
                adaptive_stop_name(r.gradient_stop_reason),r.value_converged,r.mu,
                r.estimated_abs_error_mu,elapsed,r.stats.topology_ms,r.stats.setup_ms,
                r.stats.setup_frame_ms,r.stats.setup_cuts_ms,r.stats.setup_event_ms,
                r.stats.setup_panel_ms,r.stats.unique_nodes,r.stats.node_evaluations,
                r.stats.event_double_checks,r.stats.event_dd_checks,r.stats.event_dd_accepts,
                r.stats.qf_family_constructions,r.stats.event_qf_refinements,
                r.stats.event_topology_reuses,r.stats.event_radius_reuses,
                r.stats.event_direct_qf_failures);
            for(auto q:r.grad_quality)std::printf(",%s",gradient_quality_name(q));
            for(auto reason:r.grad_reason)std::printf(",%s",gradient_reason_name(reason));
            std::putchar('\n'); std::fflush(stdout);
        }
    }
}
