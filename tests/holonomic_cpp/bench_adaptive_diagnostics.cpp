#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <array>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace lcbinint::holonomic;

struct DiagnosticCase {
    LensParams p;
    double u=0;
    std::string name;
};

static std::vector<DiagnosticCase> read_cases(const char* path) {
    std::ifstream input(path);
    std::vector<DiagnosticCase> cases;
    std::string line;
    while(std::getline(input,line)) {
        DiagnosticCase c;
        int bary=0;
        double old=0;
        std::istringstream s(line);
        if(s>>c.p.xs>>c.p.ys>>c.p.rho>>c.p.q>>c.p.a>>bary>>c.u>>old>>c.name &&
           c.name=="rand033") {
            c.p.barycentric=bary;
            cases.push_back(std::move(c));
        }
    }
    return cases;
}

int main(int argc,char** argv) {
    if(argc<3) return 2;
    const auto cases=read_cases(argv[1]);
    const std::string outdir=argv[2];
    std::ofstream epochs(outdir+"/epochs.csv");
    std::ofstream events(outdir+"/events.csv");
    std::ofstream refinements(outdir+"/refinements.csv");
    if(!epochs||!events||!refinements) return 3;
    epochs << std::setprecision(17);
    events << std::setprecision(17);
    refinements << std::setprecision(17);

    epochs << "name,u,policy,warm,rtol,stop,value_stop,gradient_stop,value_converged,mu,value_error,nodes,evaluations,event_count,refinement_count,topology_ms,setup_ms,physics_ms,estimator_ms,scheduler_ms\n";
    events << "name,u,policy,warm,rtol,index,input_radius,selected_radius,radius_lo,uncertainty,precision_tier,double_reason,dd_reason,qf_reason,double_residual,dd_residual\n";
    refinements << "name,u,policy,warm,rtol,index,panel,cell,parent,depth,level,node_evaluations,value_error,gradient_error0,gradient_error1,gradient_error2,gradient_error3,gradient_error4,gradient_phase,value_resolved,gradient_resolved0,gradient_resolved1,gradient_resolved2,gradient_resolved3,gradient_resolved4\n";

    const GradientPolicy policies[]={GradientPolicy::None,GradientPolicy::ValueFirst};
    for(const auto& c:cases) for(const auto policy:policies) for(bool warm:{false,true}) {
        for(double tol:{1e-3,1e-4}) {
            AdaptiveConfig cfg;
            cfg.gradient_policy=policy;
            cfg.with_jacobian=policy!=GradientPolicy::None;
            cfg.tol.mu_rtol=tol;
            cfg.tol.grad_rtol.fill(1e-4);
            cfg.tol.grad_atol.fill(1e-6);
            cfg.value_first_gradient_node_budget=1024;
            cfg.value_first_gradient_round_budget=3;
            cfg.collect_diagnostics=true;

            LensParams p=c.p;
            AdaptiveWorkspace workspace;
            PreparedEpochGeometry prepared;
            if(warm) {
                (void)epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared);
                p.xs+=.01*p.rho;
            }
            const auto r=warm?epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared)
                             :epoch_adaptive(p,c.u,cfg,workspace);
            epochs << c.name << ',' << c.u << ',' << gradient_policy_name(policy) << ',' << int(warm)
                   << ',' << tol << ',' << adaptive_stop_name(r.stop) << ','
                   << adaptive_stop_name(r.value_stop_reason) << ','
                   << adaptive_stop_name(r.gradient_stop_reason) << ',' << int(r.value_converged)
                   << ',' << r.mu << ',' << r.estimated_abs_error_mu << ','
                   << r.stats.unique_nodes << ',' << r.stats.node_evaluations << ','
                   << r.stats.event_diagnostics.size() << ',' << r.stats.refinement_history.size() << ','
                   << r.stats.topology_ms << ',' << r.stats.setup_ms << ','
                   << r.stats.physical_ms << ',' << r.stats.estimator_ms << ','
                   << r.stats.scheduler_ms << '\n';
            for(size_t i=0;i<r.stats.event_diagnostics.size();++i) {
                const auto& e=r.stats.event_diagnostics[i];
                events << c.name << ',' << c.u << ',' << gradient_policy_name(policy) << ',' << int(warm)
                       << ',' << tol << ',' << i << ',' << e.input_radius << ','
                       << e.selected_radius << ',' << e.radius_lo << ',' << e.uncertainty << ','
                       << e.precision_tier << ',' << event_decision_reason_name(e.double_reason) << ','
                       << event_decision_reason_name(e.dd_reason) << ','
                       << event_decision_reason_name(e.qf_reason) << ',' << e.double_residual << ','
                       << e.dd_residual << '\n';
            }
            for(size_t i=0;i<r.stats.refinement_history.size();++i) {
                const auto& h=r.stats.refinement_history[i];
                refinements << c.name << ',' << c.u << ',' << gradient_policy_name(policy) << ',' << int(warm)
                            << ',' << tol << ',' << i << ',' << h.panel << ',' << h.cell << ','
                            << h.parent << ',' << h.depth << ',' << h.level << ',' << h.node_evaluations
                            << ',' << h.value_error;
                for(double e:h.gradient_error) refinements << ',' << e;
                refinements << ',' << int(h.gradient_phase) << ',' << int(h.value_resolved);
                for(bool x:h.gradient_resolved) refinements << ',' << int(x);
                refinements << '\n';
            }
        }
    }
}
