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

static std::vector<DiagnosticCase> read_cases(const char* path,
                                              const std::string& wanted_name) {
    std::ifstream input(path);
    std::vector<DiagnosticCase> cases;
    std::string line;
    while(std::getline(input,line)) {
        DiagnosticCase c;
        int bary=0;
        double old=0;
        std::istringstream s(line);
        if(s>>c.p.xs>>c.p.ys>>c.p.rho>>c.p.q>>c.p.a>>bary>>c.u>>old>>c.name &&
           c.name==wanted_name) {
            c.p.barycentric=bary;
            cases.push_back(std::move(c));
        }
    }
    return cases;
}

int main(int argc,char** argv) {
    if(argc<3) return 2;
    // Keep rand033 as the historical default, while allowing a named hard
    // case to be captured without rewriting its provenance in the TSV.
    const auto cases=read_cases(argv[1],argc>3?argv[3]:"rand033");
    const std::string outdir=argv[2];
    std::ofstream epochs(outdir+"/epochs.csv");
    std::ofstream events(outdir+"/events.csv");
    std::ofstream refinements(outdir+"/refinements.csv");
    std::ofstream samples(outdir+"/samples.csv");
    std::ofstream sample_counts(outdir+"/sample_reason_counts.csv");
    if(!epochs||!events||!refinements||!samples||!sample_counts) return 3;
    epochs << std::setprecision(17);
    events << std::setprecision(17);
    refinements << std::setprecision(17);
    samples << std::setprecision(17);
    sample_counts << std::setprecision(17);

    epochs << "name,u,policy,warm,rtol,stop,value_stop,gradient_stop,value_converged,mu,value_error,nodes,evaluations,event_count,refinement_count,topology_ms,setup_ms,physics_ms,estimator_ms,scheduler_ms,event_double_checks,event_dd_checks,event_dd_accepts,event_qf_refinements,qf_family_constructions,event_topology_reuses,event_radius_reuses,event_direct_qf_failures\n";
    events << "name,u,policy,warm,rtol,index,input_radius,selected_radius,radius_lo,uncertainty,t_seed,d14_condition,t_seed_valid,topology_reused,precision_tier,double_reason,dd_reason,qf_reason,double_residual,dd_residual\n";
    refinements << "name,u,policy,warm,rtol,index,panel,cell,parent,depth,level,node_evaluations,value_error,gradient_error0,gradient_error1,gradient_error2,gradient_error3,gradient_error4,gradient_phase,value_resolved,gradient_resolved0,gradient_resolved1,gradient_resolved2,gradient_resolved3,gradient_resolved4\n";
    samples << "name,u,policy,warm,rtol,index,panel,node_slot,cell,level,R,reason,cold_retry,cold_retry_succeeded\n";
    sample_counts << "name,u,policy,warm,rtol,reason,count,cold_count,cold_retries,cold_retry_successes\n";

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
                   << r.stats.scheduler_ms << ',' << r.stats.event_double_checks << ','
                   << r.stats.event_dd_checks << ',' << r.stats.event_dd_accepts << ','
                   << r.stats.event_qf_refinements << ',' << r.stats.qf_family_constructions << ','
                   << r.stats.event_topology_reuses << ',' << r.stats.event_radius_reuses << ','
                   << r.stats.event_direct_qf_failures << '\n';
            for(size_t i=0;i<r.stats.event_diagnostics.size();++i) {
                const auto& e=r.stats.event_diagnostics[i];
                events << c.name << ',' << c.u << ',' << gradient_policy_name(policy) << ',' << int(warm)
                       << ',' << tol << ',' << i << ',' << e.input_radius << ','
                       << e.selected_radius << ',' << e.radius_lo << ',' << e.uncertainty << ','
                       << e.t_seed << ',' << e.d14_condition << ',' << int(e.t_seed_valid) << ','
                       << int(e.topology_reused) << ','
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
            for(size_t i=0;i<r.stats.sample_diagnostics.size();++i) {
                const auto& d=r.stats.sample_diagnostics[i];
                samples << c.name << ',' << c.u << ',' << gradient_policy_name(policy)
                        << ',' << int(warm) << ',' << tol << ',' << i << ','
                        << d.panel << ',' << d.node_slot << ',' << d.cell << ','
                        << d.level << ',' << d.R << ','
                        << adaptive_sample_reject_reason_name(d.reason) << ','
                        << int(d.cold_retry) << ',' << int(d.cold_retry_succeeded)
                        << '\n';
            }
            for(std::size_t i=0;i<r.stats.sample_reject_counts.size();++i) {
                const auto reason=static_cast<AdaptiveSampleRejectReason>(i);
                sample_counts << c.name << ',' << c.u << ','
                              << gradient_policy_name(policy) << ',' << int(warm)
                              << ',' << tol << ','
                              << adaptive_sample_reject_reason_name(reason) << ','
                              << r.stats.sample_reject_counts[i] << ','
                              << r.stats.sample_cold_reject_counts[i] << ','
                              << r.stats.sample_cold_retries << ','
                              << r.stats.sample_cold_retry_successes << '\n';
            }
        }
    }
}
