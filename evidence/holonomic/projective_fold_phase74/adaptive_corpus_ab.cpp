#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;

int main(int argc,char**argv) {
    if(argc!=3) {
        std::cerr<<"usage: adaptive_corpus_ab CASES_TSV OUTPUT_TSV\n";
        return 2;
    }
#if defined(HOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH)
    const char* variant="projective_fold";
#else
    const char* variant="baseline";
#endif
    std::ifstream in(argv[1]);
    std::ofstream out(argv[2],std::ios::trunc);
    if(!in||!out)return 2;
    out<<std::setprecision(17)
       <<"lane\trow\tcase\tu\tmu_reference\tpolicy\ttopology_status\t"
         "topology_events\ttopology_cells\tchart_events\tprojective_promotions\t"
         "event_identity_unchanged\tcell_plan_unchanged\tvalue_converged\t"
         "value_stop\tmu\tvalue_error\tnumerical_status\tstop\twall_ms\t"
         "grad_x\tgrad_y\tgrad_rho\t"
         "grad_q\tgrad_a\tquality_x\tquality_y\tquality_rho\tquality_q\t"
         "quality_a\tgradient_a_error\tnodes\tpanels\tsplits\n";
    std::string line;
    int row=0;
    while(std::getline(in,line)) {
        ++row;
        std::istringstream ss(line);
        LensParams p;
        int barycentric=0;
        double u=0,mu_reference=0;
        std::string name;
        if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>barycentric>>u>>mu_reference>>name))continue;
        p.barycentric=barycentric!=0;
        const auto topology=classify_cells(PrimaryFrame::from(p),nullptr,nullptr,true);
        auto certified=topology;
        adaptive_detail::certify_projective_p4_events(certified,PrimaryFrame::from(p));
        std::size_t chart_events=0,promotions=0;
        for(const auto& event:topology.events)chart_events+=event.kind=="chart_p4";
        for(const auto& event:certified.events)promotions+=event.projective_fold_certified;
        bool same_cells=topology.cells.size()==certified.cells.size();
        if(same_cells)for(std::size_t i=0;i<topology.cells.size();++i) {
            const auto& a=topology.cells[i];const auto& b=certified.cells[i];
            same_cells=same_cells&&a.r_lo==b.r_lo&&a.r_hi==b.r_hi&&
                       a.kind==b.kind&&a.n_crossings==b.n_crossings&&a.status==b.status;
        }
        const bool same_event_count=topology.events.size()==certified.events.size();
        bool same_event_identity=same_event_count;
        if(same_event_identity)for(std::size_t i=0;i<topology.events.size();++i)
            same_event_identity=same_event_identity&&
                topology.events[i].kind==certified.events[i].kind&&
                topology.events[i].radius==certified.events[i].radius;
        for(const auto policy:{GradientPolicy::None,GradientPolicy::ValueFirst}) {
            AdaptiveConfig cfg;
            cfg.gradient_policy=policy;
            cfg.with_jacobian=policy!=GradientPolicy::None;
            cfg.tol.mu_rtol=1e-4;
            cfg.tol.grad_rtol.fill(1e-3);
            AdaptiveWorkspace workspace;
            const auto start=std::chrono::steady_clock::now();
            const auto result=policy==GradientPolicy::None
                ? epoch_value_adaptive(p,u,cfg,workspace)
                : epoch_jacobian_adaptive(p,u,cfg,workspace);
            const double wall_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count();
            out<<variant<<'\t'<<row<<'\t'<<name<<'\t'<<u<<'\t'<<mu_reference<<'\t'
               <<gradient_policy_name(policy)<<'\t'<<static_cast<int>(topology.status)<<'\t'
               <<topology.events.size()<<'\t'<<topology.cells.size()<<'\t'
               <<chart_events<<'\t'<<promotions<<'\t'<<same_event_identity<<'\t'
               <<same_cells<<'\t'<<result.value_converged<<'\t'
               <<adaptive_stop_name(result.value_stop_reason)<<'\t'<<result.mu<<'\t'
               <<result.estimated_abs_error_mu<<'\t'
               <<static_cast<int>(result.numerical_status)<<'\t'
               <<adaptive_stop_name(result.stop)<<'\t'<<wall_ms;
            for(double g:result.grad_mu)out<<'\t'<<g;
            for(auto q:result.grad_quality)out<<'\t'<<gradient_quality_name(q);
            out<<'\t'<<result.estimated_abs_error_grad[4]<<'\t'
               <<result.stats.unique_nodes<<'\t'<<result.stats.panels<<'\t'
               <<result.stats.splits<<'\n';
        }
    }
    return 0;
}
