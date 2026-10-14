#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;

int main(int argc,char**argv) {
    if(argc!=5) {
        std::cerr<<"usage: adaptive_ab CASES_TSV RESULT_TSV PANELS_TSV EPOCH_TSV\n";
        return 2;
    }
    LensParams p;
    bool found=false;
    std::ifstream cases(argv[1]);
    std::string line;
    int row=0;
    while(std::getline(cases,line)) {
        ++row;
        std::istringstream ss(line);
        int barycentric=0;
        double u=0,mu=0;
        std::string name;
        if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>barycentric>>u>>mu>>name))continue;
        if(row==17&&name=="caustic-cross"&&u==0.0) {
            p.barycentric=barycentric!=0;
            found=true;
            break;
        }
    }
    if(!found)return 3;

    const auto topo=classify_cells(PrimaryFrame::from(p),nullptr,nullptr,true);
#if defined(HOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH)
    const char* variant="projective_fold";
#else
    const char* variant="baseline";
#endif
    std::ofstream results(argv[2],std::ios::trunc);
    std::ofstream panels(argv[3],std::ios::trunc);
    std::ofstream epochs(argv[4],std::ios::trunc);
    if(!results||!panels||!epochs)return 4;
    results<<std::setprecision(17)
       <<"lane\tpolicy\ttopology_status\tevent_count\tcell_count\t"
         "value_converged\tvalue_stop\tmu\tvalue_error\t"
         "numerical_status\tstop\tgrad_a\tgrad_a_quality\tgrad_a_reason\t"
         "grad_a_error\tgrad_a_converged\tnodes\tpanels\tsplits\t"
         "projective_map_cells\tmap_cell5_left\tmap_cell7_left\t"
         "map_cell7_right\n";
    panels<<std::setprecision(17)
       <<"lane\tpolicy\tpanel\tcell\tr_lo\tr_hi\tleft_map\tright_map\t"
         "left_radius_lo\tright_radius_lo\tleft_uncertainty\tright_uncertainty\n";
    epochs<<std::setprecision(17)
       <<"lane\tpolicy\twall_ms\tvalue_converged\tvalue_stop\tmu\t"
         "value_error\tnumerical_status\tstop\tgrad_a\tgrad_a_quality\t"
         "grad_a_reason\tgrad_a_error\tgrad_a_converged\tnodes\tpanels\t"
         "topology_ms\tphysical_ms\n";

    for(const auto lane:{std::string("value"),std::string("value+5Jac")}) {
        AdaptiveConfig cfg;
        cfg.gradient_policy=lane=="value"?GradientPolicy::None:
                                             GradientPolicy::ValueFirst;
        cfg.with_jacobian=lane!="value";
        cfg.collect_diagnostics=true;
        cfg.tol.mu_rtol=1e-4;
        cfg.tol.grad_rtol.fill(1e-3);
        AdaptiveWorkspace workspace;
        const auto result=flux_adaptive_integrate(p,0.0,topo,cfg,workspace);

        bool cell5_left=false,cell7_left=false,cell7_right=false;
        for(std::size_t i=0;i<workspace.panels.size();++i) {
            const auto& panel=workspace.panels[i];
            const auto& cell=workspace.cells[panel.cell];
            if(panel.cell==5)cell5_left=cell5_left||panel.map.left;
            if(panel.cell==7) {
                cell7_left=cell7_left||panel.map.left;
                cell7_right=cell7_right||panel.map.right;
            }
            panels<<variant<<'\t'<<lane<<'\t'<<i<<'\t'<<panel.cell<<'\t'
                  <<cell.r_lo<<'\t'<<cell.r_hi<<'\t'<<panel.map.left<<'\t'
                  <<panel.map.right<<'\t'<<panel.left_radius_lo<<'\t'
                  <<panel.right_radius_lo<<'\t'<<panel.left_uncertainty<<'\t'
                  <<panel.right_uncertainty<<'\n';
        }
        results<<variant<<'\t'<<gradient_policy_name(cfg.gradient_policy)<<'\t'
               <<static_cast<int>(topo.status)<<'\t'<<topo.events.size()<<'\t'
               <<topo.cells.size()<<'\t'<<result.value_converged<<'\t'
               <<adaptive_stop_name(result.value_stop_reason)<<'\t'<<result.mu<<'\t'
               <<result.estimated_abs_error_mu<<'\t'
               <<static_cast<int>(result.numerical_status)<<'\t'
               <<adaptive_stop_name(result.stop)<<'\t'<<result.grad_mu[4]<<'\t'
               <<gradient_quality_name(result.grad_quality[4])<<'\t'
               <<gradient_reason_name(result.grad_reason[4])<<'\t'
               <<result.estimated_abs_error_grad[4]<<'\t'
               <<result.grad_converged[4]<<'\t'<<result.stats.unique_nodes<<'\t'
               <<result.stats.panels<<'\t'<<result.stats.splits<<'\t'
               <<(int(cell5_left)+int(cell7_left))<<'\t'<<cell5_left<<'\t'<<cell7_left<<'\t'
               <<cell7_right<<'\n';

        // Whole adaptive epoch A/B: this repeats D14/topology construction
        // through the same epoch entry point used by adaptive callers.
        AdaptiveWorkspace epoch_workspace;
        const auto begin=std::chrono::steady_clock::now();
        const AdaptiveResult epoch_result = lane=="value"
            ? epoch_value_adaptive(p,0.0,cfg,epoch_workspace)
            : epoch_jacobian_adaptive(p,0.0,cfg,epoch_workspace);
        const double wall_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-begin).count();
        epochs<<variant<<'\t'<<gradient_policy_name(cfg.gradient_policy)<<'\t'
              <<wall_ms<<'\t'<<epoch_result.value_converged<<'\t'
              <<adaptive_stop_name(epoch_result.value_stop_reason)<<'\t'
              <<epoch_result.mu<<'\t'<<epoch_result.estimated_abs_error_mu<<'\t'
              <<static_cast<int>(epoch_result.numerical_status)<<'\t'
              <<adaptive_stop_name(epoch_result.stop)<<'\t'
              <<epoch_result.grad_mu[4]<<'\t'
              <<gradient_quality_name(epoch_result.grad_quality[4])<<'\t'
              <<gradient_reason_name(epoch_result.grad_reason[4])<<'\t'
              <<epoch_result.estimated_abs_error_grad[4]<<'\t'
              <<epoch_result.grad_converged[4]<<'\t'
              <<epoch_result.stats.unique_nodes<<'\t'<<epoch_result.stats.panels<<'\t'
              <<epoch_result.stats.topology_ms<<'\t'<<epoch_result.stats.physical_ms<<'\n';
    }
    return 0;
}
