#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: budget_probe CASES_TSV OUTPUT_TSV\n";
        return 2;
    }
    LensParams p;
    bool found = false;
    std::ifstream in(argv[1]);
    std::string line;
    int row = 0, barycentric = 0;
    double input_u = 0.0, reference_mu = 0.0;
    std::string name;
    while (std::getline(in, line)) {
        ++row;
        std::istringstream ss(line);
        if (!(ss >> p.xs >> p.ys >> p.rho >> p.q >> p.a >> barycentric >>
              input_u >> reference_mu >> name)) continue;
        if (row == 17 && name == "caustic-cross") {
            found = true;
            break;
        }
    }
    if (!found) return 3;
    p.barycentric = barycentric != 0;
    constexpr double u = 0.5;
    const auto topology = classify_cells(PrimaryFrame::from(p), nullptr,
                                         nullptr, true);
    if (topology.status != Status::OK) return 4;

    std::ofstream out(argv[2], std::ios::trunc);
    if (!out) return 5;
    out << std::setprecision(17)
        << "arm\tu\tvalue_rtol\tgradient_round_budget\tgradient_node_budget"
           "\tvalue_ok\tvalue_stop\tmu\tmu_error\tgrad_x\tquality_x"
           "\treason_x\tgrad_error_x\tradial_x\tinner_x\tgeometry_x"
           "\tevent_x\troundoff_x\tnodes\tevaluations\tpanels\tsplits"
           "\tfirst_grad_error_pass_nodes\tprojective_folds\n";
#if defined(HOLO_ADAPTIVE_DIAG_RADIAL_ONLY_AFTER_INNER)
    const char* arm = "projective_radial_only";
#elif defined(HOLO_ADAPTIVE_KJAC_24_DIAGNOSTIC)
    const char* arm = "projective_kjac24";
#elif defined(HOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH)
    const char* arm = "projective";
#else
    const char* arm = "baseline";
#endif
    int projective_folds = 0;
    auto marked = topology;
    adaptive_detail::certify_projective_p4_events(marked, PrimaryFrame::from(p));
    for (const auto& event : marked.events)
        projective_folds += event.projective_fold_certified ? 1 : 0;

    for (double rtol : {1e-3, 1e-4}) {
#if defined(HOLO_ADAPTIVE_DIAG_RADIAL_ONLY_AFTER_INNER)
        for (int rounds : {8, 16, 32, 64}) {
#else
        for (int rounds : {0, 1, 2, 4, 8, 16, 32, 64}) {
#endif
            AdaptiveConfig cfg;
            cfg.gradient_policy = GradientPolicy::ValueFirst;
            cfg.with_jacobian = true;
            cfg.collect_diagnostics = false;
            cfg.tol.mu_rtol = rtol;
            cfg.value_first_gradient_round_budget = rounds;
            cfg.value_first_gradient_node_budget = 65536;
            cfg.max_node_evals = 65536;
            cfg.max_level = 8;  // production p-depth cap; budget continues h-splits
            cfg.max_depth = 16;
            AdaptiveWorkspace workspace;
            const auto result = flux_adaptive_integrate(
                p, u, topology, cfg, workspace);
            out << arm << '\t' << u << '\t' << rtol << '\t' << rounds << '\t'
                << cfg.value_first_gradient_node_budget << '\t'
                << result.value_converged << '\t'
                << adaptive_stop_name(result.value_stop_reason) << '\t'
                << result.mu << '\t' << result.estimated_abs_error_mu << '\t'
                << result.grad_mu[0] << '\t'
                << gradient_quality_name(result.grad_quality[0]) << '\t'
                << gradient_reason_name(result.grad_reason[0]) << '\t'
                << result.estimated_abs_error_grad[0] << '\t'
                << result.radial_error[1] << '\t' << result.inner_error[1]
                << '\t' << result.geometry_error[1] << '\t'
                << result.event_error[1] << '\t' << result.roundoff_error[1]
                << '\t' << result.stats.unique_nodes << '\t'
                << result.stats.node_evaluations << '\t' << result.stats.panels
                << '\t' << result.stats.splits << '\t'
                << result.stats.first_gradient_error_pass_nodes << '\t'
                << projective_folds << '\n';
        }
    }
    return 0;
}
