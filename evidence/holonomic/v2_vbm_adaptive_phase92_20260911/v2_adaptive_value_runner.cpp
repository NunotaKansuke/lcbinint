#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;
using Clock = std::chrono::steady_clock;

static double median3(double a, double b, double c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return b;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: v2_adaptive_value_runner INPUT OUTPUT\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    std::ofstream out(argv[2]);
    if (!in || !out) {
        std::cerr << "cannot open input/output\n";
        return 2;
    }

    // Use the current V2 value path and keep the experimental ODE lane out of
    // this comparison.  The topology is prepared once per input row, outside
    // the timed adaptive radial body, matching the existing VBM pure-kernel
    // timing boundary.  The retained D14/fold metadata is still passed into
    // flux_adaptive_integrate, so Phase 9.2 event reuse is exercised.
    holo_mv_transport_override() = 1;
    holo_holonomic_transport_override() = 1;
    holo_ode_transport_override() = 0;

    out << "# current V2 adaptive value-only kernel; HEAD=46b82b9; "
           "topology prebuilt outside timer; adaptive RelTol; mu_atol=1e-16; "
           "gradient_policy=None; repeats=3 median\n";
    out << "# case_id configuration_id profile d_bin_index epoch_index target "
           "s q rho x y time u X reference topology_status topology_status_code "
           "topology_cells topology_uncertain_cells physical_events "
           "v2_mu value_error value_converged "
           "stop value_stop numerical_status status_code ms nodes evaluations "
           "reused_nodes panels splits setup_ms physics_ms estimator_ms "
           "scheduler_ms setup_event_ms event_topology_reuses "
           "event_radius_reuses event_qf_refinements\n";
    out << std::setprecision(17);

    std::string line;
    std::size_t count = 0;
    volatile double sink = 0.0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int case_id = 0, configuration_id = 0, d_bin_index = 0, epoch_index = 0;
        std::string profile;
        double s = 0, q = 0, rho = 0, x = 0, y = 0, time = 0, u = 0,
               X = 0, reference = 0;
        if (!(ss >> case_id >> configuration_id >> profile >> d_bin_index >>
              epoch_index >> s >> q >> rho >> x >> y >> time >> u >> X >>
              reference)) {
            std::cerr << "malformed input row: " << line << "\n";
            return 3;
        }

        // This is the verified mapping used by the existing VBM corpus.
        const LensParams p{time, y, rho, 1.0 / q, s, true};
        const PrimaryFrame pf = PrimaryFrame::from(p);
        const TopologyResult topo = classify_cells(
            pf, nullptr, nullptr, /*retain_adaptive_metadata=*/true);
        size_t topology_uncertain_cells = 0;
        for (const auto& cell : topo.cells)
            if (cell.status != Status::OK) ++topology_uncertain_cells;
        size_t physical_events = 0;
        for (const auto& event : topo.events)
            if (event.physically_real && event.kind == "physical_real")
                ++physical_events;

        for (double target : {1e-3, 1e-4}) {
            AdaptiveConfig cfg;
            cfg.gradient_policy = GradientPolicy::None;
            cfg.with_jacobian = false;
            cfg.tol.mu_atol = 1e-16;
            cfg.tol.mu_rtol = target;
            cfg.reuse_samples = true;

            AdaptiveWorkspace workspace;
            // Warm only the adaptive workspace/caches; the timed calls below
            // have the same value contract and a reusable capacity layout.
            const auto warm = flux_adaptive_integrate(p, u, topo, cfg, workspace);
            sink += warm.mu;

            double elapsed[3]{};
            AdaptiveResult measured;
            for (int rep = 0; rep < 3; ++rep) {
                const auto t0 = Clock::now();
                measured = flux_adaptive_integrate(p, u, topo, cfg, workspace);
                const auto t1 = Clock::now();
                elapsed[rep] =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                sink += measured.mu;
            }
            const double ms = median3(elapsed[0], elapsed[1], elapsed[2]);
            out << case_id << ' ' << configuration_id << ' ' << profile << ' '
                << d_bin_index << ' ' << epoch_index << ' ' << target << ' '
                << s << ' ' << q << ' ' << rho << ' ' << x << ' ' << y << ' '
                << time << ' ' << u << ' ' << X << ' ' << reference << ' '
                << to_string(topo.status) << ' '
                << static_cast<int>(topo.status) << ' ' << topo.cells.size() << ' '
                << topology_uncertain_cells << ' ' << physical_events << ' '
                << measured.mu << ' ' << measured.estimated_abs_error_mu << ' '
                << int(measured.value_converged) << ' '
                << adaptive_stop_name(measured.stop) << ' '
                << adaptive_stop_name(measured.value_stop_reason) << ' '
                << to_string(measured.numerical_status) << ' '
                << static_cast<int>(measured.numerical_status) << ' ' << ms << ' '
                << measured.stats.unique_nodes << ' '
                << measured.stats.node_evaluations << ' '
                << measured.stats.reused_nodes << ' ' << measured.stats.panels << ' '
                << measured.stats.splits << ' ' << measured.stats.setup_ms << ' '
                << measured.stats.physical_ms << ' '
                << measured.stats.estimator_ms << ' '
                << measured.stats.scheduler_ms << ' '
                << measured.stats.setup_event_ms << ' '
                << measured.stats.event_topology_reuses << ' '
                << measured.stats.event_radius_reuses << ' '
                << measured.stats.event_qf_refinements << '\n';
            ++count;
        }
        if ((count % 256) == 0)
            std::cerr << "processed " << count << " rows\n";
    }
    std::cerr << "wrote " << count << " adaptive rows; sink=" << sink << '\n';
    return 0;
}
