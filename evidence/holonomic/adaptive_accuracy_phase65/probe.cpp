#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
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
    int case_id = 0;
    int configuration_id = 0;
    std::string profile;
    int d_bin_index = 0;
    int epoch_index = 0;
    double s = 0.0;
    double q = 0.0;
    double rho = 0.0;
    double x = 0.0;
    double y = 0.0;
    double time = 0.0;
    double u = 0.0;
    double X = 0.0;
    double reference = 0.0;
};

struct EpochPlan {
    InputRow input;
    LensParams params;
    TopologyResult radial_topology;
};

struct RunCapture {
    AdaptiveResult result;
    PreparedReuseStats reuse{};
    double whole_ms = 0.0;
};

static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double x = 0.5 * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(x);
    const double f = x - static_cast<double>(lo);
    if (lo + 1 >= values.size()) return values.back();
    return values[lo] * (1.0 - f) + values[lo + 1] * f;
}

static double median3(const std::vector<double>& values) {
    return median(values);
}

static AdaptiveConfig make_config(double rtol) {
    AdaptiveConfig cfg;
    cfg.gradient_policy = GradientPolicy::None;
    cfg.with_jacobian = false;
    cfg.fold_maps = true;
    cfg.reuse_samples = true;
    cfg.tol.mu_atol = 1e-16;
    cfg.tol.mu_rtol = rtol;
    cfg.radial_error_estimator = RadialErrorEstimator::HybridEmbedded;
    cfg.nested_difference_safety = 2.0;
    return cfg;
}

static bool read_row(const std::string& line, InputRow& row) {
    if (line.empty() || line[0] == '#') return false;
    std::istringstream in(line);
    return static_cast<bool>(
        in >> row.case_id >> row.configuration_id >> row.profile >>
        row.d_bin_index >> row.epoch_index >> row.s >> row.q >> row.rho >>
        row.x >> row.y >> row.time >> row.u >> row.X >> row.reference);
}

static bool same_trajectory(const InputRow& a, const InputRow& b) {
    return a.case_id == b.case_id &&
           a.configuration_id == b.configuration_id &&
           a.profile == b.profile && a.d_bin_index == b.d_bin_index;
}

// This is the production adaptive cold control flow: no prepared state is
// passed, so every call constructs D14/topology afresh.  epoch_adaptive also
// keeps its validation outside the topology timer, exactly as the API does.
static RunCapture run_full_cold(const EpochPlan& e, const AdaptiveConfig& cfg,
                                AdaptiveWorkspace& workspace) {
    const auto start = Clock::now();
    RunCapture capture;
    capture.result = epoch_adaptive(e.params, e.input.u, cfg, workspace);
    capture.whole_ms = elapsed_ms(start);
    return capture;
}

// L2 trajectory warm path.  L1 verbatim cell-plan reuse is deliberately off;
// classify_cells remains the topology authority while its all-root D14 solve
// receives the previous epoch's certified roots.
static RunCapture run_full_warm(const EpochPlan& e, const AdaptiveConfig& cfg,
                                AdaptiveWorkspace& workspace,
                                PreparedEpochGeometry& state,
                                const PreparedReuseConfig& reuse) {
    const auto start = Clock::now();
    RunCapture capture;
    const auto topology_start = Clock::now();
    TopologyResult topo = prepared_topology(
        PrimaryFrame::from(e.params), state, reuse, &capture.reuse);
    const double topology_ms = elapsed_ms(topology_start);
    capture.result = flux_adaptive_integrate(
        e.params, e.input.u, topo, cfg, workspace);
    capture.result.stats.topology_ms = topology_ms;
    capture.whole_ms = elapsed_ms(start);
    return capture;
}

// Pure radial/adaptive body.  The same topology object is passed to every
// lane, and its D14 construction is outside this timer by design.  This is
// the scope used by the earlier q-rho adaptive benchmark.
static RunCapture run_radial_only(const EpochPlan& e, const AdaptiveConfig& cfg,
                                  AdaptiveWorkspace& workspace) {
    const auto start = Clock::now();
    RunCapture capture;
    capture.result = flux_adaptive_integrate(
        e.params, e.input.u, e.radial_topology, cfg, workspace);
    capture.whole_ms = elapsed_ms(start);
    return capture;
}

static void print_header(std::ofstream& out) {
    out << "# V2 adaptive trajectory benchmark; value-only; "
           "radial-only topology prebuilt; full-cold D14/topology per epoch; "
           "full-warm L2 D14 warm seed; repeats median; n_r=adaptive\n";
    out << "case_id configuration_id profile d_bin_index epoch_index trajectory_id trajectory_pos target "
           "s q rho x y time u X reference "
           "topology_status topology_cells topology_events "
           "cold_ms warm_ms radial_ms cold_topology_ms warm_topology_ms radial_topology_ms "
           "cold_adaptive_ms warm_adaptive_ms radial_adaptive_ms "
           "cold_setup_ms warm_setup_ms radial_setup_ms "
           "cold_physical_ms warm_physical_ms radial_physical_ms "
           "cold_estimator_ms warm_estimator_ms radial_estimator_ms "
           "cold_scheduler_ms warm_scheduler_ms radial_scheduler_ms "
           "cold_mu warm_mu radial_mu cold_error warm_error radial_error "
           "cold_value_converged warm_value_converged radial_value_converged "
           "cold_stop warm_stop radial_stop cold_status warm_status radial_status "
           "cold_nodes warm_nodes radial_nodes cold_evaluations warm_evaluations radial_evaluations "
           "cold_panels warm_panels radial_panels cold_splits warm_splits radial_splits "
           "warm_l1 warm_l2 warm_l3 warm_rescreen_fail warm_seed_used\n";
}

static void print_result_fields(std::ofstream& out, const RunCapture& cold,
                                const RunCapture& warm, const RunCapture& radial,
                                const EpochPlan& e, int trajectory_id,
                                int trajectory_pos, double target,
                                double cold_ms, double warm_ms,
                                double radial_ms) {
    const auto& c = cold.result;
    const auto& w = warm.result;
    const auto& r = radial.result;
    const double c_topo = c.stats.topology_ms;
    const double w_topo = w.stats.topology_ms;
    out << e.input.case_id << ' ' << e.input.configuration_id << ' '
        << e.input.profile << ' ' << e.input.d_bin_index << ' '
        << e.input.epoch_index << ' ' << trajectory_id << ' '
        << trajectory_pos << ' ' << target << ' ' << e.input.s << ' '
        << e.input.q << ' ' << e.input.rho << ' ' << e.input.x << ' '
        << e.input.y << ' ' << e.input.time << ' ' << e.input.u << ' '
        << e.input.X << ' ' << e.input.reference << ' '
        << to_string(e.radial_topology.status) << ' '
        << e.radial_topology.cells.size() << ' '
        << e.radial_topology.events.size() << ' '
        << cold_ms << ' ' << warm_ms << ' ' << radial_ms << ' '
        << c_topo << ' ' << w_topo << " 0 "
        << std::max(0.0, cold_ms - c_topo) << ' '
        << std::max(0.0, warm_ms - w_topo) << ' ' << radial_ms << ' '
        << c.stats.setup_ms << ' ' << w.stats.setup_ms << ' ' << r.stats.setup_ms << ' '
        << c.stats.physical_ms << ' ' << w.stats.physical_ms << ' ' << r.stats.physical_ms << ' '
        << c.stats.estimator_ms << ' ' << w.stats.estimator_ms << ' ' << r.stats.estimator_ms << ' '
        << c.stats.scheduler_ms << ' ' << w.stats.scheduler_ms << ' ' << r.stats.scheduler_ms << ' '
        << c.mu << ' ' << w.mu << ' ' << r.mu << ' '
        << c.estimated_abs_error_mu << ' ' << w.estimated_abs_error_mu << ' '
        << r.estimated_abs_error_mu << ' ' << int(c.value_converged) << ' '
        << int(w.value_converged) << ' ' << int(r.value_converged) << ' '
        << adaptive_stop_name(c.stop) << ' ' << adaptive_stop_name(w.stop) << ' '
        << adaptive_stop_name(r.stop) << ' ' << to_string(c.numerical_status) << ' '
        << to_string(w.numerical_status) << ' ' << to_string(r.numerical_status) << ' '
        << c.stats.unique_nodes << ' ' << w.stats.unique_nodes << ' ' << r.stats.unique_nodes << ' '
        << c.stats.node_evaluations << ' ' << w.stats.node_evaluations << ' '
        << r.stats.node_evaluations << ' ' << c.stats.panels << ' ' << w.stats.panels << ' '
        << r.stats.panels << ' ' << c.stats.splits << ' ' << w.stats.splits << ' '
        << r.stats.splits << ' ' << warm.reuse.l1_topology_reuse << ' '
        << warm.reuse.l2_warm_recompute << ' ' << warm.reuse.l3_cold_recompute << ' '
        << warm.reuse.rescreen_fail << ' ' << warm.reuse.warm_solve_used << '\n';
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: v2_adaptive_trajectory_runner INPUT OUTPUT [reps]\n";
        return 2;
    }
    const int reps = argc > 3 ? std::max(1, std::atoi(argv[3])) : 3;
    std::ifstream in(argv[1]);
    std::ofstream out(argv[2]);
    if (!in || !out) {
        std::cerr << "cannot open input/output\n";
        return 2;
    }

    // Match the existing V2 q-rho benchmark: force the current V2 radial
    // route and keep the experimental coupled ODE path out of the comparison.
    holo_mv_transport_override() = 1;
    holo_holonomic_transport_override() = 1;
    holo_ode_transport_override() = 0;

    std::vector<InputRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        InputRow row;
        if (read_row(line, row)) rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        std::cerr << "no input rows\n";
        return 3;
    }
    std::stable_sort(rows.begin(), rows.end(), [](const InputRow& a, const InputRow& b) {
        if (a.case_id != b.case_id) return a.case_id < b.case_id;
        if (a.configuration_id != b.configuration_id)
            return a.configuration_id < b.configuration_id;
        if (a.profile != b.profile) return a.profile < b.profile;
        if (a.d_bin_index != b.d_bin_index) return a.d_bin_index < b.d_bin_index;
        return a.epoch_index < b.epoch_index;
    });

    print_header(out);
    out << std::setprecision(17);
    const std::array<double, 4> targets{{1e-4, 1e-5, 1e-6, 1e-7}};
    const std::size_t expected_trajectories = rows.size() / 4;
    std::size_t trajectory_id = 0;
    std::size_t emitted = 0;
    volatile double sink = 0.0;

    for (std::size_t begin = 0; begin < rows.size();) {
        std::size_t end = begin + 1;
        while (end < rows.size() && same_trajectory(rows[begin], rows[end])) ++end;
        if (end - begin != 4) {
            std::cerr << "trajectory has " << (end - begin)
                      << " rows at sorted offset " << begin << "\n";
            return 4;
        }

        std::vector<EpochPlan> plan;
        plan.reserve(end - begin);
        for (std::size_t i = begin; i < end; ++i) {
            EpochPlan e;
            e.input = rows[i];
            // This is the verified mapping used by the existing VBM corpus.
            e.params = LensParams{e.input.time, e.input.y, e.input.rho,
                                  1.0 / e.input.q, e.input.s, true};
            e.radial_topology = classify_cells(
                PrimaryFrame::from(e.params), nullptr, nullptr,
                /*retain_adaptive_metadata=*/true);
            plan.push_back(std::move(e));
        }

        for (double target : targets) {
            const AdaptiveConfig cfg = make_config(target);

            // One unmeasured trajectory per lane warms instruction/cache state
            // and workspace capacity.  The measured cold lane still solves
            // D14/topology from scratch; the measured warm lane starts with a
            // new prepared state and only reuses roots between its epochs.
            {
                AdaptiveWorkspace w;
                for (const auto& e : plan) sink += run_radial_only(e, cfg, w).result.mu;
            }
            {
                AdaptiveWorkspace w;
                for (const auto& e : plan) sink += run_full_cold(e, cfg, w).result.mu;
            }
            {
                AdaptiveWorkspace w;
                PreparedEpochGeometry state;
                PreparedReuseConfig reuse;
                reuse.allow_topology_reuse = false;
                reuse.allow_warm_d14 = true;
                reuse.l2_drift = 1e18;
                for (const auto& e : plan)
                    sink += run_full_warm(e, cfg, w, state, reuse).result.mu;
            }

            const std::size_t n = plan.size();
            std::vector<std::vector<double>> cold_times(n), warm_times(n), radial_times(n);
            std::vector<std::vector<RunCapture>> cold_runs(n), warm_runs(n), radial_runs(n);
            for (int rep = 0; rep < reps; ++rep) {
                {
                    AdaptiveWorkspace w;
                    for (std::size_t i = 0; i < n; ++i) {
                        RunCapture x = run_full_cold(plan[i], cfg, w);
                        cold_times[i].push_back(x.whole_ms);
                        sink += x.result.mu;
                        cold_runs[i].push_back(std::move(x));
                    }
                }
                {
                    AdaptiveWorkspace w;
                    PreparedEpochGeometry state;
                    PreparedReuseConfig reuse;
                    reuse.allow_topology_reuse = false;
                    reuse.allow_warm_d14 = true;
                    reuse.l2_drift = 1e18;
                    for (std::size_t i = 0; i < n; ++i) {
                        RunCapture x = run_full_warm(plan[i], cfg, w, state, reuse);
                        warm_times[i].push_back(x.whole_ms);
                        sink += x.result.mu;
                        warm_runs[i].push_back(std::move(x));
                    }
                }
                {
                    AdaptiveWorkspace w;
                    for (std::size_t i = 0; i < n; ++i) {
                        RunCapture x = run_radial_only(plan[i], cfg, w);
                        radial_times[i].push_back(x.whole_ms);
                        sink += x.result.mu;
                        radial_runs[i].push_back(std::move(x));
                    }
                }
            }

            auto choose_stage_sample = [](const std::vector<RunCapture>& runs,
                                          const std::vector<double>& times) {
                const double m = median(times);
                std::size_t best = 0;
                double distance = std::numeric_limits<double>::infinity();
                for (std::size_t i = 0; i < runs.size(); ++i) {
                    const double d = std::fabs(times[i] - m);
                    if (d < distance) {
                        distance = d;
                        best = i;
                    }
                }
                return runs[best];
            };
            std::vector<RunCapture> cold_capture(n), warm_capture(n), radial_capture(n);
            for (std::size_t i = 0; i < n; ++i) {
                cold_capture[i] = choose_stage_sample(cold_runs[i], cold_times[i]);
                warm_capture[i] = choose_stage_sample(warm_runs[i], warm_times[i]);
                radial_capture[i] = choose_stage_sample(radial_runs[i], radial_times[i]);
            }

            for (std::size_t i = 0; i < n; ++i) {
                print_result_fields(out, cold_capture[i], warm_capture[i],
                                    radial_capture[i], plan[i],
                                    static_cast<int>(trajectory_id),
                                    static_cast<int>(i), target,
                                    median3(cold_times[i]), median3(warm_times[i]),
                                    median3(radial_times[i]));
                ++emitted;
            }
        }
        ++trajectory_id;
        begin = end;
        if ((trajectory_id % 32) == 0)
            std::cerr << "progress trajectories " << trajectory_id << "/"
                      << expected_trajectories << " rows=" << emitted << "\n";
    }

    std::cerr << "wrote " << emitted << " rows from " << trajectory_id
              << " trajectories, reps=" << reps << ", sink=" << sink << '\n';
    return 0;
}
