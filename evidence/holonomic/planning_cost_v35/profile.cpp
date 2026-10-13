#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"

using namespace lcbinint::holonomic;
using Clock = std::chrono::steady_clock;

namespace {

constexpr std::array<double, 2> kRtol{{1e-3, 1e-4}};

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

bool parse_row(const std::string& line, InputRow& row) {
    std::istringstream input(line);
    return static_cast<bool>(input >> row.case_id >> row.configuration_id >>
        row.profile >> row.d_bin_index >> row.epoch_index >> row.s >> row.q >>
        row.rho >> row.x >> row.y >> row.time >> row.u >> row.X >> row.reference);
}

void write_header(std::ostream& output) {
    output << "case_id\tconfiguration_id\tprofile\td_bin_index\tepoch_index"
           << "\trtol\trep\ts\tq\trho\tx\ty\tu\treference_1e-6"
           << "\tseed_ms\tfrontier_ms\tfull_ms\tplanning_ms\tintegration_ms"
           << "\tmu\testimated_abs_error\trelative_error_reference"
           << "\tvalue_converged\tcoverage_certified\tstop\tvalue_stop"
           << "\tnumerical_status\tfrontier_reason\tnodes\tevaluations"
           << "\tradius_probes\tbands\tskipped_seeds\tseed_count\tsupport_proven\tfast_counts\tsturm_counts\n";
}

void write_failure(std::ostream& output, const InputRow& row, double rtol,
                   int rep, const char* reason) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    output << row.case_id << '\t' << row.configuration_id << '\t' << row.profile << '\t'
           << row.d_bin_index << '\t' << row.epoch_index << '\t' << rtol << '\t' << rep
           << '\t' << row.s << '\t' << row.q << '\t' << row.rho << '\t' << row.x << '\t'
           << row.y << '\t' << row.u << '\t' << row.reference;
    for (int i = 0; i < 8; ++i) output << '\t' << nan;
    output << "\t0\t0\texception\texception\tEXCEPTION\t" << reason
           << "\t0\t0\t0\t0\t0\t0\t0\t0\t0\n";
}

void write_result(std::ostream& output, const InputRow& row, double rtol, int rep,
                  double seed_ms, double frontier_ms,
                  const SeededFrontierResult& result, std::size_t seed_count, bool support_proven) {
    const double mu = result.integral.mu;
    const double relerr = std::isfinite(mu) && std::isfinite(row.reference) && row.reference != 0.0
        ? std::fabs(mu - row.reference) / std::fabs(row.reference)
        : std::numeric_limits<double>::quiet_NaN();
    output << row.case_id << '\t' << row.configuration_id << '\t' << row.profile << '\t'
           << row.d_bin_index << '\t' << row.epoch_index << '\t' << rtol << '\t' << rep
           << '\t' << row.s << '\t' << row.q << '\t' << row.rho << '\t' << row.x << '\t'
           << row.y << '\t' << row.u << '\t' << row.reference << '\t'
           << seed_ms << '\t' << frontier_ms << '\t' << seed_ms + frontier_ms << '\t'
           << result.frontier.planning_ms << '\t' << result.frontier.integration_ms << '\t'
           << mu << '\t' << result.integral.estimated_abs_error_mu << '\t' << relerr << '\t'
           << result.integral.value_converged << '\t' << result.frontier.coverage_certified
           << '\t' << adaptive_stop_name(result.integral.stop) << '\t'
           << adaptive_stop_name(result.integral.value_stop_reason) << '\t'
           << to_string(result.integral.numerical_status) << '\t' << result.frontier.reason << '\t'
           << result.integral.stats.unique_nodes << '\t'
           << result.integral.stats.node_evaluations << '\t'
           << result.frontier.radius_probes << '\t' << result.frontier.bands << '\t'
           << result.frontier.skipped_seeds << '\t' << seed_count << '\t' << support_proven
#ifndef HOLO_PLANNING_BASELINE
           << '\t' << result.frontier.invariant_count_probes << '\t' << result.frontier.sturm_count_probes
#else
           << '\t' << 0 << '\t' << result.frontier.radius_probes
#endif
           << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: bench_tolerance_probe INPUT OUTPUT REPS\n";
        return 2;
    }
    const int reps = std::max(1, std::atoi(argv[3]));
    std::ifstream input(argv[1]);
    std::ofstream output(argv[2], std::ios::trunc);
    if (!input || !output) {
        std::cerr << "cannot open input or output\n";
        return 2;
    }
    output << std::setprecision(17);
    write_header(output);

    std::vector<InputRow> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("case_id\t", 0) == 0) continue;
        InputRow row;
        if (!parse_row(line, row)) {
            std::cerr << "malformed input: " << line << '\n';
            return 3;
        }
        rows.push_back(row);
    }

    std::size_t completed = 0;
    for (const auto& row : rows) {
        try {
            if ((row.profile == "uniform" && std::abs(row.u) > 1e-14) ||
                (row.profile == "linear" && std::abs(row.u - 0.5) > 1e-14)) {
                throw std::runtime_error("profile/u mismatch");
            }
            const LensParams p{row.time, row.y, row.rho, 1.0 / row.q, row.s, true};
            for (int rep = 0; rep < reps; ++rep) {
                lcbinint::magnification::PointSourceMagnifier point_magnifier;
                lcbinint::magnification::FiniteSourceSettings settings;
                settings.caustic_bins = 1400;
                lcbinint::magnification::FiniteSourceMagnifier finite_magnifier(settings);
                (void)finite_magnifier.binary_caustic_branches(p.a, p.q);

                const auto seed_start = Clock::now();
                std::vector<lcbinint::SourcePosition> center;
                const auto center_images = point_magnifier.binary_images(
                    p.a, p.q, {p.xs, p.ys});
                center.reserve(center_images.size());
                for (const auto& image : center_images) center.push_back(image.position);
                bool support_proven = false;
                const auto positions = finite_magnifier.cached_binary_image_seeds(
                    point_magnifier, p.a, p.q, {p.xs, p.ys}, p.rho,
                    std::numeric_limits<double>::infinity(), &center, &support_proven);
                const double seed_ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - seed_start).count();

                std::vector<Cplx<double>> seeds;
                seeds.reserve(positions.size());
                for (const auto& image : positions) {
                    seeds.push_back({image.x + (p.q < 1.0 ? p.a : 0.0), image.y});
                }
                if (seeds.empty()) {
                    throw std::runtime_error("seed support unavailable");
                }

                // Rotate tolerance order across repetitions to reduce clock/order bias.
                for (std::size_t j = 0; j < kRtol.size(); ++j) {
                    const std::size_t ti = rep % 2 == 0
                        ? (j + static_cast<std::size_t>(rep)) % kRtol.size()
                        : (kRtol.size() - 1 - j + static_cast<std::size_t>(rep)) % kRtol.size();
                    AdaptiveConfig cfg;
                    cfg.tol.mu_atol = 1e-8;
                    cfg.tol.mu_rtol = kRtol[ti];
                    cfg.gradient_policy = GradientPolicy::None;
                    cfg.with_jacobian = false;
                    AdaptiveWorkspace workspace;
#ifdef HOLO_PLANNING_PROFILE
                    V2Profile profile;
                    V2ProfileScope profile_scope(profile);
#endif
                    const auto start = Clock::now();
                    const auto result = epoch_seeded_radial_frontier(
                        p, row.u, seeds, cfg, workspace, false);
                    const double frontier_ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - start).count();
#ifdef HOLO_PLANNING_PROFILE
                    std::cerr << "PROFILE " << row.case_id << ' ' << row.profile << ' '
                              << kRtol[ti] << ' ' << result.frontier.planning_ms << ' '
                              << profile.frontier_count_ms << ' ' << profile.frontier_event_ms
                              << ' ' << result.frontier.radius_probes << ' ' << profile.frontier_march_ms << ' ' << profile.frontier_repair_ms << ' ' << profile.frontier_endpoint_ms << ' ' << result.frontier.setup_event_ms << ' ' << result.frontier.setup_panel_ms << '\n';
#endif
                    write_result(output, row, kRtol[ti], rep, seed_ms, frontier_ms,
                                 result, seeds.size(), support_proven);
                }
                output.flush();
            }
        } catch (const std::exception& error) {
            std::cerr << "exception case=" << row.case_id << " profile=" << row.profile
                      << " d_bin=" << row.d_bin_index << " epoch=" << row.epoch_index
                      << ": " << error.what() << '\n';
            for (int rep = 0; rep < reps; ++rep) {
                for (double rtol : kRtol) write_failure(output, row, rtol, rep, error.what());
            }
            output.flush();
        }
        ++completed;
        if (completed % 20 == 0 || completed == rows.size()) {
            std::cerr << "progress " << completed << '/' << rows.size() << '\n';
        }
    }
    std::cerr << "finished inputs=" << rows.size() << " reps=" << reps << '\n';
    return 0;
}
