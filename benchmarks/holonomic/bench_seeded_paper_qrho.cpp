#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"

using namespace lcbinint::holonomic;
using Clock = std::chrono::steady_clock;

namespace {

constexpr std::array<double, 2> kTargets{{1e-3, 1e-4}};
constexpr std::size_t kExpectedInputRows = 7216;

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

using Key = std::tuple<int, std::string, int, int, double>;

struct OneTarget {
    std::vector<double> full_ms;
    std::vector<double> radial_ms;
    std::vector<double> planning_ms;
    std::vector<double> integration_ms;
    std::vector<double> abs_error;
    std::vector<double> mu;
    std::vector<std::size_t> nodes;
    std::vector<std::size_t> evaluations;
    std::vector<std::size_t> reused_nodes;
    std::vector<std::size_t> panels;
    std::vector<std::size_t> splits;
    std::vector<std::size_t> radius_probes;
    std::vector<std::size_t> bands;
    std::vector<std::size_t> skipped_seeds;
    std::vector<std::size_t> local_repairs;
    std::vector<std::size_t> retained_nodes;
    std::vector<std::size_t> evaluated_nodes;
    std::vector<double> setup_ms;
    std::vector<double> physics_ms;
    std::vector<double> estimator_ms;
    std::vector<double> scheduler_ms;
    std::vector<double> setup_event_ms;
    std::vector<std::size_t> event_topology_reuses;
    std::vector<std::size_t> event_radius_reuses;
    std::vector<std::size_t> event_qf_refinements;
    std::vector<AdaptiveStop> stops;
    std::vector<AdaptiveStop> value_stops;
    std::vector<Status> numerical_statuses;
    std::vector<bool> value_converged;
    std::string last_frontier_reason = "not_started";
    bool coverage_certified = false;
};

double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double median_size(const std::vector<std::size_t>& values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> converted(values.begin(), values.end());
    return median(std::move(converted));
}

bool parse_row(const std::string& line, InputRow& row) {
    std::istringstream input(line);
    return static_cast<bool>(input >> row.case_id >> row.configuration_id >>
        row.profile >> row.d_bin_index >> row.epoch_index >> row.s >> row.q >>
        row.rho >> row.x >> row.y >> row.time >> row.u >> row.X >> row.reference);
}

bool read_existing(const std::string& path, std::set<Key>& completed) {
    std::ifstream input(path);
    if (!input) return false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        int case_id = 0, configuration_id = 0, d_bin_index = 0, epoch_index = 0;
        std::string profile, status;
        double target = 0.0;
        if (fields >> case_id >> configuration_id >> profile >> d_bin_index >>
            epoch_index >> target) {
            completed.emplace(case_id, profile, d_bin_index, epoch_index, target);
        }
    }
    return true;
}

void write_header(std::ostream& output) {
    output << "case_id\tconfiguration_id\tprofile\td_bin_index\tepoch_index\ttarget"
           << "\ts\tq\trho\tx\ty\ttime\tu\tX\treference_1e-6"
           << "\tstatus\tsupport_proven\tcoverage_certified\treps"
           << "\tvalue_converged_reps\tseed_count\tcenter_seed_count"
           << "\tcaustic_prebuild_ms\tseed_ms\tplanning_ms\tintegration_ms\tms"
           << "\tradius_probes\tbands\tskipped_seeds\tnodes\tevaluations"
           << "\treused_nodes\tpanels\tsplits\tlocal_repairs\tretained_nodes\tevaluated_nodes"
           << "\tsetup_ms\tphysics_ms\testimator_ms\tscheduler_ms\tsetup_event_ms"
           << "\tevent_topology_reuses\tevent_radius_reuses\tevent_qf_refinements"
           << "\tv2_mu\tvalue_error\tvalue_converged\tstop\tvalue_stop"
           << "\tnumerical_status\tfrontier_reason\n";
}

void write_result(std::ostream& output, const InputRow& row, double target,
                  bool support_proven, int support_reps, int seed_count,
                  int center_seed_count, const std::vector<double>& caustic_ms,
                  const std::vector<double>& seed_ms, const OneTarget& measured,
                  int reps) {
    const bool all_value_converged = !measured.value_converged.empty() &&
        std::all_of(measured.value_converged.begin(), measured.value_converged.end(),
                    [](bool value) { return value; });
    const bool supported = support_reps == reps && support_proven;
    const bool accepted = supported && all_value_converged &&
        std::isfinite(median(measured.mu));
    std::string status = "OK";
    if (!supported) {
        status = "SEED_SUPPORT_UNPROVEN";
    } else if (!accepted) {
        std::size_t failed_index = 0;
        while (failed_index < measured.value_converged.size() &&
               measured.value_converged[failed_index]) ++failed_index;
        status = failed_index < measured.value_stops.size()
            ? adaptive_stop_name(measured.value_stops[failed_index]) : "NONFINITE_VALUE";
    }
    const std::string stop = measured.stops.empty() ? "exception"
                                                     : adaptive_stop_name(measured.stops.back());
    const std::string value_stop = measured.value_stops.empty() ? "exception"
        : adaptive_stop_name(measured.value_stops.back());
    const std::string numerical_status = measured.numerical_statuses.empty() ? "exception"
        : to_string(measured.numerical_statuses.back());

    output << row.case_id << '\t' << row.configuration_id << '\t' << row.profile << '\t'
           << row.d_bin_index << '\t' << row.epoch_index << '\t' << target << '\t'
           << row.s << '\t' << row.q << '\t' << row.rho << '\t' << row.x << '\t'
           << row.y << '\t' << row.time << '\t' << row.u << '\t' << row.X << '\t'
           << row.reference << '\t' << status << '\t' << supported << '\t' << false << '\t'
           << reps << '\t'
           << std::count(measured.value_converged.begin(), measured.value_converged.end(), true)
           << '\t' << seed_count << '\t' << center_seed_count << '\t'
           << median(caustic_ms) << '\t' << median(seed_ms) << '\t'
           << median(measured.planning_ms) << '\t' << median(measured.integration_ms) << '\t'
           << median(measured.full_ms) << '\t'
           << median_size(measured.radius_probes) << '\t' << median_size(measured.bands) << '\t'
           << median_size(measured.skipped_seeds) << '\t' << median_size(measured.nodes) << '\t'
           << median_size(measured.evaluations) << '\t' << median_size(measured.reused_nodes) << '\t'
           << median_size(measured.panels) << '\t' << median_size(measured.splits) << '\t'
           << median_size(measured.local_repairs) << '\t'
           << median_size(measured.retained_nodes) << '\t'
           << median_size(measured.evaluated_nodes) << '\t'
           << median(measured.setup_ms) << '\t' << median(measured.physics_ms) << '\t'
           << median(measured.estimator_ms) << '\t' << median(measured.scheduler_ms) << '\t'
           << median(measured.setup_event_ms) << '\t'
           << median_size(measured.event_topology_reuses) << '\t'
           << median_size(measured.event_radius_reuses) << '\t'
           << median_size(measured.event_qf_refinements) << '\t' << median(measured.mu) << '\t'
           << median(measured.abs_error) << '\t' << all_value_converged << '\t'
           << stop << '\t' << value_stop << '\t' << numerical_status << '\t'
           << measured.last_frontier_reason << '\n';
}

void write_failure(std::ostream& output, const InputRow& row, double target,
                   int reps) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    output << row.case_id << '\t' << row.configuration_id << '\t' << row.profile << '\t'
           << row.d_bin_index << '\t' << row.epoch_index << '\t' << target << '\t'
           << row.s << '\t' << row.q << '\t' << row.rho << '\t' << row.x << '\t'
           << row.y << '\t' << row.time << '\t' << row.u << '\t' << row.X << '\t'
           << row.reference << "\tEXCEPTION\t0\t0\t" << reps << "\t0\t0\t0";
    for (int i = 0; i < 5; ++i) output << '\t' << nan;
    for (int i = 0; i < 11; ++i) output << "\t0";
    for (int i = 0; i < 5; ++i) output << '\t' << nan;
    for (int i = 0; i < 3; ++i) output << "\t0";
    output << '\t' << nan << '\t' << nan << "\t0\texception\texception\tEXCEPTION\tEXCEPTION\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "usage: bench_seeded_paper_qrho INPUT OUTPUT [REPS=3] [ROW_LIMIT=0] [RESUME=0]\n";
        return 2;
    }
    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const int reps = argc > 3 ? std::max(1, std::atoi(argv[3])) : 3;
    const std::size_t row_limit = argc > 4
        ? static_cast<std::size_t>(std::max(0, std::atoi(argv[4]))) : 0;
    const bool resume = argc > 5 && std::atoi(argv[5]) != 0;

    std::set<Key> completed;
    if (resume) read_existing(output_path, completed);
    std::ofstream output(output_path, resume ? std::ios::app : std::ios::trunc);
    std::ifstream input(input_path);
    if (!input || !output) {
        std::cerr << "cannot open input or output\n";
        return 2;
    }
    if (!resume || completed.empty()) write_header(output);
    output << std::setprecision(17);

    std::vector<InputRow> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        InputRow row;
        if (!parse_row(line, row)) {
            std::cerr << "malformed input: " << line << '\n';
            return 3;
        }
        rows.push_back(row);
    }
    if (row_limit > 0 && rows.size() > row_limit) rows.resize(row_limit);
    if (row_limit == 0 && rows.size() != kExpectedInputRows) {
        std::cerr << "expected " << kExpectedInputRows << " input rows, got "
                  << rows.size() << '\n';
        return 4;
    }

    std::size_t output_rows = 0;
    std::size_t completed_inputs = 0;
    volatile double sink = 0.0;
    for (const auto& row : rows) {
        bool all_done = true;
        for (double target : kTargets) {
            const Key key{row.case_id, row.profile, row.d_bin_index,
                          row.epoch_index, target};
            if (completed.find(key) == completed.end()) all_done = false;
        }
        if (all_done) {
            ++completed_inputs;
            continue;
        }

        std::array<OneTarget, 2> target_results;
        std::vector<double> caustic_ms;
        std::vector<double> seed_ms;
        int seed_count = 0;
        int center_seed_count = 0;
        int support_reps = 0;
        bool support_proven = true;
        try {
            // The paper corpus stores its source epoch as `time`; its verified
            // computational frame swaps the masses and reflects x, exactly as
            // in the existing V2 runner.
            const LensParams p{row.time, row.y, row.rho, 1.0 / row.q, row.s, true};
            if ((row.profile == "uniform" && std::abs(row.u) > 1e-14) ||
                (row.profile == "linear" && std::abs(row.u - 0.5) > 1e-14)) {
                throw std::runtime_error("profile/u mismatch");
            }
            for (int rep = 0; rep < reps; ++rep) {
                lcbinint::magnification::PointSourceMagnifier point_magnifier;
                lcbinint::magnification::FiniteSourceSettings settings;
                settings.caustic_bins = 1400;
                lcbinint::magnification::FiniteSourceMagnifier finite_magnifier(settings);

                // Seed generation consumes the caustic branches. Build them
                // before starting the clock; scanning them and solving roots
                // for the seeds remains inside the measured interval.
                const auto caustic_start = Clock::now();
                (void)finite_magnifier.binary_caustic_branches(p.a, p.q);
                const auto caustic_end = Clock::now();
                caustic_ms.push_back(std::chrono::duration<double, std::milli>(
                    caustic_end - caustic_start).count());

                const auto seed_start = Clock::now();
                std::vector<lcbinint::SourcePosition> center;
                const auto center_images = point_magnifier.binary_images(
                    p.a, p.q, {p.xs, p.ys});
                center.reserve(center_images.size());
                for (const auto& image : center_images) center.push_back(image.position);
                bool current_proven = false;
                const auto positions = finite_magnifier.cached_binary_image_seeds(
                    point_magnifier, p.a, p.q, {p.xs, p.ys}, p.rho,
                    std::numeric_limits<double>::infinity(), &center, &current_proven);
                std::vector<Cplx<double>> seeds;
                seeds.reserve(positions.size());
                for (const auto& image : positions) {
                    seeds.push_back({image.x + (p.q < 1.0 ? p.a : 0.0), image.y});
                }
                const auto seed_end = Clock::now();
                const double one_seed_ms = std::chrono::duration<double, std::milli>(
                    seed_end - seed_start).count();
                seed_ms.push_back(one_seed_ms);
                seed_count = static_cast<int>(seeds.size());
                center_seed_count = static_cast<int>(center.size());
                support_proven = support_proven && current_proven;
                ++support_reps;

                for (std::size_t ti = 0; ti < kTargets.size(); ++ti) {
                    AdaptiveConfig cfg;
                    cfg.tol.mu_rtol = kTargets[ti];
                    cfg.gradient_policy = GradientPolicy::None;
                    cfg.with_jacobian = false;
                    AdaptiveWorkspace workspace;
                    const auto radial_start = Clock::now();
                    const auto result = epoch_seeded_radial_frontier(
                        p, row.u, seeds, cfg, workspace, false);
                    const auto radial_end = Clock::now();
                    const double radial = std::chrono::duration<double, std::milli>(
                        radial_end - radial_start).count();
                    auto& target = target_results[ti];
                    target.full_ms.push_back(one_seed_ms + radial);
                    target.radial_ms.push_back(radial);
                    target.planning_ms.push_back(result.frontier.planning_ms);
                    target.integration_ms.push_back(result.frontier.integration_ms);
                    target.abs_error.push_back(result.integral.estimated_abs_error_mu);
                    target.mu.push_back(result.integral.mu);
                    target.nodes.push_back(result.integral.stats.unique_nodes);
                    target.evaluations.push_back(result.integral.stats.node_evaluations);
                    target.reused_nodes.push_back(result.integral.stats.reused_nodes);
                    target.panels.push_back(result.integral.stats.panels);
                    target.splits.push_back(result.integral.stats.splits);
                    target.radius_probes.push_back(result.frontier.radius_probes);
                    target.bands.push_back(result.frontier.bands);
                    target.skipped_seeds.push_back(result.frontier.skipped_seeds);
                    target.local_repairs.push_back(result.frontier.local_repairs);
                    target.retained_nodes.push_back(result.frontier.retained_nodes);
                    target.evaluated_nodes.push_back(result.frontier.evaluated_nodes);
                    target.setup_ms.push_back(result.integral.stats.setup_ms);
                    target.physics_ms.push_back(result.integral.stats.physical_ms);
                    target.estimator_ms.push_back(result.integral.stats.estimator_ms);
                    target.scheduler_ms.push_back(result.integral.stats.scheduler_ms);
                    target.setup_event_ms.push_back(result.integral.stats.setup_event_ms);
                    target.event_topology_reuses.push_back(
                        result.integral.stats.event_topology_reuses);
                    target.event_radius_reuses.push_back(
                        result.integral.stats.event_radius_reuses);
                    target.event_qf_refinements.push_back(
                        result.integral.stats.event_qf_refinements);
                    target.stops.push_back(result.integral.stop);
                    target.value_stops.push_back(result.integral.value_stop_reason);
                    target.numerical_statuses.push_back(result.integral.numerical_status);
                    target.value_converged.push_back(result.integral.value_converged);
                    target.last_frontier_reason = result.frontier.reason;
                    target.coverage_certified = result.frontier.coverage_certified;
                    sink += result.integral.mu;
                }
                sink += static_cast<double>(positions.size() + center.size());
            }

            for (std::size_t ti = 0; ti < kTargets.size(); ++ti) {
                const Key key{row.case_id, row.profile, row.d_bin_index,
                              row.epoch_index, kTargets[ti]};
                if (completed.find(key) != completed.end()) continue;
                write_result(output, row, kTargets[ti], support_proven, support_reps,
                             seed_count, center_seed_count, caustic_ms, seed_ms,
                             target_results[ti], reps);
                completed.insert(key);
                ++output_rows;
            }
            output.flush();
        } catch (const std::exception& error) {
            std::cerr << "exception case=" << row.case_id << " profile=" << row.profile
                      << " d_bin=" << row.d_bin_index << " epoch=" << row.epoch_index
                      << ": " << error.what() << '\n';
            for (double target : kTargets) {
                const Key key{row.case_id, row.profile, row.d_bin_index,
                              row.epoch_index, target};
                if (completed.find(key) != completed.end()) continue;
                write_failure(output, row, target, reps);
                completed.insert(key);
                ++output_rows;
            }
            output.flush();
        }
        ++completed_inputs;
        if ((completed_inputs % 8) == 0 || completed_inputs == rows.size()) {
            std::cerr << "progress " << completed_inputs << '/' << rows.size()
                      << " output_rows=" << output_rows << "\n";
        }
    }
    std::cerr << "finished input_rows=" << rows.size() << " output_rows="
              << output_rows << " reps=" << reps << " sink=" << sink << '\n';
    return output_rows == 0 && completed.empty() ? 2 : 0;
}
