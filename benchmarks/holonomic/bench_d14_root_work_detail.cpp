// Opt-in root-wise D14 work and exclusive-stage diagnostic.
// Input columns match the V2 adaptive trajectory snapshot.  This benchmark
// records topology/D14 only; its wall time is diagnostic and is not used as
// the whole-epoch performance result.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/v2_profile.hpp"

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

static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static const char* lane_name(bool warm) { return warm ? "warm" : "cold"; }

static void write_timing_header(std::ofstream& out) {
    out << "case_id configuration_id profile d_bin_index epoch_index rep lane "
           "q rho s x y time u X reference topology_status cells events "
           "whole_classify_ms profile_topology_ms radial_events_ms "
           "cell_topology_remainder_ms physical_real_events soft_events "
           "d14_coeff_ms d14_struct_build_ms "
           "d14_expand_ms d14_solve_ms presearch_ms d14real_ms "
           "qf_polish_ms residual_eval_ms completeness_check_ms "
           "physical_classify_ms soft_event_ms qf_warm_calls qf_cold_calls "
           "warm_seeded presearch_sweeps dd_sweeps qf_warm_sweeps "
           "presearch_active_fallbacks presearch_nonfinite_failures "
           "presearch_failure_code presearch_failure_sweep "
           "presearch_failure_root presearch_last_finite_handoffs "
           "qf_cold_sweeps d14real_calls d14real_finite_calls "
           "d14real_nonconverged completeness_fails root_count_bad "
           "conjugacy_bad vieta_bad root_clusters captured_root_records "
           "qf_root_count\n";
}

static void write_root_header(std::ofstream& out) {
    out << "case_id configuration_id profile d_bin_index epoch_index rep lane "
           "root_index source_root_index role role_hint physical_real "
           "v_re v_im final_v_re final_v_im d14real_updates cluster_id "
           "double_seed_index double_seed_v_re double_seed_v_im "
           "double_seed_valid expanded_seed_source_index "
           "expanded_seed_displacement expanded_seed_match_valid "
           "cluster_size freeze_count reactivation_count d14real_newton "
           "d14real_nearest_sep d14real_rel_correction d14real_position_error "
           "qf_escalated qf_source_match_valid qf_displacement "
           "qf_displacement_valid qf_newton qf_nearest_sep qf_position_error "
           "qf_expanded_relative_residual\n";
}

static void write_root(std::ofstream& out, const InputRow& row, int rep,
                       bool warm, const D14RootWorkRecord& r) {
    out << row.case_id << ' ' << row.configuration_id << ' ' << row.profile << ' '
        << row.d_bin_index << ' ' << row.epoch_index << ' ' << rep << ' '
        << lane_name(warm) << ' ' << r.root_index << ' ' << r.source_root_index
        << ' ' << r.role << ' ' << r.role_hint << ' ' << r.physical_real << ' '
        << r.v_re << ' ' << r.v_im << ' ' << r.final_v_re << ' ' << r.final_v_im
        << ' ' << r.d14real_updates << ' ' << r.cluster_id << ' '
        << r.double_seed_index << ' ' << r.double_seed_v_re << ' '
        << r.double_seed_v_im << ' ' << int(r.double_seed_valid) << ' '
        << r.expanded_seed_source_index << ' ' << r.expanded_seed_displacement
        << ' ' << int(r.expanded_seed_match_valid) << ' ' << r.cluster_size
        << ' ' << r.freeze_count << ' ' << r.reactivation_count << ' '
        << r.d14real_newton_correction << ' ' << r.d14real_nearest_separation
        << ' ' << r.d14real_relative_correction << ' ' << r.d14real_position_error
        << ' ' << int(r.qf_escalated) << ' ' << int(r.qf_source_match_valid)
        << ' ' << r.qf_displacement << ' ' << int(r.qf_displacement_valid)
        << ' ' << r.qf_newton_correction << ' ' << r.qf_nearest_separation
        << ' ' << r.qf_position_error << ' '
        << r.qf_expanded_relative_residual << '\n';
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: bench_d14_root_work_detail INPUT ROOTS.tsv TIMINGS.tsv "
                     "[reps=1] [cold|warm|both=both] [case=-1] [d_bin=-1] "
                     "[profile=all|linear|uniform]\n";
        return 2;
    }
    const int reps = argc > 4 ? std::max(1, std::atoi(argv[4])) : 1;
    const std::string mode = argc > 5 ? argv[5] : "both";
    const int case_filter = argc > 6 ? std::atoi(argv[6]) : -1;
    const int d_bin_filter = argc > 7 ? std::atoi(argv[7]) : -1;
    const std::string profile_filter = argc > 8 ? argv[8] : "all";
    const bool do_cold = mode == "cold" || mode == "both";
    const bool do_warm = mode == "warm" || mode == "both";
    if (!do_cold && !do_warm) {
        std::cerr << "lane must be cold, warm, or both\n";
        return 2;
    }

    std::ifstream input(argv[1]);
    std::ofstream roots_out(argv[2]);
    std::ofstream times_out(argv[3]);
    if (!input || !roots_out || !times_out) {
        std::cerr << "failed to open input/output\n";
        return 2;
    }
    roots_out << std::setprecision(17);
    times_out << std::setprecision(17);
    write_root_header(roots_out);
    write_timing_header(times_out);

    std::vector<InputRow> rows;
    std::string line;
    while (std::getline(input, line)) {
        InputRow row;
        if (!read_row(line, row)) continue;
        if (case_filter >= 0 && row.case_id != case_filter) continue;
        if (d_bin_filter >= 0 && row.d_bin_index != d_bin_filter) continue;
        if (profile_filter != "all" && row.profile != profile_filter) continue;
        rows.push_back(std::move(row));
    }
    std::stable_sort(rows.begin(), rows.end(), [](const InputRow& a,
                                                   const InputRow& b) {
        if (a.case_id != b.case_id) return a.case_id < b.case_id;
        if (a.configuration_id != b.configuration_id)
            return a.configuration_id < b.configuration_id;
        if (a.profile != b.profile) return a.profile < b.profile;
        if (a.d_bin_index != b.d_bin_index) return a.d_bin_index < b.d_bin_index;
        return a.epoch_index < b.epoch_index;
    });
    if (rows.empty()) {
        std::cerr << "no matching input rows\n";
        return 3;
    }

    D14EventPolicyScope policy(D14EventPolicy::AllComplexSoft);
    volatile double sink = 0.0;
    for (bool warm_lane : {false, true}) {
        if ((warm_lane && !do_warm) || (!warm_lane && !do_cold)) continue;
        for (int rep = 0; rep < reps; ++rep) {
            for (std::size_t begin = 0; begin < rows.size();) {
                std::size_t end = begin + 1;
                while (end < rows.size() && same_trajectory(rows[begin], rows[end]))
                    ++end;
                std::vector<Cplx<__float128>> previous_roots;
                for (std::size_t k = begin; k < end; ++k) {
                    const InputRow& row = rows[k];
                    const LensParams params{row.time, row.y, row.rho,
                                            1.0 / row.q, row.s, true};
                    const PrimaryFrame pf = PrimaryFrame::from(params);
                    V2Profile profile;
                    profile.capture_d14_root_work = true;
                    TopologyResult topology;
                    std::vector<Cplx<__float128>> current_roots;
                    double wall_ms = 0.0;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
                    if (std::getenv("HOLO_D14_QF_TRACE") &&
                        std::getenv("HOLO_D14_QF_TRACE")[0] == '1')
                        std::fprintf(stderr,
                            "D14TRACE_ROW\tcase=%d\tconfig=%d\tprofile=%s"
                            "\td_bin=%d\tepoch=%d\trep=%d\tlane=%s\n",
                            row.case_id, row.configuration_id, row.profile.c_str(),
                            row.d_bin_index, row.epoch_index, rep,
                            lane_name(warm_lane));
#endif
                    {
                        V2ProfileScope scope(profile);
                        const auto start = Clock::now();
                        topology = classify_cells(
                            pf,
                            warm_lane && !previous_roots.empty()
                                ? &previous_roots : nullptr,
                            &current_roots,
                            /*retain_adaptive_metadata=*/true);
                        wall_ms = elapsed_ms(start);
                    }
                    sink += static_cast<double>(topology.cells.size()) +
                            static_cast<double>(current_roots.size());
                    std::size_t physical = 0, soft = 0;
                    for (const auto& event : topology.events) {
                        if (event.kind == "physical_real") ++physical;
                        if (event.kind == "physical_complex") ++soft;
                    }
                    const double cell_remainder = std::max(
                        0.0, profile.topology_ms - profile.radial_events_ms);
                    times_out << row.case_id << ' ' << row.configuration_id << ' '
                        << row.profile << ' ' << row.d_bin_index << ' '
                        << row.epoch_index << ' ' << rep << ' ' << lane_name(warm_lane)
                        << ' ' << row.q << ' ' << row.rho << ' ' << row.s << ' '
                        << row.x << ' ' << row.y << ' ' << row.time << ' ' << row.u
                        << ' ' << row.X << ' ' << row.reference << ' '
                        << static_cast<int>(topology.status) << ' '
                        << topology.cells.size() << ' ' << topology.events.size()
                        << ' ' << wall_ms << ' ' << profile.topology_ms << ' '
                        << profile.radial_events_ms << ' ' << cell_remainder << ' '
                        << physical << ' ' << soft << ' '
                        << profile.d14_coeff_ms << ' '
                        << profile.d14_struct_build_ms << ' '
                        << profile.d14_expand_ms << ' ' << profile.d14_solve_ms
                        << ' ' << profile.d14_presearch_ms << ' '
                        << profile.d14_real_ms << ' ' << profile.d14_qf_polish_ms
                        << ' ' << profile.d14_residual_eval_ms << ' '
                        << profile.d14_completeness_check_ms << ' '
                        << profile.d14_event_classify_ms << ' '
                        << profile.d14_soft_event_ms << ' '
                        << profile.d14_qf_warm_calls << ' '
                        << profile.d14_qf_cold_calls << ' '
                        << profile.d14_warm_seeded << ' '
                        << profile.d14_presearch_sweeps << ' '
                        << profile.d14_dd_sweeps << ' '
                        << profile.d14_qf_warm_sweeps << ' '
                        << profile.d14_presearch_active_fallbacks << ' '
                        << profile.d14_presearch_nonfinite_failures << ' '
                        << profile.d14_presearch_failure_code << ' '
                        << profile.d14_presearch_failure_iteration << ' '
                        << profile.d14_presearch_failure_root << ' '
                        << profile.d14_presearch_last_finite_handoffs << ' '
                        << profile.d14_qf_cold_sweeps << ' '
                        << profile.d14_real_calls << ' '
                        << profile.d14_real_finite_calls << ' '
                        << profile.d14_real_nonconverged << ' '
                        << profile.d14_completeness_fails << ' '
                        << profile.d14_root_count_bad << ' '
                        << profile.d14_conjugacy_bad << ' '
                        << profile.d14_vieta_bad << ' '
                        << profile.d14_root_clusters << ' '
                        << profile.d14_root_work.size() << ' '
                        << current_roots.size() << '\n';
                    for (const auto& record : profile.d14_root_work)
                        write_root(roots_out, row, rep, warm_lane, record);
                    if (warm_lane) previous_roots = std::move(current_roots);
                }
                begin = end;
            }
        }
    }
    if (sink == -1.0) std::cerr << sink << '\n';
    return 0;
}
