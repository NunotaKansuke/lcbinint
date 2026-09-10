// Phase 7 shared-geometry A/B.
//
// One V2 classify_cells/D14 + cell/radial/root/F0 plan is built once for each
// lane and passed to both evaluators.  V2 evaluates J_half with the existing
// 16-node K-rule; PF6/fold evaluates the same node/root-pair plan with the
// equation-derived packet.  The PF6 evaluator does not call classify_cells,
// arc_intervals, or endpoint search.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/phase7_shared_harness.hpp"

using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::phase7_detail;

namespace {

struct Case {
    LensParams p;
    double u = 0.0;
    std::string name;
};

struct AngularReference {
    bool ok = true;
    double mu = 0.0;
};

std::vector<Case> load_cases(const char* path, int limit) {
    std::ifstream in(path);
    std::vector<Case> out;
    std::string line;
    while (std::getline(in, line) && (limit <= 0 || (int)out.size() < limit)) {
        std::istringstream ls(line);
        double xs, ys, rho, q, a, ignored;
        int bary;
        Case c;
        if (!(ls >> xs >> ys >> rho >> q >> a >> bary >> c.u >> ignored >>
              c.name))
            continue;
        c.p = LensParams{xs, ys, rho, q, a, bary != 0};
        out.push_back(c);
    }
    return out;
}

double relative_error(double a, double b) {
    return std::fabs(a - b) /
           (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

AngularReference angular_reference(const Case& c, int n_r, int ntheta) {
    AngularReference out;
    const PrimaryFrame pf = PrimaryFrame::from(c.p);
    const TopologyResult topo = classify_cells(pf);
    const Cheb1Dyn rr(n_r);
    constexpr double pi = 3.1415926535897932384626433832795;
    double F0 = 0.0;
    double Fh = 0.0;
    for (const auto& cell : topo.cells) {
        const double width = cell.r_hi - cell.r_lo;
        if (!(width > 0.0) || cell.kind == ArcKind::kEmpty) continue;
        const double inset = 1e-9 * width;
        const double lo = cell.r_lo + inset;
        const double hi = cell.r_hi - inset;
        const double center = 0.5 * (lo + hi);
        const double half = 0.5 * (hi - lo);
        for (int n = 0; n < n_r; ++n) {
            const double R = center + half * rr.x[n];
            const double wk = half * rr.w[n];
            if (cell.kind == ArcKind::kFull) {
                F0 += wk * R * 2.0 * pi;
                if (c.u != 0.0) {
                    double sum = 0.0;
                    for (int k = 0; k < ntheta; ++k) {
                        const double th = 2.0 * pi * (k + 0.5) / ntheta;
                        const double ph = phi_lens(R, th, pf);
                        if (!(ph > 0.0)) {
                            out.ok = false;
                            continue;
                        }
                        sum += std::sqrt(ph);
                    }
                    Fh += wk * R * 2.0 * pi * sum / ntheta;
                }
                continue;
            }
            if (cell.kind != ArcKind::kArcs) {
                out.ok = false;
                continue;
            }
            const ArcSet as = arc_intervals(R, pf);
            if (as.kind != ArcKind::kArcs) {
                out.ok = false;
                continue;
            }
            for (const auto& arc : as.arcs) {
                const PolishResult pe = polish_endpoint(R, arc[0], pf);
                const PolishResult pl = polish_endpoint(R, arc[1], pf);
                double end = pl.theta;
                if (end <= pe.theta) end += 2.0 * pi;
                F0 += wk * R * (end - pe.theta);
                if (c.u == 0.0) continue;
                double sum = 0.0;
                const double ah = 0.5 * (end - pe.theta);
                const double am = 0.5 * (end + pe.theta);
                for (int k = 1; k <= ntheta; ++k) {
                    const double x = std::cos(pi * k / (ntheta + 1.0));
                    const double s2 = std::max(0.0, 1.0 - x * x);
                    const double ph = phi_lens(R, am + ah * x, pf);
                    if (!(ph > 0.0) || !(s2 > 0.0)) {
                        out.ok = false;
                        continue;
                    }
                    sum += (pi / (ntheta + 1.0)) * s2 *
                           std::sqrt(ph / s2);
                }
                Fh += wk * R * ah * sum;
            }
        }
    }
    const double denom = pi * c.p.rho * c.p.rho * (1.0 - c.u / 3.0);
    out.mu = ((1.0 - c.u) * F0 + c.u * Fh) / denom;
    return out;
}

struct Lane {
    SharedPlanCost plan_cost{};
    SharedEpochEvaluation v2{};
    SharedEpochEvaluation pf6{};
    double v2_whole_ms = 0.0;
    double pf6_whole_ms = 0.0;
};

struct Row {
    Case c;
    Lane value_cold, value_warm;
    Lane jac_cold, jac_warm;
    AngularReference reference{};
    bool have_jac = false;
};

template <int Order>
Lane run_lane(const Case& c, int n_r, bool with_jacobian,
             const std::vector<Cplx<__float128>>* d14_warm) {
    Lane lane;
    const SharedEpochPlan plan = build_shared_plan(
        c.p, c.u, n_r, with_jacobian, d14_warm);
    lane.plan_cost = plan.cost;
    lane.v2 = evaluate_v2_k_rule(plan, with_jacobian);
    lane.pf6 = evaluate_pf6_fold<Order>(plan, with_jacobian);
    lane.v2_whole_ms = plan.cost.total_ms + lane.v2.cost.evaluation_ms;
    lane.pf6_whole_ms = plan.cost.total_ms + lane.pf6.cost.evaluation_ms;
    return lane;
}

template <class T>
double median(std::vector<T> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return (double)values[values.size() / 2];
}

void print_lane(const char* label, const std::vector<Row>& rows,
                const Lane Row::*member) {
    std::vector<double> topo, root, f0, whole_v2, whole_pf6;
    std::vector<double> v2_ld, pf6_ld, pf6_seed, pf6_packet, pf6_eval;
    std::vector<double> fold_build, fold_eval;
    long long k_ok = 0, k_fail = 0, k_fallback = 0;
    long long pf6_fail = 0, pf6_transport = 0;
    long long fold_builds = 0, fold_accept = 0, fold_reject = 0;
    long long pf6_builds = 0, pf6_accept = 0, pf6_reject = 0;
    long long total_arcs = 0, total_nodes = 0;
    int v2_true = 0, pf6_true = 0, count = 0;
    for (const auto& row : rows) {
        if (row.c.u == 0.0) continue;
        const Lane& lane = row.*member;
        ++count;
        topo.push_back(lane.plan_cost.topology_ms);
        root.push_back(lane.plan_cost.root_tracking_ms);
        f0.push_back(lane.plan_cost.f0_endpoint_ms);
        whole_v2.push_back(lane.v2_whole_ms);
        whole_pf6.push_back(lane.pf6_whole_ms);
        v2_ld.push_back(lane.v2.cost.evaluation_ms);
        pf6_ld.push_back(lane.pf6.cost.evaluation_ms);
        pf6_seed.push_back(lane.pf6.cost.physical_seed_ms);
        pf6_packet.push_back(lane.pf6.cost.pf6_packet_ms);
        pf6_eval.push_back(lane.pf6.cost.pf6_evaluation_ms);
        fold_build.push_back(lane.pf6.cost.fold_build_ms);
        fold_eval.push_back(lane.pf6.cost.fold_evaluation_ms);
        k_ok += lane.v2.cost.k_rule_success;
        k_fail += lane.v2.cost.k_rule_failed;
        k_fallback += lane.v2.cost.incumbent_angular_fallback;
        pf6_fail += lane.pf6.cost.failed;
        pf6_transport += lane.pf6.cost.transported;
        fold_builds += lane.pf6.cost.fold_packet_constructions;
        fold_accept += lane.pf6.cost.fold_packet_accepted;
        fold_reject += lane.pf6.cost.fold_packet_rejected;
        pf6_builds += lane.pf6.cost.pf6_packet_constructions;
        pf6_accept += lane.pf6.cost.pf6_packet_accepted;
        pf6_reject += lane.pf6.cost.pf6_packet_rejected;
        total_arcs += lane.pf6.cost.arc_nodes;
        total_nodes += lane.pf6.cost.radial_nodes;
        if (lane.v2.all_true) ++v2_true;
        if (lane.pf6.all_true) ++pf6_true;
    }
    std::printf(
        "PHASE7_MEDIAN %s cases=%d shared_topology_D14=%.6f "
        "shared_root_tracking=%.6f shared_F0_endpoint=%.6f "
        "shared_total=%.6f "
        "V2_K_LD=%.6f PF6_fold_LD=%.6f PF6_seed=%.6f "
        "PF6_packet=%.6f PF6_eval=%.6f fold_build=%.6f fold_eval=%.6f "
        "whole_V2=%.6f whole_PF6=%.6f\n",
        label, count, median(topo), median(root), median(f0),
        median([&] {
            std::vector<double> x;
            for (const auto& row : rows)
                if (row.c.u != 0.0) x.push_back((row.*member).plan_cost.total_ms);
            return x;
        }()),
        median(v2_ld), median(pf6_ld), median(pf6_seed), median(pf6_packet),
        median(pf6_eval), median(fold_build), median(fold_eval),
        median(whole_v2), median(whole_pf6));
    std::printf(
        "PHASE7_COUNTS %s V2_k_success=%lld V2_k_failed=%lld "
        "V2_angular_fallback=%lld V2_all_true=%d/%d "
        "PF6_transported=%lld PF6_failed=%lld PF6_all_true=%d/%d "
        "fold_construct=%lld fold_accept=%lld fold_reject=%lld "
        "PF6_construct=%lld PF6_accept=%lld PF6_reject=%lld "
        "arc_nodes=%lld radial_nodes=%lld\n",
        label, k_ok, k_fail, k_fallback, v2_true, count, pf6_transport,
        pf6_fail, pf6_true, count, fold_builds, fold_accept, fold_reject,
        pf6_builds, pf6_accept, pf6_reject, total_arcs, total_nodes);
}

template <int Order>
int run(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
                               : "evidence/holonomic/gm_coverage_cases.tsv";
    const int n_r = argc > 2 ? std::max(4, std::atoi(argv[2])) : 64;
    const int limit = argc > 3 ? std::atoi(argv[3]) : 0;
    const int ref_limit = argc > 4 ? std::max(0, std::atoi(argv[4])) : 12;
    const int ref_ntheta = argc > 6 ? std::max(64, std::atoi(argv[6])) : 2048;
    const auto cases = load_cases(path, limit);
    if (cases.empty()) {
        std::fprintf(stderr, "no cases in %s\n", path);
        return 2;
    }

    // The incumbent benchmark uses the normal V2 root path and does not allow
    // the production ODE/router to alter the shared plan.
    holo_holonomic_transport_override() = 0;
    holo_mv_transport_override() = 0;

    std::vector<Row> rows;
    rows.reserve(cases.size());
    double worst_value = 0.0, worst_jac = 0.0, worst_grad = 0.0;
    double worst_ref_v2 = 0.0, worst_ref_pf6 = 0.0;
    std::string worst_ref_v2_name, worst_ref_pf6_name;
    int ref_count = 0, ref_ok = 0, ref_comparable = 0;
    for (size_t index = 0; index < cases.size(); ++index) {
        const Case& c = cases[index];
        Row row;
        row.c = c;
        const SharedEpochPlan value_cold_plan = build_shared_plan(
            c.p, c.u, n_r, false, nullptr);
        row.value_cold.plan_cost = value_cold_plan.cost;
        row.value_cold.v2 = evaluate_v2_k_rule(value_cold_plan, false);
        row.value_cold.pf6 = evaluate_pf6_fold<Order>(value_cold_plan, false);
        row.value_cold.v2_whole_ms = value_cold_plan.cost.total_ms +
                                     row.value_cold.v2.cost.evaluation_ms;
        row.value_cold.pf6_whole_ms = value_cold_plan.cost.total_ms +
                                      row.value_cold.pf6.cost.evaluation_ms;
        const SharedEpochPlan value_warm_plan = build_shared_plan(
            c.p, c.u, n_r, false, &value_cold_plan.d14_roots);
        row.value_warm.plan_cost = value_warm_plan.cost;
        row.value_warm.v2 = evaluate_v2_k_rule(value_warm_plan, false);
        row.value_warm.pf6 = evaluate_pf6_fold<Order>(value_warm_plan, false);
        row.value_warm.v2_whole_ms = value_warm_plan.cost.total_ms +
                                     row.value_warm.v2.cost.evaluation_ms;
        row.value_warm.pf6_whole_ms = value_warm_plan.cost.total_ms +
                                      row.value_warm.pf6.cost.evaluation_ms;

        if (c.u != 0.0) {
            row.have_jac = true;
            const SharedEpochPlan jac_cold_plan = build_shared_plan(
                c.p, c.u, n_r, true, nullptr);
            row.jac_cold.plan_cost = jac_cold_plan.cost;
            row.jac_cold.v2 = evaluate_v2_k_rule(jac_cold_plan, true);
            row.jac_cold.pf6 = evaluate_pf6_fold<Order>(jac_cold_plan, true);
            row.jac_cold.v2_whole_ms = jac_cold_plan.cost.total_ms +
                                       row.jac_cold.v2.cost.evaluation_ms;
            row.jac_cold.pf6_whole_ms = jac_cold_plan.cost.total_ms +
                                        row.jac_cold.pf6.cost.evaluation_ms;
            const SharedEpochPlan jac_warm_plan = build_shared_plan(
                c.p, c.u, n_r, true, &jac_cold_plan.d14_roots);
            row.jac_warm.plan_cost = jac_warm_plan.cost;
            row.jac_warm.v2 = evaluate_v2_k_rule(jac_warm_plan, true);
            row.jac_warm.pf6 = evaluate_pf6_fold<Order>(jac_warm_plan, true);
            row.jac_warm.v2_whole_ms = jac_warm_plan.cost.total_ms +
                                       row.jac_warm.v2.cost.evaluation_ms;
            row.jac_warm.pf6_whole_ms = jac_warm_plan.cost.total_ms +
                                        row.jac_warm.pf6.cost.evaluation_ms;
        }

        if ((int)index < ref_limit && c.u != 0.0) {
            row.reference = angular_reference(c, n_r, ref_ntheta);
            ++ref_count;
            if (row.reference.ok) ++ref_ok;
            // An unresolved topology is deliberately retained as a status
            // result, but its angular number is not a meaningful accuracy
            // comparison for either evaluator.  Keep it in ref_count and
            // report the clean subset separately.
            if (row.value_cold.v2.status == Status::OK &&
                row.value_cold.pf6.status == Status::OK) {
                ++ref_comparable;
                const double ref_v2 = relative_error(
                    row.value_cold.v2.mu, row.reference.mu);
                const double ref_pf6 = relative_error(
                    row.value_cold.pf6.mu, row.reference.mu);
                if (ref_v2 > worst_ref_v2) {
                    worst_ref_v2 = ref_v2;
                    worst_ref_v2_name = c.name;
                }
                if (ref_pf6 > worst_ref_pf6) {
                    worst_ref_pf6 = ref_pf6;
                    worst_ref_pf6_name = c.name;
                }
            }
        }
        worst_value = std::max(
            worst_value,
            relative_error(row.value_cold.pf6.mu, row.value_cold.v2.mu));
        if (c.u != 0.0) {
            worst_jac = std::max(
                worst_jac,
                relative_error(row.jac_cold.pf6.mu, row.jac_cold.v2.mu));
            for (int j = 0; j < 5; ++j)
                worst_grad = std::max(
                    worst_grad,
                    relative_error(row.jac_cold.pf6.grad_mu[j],
                                   row.jac_cold.v2.grad_mu[j]));
        }
        std::printf(
            "PHASE7_CASE name=%s u=%.1f value_cold_whole=%.6f/%.6f "
            "value_warm_whole=%.6f/%.6f value_ld=%.6f/%.6f "
            "status=%s/%s true=%d/%d",
            c.name.c_str(), c.u, row.value_cold.v2_whole_ms,
            row.value_cold.pf6_whole_ms, row.value_warm.v2_whole_ms,
            row.value_warm.pf6_whole_ms, row.value_cold.v2.cost.evaluation_ms,
            row.value_cold.pf6.cost.evaluation_ms,
            to_string(row.value_cold.v2.status),
            to_string(row.value_cold.pf6.status),
            (int)row.value_cold.v2.all_true,
            (int)row.value_cold.pf6.all_true);
        if (row.have_jac)
            std::printf(
                " jac_cold_whole=%.6f/%.6f jac_warm_whole=%.6f/%.6f "
                "jac_ld=%.6f/%.6f jac_status=%s/%s true=%d/%d",
                row.jac_cold.v2_whole_ms, row.jac_cold.pf6_whole_ms,
                row.jac_warm.v2_whole_ms, row.jac_warm.pf6_whole_ms,
                row.jac_cold.v2.cost.evaluation_ms,
                row.jac_cold.pf6.cost.evaluation_ms,
                to_string(row.jac_cold.v2.status),
                to_string(row.jac_cold.pf6.status),
                (int)row.jac_cold.v2.all_true,
                (int)row.jac_cold.pf6.all_true);
        std::printf("\n");
        rows.push_back(std::move(row));
    }

    std::printf(
        "PHASE7_RESULT order=%d cases=%zu nr=%d limb=%zu "
        "same_plan=1 reference_ntheta=%d production_router=unchanged\n",
        Order, rows.size(), n_r,
        std::count_if(rows.begin(), rows.end(),
                      [](const Row& r) { return r.c.u != 0.0; }), ref_ntheta);
    print_lane("value_cold", rows, &Row::value_cold);
    print_lane("value_warm", rows, &Row::value_warm);
    print_lane("jac_cold", rows, &Row::jac_cold);
    print_lane("jac_warm", rows, &Row::jac_warm);
    std::printf(
        "PHASE7_ACCURACY worst_value_pf6_vs_v2=%.3e "
        "worst_jac_mu_pf6_vs_v2=%.3e worst_jac_grad_pf6_vs_v2=%.3e "
        "angular_reference=%d/%d clean_comparable=%d worst_v2=%.3e(%s) "
        "worst_pf6=%.3e(%s)\n",
        worst_value, worst_jac, worst_grad, ref_ok, ref_count, ref_comparable,
        worst_ref_v2, worst_ref_v2_name.c_str(), worst_ref_pf6,
        worst_ref_pf6_name.c_str());

    std::vector<double> value_jac_v2, value_jac_pf6, whole_jac_v2,
        whole_jac_pf6;
    for (const auto& row : rows) {
        if (!row.have_jac) continue;
        value_jac_v2.push_back(row.jac_cold.v2.cost.evaluation_ms -
                               row.value_cold.v2.cost.evaluation_ms);
        value_jac_pf6.push_back(row.jac_cold.pf6.cost.evaluation_ms -
                                row.value_cold.pf6.cost.evaluation_ms);
        whole_jac_v2.push_back(row.jac_cold.v2_whole_ms -
                               row.value_cold.v2_whole_ms);
        whole_jac_pf6.push_back(row.jac_cold.pf6_whole_ms -
                                row.value_cold.pf6_whole_ms);
    }
    std::printf(
        "PHASE7_JAC_MARGINAL cold_eval_V2=%.6f cold_eval_PF6=%.6f "
        "cold_whole_V2=%.6f cold_whole_PF6=%.6f ms\n",
        median(value_jac_v2), median(value_jac_pf6), median(whole_jac_v2),
        median(whole_jac_pf6));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const int order = argc > 5 ? std::atoi(argv[5]) : 12;
    switch (order) {
        case 8: return run<8>(argc, argv);
        case 10: return run<10>(argc, argv);
        case 12: return run<12>(argc, argv);
        case 14: return run<14>(argc, argv);
        case 16: return run<16>(argc, argv);
        case 20: return run<20>(argc, argv);
        default:
            std::fprintf(stderr, "order must be one of 8,10,12,14,16,20\n");
            return 2;
    }
}
