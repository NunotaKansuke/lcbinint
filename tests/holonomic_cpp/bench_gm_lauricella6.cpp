// Whole-epoch benchmark for the physical FD6 -> FD5 six-state kernel.
//
// The PF6 lane includes topology, one physical Euler seed per arc, algebraic
// endpoint geometry, adaptive Taylor connection packets, dense output,
// fail-closed gates, and (optionally) the five-parameter chain rule.  The
// reference lane is the independent angular physical integral.  V2 is timed
// separately in the same process and the production router is never changed.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/gm_lauricella6.hpp"
#include "lcbinint/magnification/holonomic/gm_physical_transport.hpp"

using namespace lcbinint::holonomic;
using Clock = std::chrono::steady_clock;

namespace {

struct Case {
    LensParams p;
    double u = 0.0;
    std::string name;
};

struct AngularEpoch {
    bool ok = true;
    double F0 = 0.0;
    double F_half = 0.0;
    double mu = 0.0;
};

double rel(double a, double b) {
    return std::fabs(a - b) /
           (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

std::vector<Case> load(const char* path, int limit) {
    std::ifstream in(path);
    std::vector<Case> out;
    std::string line;
    while (std::getline(in, line) && (limit <= 0 || (int)out.size() < limit)) {
        std::istringstream ls(line);
        double xs, ys, rho, q, a, ignored;
        int bary;
        Case c;
        if (!(ls >> xs >> ys >> rho >> q >> a >> bary >> c.u >> ignored >> c.name))
            continue;
        c.p = LensParams{xs, ys, rho, q, a, bary != 0};
        out.push_back(c);
    }
    return out;
}

AngularEpoch angular_epoch(const LensParams& p, double u, int nr, int ntheta) {
    AngularEpoch out;
    const PrimaryFrame pf = PrimaryFrame::from(p);
    const TopologyResult topo = classify_cells(pf);
    const Cheb1Dyn rr(nr);
    constexpr double pi = 3.1415926535897932384626433832795;
    for (const auto& cell : topo.cells) {
        const double width = cell.r_hi - cell.r_lo;
        if (!(width > 0.0) || cell.kind == ArcKind::kEmpty) continue;
        const double inset = 1e-9 * width;
        const double lo = cell.r_lo + inset;
        const double hi = cell.r_hi - inset;
        const double center = 0.5 * (lo + hi);
        const double half = 0.5 * (hi - lo);
        for (int n = 0; n < nr; ++n) {
            const double R = center + half * rr.x[n];
            const double wk = half * rr.w[n];
            if (cell.kind == ArcKind::kFull) {
                out.F0 += wk * R * 2.0 * pi;
                if (u != 0.0) {
                    double sum = 0.0;
                    for (int k = 0; k < ntheta; ++k) {
                        const double th = 2.0 * pi * (k + 0.5) / ntheta;
                        const double ph = phi_lens(R, th, pf);
                        if (!(ph > 0.0)) { out.ok = false; break; }
                        sum += std::sqrt(ph);
                    }
                    out.F_half += wk * R * 2.0 * pi * sum / ntheta;
                }
                continue;
            }
            if (cell.kind != ArcKind::kArcs) { out.ok = false; continue; }
            const ArcSet as = arc_intervals(R, pf);
            if (as.kind != ArcKind::kArcs) { out.ok = false; continue; }
            for (const auto& arc : as.arcs) {
                const auto pe = polish_endpoint(R, arc[0], pf);
                const auto pl = polish_endpoint(R, arc[1], pf);
                double end = pl.theta;
                if (end <= pe.theta) end += 2.0 * pi;
                out.F0 += wk * R * (end - pe.theta);
                if (u == 0.0) continue;
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
                out.F_half += wk * R * ah * sum;
            }
        }
    }
    const double denom = pi * p.rho * p.rho * (1.0 - u / 3.0);
    out.mu = ((1.0 - u) * out.F0 + u * out.F_half) / denom;
    return out;
}

template <class F>
double elapsed(F&& f) {
    const auto t0 = Clock::now();
    volatile double sink = f();
    (void)sink;
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Row {
    const Case* c = nullptr;
    GmLauricella6Epoch cold;
    GmLauricella6Epoch warm;
    GmLauricella6Epoch cold_jac;
    GmLauricella6Epoch warm_jac;
    EpochValue v2_value{};
    EpochJacobian v2_jac{};
    AngularEpoch angular{};
    double cold_ms = 0.0;
    double warm_ms = 0.0;
    double cold_jac_ms = 0.0;
    double warm_jac_ms = 0.0;
    double v2_value_ms = 0.0;
    double v2_jac_ms = 0.0;
    double angular_ms = 0.0;
};

template <int Order>
void run_pf6_case(const Case& c, int nr, Row& row) {
    std::vector<Cplx<__float128>> roots;
    row.cold_ms = elapsed([&] {
        row.cold = gm_lauricella6_epoch<Order>(c.p, c.u, nr, &roots, false);
        return row.cold.mu;
    });
    row.warm_ms = elapsed([&] {
        row.warm = gm_lauricella6_epoch<Order>(c.p, c.u, nr, &roots, false);
        return row.warm.mu;
    });
    if (c.u != 0.0) {
        std::vector<Cplx<__float128>> jac_roots;
        row.cold_jac_ms = elapsed([&] {
            row.cold_jac = gm_lauricella6_epoch<Order>(
                c.p, c.u, nr, &jac_roots, true);
            return row.cold_jac.mu;
        });
        row.warm_jac_ms = elapsed([&] {
            row.warm_jac = gm_lauricella6_epoch<Order>(
                c.p, c.u, nr, &jac_roots, true);
            return row.warm_jac.mu;
        });
    }
}

using StageSamples = std::array<std::vector<double>, 8>;

struct PacketAggregate {
    long long candidates = 0;
    long long built = 0;
    long long accepted = 0;
    long long rejected = 0;
    long long quality_rejected = 0;
    long long build_failed = 0;
    std::array<long long, 17> depth_hist{};
    std::array<long long, 65> accepted_node_hist{};
    double endpoint_root_ms = 0.0;
    double algebraic_geometry_ms = 0.0;
    double log_derivative_ms = 0.0;
    double pfaffian_recurrence_ms = 0.0;
    double quality_tail_ms = 0.0;
    double accepted_candidate_ms = 0.0;
    double rejected_candidate_ms = 0.0;
};

void append_packet(PacketAggregate& sum, const GmLauricella6Cost& cost) {
    const auto& p = cost.packet_profile;
    sum.candidates += p.candidates;
    sum.built += p.built;
    sum.accepted += p.accepted;
    sum.rejected += p.rejected;
    sum.quality_rejected += p.quality_rejected;
    sum.build_failed += p.build_failed;
    for (size_t i = 0; i < sum.depth_hist.size(); ++i)
        sum.depth_hist[i] += p.depth_hist[i];
    for (size_t i = 0; i < sum.accepted_node_hist.size(); ++i)
        sum.accepted_node_hist[i] += p.accepted_node_hist[i];
    sum.endpoint_root_ms += p.endpoint_root_ms;
    sum.algebraic_geometry_ms += p.algebraic_geometry_ms;
    sum.log_derivative_ms += p.log_derivative_ms;
    sum.pfaffian_recurrence_ms += p.pfaffian_recurrence_ms;
    sum.quality_tail_ms += p.quality_tail_ms;
    sum.accepted_candidate_ms += p.accepted_candidate_ms;
    sum.rejected_candidate_ms += p.rejected_candidate_ms;
}

void append_stages(StageSamples& samples, const GmLauricella6Epoch& epoch) {
    samples[0].push_back(epoch.cost.topology_ms);
    samples[1].push_back(epoch.cost.arc_geometry_ms);
    samples[2].push_back(epoch.cost.algebraic_geometry_ms);
    samples[3].push_back(epoch.cost.physical_seed_ms);
    samples[4].push_back(epoch.cost.connection_ms);
    samples[5].push_back(epoch.cost.transport_ms);
    samples[6].push_back(epoch.cost.analytic_jacobian_ms);
    samples[7].push_back(epoch.cost.physical_fallback_ms);
}

double jac_error(const GmLauricella6Epoch& a, const EpochJacobian& b) {
    double e = 0.0;
    for (int j = 0; j < 5; ++j) e = std::max(e, rel(a.grad_mu[j], b.grad_mu[j]));
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "evidence/holonomic/gm_coverage_cases.tsv";
    const int nr = argc > 2 ? std::max(4, std::atoi(argv[2])) : 64;
    const int limit = argc > 3 ? std::atoi(argv[3]) : 0;
    const int angular_arg = argc > 4 ? std::atoi(argv[4]) : 0;
    const int angular_n = angular_arg > 0 ? std::max(32, angular_arg) : 0;
    const int order = argc > 5 ? std::atoi(argv[5]) : 20;
    if (order != 8 && order != 10 && order != 12 && order != 14 &&
        order != 16 && order != 20) {
        std::fprintf(stderr, "order must be one of 8,10,12,14,16,20\n");
        return 2;
    }
    const auto cases = load(path, limit);
    if (cases.empty()) {
        std::fprintf(stderr, "no cases in %s\n", path);
        return 2;
    }

    // Explicitly pin the incumbent benchmark to the normal V2 route.  No
    // PF6 result is wired into these flags.
    holo_holonomic_transport_override() = 0;
    holo_mv_transport_override() = 0;

    std::vector<Row> rows;
    rows.reserve(cases.size());
    int cold_true = 0, warm_true = 0, cold_jac_true = 0, warm_jac_true = 0;
    int cold_nodes = 0, cold_transported = 0;
    int warm_nodes = 0, warm_transported = 0;
    int cold_jac_nodes = 0, cold_jac_transported = 0;
    int warm_jac_nodes = 0, warm_jac_transported = 0;
    double worst_cold_v2 = 0.0, worst_warm_v2 = 0.0;
    double worst_cold_jac_v2 = 0.0, worst_warm_jac_v2 = 0.0;
    double worst_cold_jac_grad = 0.0, worst_warm_jac_grad = 0.0;
    double worst_cold_ang = 0.0, worst_warm_ang = 0.0;
    long long cold_arcs = 0, warm_arcs = 0;
    long long cold_seeds = 0, warm_seeds = 0;
    long long cold_connections = 0, warm_connections = 0;
    long long cold_fallbacks = 0, warm_fallbacks = 0;
    long long cold_reseeds = 0, warm_reseeds = 0;
    long long cold_promotions = 0, warm_promotions = 0;
    long long cold_precision_failures = 0, warm_precision_failures = 0;
    long long jac_cold_arcs = 0, jac_warm_arcs = 0;
    long long jac_cold_seeds = 0, jac_warm_seeds = 0;
    long long jac_cold_connections = 0, jac_warm_connections = 0;
    long long jac_cold_fallbacks = 0, jac_warm_fallbacks = 0;
    long long jac_cold_reseeds = 0, jac_warm_reseeds = 0;
    long long jac_cold_promotions = 0, jac_warm_promotions = 0;
    long long jac_cold_precision_failures = 0, jac_warm_precision_failures = 0;
    std::vector<double> cold_jac_marginal_ms, warm_jac_marginal_ms;
    std::vector<double> cold_analytic_jac_ms, warm_analytic_jac_ms;
    StageSamples value_cold_stages{}, value_warm_stages{};
    StageSamples jac_cold_stages{}, jac_warm_stages{};
    for (const auto& c : cases) {
        Row row;
        row.c = &c;
        row.v2_value_ms = elapsed([&] {
            const auto v = epoch_value(c.p, c.u, nr, false);
            return v.mu;
        });
        row.v2_value = epoch_value(c.p, c.u, nr, false);
        row.v2_jac_ms = elapsed([&] {
            const auto v = epoch_jacobian(c.p, c.u, nr, false);
            return v.mu;
        });
        row.v2_jac = epoch_jacobian(c.p, c.u, nr, false);
        switch (order) {
            case 8: run_pf6_case<8>(c, nr, row); break;
            case 10: run_pf6_case<10>(c, nr, row); break;
            case 12: run_pf6_case<12>(c, nr, row); break;
            case 14: run_pf6_case<14>(c, nr, row); break;
            case 16: run_pf6_case<16>(c, nr, row); break;
            case 20: run_pf6_case<20>(c, nr, row); break;
        }
        if (c.u != 0.0) {
            cold_jac_marginal_ms.push_back(
                std::max(0.0, row.cold_jac_ms - row.cold_ms));
            warm_jac_marginal_ms.push_back(
                std::max(0.0, row.warm_jac_ms - row.warm_ms));
            cold_analytic_jac_ms.push_back(
                row.cold_jac.cost.analytic_jacobian_ms);
            warm_analytic_jac_ms.push_back(
                row.warm_jac.cost.analytic_jacobian_ms);
            append_stages(value_cold_stages, row.cold);
            append_stages(value_warm_stages, row.warm);
            append_stages(jac_cold_stages, row.cold_jac);
            append_stages(jac_warm_stages, row.warm_jac);
        }
        if (angular_n > 0 && (int)rows.size() < 12) {
            row.angular_ms = elapsed([&] {
                row.angular = angular_epoch(c.p, c.u, nr, angular_n);
                return row.angular.mu;
            });
            worst_cold_ang = std::max(worst_cold_ang,
                                      rel(row.cold.mu, row.angular.mu));
            worst_warm_ang = std::max(worst_warm_ang,
                                      rel(row.warm.mu, row.angular.mu));
        }
        worst_cold_v2 = std::max(worst_cold_v2,
                                  rel(row.cold.mu, row.v2_value.mu));
        worst_warm_v2 = std::max(worst_warm_v2,
                                  rel(row.warm.mu, row.v2_value.mu));
        if (row.cold.all_true_transport) ++cold_true;
        if (row.warm.all_true_transport) ++warm_true;
        if (c.u != 0.0) {
            if (row.cold_jac.status == Status::OK &&
                row.cold_jac.all_true_transport &&
                row.cold_jac.cost.jacobian_nodes ==
                    row.cold_jac.cost.nodes)
                ++cold_jac_true;
            if (row.warm_jac.status == Status::OK &&
                row.warm_jac.all_true_transport &&
                row.warm_jac.cost.jacobian_nodes ==
                    row.warm_jac.cost.nodes)
                ++warm_jac_true;
            worst_cold_jac_v2 = std::max(
                worst_cold_jac_v2, rel(row.cold_jac.mu, row.v2_jac.mu));
            worst_warm_jac_v2 = std::max(
                worst_warm_jac_v2, rel(row.warm_jac.mu, row.v2_jac.mu));
            worst_cold_jac_grad = std::max(
                worst_cold_jac_grad, jac_error(row.cold_jac, row.v2_jac));
            worst_warm_jac_grad = std::max(
                worst_warm_jac_grad, jac_error(row.warm_jac, row.v2_jac));
        }
        cold_nodes += row.cold.cost.nodes;
        cold_transported += row.cold.cost.transported;
        warm_nodes += row.warm.cost.nodes;
        warm_transported += row.warm.cost.transported;
        cold_jac_nodes += row.cold_jac.cost.nodes;
        cold_jac_transported += row.cold_jac.cost.transported;
        warm_jac_nodes += row.warm_jac.cost.nodes;
        warm_jac_transported += row.warm_jac.cost.transported;
        cold_arcs += row.cold.cost.arcs;
        warm_arcs += row.warm.cost.arcs;
        cold_seeds += row.cold.cost.physical_seeds;
        warm_seeds += row.warm.cost.physical_seeds;
        cold_connections += row.cold.cost.connection_constructions;
        warm_connections += row.warm.cost.connection_constructions;
        cold_fallbacks += row.cold.cost.physical_fallback;
        warm_fallbacks += row.warm.cost.physical_fallback;
        cold_reseeds += row.cold.cost.reseeds;
        warm_reseeds += row.warm.cost.reseeds;
        cold_promotions += row.cold.cost.precision_promotions;
        warm_promotions += row.warm.cost.precision_promotions;
        cold_precision_failures += row.cold.cost.precision_failures;
        warm_precision_failures += row.warm.cost.precision_failures;
        jac_cold_arcs += row.cold_jac.cost.arcs;
        jac_warm_arcs += row.warm_jac.cost.arcs;
        jac_cold_seeds += row.cold_jac.cost.physical_seeds;
        jac_warm_seeds += row.warm_jac.cost.physical_seeds;
        jac_cold_connections += row.cold_jac.cost.connection_constructions;
        jac_warm_connections += row.warm_jac.cost.connection_constructions;
        jac_cold_fallbacks += row.cold_jac.cost.physical_fallback;
        jac_warm_fallbacks += row.warm_jac.cost.physical_fallback;
        jac_cold_reseeds += row.cold_jac.cost.reseeds;
        jac_warm_reseeds += row.warm_jac.cost.reseeds;
        jac_cold_promotions += row.cold_jac.cost.precision_promotions;
        jac_warm_promotions += row.warm_jac.cost.precision_promotions;
        jac_cold_precision_failures += row.cold_jac.cost.precision_failures;
        jac_warm_precision_failures += row.warm_jac.cost.precision_failures;
        std::printf(
            "case %-16s u=%.1f cold=% .9g warm=% .9g v2=% .9g "
            "err(c/w)=%.3e/%.3e status(c/w)=%s/%s true(c/w)=%d/%d "
            "ms(c/w/v2value)=%.3f/%.3f/%.3f blocks(c/w)=%d/%d "
            "nodes(c/w)=%d/%d transported=%d/%d\n",
            c.name.c_str(), c.u, row.cold.mu, row.warm.mu, row.v2_value.mu,
            rel(row.cold.mu, row.v2_value.mu),
            rel(row.warm.mu, row.v2_value.mu),
            to_string(row.cold.status), to_string(row.warm.status),
            (int)row.cold.all_true_transport, (int)row.warm.all_true_transport,
            row.cold_ms, row.warm_ms, row.v2_value_ms,
            row.cold.cost.blocks, row.warm.cost.blocks,
            row.cold.cost.nodes, row.warm.cost.nodes,
            row.cold.cost.transported, row.warm.cost.transported);
        if (c.u != 0.0) {
            std::printf(
                "  jac cold=% .9g warm=% .9g v2=% .9g err(c/w)=%.3e/%.3e "
                "status(c/w)=%s/%s true(c/w)=%d/%d "
                "ms(c/w/v2)=%.3f/%.3f/%.3f jacnodes=%d/%d "
                "grad_err(c/w)=%.3e/%.3e\n",
                row.cold_jac.mu, row.warm_jac.mu, row.v2_jac.mu,
                rel(row.cold_jac.mu, row.v2_jac.mu),
                rel(row.warm_jac.mu, row.v2_jac.mu),
                to_string(row.cold_jac.status),
                to_string(row.warm_jac.status),
                (int)(row.cold_jac.status == Status::OK),
                (int)(row.warm_jac.status == Status::OK),
                row.cold_jac_ms, row.warm_jac_ms, row.v2_jac_ms,
                row.cold_jac.cost.jacobian_nodes,
                row.warm_jac.cost.jacobian_nodes,
                jac_error(row.cold_jac, row.v2_jac),
                jac_error(row.warm_jac, row.v2_jac));
            std::printf(
                "  cost(value) cold topo/arc/geom/seed/conn/trans/jac/fallback="
                "%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f "
                "warm=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f\n",
                row.cold.cost.topology_ms, row.cold.cost.arc_geometry_ms,
                row.cold.cost.algebraic_geometry_ms,
                row.cold.cost.physical_seed_ms, row.cold.cost.connection_ms,
                row.cold.cost.transport_ms,
                row.cold.cost.analytic_jacobian_ms,
                row.cold.cost.physical_fallback_ms,
                row.warm.cost.topology_ms, row.warm.cost.arc_geometry_ms,
                row.warm.cost.algebraic_geometry_ms,
                row.warm.cost.physical_seed_ms, row.warm.cost.connection_ms,
                row.warm.cost.transport_ms,
                row.warm.cost.analytic_jacobian_ms,
                row.warm.cost.physical_fallback_ms);
            std::printf(
                "  cost(jac) cold topo/arc/geom/seed/conn/trans/jac/fallback="
                "%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f "
                "warm=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f\n",
                row.cold_jac.cost.topology_ms,
                row.cold_jac.cost.arc_geometry_ms,
                row.cold_jac.cost.algebraic_geometry_ms,
                row.cold_jac.cost.physical_seed_ms,
                row.cold_jac.cost.connection_ms,
                row.cold_jac.cost.transport_ms,
                row.cold_jac.cost.analytic_jacobian_ms,
                row.cold_jac.cost.physical_fallback_ms,
                row.warm_jac.cost.topology_ms,
                row.warm_jac.cost.arc_geometry_ms,
                row.warm_jac.cost.algebraic_geometry_ms,
                row.warm_jac.cost.physical_seed_ms,
                row.warm_jac.cost.connection_ms,
                row.warm_jac.cost.transport_ms,
                row.warm_jac.cost.analytic_jacobian_ms,
                row.warm_jac.cost.physical_fallback_ms);
        }
        if (row.cold.cost.chart_failed || row.cold.cost.branch_failed ||
            row.cold.cost.geometry_failed || row.cold.cost.seed_failed ||
            row.cold.cost.connection_failed || row.cold.cost.coverage_failed ||
            row.cold.cost.node_failed)
            std::printf("  failures cold chart/branch/geom/seed/conn/cover/node="
                        "%d/%d/%d/%d/%d/%d/%d warm=%d/%d/%d/%d/%d/%d/%d\n",
                        row.cold.cost.chart_failed, row.cold.cost.branch_failed,
                        row.cold.cost.geometry_failed, row.cold.cost.seed_failed,
                        row.cold.cost.connection_failed, row.cold.cost.coverage_failed,
                        row.cold.cost.node_failed,
                        row.warm.cost.chart_failed, row.warm.cost.branch_failed,
                        row.warm.cost.geometry_failed, row.warm.cost.seed_failed,
                        row.warm.cost.connection_failed, row.warm.cost.coverage_failed,
                        row.warm.cost.node_failed);
        for (const auto& rr : row.cold.cost.coverage_failed_ranges)
            std::printf("  coverage_failed_range=[%.17g,%.17g]\n",
                        rr[0], rr[1]);
        rows.push_back(std::move(row));
    }

    std::vector<double> cold_ms, warm_ms, v2_value_ms;
    std::vector<double> cold_ld_ms, warm_ld_ms, v2_ld_value_ms;
    std::vector<double> cold_jac_ms, warm_jac_ms, v2_jac_ms;
    for (const auto& r : rows) {
        cold_ms.push_back(r.cold_ms);
        warm_ms.push_back(r.warm_ms);
        v2_value_ms.push_back(r.v2_value_ms);
        if (r.c->u != 0.0) {
            cold_ld_ms.push_back(r.cold_ms);
            warm_ld_ms.push_back(r.warm_ms);
            v2_ld_value_ms.push_back(r.v2_value_ms);
            cold_jac_ms.push_back(r.cold_jac_ms);
            warm_jac_ms.push_back(r.warm_jac_ms);
            v2_jac_ms.push_back(r.v2_jac_ms);
        }
    }
    auto med = [](std::vector<double> x) {
        std::sort(x.begin(), x.end());
        return x[x.size() / 2];
    };
    int ld = 0, uniform = 0;
    for (const auto& c : cases) (c.u == 0.0 ? ++uniform : ++ld);
    std::printf(
        "TOTAL cases=%zu uniform=%d limb=%d nr=%d cold_true=%d warm_true=%d "
        "cold_jac_ok=%d warm_jac_ok=%d "
        "nodes(c/w/jc/jw)=%d/%d/%d/%d transported=%d/%d/%d/%d "
        "rates(c/w/jc/jw)=%.6f/%.6f/%.6f/%.6f "
        "median_value_ms(c/w/v2)=%.3f/%.3f/%.3f "
        "median_jac_ms(c/w/v2)=%.3f/%.3f/%.3f "
        "worst_v2_value(c/w)=%.3e/%.3e worst_v2_jac_mu(c/w)=%.3e/%.3e "
        "worst_v2_jac_grad(c/w)=%.3e/%.3e "
        "angular_first12(c/w)=%.3e/%.3e\n",
        rows.size(), uniform, ld, nr, cold_true, warm_true, cold_jac_true,
        warm_jac_true, cold_nodes, warm_nodes, cold_jac_nodes, warm_jac_nodes,
        cold_transported, warm_transported, cold_jac_transported,
        warm_jac_transported,
        cold_nodes ? (double)cold_transported / cold_nodes : 0.0,
        warm_nodes ? (double)warm_transported / warm_nodes : 0.0,
        cold_jac_nodes ? (double)cold_jac_transported / cold_jac_nodes : 0.0,
        warm_jac_nodes ? (double)warm_jac_transported / warm_jac_nodes : 0.0,
        med(cold_ms), med(warm_ms), med(v2_value_ms),
        cold_jac_ms.empty() ? 0.0 : med(cold_jac_ms),
        warm_jac_ms.empty() ? 0.0 : med(warm_jac_ms),
        v2_jac_ms.empty() ? 0.0 : med(v2_jac_ms),
        worst_cold_v2, worst_warm_v2, worst_cold_jac_v2,
        worst_warm_jac_v2, worst_cold_jac_grad, worst_warm_jac_grad,
        worst_cold_ang, worst_warm_ang);
    std::printf("median_value_ld_ms(c/w/v2)=%.3f/%.3f/%.3f\n",
                med(cold_ld_ms), med(warm_ld_ms), med(v2_ld_value_ms));

    auto print_median_pair = [](const char* label,
                                std::vector<double> cold,
                                std::vector<double> warm) {
        std::sort(cold.begin(), cold.end());
        std::sort(warm.begin(), warm.end());
        const double c = cold.empty() ? 0.0 : cold[cold.size() / 2];
        const double w = warm.empty() ? 0.0 : warm[warm.size() / 2];
        std::printf("%s(c/w)=%.3f/%.3f ms\n", label, c, w);
    };
    std::printf(
        "COUNTS(value) arcs(c/w)=%lld/%lld seeds=%lld/%lld "
        "connections=%lld/%lld fallback=%lld/%lld reseed=%lld/%lld "
        "promotions=%lld/%lld precision_fail=%lld/%lld\n",
        cold_arcs, warm_arcs, cold_seeds, warm_seeds, cold_connections,
        warm_connections, cold_fallbacks, warm_fallbacks, cold_reseeds,
        warm_reseeds, cold_promotions, warm_promotions,
        cold_precision_failures, warm_precision_failures);
    std::printf(
        "COUNTS(jac) arcs(c/w)=%lld/%lld seeds=%lld/%lld "
        "connections=%lld/%lld fallback=%lld/%lld reseed=%lld/%lld "
        "promotions=%lld/%lld precision_fail=%lld/%lld\n",
        jac_cold_arcs, jac_warm_arcs, jac_cold_seeds, jac_warm_seeds,
        jac_cold_connections, jac_warm_connections, jac_cold_fallbacks,
        jac_warm_fallbacks, jac_cold_reseeds, jac_warm_reseeds,
        jac_cold_promotions, jac_warm_promotions,
        jac_cold_precision_failures, jac_warm_precision_failures);
    print_median_pair("jacobian_marginal_ms", cold_jac_marginal_ms,
                      warm_jac_marginal_ms);
    print_median_pair("analytic_jacobian_cost_ms", cold_analytic_jac_ms,
                      warm_analytic_jac_ms);
    auto print_stage_medians = [](const char* label,
                                  const StageSamples& samples) {
        std::array<double, 8> m{};
        for (int i = 0; i < 8; ++i) {
            std::vector<double> values = samples[i];
            std::sort(values.begin(), values.end());
            m[i] = values.empty() ? 0.0 : values[values.size() / 2];
        }
        std::printf(
            "COST_MEDIAN(%s) topology/D14=%.3f arc=%.3f geometry=%.3f "
            "seed=%.3f connection=%.3f transport=%.3f "
            "analytic_jac=%.3f fallback=%.3f ms\n",
            label, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
    };
    print_stage_medians("value_cold", value_cold_stages);
    print_stage_medians("value_warm", value_warm_stages);
    print_stage_medians("jac_cold", jac_cold_stages);
    print_stage_medians("jac_warm", jac_warm_stages);

    auto print_packet_aggregate = [&](const char* label, auto getter) {
        PacketAggregate sum;
        for (const auto& r : rows)
            if (r.c->u != 0.0) append_packet(sum, getter(r));
        int max_depth = 0;
        for (int d = 0; d < (int)sum.depth_hist.size(); ++d)
            if (sum.depth_hist[d] != 0) max_depth = d;
        std::printf(
            "PACKET_SUMMARY(%s) candidates=%lld built=%lld accepted=%lld "
            "rejected=%lld quality_rejected=%lld build_failed=%lld "
            "max_depth=%d stages_ms(endpoint/root=%.6f algebraic=%.6f "
            "logderiv=%.6f pfaffian=%.6f quality=%.6f "
            "accepted=%.6f rejected=%.6f)\n",
            label, sum.candidates, sum.built, sum.accepted, sum.rejected,
            sum.quality_rejected, sum.build_failed, max_depth,
            sum.endpoint_root_ms, sum.algebraic_geometry_ms,
            sum.log_derivative_ms, sum.pfaffian_recurrence_ms,
            sum.quality_tail_ms, sum.accepted_candidate_ms,
            sum.rejected_candidate_ms);
        std::printf("PACKET_DEPTH(%s)", label);
        for (int d = 0; d <= max_depth; ++d)
            std::printf(" %d:%lld", d, sum.depth_hist[d]);
        std::printf("\nPACKET_ACCEPTED_NODES(%s)", label);
        for (int n = 0; n < (int)sum.accepted_node_hist.size(); ++n)
            if (sum.accepted_node_hist[n] != 0)
                std::printf(" %d:%lld", n, sum.accepted_node_hist[n]);
        std::printf("\n");
    };
    print_packet_aggregate("value_cold",
                           [](const auto& r) -> const GmLauricella6Cost& {
                               return r.cold.cost;
                           });
    print_packet_aggregate("value_warm",
                           [](const auto& r) -> const GmLauricella6Cost& {
                               return r.warm.cost;
                           });
    print_packet_aggregate("jac_cold",
                           [](const auto& r) -> const GmLauricella6Cost& {
                               return r.cold_jac.cost;
                           });
    print_packet_aggregate("jac_warm",
                           [](const auto& r) -> const GmLauricella6Cost& {
                               return r.warm_jac.cost;
                           });

    const char* packet_path = std::getenv("GM6_PACKET_PROFILE_PATH");
    if (packet_path && packet_path[0] != '\0') {
        std::ofstream profile(packet_path);
        profile << "# PF6 Phase 5 packet profile; timing requires "
                   "GM6_PACKET_PROFILE=1\n";
        profile << "SUMMARY lane case u candidates built accepted rejected "
                   "quality_rejected build_failed max_depth endpoint_root_ms "
                   "algebraic_geometry_ms log_derivative_ms "
                   "pfaffian_recurrence_ms quality_tail_ms accepted_ms "
                   "rejected_ms\n";
        profile << "ARC lane case u arc_index cell_index cell_lo cell_hi "
                   "radial_center theta_lo theta_hi event_gap_ratio "
                   "nearest_event_kind xi_min_divisor chart2 constructions "
                   "built accepted rejected quality_rejected build_failed "
                   "max_depth accepted_node_sum accepted_node_min "
                   "accepted_node_max node_failed first_failed_R "
                   "first_failed_packet_center first_failed_packet_lo "
                   "first_failed_packet_hi first_failed_block_lo "
                   "first_failed_block_hi first_failed_phase_error "
                   "first_failed_packet_tail construction_ms\n";
        auto write_profile = [&](const char* lane,
                                 const GmLauricella6Epoch& epoch,
                                 const Case& c) {
            const auto& p = epoch.cost.packet_profile;
            profile << "SUMMARY " << lane << ' ' << c.name << ' ' << c.u << ' '
                    << p.candidates << ' ' << p.built << ' ' << p.accepted
                    << ' ' << p.rejected << ' ' << p.quality_rejected << ' '
                    << p.build_failed << ' ';
            int max_depth = 0;
            for (int d = 0; d < (int)p.depth_hist.size(); ++d)
                if (p.depth_hist[d] != 0) max_depth = d;
            profile << max_depth << ' ' << std::setprecision(17)
                    << p.endpoint_root_ms << ' ' << p.algebraic_geometry_ms
                    << ' ' << p.log_derivative_ms << ' '
                    << p.pfaffian_recurrence_ms << ' ' << p.quality_tail_ms
                    << ' ' << p.accepted_candidate_ms << ' '
                    << p.rejected_candidate_ms << '\n';
            for (const auto& a : epoch.cost.packet_arcs) {
                profile << "ARC " << lane << ' ' << c.name << ' ' << c.u << ' '
                        << a.arc_index << ' ' << a.cell_index << ' '
                        << a.cell_lo << ' ' << a.cell_hi << ' '
                        << a.radial_center << ' ' << a.theta_lo << ' '
                        << a.theta_hi << ' ' << a.event_gap_ratio << ' '
                        << a.nearest_event_kind << ' ' << a.xi_min_divisor
                        << ' ' << a.chart2 << ' ' << a.constructions << ' '
                        << a.built << ' ' << a.accepted << ' ' << a.rejected
                        << ' ' << a.quality_rejected << ' ' << a.build_failed
                        << ' ' << a.max_depth << ' ' << a.accepted_node_sum
                        << ' ' << a.accepted_node_min << ' '
                        << a.accepted_node_max << ' ' << a.node_failed << ' '
                        << a.first_failed_R << ' '
                        << a.first_failed_packet_center << ' '
                        << a.first_failed_packet_lo << ' '
                        << a.first_failed_packet_hi << ' '
                        << a.first_failed_block_lo << ' '
                        << a.first_failed_block_hi << ' '
                        << a.first_failed_phase_error << ' '
                        << a.first_failed_packet_tail << ' '
                        << a.construction_ms
                        << '\n';
            }
        };
        for (const auto& r : rows) {
            write_profile("value_cold", r.cold, *r.c);
            write_profile("value_warm", r.warm, *r.c);
            if (r.c->u != 0.0) {
                write_profile("jac_cold", r.cold_jac, *r.c);
                write_profile("jac_warm", r.warm_jac, *r.c);
            }
        }
        std::printf("PACKET_PROFILE_FILE %s\n", packet_path);
    }

    if (angular_n > 0) {
        std::printf("ANGULAR_REFERENCE ntheta=%d only_first12=1\n", angular_n);
        for (const auto& r : rows) {
            if (r.angular_ms == 0.0) continue;
            std::printf("angular %-16s mu=% .12g cold_err=%.3e warm_err=%.3e ok=%d\n",
                        r.c->name.c_str(), r.angular.mu,
                        rel(r.cold.mu, r.angular.mu),
                        rel(r.warm.mu, r.angular.mu), (int)r.angular.ok);
        }
    }
    std::printf("ORDER %d\n", order);
    return 0;
}
