// End-to-end true-GM diagnostic.
//
// The measured path is physical period seed -> connection Taylor transport
// -> physical observable reconstruction -> F0/Fhalf -> mu.  It is kept
// separate from V2/V3: the incumbent is called in the same process only for
// a same-case wall-time/status/value comparison, and no K-rule packet is
// called by the GM path.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/gm_physical_transport.hpp"

using namespace lcbinint::holonomic;
using clk = std::chrono::steady_clock;

namespace {

struct Case {
    LensParams p;
    double u = 0.0;
    std::string name;
};

struct AngularReference {
    bool ok = true;
    double F0 = 0.0;
    double F_half = 0.0;
    double mu = 0.0;
};

struct GmJacobianDiagnostic {
    std::array<double, 5> grad{};
    double checksum = 0.0;
};

enum class GmLane { kDouble8, kDd8, kQf10 };

GmLane parse_lane(const char* text) {
    if (text && std::string(text) == "double8") return GmLane::kDouble8;
    if (text && std::string(text) == "qf10") return GmLane::kQf10;
    return GmLane::kDd8;
}

const char* lane_name(GmLane lane) {
    switch (lane) {
        case GmLane::kDouble8: return "double8";
        case GmLane::kQf10: return "qf10";
        default: return "dd8";
    }
}

double rel(double a, double b) {
    return std::fabs(a - b) / (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t i = std::min(v.size() - 1,
                              (size_t)std::floor(q * (v.size() - 1)));
    return v[i];
}

std::vector<Case> load(const char* path, int limit) {
    std::ifstream in(path);
    std::vector<Case> out;
    std::string line;
    while (std::getline(in, line) && (int)out.size() < limit) {
        std::istringstream ls(line);
        double xs, ys, rho, q, a, tj;
        int bary;
        Case c;
        if (!(ls >> xs >> ys >> rho >> q >> a >> bary >> c.u >> tj >> c.name))
            continue;
        c.p = LensParams{xs, ys, rho, q, a, (bool)bary};
        out.push_back(c);
    }
    return out;
}

AngularReference angular_reference(const LensParams& p, double u, int n_r,
                                   int n_theta) {
    AngularReference out;
    const PrimaryFrame pf = PrimaryFrame::from(p);
    const TopologyResult topo = classify_cells(pf);
    Cheb1Dyn rr(n_r);
    const bool want_fhalf = u != 0.0;
    constexpr double pi = 3.14159265358979323846;
    for (const auto& cell : topo.cells) {
        const double width = cell.r_hi - cell.r_lo;
        if (!(width > 0.0) || cell.kind == ArcKind::kEmpty) continue;
        const double inset = 1e-9 * width;
        const double lo = cell.r_lo + inset;
        const double hi = cell.r_hi - inset;
        const double Rc = 0.5 * (lo + hi);
        const double rh = 0.5 * (hi - lo);
        if (cell.kind == ArcKind::kFull) {
            for (int n = 0; n < n_r; ++n) {
                const double R = Rc + rh * rr.x[n];
                const double wk = rh * rr.w[n];
                out.F0 += wk * R * 2.0 * pi;
                if (want_fhalf) {
                    double fh = 0.0;
                    if (!gm_transport_detail::full_circle_fhalf(
                            R, pf, n_theta, fh)) out.ok = false;
                    out.F_half += wk * fh;
                }
            }
            continue;
        }
        QuarticWarm qw;
        RootPairWarm rpw;
        rpw.certify = topo.from_warm_d14;
        for (int n = 0; n < n_r; ++n) {
            const double R = Rc + rh * rr.x[n];
            const double wk = rh * rr.w[n];
            const ArcSet arcs = arc_intervals(R, pf, &qw, &rpw);
            if (arcs.kind != ArcKind::kArcs || arcs.arcs.empty()) {
                out.ok = false;
                continue;
            }
            double f0 = 0.0;
            for (const auto& arc : arcs.arcs)
                f0 += gm_transport_detail::arc_measure(arc);
            out.F0 += wk * R * f0;
            if (!want_fhalf) continue;
            for (const auto& arc : arcs.arcs) {
                double fh = 0.0;
                if (!gm_physical_arc_angular(R, pf, arc, n_theta, fh)) {
                    out.ok = false;
                    continue;
                }
                out.F_half += wk * fh;
            }
        }
    }
    const double denom = pi * p.rho * p.rho * (1.0 - u / 3.0);
    out.mu = ((1.0 - u) * out.F0 + u * out.F_half) / denom;
    return out;
}

template <class Fn>
double best(int reps, Fn&& fn) {
    double ans = 1e300;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = clk::now();
        volatile double sink = fn();
        (void)sink;
        const auto t1 = clk::now();
        ans = std::min(ans, std::chrono::duration<double, std::milli>(
                                  t1 - t0).count());
    }
    return ans;
}

// The physical seed currently has no analytic endpoint-parameter derivative
// path.  Use central differences of the complete true-GM value lane as an
// explicit diagnostic for the requested value+5Jac wall cost.  This is kept
// separate from the point GmDual5 benchmark and is never used by production.
GmEpochValue gm_value_lane(const Case& c, int n_r, GmLane lane,
                           bool direct = false) {
    switch (lane) {
        case GmLane::kDouble8:
            return direct ? gm_epoch_value_direct_physical<8, double>(
                                c.p, c.u, n_r)
                          : gm_epoch_value<8, double>(c.p, c.u, n_r);
        case GmLane::kQf10:
            return direct ? gm_epoch_value_direct_physical<10, __float128>(
                                c.p, c.u, n_r)
                          : gm_epoch_value<10, __float128>(c.p, c.u, n_r);
        default:
            return direct ? gm_epoch_value_direct_physical<8, DD>(
                                c.p, c.u, n_r)
                          : gm_epoch_value<8, DD>(c.p, c.u, n_r);
    }
}

GmJacobianDiagnostic gm_fd_jacobian(const Case& c, int n_r, GmLane lane) {
    GmJacobianDiagnostic out;
    const double base[5] = {c.p.xs, c.p.ys, c.p.rho, c.p.q, c.p.a};
    for (int j = 0; j < 5; ++j) {
        double step = 2.0e-6 * std::max(1.0, std::fabs(base[j]));
        if ((j == 2 || j == 4) && base[j] > 0.0 &&
            step >= 0.25 * base[j]) step = 0.25 * base[j];
        LensParams plus = c.p, minus = c.p;
        double* plus_v[5] = {&plus.xs, &plus.ys, &plus.rho, &plus.q, &plus.a};
        double* minus_v[5] = {&minus.xs, &minus.ys, &minus.rho, &minus.q, &minus.a};
        *plus_v[j] += step;
        *minus_v[j] -= step;
        const Case cp{plus, c.u, c.name};
        const Case cm{minus, c.u, c.name};
        const GmEpochValue vp = gm_value_lane(cp, n_r, lane);
        const GmEpochValue vm = gm_value_lane(cm, n_r, lane);
        out.grad[j] = (vp.mu - vm.mu) / (2.0 * step);
        out.checksum += out.grad[j] * (j + 1.0);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    const int reps = argc > 2 ? std::max(1, std::atoi(argv[2])) : 1;
    const int limit = argc > 3 ? std::max(1, std::atoi(argv[3])) : 12;
    const int n_r = argc > 4 ? std::max(4, std::atoi(argv[4])) : 16;
    const int angular_arg = argc > 5 ? std::atoi(argv[5]) : 0;
    const int angular_n = angular_arg > 0 ? std::max(8, angular_arg) : 0;
    const GmLane lane = parse_lane(argc > 6 ? argv[6] : "dd8");
    const std::vector<Case> cases = load(path, limit);
    if (cases.empty()) {
        std::fprintf(stderr, "no cases in %s\n", path);
        return 2;
    }

    // V2 means the incumbent semi-holonomic value+Jacobian route.  The GM
    // path below does not observe this switch.
    holo_mv_transport_override() = 1;
    holo_holonomic_transport_override() = 1;

    std::vector<double> tgm, tgm_value, tgm_direct, tv2, tv2_value;
    std::vector<double> tgm_jac, tgm_jac_base, err_gm_jac;
    std::vector<double> err_v2, err_v0, err_gm_direct, err_direct_v0;
    std::vector<double> err_value_v2;
    std::vector<double> err_direct_angular, err_angular_v0;
    int gm_ok = 0, gm_ok_value = 0, gm_ok_jac = 0;
    int gm_true_jac = 0, v2_ok = 0, status_changes = 0;
    int value_cases = 0, jac_cases = 0, value_gm_ok = 0;
    double worst_f0 = 0.0, worst_fh = 0.0;
    double worst_f0_direct = 0.0, worst_fh_direct = 0.0;
    double worst_mu_v0 = 0.0, worst_gm_direct = 0.0;
    double sum_seed = 0.0, sum_conn = 0.0, sum_transport = 0.0;
    double sum_arc = 0.0, sum_topology = 0.0, sum_direct = 0.0;
    double sum_direct_lane = 0.0;
    double sum_angular_ms = 0.0;
    int sum_cells = 0, sum_arcs = 0, sum_seeds = 0, sum_transported = 0;
    int sum_conn_attempts = 0, sum_conn_failed = 0;
    int sum_transport_attempts = 0, sum_transport_failed = 0;
    int sum_transport_rejected = 0, sum_direct_nodes = 0;
    double max_transport_tail = 0.0;

    for (const Case& c : cases) {
        const EpochJacobian v2 = epoch_jacobian(c.p, c.u, 64, false);
        const EpochValue v2_value = epoch_value(c.p, c.u, 64, false);
        holo_holonomic_transport_override() = 0;
        const EpochJacobian v0 = epoch_jacobian(c.p, c.u, 64, false);
        holo_holonomic_transport_override() = 1;
        if (v2.status == Status::OK) ++v2_ok;

        const GmEpochValue gm = gm_value_lane(c, n_r, lane);
        const GmEpochValue direct = gm_value_lane(c, n_r, lane, true);
        AngularReference angular;
        const auto angular_t0 = clk::now();
        if (angular_n > 0) angular = angular_reference(c.p, c.u, n_r,
                                                        angular_n);
        const auto angular_t1 = clk::now();
        if (angular_n > 0) {
            sum_angular_ms += std::chrono::duration<double, std::milli>(
                                  angular_t1 - angular_t0).count();
            err_direct_angular.push_back(rel(direct.mu, angular.mu));
            err_angular_v0.push_back(rel(angular.mu, v0.mu));
        }
        const double tg = best(reps, [&] {
            return gm_value_lane(c, n_r, lane).mu;
        });
        const double tgd = best(reps, [&] {
            return gm_value_lane(c, n_r, lane, true).mu;
        });
        const double tv = best(reps, [&] {
            return epoch_jacobian(c.p, c.u, 64, false).mu;
        });
        const double tgv = best(reps, [&] {
            return gm_value_lane(c, n_r, lane).mu;
        });
        const double tvv = best(reps, [&] {
            return epoch_value(c.p, c.u, 64, false).mu;
        });
        tgm.push_back(tg);
        tgm_direct.push_back(tgd);
        tv2.push_back(tv);
        if (c.u == 0.0) {
            tgm_value.push_back(tgv);
            tv2_value.push_back(tvv);
            err_value_v2.push_back(rel(gm.mu, v2_value.mu));
        }
        err_v2.push_back(rel(gm.mu, v2.mu));
        err_v0.push_back(rel(gm.mu, v0.mu));
        err_gm_direct.push_back(rel(gm.mu, direct.mu));
        err_direct_v0.push_back(rel(direct.mu, v0.mu));
        worst_mu_v0 = std::max(worst_mu_v0, err_v0.back());
        worst_gm_direct = std::max(worst_gm_direct, err_gm_direct.back());
        worst_f0 = std::max(worst_f0, rel(gm.F0, v0.F0));
        worst_f0_direct = std::max(worst_f0_direct,
                                   rel(direct.F0, v0.F0));
        if (c.u != 0.0) {
            worst_fh = std::max(worst_fh, rel(gm.F_half, v0.F_half));
            worst_fh_direct = std::max(worst_fh_direct,
                                       rel(direct.F_half, v0.F_half));
            ++jac_cases;

            const GmJacobianDiagnostic gm_jac =
                gm_fd_jacobian(c, n_r, lane);
            const double tgj = best(reps, [&] {
                return gm_fd_jacobian(c, n_r, lane).checksum;
            });
            tgm_jac.push_back(tgj);
            tgm_jac_base.push_back(tg);
            double jac_err = 0.0;
            for (int j = 0; j < 5; ++j)
                jac_err = std::max(jac_err, rel(gm_jac.grad[j],
                                                  v2.grad_mu[j]));
            err_gm_jac.push_back(jac_err);
        } else {
            ++value_cases;
            if (gm.status == Status::OK) ++value_gm_ok;
        }
        if (gm.status == Status::OK) {
            ++gm_ok;
            if (c.u == 0.0) ++gm_ok_value;
            else ++gm_ok_jac;
        }
        if (gm.all_true_transport) {
            ++gm_true_jac;
        }
        if ((gm.status == Status::OK) != (v2.status == Status::OK)) ++status_changes;
        sum_seed += gm.cost.seed_ms;
        sum_conn += gm.cost.connection_ms;
        sum_transport += gm.cost.transport_ms;
        sum_arc += gm.cost.arc_ms;
        sum_topology += gm.cost.topology_ms;
        sum_direct += gm.cost.direct_ms;
        sum_direct_lane += direct.cost.direct_ms;
        sum_cells += gm.cost.cells;
        sum_arcs += gm.cost.arcs;
        sum_seeds += gm.cost.seeds;
        sum_transported += gm.cost.transported;
        sum_conn_attempts += gm.cost.connection_attempts;
        sum_conn_failed += gm.cost.connection_failed;
        sum_transport_attempts += gm.cost.transport_attempts;
        sum_transport_failed += gm.cost.transport_failed;
        sum_transport_rejected += gm.cost.transport_rejected;
        sum_direct_nodes += gm.cost.direct_fallback;
        max_transport_tail = std::max(max_transport_tail,
                                      gm.cost.transport_tail_max);
        std::printf("case %-16s gm=% .9g v2=% .9g v0=% .9g "
                    "direct=% .9g err(gm/v2/gm-direct/direct-v0)=%.3e/%.3e/%.3e/%.3e "
                    "status=%s all_transport=%d "
                    "cells=%d arcs=%d seed/conn/trans/direct=%.3f/%.3f/%.3f/%.3f ms "
                    "transport-reject=%d tail=%.3e\n",
                    c.name.c_str(), gm.mu, v2.mu, v0.mu, direct.mu,
                    err_v2.back(), err_v0.back(), err_gm_direct.back(),
                    err_direct_v0.back(), to_string(gm.status),
                    gm.all_true_transport ? 1 : 0, gm.cost.cells, gm.cost.arcs,
                    gm.cost.seed_ms, gm.cost.connection_ms,
                    gm.cost.transport_ms, gm.cost.direct_ms,
                    gm.cost.transport_rejected, gm.cost.transport_tail_max);
    }

    std::printf("true GM whole-epoch diagnostic: lane=%s, n_r=%d, cases=%zu\n",
                lane_name(lane), n_r, cases.size());
    std::printf("GM status OK %d/%zu (value-only %d/%d, value+5Jac %d/%d); "
                "all-transport %d/%d value+5Jac; V2 status OK %d/%zu; "
                "status changes %d\n", gm_ok, cases.size(), gm_ok_value,
                value_cases, gm_ok_jac, jac_cases, gm_true_jac, jac_cases,
                v2_ok, cases.size(), status_changes);
    std::printf("mu rel error vs V2 median/p90/max %.3e %.3e %.3e\n",
                pct(err_v2, .5), pct(err_v2, .9), pct(err_v2, 1.0));
    std::printf("mu rel error vs V0 median/p90/max %.3e %.3e %.3e\n",
                pct(err_v0, .5), pct(err_v0, .9), pct(err_v0, 1.0));
    std::printf("GM-direct rel median/p90/max %.3e %.3e %.3e; direct-v0 %.3e %.3e %.3e\n",
                pct(err_gm_direct, .5), pct(err_gm_direct, .9),
                pct(err_gm_direct, 1.0), pct(err_direct_v0, .5),
                pct(err_direct_v0, .9), pct(err_direct_v0, 1.0));
    std::printf("F0/Fhalf GM-v0 worst %.3e / %.3e; direct-v0 %.3e / %.3e\n",
                worst_f0, worst_fh, worst_f0_direct, worst_fh_direct);
    std::printf("whole-epoch value-only cases=%d GM status OK %d/%d; "
                "GM/V2-value median/p90/max %.3e %.3e %.3e\n", value_cases,
                value_gm_ok, value_cases, pct(err_value_v2, .5),
                pct(err_value_v2, .9), pct(err_value_v2, 1.0));
    std::printf("whole-epoch value+5Jac cases=%d (GM status is the line above's full blend status)\n",
                jac_cases);
    std::printf("GM value+5Jac FD diagnostic cases=%zu wall median/p90 %.4f %.4f ms; "
                "same-case GM value median/p90 %.4f %.4f ms; "
                "Jac/value ratio %.3fx; gradient rel vs V2 median/p90/max %.3e %.3e %.3e\n",
                tgm_jac.size(), pct(tgm_jac, .5), pct(tgm_jac, .9),
                pct(tgm_jac_base, .5), pct(tgm_jac_base, .9),
                pct(tgm_jac, .5) / std::max(pct(tgm_jac_base, .5), 1e-300),
                pct(err_gm_jac, .5), pct(err_gm_jac, .9),
                pct(err_gm_jac, 1.0));
    std::printf("wall GM median/p90 %.4f %.4f ms; V2 median/p90 %.4f %.4f ms; "
                "V2/GM median %.3fx\n", pct(tgm, .5), pct(tgm, .9),
                pct(tv2, .5), pct(tv2, .9), pct(tv2, .5) / pct(tgm, .5));
    std::printf("wall value-only GM/V2 median/p90 %.4f/%.4f %.4f/%.4f ms; "
                "direct-GM median %.4f ms\n", pct(tgm_value, .5),
                pct(tgm_value, .9), pct(tv2_value, .5), pct(tv2_value, .9),
                pct(tgm_direct, .5));
    std::printf("cost sums seed %.3f ms, connection %.3f ms, transport %.3f ms, "
                "arc %.3f ms, topology %.3f ms, GM-direct fallback %.3f ms, "
                "direct-lane %.3f ms\n", sum_seed, sum_conn, sum_transport,
                sum_arc, sum_topology, sum_direct, sum_direct_lane);
    std::printf("transport rejected %d, max full/(full-2) observable tail %.3e\n",
                sum_transport_rejected, max_transport_tail);
    std::printf("connection attempts/failed %d/%d; transport attempts/failed "
                "%d/%d\n", sum_conn_attempts, sum_conn_failed,
                sum_transport_attempts, sum_transport_failed);
    if (angular_n > 0) {
        std::printf("independent angular reference n_theta=%d: direct/ref "
                    "median/p90/max %.3e %.3e %.3e; ref/v0 %.3e %.3e %.3e; "
                    "wall %.3f ms total\n", angular_n,
                    pct(err_direct_angular, .5), pct(err_direct_angular, .9),
                    pct(err_direct_angular, 1.0), pct(err_angular_v0, .5),
                    pct(err_angular_v0, .9), pct(err_angular_v0, 1.0),
                    sum_angular_ms);
    }
    std::printf("counts cells %d arcs %d physical seeds %d transported nodes %d "
                "direct fallback nodes %d\n", sum_cells, sum_arcs, sum_seeds,
                sum_transported, sum_direct_nodes);
    return 0;
}
