// ATPT holonomic solver -- full coupled-state ODE transport benchmark.
//
// The production question (user request, 2026-09-09): does the *full* coupled
// (m, v, K, F0, F_half) ODE transport -- one RK4 march per cell that removes
// the per-radial-node quartic re-solve, the per-node angular sqrt(phi)
// quadrature, and the normal radial flux quadrature -- beat both the current
// direct solver and the current semi-holonomic (per-node v*K) path in real
// C++ ?  Judge on whole-epoch wall-clock + accuracy + Jacobian reliability +
// p90/p95/p99 + fallback rate.
//
//   V0  current direct solver       -- every transport OFF
//   V1  + (m,v) root-pair transport  -- production default (no quartic re-solve,
//                                       still per-node angular sweep for F_half)
//   V2  + semi-holonomic v*K         -- V1 + HOLO_HOLONOMIC_TRANSPORT: the
//                                       per-node 16-node GC2 deflated x-chart
//                                       K-rule replaces the angular sqrt(phi)
//                                       sweep (still a per-node quartic solve)
//   V3  + full coupled-state ODE     -- V1 + HOLO_ODE_TRANSPORT: one RK4 march
//                                       of [(m,v) per arc | F0 | F_half | dF0i |
//                                       dFhi] per cell.  No per-node quartic
//                                       solve, no per-node angular quadrature,
//                                       no normal radial flux quadrature on the
//                                       ODE path.  Fold-edge sliver -> u-sub
//                                       Gauss-Legendre tail.  Any bail -> the
//                                       incumbent per-node loop (fail closed).
//                                       Epoch rho-derivative catastrophic
//                                       cancellation (kOdeRhoCancelMax) -> the
//                                       whole epoch re-runs per-node.
//
// All four share classify_cells (D14 oracle), the fixed GC64 radial rule for
// the fallback path, and the internal->user chain rule.
//
// AUDIT: the run ends with a summed-OdeCounters block that asserts the V3
// Jacobian lane took ZERO angular-sweep nodes and ZERO per-node quartic solves
// on the ODE path -- i.e. the timing above is a real "full ODE" measurement,
// not a semi-holonomic path in disguise.
//
// Build: tests/holonomic_cpp/CMakeLists.txt (isolated project).
// Run:   taskset -c 0-7 ./build-holonomic-m7/bench_holonomic_ode [cases.tsv] [reps]

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

#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/holonomic_ode_transport.hpp"
#include "lcbinint/magnification/holonomic/holonomic_transport.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"
#include "lcbinint/magnification/holonomic/status.hpp"

using namespace lcbinint::holonomic;
using clk = std::chrono::steady_clock;

namespace {

struct BC {
    double xs, ys, rho, q, a;
    int bary;
    double u;
    double t_jac_incumbent;
    std::string name;
};

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    double frac = idx - lo;
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] * (1 - frac) + v[lo + 1] * frac;
}

// V0 = 0, V1 = 1, V2 = 2 (semi-holonomic v*K), V3 = 3 (full ODE).
void set_variant(int v) {
    holo_mv_transport_override() = (v >= 1) ? 1 : 0;
    holo_holonomic_transport_override() = (v == 2) ? 1 : 0;
    holo_ode_transport_override() = (v == 3) ? 1 : 0;
}
void restore_variant() {
    holo_mv_transport_override() = -1;
    holo_holonomic_transport_override() = -1;
    holo_ode_transport_override() = -1;
}

double grad_reldiff(const EpochJacobian& a, const EpochJacobian& b, int* jw = nullptr) {
    const double floor = 1e-6 * (std::fabs(a.mu) + 1.0);
    double worst = 0.0;
    for (int j = 0; j < 5; ++j) {
        double base = std::max({std::fabs(a.grad_mu[j]), std::fabs(b.grad_mu[j]),
                                floor});
        double d = std::fabs(a.grad_mu[j] - b.grad_mu[j]) / base;
        if (d > worst) { worst = d; if (jw) *jw = j; }
    }
    return worst;
}

struct LaneStat {
    std::vector<double> t[4];       // whole-lane wall time per case, per variant
    std::vector<double> mupar[4];   // |dmu/mu| vs V0
    std::vector<double> gpar[4];    // worst grad rel vs V0 (jac lane only)
    int ok[4] = {0, 0, 0, 0};
    int chg[4] = {0, 0, 0, 0};
    double worst_mu = -1.0;   std::string worst_mu_name;
    double worst_g = -1.0;    std::string worst_g_name; int worst_g_j = -1;
    double worst_g_v0 = 0, worst_g_v3 = 0, worst_g_rho = 0;
};

void row(const char* tag, std::vector<double>& t) {
    std::fprintf(stderr,
        "  %-3s  median %8.4f  p90 %8.4f  p95 %8.4f  p99 %8.4f  max %8.4f  (ms)\n",
        tag, pct(t, 50), pct(t, 90), pct(t, 95), pct(t, 99),
        t.empty() ? 0.0 : *std::max_element(t.begin(), t.end()));
}

void report_lane(const char* title, LaneStat& s, size_t ncases, bool jac) {
    std::fprintf(stderr, "\n======== %s ========\n", title);
    std::fprintf(stderr, "-- whole-epoch wall time (best-of-reps per case) --\n");
    for (int v = 0; v < 4; ++v) {
        char tag[4] = {'V', (char)('0' + v), 0, 0};
        row(tag, s.t[v]);
    }
    std::fprintf(stderr, "-- speedup vs V0 (median-of-medians) --\n");
    for (int v = 1; v < 4; ++v) {
        double a = pct(s.t[0], 50), b = pct(s.t[v], 50);
        std::fprintf(stderr, "   V0/V%d  %.3fx   (p90 %.3fx  p99 %.3fx)\n", v,
                     b > 0 ? a / b : 0.0,
                     pct(s.t[v], 90) > 0 ? pct(s.t[0], 90) / pct(s.t[v], 90) : 0.0,
                     pct(s.t[v], 99) > 0 ? pct(s.t[0], 99) / pct(s.t[v], 99) : 0.0);
    }
    std::fprintf(stderr, "-- accuracy vs V0 --\n");
    for (int v = 1; v < 4; ++v)
        std::fprintf(stderr,
            "   V%d  mu max|dmu/mu| %.3e  median %.3e%s\n", v,
            pct(s.mupar[v], 100), pct(s.mupar[v], 50),
            (jac ? "" : ""));
    if (jac) {
        for (int v = 1; v < 4; ++v)
            std::fprintf(stderr,
                "   V%d  grad max rel %.3e  median %.3e\n", v,
                pct(s.gpar[v], 100), pct(s.gpar[v], 50));
        std::fprintf(stderr,
            "   V3 worst grad: %s  j=%d  rho=%.3g  V0 %.6e  V3 %.6e  (abs d %.2e)\n",
            s.worst_g_name.c_str(), s.worst_g_j, s.worst_g_rho,
            s.worst_g_v0, s.worst_g_v3, std::fabs(s.worst_g_v0 - s.worst_g_v3));
    }
    std::fprintf(stderr, "   V3 worst |dmu/mu|: %s\n", s.worst_mu_name.c_str());
    std::fprintf(stderr, "-- Jacobian reliability (status) --\n");
    for (int v = 0; v < 4; ++v)
        std::fprintf(stderr, "   V%d  OK %d/%zu   status-change vs V0 %d\n", v,
                     s.ok[v], ncases, s.chg[v]);
}

}  // namespace

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 150;

    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }
    std::vector<BC> cases;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        BC c;
        if (!(ls >> c.xs >> c.ys >> c.rho >> c.q >> c.a >> c.bary >> c.u >>
              c.t_jac_incumbent >> c.name))
            continue;
        cases.push_back(c);
    }
    std::fprintf(stderr, "loaded %zu bench cases, reps=%d\n", cases.size(), reps);

    LaneStat jac, val;
    std::vector<double> rp0, rp3;   // jac radial pass only (topo hoisted): V0, V3
    double checksum = 0.0;

    for (const auto& c : cases) {
        LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        const double u = c.u;

        // ---- correctness snapshot (single call per variant) ----
        EpochJacobian ej[4];
        EpochValue ev[4];
        for (int v = 0; v < 4; ++v) {
            set_variant(v);
            ej[v] = epoch_jacobian(p, u, 64, false);
            ev[v] = epoch_value(p, u, 64, false);
        }
        checksum += ej[0].mu + ej[3].mu + ev[0].mu + ev[3].mu;

        auto acc = [&](LaneStat& s, int v, double mu_v, double mu0, Status st,
                       Status st0, const EpochJacobian* g0, const EpochJacobian* gv) {
            s.ok[v] += (st == Status::OK);
            s.chg[v] += (st != st0);
            double base = std::fabs(mu0) > 0.0 ? std::fabs(mu0) : 1.0;
            double dm = std::fabs(mu_v - mu0) / base;
            s.mupar[v].push_back(dm);
            if (v == 3 && dm > s.worst_mu) { s.worst_mu = dm; s.worst_mu_name = c.name; }
            if (g0 && gv) {
                int jj = -1;
                double gd = grad_reldiff(*g0, *gv, &jj);
                s.gpar[v].push_back(gd);
                if (v == 3 && gd > s.worst_g) {
                    s.worst_g = gd; s.worst_g_name = c.name; s.worst_g_j = jj;
                    s.worst_g_v0 = g0->grad_mu[jj]; s.worst_g_v3 = gv->grad_mu[jj];
                    s.worst_g_rho = c.rho;
                }
            }
        };
        for (int v = 1; v < 4; ++v) {
            acc(jac, v, ej[v].mu, ej[0].mu, ej[v].status, ej[0].status, &ej[0], &ej[v]);
            acc(val, v, ev[v].mu, ev[0].mu, ev[v].status, ev[0].status, nullptr, nullptr);
        }
        jac.ok[0] += (ej[0].status == Status::OK);
        val.ok[0] += (ev[0].status == Status::OK);

        // ---- timing: value+5-Jac lane ----
        double bj[4] = {1e30, 1e30, 1e30, 1e30};
        for (int r = 0; r < reps; ++r)
            for (int v = 0; v < 4; ++v) {
                set_variant(v);
                auto ta = clk::now();
                auto e = epoch_jacobian(p, u, 64, false);
                auto tb = clk::now();
                bj[v] = std::min(bj[v],
                    std::chrono::duration<double, std::milli>(tb - ta).count());
                checksum += e.mu;
            }
        for (int v = 0; v < 4; ++v) jac.t[v].push_back(bj[v]);

        // ---- timing: value-only lane ----
        double bv[4] = {1e30, 1e30, 1e30, 1e30};
        for (int r = 0; r < reps; ++r)
            for (int v = 0; v < 4; ++v) {
                set_variant(v);
                auto ta = clk::now();
                auto e = epoch_value(p, u, 64, false);
                auto tb = clk::now();
                bv[v] = std::min(bv[v],
                    std::chrono::duration<double, std::milli>(tb - ta).count());
                checksum += e.mu;
            }
        for (int v = 0; v < 4; ++v) val.t[v].push_back(bv[v]);

        // ---- jac radial pass only: classify_cells (D14) hoisted ----
        PrimaryFrame pf = PrimaryFrame::from(p);
        TopologyResult topo = classify_cells(pf);
        double q0 = 1e30, q3 = 1e30;
        for (int r = 0; r < reps; ++r) {
            set_variant(0);
            auto ta = clk::now();
            auto f0 = flux_jacobian_integrate_gated(p, 64, pf, topo);
            auto tb = clk::now();
            set_variant(3);
            auto f3 = flux_jacobian_integrate_gated(p, 64, pf, topo);
            auto tc = clk::now();
            q0 = std::min(q0, std::chrono::duration<double, std::milli>(tb - ta).count());
            q3 = std::min(q3, std::chrono::duration<double, std::milli>(tc - tb).count());
            checksum += f0.F_half + f3.F_half;
        }
        rp0.push_back(q0);
        rp3.push_back(q3);
    }
    restore_variant();

    report_lane("value + 5-Jacobian lane (epoch_jacobian)", jac, cases.size(), true);
    report_lane("value-only lane (epoch_value)", val, cases.size(), false);

    std::fprintf(stderr,
        "\n-- jac radial pass only (classify_cells hoisted; 1.47x ceiling term) --\n");
    {
        std::vector<double> rr;
        for (size_t i = 0; i < rp0.size(); ++i)
            rr.push_back(rp3[i] > 0 ? rp0[i] / rp3[i] : 0.0);
        std::fprintf(stderr,
            "   V0 median %.4f  V3 median %.4f ms   V0/V3 median %.3fx"
            "  p90 %.3fx  p95 %.3fx  min %.3fx\n",
            pct(rp0, 50), pct(rp3, 50),
            pct(rr, 50), pct(rr, 90), pct(rr, 95),
            rr.empty() ? 0.0 : *std::min_element(rr.begin(), rr.end()));
        std::fprintf(stderr, "   radial fraction of whole epoch (V0): %.1f%%\n",
                     100.0 * pct(rp0, 50) / pct(jac.t[0], 50));
    }

    // ---- AUDIT: is V3 really a full-ODE measurement? ----
    std::fprintf(stderr, "\n======== full-ODE audit (V3 Jacobian lane) ========\n");
    OdeCounters sum;
    sum.reset();
    long n_epochs = 0;
    for (const auto& c : cases) {
        LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        set_variant(3);
        ode_counters().reset();
        auto e = epoch_jacobian(p, c.u, 64, false);
        checksum += e.mu;
        const OdeCounters& cc = ode_counters();
        ++n_epochs;
        sum.cells_seen += cc.cells_seen;
        sum.cells_ode += cc.cells_ode;
        sum.cells_fallback += cc.cells_fallback;
        sum.seed_quartic_solves += cc.seed_quartic_solves;
        sum.rk4_steps += cc.rk4_steps;
        sum.krule_gc2_evals += cc.krule_gc2_evals;
        sum.angular_sweep_nodes += cc.angular_sweep_nodes;
        sum.per_node_quartic_solves += cc.per_node_quartic_solves;
        sum.regime1 += cc.regime1;
        sum.regime2 += cc.regime2;
        sum.fb_seed_notarcs += cc.fb_seed_notarcs;
        sum.fb_seed_polish += cc.fb_seed_polish;
        sum.fb_seed_tmax += cc.fb_seed_tmax;
        sum.fb_route_det += cc.fb_route_det;
        sum.fb_march_mvdet += cc.fb_march_mvdet;
        sum.fb_march_vfloor += cc.fb_march_vfloor;
        sum.fb_march_rpdr += cc.fb_march_rpdr;
        sum.fb_march_krule += cc.fb_march_krule;
        sum.fb_march_dtheta += cc.fb_march_dtheta;
        sum.fb_march_steps += cc.fb_march_steps;
        sum.fb_nonfinite += cc.fb_nonfinite;
        sum.fb_march_tail += cc.fb_march_tail;
        sum.fold_tails += cc.fold_tails;
        sum.rho_cancel_gated += cc.rho_cancel_gated;
        sum.jet_eligible_cells += cc.jet_eligible_cells;
        sum.jet_cells += cc.jet_cells;
        sum.jet_seed_krule_evals += cc.jet_seed_krule_evals;
        sum.jet_rhs_evals += cc.jet_rhs_evals;
        sum.jet_demote_decay += cc.jet_demote_decay;
        sum.jet_demote_seed += cc.jet_demote_seed;
    }
    restore_variant();

    std::fprintf(stderr, "  kArcs cells offered        %ld\n", sum.cells_seen);
    std::fprintf(stderr, "  cells completed on ODE     %ld  (%.1f%%)\n",
                 sum.cells_ode,
                 sum.cells_seen ? 100.0 * sum.cells_ode / sum.cells_seen : 0.0);
    std::fprintf(stderr, "  cells fallback -> per-node %ld  (%.1f%%)\n",
                 sum.cells_fallback,
                 sum.cells_seen ? 100.0 * sum.cells_fallback / sum.cells_seen : 0.0);
    std::fprintf(stderr, "  epochs rho-cancel GATED    %ld / %ld  (%.1f%%; ODE pass"
                 " discarded, whole epoch re-run per-node)\n",
                 sum.rho_cancel_gated, n_epochs,
                 n_epochs ? 100.0 * sum.rho_cancel_gated / n_epochs : 0.0);
    std::fprintf(stderr, "  seed quartic solves        %ld  (1 per ODE cell seed)\n",
                 sum.seed_quartic_solves);
    std::fprintf(stderr, "  RK4 steps                  %ld\n", sum.rk4_steps);
    std::fprintf(stderr, "  fold tails (u-sub GL)      %ld\n", sum.fold_tails);
    std::fprintf(stderr, "  K-rule 16-node GC2 evals   %ld  (on-march, not per radial node)\n",
                 sum.krule_gc2_evals);
    std::fprintf(stderr, "  regime1 (K) / regime2 (jet) %ld / %ld\n",
                 sum.regime1, sum.regime2);
    std::fprintf(stderr, "  jet: eligible cells %ld  active %ld  demote(decay/seed) %ld/%ld\n",
                 sum.jet_eligible_cells, sum.jet_cells,
                 sum.jet_demote_decay, sum.jet_demote_seed);
    std::fprintf(stderr, "  jet: seed K-rule evals %ld  Horner RHS arc-evals %ld\n",
                 sum.jet_seed_krule_evals, sum.jet_rhs_evals);
    std::fprintf(stderr, "  K-rule evals NOT from jet seeds: %ld\n",
                 sum.krule_gc2_evals - sum.jet_seed_krule_evals);
    std::fprintf(stderr, "  >>> angular_sweep_nodes    %ld   (MUST be 0)\n",
                 sum.angular_sweep_nodes);
    std::fprintf(stderr, "  >>> per_node_quartic_solv  %ld   (MUST be 0)\n",
                 sum.per_node_quartic_solves);
    std::fprintf(stderr, "  fallback breakdown: seed(notarcs/polish/tmax)=%ld/%ld/%ld"
                 "  route_det=%ld  march(mvdet/vfloor/rpdr/krule/dtheta/steps/tail)"
                 "=%ld/%ld/%ld/%ld/%ld/%ld/%ld  nonfinite=%ld\n",
                 sum.fb_seed_notarcs, sum.fb_seed_polish, sum.fb_seed_tmax,
                 sum.fb_route_det, sum.fb_march_mvdet, sum.fb_march_vfloor,
                 sum.fb_march_rpdr, sum.fb_march_krule, sum.fb_march_dtheta,
                 sum.fb_march_steps, sum.fb_march_tail, sum.fb_nonfinite);

    const bool audit_ok =
        (sum.angular_sweep_nodes == 0 && sum.per_node_quartic_solves == 0);
    std::fprintf(stderr, "\n  AUDIT %s -- V3 timing is %s a full-ODE measurement.\n",
                 audit_ok ? "PASS" : "FAIL",
                 audit_ok ? "" : "NOT");

    std::fprintf(stderr, "\n  checksum %.6f\n", checksum);
    return audit_ok ? 0 : 1;
}
