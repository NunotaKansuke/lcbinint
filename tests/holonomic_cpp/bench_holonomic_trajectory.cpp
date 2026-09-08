// ATPT holonomic solver -- prepared-geometry trajectory reuse (Phase E) +
// the Phase B2 all-root D14 warm-start.
//
// For each bench config an N-epoch source track is synthesised (the lens
// params q, a, rho held fixed; the source swept along a straight line long
// enough to cross the config's boundary structure).  Each track is run
// three ways in the same process:
//
//   V0  epoch_jacobian(p,u,64,false)          -- cold classify_cells / D14
//                                                every epoch (today's path)
//   V1  epoch_jacobian_prepared, L2 only      -- classify_cells (the D14
//                                                authority) EVERY epoch, but
//                                                the D14 root solve is warm-
//                                                seeded by the previous
//                                                epoch's 14 roots (Phase B2).
//                                                Every completeness / residual
//                                                gate intact -> exact parity.
//   V2  epoch_jacobian_prepared, L1 + L2      -- adds the L1 verbatim cell-
//                                                plan reuse (quartic-only
//                                                sec.8 re-screen).  Kept as a
//                                                measurement of why L1 is NOT
//                                                default-safe: the re-screen
//                                                cannot see D14's complex-root
//                                                panel boundaries (the
//                                                deferred cheaper-than-D14
//                                                certificate), so near-caustic
//                                                epochs leak false reuse.
//
// Reports, aggregated over every (config, epoch):
//   * mu / grad_mu / status parity  V1,V2  vs V0  (fail-closed: never better)
//   * FALSE-REUSE audit: an L1-reused epoch whose mu disagrees with V0 by
//     > 1e-3 means the re-screen passed a stale topology -- MUST be 0
//   * L1 / L2 / L3 mix and the re-screen miss rate
//   * per-epoch wall time V0/V1/V2 (median / p90 / p99), steady state
//     (epoch 0 -- the unavoidable cold build -- excluded)
//
//   taskset -c 0-7 ./build-holonomic-m7/bench_holonomic_trajectory [cases.tsv] [reps] [N]

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
#include "lcbinint/magnification/holonomic/prepared_geometry.hpp"

using namespace lcbinint::holonomic;
using clk = std::chrono::steady_clock;

struct BC {
    double xs, ys, rho, q, a;
    int bary;
    double u, t_jac;
    std::string name;
};

static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    double frac = idx - lo;
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] * (1 - frac) + v[lo + 1] * frac;
}

static const char* st_name(Status s) { return to_string(s); }

// rank a status for the fail-closed check: OK is the "best"; anything else
// is a downgrade.  V1/V2 must never report a strictly better status than V0.
static int st_rank(Status s) { return s == Status::OK ? 0 : 1; }

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 30;
    int N = argc > 3 ? std::atoi(argv[3]) : 32;

    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }
    std::vector<BC> cases;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        BC c;
        if (!(ls >> c.xs >> c.ys >> c.rho >> c.q >> c.a >> c.bary >> c.u >>
              c.t_jac >> c.name))
            continue;
        cases.push_back(c);
    }
    std::fprintf(stderr, "loaded %zu bench cases, reps=%d, N=%d epochs/track\n\n",
                 cases.size(), reps, N);

    PreparedReuseConfig cfg_v1;
    cfg_v1.allow_topology_reuse = false;  // L1 off: no quartic-only reuse
    cfg_v1.allow_warm_d14 = true;         // L2 on: warm-seeded D14 every epoch
    cfg_v1.l2_drift = 1e18;               // always attempt the warm seed;
                                          // solve_d14 fails closed on a bad one
    PreparedReuseConfig cfg_v2 = cfg_v1;
    cfg_v2.allow_topology_reuse = true;   // + L1 verbatim cell-plan reuse

    // aggregate accumulators
    std::vector<double> mu_rel_v1, mu_rel_v2, grad_rel_v1, grad_rel_v2;
    std::vector<double> t0_ep, t1_ep, t2_ep;         // per-epoch (all epochs)
    std::vector<double> t1_ss, t2_ss;                // steady state (epoch>0)
    long n_status_worse_v1 = 0, n_status_worse_v2 = 0;
    long false_reuse_v1 = 0, false_reuse_v2 = 0;
    PreparedReuseStats agg_v1, agg_v2;
    long total_epochs = 0;
    double checksum = 0.0;

    const char* only = std::getenv("HOLO_ONLY");
    const char* only_ep = std::getenv("HOLO_ONLY_EP");
    for (const auto& c : cases) {
        if (only && c.name.find(only) == std::string::npos) continue;
        // straight source track through (xs,ys); length scaled to guarantee
        // it meets the config's boundary structure.
        const double L = std::max(8.0 * c.rho, 0.5 * c.a);
        const double dirx = 0.94868, diry = 0.31623;  // ~ (3,1)/sqrt(10)
        auto epoch_params = [&](int k) {
            double s = ((double)k / (N - 1) - 0.5) * L;
            BC e = c;
            e.xs = c.xs + s * dirx;
            e.ys = c.ys + s * diry;
            return e;
        };

        // ---- correctness pass (one rep) ----
        PreparedEpochGeometry st1{}, st2{};
        for (int k = 0; k < N; ++k) {
            BC e = epoch_params(k);
            LensParams p{e.xs, e.ys, e.rho, e.q, e.a, (bool)e.bary};

            const bool trace = std::getenv("HOLO_MV_DEBUG") &&
                               (!only_ep || std::atoi(only_ep) == k);
            if (trace) std::fprintf(stderr, "=== %s k=%d PATH=V0\n", c.name.c_str(), k);
            EpochJacobian b = epoch_jacobian(p, e.u, 64, false);
            PreparedReuseStats s1{}, s2{};
            if (trace) std::fprintf(stderr, "=== %s k=%d PATH=V1\n", c.name.c_str(), k);
            EpochJacobian r1 =
                epoch_jacobian_prepared(p, e.u, 64, st1, cfg_v1, &s1);
            if (trace) std::fprintf(stderr, "=== %s k=%d PATH=V2\n", c.name.c_str(), k);
            EpochJacobian r2 =
                epoch_jacobian_prepared(p, e.u, 64, st2, cfg_v2, &s2);
            agg_v1.l1_topology_reuse += s1.l1_topology_reuse;
            agg_v1.l2_warm_recompute += s1.l2_warm_recompute;
            agg_v1.l3_cold_recompute += s1.l3_cold_recompute;
            agg_v1.rescreen_fail += s1.rescreen_fail;
            agg_v1.warm_solve_used += s1.warm_solve_used;
            agg_v2.l1_topology_reuse += s2.l1_topology_reuse;
            agg_v2.l2_warm_recompute += s2.l2_warm_recompute;
            agg_v2.l3_cold_recompute += s2.l3_cold_recompute;
            agg_v2.rescreen_fail += s2.rescreen_fail;
            agg_v2.warm_solve_used += s2.warm_solve_used;
            total_epochs++;

            const double md = std::fabs(b.mu) > 0 ? std::fabs(b.mu) : 1.0;
            const double e1 = std::fabs(r1.mu - b.mu) / md;
            const double e2 = std::fabs(r2.mu - b.mu) / md;
            mu_rel_v1.push_back(e1);
            mu_rel_v2.push_back(e2);

            if (std::getenv("HOLO_DUMP")) {
                std::printf("DUMP\t%s\t%d\t%.17g\t%.17g\t%.17g\n",
                            c.name.c_str(), k, b.mu, r1.mu, r2.mu);
            }

            auto grad_rel = [&](const EpochJacobian& r) {
                double num = 0, den = 0;
                for (int j = 0; j < 5; ++j) {
                    num += (r.grad_mu[j] - b.grad_mu[j]) *
                           (r.grad_mu[j] - b.grad_mu[j]);
                    den += b.grad_mu[j] * b.grad_mu[j];
                }
                return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
            };
            grad_rel_v1.push_back(grad_rel(r1));
            grad_rel_v2.push_back(grad_rel(r2));

            if (st_rank(r1.status) < st_rank(b.status)) ++n_status_worse_v1;
            if (st_rank(r2.status) < st_rank(b.status)) ++n_status_worse_v2;

            if (s1.l1_topology_reuse && e1 > 1e-3) ++false_reuse_v1;
            if (s2.l1_topology_reuse && e2 > 1e-3) ++false_reuse_v2;

            checksum += b.mu + r1.mu + r2.mu;
        }

        // ---- timing pass ----
        std::vector<double> bt0(N, 1e30), bt1(N, 1e30), bt2(N, 1e30);
        for (int r = 0; r < reps; ++r) {
            PreparedEpochGeometry g1{}, g2{};
            for (int k = 0; k < N; ++k) {
                BC e = epoch_params(k);
                LensParams p{e.xs, e.ys, e.rho, e.q, e.a, (bool)e.bary};
                PreparedReuseStats s1{}, s2{};

                auto ta = clk::now();
                EpochJacobian x0 = epoch_jacobian(p, e.u, 64, false);
                auto tb = clk::now();
                EpochJacobian x1 =
                    epoch_jacobian_prepared(p, e.u, 64, g1, cfg_v1, &s1);
                auto tc = clk::now();
                EpochJacobian x2 =
                    epoch_jacobian_prepared(p, e.u, 64, g2, cfg_v2, &s2);
                auto td = clk::now();
                checksum += x0.mu + x1.mu + x2.mu;
                bt0[k] = std::min(
                    bt0[k],
                    std::chrono::duration<double, std::milli>(tb - ta).count());
                bt1[k] = std::min(
                    bt1[k],
                    std::chrono::duration<double, std::milli>(tc - tb).count());
                bt2[k] = std::min(
                    bt2[k],
                    std::chrono::duration<double, std::milli>(td - tc).count());
            }
        }
        for (int k = 0; k < N; ++k) {
            t0_ep.push_back(bt0[k]);
            t1_ep.push_back(bt1[k]);
            t2_ep.push_back(bt2[k]);
            if (k > 0) {
                t1_ss.push_back(bt1[k]);
                t2_ss.push_back(bt2[k]);
            }
        }
    }

    auto row = [](const char* tag, std::vector<double>& v) {
        std::fprintf(stderr,
                     "  %-22s median %7.4f  p90 %7.4f  p95 %7.4f  p99 %7.4f  "
                     "max %7.4f  ms\n",
                     tag, pct(v, 50), pct(v, 90), pct(v, 95), pct(v, 99),
                     pct(v, 100));
    };

    std::fprintf(stderr, "== parity vs V0 (cold classify_cells every epoch) ==\n");
    std::fprintf(stderr,
                 "  |dmu|/|mu|      V1: median %.2e p90 %.2e max %.2e\n"
                 "                 V2: median %.2e p90 %.2e max %.2e\n",
                 pct(mu_rel_v1, 50), pct(mu_rel_v1, 90), pct(mu_rel_v1, 100),
                 pct(mu_rel_v2, 50), pct(mu_rel_v2, 90), pct(mu_rel_v2, 100));
    std::fprintf(stderr,
                 "  ||dgrad||/||g|| V1: median %.2e p90 %.2e max %.2e\n"
                 "                 V2: median %.2e p90 %.2e max %.2e\n",
                 pct(grad_rel_v1, 50), pct(grad_rel_v1, 90),
                 pct(grad_rel_v1, 100), pct(grad_rel_v2, 50),
                 pct(grad_rel_v2, 90), pct(grad_rel_v2, 100));
    std::fprintf(stderr, "  status downgraded vs V0 (must be 0) : V1 %ld  V2 %ld\n",
                 n_status_worse_v1, n_status_worse_v2);
    std::fprintf(stderr,
                 "  FALSE REUSE (L1 reuse, |dmu|/|mu|>1e-3; MUST be 0) : "
                 "V1 %ld  V2 %ld\n",
                 false_reuse_v1, false_reuse_v2);

    auto mix = [&](const char* tag, const PreparedReuseStats& a) {
        const long tot = a.l1_topology_reuse + a.l2_warm_recompute +
                         a.l3_cold_recompute;
        std::fprintf(stderr,
                     "  %s  L1 %ld (%.1f%%)  L2 %ld (%.1f%%)  L3 %ld (%.1f%%)  "
                     "rescreen-miss %ld  warm-seed-used %ld\n",
                     tag, a.l1_topology_reuse,
                     tot ? 100.0 * a.l1_topology_reuse / tot : 0.0,
                     a.l2_warm_recompute,
                     tot ? 100.0 * a.l2_warm_recompute / tot : 0.0,
                     a.l3_cold_recompute,
                     tot ? 100.0 * a.l3_cold_recompute / tot : 0.0,
                     a.rescreen_fail, a.warm_solve_used);
    };
    std::fprintf(stderr, "\n== reuse mix (%ld epochs total) ==\n", total_epochs);
    mix("V1", agg_v1);
    mix("V2", agg_v2);

    std::fprintf(stderr, "\n== wall time per epoch, best-of-%d ==\n", reps);
    row("V0 cold (all epochs)", t0_ep);
    row("V1 L2 warm-D14 (all)", t1_ep);
    row("V2 L1+L2 (all)", t2_ep);
    row("V1 L2 steady state", t1_ss);
    row("V2 L1+L2 steady state", t2_ss);
    std::fprintf(stderr,
                 "  V1 steady speedup vs V0 median : %.2fx    V2 : %.2fx\n",
                 pct(t0_ep, 50) / std::max(1e-12, pct(t1_ss, 50)),
                 pct(t0_ep, 50) / std::max(1e-12, pct(t2_ss, 50)));

    std::fprintf(stderr, "\nchecksum %.6f\n", checksum);
    return (n_status_worse_v1 || n_status_worse_v2 || false_reuse_v1 ||
            false_reuse_v2)
               ? 1
               : 0;
}
