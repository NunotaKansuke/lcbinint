// ATPT holonomic solver -- fast-planner topology vs classify_cells, full epoch.
//
// Phase A step 4 (unified-engine implementation order, checkpoint sec.0.7 /
// sec.16.8).  For each of the 108 (config, u) bench points, runs the fused
// value+5-Jacobian epoch BOTH ways in the same process:
//   base = epoch_jacobian(p, u, 64, /*use_fast=*/false)   (classify_cells / D14)
//   fast = epoch_jacobian(p, u, 64, /*use_fast=*/true )   (fast_bands + screen,
//                                                          D14 fallback)
// and reports:
//   * fast-path adoption rate (screen pass) vs classify_cells fallback;
//   * Status parity (must be 0 changes);
//   * mu / grad_mu / dmu_du agreement base-vs-fast;
//   * wall time both ways (median / p90 / p95 / p99) -- the decision-20 gate
//     is re-scored on the fast path.
//
//   taskset -c 0-7 ./build-holonomic-m7/bench_fast_topology [cases.tsv] [reps]

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
#include "lcbinint/magnification/holonomic/fast_topology.hpp"

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

static const char* st_name(Status s) {
    switch (s) {
        case Status::OK: return "OK";
        case Status::GRADIENT_UNRELIABLE: return "GRAD_UNREL";
        case Status::TOPOLOGY_UNCERTAIN: return "TOPO_UNCERT";
    }
    return "?";
}

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 200;

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
    std::fprintf(stderr, "loaded %zu bench cases, reps=%d\n\n", cases.size(),
                 reps);

    int n_fast_adopted = 0, n_fallback = 0;
    int n_status_change = 0;
    std::vector<double> mu_rel, grad_rel, dmudu_rel, t_base, t_fast, spd;
    std::vector<std::string> notes;
    double checksum = 0.0;

    for (const auto& c : cases) {
        LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        PrimaryFrame pf = PrimaryFrame::from(p);

        FastTopologyStats fts;
        {
            ScopedFlushDenormals _ftz;
            classify_cells_fast(pf, &fts);
        }
        if (fts.used_fast) ++n_fast_adopted;
        if (fts.fell_back) ++n_fallback;

        EpochJacobian base = epoch_jacobian(p, c.u, 64, false);
        EpochJacobian fast = epoch_jacobian(p, c.u, 64, true);

        if (base.status != fast.status) {
            ++n_status_change;
            char buf[192];
            std::snprintf(buf, sizeof buf,
                          "%-16s u=%.1f  STATUS %s -> %s   (%s)", c.name.c_str(),
                          c.u, st_name(base.status), st_name(fast.status),
                          fts.fell_back ? fts.reason : "fast path");
            notes.push_back(buf);
        }

        const double md = std::fabs(base.mu) > 0 ? std::fabs(base.mu) : 1.0;
        mu_rel.push_back(std::fabs(fast.mu - base.mu) / md);

        double num = 0.0, den = 0.0;
        for (int j = 0; j < 5; ++j) {
            num += (fast.grad_mu[j] - base.grad_mu[j]) *
                   (fast.grad_mu[j] - base.grad_mu[j]);
            den += base.grad_mu[j] * base.grad_mu[j];
        }
        grad_rel.push_back(den > 0 ? std::sqrt(num / den) : std::sqrt(num));

        const double dd =
            std::fabs(base.dmu_du) > 0 ? std::fabs(base.dmu_du) : 1.0;
        dmudu_rel.push_back(std::fabs(fast.dmu_du - base.dmu_du) / dd);

        checksum += fast.mu + base.mu;

        // timing
        double bb = 1e30, bf = 1e30;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            EpochJacobian e0 = epoch_jacobian(p, c.u, 64, false);
            auto t1 = clk::now();
            EpochJacobian e1 = epoch_jacobian(p, c.u, 64, true);
            auto t2 = clk::now();
            checksum += e0.mu + e1.mu;
            bb = std::min(
                bb, std::chrono::duration<double, std::milli>(t1 - t0).count());
            bf = std::min(
                bf, std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        t_base.push_back(bb);
        t_fast.push_back(bf);
        spd.push_back(c.t_jac * 1e3 / bf);
    }

    const int N = (int)cases.size();
    std::fprintf(stderr, "== fast-topology adoption (%d epochs) ==\n", N);
    std::fprintf(stderr, "  fast band set adopted (screen pass) : %d  (%.1f%%)\n",
                 n_fast_adopted, 100.0 * n_fast_adopted / N);
    std::fprintf(stderr, "  fell back to classify_cells / D14   : %d  (%.1f%%)\n",
                 n_fallback, 100.0 * n_fallback / N);
    std::fprintf(stderr, "\n== parity: fast vs classify_cells (base) ==\n");
    std::fprintf(stderr, "  Status changes            : %d  %s\n", n_status_change,
                 n_status_change == 0 ? "(PASS)" : "(FAIL)");
    std::fprintf(stderr, "  |mu_fast - mu_base|/|mu|   : median %.2e  p90 %.2e  "
                 "max %.2e\n",
                 pct(mu_rel, 50), pct(mu_rel, 90), pct(mu_rel, 100));
    std::fprintf(stderr, "  ||dgrad|| / ||grad_base||  : median %.2e  p90 %.2e  "
                 "max %.2e\n",
                 pct(grad_rel, 50), pct(grad_rel, 90), pct(grad_rel, 100));
    std::fprintf(stderr, "  |ddmu_du| / |dmu_du_base|  : median %.2e  p90 %.2e  "
                 "max %.2e\n",
                 pct(dmudu_rel, 50), pct(dmudu_rel, 90), pct(dmudu_rel, 100));

    if (!notes.empty()) {
        std::fprintf(stderr, "\n  -- status-change detail --\n");
        for (auto& s : notes) std::fprintf(stderr, "     %s\n", s.c_str());
    }

    std::fprintf(stderr,
                 "\n== wall time, best-of-%d per epoch (value + 5-Jac) ==\n",
                 reps);
    std::fprintf(stderr,
                 "  classify_cells path : median %7.3f  p90 %7.3f  p95 %7.3f  "
                 "p99 %7.3f  ms\n",
                 pct(t_base, 50), pct(t_base, 90), pct(t_base, 95),
                 pct(t_base, 99));
    std::fprintf(stderr,
                 "  fast-planner path   : median %7.3f  p90 %7.3f  p95 %7.3f  "
                 "p99 %7.3f  ms\n",
                 pct(t_fast, 50), pct(t_fast, 90), pct(t_fast, 95),
                 pct(t_fast, 99));
    std::fprintf(stderr, "  speedup vs incumbent (fast path) : median %.1fx  "
                 "min %.1fx  p10 %.1fx\n",
                 pct(spd, 50), pct(spd, 0), pct(spd, 10));
    std::fprintf(stderr,
                 "  DECISION-20 (fast path median <= 7.44 ms) : %s\n",
                 pct(t_fast, 50) <= 7.44 ? "PASS" : "FAIL");

    std::fprintf(stderr, "\nchecksum %.6f\n", checksum);
    return 0;
}
