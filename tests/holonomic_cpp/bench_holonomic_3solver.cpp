// ATPT holonomic solver -- 3-solver whole-epoch benchmark.
//
// The production question (checkpoint sec. 28.5): does regularized transport
// deliver measurable additional value over the current direct solver in real
// C++ ?  Judge on whole-epoch wall-clock + accuracy + Jacobian reliability +
// p90/p95/p99 -- NOT the kernel-level 15x/66x spike estimates.
//
//   V0  current direct solver          -- epoch_jacobian, both transports OFF
//   V1  + (m,v) root-pair transport    -- (m,v) transport (production default)
//   V2  + regularized holonomic        -- (m,v) transport + HOLO_HOLONOMIC_TRANSPORT
//                                         (deflated x-chart K-rule replaces the
//                                          64-pt angular sqrt(phi) sweep for
//                                          the F_half value + Jacobian)
//
// All three share classify_cells (D14 oracle), the fixed GC64 radial rule,
// boundary-quartic / QuarticWarm / polish path, and the internal->user chain
// rule.  Only the per-node radial-terms kernel differs.  Variants are toggled
// in-process via the holo_*_transport_override() hooks.
//
// Epoch theoretical ceiling from the profile: x = 1/(1 - 0.32 f), f = covered
// arc fraction; f = 1 -> 1.47x.  This bench measures how close V2 gets.
//
// Build: tests/holonomic_cpp/CMakeLists.txt (isolated project).
// Run:   taskset -c 0-7 ./build-holonomic-m7/bench_holonomic_3solver [cases.tsv] [reps]

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
double mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / v.size();
}

void set_variant(int v) {  // 0 = V0, 1 = V1, 2 = V2
    holo_mv_transport_override() = (v >= 1) ? 1 : 0;
    holo_holonomic_transport_override() = (v >= 2) ? 1 : 0;
}

double grad_reldiff(const EpochJacobian& a, const EpochJacobian& b, int* jw = nullptr) {
    // Relative-or-absolute: on-axis geometries have grad_mu components that are
    // exactly zero by symmetry and come back as ~1e-12 rounding noise in both
    // variants; a bare relative metric turns that nothing into a huge ratio.
    // Floor the denominator at 1e-6 * (|mu| + 1) -- a genuine grad component is
    // O(mu / parameter-scale), so anything below that floor is numerically zero.
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

}  // namespace

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
              c.t_jac_incumbent >> c.name))
            continue;
        cases.push_back(c);
    }
    std::fprintf(stderr, "loaded %zu bench cases, reps=%d\n", cases.size(), reps);

    std::vector<double> t0, t1, t2;
    std::vector<double> rp0, rp2;             // radial pass only (topo hoisted)
    std::vector<double> r10, r20;             // per-case speedup V0/V1, V0/V2
    std::vector<double> mupar1, mupar2;       // |dmu/mu| vs V0
    std::vector<double> gpar1, gpar2;         // worst |dgrad/grad| vs V0
    int ok0 = 0, ok1 = 0, ok2 = 0;
    int chg1 = 0, chg2 = 0;
    std::string worst_name;
    double worst_mu2 = -1.0;
    std::string worst_gname;
    double worst_g2 = -1.0;
    int worst_gj = -1;
    double worst_g_v0 = 0, worst_g_v2 = 0, worst_g_mu = 0, worst_g_rho = 0,
           worst_g_q = 0;
    double checksum = 0.0;

    for (const auto& c : cases) {
        LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        const double u = c.u;

        set_variant(0);
        EpochJacobian e0 = epoch_jacobian(p, u, 64, false);
        set_variant(1);
        EpochJacobian e1 = epoch_jacobian(p, u, 64, false);
        set_variant(2);
        EpochJacobian e2 = epoch_jacobian(p, u, 64, false);
        checksum += e0.mu + e1.mu + e2.mu;

        ok0 += (e0.status == Status::OK);
        ok1 += (e1.status == Status::OK);
        ok2 += (e2.status == Status::OK);
        chg1 += (e1.status != e0.status);
        chg2 += (e2.status != e0.status);

        double base = std::fabs(e0.mu) > 0.0 ? std::fabs(e0.mu) : 1.0;
        mupar1.push_back(std::fabs(e1.mu - e0.mu) / base);
        double mp2 = std::fabs(e2.mu - e0.mu) / base;
        mupar2.push_back(mp2);
        gpar1.push_back(grad_reldiff(e0, e1));
        int gj2 = -1;
        double gp2 = grad_reldiff(e0, e2, &gj2);
        gpar2.push_back(gp2);
        if (mp2 > worst_mu2) { worst_mu2 = mp2; worst_name = c.name; }
        if (gp2 > worst_g2) {
            worst_g2 = gp2; worst_gname = c.name; worst_gj = gj2;
            worst_g_v0 = e0.grad_mu[gj2]; worst_g_v2 = e2.grad_mu[gj2];
            worst_g_mu = e0.mu; worst_g_rho = c.rho; worst_g_q = c.q;
        }

        double b0 = 1e30, b1 = 1e30, b2 = 1e30;
        for (int r = 0; r < reps; ++r) {
            set_variant(0);
            auto ta = clk::now();
            auto v0 = epoch_jacobian(p, u, 64, false);
            auto tb = clk::now();
            set_variant(1);
            auto v1 = epoch_jacobian(p, u, 64, false);
            auto tc = clk::now();
            set_variant(2);
            auto v2 = epoch_jacobian(p, u, 64, false);
            auto tdd = clk::now();
            b0 = std::min(b0, std::chrono::duration<double, std::milli>(tb - ta).count());
            b1 = std::min(b1, std::chrono::duration<double, std::milli>(tc - tb).count());
            b2 = std::min(b2, std::chrono::duration<double, std::milli>(tdd - tc).count());
            checksum += v0.mu + v1.mu + v2.mu;
        }
        t0.push_back(b0);
        t1.push_back(b1);
        t2.push_back(b2);
        r10.push_back(b1 > 0 ? b0 / b1 : 0.0);
        r20.push_back(b2 > 0 ? b0 / b2 : 0.0);

        // radial pass in isolation: classify_cells (D14, common to all
        // variants) hoisted out of the timed region -- this is the term the
        // 1.47x profile ceiling refers to.
        PrimaryFrame pf = PrimaryFrame::from(p);
        TopologyResult topo = classify_cells(pf);
        double q0 = 1e30, q2 = 1e30;
        for (int r = 0; r < reps; ++r) {
            set_variant(0);
            auto ta = clk::now();
            auto f0 = flux_jacobian_integrate(p, 64, pf, topo);
            auto tb = clk::now();
            set_variant(2);
            auto f2 = flux_jacobian_integrate(p, 64, pf, topo);
            auto tc = clk::now();
            q0 = std::min(q0, std::chrono::duration<double, std::milli>(tb - ta).count());
            q2 = std::min(q2, std::chrono::duration<double, std::milli>(tc - tb).count());
            checksum += f0.F_half + f2.F_half;
        }
        rp0.push_back(q0);
        rp2.push_back(q2);
    }
    holo_mv_transport_override() = -1;  // restore env-var control
    holo_holonomic_transport_override() = -1;

    auto row = [](const char* tag, std::vector<double>& t) {
        std::fprintf(stderr,
            "  %-3s  median %8.4f  p90 %8.4f  p95 %8.4f  p99 %8.4f  max %8.4f  (ms)\n",
            tag, pct(t, 50), pct(t, 90), pct(t, 95), pct(t, 99),
            *std::max_element(t.begin(), t.end()));
    };
    std::fprintf(stderr, "\n== whole-epoch wall time, best-of-%d per case ==\n", reps);
    row("V0", t0);
    row("V1", t1);
    row("V2", t2);

    std::fprintf(stderr, "\n== speedup vs V0 (per-case ratio, then percentiles) ==\n");
    std::fprintf(stderr,
        "  V0/V1 (+m,v):     median %.3fx  p90 %.3fx  p95 %.3fx  p99 %.3fx  min %.3fx\n",
        pct(r10, 50), pct(r10, 90), pct(r10, 95), pct(r10, 99),
        *std::min_element(r10.begin(), r10.end()));
    std::fprintf(stderr,
        "  V0/V2 (+holo):    median %.3fx  p90 %.3fx  p95 %.3fx  p99 %.3fx  min %.3fx\n",
        pct(r20, 50), pct(r20, 90), pct(r20, 95), pct(r20, 99),
        *std::min_element(r20.begin(), r20.end()));
    std::fprintf(stderr,
        "  aggregate median-of-medians:  V0 %.4f  V1 %.4f  V2 %.4f ms"
        "  ->  V2 %.3fx\n",
        pct(t0, 50), pct(t1, 50), pct(t2, 50),
        pct(t1, 50) > 0 ? pct(t0, 50) / pct(t2, 50) : 0.0);

    std::fprintf(stderr,
        "\n== radial pass only (classify_cells hoisted; 1.47x ceiling term) ==\n");
    {
        std::vector<double> rr;
        for (size_t i = 0; i < rp0.size(); ++i)
            rr.push_back(rp2[i] > 0 ? rp0[i] / rp2[i] : 0.0);
        std::fprintf(stderr,
            "  V0 median %.4f  V2 median %.4f ms   speedup median %.3fx"
            "  p90 %.3fx  p95 %.3fx  min %.3fx\n",
            pct(rp0, 50), pct(rp2, 50),
            pct(rr, 50), pct(rr, 90), pct(rr, 95),
            *std::min_element(rr.begin(), rr.end()));
        std::fprintf(stderr,
            "  radial fraction of whole epoch (V0): %.1f%%\n",
            100.0 * pct(rp0, 50) / pct(t0, 50));
    }

    std::fprintf(stderr, "\n== accuracy vs V0 ==\n");
    std::fprintf(stderr, "  mu:      V1 max |dmu/mu| %.3e  median %.3e\n",
                 pct(mupar1, 100), pct(mupar1, 50));
    std::fprintf(stderr, "           V2 max |dmu/mu| %.3e  median %.3e   (worst: %s)\n",
                 pct(mupar2, 100), pct(mupar2, 50), worst_name.c_str());
    std::fprintf(stderr, "  grad_mu: V1 max rel %.3e  median %.3e\n",
                 pct(gpar1, 100), pct(gpar1, 50));
    std::fprintf(stderr, "           V2 max rel %.3e  median %.3e\n",
                 pct(gpar2, 100), pct(gpar2, 50));
    std::fprintf(stderr,
                 "           V2 worst: %s  j=%d  rho=%.3g q=%.3g mu=%.4g\n"
                 "             grad_mu[j]: V0 %.6e  V2 %.6e  (abs d %.2e)\n",
                 worst_gname.c_str(), worst_gj, worst_g_rho, worst_g_q,
                 worst_g_mu, worst_g_v0, worst_g_v2,
                 std::fabs(worst_g_v0 - worst_g_v2));

    std::fprintf(stderr, "\n== Jacobian reliability (status) ==\n");
    std::fprintf(stderr, "  OK count:  V0 %d/%zu   V1 %d/%zu   V2 %d/%zu\n",
                 ok0, cases.size(), ok1, cases.size(), ok2, cases.size());
    std::fprintf(stderr, "  status-change vs V0:  V1 %d   V2 %d\n", chg1, chg2);

    std::fprintf(stderr, "\n== decision-20 gate ==\n");
    std::fprintf(stderr,
        "  full-Jac median >= 2x faster AND p90/p95/p99 non-regressing.\n"
        "  V2 median %.4f ms vs V0 %.4f ms  (%.3fx)  %s\n",
        pct(t2, 50), pct(t0, 50), pct(t0, 50) / pct(t2, 50),
        pct(t0, 50) / pct(t2, 50) >= 2.0 ? "MET" : "not met (see note)");
    std::fprintf(stderr,
        "  p90 %s  p95 %s  p99 %s  (V2 <= V0 => non-regressing)\n",
        pct(t2, 90) <= pct(t0, 90) ? "ok" : "REGRESS",
        pct(t2, 95) <= pct(t0, 95) ? "ok" : "REGRESS",
        pct(t2, 99) <= pct(t0, 99) ? "ok" : "REGRESS");
    std::fprintf(stderr,
        "\n  NOTE: whole-epoch includes classify_cells (D14) which is common to\n"
        "  all 3 -- the profile ceiling (1.47x) is on the radial pass only.\n");

    std::fprintf(stderr, "\n  checksum %.6f\n", checksum);
    (void)mean;
    return 0;
}
