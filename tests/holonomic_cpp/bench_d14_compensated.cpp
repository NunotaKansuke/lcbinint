// ATPT holonomic solver -- Phase B1: compensated (double-double) D14 polish.
//
// Direct A/B of re_detail::solve_d14(desc, deg, compensated):
//   compensated == false : the retained __float128 warm polish (legacy)
//   compensated == true  : double-double (EFT) warm polish, __float128 only
//                          for a residual-gate failure
//
// Per case (108 (config,u) points + the 7 M7 reference configs):
//   * PARITY: positive-real root set, complex-Re>0 root set, and the
//     per-root __float128 residual must match the legacy path.
//   * TIMING: best-of-N wall time of the whole solve_d14 call, each way.
//   * TIER histogram for the compensated path.
//
//   taskset -c 0-7 ./build-holonomic-m7/bench_d14_compensated [cases.tsv] [reps]

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

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/radial_events.hpp"

using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::re_detail;
using clk = std::chrono::steady_clock;
using qf = __float128;

static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    double frac = idx - lo;
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] * (1 - frac) + v[lo + 1] * frac;
}
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static std::vector<double> pos_real(const std::vector<Cplx<qf>>& z) {
    return positive_real_roots(z, 1e-8, 1e-9);
}
static std::vector<double> cplx_re(const std::vector<Cplx<qf>>& z) {
    std::vector<double> v;
    for (const auto& r : z) {
        double re = (double)r.re, im = (double)r.im;
        if (re > 0.0 && std::fabs(im) >= 1e-8 * (1.0 + std::fabs(re)))
            v.push_back(re);
    }
    std::sort(v.begin(), v.end());
    std::vector<double> d;
    for (double x : v)
        if (d.empty() || x - d.back() > 1e-9) d.push_back(x);
    return d;
}
static double max_set_diff(std::vector<double> a, std::vector<double> b) {
    if (a.size() != b.size()) return 1e9;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double rel = std::fabs(a[i] - b[i]) / (std::fabs(b[i]) + 1e-30);
        m = std::max(m, rel);
    }
    return m;
}

struct BC {
    double xs, ys, rho, q, a;
    int bary;
    double u, t_jac;
    std::string name;
};

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 200;

    std::vector<BC> cases;
    {
        std::ifstream in(path);
        if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            BC c;
            if (ls >> c.xs >> c.ys >> c.rho >> c.q >> c.a >> c.bary >> c.u >>
                c.t_jac >> c.name)
                cases.push_back(c);
        }
    }
    struct RC { double xs, ys, rho, q, a; const char* nm; };
    // The 7 M7 reference configs (xs ys rho q a), evidence/holonomic/m7_reference.tsv.
    const RC refs[] = {
        {1.15, 0.7142857142857143, 0.35, 0.22, 0.05, "benign"},
        {0.55, 0.5555555555555556, 0.4, -0.05, 0.09, "close"},
        {0.9, 0.7692307692307692, 0.05, 0.02, 0.05, "resonant"},
        {1.2, 0.6666666666666666, 0.2, 0.14285714285714285, 0.125, "plan15"},
        {2.5, 0.9990009990009991, 1.4, 0.1, 0.03, "wide-planet"},
        {0.7, 0.8, 0.9, -0.3, 0.05, "cusp"},
        {1.0, 0.7692307692307692, 0.1, 0.02, 0.005, "tiny-rho"},
    };
    for (const auto& r : refs)
        cases.push_back({r.xs, r.ys, r.rho, r.q, r.a, 1, 0.0, 0.0, r.nm});

    std::fprintf(stderr, "loaded %zu cases (%d bench + 7 ref), reps=%d\n\n",
                 cases.size(), (int)cases.size() - 7, reps);

    std::vector<double> T_legacy, T_comp;
    std::vector<double> T_re, T_epoch;
    int n_solved = 0, n_parity_fail = 0;
    int tier_hist[4] = {0, 0, 0, 0};
    int legacy_cold = 0, comp_escalate = 0, comp_cold = 0;
    double worst_root_reldiff = 0.0;
    qf worst_res_legacy = 0, worst_res_comp = 0;
    double checksum = 0.0;

    struct Bad { std::string nm; double u; double rd; const char* what; };
    std::vector<Bad> bad;

    for (const auto& c : cases) {
        LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        PrimaryFrame pf = PrimaryFrame::from(p);
        ScopedFlushDenormals _ftz;

        PolyFamilyR fam = p_coeffs_in_R((qf)pf.a, (qf)pf.m0, (qf)pf.X, (qf)pf.Y,
                                        (qf)pf.rho);
        std::vector<qf> d14 = d14_coeffs(fam);
        if (d14.empty()) continue;
        ++n_solved;
        const int deg = (int)d14.size() - 1;
        std::vector<qf> desc(deg + 1);
        for (int i = 0; i <= deg; ++i) desc[i] = d14[deg - i];

        D14Solve sl = solve_d14(desc, deg, false);
        D14Solve sc = solve_d14(desc, deg, true);

        if (sl.tier == 2 || sl.tier == -1) ++legacy_cold;
        tier_hist[std::max(0, sc.tier)]++;
        if (sc.tier == 1) ++comp_escalate;
        if (sc.tier == 2) ++comp_cold;

        auto prl = pos_real(sl.roots), prc = pos_real(sc.roots);
        auto crl = cplx_re(sl.roots), crc = cplx_re(sc.roots);
        double d1 = max_set_diff(prc, prl), d2 = max_set_diff(crc, crl);
        worst_root_reldiff = std::max({worst_root_reldiff, d1, d2});
        worst_res_legacy = std::max(worst_res_legacy, sl.worst_res);
        worst_res_comp = std::max(worst_res_comp, sc.worst_res);

        bool ok = true;
        if (d1 > 1e-7) { ok = false; bad.push_back({c.name, c.u, d1, "pos-real root set"}); }
        if (d2 > 1e-6) { ok = false; bad.push_back({c.name, c.u, d2, "complex-Re root set"}); }
        if ((double)sc.worst_res > 10.0 * (double)sl.worst_res + 1e-13) {
            ok = false;
            bad.push_back({c.name, c.u, (double)sc.worst_res, "residual regressed"});
        }
        if (!ok) ++n_parity_fail;

        double bl = 1e30, bc = 1e30;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            auto x = solve_d14(desc, deg, false);
            auto t1 = clk::now();
            bl = std::min(bl, ms(t0, t1));
            checksum += (double)x.roots[0].re;
        }
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            auto x = solve_d14(desc, deg, true);
            auto t1 = clk::now();
            bc = std::min(bc, ms(t0, t1));
            checksum += (double)x.roots[0].re;
        }
        T_legacy.push_back(bl);
        T_comp.push_back(bc);

        // whole radial_events + full value+5-Jac epoch, compiled default
        // solve path (set HOLO_D14_LEGACY_SOLVE=1 to A/B the aggregate).
        double bre = 1e30, bep = 1e30;
        for (int r = 0; r < reps; ++r) {
            double rmax = 0.0;
            auto t0 = clk::now();
            auto ev = radial_events(pf, &rmax);
            auto t1 = clk::now();
            bre = std::min(bre, ms(t0, t1));
            checksum += (double)ev.size();
        }
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            EpochJacobian ej = epoch_jacobian(p, c.u, 64, false);
            auto t1 = clk::now();
            bep = std::min(bep, ms(t0, t1));
            checksum += ej.mu;
        }
        T_re.push_back(bre);
        T_epoch.push_back(bep);
    }

    std::fprintf(stderr, "== solve_d14  A/B  (best-of-%d per case) ==\n", reps);
    std::fprintf(stderr, "  cases with a D14 : %d\n", n_solved);
    std::fprintf(stderr,
                 "  legacy (__float128 warm)  : median %8.4f  p90 %8.4f  "
                 "p99 %8.4f  max %8.4f  ms\n",
                 pct(T_legacy, 50), pct(T_legacy, 90), pct(T_legacy, 99),
                 pct(T_legacy, 100));
    std::fprintf(stderr,
                 "  compensated (double-double): median %8.4f  p90 %8.4f  "
                 "p99 %8.4f  max %8.4f  ms\n",
                 pct(T_comp, 50), pct(T_comp, 90), pct(T_comp, 99),
                 pct(T_comp, 100));
    double sp_med = pct(T_legacy, 50) / std::max(1e-12, pct(T_comp, 50));
    double sp_p90 = pct(T_legacy, 90) / std::max(1e-12, pct(T_comp, 90));
    std::fprintf(stderr, "  speedup   median %.2fx   p90 %.2fx\n", sp_med, sp_p90);
    std::fprintf(stderr,
                 "\n  compensated tier histogram : "
                 "dd-sufficed %d   qf-escalate %d   cold-quad %d\n",
                 tier_hist[0], comp_escalate, comp_cold);
    std::fprintf(stderr, "  legacy cold-quad fallbacks : %d\n", legacy_cold);
    std::fprintf(stderr, "\n  PARITY vs legacy:\n");
    std::fprintf(stderr, "    worst root-set rel diff (comp vs legacy) : %.3e\n",
                 worst_root_reldiff);
    std::fprintf(stderr, "    worst __float128 residual  legacy %.3e   comp %.3e\n",
                 (double)worst_res_legacy, (double)worst_res_comp);
    std::fprintf(stderr, "    parity failures : %d / %d\n", n_parity_fail, n_solved);
    for (const auto& b : bad)
        std::fprintf(stderr, "      FAIL %-14s u=%.1f  %s  (%.3e)\n",
                     b.nm.c_str(), b.u, b.what, b.rd);

    std::fprintf(stderr,
                 "\n  whole radial_events (compiled default path) : "
                 "median %8.4f  p90 %8.4f  p99 %8.4f  max %8.4f  ms\n",
                 pct(T_re, 50), pct(T_re, 90), pct(T_re, 99), pct(T_re, 100));
    std::fprintf(stderr,
                 "  full value+5-Jac epoch                      : "
                 "median %8.4f  p90 %8.4f  p99 %8.4f  max %8.4f  ms\n",
                 pct(T_epoch, 50), pct(T_epoch, 90), pct(T_epoch, 99),
                 pct(T_epoch, 100));

    std::fprintf(stderr, "\nchecksum %.6f\n", checksum);
    return n_parity_fail == 0 ? 0 : 1;
}
