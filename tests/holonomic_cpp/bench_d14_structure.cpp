// ATPT holonomic solver -- D14 C3/G4/Z3 block-form evaluation A/B.
//
// Two things change together behind HOLO_D14_STRUCT=1 (d14_structure.hpp):
//   (1) the expanded degree-14 coefficient vector is assembled from the
//       exact low-degree blocks instead of the degree-36 I,J expansion
//       (d14_expanded_from_struct vs re_detail::d14_coeffs);
//   (2) the Aberth polish evaluates D / D' in block form instead of
//       Horner on that expanded vector.
//
// This bench runs with HOLO_D14_STRUCT=1 exported (so solve_d14's
// use_struct gate is live) and A/Bs, per case, in ONE process:
//   base : solve_d14(desc_expansion, deg, true, nullptr, nullptr)
//   strt : solve_d14(desc_blocks,    deg, true, nullptr, &d14s)
//
//   * PARITY: positive-real root set + complex-Re>0 root set must match.
//   * TIMING: best-of-N and MEDIAN/p90/p99 of the whole solve_d14 call.
//   * TIER histogram each way.
// Run the binary twice (HOLO_D14_STRUCT unset vs =1) for the
// radial_events / full-epoch aggregate A/B.
//
//   HOLO_D14_STRUCT=1 taskset -c 0-7 ./build-holonomic-m7/bench_d14_structure \
//       /tmp/bench_cases.tsv 200

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
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(a[i] - b[i]) / (std::fabs(b[i]) + 1e-30));
    return m;
}

struct BC { double xs, ys, rho, q, a; int bary; double u, t_jac; std::string name; };

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 200;
    // Block form is now the default (checkpoint 34); HOLO_D14_STRUCT_LEGACY=1
    // reverts to expanded-Horner.  flag_on == "block form active".
    const char* e = std::getenv("HOLO_D14_STRUCT_LEGACY");
    const bool flag_on = !(e && e[0] == '1');

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

    std::fprintf(stderr, "loaded %zu cases (%d bench + 7 ref), reps=%d  "
                 "block-form=%d\n\n",
                 cases.size(), (int)cases.size() - 7, reps, (int)flag_on);
    if (!flag_on)
        std::fprintf(stderr,
                     "  NOTE: HOLO_D14_STRUCT_LEGACY=1 -> solve A/B is a no-op "
                     "(use_struct gate off).  Aggregates below are the\n"
                     "  expanded-Horner baseline; unset the var for the "
                     "block-form aggregate.\n\n");

    std::vector<double> T_base, T_strt, T_re, T_epoch;
    int n_solved = 0, n_parity_fail = 0;
    int tb[4] = {0}, tsr[4] = {0};
    int base_escal = 0, strt_escal = 0, base_cold = 0, strt_cold = 0;
    int coeff_src_fail = 0;
    double worst_root_reldiff = 0.0, worst_coeff_reldiff = 0.0;
    qf worst_res_base = 0, worst_res_strt = 0;
    double checksum = 0.0;
    struct Bad { std::string nm; double u; double rd; const char* what; };
    std::vector<Bad> bad;

    for (const auto& c : cases) {
        LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        PrimaryFrame pf = PrimaryFrame::from(p);
        ScopedFlushDenormals _ftz;

        PolyFamilyR fam = p_coeffs_in_R((qf)pf.a, (qf)pf.m0, (qf)pf.X, (qf)pf.Y,
                                        (qf)pf.rho);
        std::vector<qf> d14_exp = d14_coeffs(fam);
        D14StructQf d14s = d14_struct_build((qf)pf.a, (qf)pf.m0, (qf)pf.X,
                                            (qf)pf.Y, (qf)pf.rho);
        std::vector<qf> d14_blk = d14_expanded_from_struct(d14s);
        if (d14_exp.empty() || d14_blk.empty()) continue;
        ++n_solved;

        // coeff-source parity (block-assembled vs I,J-expansion)
        if (d14_exp.size() == d14_blk.size()) {
            qf sc = 0;
            for (qf v : d14_exp) { qf a = fabsq(v); if (a > sc) sc = a; }
            double cr = 0.0;
            for (size_t i = 0; i < d14_exp.size(); ++i)
                cr = std::max(cr, (double)(fabsq(d14_exp[i] - d14_blk[i]) /
                                           (sc + (qf)1e-300)));
            worst_coeff_reldiff = std::max(worst_coeff_reldiff, cr);
        } else {
            ++coeff_src_fail;
        }

        const int deg = (int)d14_exp.size() - 1;
        std::vector<qf> desc_e(deg + 1), desc_b(deg + 1);
        for (int i = 0; i <= deg; ++i) {
            desc_e[i] = d14_exp[deg - i];
            desc_b[i] = d14_blk[deg - i];
        }

        D14Solve sb = solve_d14(desc_e, deg, true, nullptr, nullptr);
        D14Solve ss = solve_d14(desc_b, deg, true, nullptr, &d14s);

        tb[std::max(0, sb.tier)]++;
        tsr[std::max(0, ss.tier)]++;
        if (sb.tier == 1) ++base_escal;
        if (ss.tier == 1) ++strt_escal;
        if (sb.tier == 2) ++base_cold;
        if (ss.tier == 2) ++strt_cold;

        auto prb = pos_real(sb.roots), prs = pos_real(ss.roots);
        auto crb = cplx_re(sb.roots), crs = cplx_re(ss.roots);
        double d1 = max_set_diff(prs, prb), d2 = max_set_diff(crs, crb);
        worst_root_reldiff = std::max({worst_root_reldiff, d1, d2});
        worst_res_base = std::max(worst_res_base, sb.worst_res);
        worst_res_strt = std::max(worst_res_strt, ss.worst_res);

        bool ok = true;
        if (d1 > 1e-7) { ok = false; bad.push_back({c.name, c.u, d1, "pos-real root set"}); }
        if (d2 > 1e-6) { ok = false; bad.push_back({c.name, c.u, d2, "complex-Re root set"}); }
        if ((double)ss.worst_res > 10.0 * (double)sb.worst_res + 1e-13) {
            ok = false;
            bad.push_back({c.name, c.u, (double)ss.worst_res, "residual regressed"});
        }
        if (!ok) ++n_parity_fail;

        double bb = 1e30, bs = 1e30;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            auto x = solve_d14(desc_e, deg, true, nullptr, nullptr);
            auto t1 = clk::now();
            bb = std::min(bb, ms(t0, t1));
            checksum += (double)x.roots[0].re;
        }
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            auto x = solve_d14(desc_b, deg, true, nullptr, &d14s);
            auto t1 = clk::now();
            bs = std::min(bs, ms(t0, t1));
            checksum += (double)x.roots[0].re;
        }
        T_base.push_back(bb);
        T_strt.push_back(bs);

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

    std::fprintf(stderr, "== solve_d14  A/B (base = expansion+Horner, "
                 "strt = blocks+block-eval; best-of-%d) ==\n", reps);
    std::fprintf(stderr, "  cases with a D14 : %d\n", n_solved);
    std::fprintf(stderr,
                 "  base : median %8.4f  p90 %8.4f  p99 %8.4f  max %8.4f  ms\n",
                 pct(T_base, 50), pct(T_base, 90), pct(T_base, 99), pct(T_base, 100));
    std::fprintf(stderr,
                 "  strt : median %8.4f  p90 %8.4f  p99 %8.4f  max %8.4f  ms\n",
                 pct(T_strt, 50), pct(T_strt, 90), pct(T_strt, 99), pct(T_strt, 100));
    std::fprintf(stderr, "  speedup  median %.2fx  p90 %.2fx  p99 %.2fx\n",
                 pct(T_base, 50) / std::max(1e-12, pct(T_strt, 50)),
                 pct(T_base, 90) / std::max(1e-12, pct(T_strt, 90)),
                 pct(T_base, 99) / std::max(1e-12, pct(T_strt, 99)));
    std::fprintf(stderr,
                 "\n  base tier hist : dd %d  qf-escal %d  cold %d\n",
                 tb[0], base_escal, base_cold);
    std::fprintf(stderr,
                 "  strt tier hist : dd %d  qf-escal %d  cold %d\n",
                 tsr[0], strt_escal, strt_cold);
    std::fprintf(stderr, "\n  PARITY:\n");
    std::fprintf(stderr, "    coeff vector (blocks vs expansion) worst rel : %.3e  "
                 "(deg-mismatch %d)\n", worst_coeff_reldiff, coeff_src_fail);
    std::fprintf(stderr, "    root-set rel diff (strt vs base) worst       : %.3e\n",
                 worst_root_reldiff);
    std::fprintf(stderr, "    worst __float128 residual  base %.3e  strt %.3e\n",
                 (double)worst_res_base, (double)worst_res_strt);
    std::fprintf(stderr, "    parity failures : %d / %d\n", n_parity_fail, n_solved);
    for (const auto& b : bad)
        std::fprintf(stderr, "      FAIL %-14s u=%.1f  %s  (%.3e)\n",
                     b.nm.c_str(), b.u, b.what, b.rd);

    std::fprintf(stderr,
                 "\n  whole radial_events (compiled path, block-form=%d) : "
                 "median %8.4f  p90 %8.4f  p99 %8.4f  max %8.4f  ms\n",
                 (int)flag_on, pct(T_re, 50), pct(T_re, 90), pct(T_re, 99),
                 pct(T_re, 100));
    std::fprintf(stderr,
                 "  full value+5-Jac epoch  (block-form=%d)             : "
                 "median %8.4f  p90 %8.4f  p99 %8.4f  max %8.4f  ms\n",
                 (int)flag_on, pct(T_epoch, 50), pct(T_epoch, 90),
                 pct(T_epoch, 99), pct(T_epoch, 100));

    std::fprintf(stderr, "\nchecksum %.6f\n", checksum);
    return n_parity_fail == 0 ? 0 : 1;
}
