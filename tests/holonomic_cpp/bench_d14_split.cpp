// ATPT holonomic solver -- radial_events / D14 cost split (Phase B step 0).
//
// The phase-B direction is "keep every bit of information D14 carries (real
// events AND complex-root panel boundaries), and cut the cost of *solving*
// D14".  Before choosing between (1) compensated Ehrlich-Aberth, (2) secular
// transformation, (3) exact on-axis 6+4 factorization, and (4) all-root
// warm-start, we must know where the time goes inside radial_events -- in
// particular whether the __float128 COEFFICIENT CONSTRUCTION rivals the
// ROOT SOLVE.
//
//   taskset -c 0-7 ./build-holonomic-m7/bench_d14_split [cases.tsv] [reps]

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

struct BC {
    double xs, ys, rho, q, a;
    int bary;
    double u, t_jac;
    std::string name;
};

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 120;

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
    std::fprintf(stderr, "loaded %zu cases, reps=%d\n\n", cases.size(), reps);

    std::vector<double> T_pcoef, T_d14c, T_dsrch, T_qpol, T_qcold, T_drr, T_side,
        T_re, T_epoch;
    std::vector<double> frac_solve;
    int n_empty = 0, n_coldfallback = 0;
    double checksum = 0.0;

    struct Row { std::string name; double u, re, d14c, solve, epoch; };
    std::vector<Row> rows;

    for (const auto& c : cases) {
        LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        PrimaryFrame pf = PrimaryFrame::from(p);
        ScopedFlushDenormals _ftz;

        double b_pcoef = 1e30, b_d14c = 1e30;
        PolyFamilyR fam;
        std::vector<qf> d14;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            fam = p_coeffs_in_R((qf)pf.a, (qf)pf.m0, (qf)pf.X, (qf)pf.Y,
                                (qf)pf.rho);
            auto t1 = clk::now();
            d14 = d14_coeffs(fam);
            auto t2 = clk::now();
            b_pcoef = std::min(b_pcoef, ms(t0, t1));
            b_d14c = std::min(b_d14c, ms(t1, t2));
        }
        T_pcoef.push_back(b_pcoef);
        T_d14c.push_back(b_d14c);

        double b_dsrch = 0.0, b_qpol = 0.0, b_qcold = 0.0, b_drr = 0.0;
        if (d14.empty()) {
            ++n_empty;
        } else {
            const int deg = (int)d14.size() - 1;
            std::vector<qf> desc(deg + 1);
            for (int i = 0; i <= deg; ++i) desc[i] = d14[deg - i];
            qf dscale = 0;
            for (int i = 0; i <= deg; ++i)
                dscale = std::max(dscale, fabsq(desc[i]));
            double sscale = 1.0;
            if ((double)fabsq(desc[deg]) > 0.0 && (double)fabsq(desc[0]) > 0.0) {
                double ratio = (double)(fabsq(desc[deg]) / fabsq(desc[0]));
                sscale = std::pow(ratio, 1.0 / deg);
                if (!(sscale > 0.0) || !std::isfinite(sscale)) sscale = 1.0;
            }
            std::vector<double> descd(deg + 1);
            {
                double sp = 1.0;
                for (int i = deg; i >= 0; --i) {
                    descd[i] = (double)desc[i] * sp;
                    sp *= sscale;
                }
            }

            b_dsrch = 1e30;
            std::vector<Cplx<double>> zd;
            for (int r = 0; r < reps; ++r) {
                auto t0 = clk::now();
                zd = aberth<double>(descd.data(), deg, 200);
                auto t1 = clk::now();
                b_dsrch = std::min(b_dsrch, ms(t0, t1));
            }
            checksum += zd[0].re;

            std::vector<Cplx<qf>> seed(deg);
            for (int i = 0; i < deg; ++i)
                seed[i] = Cplx<qf>((qf)zd[i].re * (qf)sscale,
                                   (qf)zd[i].im * (qf)sscale);

            b_qpol = 1e30;
            std::vector<Cplx<qf>> roots;
            for (int r = 0; r < reps; ++r) {
                auto t0 = clk::now();
                roots = aberth<qf>(desc.data(), deg, 24, seed.data(),
                                   (qf)1e-20);
                auto t1 = clk::now();
                b_qpol = std::min(b_qpol, ms(t0, t1));
            }
            qf worst = 0;
            for (const auto& rr : roots) {
                qf res = cabs(poly_eval_c(desc.data(), deg, rr)) /
                         (dscale + (qf)1e-300);
                if (res > worst) worst = res;
            }
            if (!(worst <= (qf)1e-12)) ++n_coldfallback;
            checksum += (double)roots[0].re;

            b_qcold = 1e30;
            for (int r = 0; r < reps; ++r) {
                auto t0 = clk::now();
                auto rc = aberth<qf>(desc.data(), deg, 400, nullptr, (qf)1e-22);
                auto t1 = clk::now();
                b_qcold = std::min(b_qcold, ms(t0, t1));
                checksum += (double)rc[0].re;
            }

            auto rv = positive_real_roots(roots, 1e-8, 1e-9);
            b_drr = 1e30;
            for (int r = 0; r < reps; ++r) {
                auto t0 = clk::now();
                for (double v : rv) {
                    double R = std::sqrt(v);
                    volatile bool x = double_root_is_real(R, pf);
                    (void)x;
                }
                auto t1 = clk::now();
                b_drr = std::min(b_drr, ms(t0, t1));
            }
        }
        T_dsrch.push_back(b_dsrch);
        T_qpol.push_back(b_qpol);
        T_qcold.push_back(b_qcold);
        T_drr.push_back(b_drr);

        double b_re = 1e30, b_epoch = 1e30;
        for (int r = 0; r < reps; ++r) {
            double rmax = 0.0;
            auto t0 = clk::now();
            auto ev = radial_events(pf, &rmax);
            auto t1 = clk::now();
            b_re = std::min(b_re, ms(t0, t1));
            checksum += (double)ev.size();
        }
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            EpochJacobian ej = epoch_jacobian(p, c.u, 64, false);
            auto t1 = clk::now();
            b_epoch = std::min(b_epoch, ms(t0, t1));
            checksum += ej.mu;
        }
        T_re.push_back(b_re);
        T_epoch.push_back(b_epoch);

        double side = std::max(0.0, b_re - b_pcoef - b_d14c - b_dsrch - b_qpol -
                                        b_drr);
        T_side.push_back(side);
        if (b_re > 0) frac_solve.push_back((b_dsrch + b_qpol) / b_re);

        rows.push_back({c.name, c.u, b_re, b_d14c, b_dsrch + b_qpol, b_epoch});
    }

    auto report = [&](const char* nm, std::vector<double>& v) {
        double med = pct(v, 50), p90 = pct(v, 90), mx = pct(v, 100);
        double re_med = pct(T_re, 50), ep_med = pct(T_epoch, 50);
        std::fprintf(stderr,
                     "  %-8s  median %8.4f  p90 %8.4f  max %8.4f  ms   "
                     "( %5.1f%% of radial_events , %5.1f%% of epoch )\n",
                     nm, med, p90, mx, re_med > 0 ? 100.0 * med / re_med : 0.0,
                     ep_med > 0 ? 100.0 * med / ep_med : 0.0);
    };

    std::fprintf(stderr, "== radial_events cost split (best-of-%d per case) ==\n",
                 reps);
    report("pcoef", T_pcoef);
    report("d14c", T_d14c);
    report("dsrch", T_dsrch);
    report("qpol", T_qpol);
    report("drr", T_drr);
    report("side", T_side);
    std::fprintf(stderr, "  %-8s  median %8.4f  p90 %8.4f  max %8.4f  ms\n",
                 "--RE--", pct(T_re, 50), pct(T_re, 90), pct(T_re, 100));
    std::fprintf(stderr, "  %-8s  median %8.4f  p90 %8.4f  max %8.4f  ms\n",
                 "epoch", pct(T_epoch, 50), pct(T_epoch, 90), pct(T_epoch, 100));
    std::fprintf(stderr,
                 "\n  cold-quad (fallback) alone : median %8.4f  p90 %8.4f  "
                 "max %8.4f  ms\n",
                 pct(T_qcold, 50), pct(T_qcold, 90), pct(T_qcold, 100));
    std::fprintf(stderr, "  (dsrch+qpol) / radial_events : median %.1f%%  p90 %.1f%%\n",
                 100.0 * pct(frac_solve, 50), 100.0 * pct(frac_solve, 90));
    std::fprintf(stderr, "  d14_coeffs empty (structural degen) : %d / %zu\n",
                 n_empty, cases.size());
    std::fprintf(stderr, "  warm-polish -> cold-quad fallback   : %d / %zu\n",
                 n_coldfallback, cases.size());

    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.re > b.re; });
    std::fprintf(stderr, "\n  -- top 12 by radial_events wall time --\n");
    std::fprintf(stderr, "     %-14s %4s  %9s %9s %9s %9s\n", "case", "u",
                 "RE(ms)", "d14c", "solve", "epoch");
    for (int i = 0; i < 12 && i < (int)rows.size(); ++i)
        std::fprintf(stderr, "     %-14s %4.1f  %9.4f %9.4f %9.4f %9.4f\n",
                     rows[i].name.c_str(), rows[i].u, rows[i].re, rows[i].d14c,
                     rows[i].solve, rows[i].epoch);

    std::fprintf(stderr, "\nchecksum %.6f\n", checksum);
    return 0;
}
