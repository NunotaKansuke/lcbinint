// Timing A/B for chart_p4: two cubic Aberth solves vs the legacy degree-6
// generic solve.  The factorization is valid for arbitrary physical
// parameters; the environment switch is only for this isolated benchmark.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/radial_events.hpp"

using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::re_detail;
using clk = std::chrono::steady_clock;

struct C { LensParams p; std::string name; };

double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double factor_run(const PrimaryFrame& pf) {
    const auto roots = chart_p4_factor_roots(pf);
    double s = 0.0;
    for (double x : roots) s += x;
    return s;
}

double generic_run(const PrimaryFrame& pf) {
    const auto fam = p_coeffs_in_R((__float128)pf.a, (__float128)pf.m0,
                                   (__float128)pf.X, (__float128)pf.Y,
                                   (__float128)pf.rho);
    int deg = fam.p[4].deg;
    while (deg > 0 && fabsq(fam.p[4].c[deg]) == 0) --deg;
    if (deg <= 0) return 0.0;
    std::vector<double> desc(deg + 1);
    for (int i = 0; i <= deg; ++i) desc[i] = (double)fam.p[4].c[deg - i];
    const auto roots = positive_real_roots(aberth<double>(desc.data(), deg, 200), 1e-8, 1e-9);
    double s = 0.0;
    for (double x : roots) s += x;
    return s;
}

double pct(std::vector<double> x, double f) {
    if (x.empty()) return 0.0;
    std::sort(x.begin(), x.end());
    const size_t i = std::min(x.size() - 1, (size_t)std::floor(f * (x.size() - 1)));
    return x[i];
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    const int reps = argc > 2 ? std::max(1, std::atoi(argv[2])) : 10;
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }
    std::vector<C> cases;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        double xs, ys, rho, q, a, u, tj;
        int bary;
        std::string name;
        if (ls >> xs >> ys >> rho >> q >> a >> bary >> u >> tj >> name)
            cases.push_back({LensParams{xs, ys, rho, q, a, (bool)bary}, name});
    }
    std::vector<double> tf, tg, tf_app, tg_app;
    double sf = 0.0, sg = 0.0;
    for (const auto& c : cases) {
        const PrimaryFrame pf = PrimaryFrame::from(c.p);
        double bf = 1e300, bg = 1e300;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now(); sf += factor_run(pf); auto t1 = clk::now();
            auto t2 = clk::now(); sg += generic_run(pf); auto t3 = clk::now();
            bf = std::min(bf, ms(t0, t1));
            bg = std::min(bg, ms(t2, t3));
        }
        tf.push_back(bf); tg.push_back(bg);
        if (pf.rho >= std::fabs(pf.Y)) {
            tf_app.push_back(bf);
            tg_app.push_back(bg);
        }
    }
    std::printf("chart_p4 cubic factor benchmark\n");
    std::printf("cases                 %zu  reps=%d\n", cases.size(), reps);
    std::printf("factor median/p90/p99  %.4f %.4f %.4f ms\n",
                pct(tf,.50),pct(tf,.90),pct(tf,.99));
    std::printf("generic median/p90/p99 %.4f %.4f %.4f ms\n",
                pct(tg,.50),pct(tg,.90),pct(tg,.99));
    std::printf("generic/factor median  %.3fx\n", pct(tg,.50) / std::max(1e-12,pct(tf,.50)));
    std::printf("applicable cases       %zu\n", tf_app.size());
    std::printf("factor applicable      median %.4f p90 %.4f p99 %.4f ms\n",
                pct(tf_app,.50),pct(tf_app,.90),pct(tf_app,.99));
    std::printf("generic applicable     median %.4f p90 %.4f p99 %.4f ms\n",
                pct(tg_app,.50),pct(tg_app,.90),pct(tg_app,.99));
    std::printf("generic/factor app     %.3fx\n", pct(tg_app,.50) / std::max(1e-12,pct(tf_app,.50)));
    std::printf("checksum              %.12g %.12g\n", sf, sg);
    return 0;
}
