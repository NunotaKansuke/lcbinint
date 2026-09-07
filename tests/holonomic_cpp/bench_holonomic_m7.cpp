// ATPT holonomic solver (M7) -- timing harness.
//
// Times the fused value+Jacobian path (epoch_jacobian, n_r=64) over the
// same 108 (config, u) points the M0 baseline probe used
// (evidence/holonomic/baseline_M0.json, exported to a flat TSV), and
// prints the median / p90 / p95 / p99 the decision-20 gate is scored on.
//
// Build:
//   g++ -std=c++17 -O3 -march=native -funroll-loops -I src \
//       tests/holonomic_cpp/bench_holonomic_m7.cpp \
//       -o build-holonomic-m7/bench_holonomic_m7 -lquadmath
//
// Run:  taskset -c 0-7 ./build-holonomic-m7/bench_holonomic_m7 [cases.tsv] [reps]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"

using namespace lcbinint::holonomic;
using clk = std::chrono::steady_clock;

struct BC {
    double xs, ys, rho, q, a;
    int bary;
    double u;
    double t_jac_incumbent;
    std::string name;
};

static double pct(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    double frac = idx - lo;
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] * (1 - frac) + v[lo + 1] * frac;
}

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 200;

    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return 2;
    }
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

    std::vector<double> per_case_ms, speedup;
    int ok = 0, nonok = 0;
    double checksum = 0.0;

    for (const auto& c : cases) {
        LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        // warm
        auto ej0 = epoch_jacobian(p, c.u, 64);
        checksum += ej0.mu;
        if (ej0.status == Status::OK) ++ok; else ++nonok;

        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            auto ej = epoch_jacobian(p, c.u, 64);
            auto t1 = clk::now();
            double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            best = std::min(best, ms);
            checksum += ej.grad_mu[0];
        }
        per_case_ms.push_back(best);
        speedup.push_back(c.t_jac_incumbent * 1e3 / best);
    }

    auto p50 = pct(per_case_ms, 50), p90 = pct(per_case_ms, 90),
         p95 = pct(per_case_ms, 95), p99 = pct(per_case_ms, 99);
    std::fprintf(stderr, "\n== C++ epoch_jacobian (value+5-Jac), best-of-%d per case ==\n", reps);
    std::fprintf(stderr, "  median %8.3f ms   p90 %8.3f ms   p95 %8.3f ms   p99 %8.3f ms   max %8.3f ms\n",
                 p50, p90, p95, p99, per_case_ms.back());

    auto s50 = pct(speedup, 50);
    std::sort(speedup.begin(), speedup.end());
    std::fprintf(stderr, "  speedup vs incumbent t_jac:  median %.1fx   min %.1fx   (p10 %.1fx)\n",
                 s50, speedup.front(), pct(speedup, 10));
    std::fprintf(stderr, "  incumbent: median %.2f ms  p90 %.1f ms  p95 %.0f ms  p99 %.0f ms\n",
                 14.89, 145.3, 4392.0, 4628.0);
    std::fprintf(stderr, "  status: %d OK, %d non-OK   checksum %.6f\n", ok, nonok, checksum);

    std::fprintf(stderr, "\n  GATE median>=2x:  %s (incumbent median 14.89 ms, need <= 7.44 ms)\n",
                 p50 <= 7.44 ? "PASS" : "FAIL");
    std::fprintf(stderr, "  GATE p95 non-regress: %s\n", p95 <= 4392.0 ? "PASS" : "FAIL");
    return 0;
}
