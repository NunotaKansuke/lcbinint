// Single-case algebraic-boundary timing probe.  Run one (config,u) point so an
// outer `timeout` can bound the local backend's non-terminating paths.
//
//   argv: xs ys rho q a u [reps]
//   stdout (one line): OK  val_ms  jac_ms  mu_val  mu_jac  succ  grad_reliable  grad_err  "reason"
//   (exit 0 always on completion; the caller treats a `timeout` kill as HANG)
//
// Build:  see tests/holonomic_cpp/README_threeway (same flags as bench_three_way)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
namespace mg = lcbinint::magnification;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "need xs ys rho q a u [reps]\n"); return 2; }
    double xs = atof(argv[1]), ys = atof(argv[2]), rho = atof(argv[3]);
    double q = atof(argv[4]), a = atof(argv[5]), u = atof(argv[6]);
    int reps = argc > 7 ? atoi(argv[7]) : 100;

    const double m1f = q / (1.0 + q);
    lcbinint::SourcePosition src{xs - m1f * a, ys};
    mg::FiniteSourceSettings st;
    st.limb_darkening_c = u;
    st.limb_darkening_d = 0.0;
    mg::FiniteSourceMagnifier magn(st);

    auto v0 = magn.experimental_algebraic_boundary_binary_mag(a, q, src, rho);
    auto j0 = magn.experimental_algebraic_boundary_binary_jacobian(a, q, src, rho);

    double bv = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        auto v = magn.experimental_algebraic_boundary_binary_mag(a, q, src, rho);
        auto t1 = clk::now();
        bv = std::min(bv, std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (v.magnification == -1e99) std::printf("x");
    }
    double bj = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        auto j = magn.experimental_algebraic_boundary_binary_jacobian(a, q, src, rho);
        auto t1 = clk::now();
        bj = std::min(bj, std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (j.magnification == -1e99) std::printf("x");
    }
    std::string reason = v0.unsafe_reason.empty() ? j0.unsafe_reason : v0.unsafe_reason;
    for (auto& ch : reason) if (ch == ' ' || ch == '\t') ch = '_';
    if (reason.empty()) reason = "-";
    std::printf("OK %.5f %.5f %.10g %.10g %d %d %.3e %s\n",
                bv, bj, v0.magnification, j0.magnification,
                (int)(v0.success), (int)(j0.success && j0.grad_reliable),
                j0.grad_error, reason.c_str());
    return 0;
}
