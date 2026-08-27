// Speed + accuracy probe for the forward-mode algebraic-boundary Jacobian.
//
// Reports, per case: primal value time, value+Jacobian time, the overhead
// factor, and the worst central-FD mismatch of the analytic Jacobian. CPU time
// (std::clock) with a min-of-N reduction, since the box is oversubscribed and
// wall clock is meaningless here.

#include "lcbinint/magnification/finite_source_magnifier.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <vector>

namespace mg = lcbinint::magnification;

namespace {

template <typename F>
double min_cpu(F&& f, int outer, int inner)
{
    double best = std::numeric_limits<double>::infinity();
    for (int o = 0; o < outer; ++o) {
        const std::clock_t t0 = std::clock();
        for (int i = 0; i < inner; ++i) f();
        const double dt = double(std::clock() - t0) / CLOCKS_PER_SEC / inner;
        if (dt < best) best = dt;
    }
    return best;
}

struct BCase {
    const char* name;
    double sep, q, rho;
    lcbinint::SourcePosition src;
};

}  // namespace

int main()
{
    const BCase cases[] = {
        {"wide/off-caustic", 1.35, 0.28, 0.05, {0.22, 0.15}},
        {"close/off-caustic", 0.72, 0.45, 0.03, {0.05, 0.18}},
        {"resonant-ish", 1.05, 0.60, 0.02, {0.11, 0.04}},
        {"small-source", 1.20, 0.20, 0.008, {0.30, 0.02}},
    };
    struct Prof { const char* name; double c, d; };
    const Prof profs[] = {{"uniform", 0.0, 0.0}, {"linear-LD", 0.6, 0.0}};

    std::printf("%-18s %-10s %12s %12s %8s %12s\n",
                "case", "profile", "value[us]", "val+jac[us]", "x", "max_fd_err");
    for (const auto& pr : profs) {
        for (const auto& cs : cases) {
            mg::FiniteSourceSettings st;
            st.limb_darkening_c = pr.c;
            st.limb_darkening_d = pr.d;
            mg::FiniteSourceMagnifier magnifier(st);

            auto val = [&] {
                return magnifier.experimental_algebraic_boundary_binary_mag(
                    cs.sep, cs.q, cs.src, cs.rho).magnification;
            };
            auto vj = [&] {
                return magnifier.experimental_algebraic_boundary_binary_jacobian(
                    cs.sep, cs.q, cs.src, cs.rho).grad_error;
            };
            const double tv = min_cpu(val, 5, 200);
            const double tj = min_cpu(vj, 5, 200);

            const auto jac = magnifier.experimental_algebraic_boundary_binary_jacobian(
                cs.sep, cs.q, cs.src, cs.rho);
            double max_err = 0.0;
            if (jac.success && jac.grad_reliable) {
                const double base[5] = {cs.src.x, cs.src.y, cs.rho, cs.q, cs.sep};
                // Primal magnification carries ~1e-6 internal tolerance noise,
                // so a central FD needs a generous step to stay above it.
                const double step[5] = {1e-5, 1e-5, 5e-6, 1e-5, 1e-5};
                auto mag_at = [&](const double* b) {
                    return magnifier.experimental_algebraic_boundary_binary_mag(
                        b[4], b[3], {b[0], b[1]}, b[2]).magnification;
                };
                for (int p = 0; p < 5; ++p) {
                    double bp[5], bm[5];
                    for (int k = 0; k < 5; ++k) { bp[k] = base[k]; bm[k] = base[k]; }
                    bp[p] += step[p]; bm[p] -= step[p];
                    const double fd = (mag_at(bp) - mag_at(bm)) / (2.0 * step[p]);
                    const double e = std::fabs(jac.jacobian[p] - fd) /
                        (1.0 + std::fabs(fd));
                    if (e > max_err) max_err = e;
                }
            } else {
                max_err = std::nan("");
            }

            std::printf("%-18s %-10s %12.2f %12.2f %8.2f %12.2e\n",
                        cs.name, pr.name, tv * 1e6, tj * 1e6, tj / tv, max_err);
        }
    }
    return 0;
}
