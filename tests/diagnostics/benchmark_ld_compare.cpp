// Light linear-limb-darkening speed comparison on matched geometries:
//   * lcbinint mainstream  (FiniteSourceMagnifier::binary_mag, auto mode)
//   * lcbinint algebraic-boundary primal   (value only)
//   * lcbinint algebraic-boundary + Jacobian  (value + analytic d/dp)
//
// A companion Python driver (benchmark_ld_compare.py) adds VBMicrolensing's
// BinaryMagDark on the identical geometries and prints the comparison table.
//
// Each case is a short source track so the mainstream one-entry result cache
// does not alias consecutive calls.  CPU time (std::clock), min-of-N.

#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>

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
    double x0, y0, x1, y1;
};

}  // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    const BCase cases[] = {
        {"off_caustic_wide",    1.35, 0.28,  0.05,   0.18, 0.15,  0.28, 0.15},
        {"off_caustic_close",   0.72, 0.45,  0.03,   0.02, 0.20,  0.10, 0.14},
        {"caustic_crossing",    1.00, 0.20,  0.02,   0.05, 0.02,  0.14, 0.06},
        {"resonant_equal",      1.00, 1.00,  0.01,  -0.02, 0.03,  0.05, 0.00},
        {"large_source_smooth", 1.00, 0.10,  0.40,   0.20, 0.30,  0.55, 0.30},
        {"clear_wide_small",    1.60, 0.30,  0.01,   0.90, 0.40,  1.00, 0.40},
        {"clear_close_small",   0.60, 0.20,  0.01,   0.55, 0.35,  0.65, 0.35},
    };
    const double tol = 1.0e-3;
    const double limb_c = 0.5;
    constexpr int kTrack = 5;

    std::printf("case,tolerance,sep,q,rho,limb_c,mid_x,mid_y,"
                "mainstream_us,alg_primal_us,alg_jacobian_us,alg_multipole_hits,"
                "mainstream_mag,alg_primal_mag\n");

    for (const auto& cs : cases) {
        std::array<lcbinint::SourcePosition, kTrack> track;
        for (int i = 0; i < kTrack; ++i) {
            const double t = double(i) / (kTrack - 1);
            track[static_cast<std::size_t>(i)] = {
                cs.x0 + t * (cs.x1 - cs.x0), cs.y0 + t * (cs.y1 - cs.y0)};
        }

        mg::FiniteSourceSettings st;
        st.limb_darkening_c = limb_c;
        st.finite_mode = 4;
        st.finite_source_reltol = tol;
        mg::FiniteSourceMagnifier magnifier(st);

        mg::AlgebraicBoundarySettings alg;
        alg.relative_tolerance = tol;
        alg.absolute_tolerance = 0.0;

        mg::PointSourceMagnifier point;

        auto mainstream = [&] {
            double s = 0.0;
            for (const auto& src : track) {
                const double a0 = point.binary_mag0(cs.sep, cs.q, src).magnification;
                s += magnifier.binary_mag(
                    cs.sep, cs.q, src, cs.rho, a0, nullptr, true, &point).magnification;
            }
            return s;
        };
        auto primal = [&] {
            double s = 0.0;
            for (const auto& src : track)
                s += magnifier.experimental_algebraic_boundary_binary_mag(
                    cs.sep, cs.q, src, cs.rho, alg).magnification;
            return s;
        };
        auto jacobian = [&] {
            double s = 0.0;
            for (const auto& src : track)
                s += magnifier.experimental_algebraic_boundary_binary_jacobian(
                    cs.sep, cs.q, src, cs.rho, alg).magnification;
            return s;
        };

        // Warm the per-lens caustic / limb-darkening caches so the min-of-N
        // measures steady-state epoch cost, not one-time construction.
        mainstream(); primal(); jacobian();

        const double tm = min_cpu(mainstream, 3, 5) / kTrack;
        const double tp = min_cpu(primal, 3, 5) / kTrack;
        const double tj = min_cpu(jacobian, 3, 5) / kTrack;

        int multipole_hits = 0;
        for (const auto& src : track) {
            const auto r = magnifier.experimental_algebraic_boundary_binary_mag(
                cs.sep, cs.q, src, cs.rho, alg);
            multipole_hits += r.diagnostics.multipole_shortcut_count;
        }
        // Value check at the mid track point (matched with the VBM call).
        const auto mid = track[kTrack / 2];
        const double a0_mid = point.binary_mag0(cs.sep, cs.q, mid).magnification;
        const double mm = magnifier.binary_mag(
            cs.sep, cs.q, mid, cs.rho, a0_mid, nullptr, true, &point).magnification;
        const double pm = magnifier.experimental_algebraic_boundary_binary_mag(
            cs.sep, cs.q, mid, cs.rho, alg).magnification;

        std::printf("%s,%.0e,%.6g,%.6g,%.6g,%.3g,%.6g,%.6g,%.3f,%.3f,%.3f,%d,%.10f,%.10f\n",
            cs.name, tol, cs.sep, cs.q, cs.rho, limb_c,
            track[kTrack / 2].x, track[kTrack / 2].y,
            tm * 1e6, tp * 1e6, tj * 1e6, multipole_hits, mm, pm);
    }
    return 0;
}
