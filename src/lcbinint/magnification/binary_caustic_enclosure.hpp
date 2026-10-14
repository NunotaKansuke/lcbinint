#pragma once

#include "lcbinint/math/polynomial_roots.hpp"
#include "lcbinint/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace lcbinint::magnification::detail {

using CausticEnclosureDisk = std::pair<SourcePosition, double>;

// A continuous, geometry-only strip bound, requiring no critical roots.
// Write A=m1/|z-a|^2 and B=m2/|z|^2. On a critical curve, |kappa|=1
// implies |A-B|<=1<=A+B, while Im(w)=Im(z)*(1-A-B). Hence
// |Im(w)| <= 2 min(m1*|Im(z)|/|z-a|^2, m2*|Im(z)|/|z|^2).
// For t=sqrt(m1*m2)/s, the sets m_i*|Im(z)|/|z-a_i|^2>t are
// two disjoint open disks: their radii m_i/(2t) sum to the separation
// of their centres. Thus the minimum is <=t everywhere, and every
// caustic lies in |Im(w)| <= 2*sqrt(m1*m2)/s. Real barycentre/frame
// shifts and swapping the lens masses do not change this strip.
inline double binary_caustic_strip_clearance(
    double separation, double mass_ratio, SourcePosition source)
{
    const double s = std::abs(separation), q_input = std::abs(mass_ratio);
    if (!(s > 0.0 && q_input > 0.0) ||
        !std::isfinite(s) || !std::isfinite(q_input) || !std::isfinite(source.y)) return 0.0;
    const double q = q_input < 1.0 ? q_input : 1.0/q_input;
    const double half_width = (2.0*std::sqrt(q)/(1.0+q))/s;
    if (!std::isfinite(half_width)) return 0.0;
    const double roundoff = 256.0*std::numeric_limits<double>::epsilon()*
        (1.0+half_width+std::abs(source.y));
    return std::max(0.0, std::abs(source.y)-half_width-roundoff);
}

// Enclose *all* critical phases, not just sampled caustic vertices. At a phase
// centre, Q(z)=-phase*product(z-root_j). On |z-root_i|=r, Rouche gives one root
// throughout the phase interval if
//   |phase| r product_{j!=i}(|root_i-root_j|-r)
//       > |delta_phase| (|root_i|+r)^2 (|root_i-a|+r)^2 + coefficient_error.
// Disjoint disks account for all four roots of the exact quartic. Mapping a
// disk which avoids both lens poles increases its radius by at most
//   r [1 + m1/(d1(d1-r)) + m2/(d2(d2-r))].
// The resulting source-plane disks cover the whole continuous caustic. Root
// factorisation error and generous double-roundoff padding enter the upper
// bound. If any interval cannot be covered, discard the entire enclosure.
inline std::vector<CausticEnclosureDisk> build_binary_caustic_enclosure(
    double separation, double mass_ratio)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    if (!(s > 0.0 && q_input > 0.0) || !std::isfinite(s) || !std::isfinite(q_input)) {
        return {};
    }
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const double a = q_input < 1.0 ? -s : s;
    const double m1 = 1.0 / (1.0 + q);
    const double m2 = q * m1;
    constexpr double pi = 3.14159265358979323846;
    constexpr double padding = 256.0 * std::numeric_limits<double>::epsilon();
    constexpr int initial_intervals = 32;
    constexpr int max_depth = 4;
    struct Interval { double center, half_width; int depth; };
    std::vector<Interval> pending;
    // Real lens positions/masses imply Q(conj(z), conj(phase))=conj(Q(z,phase)).
    // Prove the upper half of the phase circle once and reflect its disks to
    // cover the lower half. The lens map has the same conjugation symmetry.
    for (int k = initial_intervals/2 - 1; k >= 0; --k) {
        pending.push_back({2.0*pi*(k+0.5)/initial_intervals, pi/initial_intervals, 0});
    }
    std::vector<CausticEnclosureDisk> cover;
    cover.reserve(256);
    math::PolynomialRootSolver solver;
    while (!pending.empty()) {
        const auto interval = pending.back();
        pending.pop_back();
        const Complex phase = std::polar(1.0, interval.center);
        const std::vector<Complex> coefficients {
            m2*a*a, -2.0*m2*a, 1.0-phase*a*a, 2.0*phase*a, -phase,
        };
        const auto solved = solver.solve(coefficients);
        if (solved.status != math::RootSolverStatus::ok || solved.roots.size() != 4) return {};

        // Reconstruct the polynomial represented by these numerical roots.
        // The absolute-value product bounds coefficient-formation roundoff,
        // including cancellation in the reconstructed coefficients.
        std::array<Complex, 5> reconstructed {};
        std::array<double, 5> envelope {};
        reconstructed[0] = -phase;
        envelope[0] = std::abs(phase);
        int degree = 0;
        for (const auto root : solved.roots) {
            if (!std::isfinite(root.real()) || !std::isfinite(root.imag())) return {};
            for (int j = degree + 1; j >= 0; --j) {
                reconstructed[j] = (j > 0 ? reconstructed[j-1] : Complex{}) -
                    (j <= degree ? root * reconstructed[j] : Complex{});
                envelope[j] = (j > 0 ? envelope[j-1] : 0.0) +
                    (j <= degree ? std::abs(root)*envelope[j] : 0.0);
            }
            ++degree;
        }
        std::array<double, 5> errors {};
        for (int j = 0; j < 5; ++j) {
            errors[j] = std::abs(coefficients[j]-reconstructed[j]) +
                padding*(1.0 + std::abs(coefficients[j]) + envelope[j]);
        }
        const double delta = 2.0*std::sin(interval.half_width/2.0) + padding;
        std::array<CausticEnclosureDisk, 4> disks;
        bool enclosed = true;
        for (int i = 0; i < 4; ++i) {
            const auto root = solved.roots[i];
            const double d1 = std::abs(root-a), d2 = std::abs(root);
            std::array<double, 3> separations;
            int n = 0;
            for (int j = 0; j < 4; ++j) if (j != i) {
                separations[n++] = std::abs(root-solved.roots[j]);
            }
            const double limit = 0.45*std::min({d1, d2,
                separations[0], separations[1], separations[2]});
            const double derivative = std::abs(phase)*
                separations[0]*separations[1]*separations[2];
            if (!(limit > 0.0 && derivative > 0.0)) { enclosed = false; break; }
            double coefficient_error = errors[4];
            for (int j = 3; j >= 0; --j) coefficient_error = coefficient_error*d2 + errors[j];
            double radius = std::max(padding*(1.0+d2),
                1.25*(delta*d2*d2*d1*d1 + coefficient_error)/derivative);
            bool proved = false;
            for (; radius <= limit; radius *= 1.25) {
                const double upper_z = d2+radius, upper_a = d1+radius;
                double error_bound = errors[4];
                for (int j = 3; j >= 0; --j) error_bound = error_bound*upper_z + errors[j];
                const double lower = std::abs(phase)*radius*
                    (separations[0]-radius)*(separations[1]-radius)*(separations[2]-radius);
                const double upper = delta*upper_z*upper_z*upper_a*upper_a + error_bound;
                if (std::isfinite(lower) && std::isfinite(upper) &&
                    lower*(1.0-padding) > upper*(1.0+padding)) {
                    proved = true;
                    break;
                }
            }
            if (!proved) { enclosed = false; break; }
            const auto mapped = root - m1/(std::conj(root)-a) - m2/std::conj(root) - a*m1;
            const double map_radius = radius*(1.0 +
                m1/(d1*(d1-radius)) + m2/(d2*(d2-radius)));
            const double roundoff = padding*(1.0+std::abs(root)+std::abs(mapped)+
                m1/(d1-radius) + m2/(d2-radius));
            if (!std::isfinite(map_radius) || !std::isfinite(roundoff)) return {};
            disks[i] = {{mapped.real(), mapped.imag()}, map_radius*(1.0+padding)+roundoff};
        }
        if (enclosed) {
            cover.insert(cover.end(), disks.begin(), disks.end());
            for (const auto& disk : disks) {
                cover.push_back({{disk.first.x, -disk.first.y}, disk.second});
            }
        } else {
            if (interval.depth >= max_depth) return {};
            const double half = interval.half_width/2.0;
            pending.push_back({interval.center+half, half, interval.depth+1});
            pending.push_back({interval.center-half, half, interval.depth+1});
        }
    }
    return cover;
}

inline double binary_caustic_enclosure_clearance(
    const std::vector<CausticEnclosureDisk>& cover, SourcePosition source)
{
    if (cover.empty()) return 0.0;
    double distance = std::numeric_limits<double>::infinity();
    constexpr double padding = 512.0*std::numeric_limits<double>::epsilon();
    const double source_scale = 1.0+std::abs(source.x)+std::abs(source.y);
    for (const auto& disk : cover) {
        const double dx = source.x-disk.first.x, dy = source.y-disk.first.y;
        // Most disks cannot improve the running minimum. Compare squared
        // distances first, with outward padding for subtraction/roundoff;
        // retain hypot whenever the comparison is close or overflows.
        const double limit = (distance+disk.second)*(1.0+padding) +
            padding*(source_scale+std::abs(disk.first.x)+std::abs(disk.first.y));
        const double squared = dx*dx+dy*dy;
        if (limit > 0.0 && std::isfinite(limit) &&
            std::isfinite(squared) && squared > limit*limit) continue;
        distance = std::min(distance,
            std::hypot(dx, dy) - disk.second);
    }
    // Pad the source-coordinate/distance evaluation as well as disk mapping.
    const double roundoff = 256.0*std::numeric_limits<double>::epsilon()*
        (1.0+std::abs(source.x)+std::abs(source.y)+std::abs(distance));
    return std::isfinite(distance) ? std::max(0.0, distance-roundoff) : 0.0;
}

} // namespace lcbinint::magnification::detail
