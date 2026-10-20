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

// On a critical curve min(d1,d2)<=1, so its barycentric radius is at
// most max(m1,m2)*s+1. If di<=s/2, the reverse triangle inequality for
// kappa gives mi/di^2<=1+4*mj/s^2; otherwise di>s/2. These lower pole
// distances bound the lens-map deflection without solving critical roots.
inline double binary_caustic_radial_clearance(
    double separation, double mass_ratio, SourcePosition source)
{
    const double s = std::abs(separation), q_input = std::abs(mass_ratio);
    if (!(s > 0.0 && q_input > 0.0) || !std::isfinite(s) ||
        !std::isfinite(q_input) || !std::isfinite(source.x) ||
        !std::isfinite(source.y)) return 0.0;
    const double q = std::min(q_input, 1.0/q_input);
    const double m1 = 1.0/(1.0+q), m2 = q*m1;
    const double deflection1 = std::max(2.0*m1/s, std::sqrt(m1)*std::hypot(1.0, 2.0*std::sqrt(m2)/s));
    const double deflection2 = std::max(2.0*m2/s, std::sqrt(m2)*std::hypot(1.0, 2.0*std::sqrt(m1)/s));
    const double radius = m1*s + 1.0 + deflection1 + deflection2;
    const double distance = std::hypot(source.x, source.y);
    if (!std::isfinite(radius) || !std::isfinite(distance)) return 0.0;
    const double roundoff = 512.0*std::numeric_limits<double>::epsilon()*
        (1.0+radius+distance);
    return std::max(0.0, distance-radius-roundoff);
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
inline std::vector<CausticEnclosureDisk> cover_binary_caustic_phase(
    double separation, double mass_ratio, double phase_center, double phase_half_width,
    double separation_half_width, std::vector<Complex>* root_centers = nullptr,
    bool reuse_centers = true, bool polish_centers = true,
    std::vector<double>* image_radii = nullptr)
{

    if (!std::isfinite(phase_center) || !(phase_half_width >= 0.0 &&
        phase_half_width <= 3.14159265358979323846)) return {};
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    if (!(s > 0.0 && q_input > 0.0) || !std::isfinite(s) || !std::isfinite(q_input)) {
        return {};
    }
    if (!(std::isfinite(separation_half_width) && separation_half_width >= 0.0 &&
          separation_half_width < s)) return {};
    const double da = separation_half_width;
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const double a = q_input < 1.0 ? -s : s;
    const double m1 = 1.0 / (1.0 + q);
    const double m2 = q * m1;
    constexpr double padding = 256.0 * std::numeric_limits<double>::epsilon();
    math::PolynomialRootSolver solver;
    const Complex phase = std::polar(1.0, phase_center);
    const std::array<Complex,5> coefficients {
        m2*a*a, -2.0*m2*a, 1.0-phase*a*a, 2.0*phase*a, -phase,
    };
    const bool cached = reuse_centers && root_centers != nullptr && root_centers->size() == 4;
    std::vector<Complex> local_roots;
    if (!cached) {
        local_roots=solver.solve(std::vector<Complex>(coefficients.begin(),coefficients.end())).roots;
        if(root_centers!=nullptr)*root_centers=std::move(local_roots);
    }
    const auto retry = [&]() {
        if(cached && polish_centers) {
            // Approximate centers only: the following enclosure still checks
            // the complete coefficient discrepancy and disjoint root disks.
            for(auto& z:*root_centers) for(int k=0;k<2;++k) {
                Complex value=coefficients[4],derivative{};
                for(int j=3;j>=0;--j) {derivative=derivative*z+value;value=value*z+coefficients[j];}
                if(std::abs(derivative)==0.0)break;
                const double denominator=std::norm(derivative);
                if(!(denominator>0.0 && std::isfinite(denominator)))break;
                const Complex step((value.real()*derivative.real()+value.imag()*derivative.imag())/denominator,
                    (value.imag()*derivative.real()-value.real()*derivative.imag())/denominator);
                const auto next=z-step;
                if(!std::isfinite(next.real())||!std::isfinite(next.imag()))break;
                z=next;
            }
            return cover_binary_caustic_phase(separation,mass_ratio,phase_center,
                phase_half_width,separation_half_width,root_centers,true,false,image_radii);
        }
        return cover_binary_caustic_phase(separation,mass_ratio,phase_center,
            phase_half_width,separation_half_width,root_centers,false,false,image_radii);
    };
    const auto& roots=root_centers!=nullptr?*root_centers:local_roots;
    if (roots.size()!=4) return {};

    // Reconstruct the polynomial represented by these numerical roots.
    // The absolute-value product bounds coefficient-formation roundoff,
    // including cancellation in the reconstructed coefficients.
    std::array<Complex, 5> reconstructed {};
    std::array<double, 5> envelope {};
    reconstructed[0] = -phase;
    envelope[0] = std::abs(phase);
    int degree = 0;
    for (const auto root : roots) {
        if (!std::isfinite(root.real()) || !std::isfinite(root.imag())) {
            if(cached)return retry();
            return {};
        }
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
    const double delta = 2.0*std::sin(phase_half_width/2.0) + padding;
    std::array<CausticEnclosureDisk, 4> disks;
    std::array<double,4> certified_radii {};
    bool enclosed = true;
    for (int i = 0; i < 4; ++i) {
        const auto root = roots[i];
        const double d1 = std::abs(root-a), d2 = std::abs(root);
        std::array<double, 3> separations;
        int n = 0;
        for (int j = 0; j < 4; ++j) if (j != i) {
            separations[n++] = std::abs(root-roots[j]);
        }
        const double limit = 0.45*std::min({d1-da, d2,
            separations[0], separations[1], separations[2]});
        const double derivative = std::abs(phase)*
            separations[0]*separations[1]*separations[2];
        if (!(limit > 0.0 && derivative > 0.0)) { enclosed = false; break; }
        double coefficient_error = errors[4];
        for (int j = 3; j >= 0; --j) coefficient_error = coefficient_error*d2 + errors[j];
        // Root centres may belong to a nearby separation. Reconstruction
        // error accounts for that drift in the Rouché inequality below.
        const double phase_error = delta*d2*d2*(d1+da)*(d1+da);
        if (cached && coefficient_error > 0.25*phase_error)
            return retry();
        double radius = std::max(padding*(1.0+d2),
            1.25*(delta*d2*d2*(d1+da)*(d1+da) +
                (m2+d2*d2)*(2.0*da*d1+da*da) + coefficient_error)/derivative);
        bool proved = false;
        for (; radius <= limit; radius *= 1.25) {
            const double upper_z = d2+radius, upper_a = d1+radius;
            double error_bound = errors[4];
            for (int j = 3; j >= 0; --j) error_bound = error_bound*upper_z + errors[j];
            const double lower = std::abs(phase)*radius*
                (separations[0]-radius)*(separations[1]-radius)*(separations[2]-radius);
            // Q(z,a+da)-Q(z,a)=(m2-phase*z^2)*(-2*da*(z-a)+da^2).
            // Bound this over the whole separation interval, then add
            // the independent critical-phase perturbation at the moved pole.
            const double parameter_error = (m2+upper_z*upper_z)*
                (2.0*da*upper_a+da*da);
            const double upper = delta*upper_z*upper_z*(upper_a+da)*(upper_a+da)
                + parameter_error + error_bound;
            if (std::isfinite(lower) && std::isfinite(upper) &&
                lower*(1.0-padding) > upper*(1.0+padding)) {
                proved = true;
                break;
            }
        }
        if (!proved) { enclosed = false; break; }
        certified_radii[i]=radius;
        const auto mapped = root - m1/(std::conj(root)-a) - m2/std::conj(root) - a*m1;
        const double map_radius = radius +
            m1*(radius+da)/(d1*(d1-radius-da)) +
            m2*radius/(d2*(d2-radius)) + m1*da;
        const double roundoff = padding*(1.0+std::abs(root)+std::abs(mapped)+
            m1/(d1-radius-da) + m2/(d2-radius) + m1*da);
        if (!std::isfinite(map_radius) || !std::isfinite(roundoff)) return {};
        disks[i] = {{mapped.real(), mapped.imag()}, map_radius*(1.0+padding)+roundoff};
    }
    // A failed phase/separation enclosure needs interval refinement. Re-solving
    // accurate centres cannot remove that perturbation. Coefficient-dominated
    // centre errors already trigger the guarded retry above.
    if (!enclosed) return {};
    if(image_radii != nullptr)image_radii->assign(certified_radii.begin(),certified_radii.end());
    std::vector<CausticEnclosureDisk> cover;
    cover.reserve(8);
    cover.insert(cover.end(),disks.begin(),disks.end());
    for (const auto& disk:disks) cover.push_back({{disk.first.x,-disk.first.y},disk.second});
    return cover;
}

inline std::vector<CausticEnclosureDisk> build_binary_caustic_enclosure(
    double separation, double mass_ratio, int max_depth = 4, int initial_intervals = 32,
    double separation_half_width = 0.0)
{
    if(initial_intervals<2 || initial_intervals%2!=0 || max_depth<0) return {};
    constexpr double pi=3.14159265358979323846;
    struct Interval {double center,half_width;int depth;};
    std::vector<Interval> pending;
    for(int k=initial_intervals/2-1;k>=0;--k)
        pending.push_back({2.0*pi*(k+0.5)/initial_intervals,pi/initial_intervals,0});
    std::vector<CausticEnclosureDisk> cover;
    while(!pending.empty()) {
        const auto interval=pending.back();pending.pop_back();
        const auto disks=cover_binary_caustic_phase(separation,mass_ratio,
            interval.center,interval.half_width,separation_half_width);
        if(!disks.empty()) cover.insert(cover.end(),disks.begin(),disks.end());
        else {
            if(interval.depth>=max_depth)return {};
            const double half=interval.half_width/2.0;
            pending.push_back({interval.center+half,half,interval.depth+1});
            pending.push_back({interval.center-half,half,interval.depth+1});
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
