#pragma once

// Physical period seed for the equation-derived Gauss--Manin experiments.
//
// This is deliberately separate from the K-rule / Chebyshev packet path.  It
// evaluates the actual boundary period on a real image arc by deflating the
// two endpoint roots and applying a high-order Gauss--Chebyshev rule to the
// smooth remainder.  The returned I_half is the real-segment vector
//
//     I_k = int_arc t^k dt / sqrt(Q),
//
// while eta_closed = 2 I_half is the closed-cycle normalization used by the
// period reduction reference.  The observable identity is
//
//     Phi_arc = h^T I_half,       F_half,arc = 2 Phi_arc / rho.
//
// The helper is a seed/reference facility.  It does not generate connection
// coefficients and it is never called by the existing V2/V3 production
// router.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/gm_connection.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"

namespace lcbinint::holonomic {

struct GmPhysicalSeed {
    bool ok = false;
    bool chart2 = false;
    PrimaryFrame chart_pf{};
    double theta_enter = 0.0;
    double theta_leave = 0.0;
    double t_lo = 0.0;
    double t_hi = 0.0;
    std::array<double, 7> I_half{};
    std::array<double, 7> eta_closed{};
    std::array<double, 6> psi_closed{};
    std::array<double, 7> h{};
    double phi_arc = 0.0;
    double fhalf_arc = 0.0;
    double residue_check = 0.0;
};

inline double gm_wrap_pi(double x) {
    constexpr double pi = 3.1415926535897932384626433832795;
    constexpr double two_pi = 2.0 * pi;
    x = std::fmod(x + pi, two_pi);
    if (x < 0.0) x += two_pi;
    return x - pi;
}

inline std::array<double, 9> gm_q_coefficients(double R,
                                                const PrimaryFrame& pf) {
    const auto pc = boundary_quartic(R, pf).p;
    const double bm = (R - pf.a) * (R - pf.a);
    const double bp = (R + pf.a) * (R + pf.a);
    std::array<double, 9> q{};
    // Q = P (1+t^2) (bm + bp t^2), expanded without a generic convolution.
    const double d0 = bm;
    const double d2 = bm + bp;
    const double d4 = bp;
    q[0] = pc[0] * d0;
    q[1] = pc[1] * d0;
    q[2] = pc[2] * d0 + pc[0] * d2;
    q[3] = pc[3] * d0 + pc[1] * d2;
    q[4] = pc[4] * d0 + pc[2] * d2 + pc[0] * d4;
    q[5] = pc[3] * d2 + pc[1] * d4;
    q[6] = pc[4] * d2 + pc[2] * d4;
    q[7] = pc[3] * d4;
    q[8] = pc[4] * d4;
    return q;
}

inline std::array<double, 7> gm_observable_h(double R,
                                             const PrimaryFrame& pf) {
    const auto P = boundary_quartic(R, pf).p;
    const double bp = (R + pf.a) * (R + pf.a);
    const double bm = (R - pf.a) * (R - pf.a);
    const double inner_scale = 8.0 * pf.a * R;
    std::array<double, 7> out{};
    if (inner_scale == 0.0 || !std::isfinite(inner_scale)) return out;

    std::array<double, 6> inner{};
    const double dP[4] = {P[1], 2.0 * P[2], 3.0 * P[3], 4.0 * P[4]};
    const double B[3] = {bm, 0.0, bp};
    const double dB[2] = {0.0, 2.0 * bp};
    for (int i = 0; i <= 2; ++i)
        for (int j = 0; j <= 3; ++j) inner[i + j] += B[i] * dP[j];
    for (int i = 0; i <= 1; ++i)
        for (int j = 0; j <= 4; ++j) inner[i + j] += dB[i] * P[j];

    for (int k = 0; k <= 6; ++k) {
        const double derivative_term = k > 0 ? inner[k - 1] : 0.0;
        const double direct_term = k <= 4 ? 2.0 * bp * P[k] : 0.0;
        out[k] = (direct_term + derivative_term) / inner_scale;
    }
    return out;
}

inline std::array<double, 3> gm_residue_b(const std::array<double, 9>& q) {
    const double q8 = q[8];
    if (q8 == 0.0) return {0.0, 0.0, 0.0};
    const double q8_2 = q8 * q8;
    return {-q[7] / (2.0 * q8),
            -q[6] / (2.0 * q8) + 3.0 * q[7] * q[7] / (8.0 * q8_2),
            -q[5] / (2.0 * q8) + 3.0 * q[7] * q[6] / (4.0 * q8_2)
                - 5.0 * q[7] * q[7] * q[7] / (16.0 * q8_2 * q8)};
}

inline GmPhysicalSeed gm_physical_period_seed(
    double R, const PrimaryFrame& pf, const std::array<double, 2>& arc,
    int n = 256) {
    GmPhysicalSeed out;
    if (!(R > 0.0) || !(pf.rho > 0.0) || n < 8) return out;

    constexpr double pi = 3.1415926535897932384626433832795;
    constexpr double two_pi = 2.0 * pi;
    double enter = arc[0], leave = arc[1];
    if (leave <= enter) leave += two_pi;
    const double measure = leave - enter;
    if (!(measure > 0.0) || !(measure < two_pi)) return out;
    const double midpoint = 0.5 * (enter + leave);

    out.chart2 = std::cos(midpoint) < 0.0;
    const double shift = out.chart2 ? pi : 0.0;
    const double singular = out.chart2 ? 0.0 : pi;
    double d_sing = std::fmod(singular - enter, two_pi);
    if (d_sing < 0.0) d_sing += two_pi;
    if (d_sing > 0.0 && d_sing < measure) return out;

    out.chart_pf = out.chart2
        ? PrimaryFrame{-pf.a, pf.m0, -pf.X, -pf.Y, pf.rho}
        : pf;
    out.theta_enter = enter;
    out.theta_leave = leave;
    const double w1 = std::tan(gm_wrap_pi(enter - shift) * 0.5);
    const double w2 = std::tan(gm_wrap_pi(leave - shift) * 0.5);
    out.t_lo = std::min(w1, w2);
    out.t_hi = std::max(w1, w2);
    if (!(out.t_hi > out.t_lo) || !std::isfinite(out.t_lo) ||
        !std::isfinite(out.t_hi)) return out;

    const auto P = boundary_quartic(R, out.chart_pf).p;
    const double p4 = P[4];
    if (p4 == 0.0 || !std::isfinite(p4)) return out;
    const double beta = -(out.t_lo + out.t_hi);
    const double gamma = out.t_lo * out.t_hi;
    const double d1 = P[3] - beta * p4;
    const double d0 = P[2] - beta * d1 - gamma * p4;
    const double mid_t = 0.5 * (out.t_lo + out.t_hi);
    const double half_t = 0.5 * (out.t_hi - out.t_lo);
    const double bm = (R - out.chart_pf.a) * (R - out.chart_pf.a);
    const double bp = (R + out.chart_pf.a) * (R + out.chart_pf.a);
    auto g = [&](double t) {
        return -(d0 + t * (d1 + t * p4)) * (1.0 + t * t) *
               (bm + bp * t * t);
    };
    if (!(g(mid_t) > 0.0) || !std::isfinite(g(mid_t))) return out;

    const double weight = pi / n;
    for (int j = 0; j < n; ++j) {
        const double u = std::cos((j + 0.5) * pi / n);
        const double t = mid_t + half_t * u;
        const double gt = g(t);
        if (!(gt > 0.0) || !std::isfinite(gt)) return GmPhysicalSeed{};
        const double inv_sqrt = 1.0 / std::sqrt(gt);
        double tk = 1.0;
        for (int k = 0; k <= 6; ++k) {
            out.I_half[k] += weight * tk * inv_sqrt;
            tk *= t;
        }
    }
    for (double x : out.I_half)
        if (!std::isfinite(x)) return GmPhysicalSeed{};

    out.eta_closed = out.I_half;
    for (double& x : out.eta_closed) x *= 2.0;
    const auto q = gm_q_coefficients(R, out.chart_pf);
    const auto b = gm_residue_b(q);
    out.psi_closed = {out.eta_closed[0], out.eta_closed[1], out.eta_closed[2],
                      out.eta_closed[4] - b[0] * out.eta_closed[3],
                      out.eta_closed[5] - b[1] * out.eta_closed[3],
                      out.eta_closed[6] - b[2] * out.eta_closed[3]};
    out.h = gm_observable_h(R, out.chart_pf);
    out.residue_check = out.h[3] + b[0] * out.h[4] + b[1] * out.h[5] +
                        b[2] * out.h[6];
    for (int k = 0; k <= 6; ++k) out.phi_arc += out.h[k] * out.I_half[k];
    out.fhalf_arc = 2.0 * out.phi_arc / pf.rho;
    out.ok = std::isfinite(out.phi_arc) && std::isfinite(out.fhalf_arc) &&
             std::isfinite(out.residue_check);
    return out;
}

// Independent physical check for one arc.  This integrates the original
// angular sqrt(phi) expression with a Gauss--Chebyshev-2 rule.  The endpoint
// factor is divided out before evaluation, so the rule sees a smooth
// integrand.  It does not use P, Q, H, or the period basis.
inline bool gm_physical_arc_angular(double R, const PrimaryFrame& pf,
                                    const std::array<double, 2>& arc,
                                    int n, double& fhalf) {
    if (!(R > 0.0) || !(pf.rho > 0.0) || n < 8) return false;
    constexpr double pi = 3.1415926535897932384626433832795;
    constexpr double two_pi = 2.0 * pi;
    double enter = arc[0], leave = arc[1];
    if (leave <= enter) leave += two_pi;
    const double half = 0.5 * (leave - enter);
    const double mid = 0.5 * (enter + leave);
    if (!(half > 0.0) || !(half < pi)) return false;
    double sum = 0.0;
    for (int k = 1; k <= n; ++k) {
        const double x = std::cos(k * pi / (n + 1));
        const double s2 = std::max(0.0, 1.0 - x * x);
        const double theta = mid + half * x;
        const double phi = phi_lens(R, theta, pf);
        if (!(phi > 0.0) || !(s2 > 0.0)) return false;
        const double smooth = std::sqrt(phi / s2);
        sum += (pi / (n + 1)) * s2 * smooth;
    }
    fhalf = R * half * sum;
    return std::isfinite(fhalf);
}

}  // namespace lcbinint::holonomic
