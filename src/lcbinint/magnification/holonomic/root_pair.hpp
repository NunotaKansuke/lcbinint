#pragma once

// ATPT holonomic solver -- the boundary root pair (m, v) of one image arc.
//
// Direct port of python/lcbinint/holonomic_ref/root_pair.py (plan sec. 8, 11).
// An image arc on |z| = R is bounded by two real roots t_- < t_+ of the
// boundary quartic P(t; R), t = tan(theta/2).  Instead of the two roots the
// radial transport carries the symmetric pair
//
//     m = (t_+ + t_-) / 2 ,      v = ((t_+ - t_-) / 2)^2 ,
//
// so a tangency (t_+ -> t_-) is the smooth boundary v -> 0 rather than a
// collision of two coordinates.  For a quartic
//
//     E(m, v) := (P(m+s) + P(m-s)) / 2 = P + (v/2) P'' + (v^2/24) P''''
//     O(m, v) := (P(m+s) - P(m-s)) / (2 s) = P' + (v/6) P'''
//
// with s = sqrt(v) and NO truncation (higher derivatives of a quartic
// vanish), so E = O = 0 iff P(t_-) = P(t_+) = 0 exactly.  The Jacobian at a
// genuine tangency (P'(m) = 0, v = 0) is det d(E,O)/d(m,v) = -1/2 P''(m)^2.
//
// Radial motion of the pair is the 2x2 implicit-function solve
//     [[E_m, E_v], [O_m, O_v]] [dm/dR, dv/dR]^T = -[E_R, O_R]^T
// and of a single endpoint  dt/dR = -P_R(t) / P_t(t).

#include <array>
#include <cmath>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"

namespace lcbinint::holonomic {

constexpr double kRootPairDetFloor = 1e-300;

struct RootPair {
    double m = 0.0;
    double v = 0.0;  // squared half-gap; v = 0 is a tangency

    double t_minus() const { return m - std::sqrt(v > 0.0 ? v : 0.0); }
    double t_plus() const { return m + std::sqrt(v > 0.0 ? v : 0.0); }

    // Angular width of the arc this pair bounds (theta = 2 arctan t).
    double delta_theta() const {
        const double s = std::sqrt(v > 0.0 ? v : 0.0);
        return 2.0 * std::atan2(2.0 * s, 1.0 + m * m - v);
    }
};

inline RootPair root_pair_from_endpoints(double t_lo, double t_hi) {
    const double half = 0.5 * (t_hi - t_lo);
    return RootPair{0.5 * (t_lo + t_hi), half * half};
}

// (P, P', P'', P''', P'''') of the ascending quartic `pc` at t.
struct PDerivs {
    double P, P1, P2, P3, P4;
};
inline PDerivs p_derivs(const std::array<double, 5>& pc, double t) {
    const double p0 = pc[0], p1 = pc[1], p2 = pc[2], p3 = pc[3], p4 = pc[4];
    PDerivs d;
    d.P = p0 + t * (p1 + t * (p2 + t * (p3 + t * p4)));
    d.P1 = p1 + t * (2 * p2 + t * (3 * p3 + t * 4 * p4));
    d.P2 = 2 * p2 + t * (6 * p3 + t * 12 * p4);
    d.P3 = 6 * p3 + t * 24 * p4;
    d.P4 = 24 * p4;
    return d;
}

// (E, O) -- both zero iff t_- and t_+ are roots of P.
struct EO {
    double E, O;
};
inline EO eo_residuals(const RootPair& rp, const std::array<double, 5>& pc) {
    const double m = rp.m, v = rp.v;
    const PDerivs d = p_derivs(pc, m);
    return {d.P + 0.5 * v * d.P2 + (v * v / 24.0) * d.P4,
            d.P1 + (v / 6.0) * d.P3};
}

// [[E_m, E_v], [O_m, O_v]] at (m, v) -- exact for a quartic.
struct EOJac {
    double E_m, E_v, O_m, O_v;
};
inline EOJac eo_jacobian(const RootPair& rp, const std::array<double, 5>& pc) {
    const double m = rp.m, v = rp.v, p4 = pc[4];
    const PDerivs d = p_derivs(pc, m);
    return {d.P1 + 0.5 * v * d.P3,   // E_m
            0.5 * d.P2 + 2.0 * v * p4,  // E_v
            d.P2 + 4.0 * v * p4,     // O_m
            d.P3 / 6.0};             // O_v
}

// The boxed identity value -1/2 P''(m)^2 (Jacobian det at v = 0).
inline double tangency_determinant(double m, const std::array<double, 5>& pc) {
    const double P2 = p_derivs(pc, m).P2;
    return -0.5 * P2 * P2;
}

// dt/dR = -P_R(t) / P_t(t) for one boundary root t at radius R.
struct EndpointDR {
    double dt_dR;
    bool ok;
};
inline EndpointDR endpoint_dR(double t, double R, const PrimaryFrame& pf) {
    const QuarticCoeffs pc = boundary_quartic(R, pf);
    const QuarticCoeffs pcR = boundary_quartic_dR(R, pf);
    const double P_t = p_derivs(pc.p, t).P1;
    const double P_R = poly_eval(pcR.p, t);
    if (std::fabs(P_t) < kRootPairDetFloor || !std::isfinite(P_t))
        return {0.0, false};
    return {-P_R / P_t, true};
}

// (dm/dR, dv/dR) by the 2x2 implicit-function solve.  ok == false if the
// pair Jacobian is singular (tangency / degenerate arc) -- fail closed, the
// caller must switch to a cold quartic solve.
struct RootPairDR {
    double dm_dR, dv_dR;
    bool ok;
};

// Pre-built coefficient / dR-coefficient arrays (the hot path -- the
// integrator already holds them).
inline RootPairDR root_pair_dR(const RootPair& rp,
                               const std::array<double, 5>& pc,
                               const std::array<double, 5>& pcR) {
    const double m = rp.m, v = rp.v;
    const EOJac J = eo_jacobian(rp, pc);
    // E_R = P_R(m) + v/2 P''_R(m) + v^2 p4_R ; O_R = P'_R(m) + v/6 P'''_R(m)
    const double PR0 = pcR[0] + m * (pcR[1] + m * (pcR[2] + m * (pcR[3] +
                                                                m * pcR[4])));
    const double PR1 =
        pcR[1] + m * (2 * pcR[2] + m * (3 * pcR[3] + m * 4 * pcR[4]));
    const double PR2 = 2 * pcR[2] + m * (6 * pcR[3] + m * 12 * pcR[4]);
    const double PR3 = 6 * pcR[3] + m * 24 * pcR[4];
    const double E_R = PR0 + 0.5 * v * PR2 + v * v * pcR[4];
    const double O_R = PR1 + (v / 6.0) * PR3;
    const double det = J.E_m * J.O_v - J.E_v * J.O_m;
    if (std::fabs(det) < kRootPairDetFloor || !std::isfinite(det))
        return {0.0, 0.0, false};
    const double dm = -(J.O_v * E_R - J.E_v * O_R) / det;
    const double dv = -(-J.O_m * E_R + J.E_m * O_R) / det;
    return {dm, dv, true};
}

// Convenience: build the coefficients from the frame at R.
inline RootPairDR root_pair_dR(const RootPair& rp, double R,
                               const PrimaryFrame& pf) {
    return root_pair_dR(rp, boundary_quartic(R, pf).p,
                        boundary_quartic_dR(R, pf).p);
}

}  // namespace lcbinint::holonomic
