#pragma once

// ATPT holonomic solver (M7) -- the boundary quartic P(t; R) and its exact
// parameter derivatives.  Direct port of
// python/lcbinint/holonomic_ref/polynomial_family.py:
//   boundary_quartic, boundary_quartic_dp
// and jacobian.py's use of boundary_quartic_dp (t-chart cross-check).
//
//   t = tan(theta/2),  A(t) = 1 + t^2
//   B(t;R) = (R-a)^2 + (R+a)^2 t^2
//   n0 = -zeta R^2, n1 = R (R^2 - 1 + a zeta), n2 = a (m0 - R^2)
//   T(t) = cT0 + cT1 t + cT2 t^2,  cT0 = n0+n1+n2, cT1 = 2i(n2-n0), cT2 = n1-n0-n2
//   P(t;R) = rho^2 R^2 A B - T conj(T)          (real, ascending [p0..p4])

#include <array>
#include <complex>

#include "lcbinint/magnification/holonomic/lens_frame.hpp"

namespace lcbinint::holonomic {

using cd = std::complex<double>;

struct QuarticCoeffs {
    std::array<double, 5> p;  // ascending: p0 + p1 t + ... + p4 t^4
};

inline void t_complex_coeffs(double R, double a, double m0, double X, double Y,
                             cd& cT0, cd& cT1, cd& cT2) {
    const cd zeta(X, Y);
    const cd n0 = -zeta * (R * R);
    const cd n1 = R * (R * R - 1.0 + a * zeta);
    const cd n2 = cd(a * (m0 - R * R), 0.0);
    cT0 = n0 + n1 + n2;
    cT1 = cd(0.0, 2.0) * (n2 - n0);
    cT2 = n1 - n0 - n2;
}

inline QuarticCoeffs boundary_quartic(double R, const PrimaryFrame& pf) {
    const double rho2 = pf.rho * pf.rho;
    const double R2 = R * R;
    const double bm = (R - pf.a) * (R - pf.a);
    const double bp = (R + pf.a) * (R + pf.a);
    cd cT0, cT1, cT2;
    t_complex_coeffs(R, pf.a, pf.m0, pf.X, pf.Y, cT0, cT1, cT2);

    const double k = rho2 * R2;
    const double lin0 = k * bm, lin2 = k * (bm + bp), lin4 = k * bp;

    const double a0 = std::norm(cT0);
    const double a1 = 2.0 * (cT0 * std::conj(cT1)).real();
    const double a2 = std::norm(cT1) + 2.0 * (cT0 * std::conj(cT2)).real();
    const double a3 = 2.0 * (cT1 * std::conj(cT2)).real();
    const double a4 = std::norm(cT2);

    return QuarticCoeffs{{lin0 - a0, -a1, lin2 - a2, -a3, lin4 - a4}};
}

// Exact d P / d(xs, ys, rho, m0, a) in the *primary frame* (xs,ys == X,Y).
// Returns dp[param][coeff], param order (X, Y, rho, m0, a), coeff ascending.
struct QuarticParamJac {
    std::array<std::array<double, 5>, 5> dp;  // dp[j] = d/dP_j of [p0..p4]
};

inline QuarticParamJac boundary_quartic_dp(double R, const PrimaryFrame& pf) {
    const double a = pf.a, m0 = pf.m0, X = pf.X, Y = pf.Y, rho = pf.rho;
    const double rho2 = rho * rho;
    const double k = rho2 * R * R;
    const double bm = (R - a) * (R - a);
    const double bp = (R + a) * (R + a);
    cd cT0, cT1, cT2;
    t_complex_coeffs(R, a, m0, X, Y, cT0, cT1, cT2);
    const cd zeta(X, Y);

    auto assemble = [&](cd dcT0, cd dcT1, cd dcT2, double dlin0, double dlin2,
                        double dlin4) -> std::array<double, 5> {
        const double da0 = 2.0 * (dcT0 * std::conj(cT0)).real();
        const double da1 =
            2.0 * (dcT0 * std::conj(cT1) + cT0 * std::conj(dcT1)).real();
        const double da2 =
            2.0 * (dcT1 * std::conj(cT1)).real() +
            2.0 * (dcT0 * std::conj(cT2) + cT0 * std::conj(dcT2)).real();
        const double da3 =
            2.0 * (dcT1 * std::conj(cT2) + cT1 * std::conj(dcT2)).real();
        const double da4 = 2.0 * (dcT2 * std::conj(cT2)).real();
        return {dlin0 - da0, -da1, dlin2 - da2, -da3, dlin4 - da4};
    };
    auto dcT = [](cd dn0, cd dn1, cd dn2, cd& d0, cd& d1, cd& d2) {
        d0 = dn0 + dn1 + dn2;
        d1 = cd(0.0, 2.0) * (dn2 - dn0);
        d2 = dn1 - dn0 - dn2;
    };

    QuarticParamJac out;
    cd d0, d1, d2;

    // d/dX : dn0 = -R^2, dn1 = a R, dn2 = 0
    dcT(cd(-R * R, 0.0), cd(a * R, 0.0), cd(0.0, 0.0), d0, d1, d2);
    out.dp[0] = assemble(d0, d1, d2, 0.0, 0.0, 0.0);
    // d/dY : dn0 = -i R^2, dn1 = i a R, dn2 = 0
    dcT(cd(0.0, -R * R), cd(0.0, a * R), cd(0.0, 0.0), d0, d1, d2);
    out.dp[1] = assemble(d0, d1, d2, 0.0, 0.0, 0.0);
    // d/drho : T unaffected, dk = 2 rho R^2
    {
        const double dk = 2.0 * rho * R * R;
        out.dp[2] = assemble(cd(0, 0), cd(0, 0), cd(0, 0), dk * bm,
                             dk * (bm + bp), dk * bp);
    }
    // d/dm0 : dn2 = a
    dcT(cd(0, 0), cd(0, 0), cd(a, 0), d0, d1, d2);
    out.dp[3] = assemble(d0, d1, d2, 0.0, 0.0, 0.0);
    // d/da : dn1 = R zeta, dn2 = m0 - R^2 ; dbm = -2(R-a), dbp = 2(R+a)
    {
        const double dbm = -2.0 * (R - a), dbp = 2.0 * (R + a);
        dcT(cd(0, 0), R * zeta, cd(m0 - R * R, 0.0), d0, d1, d2);
        out.dp[4] = assemble(d0, d1, d2, k * dbm, k * (dbm + dbp), k * dbp);
    }
    return out;
}

inline double poly_eval(const std::array<double, 5>& c, double t) {
    double r = 0.0;
    for (int i = 4; i >= 0; --i) r = r * t + c[i];
    return r;
}

}  // namespace lcbinint::holonomic
