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

// Exact reciprocal chart transform for u = -1/t:
//
//     u^4 P(-1/u) = p4 - p3 u + p2 u^2 - p1 u^3 + p0 u^4.
//
// This is kept as an algebraic primitive for the isolated chart experiment;
// the V2 router remains on the incumbent t chart until a whole-epoch
// condition-driven A/B demonstrates a benefit.
inline QuarticCoeffs boundary_quartic_reciprocal(const QuarticCoeffs& in) {
    return QuarticCoeffs{{in.p[4], -in.p[3], in.p[2], -in.p[1], in.p[0]}};
}

inline double reciprocal_t_from_u(double u) { return -1.0 / u; }
inline double reciprocal_u_from_t(double t) { return -1.0 / t; }

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

// Exact d P / d R (ascending [dp0..dp4]).  Closed form -- P's coefficients
// are explicit polynomials in R through k = rho^2 R^2, bm = (R-a)^2,
// bp = (R+a)^2 and the complex T coefficients.  Ports
// polynomial_family.boundary_quartic_dR.
inline QuarticCoeffs boundary_quartic_dR(double R, const PrimaryFrame& pf) {
    const double a = pf.a, m0 = pf.m0, X = pf.X, Y = pf.Y, rho = pf.rho;
    const double rho2 = rho * rho;
    const double k = rho2 * R * R, dk = 2.0 * rho2 * R;
    const double bm = (R - a) * (R - a), bp = (R + a) * (R + a);
    const double dbm = 2.0 * (R - a), dbp = 2.0 * (R + a);

    const double dlin0 = dk * bm + k * dbm;
    const double dlin2 = dk * (bm + bp) + k * (dbm + dbp);
    const double dlin4 = dk * bp + k * dbp;

    cd cT0, cT1, cT2;
    t_complex_coeffs(R, a, m0, X, Y, cT0, cT1, cT2);
    const cd zeta(X, Y);
    const cd dn0 = -2.0 * zeta * R;
    const cd dn1 = cd(3.0 * R * R - 1.0, 0.0) + a * zeta;
    const cd dn2(-2.0 * a * R, 0.0);
    const cd dcT0 = dn0 + dn1 + dn2;
    const cd dcT1 = cd(0.0, 2.0) * (dn2 - dn0);
    const cd dcT2 = dn1 - dn0 - dn2;

    const double da0 = 2.0 * (dcT0 * std::conj(cT0)).real();
    const double da1 =
        2.0 * (dcT0 * std::conj(cT1) + cT0 * std::conj(dcT1)).real();
    const double da2 =
        2.0 * (dcT1 * std::conj(cT1)).real() +
        2.0 * (dcT0 * std::conj(cT2) + cT0 * std::conj(dcT2)).real();
    const double da3 =
        2.0 * (dcT1 * std::conj(cT2) + cT1 * std::conj(dcT2)).real();
    const double da4 = 2.0 * (dcT2 * std::conj(cT2)).real();

    return QuarticCoeffs{
        {dlin0 - da0, -da1, dlin2 - da2, -da3, dlin4 - da4}};
}

// Exact d P / d(xs, ys, rho, m0, a) in the *primary frame* (xs,ys == X,Y).
// Returns dp[param][coeff], param order (X, Y, rho, m0, a), coeff ascending.
struct QuarticParamJac {
    std::array<std::array<double, 5>, 5> dp;  // dp[j] = d/dP_j of [p0..p4]
};

inline QuarticParamJac boundary_quartic_reciprocal_dp(
    const QuarticParamJac& in) {
    QuarticParamJac out{};
    for (int j = 0; j < 5; ++j)
        out.dp[j] = {in.dp[j][4], -in.dp[j][3], in.dp[j][2],
                     -in.dp[j][1], in.dp[j][0]};
    return out;
}

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

// Direct local P(R,t) quantities for the physical-fold event refinement.
// This is deliberately separate from the generic R-polynomial family: the
// event solver only needs P, P_t, P_R, P_tt and P_tR at one (R,t), and the
// explicit construction preserves the input precision all the way through.
// T is double, DD, or __float128.  The formulas are the same ones used by
// boundary_quartic()/boundary_quartic_dR(), written without std::complex so
// the DD path does not round high-precision coefficients through binary64.
template <class T>
struct LocalFoldQuantities {
    T P{};
    T Pt{};
    T PR{};
    T Ptt{};
    T Ptr{};
    T Psss{},Pssss{};
};

template <class T>
inline LocalFoldQuantities<T> local_fold_quantities(T R, T t,
                                                    const PrimaryFrame& pf,
                                                    bool reciprocal = false) {
    const T a = T(pf.a), m0 = T(pf.m0), X = T(pf.X), Y = T(pf.Y);
    const T rho = T(pf.rho);
    const T R2 = R * R;
    const T n0r = -X * R2, n0i = -Y * R2;
    const T n1r = R * (R2 - T(1.0) + a * X), n1i = R * a * Y;
    const T n2r = a * (m0 - R2), n2i = T(0.0);
    const T c0r = n0r + n1r + n2r, c0i = n0i + n1i;
    const T c1r = -T(2.0) * (n2i - n0i);
    const T c1i = T(2.0) * (n2r - n0r);
    const T c2r = n1r - n0r - n2r, c2i = n1i - n0i - n2i;

    const T dn0r = -T(2.0) * X * R, dn0i = -T(2.0) * Y * R;
    const T dn1r = T(3.0) * R2 - T(1.0) + a * X, dn1i = a * Y;
    const T dn2r = -T(2.0) * a * R, dn2i = T(0.0);
    const T dc0r = dn0r + dn1r + dn2r, dc0i = dn0i + dn1i;
    const T dc1r = -T(2.0) * (dn2i - dn0i);
    const T dc1i = T(2.0) * (dn2r - dn0r);
    const T dc2r = dn1r - dn0r - dn2r, dc2i = dn1i - dn0i - dn2i;

    const T dot00 = c0r * c0r + c0i * c0i;
    const T dot01 = c0r * c1r + c0i * c1i;
    const T dot02 = c0r * c2r + c0i * c2i;
    const T dot11 = c1r * c1r + c1i * c1i;
    const T dot12 = c1r * c2r + c1i * c2i;
    const T dot22 = c2r * c2r + c2i * c2i;
    const T d00 = c0r * dc0r + c0i * dc0i;
    const T d01 = dc0r * c1r + dc0i * c1i + c0r * dc1r + c0i * dc1i;
    const T d02 = dc0r * c2r + dc0i * c2i + c0r * dc2r + c0i * dc2i;
    const T d11 = c1r * dc1r + c1i * dc1i;
    const T d12 = dc1r * c2r + dc1i * c2i + c1r * dc2r + c1i * dc2i;
    const T d22 = c2r * dc2r + c2i * dc2i;

    const T k = rho * rho * R2, dk = T(2.0) * rho * rho * R;
    const T rm = R - a, rp = R + a;
    const T bm = rm * rm, bp = rp * rp;
    const T dbm = T(2.0) * rm, dbp = T(2.0) * rp;
    const T p0 = k * bm - dot00;
    const T p1 = -T(2.0) * dot01;
    const T p2 = k * (bm + bp) - dot11 - T(2.0) * dot02;
    const T p3 = -T(2.0) * dot12;
    const T p4 = k * bp - dot22;
    const T r0 = dk * bm + k * dbm - T(2.0) * d00;
    const T r1 = -T(2.0) * d01;
    const T r2 = dk * (bm + bp) + k * (dbm + dbp) -
                 T(2.0) * d11 - T(2.0) * d02;
    const T r3 = -T(2.0) * d12;
    const T r4 = dk * bp + k * dbp - T(2.0) * d22;

    // Exact homogeneous reversal; no division by the chart coordinate.
    const T c0=reciprocal?p4:p0,c1=reciprocal?-p3:p1,c2=p2;
    const T c3=reciprocal?-p1:p3,c4=reciprocal?p0:p4;
    const T d0=reciprocal?r4:r0,d1=reciprocal?-r3:r1,d2=r2;
    const T d3=reciprocal?-r1:r3,d4=reciprocal?r0:r4;
    LocalFoldQuantities<T> out;
    out.P = (((c4 * t + c3) * t + c2) * t + c1) * t + c0;
    out.Pt = ((T(4.0) * c4 * t + T(3.0) * c3) * t +
              T(2.0) * c2) * t + c1;
    out.PR = (((d4 * t + d3) * t + d2) * t + d1) * t + d0;
    out.Ptt = (T(12.0) * c4 * t + T(6.0) * c3) * t + T(2.0) * c2;
    out.Psss=T(6.0)*c3+T(24.0)*c4*t;
    out.Pssss=T(24.0)*c4;
    out.Ptr = ((T(4.0) * d4 * t + T(3.0) * d3) * t +
               T(2.0) * d2) * t + d1;
    return out;
}

inline double poly_eval(const std::array<double, 5>& c, double t) {
    double r = 0.0;
    for (int i = 4; i >= 0; --i) r = r * t + c[i];
    return r;
}

}  // namespace lcbinint::holonomic
