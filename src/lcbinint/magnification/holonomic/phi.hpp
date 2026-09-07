#pragma once

// ATPT holonomic solver (M7) -- phi and its closed-form gradient.
// Direct port of python/lcbinint/holonomic_ref/jacobian.py:phi_grad.
//
//   phi = 1 - |f(z) - zeta|^2 / rho^2,  z = R e^{i theta}
//   f(z) = z - m0/zbar - m1/(zbar - a),  m1 = 1 - m0,  zeta = X + iY
//
// Derivative order matches jacobian.py: (dphi/dX, dphi/dY, dphi/drho,
// dphi/dm0, dphi/da).
//
// Complex arithmetic is hand-rolled on real/imag pairs -- no
// std::complex (its C99 Annex-G inf/nan bookkeeping on every divide is
// ~4x slower and never needed here: zbar and zbar-a are bounded away
// from 0 on the integration cells).  This is the hottest routine in the
// solver (tens of thousands of calls per epoch).

#include <array>
#include <cmath>

#include "lcbinint/magnification/holonomic/lens_frame.hpp"

namespace lcbinint::holonomic {

struct PhiGrad {
    double phi;
    double dphi_dtheta;
    std::array<double, 5> dP;  // d/d(X, Y, rho, m0, a)
};

namespace phi_detail {
struct C {
    double re, im;
};
inline C operator+(C a, C b) { return {a.re + b.re, a.im + b.im}; }
inline C operator-(C a, C b) { return {a.re - b.re, a.im - b.im}; }
inline C operator*(C a, C b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
inline C operator/(C a, C b) {
    double d = 1.0 / (b.re * b.re + b.im * b.im);
    return {(a.re * b.re + a.im * b.im) * d, (a.im * b.re - a.re * b.im) * d};
}
inline C conj(C a) { return {a.re, -a.im}; }
inline C muli(C a) { return {-a.im, a.re}; }  // i * a
inline double re_prod(C a, C b) {  // Re(a * b)
    return a.re * b.re - a.im * b.im;
}
}  // namespace phi_detail

inline PhiGrad phi_grad(double R, double theta, const PrimaryFrame& pf) {
    using phi_detail::C;
    using phi_detail::conj;
    using phi_detail::muli;
    using phi_detail::re_prod;
    const double a = pf.a, m0 = pf.m0, X = pf.X, Y = pf.Y, rho = pf.rho;
    double ct, st;
    __builtin_sincos(theta, &st, &ct);
    const C z{R * ct, R * st};
    const C zb{z.re, -z.im};
    const double m1 = 1.0 - m0;
    const C zba = zb - C{a, 0.0};
    const C inv_zb = C{1.0, 0.0} / zb;
    const C inv_zba = C{1.0, 0.0} / zba;
    const C f = z - C{m0, 0.0} * inv_zb - C{m1, 0.0} * inv_zba;
    const C g = f - C{X, Y};
    const double r2 = rho * rho;
    const double inv_r2 = 1.0 / r2;
    const double g2 = g.re * g.re + g.im * g.im;

    PhiGrad out;
    out.phi = 1.0 - g2 * inv_r2;

    const C inv_zb2 = inv_zb * inv_zb;
    const C inv_zba2 = inv_zba * inv_zba;
    const C fzb = C{m0, 0.0} * inv_zb2 + C{m1, 0.0} * inv_zba2;
    const C dg_dtheta = muli(z) - muli(zb * fzb);
    out.dphi_dtheta = -(2.0 * inv_r2) * re_prod(conj(g), dg_dtheta);

    const double dphi_dX = (2.0 * inv_r2) * g.re;
    const double dphi_dY = (2.0 * inv_r2) * g.im;
    const double dphi_drho = 2.0 * g2 / (rho * r2);
    const C df_dm0 = C{0.0, 0.0} - inv_zb + inv_zba;
    const double dphi_dm0 = -(2.0 * inv_r2) * re_prod(conj(g), df_dm0);
    const C df_da = C{-m1, 0.0} * inv_zba2;
    const double dphi_da = -(2.0 * inv_r2) * re_prod(conj(g), df_da);
    out.dP = {dphi_dX, dphi_dY, dphi_drho, dphi_dm0, dphi_da};
    return out;
}

// phi only, straight from the lens equation (topology._phi) -- the
// independent evaluation used for arc mid-point sign tests.
inline double phi_lens(double R, double theta, const PrimaryFrame& pf) {
    using phi_detail::C;
    double ct, st;
    __builtin_sincos(theta, &st, &ct);
    const C z{R * ct, R * st};
    const C zb{z.re, -z.im};
    const C zba = zb - C{pf.a, 0.0};
    const C fz = z - C{pf.m0, 0.0} / zb - C{1.0 - pf.m0, 0.0} / zba;
    const C d = fz - C{pf.X, pf.Y};
    return 1.0 - (d.re * d.re + d.im * d.im) / (pf.rho * pf.rho);
}

}  // namespace lcbinint::holonomic
