#pragma once

// ATPT holonomic solver (M7) -- phi and its closed-form gradient.
// Direct port of python/lcbinint/holonomic_ref/jacobian.py:phi_grad.
//
//   phi = 1 - |f(z) - zeta|^2 / rho^2,  z = R e^{i theta}
//   f(z) = z - m0/zbar - m1/(zbar - a),  m1 = 1 - m0,  zeta = X + iY
//
// Derivative order matches jacobian.py: (dphi/dX, dphi/dY, dphi/drho,
// dphi/dm0, dphi/da).

#include <array>
#include <cmath>
#include <complex>

#include "lcbinint/magnification/holonomic/lens_frame.hpp"

namespace lcbinint::holonomic {

struct PhiGrad {
    double phi;
    double dphi_dtheta;
    std::array<double, 5> dP;  // d/d(X, Y, rho, m0, a)
};

inline PhiGrad phi_grad(double R, double theta, const PrimaryFrame& pf) {
    const double a = pf.a, m0 = pf.m0, X = pf.X, Y = pf.Y, rho = pf.rho;
    const double ct = std::cos(theta), st = std::sin(theta);
    const std::complex<double> z(R * ct, R * st);
    const std::complex<double> zb = std::conj(z);
    const double m1 = 1.0 - m0;
    const std::complex<double> f = z - m0 / zb - m1 / (zb - a);
    const std::complex<double> g = f - std::complex<double>(X, Y);
    const double r2 = rho * rho;
    const double g2 = g.real() * g.real() + g.imag() * g.imag();

    PhiGrad out;
    out.phi = 1.0 - g2 / r2;

    const std::complex<double> fzb =
        m0 / (zb * zb) + m1 / ((zb - a) * (zb - a));
    const std::complex<double> i(0.0, 1.0);
    const std::complex<double> dg_dtheta = i * z - i * zb * fzb;
    out.dphi_dtheta = -(2.0 / r2) * (std::conj(g) * dg_dtheta).real();

    const double dphi_dX = (2.0 / r2) * g.real();
    const double dphi_dY = (2.0 / r2) * g.imag();
    const double dphi_drho = 2.0 * g2 / (rho * r2);
    const std::complex<double> df_dm0 = -1.0 / zb + 1.0 / (zb - a);
    const double dphi_dm0 = -(2.0 / r2) * (std::conj(g) * df_dm0).real();
    const std::complex<double> df_da = -m1 / ((zb - a) * (zb - a));
    const double dphi_da = -(2.0 / r2) * (std::conj(g) * df_da).real();
    out.dP = {dphi_dX, dphi_dY, dphi_drho, dphi_dm0, dphi_da};
    return out;
}

// phi only, straight from the lens equation (topology._phi) -- the
// independent evaluation used for arc mid-point sign tests.
inline double phi_lens(double R, double theta, const PrimaryFrame& pf) {
    const std::complex<double> z(R * std::cos(theta), R * std::sin(theta));
    const std::complex<double> zb = std::conj(z);
    const std::complex<double> fz =
        z - pf.m0 / zb - (1.0 - pf.m0) / (zb - pf.a);
    const std::complex<double> d = fz - std::complex<double>(pf.X, pf.Y);
    return 1.0 - (d.real() * d.real() + d.imag() * d.imag()) / (pf.rho * pf.rho);
}

}  // namespace lcbinint::holonomic
