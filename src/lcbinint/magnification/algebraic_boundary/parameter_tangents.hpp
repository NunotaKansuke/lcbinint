#pragma once
// Parameter conversion shared by the value and tangent controllers.
#include <array>
#include <cmath>
#include "lcbinint/magnification/algebraic_boundary/angular_rule.hpp"
#include "lcbinint/magnification/algebraic_boundary/cells.hpp"
#include "lcbinint/magnification/algebraic_boundary/fp_env.hpp"
#include "lcbinint/magnification/algebraic_boundary/lens_frame.hpp"
#include "lcbinint/magnification/algebraic_boundary/radius_terms.hpp"
#include "lcbinint/magnification/algebraic_boundary/status.hpp"
namespace lcbinint::algebraic_boundary {
constexpr double kPi = 3.14159265358979323846;
constexpr double kSingularZetaTol = 1.0e-3;
// dvec_internal order: (dX, dY, drho, dm0, da_pf).  Returns d/d(xs,ys,rho,q,a).
inline std::array<double, 5> internal_to_user_jac(
    const std::array<double, 5>& d, const LensParams& p) {
    const double q = p.q, a = p.a;
    const double inv = 1.0 / (1.0 + q);
    const double dm0_dq = -inv * inv;
    const double dm1_dq = inv * inv;
    std::array<double, 5> out{d[0], d[1], d[2], 0.0, 0.0};
    if (p.barycentric) {
        out[3] = d[3] * dm0_dq + d[0] * (a * dm1_dq);
        out[4] = d[4] + d[0] * (q * inv);
    } else {
        out[3] = d[3] * dm0_dq;
        out[4] = d[4];
    }
    return out;
}

inline bool near_origin_source(const PrimaryFrame& pf,
                               double tol = kSingularZetaTol) {
    return std::hypot(pf.X, pf.Y) < tol;
}

}
