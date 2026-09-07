#pragma once

#include "lcbinint/magnification/algebraic_boundary.hpp"
#include <cmath>

namespace lcbinint::magnification {

struct PhysicalAngularBoundary {
    long double residual;
    long double derivative;
};

// Evaluate the uncleared equation without cancellation between O(1) quartic
// coefficients. Promote the same frozen binary64 geometry used by the event
// polynomial. Extended precision is only a polishing aid, not a certificate.
inline PhysicalAngularBoundary physical_angular_boundary(
    const AlgebraicBoundaryLens& lens, SourcePosition source, double rho,
    double radius, long double theta)
{
    const long double a = lens.lens_position, m1 = lens.m1, m2 = lens.m2;
    const long double sx = source.x + lens.lens_position * lens.m1;
    const long double x = radius * std::cos(theta), y = radius * std::sin(theta);
    const long double xp = -y, yp = x, xa = x - a;
    const long double d0 = x*x + y*y, d1 = xa*xa + y*y;
    const long double d1p = 2*(xa*xp + y*yp);
    const long double dx = x - m1*xa/d1 - m2*x/d0 - sx;
    const long double dy = y - m1*y/d1 - m2*y/d0 - source.y;
    const long double dxp = xp - m1*(xp*d1-xa*d1p)/(d1*d1) - m2*xp/d0;
    const long double dyp = yp - m1*(yp*d1-y*d1p)/(d1*d1) - m2*yp/d0;
    return {dx*dx + dy*dy - static_cast<long double>(rho)*rho,
            2*(dx*dxp + dy*dyp)};
}

}  // namespace lcbinint::magnification
