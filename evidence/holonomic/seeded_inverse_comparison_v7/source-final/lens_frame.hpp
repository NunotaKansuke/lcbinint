#pragma once

// ATPT holonomic solver (M7) -- lens parametrisation.
//
// Ports python/lcbinint/holonomic_ref/polynomial_family.py:LensParams.
//
//   lenses at 0 and a ; masses m0 = 1/(1+q), m1 = q/(1+q)
//   user parameters      p = (xs, ys, rho, q, a)
//   primary-frame source (X, Y):  barycentric  -> X = xs + m1 a, Y = ys
//                                 else         -> X = xs,        Y = ys
//   internal parameters  P = (X, Y, rho, m0, a)   (jacobian.py convention)

#include <cmath>

namespace lcbinint::holonomic {

struct LensParams {
    double xs = 0.0;
    double ys = 0.0;
    double rho = 0.0;
    double q = 0.0;
    double a = 0.0;
    bool barycentric = true;

    double m0() const { return 1.0 / (1.0 + q); }
    double m1() const { return q / (1.0 + q); }

    // primary-frame source coordinates
    double X() const { return barycentric ? xs + m1() * a : xs; }
    double Y() const { return ys; }
};

// Internal (primary-frame) parameter pack passed to the polynomial family
// and phi: order (a, m0, X, Y, rho) -- matches the Python `pf` tuple used in
// jacobian.py (a, m0, X, Y, rho).
struct PrimaryFrame {
    double a;
    double m0;
    double X;
    double Y;
    double rho;

    static PrimaryFrame from(const LensParams& p) {
        return PrimaryFrame{p.a, p.m0(), p.X(), p.Y(), p.rho};
    }
};

}  // namespace lcbinint::holonomic
