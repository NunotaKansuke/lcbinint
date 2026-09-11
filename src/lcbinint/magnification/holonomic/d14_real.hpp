#pragma once

// Small twofold arithmetic surface dedicated to the D14 root kernel.
//
// This is deliberately separate from DD.  It has the same two-double
// representation, but exposes only the operations used by the D14 block
// evaluator and root correction.  In particular, it does not grow into a
// general arbitrary-precision number type.  The qf conversion helpers are
// retained for the existing independent certificate and backstop.

#include <cmath>

#include <quadmath.h>

namespace lcbinint::holonomic {

struct D14Real {
    double hi{0.0};
    double lo{0.0};

    D14Real() = default;
    D14Real(double h) : hi(h), lo(0.0) {}  // intentional implicit conversion
    D14Real(double h, double l) : hi(h), lo(l) {}

    explicit operator double() const { return hi + lo; }
    bool finite() const { return std::isfinite(hi) && std::isfinite(lo); }
};

inline D14Real d14_two_sum(double a, double b) {
    const double s = a + b;
    const double bb = s - a;
    const double e = (a - (s - bb)) + (b - bb);
    return {s, e};
}

inline D14Real d14_normalize(double hi, double lo) {
    const double s = hi + lo;
    const double e = lo - (s - hi);
    return {s, e};
}

inline D14Real d14_from_qf(__float128 x) {
    const double hi = static_cast<double>(x);
    const double lo = static_cast<double>(x - static_cast<__float128>(hi));
    return {hi, lo};
}

inline __float128 d14_to_qf(D14Real x) {
    return static_cast<__float128>(x.hi) + static_cast<__float128>(x.lo);
}

inline D14Real operator-(D14Real x) { return {-x.hi, -x.lo}; }

inline D14Real operator+(D14Real a, D14Real b) {
    const D14Real s = d14_two_sum(a.hi, b.hi);
    return d14_normalize(s.hi, s.lo + a.lo + b.lo);
}

inline D14Real operator-(D14Real a, D14Real b) { return a + (-b); }

inline D14Real operator*(D14Real a, D14Real b) {
    const double p = a.hi * b.hi;
    const double e = std::fma(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi;
    return d14_normalize(p, e);
}

inline D14Real operator/(D14Real a, D14Real b) {
    const double q1 = a.hi / b.hi;
    D14Real r = a - b * D14Real(q1);
    const double q2 = r.hi / b.hi;
    r = r - b * D14Real(q2);
    const double q3 = r.hi / b.hi;
    const D14Real q12 = d14_normalize(q1, q2);
    return q12 + D14Real(q3);
}

inline bool operator>(D14Real a, D14Real b) {
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}
inline bool operator<(D14Real a, D14Real b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
inline bool operator>=(D14Real a, D14Real b) { return !(a < b); }
inline bool operator<=(D14Real a, D14Real b) { return !(a > b); }
inline bool operator==(D14Real a, D14Real b) {
    return a.hi == b.hi && a.lo == b.lo;
}
inline bool operator!=(D14Real a, D14Real b) { return !(a == b); }

inline D14Real qabs_(D14Real a) {
    return a.hi < 0.0 ? D14Real(-a.hi, -a.lo) : a;
}

inline D14Real qsqrt_(D14Real a) {
    if (a.hi <= 0.0) return D14Real(0.0);
    const double x = 1.0 / std::sqrt(a.hi);
    const double ax = a.hi * x;
    const D14Real axd(ax);
    const D14Real diff = a - axd * axd;
    return axd + D14Real(diff.hi * (x * 0.5));
}

inline bool qfinite_(D14Real a) { return a.finite(); }

}  // namespace lcbinint::holonomic
