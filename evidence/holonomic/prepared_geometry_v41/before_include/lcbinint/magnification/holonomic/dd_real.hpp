#pragma once

// ATPT holonomic solver (M7) -- double-double real arithmetic (Phase B1).
//
// D14(v) (degree 14, monomial basis kappa ~ 1e9) currently gets an
// __float128 (libquadmath, ~113-bit, fully software) Aberth-Ehrlich
// polish.  That polish is ~68% of `radial_events` and ~32% of the full
// value+5-Jacobian epoch (evidence/holonomic/d14_solve_cost_split.txt).
//
// `DD` is an unevaluated-sum double-double: value = hi + lo, |lo| <= ulp(hi)/2.
// ~106 significant bits, all in hardware `double` + one FMA per product
// (error-free transformations, Dekker/Knuth/Kahan).  kappa ~ 1e9 sits well
// inside its range, so the Aberth *polish* can run here instead of in
// libquadmath; only a genuinely near-multiple cluster (local kappa > ~1e14)
// still needs __float128.
//
// The type deliberately mirrors the small surface `aberth<R>()` /
// `Cplx<R>` need (arithmetic ops, ordering, `qabs_`, `qsqrt_`, an explicit
// `double` cast, construction from `double`), so `aberth<DD>` and
// `Cplx<DD>` instantiate with no change to poly_roots.hpp.
//
// EFT REQUIRES round-to-nearest and NO flush-to-zero: the `lo` limbs and
// `two_*` error terms must be representable.  In this solver's magnitude
// regime (balanced roots O(10), coeffs <= ~1e11, residuals ~1e-26 abs)
// nothing approaches the 2.2e-308 normal floor, so FTZ/DAZ is *inert*
// here -- but the compensated solve still wraps itself in
// `ScopedNoFlushDenormals` so the guarantee is structural, not empirical.
//
// Contraction note: built with -ffp-contract=fast.  `two_sum` uses only
// +/- (unaffected by mul-add contraction); `two_prod` uses an explicit
// `std::fma` against a separately-stored product, which is contraction
// immune by construction.

#include <cmath>

namespace lcbinint::holonomic {

struct DD {
    double hi{0.0}, lo{0.0};
    DD() = default;
    DD(double h) : hi(h), lo(0.0) {}                 // NOLINT: implicit on purpose
    DD(double h, double l) : hi(h), lo(l) {}
    explicit operator double() const { return hi + lo; }
};

// ---- error-free transformations ---------------------------------------
inline void dd_two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
inline void dd_fast_two_sum(double a, double b, double& s, double& e) {
    // requires |a| >= |b|
    s = a + b;
    e = b - (s - a);
}
inline void dd_two_prod(double a, double b, double& p, double& e) {
    p = a * b;
    e = std::fma(a, b, -p);
}

// ---- arithmetic ------------------------------------------------------
inline DD operator-(DD a) { return {-a.hi, -a.lo}; }

inline DD operator+(DD a, DD b) {
    double s, e;
    dd_two_sum(a.hi, b.hi, s, e);
    e += a.lo + b.lo;
    double hi, lo;
    dd_fast_two_sum(s, e, hi, lo);
    return {hi, lo};
}
inline DD operator-(DD a, DD b) { return a + (-b); }

inline DD operator*(DD a, DD b) {
    double p, e;
    dd_two_prod(a.hi, b.hi, p, e);
    e += a.hi * b.lo + a.lo * b.hi;
    double hi, lo;
    dd_fast_two_sum(p, e, hi, lo);
    return {hi, lo};
}

inline DD operator/(DD a, DD b) {
    double q1 = a.hi / b.hi;
    DD r = a - b * DD(q1);
    double q2 = r.hi / b.hi;
    r = r - b * DD(q2);
    double q3 = r.hi / b.hi;
    double hi, lo;
    dd_fast_two_sum(q1, q2, hi, lo);
    return DD(hi, lo) + DD(q3);
}

// ---- ordering (lexicographic on the unevaluated sum) -----------------
inline bool operator>(DD a, DD b) {
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}
inline bool operator<(DD a, DD b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

// ---- the two primitives aberth<R>()/Cplx<R> call by name -------------
inline DD qabs_(DD a) { return (a.hi < 0.0) ? DD(-a.hi, -a.lo) : a; }

inline DD qsqrt_(DD a) {
    if (a.hi <= 0.0) return DD(0.0);
    // Karp's high-precision sqrt: one Newton step in DD off a double seed.
    double x = 1.0 / std::sqrt(a.hi);   // ~ 1/sqrt(a), double precision
    double ax = a.hi * x;               // ~ sqrt(a)
    DD axd(ax);
    DD diff = a - axd * axd;            // residual, exact head
    DD corr = DD(diff.hi * (x * 0.5));  // + diff/(2 sqrt(a))
    return axd + corr;
}

// ---- interop with the retained __float128 path ----------------------
inline DD dd_from_qf(__float128 q) {
    double hi = (double)q;
    double lo = (double)(q - (__float128)hi);
    return {hi, lo};
}
inline __float128 qf_from_dd(DD d) {
    return (__float128)d.hi + (__float128)d.lo;
}

}  // namespace lcbinint::holonomic
