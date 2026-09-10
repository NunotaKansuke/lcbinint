#pragma once

// ATPT holonomic solver (M7) -- polynomial root finding.
//
// Two needs, two precisions:
//
//  * the boundary quartic P(t; R) (degree 4, ascending real coeffs) -- arc
//    endpoints; well-conditioned (radial_events.py: "np.roots ... accurate"),
//    solved at `double` with Aberth-Ehrlich;
//  * the radial-event polynomials D14(v) (degree 14), p4(R) (degree 6),
//    L(v) (degree 2).  D14's monomial basis is ill-conditioned
//    (radial_events.py docstring, kappa ~ 1e9); a `double` companion
//    silently turns real double roots into spurious complex pairs and
//    misses band births.  Solved with Aberth-Ehrlich in `__float128`
//    (~34 digits): the retained D14 coefficients are built with no
//    catastrophic cancellation (only the two spurious top coefficients,
//    ~1e-25 relative, cancel -- they are trimmed), and 113-bit
//    root-finding recovers every real root to < 1e-9 even for the
//    near-double clusters (verified against sympy exact isolation on the
//    7 M7 cases).
//
// Aberth-Ehrlich: simultaneous iteration with cubic local convergence,
// robust for clustered roots.  Fixed work, no heap, no dynamic dispatch.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include <quadmath.h>

namespace lcbinint::holonomic {

// ---- minimal complex over an arbitrary real type ------------------------
template <class R>
struct Cplx {
    R re{}, im{};
    Cplx() = default;
    Cplx(R r, R i) : re(r), im(i) {}
    explicit Cplx(R r) : re(r), im(R(0)) {}
};
template <class R> inline Cplx<R> operator+(Cplx<R> a, Cplx<R> b) { return {a.re + b.re, a.im + b.im}; }
template <class R> inline Cplx<R> operator-(Cplx<R> a, Cplx<R> b) { return {a.re - b.re, a.im - b.im}; }
template <class R> inline Cplx<R> operator*(Cplx<R> a, Cplx<R> b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

inline double qabs_(double x) { return std::fabs(x); }
inline double qsqrt_(double x) { return std::sqrt(x); }
inline __float128 qabs_(__float128 x) { return fabsq(x); }
inline __float128 qsqrt_(__float128 x) { return sqrtq(x); }
inline bool qfinite_(double x) { return std::isfinite(x); }
inline bool qfinite_(long double x) { return std::isfinite(x); }
inline bool qfinite_(__float128 x) { return finiteq(x); }
template <class R> inline bool qfinite_(const R&) { return true; }

// The original quotient is retained behind an environment switch so that
// isolated A/B runs can compare arithmetic changes in a fresh process.  The
// default reciprocal first scales the denominator, avoiding the squared-norm
// overflow/underflow path and reducing a complex quotient to one reciprocal
// followed by a multiply.
inline bool holo_legacy_complex_ops() {
    static const bool legacy = [] {
        const char* e = std::getenv("HOLO_D14_LEGACY_COMPLEX");
        return e && e[0] == '1';
    }();
    return legacy;
}

template <class R>
inline Cplx<R> cdiv_legacy(Cplx<R> a, Cplx<R> b) {
    R d = b.re * b.re + b.im * b.im;
    return {(a.re * b.re + a.im * b.im) / d,
            (a.im * b.re - a.re * b.im) / d};
}

template <class R>
inline Cplx<R> crecip(Cplx<R> b) {
    const R ar = qabs_(b.re), ai = qabs_(b.im);
    const R scale = ar > ai ? ar : ai;
    if (!(scale > R(0))) {
        // Preserve the old IEEE invalid/zero behaviour for a zero divisor;
        // all valid Aberth denominators take the scaled branch.
        return cdiv_legacy(Cplx<R>(R(1), R(0)), b);
    }
    const R br = b.re / scale;
    const R bi = b.im / scale;
    const R n2 = br * br + bi * bi;
    const R inv_scale = R(1) / scale;
    return {br * inv_scale / n2, -bi * inv_scale / n2};
}

template <class R>
inline Cplx<R> cdiv_fast(Cplx<R> a, Cplx<R> b) {
    // The usual root-scale range fits the squared norm.  Reuse one real
    // reciprocal for both components; this keeps the fast path cheaper than
    // forming a reciprocal complex and multiplying by it.  The scaled path
    // remains available for tiny/large denominators.
    const R d = b.re * b.re + b.im * b.im;
    if (d > R(0) && qfinite_(d)) {
        const R inv = R(1) / d;
        return {(a.re * b.re + a.im * b.im) * inv,
                (a.im * b.re - a.re * b.im) * inv};
    }
    return a * crecip(b);
}

template <class R>
inline Cplx<R> operator/(Cplx<R> a, Cplx<R> b) {
    return holo_legacy_complex_ops() ? cdiv_legacy(a, b) : cdiv_fast(a, b);
}

template <class R> inline R cabs2(Cplx<R> z) { return z.re * z.re + z.im * z.im; }
template <class R> inline R cabs(Cplx<R> z) { return qsqrt_(cabs2(z)); }

// Horner on descending coeffs c[0]*x^n + ... + c[n].
template <class R>
inline Cplx<R> poly_eval_c(const R* c, int n, Cplx<R> x) {
    Cplx<R> r(c[0], R(0));
    for (int i = 1; i <= n; ++i) r = r * x + Cplx<R>(c[i], R(0));
    return r;
}
template <class R>
inline Cplx<R> polyder_eval_c(const R* c, int n, Cplx<R> x) {
    // derivative of the degree-n polynomial, Horner
    Cplx<R> r(c[0] * R(n), R(0));
    for (int i = 1; i < n; ++i) r = r * x + Cplx<R>(c[i] * R(n - i), R(0));
    return r;
}

// Aberth-Ehrlich.  coeffs: descending, length deg+1, coeffs[0] != 0.
// Returns deg complex roots.  `max_iter` fixed; converges in ~10-40 for a
// cold start, ~3 warm.  If `seed` is non-null it is the warm-start guess.
template <class R>
inline std::vector<Cplx<R>> aberth(const R* coeffs, int deg, int max_iter = 200,
                                   const Cplx<R>* seed = nullptr,
                                   R tol_override = R(0),
                                   double* final_step = nullptr,
                                   int* iterations = nullptr) {
    std::vector<Cplx<R>> z(deg);
    if (iterations) *iterations = 0;

    // Initial guesses on a circle of radius ~ Cauchy bound (Aberth's
    // spread).  A warm seed already contains the basin information, so do
    // not spend time scanning all coefficients to rebuild the cold bound.
    R bound = R(1);
    if (!seed) {
        R an = qabs_(coeffs[0]);
        for (int i = 1; i <= deg; ++i) {
            R v = qabs_(coeffs[i]) / an;
            if (v > bound - R(1)) bound = R(1) + v;
        }
    }
    const R pi = R(3.14159265358979323846264338327950288L);
    for (int i = 0; i < deg; ++i) {
        if (seed) {
            z[i] = seed[i];
        } else {
            R ang = (R(2) * pi * R(i)) / R(deg) + R(0.4);
            R rad = bound * (R(0.5) + R(0.5) * R(i) / R(deg));
            z[i] = Cplx<R>(rad * R(cos((double)ang)), rad * R(sin((double)ang)));
        }
    }

    const R tol =
        tol_override > R(0)
            ? tol_override
            : ((sizeof(R) > 8) ? R(1e-24) : R(1e-15));
    R maxstep2 = R(0);
    const bool legacy = holo_legacy_complex_ops();
    for (int it = 0; it < max_iter; ++it) {
        if (iterations) *iterations = it + 1;
        maxstep2 = R(0);
        for (int i = 0; i < deg; ++i) {
            Cplx<R> p = poly_eval_c(coeffs, deg, z[i]);
            Cplx<R> dp = polyder_eval_c(coeffs, deg, z[i]);
            Cplx<R> sum(R(0), R(0));
            for (int j = 0; j < deg; ++j) {
                if (j == i) continue;
                Cplx<R> d = z[i] - z[j];
                sum = sum + Cplx<R>(R(1), R(0)) / d;
            }
            // Algebraically this is p / (p' - p sum_j 1/(z-z_j)).  It
            // removes the intermediate complex quotient p/p' from the hot
            // loop.  Keep the old form under the A/B switch to isolate the
            // effect of the rewrite from the reciprocal implementation.
            Cplx<R> w;
            if (legacy) {
                Cplx<R> ratio = p / dp;
                Cplx<R> denom = Cplx<R>(R(1), R(0)) - ratio * sum;
                w = ratio / denom;
            } else {
                w = p / (dp - p * sum);
            }
            z[i] = z[i] - w;
            R s2 = cabs2(w);
            if (s2 > maxstep2) maxstep2 = s2;
        }
        if (maxstep2 < tol * tol) break;
    }
    if (final_step) *final_step = (double)qsqrt_(maxstep2);
    return z;
}

// Positive real roots (Im/|Re| below `im_rel`), ascending, deduped at
// `merge`.  `x0` optionally shifts the "positive" test (unused here).
template <class R>
inline std::vector<double> positive_real_roots(const std::vector<Cplx<R>>& z,
                                               double im_rel, double merge) {
    std::vector<double> out;
    for (const auto& r : z) {
        double re = (double)r.re, im = (double)r.im;
        if (re <= 0.0) continue;
        if (qabs_((double)im) > im_rel * (1.0 + qabs_(re))) continue;
        out.push_back(re);
    }
    std::sort(out.begin(), out.end());
    std::vector<double> ded;
    for (double x : out)
        if (ded.empty() || x - ded.back() > merge) ded.push_back(x);
    return ded;
}

// Real roots of a real polynomial (any sign), ascending, deduped.
template <class R>
inline std::vector<double> real_roots(const std::vector<Cplx<R>>& z,
                                      double im_rel, double merge) {
    std::vector<double> out;
    for (const auto& r : z) {
        double re = (double)r.re, im = (double)r.im;
        if (qabs_((double)im) > im_rel * (1.0 + qabs_(re))) continue;
        out.push_back(re);
    }
    std::sort(out.begin(), out.end());
    std::vector<double> ded;
    for (double x : out)
        if (ded.empty() || x - ded.back() > merge) ded.push_back(x);
    return ded;
}

// ---- boundary quartic: real roots at double ----------------------------
// p ascending [p0..p4].  Returns the real roots (any multiplicity once),
// ascending.  Matches jacobian._real_root_thetas' np.roots + real filter
// (_ROOT_IM_REL = 1e-8, near-double merge 1e-11 -- applied by the caller
// in theta space; here we merge in t at 1e-11).
inline std::vector<double> quartic_real_roots(const std::array<double, 5>& p) {
    // descending, normalised
    double c[5] = {p[4], p[3], p[2], p[1], p[0]};
    int deg = 4;
    while (deg > 0 && c[0] == 0.0) {
        for (int i = 0; i < deg; ++i) c[i] = c[i + 1];
        --deg;
    }
    if (deg <= 0) return {};
    auto z = aberth<double>(c, deg, 120);
    return real_roots(z, 1e-8, 1e-11);
}

}  // namespace lcbinint::holonomic
