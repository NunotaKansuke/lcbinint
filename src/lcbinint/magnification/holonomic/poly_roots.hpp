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
template <class R> inline Cplx<R> operator/(Cplx<R> a, Cplx<R> b) {
    R d = b.re * b.re + b.im * b.im;
    return {(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d};
}

inline double qabs_(double x) { return std::fabs(x); }
inline double qsqrt_(double x) { return std::sqrt(x); }
inline __float128 qabs_(__float128 x) { return fabsq(x); }
inline __float128 qsqrt_(__float128 x) { return sqrtq(x); }

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
                                   const Cplx<R>* seed = nullptr) {
    std::vector<Cplx<R>> z(deg);

    // initial guesses on a circle of radius ~ Cauchy bound (Aberth's spread)
    R an = qabs_(coeffs[0]);
    R bound = R(0);
    for (int i = 1; i <= deg; ++i) {
        R v = qabs_(coeffs[i]) / an;
        if (v > bound) bound = v;
    }
    bound = R(1) + bound;
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

    const R tol = (sizeof(R) > 8) ? R(1e-30) : R(1e-15);
    for (int it = 0; it < max_iter; ++it) {
        R maxstep = R(0);
        for (int i = 0; i < deg; ++i) {
            Cplx<R> p = poly_eval_c(coeffs, deg, z[i]);
            Cplx<R> dp = polyder_eval_c(coeffs, deg, z[i]);
            Cplx<R> ratio = p / dp;  // Newton step p/p'
            Cplx<R> sum(R(0), R(0));
            for (int j = 0; j < deg; ++j) {
                if (j == i) continue;
                Cplx<R> d = z[i] - z[j];
                sum = sum + Cplx<R>(R(1), R(0)) / d;
            }
            Cplx<R> denom = Cplx<R>(R(1), R(0)) - ratio * sum;
            Cplx<R> w = ratio / denom;
            z[i] = z[i] - w;
            R s = cabs(w);
            if (s > maxstep) maxstep = s;
        }
        if (maxstep < tol) break;
    }
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
