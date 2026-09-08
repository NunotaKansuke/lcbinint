#pragma once

// ATPT holonomic solver -- binary point-source image positions.
//
// Phase A (unified-engine implementation order, checkpoint sec.0.7): the fast
// radial-band planner (fast_bands.hpp, port of algebraic find_radial_bands) is
// SEED-ANCHORED -- it marches outward from each certified point-source image
// radius.  The holonomic path never needed image seeds before (radial_events /
// D14 gives the band structure straight from the angular discriminant), so this
// header adds the standard binary point-source solve.
//
// Lens: mass m0 at z = 0, mass m1 = 1 - m0 at z = a (real), source w = X + iY,
// all in the PrimaryFrame.  Lens equation
//
//     w = z - m0 / conj(z) - m1 / (conj(z) - a).
//
// Substituting conj(z) from the conjugate equation and clearing denominators
// gives a degree-5 complex polynomial in z (Witt & Mao 1995).  With
//
//     Q(z)  = z^2 - a z                              (from conj(z) = N~/Q)
//     N~(z) = wbar z^2 + (1 - wbar a) z - m0 a
//     M(z)  = N~(z) - a Q(z)
//            = (wbar - a) z^2 + (1 - wbar a + a^2) z - m0 a
//
// the image polynomial is
//
//     P(z) = (w - z) * N~(z) * M(z)  +  m0 * Q(z) * M(z)  +  m1 * Q(z) * N~(z).
//
// The monomial basis is ill-conditioned for extreme mass ratios (q ~ 1e-8:
// two roots crowd the planet at z = a while the coefficients span ~1e16).
// The solve is TWO-TIER: a `double` complex Aberth-Ehrlich pass first, its
// roots verified against the ORIGINAL non-holomorphic lens equation by
// residual; if that yields a clean odd image set (3 or 5, residuals well
// inside tolerance) it is returned directly.  Otherwise the build + solve is
// repeated in __float128 (~34 digits), which resolves the extreme-q clusters.
// Q(z) multiplication can introduce z = 0 / z = a as spurious roots; the
// residual gate (conj(z) taken directly) drops those and any non-image root.
// A binary lens has 3 or 5 real images (odd).

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/poly_roots.hpp"

namespace lcbinint::holonomic {

struct PointImage {
    Cplx<double> z;       // image position
    double radius;        // |z|
    double mag;           // point-source magnification of this image, 1/|det J|
    double residual;      // |lens-equation residual| / (1 + |w|)
    bool positive_parity; // det J > 0
};

struct PointImages {
    std::vector<PointImage> images;  // certified, ascending in radius
    double mu_total = 0.0;           // sum of |mag| -- point-source magnification
    int n_raw = 0;                   // roots returned by the degree-5 solve
    bool reliable = true;            // false: image count even / residuals large
};

namespace pimg_detail {

template <class R>
using Zt = Cplx<R>;

// ascending-coeff complex polynomial product / sum / scale
template <class R>
inline std::vector<Zt<R>> cmul(const std::vector<Zt<R>>& a,
                               const std::vector<Zt<R>>& b) {
    std::vector<Zt<R>> r(a.size() + b.size() - 1, Zt<R>(R(0), R(0)));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) r[i + j] = r[i + j] + a[i] * b[j];
    return r;
}
template <class R>
inline std::vector<Zt<R>> cadd(const std::vector<Zt<R>>& a,
                               const std::vector<Zt<R>>& b) {
    std::vector<Zt<R>> r = a.size() >= b.size() ? a : b;
    const std::vector<Zt<R>>& s = a.size() >= b.size() ? b : a;
    for (size_t i = 0; i < s.size(); ++i) r[i] = r[i] + s[i];
    return r;
}
template <class R>
inline std::vector<Zt<R>> cscale(std::vector<Zt<R>> a, R s) {
    for (auto& c : a) c = c * Zt<R>(s, R(0));
    return a;
}

// Complex Aberth-Ehrlich on an ascending-coefficient polynomial.
template <class R>
inline std::vector<Zt<R>> complex_aberth(const std::vector<Zt<R>>& asc,
                                         int max_iter, R tol) {
    const int deg = (int)asc.size() - 1;
    std::vector<Zt<R>> z(deg);
    // Cauchy-bound spread of initial guesses
    R an = cabs(asc[deg]);
    R bound = R(0);
    for (int i = 0; i < deg; ++i) {
        R v = cabs(asc[i]) / an;
        if (v > bound) bound = v;
    }
    bound = R(1) + bound;
    for (int i = 0; i < deg; ++i) {
        double ang = 2.0 * M_PI * i / deg + 0.4;
        double rad = (double)bound * (0.5 + 0.5 * i / deg);
        z[i] = Zt<R>(R(rad * std::cos(ang)), R(rad * std::sin(ang)));
    }
    auto peval = [&](Zt<R> x) {
        Zt<R> r = asc[deg];
        for (int i = deg - 1; i >= 0; --i) r = r * x + asc[i];
        return r;
    };
    auto pdeval = [&](Zt<R> x) {
        Zt<R> r = Zt<R>(asc[deg].re * R(deg), asc[deg].im * R(deg));
        for (int i = deg - 1; i >= 1; --i)
            r = r * x + Zt<R>(asc[i].re * R(i), asc[i].im * R(i));
        return r;
    };
    for (int it = 0; it < max_iter; ++it) {
        R maxstep = R(0);
        for (int i = 0; i < deg; ++i) {
            Zt<R> p = peval(z[i]);
            Zt<R> dp = pdeval(z[i]);
            Zt<R> ratio = p / dp;
            Zt<R> sum(R(0), R(0));
            for (int j = 0; j < deg; ++j) {
                if (j == i) continue;
                sum = sum + Zt<R>(R(1), R(0)) / (z[i] - z[j]);
            }
            Zt<R> denom = Zt<R>(R(1), R(0)) - ratio * sum;
            Zt<R> w = ratio / denom;
            z[i] = z[i] - w;
            R s = cabs(w);
            if (s > maxstep) maxstep = s;
        }
        if (maxstep < tol) break;
    }
    return z;
}

// Build P(z) in real type R, solve, verify against the non-holomorphic lens
// equation, and populate `out`.  Returns the worst retained residual (or a
// large value if the image count came out even / empty).
template <class R>
inline double pimg_solve_verify(const PrimaryFrame& pf, double residual_tol,
                                PointImages* out) {
    using Z = Cplx<R>;
    using pimg_detail::cadd;
    using pimg_detail::cmul;
    using pimg_detail::cscale;

    const R a = R(pf.a), m0 = R(pf.m0), m1 = R(1) - R(pf.m0);
    const Z w{R(pf.X), R(pf.Y)};
    const Z wb{R(pf.X), R(-pf.Y)};

    // Q = z^2 - a z
    std::vector<Z> Q{Z(R(0), R(0)), Z(-a, R(0)), Z(R(1), R(0))};
    // N~ = wbar z^2 + (1 - wbar a) z - m0 a
    std::vector<Z> Nt{Z(-m0 * a, R(0)), Z(R(1), R(0)) - wb * Z(a, R(0)), wb};
    // M = N~ - a Q = (wbar - a) z^2 + (1 - wbar a + a^2) z - m0 a
    std::vector<Z> M{Z(-m0 * a, R(0)),
                     Z(R(1), R(0)) - wb * Z(a, R(0)) + Z(a * a, R(0)),
                     wb - Z(a, R(0))};
    // w - z
    std::vector<Z> wz{w, Z(R(-1), R(0))};

    std::vector<Z> P = cadd(cadd(cmul(cmul(wz, Nt), M), cscale(cmul(Q, M), m0)),
                            cscale(cmul(Q, Nt), m1));
    // P is degree 5 (6 coeffs).  Guard against a vanished leading coeff.
    R amax = R(0);
    for (auto& c : P) amax = std::max(amax, cabs(c));
    while (P.size() > 1 && cabs(P.back()) < R(1e-30) * amax) P.pop_back();
    const int deg = (int)P.size() - 1;

    out->images.clear();
    out->mu_total = 0.0;
    out->n_raw = deg;
    if (deg < 1) return 1e30;

    const R atol = (sizeof(R) > 8) ? R(1e-26) : R(1e-13);
    const int aiter = (sizeof(R) > 8) ? 200 : 80;
    std::vector<Z> cz = pimg_detail::complex_aberth<R>(P, aiter, atol);

    // Verify each root against the ORIGINAL non-holomorphic equation.  A
    // genuine image satisfies it to ~machine epsilon; a spurious root (from
    // the Q(z) factor, or a complex image pair forced real by conditioning)
    // fails the residual gate by orders of magnitude.  No Newton polish on the
    // non-holomorphic equation -- it only drags a spurious root onto the image
    // manifold near a real image and manufactures a duplicate.
    const double wnorm = 1.0 + std::hypot(pf.X, pf.Y);
    double worst = 0.0;
    for (const Z& zq : cz) {
        const Cplx<double> z{(double)zq.re, (double)zq.im};
        const Cplx<double> zb{z.re, -z.im};
        const Cplx<double> zba = zb - Cplx<double>(pf.a, 0.0);
        const double d1 = std::hypot(zb.re, zb.im);
        const double d2 = std::hypot(zba.re, zba.im);
        if (d1 < 1e-12 || d2 < 1e-12) continue;  // exact pole root
        const Cplx<double> fz = z - Cplx<double>(pf.m0, 0.0) / zb -
                                Cplx<double>(1.0 - pf.m0, 0.0) / zba;
        const Cplx<double> res = Cplx<double>(pf.X, pf.Y) - fz;
        const double rr = std::hypot(res.re, res.im) / wnorm;
        if (rr > residual_tol) continue;
        bool dup = false;
        for (const auto& e : out->images)
            if (std::hypot(e.z.re - z.re, e.z.im - z.im) <
                1e-9 * (1.0 + std::hypot(z.re, z.im))) {
                dup = true;
                break;
            }
        if (dup) continue;
        // point-source magnification: dw/dzbar = m0/zbar^2 + m1/(zbar-a)^2
        const Cplx<double> dwdzb = Cplx<double>(pf.m0, 0.0) / (zb * zb) +
                                   Cplx<double>(1.0 - pf.m0, 0.0) / (zba * zba);
        const double mu2 = dwdzb.re * dwdzb.re + dwdzb.im * dwdzb.im;
        const double detJ = 1.0 - mu2;
        const double mag = std::fabs(detJ) > 1e-300 ? 1.0 / std::fabs(detJ) : 0.0;
        out->images.push_back({z, std::hypot(z.re, z.im), mag, rr, detJ > 0.0});
        out->mu_total += mag;
        if (rr > worst) worst = rr;
    }
    std::sort(out->images.begin(), out->images.end(),
              [](const PointImage& x, const PointImage& y) {
                  return x.radius < y.radius;
              });
    const int n = (int)out->images.size();
    if (n != 3 && n != 5) return 1e30;
    return worst;
}

}  // namespace pimg_detail

inline PointImages binary_point_images(const PrimaryFrame& pf,
                                       double residual_tol = 1e-9) {
    PointImages out;
    // Tier 1: double.  Accept a clean odd image set (3 or 5) whose worst
    // verified residual is inside tolerance -- for non-extreme q the degree-5
    // double Aberth is reliable, and the seeds only anchor the radial march
    // (band edges come from independent bisection on arc_intervals).  A broken
    // double solve (extreme q: the planet cluster collapses, count drops to 2)
    // fails the odd-count / residual gate and escalates.
    const double w_d =
        pimg_detail::pimg_solve_verify<double>(pf, residual_tol, &out);
    if (w_d <= residual_tol) {
        out.reliable = true;
        return out;
    }
    // Tier 2: __float128.
    const double w_q =
        pimg_detail::pimg_solve_verify<__float128>(pf, residual_tol, &out);
    const int n = (int)out.images.size();
    out.reliable = (n == 3 || n == 5) && w_q <= residual_tol;
    return out;
}

}  // namespace lcbinint::holonomic
