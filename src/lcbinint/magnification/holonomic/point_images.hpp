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
// Q(z) multiplication can introduce z = 0 / z = a as spurious roots; every
// candidate root is verified against the ORIGINAL lens equation by residual
// (conj(z) taken directly, not via N~/Q), so spurious and non-image roots are
// dropped.  A binary lens has 3 or 5 real images (odd).

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

using Z = Cplx<double>;

// ascending-coeff complex polynomial product
inline std::vector<Z> cmul(const std::vector<Z>& a, const std::vector<Z>& b) {
    std::vector<Z> r(a.size() + b.size() - 1, Z(0.0, 0.0));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            r[i + j] = r[i + j] + a[i] * b[j];
    return r;
}
inline std::vector<Z> cadd(const std::vector<Z>& a, const std::vector<Z>& b) {
    std::vector<Z> r = a.size() >= b.size() ? a : b;
    const std::vector<Z>& s = a.size() >= b.size() ? b : a;
    for (size_t i = 0; i < s.size(); ++i) r[i] = r[i] + s[i];
    return r;
}
inline std::vector<Z> cscale(std::vector<Z> a, double s) {
    for (auto& c : a) c = c * Z(s, 0.0);
    return a;
}

}  // namespace pimg_detail

inline PointImages binary_point_images(const PrimaryFrame& pf,
                                       double residual_tol = 1e-9) {
    using pimg_detail::Z;
    using pimg_detail::cadd;
    using pimg_detail::cmul;
    using pimg_detail::cscale;

    const double a = pf.a, m0 = pf.m0, m1 = 1.0 - pf.m0;
    const Z w{pf.X, pf.Y};
    const Z wb{pf.X, -pf.Y};

    // Q = z^2 - a z
    std::vector<Z> Q{Z(0.0, 0.0), Z(-a, 0.0), Z(1.0, 0.0)};
    // N~ = wbar z^2 + (1 - wbar a) z - m0 a
    std::vector<Z> Nt{Z(-m0 * a, 0.0), Z(1.0, 0.0) - wb * Z(a, 0.0), wb};
    // M = N~ - a Q = (wbar - a) z^2 + (1 - wbar a + a^2) z - m0 a
    std::vector<Z> M{Z(-m0 * a, 0.0),
                     Z(1.0, 0.0) - wb * Z(a, 0.0) + Z(a * a, 0.0),
                     wb - Z(a, 0.0)};
    // w - z
    std::vector<Z> wz{w, Z(-1.0, 0.0)};

    std::vector<Z> P = cadd(cadd(cmul(cmul(wz, Nt), M),
                                 cscale(cmul(Q, M), m0)),
                            cscale(cmul(Q, Nt), m1));
    // P is degree 5 (6 coeffs).  Guard against a vanished leading coeff.
    while (P.size() > 1 &&
           std::hypot(P.back().re, P.back().im) <
               1e-14 * [&] {
                   double s = 0.0;
                   for (auto& c : P) s = std::max(s, std::hypot(c.re, c.im));
                   return s;
               }())
        P.pop_back();
    int deg = (int)P.size() - 1;

    PointImages out;
    if (deg < 1) { out.reliable = false; return out; }

    // descending coeffs for aberth
    std::vector<double> dummy;  // (aberth wants real coeffs -> use complex path)
    // aberth<double> takes real coeffs; the image polynomial is complex, so
    // run a small complex Aberth here directly.
    std::vector<Z> cz(deg);
    {
        // Cauchy-bound spread
        double an = std::hypot(P[deg].re, P[deg].im);
        double bound = 0.0;
        for (int i = 0; i < deg; ++i)
            bound = std::max(bound, std::hypot(P[i].re, P[i].im) / an);
        bound = 1.0 + bound;
        for (int i = 0; i < deg; ++i) {
            double ang = 2.0 * M_PI * i / deg + 0.4;
            double rad = bound * (0.5 + 0.5 * i / deg);
            cz[i] = Z(rad * std::cos(ang), rad * std::sin(ang));
        }
        auto peval = [&](Z x) {
            Z r = P[deg];
            for (int i = deg - 1; i >= 0; --i) r = r * x + P[i];
            return r;
        };
        auto pdeval = [&](Z x) {
            Z r = Z(P[deg].re * deg, P[deg].im * deg);
            for (int i = deg - 1; i >= 1; --i)
                r = r * x + Z(P[i].re * i, P[i].im * i);
            return r;
        };
        for (int it = 0; it < 120; ++it) {
            double maxstep = 0.0;
            for (int i = 0; i < deg; ++i) {
                Z p = peval(cz[i]);
                Z dp = pdeval(cz[i]);
                Z ratio = p / dp;
                Z sum(0.0, 0.0);
                for (int j = 0; j < deg; ++j) {
                    if (j == i) continue;
                    sum = sum + Z(1.0, 0.0) / (cz[i] - cz[j]);
                }
                Z denom = Z(1.0, 0.0) - ratio * sum;
                Z wst = ratio / denom;
                cz[i] = cz[i] - wst;
                maxstep = std::max(maxstep, std::hypot(wst.re, wst.im));
            }
            if (maxstep < 1e-14) break;
        }
    }
    out.n_raw = deg;

    // The degree-5 image polynomial is analytic and Aberth converges every
    // root to ~1e-14; a genuine image then also satisfies the ORIGINAL
    // non-holomorphic equation to ~1e-15, while a spurious root (introduced by
    // the Q(z) = z(z-a) factor, or a complex image pair forced onto the real
    // axis by ill-conditioning) fails the residual gate by many orders of
    // magnitude.  No Newton polish on the non-holomorphic equation -- it only
    // drags a spurious root onto the image manifold near a real image and
    // manufactures a duplicate.
    const double wnorm = 1.0 + std::hypot(w.re, w.im);

    for (const Z& z : cz) {
        const Z zb{z.re, -z.im};
        const Z zba = zb - Z(a, 0.0);
        const double d1 = std::hypot(zb.re, zb.im);
        const double d2 = std::hypot(zba.re, zba.im);
        if (d1 < 1e-12 || d2 < 1e-12) continue;  // exact pole root
        const Z fz = z - Z(m0, 0.0) / zb - Z(m1, 0.0) / zba;
        const Z res = w - fz;
        const double rr = std::hypot(res.re, res.im) / wnorm;
        if (rr > residual_tol) continue;
        // reject duplicate roots (aberth can converge two guesses onto one
        // image when another is complex)
        bool dup = false;
        for (const auto& e : out.images)
            if (std::hypot(e.z.re - z.re, e.z.im - z.im) <
                1e-9 * (1.0 + std::hypot(z.re, z.im))) {
                dup = true;
                break;
            }
        if (dup) continue;
        // point-source magnification: dw/dzbar = m0/zbar^2 + m1/(zbar-a)^2
        const Z dwdzb =
            Z(m0, 0.0) / (zb * zb) + Z(m1, 0.0) / (zba * zba);
        const double mu2 = dwdzb.re * dwdzb.re + dwdzb.im * dwdzb.im;
        const double detJ = 1.0 - mu2;
        const double mag = std::fabs(detJ) > 1e-300 ? 1.0 / std::fabs(detJ) : 0.0;
        out.images.push_back(
            {z, std::hypot(z.re, z.im), mag, rr, detJ > 0.0});
        out.mu_total += mag;
    }
    std::sort(out.images.begin(), out.images.end(),
              [](const PointImage& x, const PointImage& y) {
                  return x.radius < y.radius;
              });
    const int n = (int)out.images.size();
    if (n != 3 && n != 5) out.reliable = false;
    return out;
}

}  // namespace lcbinint::holonomic
