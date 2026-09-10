#pragma once

// ATPT holonomic solver -- D14 low-degree block representation
// (D14_structure_ideas_ja.md, sections 2-3).
//
// The degree-14 radial-event discriminant D14(v) (v = R^2) is normally
// built by expanding the boundary quartic's I,J invariants to degree 36 in
// R and collapsing to 15 monomial coefficients (re_detail::d14_coeffs).
// That expansion carries ~1e-28 relative coefficient error from the
// high-order cancellation, which sits *above* the double-double solve
// floor -- so a near-multiple cluster (rand028) still escalates the
// compensated Aberth polish to __float128.
//
// The exact identity (verified to 3e-28 vs the expansion route on all 108
// bench geometries, scratchpad/verify_d14_structure.cpp):
//
//   d = v-1,  e = v-m,  beta = x^2 + y^2 - h
//   L  = v d + a^2 e                                    (deg 2)
//   U  = a e d + x L + a v beta                         (deg 2)
//   C3 = v d^2 + a^2 e^2 + v(v+a^2) beta
//        + 2 a x v (3m - 1 - 2v)                        (deg 3, monic)
//   G4 = U^2 + y^2 L^2 - 4 a e x C3
//        - 4 a^2 e^2 v (4x^2 + y^2)                     (deg 4)
//   B2 = [a m + (x-a) v]^2 + v^2 (y^2 - h)              (deg 2)
//   Z3 = a^2 (1-m) y^2 (v-m) B2                         (deg 3)
//   F6 = C3^2 - 4 v G4                                  (deg 6)
//
//   Dhat = F6 G4^2 + 8 C3 (2 C3^2 - 9 v G4) Z3 - 432 v^2 Z3^2   (deg 14)
//   D14  = 4096 * Dhat        (== re_detail::d14_coeffs convention, exact)
//
// The blocks are built with NO catastrophic cancellation, so both the
// expanded coefficient vector assembled from them and the block-form
// D/D' evaluation used inside the Aberth polish are ~1e-32 accurate.
// Never conjugate v mid-evaluation (memo section 3): the identity is a
// polynomial identity valid for complex v, but only if the block
// arithmetic stays holomorphic.

#include <array>
#include <cmath>
#include <vector>

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/dd_real.hpp"
#include "lcbinint/magnification/holonomic/poly_roots.hpp"

namespace lcbinint::holonomic {
namespace re_detail {

// Low-degree block coefficients (ascending in v), scalar type T.
template <class T>
struct D14StructC {
    std::array<T, 4> c3{};  // C3 : deg 3
    std::array<T, 5> g4{};  // G4 : deg 4
    std::array<T, 4> z3{};  // Z3 : deg 3
};
using D14StructQf = D14StructC<__float128>;

// Build the blocks exactly in __float128 from the primary-frame lens
// parameters.  Cheap: a handful of degree <= 4 polynomial products.
inline D14StructQf d14_struct_build(__float128 a, __float128 m, __float128 X,
                                    __float128 Y, __float128 rho) {
    using q = __float128;
    const q x = X, y = Y, h = rho * rho;
    const q a2 = a * a, m2 = m * m;
    const q beta = x * x + y * y - h;

    // L = [ -a^2 m , a^2 - 1 , 1 ]
    const q L0 = -a2 * m, L1 = a2 - 1, L2 = 1;
    // U = [ a m (1 - a x) , -a(m+1) + x(a^2-1) + a beta , a + x ]
    const q U0 = a * m - a2 * m * x;
    const q U1 = -a * (m + 1) + x * (a2 - 1) + a * beta;
    const q U2 = a + x;

    D14StructQf s;
    // C3
    s.c3[0] = a2 * m2;
    s.c3[1] = 1 - 2 * a2 * m + a2 * beta + 2 * a * x * (3 * m - 1);
    s.c3[2] = -2 + a2 + beta - 4 * a * x;
    s.c3[3] = 1;

    // U^2  (deg 4)
    std::array<q, 5> U2p{U0 * U0, 2 * U0 * U1, U1 * U1 + 2 * U0 * U2,
                         2 * U1 * U2, U2 * U2};
    // y^2 L^2  (deg 4)
    std::array<q, 5> yL2{y * y * (L0 * L0), y * y * (2 * L0 * L1),
                         y * y * (L1 * L1 + 2 * L0 * L2), y * y * (2 * L1 * L2),
                         y * y * (L2 * L2)};
    // -4 a x * (e . C3)   e = [-m, 1]  -> deg 4
    const q k1 = -4 * a * x;
    std::array<q, 5> eC3{k1 * (-m * s.c3[0]),
                         k1 * (s.c3[0] - m * s.c3[1]),
                         k1 * (s.c3[1] - m * s.c3[2]),
                         k1 * (s.c3[2] - m * s.c3[3]),
                         k1 * (s.c3[3])};
    // -4 a^2 (4x^2+y^2) * (v e^2)   e^2 = [m^2,-2m,1] -> v e^2 = [0,m^2,-2m,1]
    const q k2 = -4 * a2 * (4 * x * x + y * y);
    std::array<q, 5> ve2{0, k2 * m2, k2 * (-2 * m), k2 * 1, 0};
    for (int i = 0; i < 5; ++i)
        s.g4[i] = U2p[i] + yL2[i] + eC3[i] + ve2[i];

    // B2 = [ a^2 m^2 , 2 a m (x-a) , (x-a)^2 + y^2 - h ]
    const q B0 = a2 * m2, B1 = 2 * a * m * (x - a), B2c = (x - a) * (x - a) + y * y - h;
    // e . B2   e = [-m,1]  -> deg 3
    const q kz = a2 * (1 - m) * y * y;
    s.z3[0] = kz * (-m * B0);
    s.z3[1] = kz * (B0 - m * B1);
    s.z3[2] = kz * (B1 - m * B2c);
    s.z3[3] = kz * (B2c);
    return s;
}

template <class T>
inline D14StructC<T> d14_struct_cast(const D14StructQf& s) {
    D14StructC<T> r;
    for (int i = 0; i < 4; ++i) r.c3[i] = T((double)s.c3[i]);
    for (int i = 0; i < 5; ++i) r.g4[i] = T((double)s.g4[i]);
    for (int i = 0; i < 4; ++i) r.z3[i] = T((double)s.z3[i]);
    return r;
}
template <>
inline D14StructC<DD> d14_struct_cast<DD>(const D14StructQf& s) {
    D14StructC<DD> r;
    for (int i = 0; i < 4; ++i) r.c3[i] = dd_from_qf(s.c3[i]);
    for (int i = 0; i < 5; ++i) r.g4[i] = dd_from_qf(s.g4[i]);
    for (int i = 0; i < 4; ++i) r.z3[i] = dd_from_qf(s.z3[i]);
    return r;
}
template <>
inline D14StructC<__float128> d14_struct_cast<__float128>(const D14StructQf& s) {
    return s;
}

// Horner value + derivative of an ascending-coeff real poly at complex x.
template <class T, std::size_t N>
inline void horner_vd(const std::array<T, N>& c, int deg, const Cplx<T>& x,
                      Cplx<T>& val, Cplx<T>& der) {
    val = Cplx<T>(c[deg], T(0));
    der = Cplx<T>(T(0), T(0));
    for (int i = deg - 1; i >= 0; --i) {
        der = der * x + val;
        val = val * x + Cplx<T>(c[i], T(0));
    }
}

// Structural D14 / D14' at complex v (memo section 3), returns
// 4096*Dhat and 4096*Dhat'.  Pure holomorphic block arithmetic.
template <class T>
inline void d14_struct_eval(const D14StructC<T>& s, const Cplx<T>& v,
                            Cplx<T>& D, Cplx<T>& Dp) {
    Cplx<T> c, cp, g, gp, z, zp;
    horner_vd(s.c3, 3, v, c, cp);
    horner_vd(s.g4, 4, v, g, gp);
    horner_vd(s.z3, 3, v, z, zp);

    const Cplx<T> two(T(2), T(0)), four(T(4), T(0)), nine(T(9), T(0));
    Cplx<T> cc = c * c;
    Cplx<T> vg = v * g;
    Cplx<T> f = cc - four * vg;
    Cplx<T> b = two * cc - nine * vg;
    Cplx<T> gg = g * g;
    Cplx<T> zz = z * z;
    Cplx<T> vv = v * v;

    Cplx<T> Dhat = f * gg + Cplx<T>(T(8), T(0)) * (c * b * z)
                 - Cplx<T>(T(432), T(0)) * (vv * zz);

    Cplx<T> ccp = c * cp;
    Cplx<T> fp = two * ccp - four * g - four * (v * gp);
    Cplx<T> bp = four * ccp - nine * g - nine * (v * gp);
    Cplx<T> Dhatp = fp * gg + two * (f * (g * gp))
                  + Cplx<T>(T(8), T(0)) * ((cp * b + c * bp) * z + c * b * zp)
                  - Cplx<T>(T(864), T(0)) * (v * zz)
                  - Cplx<T>(T(864), T(0)) * (vv * (z * zp));

    const Cplx<T> k(T(4096), T(0));
    D = k * Dhat;
    Dp = k * Dhatp;
}

// Assemble the expanded ascending-in-v degree-14 coefficient vector from
// the blocks (exact).  Returns {} on the a==x, y==0 leading-term
// degeneracy (matches d14_coeffs' empty-on-degeneracy contract).
inline std::vector<__float128> d14_expanded_from_struct(const D14StructQf& s) {
    using q = __float128;
    auto pmul = [](const std::vector<q>& x, const std::vector<q>& y) {
        std::vector<q> r(x.size() + y.size() - 1, q(0));
        for (std::size_t i = 0; i < x.size(); ++i)
            for (std::size_t j = 0; j < y.size(); ++j) r[i + j] += x[i] * y[j];
        return r;
    };
    auto padd = [](std::vector<q> x, const std::vector<q>& y) {
        if (x.size() < y.size()) x.resize(y.size(), q(0));
        for (std::size_t i = 0; i < y.size(); ++i) x[i] += y[i];
        return x;
    };
    auto pscale = [](std::vector<q> x, q k) {
        for (auto& v : x) v *= k;
        return x;
    };
    auto pshift = [](const std::vector<q>& x) {  // * v
        std::vector<q> r(x.size() + 1, q(0));
        for (std::size_t i = 0; i < x.size(); ++i) r[i + 1] = x[i];
        return r;
    };

    std::vector<q> C3(s.c3.begin(), s.c3.end());
    std::vector<q> G4(s.g4.begin(), s.g4.end());
    std::vector<q> Z3(s.z3.begin(), s.z3.end());

    std::vector<q> F6 = padd(pmul(C3, C3), pscale(pshift(G4), q(-4)));
    std::vector<q> G4sq = pmul(G4, G4);
    std::vector<q> term1 = pmul(F6, G4sq);
    std::vector<q> inner = padd(pscale(pmul(C3, C3), q(2)), pscale(pshift(G4), q(-9)));
    std::vector<q> term2 = pscale(pmul(pmul(C3, inner), Z3), q(8));
    std::vector<q> term3 = pscale(pshift(pshift(pmul(Z3, Z3))), q(-432));
    std::vector<q> Dhat = padd(padd(term1, term2), term3);
    std::vector<q> D14 = pscale(Dhat, q(4096));

    D14.resize(15, q(0));  // deg 14 by construction
    q scale = 0;
    for (q c : D14) { q av = fabsq(c); if (av > scale) scale = av; }
    if (scale == 0) return {};
    while (D14.size() > 1 && fabsq(D14.back()) < (q)1e-18 * scale) D14.pop_back();
    if ((int)D14.size() - 1 < 1) return {};
    return D14;  // ascending in v
}

// Aberth-Ehrlich on the structural D14 evaluator.  Mirrors
// poly_roots.hpp::aberth exactly (same init spread, same iteration, same
// tol semantics) but calls d14_struct_eval for p / p'.  Degree fixed 14.
// `bound_coeffs` (descending, len 15) is used only for the cold-start
// Cauchy circle when `seed` is null.
template <class T>
inline std::vector<Cplx<T>> aberth_d14_struct(const D14StructC<T>& s,
                                              const T* bound_coeffs,
                                              int max_iter,
                                              const Cplx<T>* seed,
                                              T tol_override,
                                              int* iterations = nullptr) {
    constexpr int deg = 14;
    std::vector<Cplx<T>> z(deg);
    if (iterations) *iterations = 0;

    T bound = T(1);
    if (bound_coeffs) {
        T an = qabs_(bound_coeffs[0]);
        T mx = T(0);
        for (int i = 1; i <= deg; ++i) {
            T v = qabs_(bound_coeffs[i]) / an;
            if (v > mx) mx = v;
        }
        bound = T(1) + mx;
    }
    const T pi = T(3.14159265358979323846264338327950288L);
    for (int i = 0; i < deg; ++i) {
        if (seed) {
            z[i] = seed[i];
        } else {
            T ang = (T(2) * pi * T(i)) / T(deg) + T(0.4);
            T rad = bound * (T(0.5) + T(0.5) * T(i) / T(deg));
            z[i] = Cplx<T>(rad * T(std::cos((double)ang)),
                           rad * T(std::sin((double)ang)));
        }
    }

    const T tol = tol_override > T(0)
                      ? tol_override
                      : ((sizeof(T) > 8) ? T(1e-24) : T(1e-15));
    const bool legacy = holo_legacy_complex_ops();
    for (int it = 0; it < max_iter; ++it) {
        if (iterations) *iterations = it + 1;
        T maxstep2 = T(0);
        for (int i = 0; i < deg; ++i) {
            Cplx<T> p, dp;
            d14_struct_eval(s, z[i], p, dp);
            Cplx<T> sum(T(0), T(0));
            for (int j = 0; j < deg; ++j) {
                if (j == i) continue;
                Cplx<T> d = z[i] - z[j];
                sum = sum + Cplx<T>(T(1), T(0)) / d;
            }
            Cplx<T> w;
            if (legacy) {
                Cplx<T> ratio = p / dp;
                Cplx<T> denom = Cplx<T>(T(1), T(0)) - ratio * sum;
                w = ratio / denom;
            } else {
                w = p / (dp - p * sum);
            }
            z[i] = z[i] - w;
            T sabs2 = cabs2(w);
            if (sabs2 > maxstep2) maxstep2 = sabs2;
        }
        if (maxstep2 < tol * tol) break;
    }
    return z;
}

}  // namespace re_detail
}  // namespace lcbinint::holonomic
