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
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/dd_real.hpp"
#include "lcbinint/magnification/holonomic/d14_real.hpp"
#include "lcbinint/magnification/holonomic/poly_roots.hpp"

namespace lcbinint::holonomic {
namespace re_detail {

#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
inline bool d14_qf_trace_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("HOLO_D14_QF_TRACE");
        return e && e[0] == '1';
    }();
    return enabled;
}

inline bool d14_qf_rootwise_trace_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("HOLO_D14_QF_TRACE_ROOTWISE");
        return e && e[0] == '1';
    }();
    return enabled;
}

inline void d14_qf_trace_value(const char* label, __float128 value) {
    char text[96];
    quadmath_snprintf(text, sizeof(text), "%+.36Qe", value);
    std::fprintf(stderr, "\t%s=%s", label, text);
}
#endif

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
template <>
inline D14StructC<D14Real> d14_struct_cast<D14Real>(const D14StructQf& s) {
    D14StructC<D14Real> r;
    for (int i = 0; i < 4; ++i) r.c3[i] = d14_from_qf(s.c3[i]);
    for (int i = 0; i < 5; ++i) r.g4[i] = d14_from_qf(s.g4[i]);
    for (int i = 0; i < 4; ++i) r.z3[i] = d14_from_qf(s.z3[i]);
    return r;
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

#if defined(HOLO_D14_QF_PAIR_POLISH_RESEARCH)
struct D14QfPairPolishResearchResult {
    std::vector<Cplx<__float128>> roots;
    int i = -1;
    int j = -1;
    int iterations = 0;
    int backtracks = 0;
    bool finite = false;
    bool converged = false;
    __float128 start_residual = 0;
    __float128 end_residual = 0;
    __float128 last_pair_step = 0;
    bool probe_available = false;
    bool probe_finite = false;
    bool probe_driver_stable = false;
    bool probe_viable = false;
    bool probe_global_ready = false;
    int probe_backtracks = -1;
    int probe_driver_i = -1;
    int probe_driver_j = -1;
    __float128 probe_full_step_ratio = -1;
    __float128 probe_accepted_ratio = -1;
    __float128 probe_step_over_separation = -1;
    __float128 probe_det_fraction = -1;
    __float128 probe_post_driver_step = -1;
    __float128 probe_post_outer_step = -1;
};

inline __float128 d14_qf_norm(Cplx<__float128> z) {
    return hypotq(z.re, z.im);
}

inline bool d14_qf_isfinite(Cplx<__float128> z) {
    return finiteq(z.re) && finiteq(z.im);
}

inline Cplx<__float128> d14_qf_scale(Cplx<__float128> z, __float128 a) {
    return {z.re * a, z.im * a};
}

inline Cplx<__float128> d14_qf_sqrt_near(Cplx<__float128> z,
                                         Cplx<__float128> reference) {
    const __float128 r = hypotq(z.re, z.im);
    const __float128 re2 = std::max((__float128)0, (r + z.re) / 2);
    const __float128 im2 = std::max((__float128)0, (r - z.re) / 2);
    Cplx<__float128> root(sqrtq(re2), sqrtq(im2));
    if (z.im < 0) root.im = -root.im;
    const Cplx<__float128> opposite(-root.re, -root.im);
    if (d14_qf_norm(opposite - reference) < d14_qf_norm(root - reference))
        root = opposite;
    return root;
}

inline bool d14_qf_aberth_driver_pair(const D14StructQf& s,
                                      const std::vector<Cplx<__float128>>& roots,
                                      int* pair_i, int* pair_j,
                                      __float128* max_step = nullptr,
                                      __float128* outer_max_step = nullptr) {
    constexpr int n = 14;
    if (roots.size() != n || !pair_i || !pair_j) return false;
    __float128 largest = -1;
    int best_i = -1, best_j = -1;
    __float128 step_magnitudes[n]{};
    for (int i = 0; i < n; ++i) {
        Cplx<__float128> p, dp;
        d14_struct_eval(s, roots[i], p, dp);
        Cplx<__float128> sum(0, 0);
        __float128 nearest = HUGE_VALQ;
        int partner = -1;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const Cplx<__float128> diff = roots[i] - roots[j];
            sum = sum + Cplx<__float128>(1, 0) / diff;
            const __float128 separation = d14_qf_norm(diff);
            if (separation < nearest) {
                nearest = separation;
                partner = j;
            }
        }
        const Cplx<__float128> step = p / (dp - p * sum);
        const __float128 magnitude = d14_qf_norm(step);
        if (!finiteq(magnitude)) return false;
        if (outer_max_step) step_magnitudes[i] = magnitude;
        if (magnitude > largest) {
            largest = magnitude;
            best_i = i;
            best_j = partner;
        }
    }
    if (best_i < 0 || best_j < 0) return false;
    *pair_i = best_i;
    *pair_j = best_j;
    if (max_step) *max_step = largest;
    if (outer_max_step) {
        __float128 outer = 0;
        for (int k = 0; k < n; ++k)
            if (k != best_i && k != best_j)
                outer = std::max(outer, step_magnitudes[k]);
        *outer_max_step = outer;
    }
    return true;
}

// Research-only symmetric two-root corrector. For a selected pair
// a=m+d, b=m-d, solve E=(P(a)+P(b))/2=0 and
// O=(P(a)-P(b))/(2d)=0 in (m,d^2). It never certifies a root set; callers
// must run the unchanged global Aberth-step, residual, completeness, and
// topology gates before using its result.
inline D14QfPairPolishResearchResult d14_qf_pair_polish_research(
    const D14StructQf& s, const std::vector<Cplx<__float128>>& input,
    int i, int j, int max_iter, __float128 step_tolerance,
    bool viability_probe = false) {
    using q = __float128;
    D14QfPairPolishResearchResult out;
    out.roots = input;
    out.i = i;
    out.j = j;
    if (input.size() != 14 || i < 0 || i >= 14 || j < 0 || j >= 14 ||
        i == j || max_iter < 1) return out;
    const Cplx<q> two(2, 0), four(4, 0);
    auto pair_residual = [&](Cplx<q> a, Cplx<q> b) {
        Cplx<q> pa, da, pb, db;
        d14_struct_eval(s, a, pa, da);
        d14_struct_eval(s, b, pb, db);
        if (!d14_qf_isfinite(pa) || !d14_qf_isfinite(pb))
            return HUGE_VALQ;
        return std::max(d14_qf_norm(pa), d14_qf_norm(pb));
    };

    Cplx<q> a = out.roots[i], b = out.roots[j];
    out.start_residual = pair_residual(a, b);
    out.end_residual = out.start_residual;
    if (!finiteq(out.start_residual)) return out;
    out.finite = true;
    for (int it = 0; it < max_iter; ++it) {
        bool stop_after_probe = false;
        const Cplx<q> d = d14_qf_scale(a - b, q(0.5));
        if (!(d14_qf_norm(d) > 0) || !d14_qf_isfinite(d)) break;
        const Cplx<q> m = d14_qf_scale(a + b, q(0.5));
        const Cplx<q> d2 = d * d;
        Cplx<q> pa, da, pb, db;
        d14_struct_eval(s, a, pa, da);
        d14_struct_eval(s, b, pb, db);
        const Cplx<q> two_d = two * d;
        const Cplx<q> four_d = four * d;
        const Cplx<q> four_d2 = four * d2;
        const Cplx<q> four_d3 = four_d2 * d;
        if (!d14_qf_isfinite(pa) || !d14_qf_isfinite(pb) ||
            !d14_qf_isfinite(da) || !d14_qf_isfinite(db) ||
            d14_qf_norm(two_d) == 0 || d14_qf_norm(four_d3) == 0)
            break;
        const Cplx<q> E = d14_qf_scale(pa + pb, q(0.5));
        const Cplx<q> O = (pa - pb) / two_d;
        const Cplx<q> J11 = d14_qf_scale(da + db, q(0.5));
        const Cplx<q> J12 = (da - db) / four_d;
        const Cplx<q> J21 = (da - db) / two_d;
        const Cplx<q> J22 = (da + db) / four_d2 - (pa - pb) / four_d3;
        const Cplx<q> det = J11 * J22 - J12 * J21;
        if (!d14_qf_isfinite(E) || !d14_qf_isfinite(O) ||
            !d14_qf_isfinite(det) || d14_qf_norm(det) == 0)
            break;
        const Cplx<q> dm = (Cplx<q>(-E.re, -E.im) * J22 + J12 * O) / det;
        const Cplx<q> ds = (Cplx<q>(-J11.re, -J11.im) * O + E * J21) / det;
        if (!d14_qf_isfinite(dm) || !d14_qf_isfinite(ds)) break;

        const bool first_step = out.iterations == 0;
        if (first_step && viability_probe) {
            const q det_scale = d14_qf_norm(J11 * J22) +
                                d14_qf_norm(J12 * J21);
            out.probe_det_fraction = det_scale > 0
                ? d14_qf_norm(det) / det_scale : 0;
        }

        q alpha = 1;
        bool accepted = false;
        const int max_backtracks = viability_probe && out.iterations == 0
            ? 3 : 16;
        for (int bt = 0; bt <= max_backtracks; ++bt, alpha /= 2) {
            const Cplx<q> next_m = m + d14_qf_scale(dm, alpha);
            const Cplx<q> next_d2 = d2 + d14_qf_scale(ds, alpha);
            Cplx<q> next_d = d14_qf_sqrt_near(next_d2, d);
            Cplx<q> next_a = next_m + next_d;
            Cplx<q> next_b = next_m - next_d;
            const q next_res = pair_residual(next_a, next_b);
            if (first_step && viability_probe && bt == 0) {
                out.probe_available = true;
                out.probe_finite = finiteq(next_res) &&
                    d14_qf_isfinite(next_a) && d14_qf_isfinite(next_b);
                out.probe_full_step_ratio = out.start_residual > 0
                    ? next_res / out.start_residual
                    : (next_res == 0 ? q(0) : HUGE_VALQ);
            }
            if (finiteq(next_res) && next_res < out.end_residual) {
                const q move = std::max(d14_qf_norm(next_a - a),
                                        d14_qf_norm(next_b - b));
                if (first_step && viability_probe) {
                    out.probe_backtracks = bt;
                    out.probe_finite = finiteq(next_res) && finiteq(move) &&
                        d14_qf_isfinite(next_a) && d14_qf_isfinite(next_b);
                    out.probe_accepted_ratio = out.start_residual > 0
                        ? next_res / out.start_residual
                        : (next_res == 0 ? q(0) : HUGE_VALQ);
                    const q separation = d14_qf_norm(a - b);
                    out.probe_step_over_separation = separation > 0
                        ? move / separation : HUGE_VALQ;
                }
                a = next_a;
                b = next_b;
                out.roots[i] = a;
                out.roots[j] = b;
                out.end_residual = next_res;
                out.last_pair_step = move;
                out.backtracks += bt;
                ++out.iterations;
                accepted = true;
                if (first_step && viability_probe) {
                    int post_i = -1, post_j = -1;
                    q post_step = 0, post_outer = 0;
                    if (d14_qf_aberth_driver_pair(
                            s, out.roots, &post_i, &post_j,
                            &post_step, &post_outer)) {
                        out.probe_driver_i = post_i;
                        out.probe_driver_j = post_j;
                        out.probe_post_driver_step = post_step;
                        out.probe_post_outer_step = post_outer;
                        out.probe_driver_stable =
                            ((post_i == i && post_j == j) ||
                             (post_i == j && post_j == i));
                        const bool driver_budget_ok =
                            out.probe_driver_stable
                                ? (post_outer <= step_tolerance)
                                : (post_step <= step_tolerance);
                        out.probe_viable =
                            out.probe_finite &&
                            out.probe_backtracks >= 0 &&
                            out.probe_accepted_ratio >= 0 &&
                            out.probe_accepted_ratio <= q(0.5) &&
                            driver_budget_ok;
                        out.probe_global_ready = viability_probe &&
                            out.probe_viable &&
                            !out.probe_driver_stable &&
                            post_step <= step_tolerance;
                        stop_after_probe = viability_probe &&
                            (!out.probe_viable || out.probe_global_ready);
                    }
                }
                if (first_step && viability_probe && !out.probe_viable)
                    stop_after_probe = true;
                if (move <= step_tolerance) out.converged = true;
                break;
            }
        }
        if (!accepted || out.converged || stop_after_probe ||
            (viability_probe && out.iterations == 0)) break;
    }
    return out;
}
#endif

// Assemble the expanded ascending-in-v degree-14 coefficient vector from
// the blocks (exact).  Returns {} on the a==x, y==0 leading-term
// degeneracy (matches d14_coeffs' empty-on-degeneracy contract).
inline std::vector<__float128> d14_expanded_from_struct(const D14StructQf& s) {
    using q = __float128;
    // All intermediate products fit in 15 coefficients.  Keeping the
    // coefficient storage fixed is material here: this function is called
    // once per D14 event and the old temporary vector algebra performed a
    // series of small heap allocations before returning the final vector.
    using P = std::array<q, 15>;
    auto zero = [] { return P{}; };
    auto add = [](const P& x, const P& y) {
        P r{};
        for (int i = 0; i < 15; ++i) r[i] = x[i] + y[i];
        return r;
    };
    auto scale_poly = [](const P& x, q k) {
        P r{};
        for (int i = 0; i < 15; ++i) r[i] = x[i] * k;
        return r;
    };
    auto shift = [](const P& x, int n) {
        P r{};
        for (int i = n; i < 15; ++i) r[i] = x[i - n];
        return r;
    };
    auto mul = [](const P& x, int dx, const P& y, int dy) {
        P r{};
        for (int i = 0; i <= dx; ++i)
            for (int j = 0; j <= dy && i + j < 15; ++j)
                r[i + j] += x[i] * y[j];
        return r;
    };

    P C3 = zero(), G4 = zero(), Z3 = zero();
    for (int i = 0; i < 4; ++i) C3[i] = s.c3[i];
    for (int i = 0; i < 5; ++i) G4[i] = s.g4[i];
    for (int i = 0; i < 4; ++i) Z3[i] = s.z3[i];

    const P C3sq = mul(C3, 3, C3, 3);
    const P F6 = add(C3sq, scale_poly(shift(G4, 1), q(-4)));
    const P G4sq = mul(G4, 4, G4, 4);
    const P term1 = mul(F6, 6, G4sq, 8);
    const P inner = add(scale_poly(C3sq, q(2)), scale_poly(shift(G4, 1), q(-9)));
    const P term2 = scale_poly(mul(mul(C3, 3, inner, 6), 9, Z3, 3), q(8));
    const P term3 = scale_poly(shift(mul(Z3, 3, Z3, 3), 2), q(-432));
    const P Dhat = add(add(term1, term2), term3);
    const P D14 = scale_poly(Dhat, q(4096));

    q scale = 0;
    for (q c : D14) { q av = fabsq(c); if (av > scale) scale = av; }
    if (scale == 0) return {};
    int degree = 14;
    while (degree > 0 && fabsq(D14[degree]) < (q)1e-18 * scale) --degree;
    if (degree < 1) return {};
    return std::vector<q>(D14.begin(), D14.begin() + degree + 1);
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
                                              int* iterations = nullptr,
                                              bool* converged = nullptr,
                                              const int* trace_role_hints = nullptr) {
    constexpr int deg = 14;
    std::vector<Cplx<T>> z(deg);
    if (iterations) *iterations = 0;
    if (converged) *converged = false;

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
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
    bool trace_qf = false;
    if constexpr (std::is_same<T, __float128>::value) {
        trace_qf = d14_qf_trace_enabled();
        if (trace_qf) {
            std::fprintf(stderr, "D14QF_BEGIN\tseed=%s\tmax_iter=%d",
                         seed ? "warm" : "cold", max_iter);
            d14_qf_trace_value("tol", tol);
            std::fputc('\n', stderr);
        }
    }
    const bool trace_rootwise = trace_qf && seed &&
                                d14_qf_rootwise_trace_enabled();
    bool trace_bad_seen = false;
#endif
    for (int it = 0; it < max_iter; ++it) {
        if (iterations) *iterations = it + 1;
        T maxstep2 = T(0);
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
        int trace_max_root = -1;
        T trace_max_step = T(0);
        Cplx<T> trace_max_p{}, trace_max_dp{}, trace_max_sum{};
        Cplx<T> trace_max_b{}, trace_max_w{}, trace_max_z{};
        T trace_max_nearest = T(0);
        int trace_max_partner = -1;
        std::array<T, deg> trace_steps{};
#endif
        for (int i = 0; i < deg; ++i) {
            Cplx<T> p, dp;
            d14_struct_eval(s, z[i], p, dp);
            Cplx<T> sum(T(0), T(0));
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
            T trace_nearest = T(0);
            int trace_nearest_partner = -1;
            if (trace_qf) {
                if constexpr (std::is_same<T, __float128>::value) {
                    trace_nearest = __builtin_huge_valq();
                    for (int j = 0; j < deg; ++j) {
                        if (j == i) continue;
                        const __float128 sep = hypotq(z[i].re - z[j].re,
                                                      z[i].im - z[j].im);
                        if (sep < trace_nearest) {
                            trace_nearest = sep;
                            trace_nearest_partner = j;
                        }
                    }
                }
            }
#endif
            for (int j = 0; j < deg; ++j) {
                if (j == i) continue;
                Cplx<T> d = z[i] - z[j];
                sum = sum + Cplx<T>(T(1), T(0)) / d;
            }
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
            Cplx<T> denominator;
            const Cplx<T> old_z = z[i];
#endif
            Cplx<T> w;
            if (legacy) {
                Cplx<T> ratio = p / dp;
                Cplx<T> denom = Cplx<T>(T(1), T(0)) - ratio * sum;
                w = ratio / denom;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
                denominator = dp - p * sum;
#endif
            } else {
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
                denominator = dp - p * sum;
                w = p / denominator;
#else
                w = p / (dp - p * sum);
#endif
            }
            z[i] = z[i] - w;
            T sabs2 = cabs2(w);
            if (sabs2 > maxstep2) maxstep2 = sabs2;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
            if (trace_qf) {
                if constexpr (std::is_same<T, __float128>::value) {
                    const __float128 step = hypotq(w.re, w.im);
                    trace_steps[i] = static_cast<T>(step);
                    const bool finite = qfinite_(p.re) && qfinite_(p.im) &&
                        qfinite_(dp.re) && qfinite_(dp.im) &&
                        qfinite_(sum.re) && qfinite_(sum.im) &&
                        qfinite_(denominator.re) && qfinite_(denominator.im) &&
                        qfinite_(w.re) && qfinite_(w.im) &&
                        qfinite_(z[i].re) && qfinite_(z[i].im);
                    if (!finite && !trace_bad_seen) {
                        trace_bad_seen = true;
                        std::fprintf(stderr,
                            "D14QF_FIRST_BAD\tsweep=%d\troot=%d\tpartner=%d",
                            it + 1, i, trace_nearest_partner);
                        d14_qf_trace_value("z_re", old_z.re);
                        d14_qf_trace_value("z_im", old_z.im);
                        d14_qf_trace_value("new_re", z[i].re);
                        d14_qf_trace_value("new_im", z[i].im);
                        d14_qf_trace_value("absP", hypotq(p.re, p.im));
                        d14_qf_trace_value("absDP", hypotq(dp.re, dp.im));
                        d14_qf_trace_value("absS", hypotq(sum.re, sum.im));
                        d14_qf_trace_value("absB", hypotq(denominator.re,
                                                           denominator.im));
                        d14_qf_trace_value("absStep", step);
                        d14_qf_trace_value("nearestSep", trace_nearest);
                        std::fputc('\n', stderr);
                    }
                    if (trace_rootwise) {
                        const Cplx<__float128> newton = p / dp;
                        std::fprintf(stderr,
                            "D14QF_ROOTSTEP\tseed=warm\tit=%d\tindex=%d"
                            "\trole_hint=%d",
                            it + 1, i,
                            trace_role_hints ? trace_role_hints[i] : -1);
                        d14_qf_trace_value("z_re", old_z.re);
                        d14_qf_trace_value("z_im", old_z.im);
                        d14_qf_trace_value("new_re", z[i].re);
                        d14_qf_trace_value("new_im", z[i].im);
                        d14_qf_trace_value("p_abs", hypotq(p.re, p.im));
                        d14_qf_trace_value("dp_abs", hypotq(dp.re, dp.im));
                        d14_qf_trace_value("step_re", w.re);
                        d14_qf_trace_value("step_im", w.im);
                        d14_qf_trace_value("step_abs", step);
                        d14_qf_trace_value("newton_re", newton.re);
                        d14_qf_trace_value("newton_im", newton.im);
                        d14_qf_trace_value("newton_abs",
                                           hypotq(newton.re, newton.im));
                        d14_qf_trace_value("nearest_sep", trace_nearest);
                        std::fprintf(stderr, "\tnearest_index=%d\n",
                                     trace_nearest_partner);
                    }
                    if (step >= trace_max_step || trace_max_root < 0) {
                        trace_max_root = i;
                        trace_max_step = step;
                        trace_max_p = p;
                        trace_max_dp = dp;
                        trace_max_sum = sum;
                        trace_max_b = denominator;
                        trace_max_w = w;
                        trace_max_z = old_z;
                        trace_max_nearest = trace_nearest;
                        trace_max_partner = trace_nearest_partner;
                    }
                }
            }
#endif
        }
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
        if (trace_qf) {
            if constexpr (std::is_same<T, __float128>::value) {
                __float128 min_sep = __builtin_huge_valq();
                int min_i = -1, min_j = -1, finite_roots = 0;
                for (int i = 0; i < deg; ++i) {
                    if (qfinite_(z[i].re) && qfinite_(z[i].im)) ++finite_roots;
                    for (int j = i + 1; j < deg; ++j) {
                        const __float128 sep = hypotq(z[i].re - z[j].re,
                                                      z[i].im - z[j].im);
                        if (sep < min_sep) {
                            min_sep = sep;
                            min_i = i;
                            min_j = j;
                        }
                    }
                }
                std::fprintf(stderr,
                    "D14QF_SWEEP\tit=%d\tfinite_roots=%d\tmax_root=%d"
                    "\tnearest_partner=%d\tmin_pair=%d,%d",
                    it + 1, finite_roots, trace_max_root, trace_max_partner,
                    min_i, min_j);
                d14_qf_trace_value("maxStep", trace_max_step);
                d14_qf_trace_value("maxRoot_nearest", trace_max_nearest);
                d14_qf_trace_value("maxRoot_absP",
                                   hypotq(trace_max_p.re, trace_max_p.im));
                d14_qf_trace_value("maxRoot_absDP",
                                   hypotq(trace_max_dp.re, trace_max_dp.im));
                d14_qf_trace_value("maxRoot_absS",
                                   hypotq(trace_max_sum.re, trace_max_sum.im));
                d14_qf_trace_value("maxRoot_absB",
                                   hypotq(trace_max_b.re, trace_max_b.im));
                d14_qf_trace_value("maxRoot_absStep",
                                   hypotq(trace_max_w.re, trace_max_w.im));
                d14_qf_trace_value("maxRoot_zRe", trace_max_z.re);
                d14_qf_trace_value("maxRoot_zIm", trace_max_z.im);
                d14_qf_trace_value("min_pair_sep", min_sep);
                std::fputc('\n', stderr);
                if (trace_rootwise && trace_max_root >= 0 &&
                    trace_max_partner >= 0) {
                    const int pair_i = trace_max_root;
                    const int pair_j = trace_max_partner;
                    const auto& zi = z[pair_i];
                    const auto& zj = z[pair_j];
                    const __float128 m_re = (zi.re + zj.re) / 2;
                    const __float128 m_im = (zi.im + zj.im) / 2;
                    const __float128 d_re = zi.re - zj.re;
                    const __float128 d_im = zi.im - zj.im;
                    const __float128 sep = hypotq(d_re, d_im);
                    std::fprintf(stderr,
                        "D14QF_PAIR\tseed=warm\tit=%d\ti=%d\tj=%d",
                        it + 1, pair_i, pair_j);
                    d14_qf_trace_value("sep", sep);
                    d14_qf_trace_value("m_re", m_re);
                    d14_qf_trace_value("m_im", m_im);
                    d14_qf_trace_value("d2_re", d_re * d_re - d_im * d_im);
                    d14_qf_trace_value("d2_im", (__float128)2 * d_re * d_im);
                    d14_qf_trace_value("step_i", trace_steps[pair_i]);
                    d14_qf_trace_value("step_j", trace_steps[pair_j]);
                    std::fprintf(stderr, "\trole_i=%d\trole_j=%d\n",
                        trace_role_hints ? trace_role_hints[pair_i] : -1,
                        trace_role_hints ? trace_role_hints[pair_j] : -1);
                }
            }
        }
#endif
        if (maxstep2 < tol * tol) {
            if (converged) *converged = true;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
            if (trace_qf) std::fprintf(stderr, "D14QF_STOP\tit=%d\treason=step_tol\n", it + 1);
#endif
            break;
        }
    }
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
    if (trace_qf) {
        std::fprintf(stderr, "D14QF_END\tsweeps=%d\n",
                     iterations ? *iterations : max_iter);
        if constexpr (std::is_same<T, __float128>::value) {
            for (int i = 0; i < deg; ++i) {
                std::fprintf(stderr, "D14QF_ROOT\tseed=%s\tindex=%d",
                             seed ? "warm" : "cold", i);
                d14_qf_trace_value("re", z[i].re);
                d14_qf_trace_value("im", z[i].im);
                std::fputc('\n', stderr);
            }
        }
    }
#endif
    return z;
}

// Fixed-capacity D14 kernel.  The residual and derivative remain twofold,
// while the root interaction sum uses compensated double arithmetic unless a
// pair is numerically dangerous.  A dangerous pair is recomputed in the
// same D14Real arithmetic; this decision depends only on the current root
// separation and scale.  The in-place update order is identical to the
// incumbent Aberth loop, and the caller still performs the independent qf
// residual/completeness certificate.
struct D14RealAberthResult {
    std::array<Cplx<D14Real>, 14> roots{};
    std::array<int, 14> root_updates{};
    std::array<int, 14> root_freezes{};
    std::array<int, 14> root_reactivations{};
    std::array<int, 14> cluster_id{};
    std::array<int, 14> cluster_size{};
    std::array<double, 14> final_newton_correction{};
    std::array<double, 14> nearest_separation{};
    std::array<double, 14> relative_newton_correction{};
    int iterations = 0;
    int skipped_updates = 0;
    int reactivations = 0;
    int cluster_wakeups = 0;
    bool finite = true;
    bool converged = false;
    int mixed_pairs = 0;
    int dangerous_pairs = 0;
    int dangerous_rows = 0;
    int full_recompute_rows = 0;
};

struct D14RealScheduleConfig {
    bool enabled = false;
    bool capture_diagnostics = false;
    double absolute_correction_tolerance = 0.0;
    double relative_correction_separation_tolerance = 0.0;
    int patience = 2;
    double cluster_relative_separation = 0.0;
    double reactivate_step_separation = 0.0;
    bool consumer_precision = false;
    double physical_position_rtol = 0.0;
    double soft_cut_position_rtol = 0.0;
    double other_correction_separation_rtol = 0.0;
};

struct D14RealNewtonResult {
    std::array<Cplx<D14Real>, 14> roots{};
    int iterations = 0;
    bool finite = true;
    bool converged = false;
};

// Warm-only local corrector.  Each previous root is corrected independently
// with D/D' from the current structural polynomial.  It is intentionally
// never used as a cold all-root method: the qf residual, conjugacy, Vieta and
// completeness checks in solve_d14 decide whether this cheap trajectory
// candidate is usable.  A stale seed therefore falls back to the incumbent
// basin search.
inline D14RealNewtonResult d14_real_warm_newton(
    const D14StructC<D14Real>& s,
    const std::array<Cplx<D14Real>, 14>& initial,
    int max_iter = 10, D14Real tol = D14Real(1e-26)) {
    D14RealNewtonResult out;
    out.roots = initial;
    constexpr int deg = 14;
    for (int it = 0; it < max_iter; ++it) {
        out.iterations = it + 1;
        bool all_small = true;
        for (int i = 0; i < deg; ++i) {
            Cplx<D14Real> p, dp;
            d14_struct_eval(s, out.roots[i], p, dp);
            if (!qfinite_(p.re) || !qfinite_(p.im) ||
                !qfinite_(dp.re) || !qfinite_(dp.im) ||
                (qabs_(dp.re) == D14Real(0.0) &&
                 qabs_(dp.im) == D14Real(0.0))) {
                out.finite = false;
                return out;
            }
            const Cplx<D14Real> step = p / dp;
            if (!qfinite_(step.re) || !qfinite_(step.im)) {
                out.finite = false;
                return out;
            }
            out.roots[i] = out.roots[i] - step;
            if (!(cabs2(step) < tol * tol)) all_small = false;
        }
        if (all_small) {
            out.converged = true;
            break;
        }
    }
    return out;
}

inline void d14_kahan_add(double x, double& sum, double& correction) {
    const double y = x - correction;
    const double t = sum + y;
    correction = (t - sum) - y;
    sum = t;
}

inline D14RealAberthResult aberth_d14_real_mixed(
    const D14StructC<D14Real>& s,
    const std::array<Cplx<D14Real>, 14>& initial,
    int max_iter = 25, D14Real tol = D14Real(1e-26),
    bool pair_local = false,
    const D14RealScheduleConfig* schedule = nullptr,
    const std::array<int, 14>* root_roles = nullptr) {
    constexpr int deg = 14;
    D14RealAberthResult out;
    out.roots = initial;
    const double separation_factor = 64.0 * std::sqrt(std::numeric_limits<double>::epsilon());
    const bool scheduled = schedule && schedule->enabled;
    const bool capture_diagnostics = schedule && schedule->capture_diagnostics;
    const bool track_roots = scheduled || capture_diagnostics;
    std::array<bool, deg> active{};
    std::array<int, deg> stable{};
    std::array<double, deg> last_aberth_step{};
    active.fill(true);
    last_aberth_step.fill(std::numeric_limits<double>::infinity());

    auto abs_complex = [](const Cplx<D14Real>& z) {
        return std::hypot(static_cast<double>(z.re), static_cast<double>(z.im));
    };
    auto make_clusters = [&](double relative_separation,
                             std::array<int, deg>& ids,
                             std::array<int, deg>& sizes) {
        std::array<int, deg> parent{};
        for (int i = 0; i < deg; ++i) parent[i] = i;
        auto root_of = [&](int x) {
            while (parent[x] != x) x = parent[x];
            return x;
        };
        auto join = [&](int a, int b) {
            const int ra = root_of(a), rb = root_of(b);
            if (ra != rb) parent[rb] = ra;
        };
        if (relative_separation > 0.0) {
            for (int i = 0; i < deg; ++i) {
                for (int j = i + 1; j < deg; ++j) {
                    const double dr = static_cast<double>(out.roots[i].re) -
                                      static_cast<double>(out.roots[j].re);
                    const double di = static_cast<double>(out.roots[i].im) -
                                      static_cast<double>(out.roots[j].im);
                    const double distance = std::hypot(dr, di);
                    const double scale = std::max(
                        {1.0, std::fabs(static_cast<double>(out.roots[i].re)),
                         std::fabs(static_cast<double>(out.roots[i].im)),
                         std::fabs(static_cast<double>(out.roots[j].re)),
                         std::fabs(static_cast<double>(out.roots[j].im))});
                    if (distance <= relative_separation * scale) join(i, j);
                }
            }
        }
        ids.fill(-1);
        sizes.fill(0);
        std::array<int, deg> compact{};
        compact.fill(-1);
        int count = 0;
        for (int i = 0; i < deg; ++i) {
            const int r = root_of(i);
            if (compact[r] < 0) compact[r] = count++;
            ids[i] = compact[r];
            ++sizes[ids[i]];
        }
    };

    auto criterion_met = [&](int i, double newton_correction,
                             double nearest_separation) {
        if (!scheduled) return false;
        bool has_criterion = false;
        bool accepted = true;
        if (schedule->absolute_correction_tolerance > 0.0) {
            has_criterion = true;
            accepted = accepted &&
                newton_correction <= schedule->absolute_correction_tolerance;
        }
        if (schedule->relative_correction_separation_tolerance > 0.0) {
            has_criterion = true;
            const double ratio = nearest_separation > 0.0
                ? newton_correction / nearest_separation
                : std::numeric_limits<double>::infinity();
            accepted = accepted && ratio <=
                schedule->relative_correction_separation_tolerance;
        }
        if (schedule->consumer_precision && root_roles) {
            const int role = (*root_roles)[i];
            const double vr = static_cast<double>(out.roots[i].re);
            const double vi = static_cast<double>(out.roots[i].im);
            const double projected_r = std::sqrt(std::max(0.0, vr));
            const double position_error = projected_r > 0.0
                ? newton_correction / (2.0 * projected_r)
                : std::numeric_limits<double>::infinity();
            double budget = 0.0;
            if (role == 1) budget = schedule->physical_position_rtol;
            else if (role == 2) budget = schedule->soft_cut_position_rtol;
            else budget = schedule->other_correction_separation_rtol;
            if (budget > 0.0) {
                has_criterion = true;
                if (role == 1 || role == 2) {
                    accepted = accepted && position_error /
                        (1.0 + projected_r) <= budget;
                } else {
                    const double ratio = nearest_separation > 0.0
                        ? newton_correction / nearest_separation
                        : std::numeric_limits<double>::infinity();
                    accepted = accepted && ratio <= budget;
                }
            }
            (void)vi;
        }
        return has_criterion && accepted;
    };

    for (int it = 0; it < max_iter; ++it) {
        out.iterations = it + 1;
        D14Real max_step2(0.0);
        std::array<int, deg> sweep_cluster{};
        std::array<int, deg> sweep_cluster_size{};
        std::array<bool, deg> reactivate_cluster{};
        if (scheduled) {
            make_clusters(schedule->cluster_relative_separation,
                          sweep_cluster, sweep_cluster_size);
            std::array<bool, deg> component_active{};
            for (int i = 0; i < deg; ++i)
                if (active[i]) component_active[sweep_cluster[i]] = true;
            for (int i = 0; i < deg; ++i) {
                if (!active[i] && component_active[sweep_cluster[i]]) {
                    active[i] = true;
                    stable[i] = 0;
                    ++out.root_reactivations[i];
                    ++out.reactivations;
                    ++out.cluster_wakeups;
                }
            }
        }
        int active_count = 0;
        for (int i = 0; i < deg; ++i) {
            if (scheduled && !active[i]) {
                ++out.skipped_updates;
                continue;
            }
            ++active_count;
            Cplx<D14Real> p, dp;
            d14_struct_eval(s, out.roots[i], p, dp);
            if (!qfinite_(p.re) || !qfinite_(p.im) ||
                !qfinite_(dp.re) || !qfinite_(dp.im)) {
                out.finite = false;
                return out;
            }
            double newton_correction = std::numeric_limits<double>::infinity();
            if (scheduled &&
                !(dp.re == D14Real(0.0) && dp.im == D14Real(0.0))) {
                const Cplx<D14Real> newton = p / dp;
                if (qfinite_(newton.re) && qfinite_(newton.im))
                    newton_correction = abs_complex(newton);
            }

            double sr = 0.0, si = 0.0, cr = 0.0, ci = 0.0;
            bool dangerous = false;
            std::array<int, deg - 1> dangerous_index{};
            int dangerous_count = 0;
            double nearest_separation = std::numeric_limits<double>::infinity();
            const double xir = static_cast<double>(out.roots[i].re);
            const double xii = static_cast<double>(out.roots[i].im);
            if (!std::isfinite(xir) || !std::isfinite(xii)) {
                out.finite = false;
                return out;
            }
            for (int j = 0; j < deg; ++j) {
                if (j == i) continue;
                const double xjr = static_cast<double>(out.roots[j].re);
                const double xji = static_cast<double>(out.roots[j].im);
                const double dr = xir - xjr, di = xii - xji;
                const double dn = std::hypot(dr, di);
                const double scale = std::max({1.0, std::fabs(xir),
                                               std::fabs(xii), std::fabs(xjr),
                                               std::fabs(xji)});
                if (scheduled) nearest_separation = std::min(nearest_separation, dn);
                if (!(dn > 0.0) || !std::isfinite(dn)) {
                    out.finite = false;
                    return out;
                }
                if (dn <= separation_factor * scale) {
                    dangerous = true;
                    dangerous_index[dangerous_count++] = j;
                } else {
                    const double inv = 1.0 / (dr * dr + di * di);
                    d14_kahan_add(dr * inv, sr, cr);
                    d14_kahan_add(-di * inv, si, ci);
                    ++out.mixed_pairs;
                }
            }

            Cplx<D14Real> sum;
            if (dangerous) ++out.dangerous_rows;
            if (!dangerous) {
                if (pair_local) {
                    // Kahan's residual has the opposite sign to the lost
                    // low part: exact ~= sum - correction.  Retain it in the
                    // D14Real low limb when the local-pair path is active.
                    sum = Cplx<D14Real>(D14Real(sr, -cr),
                                        D14Real(si, -ci));
                } else {
                    sum = Cplx<D14Real>(D14Real(sr), D14Real(si));
                }
            } else {
                if (pair_local) {
                    // Keep the safe interaction sum in double and promote
                    // only the genuinely close differences.  The close
                    // terms are accumulated in D14Real and remain in the
                    // same denominator as the D14Real D/D' evaluation.
                    sum = Cplx<D14Real>(D14Real(sr, -cr),
                                        D14Real(si, -ci));
                    for (int k = 0; k < dangerous_count; ++k) {
                        const int j = dangerous_index[k];
                        sum = sum + Cplx<D14Real>(D14Real(1.0), D14Real(0.0)) /
                                          (out.roots[i] - out.roots[j]);
                        ++out.dangerous_pairs;
                    }
                } else {
                    ++out.full_recompute_rows;
                    sum = Cplx<D14Real>(D14Real(0.0), D14Real(0.0));
                    for (int j = 0; j < deg; ++j) {
                        if (j == i) continue;
                        sum = sum + Cplx<D14Real>(D14Real(1.0), D14Real(0.0)) /
                                          (out.roots[i] - out.roots[j]);
                        ++out.dangerous_pairs;
                    }
                }
            }
            const Cplx<D14Real> step = p / (dp - p * sum);
            if (!qfinite_(step.re) || !qfinite_(step.im)) {
                out.finite = false;
                return out;
            }
            out.roots[i] = out.roots[i] - step;
            const D14Real step2 = cabs2(step);
            if (step2 > max_step2) max_step2 = step2;
            const double step_abs = track_roots ? abs_complex(step) : 0.0;
            if (track_roots) {
                last_aberth_step[i] = step_abs;
                ++out.root_updates[i];
            }

            if (scheduled) {
                if (criterion_met(i, newton_correction, nearest_separation)) {
                    if (stable[i] < schedule->patience) ++stable[i];
                    if (stable[i] >= schedule->patience && active[i]) {
                        active[i] = false;
                        ++out.root_freezes[i];
                    }
                } else {
                    stable[i] = 0;
                }
                if (schedule->reactivate_step_separation > 0.0 &&
                    nearest_separation > 0.0 &&
                    step_abs / nearest_separation >=
                        schedule->reactivate_step_separation) {
                    reactivate_cluster[sweep_cluster[i]] = true;
                }
            }
        }
        if (scheduled) {
            for (int i = 0; i < deg; ++i) {
                if (!reactivate_cluster[sweep_cluster[i]] || active[i]) continue;
                active[i] = true;
                stable[i] = 0;
                ++out.root_reactivations[i];
                ++out.reactivations;
                ++out.cluster_wakeups;
            }
            if (active_count == 0) break;
            bool all_below_incumbent_tolerance = true;
            const double tol_double = static_cast<double>(tol);
            for (int i = 0; i < deg; ++i)
                all_below_incumbent_tolerance = all_below_incumbent_tolerance &&
                    last_aberth_step[i] < tol_double;
            if (all_below_incumbent_tolerance) {
                out.converged = true;
                break;
            }
            bool any_active = false;
            for (bool value : active) any_active = any_active || value;
            if (!any_active) break;
        } else if (max_step2 < tol * tol) {
            out.converged = true;
            break;
        }
    }

    // Final per-root estimates are retained only for the opt-in research
    // report.  The scheduled production experiment needs the aggregate
    // update/freeze counters above, but does not consume these estimates; in
    // particular, do not pay for another 14 D14Real polynomial evaluations
    // and nearest-neighbor scan on every scheduled solve.  They are not
    // acceptance gates: the caller still runs the unchanged qf residual and
    // global completeness checks.
    if (capture_diagnostics) for (int i = 0; i < deg; ++i) {
        Cplx<D14Real> p, dp;
        d14_struct_eval(s, out.roots[i], p, dp);
        if (!qfinite_(p.re) || !qfinite_(p.im) ||
            !qfinite_(dp.re) || !qfinite_(dp.im) ||
            (dp.re == D14Real(0.0) && dp.im == D14Real(0.0))) {
            out.finite = false;
            continue;
        }
        const Cplx<D14Real> correction = p / dp;
        if (!qfinite_(correction.re) || !qfinite_(correction.im)) {
            out.finite = false;
            continue;
        }
        out.final_newton_correction[i] = abs_complex(correction);
        double nearest = std::numeric_limits<double>::infinity();
        for (int j = 0; j < deg; ++j) {
            if (i == j) continue;
            nearest = std::min(nearest, std::hypot(
                static_cast<double>(out.roots[i].re) -
                    static_cast<double>(out.roots[j].re),
                static_cast<double>(out.roots[i].im) -
                    static_cast<double>(out.roots[j].im)));
        }
        out.nearest_separation[i] = nearest;
        out.relative_newton_correction[i] = nearest > 0.0
            ? out.final_newton_correction[i] / nearest
            : std::numeric_limits<double>::infinity();
    }
    if (capture_diagnostics) {
        std::array<int, deg> final_cluster{};
        std::array<int, deg> final_cluster_size{};
        const double final_cluster_relative = scheduled &&
            schedule->cluster_relative_separation > 0.0
                ? schedule->cluster_relative_separation : separation_factor;
        make_clusters(final_cluster_relative, final_cluster, final_cluster_size);
        for (int i = 0; i < deg; ++i) {
            out.cluster_id[i] = final_cluster[i];
            out.cluster_size[i] = final_cluster_size[final_cluster[i]];
        }
    }
    return out;
}

}  // namespace re_detail
}  // namespace lcbinint::holonomic
