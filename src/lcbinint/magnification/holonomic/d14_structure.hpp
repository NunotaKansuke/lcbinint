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
#include <limits>
#include <vector>

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/dd_real.hpp"
#include "lcbinint/magnification/holonomic/d14_real.hpp"
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
