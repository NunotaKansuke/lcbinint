#pragma once

// ATPT holonomic solver (M7) -- radial event enumeration (plan sec. 3).
// Ports python/lcbinint/holonomic_ref/radial_events.py.
//
// The circle |z| = R meets phi = 0 in an arc count that is piecewise
// constant in R and changes only at a radial event:
//
//   physical_real / physical_complex : positive real root v of D14(v),
//     v = R^2 ; Disc_t P = R^4 D14(R^2), deg_v D14 = 14.  Real double
//     root of P -> band birth/death (physical_real); complex double root
//     -> soft boundary (physical_complex).  Also D14's complex roots
//     (Re v > 0) are soft boundaries.
//   R_eq_a, R_eq_sqrt_m0          : representation events (always present
//                                   if in range).
//   L_root                        : on-axis (Y = 0) only, roots of L(v)
//                                   from Res(P, B).
//   chart_p4                      : p4(R) = 0, a boundary point crossing
//                                   theta = pi.
//
// D14 is built in __float128 by the same Iq/Jq discriminant route as
// radial_events._d14_poly, then its roots are found with Aberth-Ehrlich in
// __float128 (see poly_roots.hpp for why `double` is unsafe here).  The
// retained coefficients carry no catastrophic cancellation; the two
// spurious top coefficients are trimmed.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/d14_structure.hpp"
#include "lcbinint/magnification/holonomic/d14_positive_roots.hpp"
#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
#include "lcbinint/magnification/holonomic/d14_rouche.hpp"
#endif
#include "lcbinint/magnification/holonomic/dd_real.hpp"
#include "lcbinint/magnification/holonomic/d14_lifted.hpp"
#include "lcbinint/magnification/holonomic/d14_hybrid.hpp"
#include "lcbinint/magnification/holonomic/fp_env.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/poly_roots.hpp"
#include "lcbinint/magnification/holonomic/v2_profile.hpp"

namespace lcbinint::holonomic {

struct RadialEvent {
    double radius = 0.0;
    std::string kind;
    bool physically_real = false;
    std::string detail;
    // Optional high-precision/local data retained for adaptive consumers.
    // `radius` remains the public binary64 cell boundary; `radius_lo` is the
    // unevaluated qf remainder relative to that boundary.  These fields are
    // metadata only and do not change the topology merge policy.
    double radius_lo = 0.0;
    double radius_uncertainty = std::numeric_limits<double>::infinity();
    double fold_t_seed = 0.0;
    bool fold_t_seed_valid = false;
    int precision_tier = 0;  // 0=double, 1=DD, 2=__float128 source
    bool positive_certified = false;
    bool atlas_anchor = false;
    int positive_root_id = -1;
    __float128 certified_radius_lo=0, certified_radius_hi=0;
    double d14_condition = std::numeric_limits<double>::infinity();
};

#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
// Research-only snapshots taken before the incumbent qf-cold backstop may
// replace a finite lower-tier root set.  They never participate in solver
// acceptance.  A standalone harness uses them to ask whether the downstream
// event/cell contract agrees with the final, fully certified root set.
struct D14EventContractCandidate {
    std::string stage;
    std::vector<Cplx<__float128>> roots;
    double worst_residual = std::numeric_limits<double>::infinity();
    bool converged = false;
    bool finite = false;
};

struct D14EventContractCapture {
    std::vector<D14EventContractCandidate> candidates;
    std::vector<Cplx<__float128>> oracle_roots;
    int oracle_tier = -99;
    double r_max = 0.0;
};

inline thread_local D14EventContractCapture* d14_event_contract_capture = nullptr;

struct D14EventContractCaptureScope {
    D14EventContractCapture* previous;
    explicit D14EventContractCaptureScope(D14EventContractCapture& capture)
        : previous(d14_event_contract_capture) {
        capture = D14EventContractCapture{};
        d14_event_contract_capture = &capture;
    }
    ~D14EventContractCaptureScope() { d14_event_contract_capture = previous; }
    D14EventContractCaptureScope(const D14EventContractCaptureScope&) = delete;
};
#endif

namespace re_detail {

using qf = __float128;

// complex-over-qf fixed-capacity polynomial in R (ascending), degree < N.
template <int N>
struct CPoly {
    std::array<Cplx<qf>, N> c{};
    int deg = 0;
};
template <int N>
struct RPoly {
    std::array<qf, N> c{};
    int deg = 0;
};

template <int N>
inline CPoly<N> cadd(const CPoly<N>& a, const CPoly<N>& b) {
    CPoly<N> r;
    r.deg = std::max(a.deg, b.deg);
    for (int i = 0; i <= r.deg; ++i) r.c[i] = a.c[i] + b.c[i];
    return r;
}
template <int N>
inline CPoly<N> cmul(const CPoly<N>& a, const CPoly<N>& b) {
    CPoly<N> r;
    r.deg = a.deg + b.deg;
    for (int i = 0; i <= a.deg; ++i)
        for (int j = 0; j <= b.deg; ++j)
            r.c[i + j] = r.c[i + j] + a.c[i] * b.c[j];
    return r;
}
template <int N>
inline CPoly<N> cscale(const CPoly<N>& a, Cplx<qf> s) {
    CPoly<N> r;
    r.deg = a.deg;
    for (int i = 0; i <= a.deg; ++i) r.c[i] = a.c[i] * s;
    return r;
}
template <int N>
inline CPoly<N> conj_poly(const CPoly<N>& a) {
    CPoly<N> r;
    r.deg = a.deg;
    for (int i = 0; i <= a.deg; ++i) r.c[i] = Cplx<qf>(a.c[i].re, -a.c[i].im);
    return r;
}
template <int N>
inline RPoly<N> real_of(const CPoly<N>& a) {
    RPoly<N> r;
    r.deg = a.deg;
    for (int i = 0; i <= a.deg; ++i) r.c[i] = a.c[i].re;
    return r;
}
template <int N>
inline RPoly<N> radd(const RPoly<N>& a, const RPoly<N>& b) {
    RPoly<N> r;
    r.deg = std::max(a.deg, b.deg);
    for (int i = 0; i <= r.deg; ++i) r.c[i] = a.c[i] + b.c[i];
    return r;
}
template <int N>
inline RPoly<N> rmul(const RPoly<N>& a, const RPoly<N>& b) {
    RPoly<N> r;
    r.deg = a.deg + b.deg;
    for (int i = 0; i <= a.deg; ++i)
        for (int j = 0; j <= b.deg; ++j) r.c[i + j] += a.c[i] * b.c[j];
    return r;
}
template <int N>
inline RPoly<N> rscale(const RPoly<N>& a, qf s) {
    RPoly<N> r;
    r.deg = a.deg;
    for (int i = 0; i <= a.deg; ++i) r.c[i] = a.c[i] * s;
    return r;
}

constexpr int NC = 8;   // cT polys: degree <= 3
constexpr int NP = 10;  // P coeff polys in R: degree <= 6
constexpr int ND = 48;  // Iq^3 / Jq^2: degree <= 36

// P_i(R) coefficients (each a real poly in R), i = 0..4.  Mirrors
// radial_events._P_coeffs_in_R_exact but in __float128.
struct PolyFamilyR {
    RPoly<NP> p[5];
};

inline PolyFamilyR p_coeffs_in_R(qf a, qf m0, qf X, qf Y, qf rho) {
    auto cx = [](qf r, qf i) { return Cplx<qf>(r, i); };
    CPoly<NC> R1;
    R1.deg = 1;
    R1.c[1] = cx(1, 0);
    CPoly<NC> R2;
    R2.deg = 2;
    R2.c[2] = cx(1, 0);
    Cplx<qf> zeta = cx(X, Y);

    CPoly<NC> n0 = cscale(R2, cx(-zeta.re, -zeta.im));  // -zeta R^2
    CPoly<NC> n1;                                       // R^3 + (a zeta - 1) R
    n1.deg = 3;
    n1.c[3] = cx(1, 0);
    n1.c[1] = Cplx<qf>(a, 0) * zeta - cx(1, 0);
    CPoly<NC> n2 = cscale(R2, cx(-a, 0));  // -a R^2 + a m0
    n2.c[0] = cx(a * m0, 0);
    if (n2.deg < 2) n2.deg = 2;

    CPoly<NC> cT0 = cadd(cadd(n0, n1), n2);
    CPoly<NC> tmp = cadd(n2, cscale(n0, cx(-1, 0)));
    CPoly<NC> cT1 = cscale(tmp, cx(0, 2));  // 2i (n2 - n0)
    CPoly<NC> cT2 = cadd(cadd(n1, cscale(n0, cx(-1, 0))), cscale(n2, cx(-1, 0)));

    // bm = (R-a)^2, bp = (R+a)^2 ; k = rho^2 R^2
    RPoly<NP> bm;
    bm.deg = 2;
    bm.c[0] = a * a;
    bm.c[1] = -2 * a;
    bm.c[2] = 1;
    RPoly<NP> bp;
    bp.deg = 2;
    bp.c[0] = a * a;
    bp.c[1] = 2 * a;
    bp.c[2] = 1;
    RPoly<NP> kk;
    kk.deg = 2;
    kk.c[2] = rho * rho;

    auto cabs2poly = [](const CPoly<NC>& x) {
        return real_of(cmul(x, conj_poly(x)));
    };
    auto cReCross = [](const CPoly<NC>& x, const CPoly<NC>& y) {
        // Re( x conj(y) + y conj(x) ) as a real poly = 2 Re(x conj y)
        return real_of(cadd(cmul(x, conj_poly(y)), cmul(y, conj_poly(x))));
    };

    RPoly<NP> a0p, a1p, a2p, a3p, a4p;
    {
        // promote NC-> NP by copy
        auto up = [](const RPoly<NC>& s) {
            RPoly<NP> r;
            r.deg = s.deg;
            for (int i = 0; i <= s.deg; ++i) r.c[i] = s.c[i];
            return r;
        };
        a0p = up(cabs2poly(cT0));
        a1p = up(cReCross(cT0, cT1));
        RPoly<NC> a2a = cabs2poly(cT1);
        RPoly<NC> a2b = cReCross(cT0, cT2);
        RPoly<NP> a2 = radd(up(a2a), up(a2b));
        a2p = a2;
        a3p = up(cReCross(cT1, cT2));
        a4p = up(cabs2poly(cT2));
    }

    RPoly<NP> lin0 = rmul(kk, bm);
    RPoly<NP> lin2 = rmul(kk, radd(bm, bp));
    RPoly<NP> lin4 = rmul(kk, bp);

    PolyFamilyR out;
    out.p[0] = radd(lin0, rscale(a0p, -1));
    out.p[1] = rscale(a1p, -1);
    out.p[2] = radd(lin2, rscale(a2p, -1));
    out.p[3] = rscale(a3p, -1);
    out.p[4] = radd(lin4, rscale(a4p, -1));
    return out;
}

// D14(v) ascending coefficients (v = R^2), degree 14.  Returns empty on a
// structural degeneracy (leading coeff vanishes / not even in R).
inline std::vector<qf> d14_coeffs(const PolyFamilyR& fam) {
    auto up = [](const RPoly<NP>& s) {
        RPoly<ND> r;
        r.deg = s.deg;
        for (int i = 0; i <= s.deg; ++i) r.c[i] = s.c[i];
        return r;
    };
    RPoly<ND> p0 = up(fam.p[0]), p1 = up(fam.p[1]), p2 = up(fam.p[2]),
              p3 = up(fam.p[3]), p4 = up(fam.p[4]);

    RPoly<ND> Iq = radd(radd(rscale(rmul(p4, p0), 12), rscale(rmul(p3, p1), -3)),
                        rmul(p2, p2));
    RPoly<ND> Jq = radd(
        radd(radd(rscale(rmul(rmul(p4, p2), p0), 72),
                  rscale(rmul(rmul(p3, p2), p1), 9)),
             radd(rscale(rmul(rmul(p4, p1), p1), -27),
                  rscale(rmul(rmul(p3, p3), p0), -27))),
        rscale(rmul(rmul(p2, p2), p2), -2));

    RPoly<ND> disc = radd(rscale(rmul(rmul(Iq, Iq), Iq), 4),
                          rscale(rmul(Jq, Jq), -1));
    for (int i = 0; i <= disc.deg; ++i) disc.c[i] /= 27;

    // scale for trimming
    qf scale = 0;
    for (int i = 0; i <= disc.deg; ++i) {
        qf v = fabsq(disc.c[i]);
        if (v > scale) scale = v;
    }
    if (scale == 0) return {};

    // divisible by R^4
    for (int i = 0; i < 4; ++i)
        if (fabsq(disc.c[i]) > (qf)1e-18 * scale) return {};
    // even in R above R^4  (odd coeffs ~ 0)
    std::vector<qf> even;
    for (int i = 4; i <= disc.deg; i += 2) {
        if (i + 1 <= disc.deg && fabsq(disc.c[i + 1]) > (qf)1e-16 * scale)
            return {};
        even.push_back(disc.c[i]);  // ascending in v
    }
    // trim spurious high-degree coeffs
    while (even.size() > 1 && fabsq(even.back()) < (qf)1e-18 * scale)
        even.pop_back();
    return even;
}

// p4(R) has a parameter-independent factorization whenever |Y| <= rho.
// Writing
//
//   f(R) = R^3 + (X+a)R^2 + (aX-1)R - a m0,
//   b^2 = (rho-|Y|)(rho+|Y|),
//
// gives p4(R) = -(f-bR(R+a))(f+bR(R+a)).  The two monic factors are cubics
// with coefficients [1, a+X +/- b, a(X +/- b)-1, -a m0].  If |Y| > rho,
// p4 is strictly negative for R>0 and has no positive roots.  This helper
// keeps the old degree-6 Aberth route available in radial_events behind
// HOLO_CHART_P4_LEGACY=1 for parity and timing A/Bs.
inline bool holo_chart_p4_factor_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_CHART_P4_LEGACY");
        return !(e && e[0] == '1');
    }();
    return on;
}

inline std::vector<double> chart_p4_factor_roots(const PrimaryFrame& pf) {
    std::vector<double> out;
    const double ay = std::fabs(pf.Y);
    if (pf.rho < ay) return out;

    // Difference-of-squares form avoids subtracting rho^2 and Y^2 when the
    // chart crossing is close to the |Y|=rho boundary.
    const double b2 = (pf.rho - ay) * (pf.rho + ay);
    const double b = std::sqrt(b2);
    // Only the exact algebraic boundary b^2 == 0 has one factor.  Nearly
    // coincident cubic roots are intentionally left for radial_events'
    // existing event merge below; the factorization itself has no tolerance
    // based branch.
    const int nfactor = b2 == 0.0 ? 1 : 2;
    for (int s = -1; s <= 1; s += 2) {
        if (nfactor == 1 && s > 0) break;
        const double sb = s * b;
        const double desc[4] = {1.0, pf.a + pf.X + sb,
                                pf.a * (pf.X + sb) - 1.0,
                                -pf.a * pf.m0};
        auto z = aberth<double>(desc, 3, 120);
        auto roots = positive_real_roots(z, 1e-8, 1e-10);
        out.insert(out.end(), roots.begin(), roots.end());
    }
    std::sort(out.begin(), out.end());
    // Keep both roots when b^2 is nonzero.  Coincident or near-coincident
    // events are merged by radial_events(), after all event kinds have been
    // collected; the algebraic factorization itself has no numerical
    // proximity policy.
    return out;
}

// At a discriminant zero P has a double root t*.  Keep the stationary-root
// probe instead of returning only a bool: adaptive event refinement can use
// the same branch seed without solving the derivative cubic again.
struct PhysicalRootProbe {
    bool physically_real = false;
    bool stationary_valid = false;
    double stationary_t = 0.0;
    double normalized_residual = std::numeric_limits<double>::infinity();
};

// The degree-one polynomial in the quartic/subresultant PRS is a cheap
// stationary-root seed at an ordinary double root.  It is only a seed: the
// residual check below decides whether it is usable, and the existing cubic
// probe remains the fail-closed fallback for a vanishing or ill-conditioned
// subresultant.
struct FoldSubresultantSeed {
    bool valid = false;
    double t = 0.0;
    double normalized_residual = std::numeric_limits<double>::infinity();
};

inline FoldSubresultantSeed quartic_fold_subresultant_seed(
    const QuarticCoeffs& q) {
    const double a = q.p[4], b = q.p[3], c = q.p[2], d = q.p[1], e = q.p[0];
    const double scale = std::max({1.0, std::fabs(a), std::fabs(b),
                                   std::fabs(c), std::fabs(d), std::fabs(e)});
    if (!std::isfinite(scale) || a == 0.0) return {};

    // S_3(t) = -a (U t + V), from the exact subresultant sequence of P,P_t.
    const double U =
        32.0 * a * a * c * e - 36.0 * a * a * d * d -
        12.0 * a * b * b * e + 28.0 * a * b * c * d -
        8.0 * a * c * c * c - 6.0 * b * b * b * d +
        2.0 * b * b * c * c;
    const double V =
        -48.0 * a * a * d * e + 32.0 * a * b * c * e +
        3.0 * a * b * d * d - 4.0 * a * c * c * d -
        9.0 * b * b * b * e + b * b * c * d;
    if (!std::isfinite(U) || !std::isfinite(V) ||
        std::fabs(U) <= 64.0 * std::numeric_limits<double>::epsilon() * scale)
        return {};
    const double t = -V / U;
    if (!std::isfinite(t)) return {};

    const double p = ((a * t + b) * t + c) * t * t + d * t + e;
    const double pt = (4.0 * a * t + 3.0 * b) * t * t + 2.0 * c * t + d;
    const double pscale = std::max(
        {1.0, std::fabs(a * t * t * t * t), std::fabs(b * t * t * t),
         std::fabs(c * t * t), std::fabs(d * t), std::fabs(e)});
    const double residual = std::hypot(p, pt) / pscale;
    return {std::isfinite(residual), t, residual};
}

// Port of radial_events._double_root_is_real, with the selected stationary
// root retained for downstream local P=P_t refinement.
inline PhysicalRootProbe probe_double_root(double R, const PrimaryFrame& pf,
                                            double tol = 1e-6) {
    QuarticCoeffs q = boundary_quartic(R, pf);  // ascending
    V2Profile* prof = v2_profile_current();
    if (prof) ++prof->d14_fold_seed_attempts;
    auto fold_seed_begin = V2Clock::now();
    const FoldSubresultantSeed sub = quartic_fold_subresultant_seed(q);
    if (sub.valid && sub.normalized_residual < tol) {
        if (prof) {
            ++prof->d14_fold_seed_success;
            v2_profile_add_ms(&V2Profile::d14_fold_seed_ms, fold_seed_begin,
                              V2Clock::now());
        }
        PhysicalRootProbe out;
        out.stationary_t = sub.t;
        out.normalized_residual = sub.normalized_residual;
        out.stationary_valid = true;
        out.physically_real = true;
        return out;
    }
    if (prof) {
        ++prof->d14_fold_seed_fallback;
        v2_profile_add_ms(&V2Profile::d14_fold_seed_ms, fold_seed_begin,
                          V2Clock::now());
    }
    double Pdesc[5] = {q.p[4], q.p[3], q.p[2], q.p[1], q.p[0]};
    double scale = 0.0;
    for (double x : q.p) scale += std::fabs(x);
    scale += 1e-300;
    double dc[4] = {4 * Pdesc[0], 3 * Pdesc[1], 2 * Pdesc[2], Pdesc[3]};
    int dd = 3;
    while (dd > 0 && dc[0] == 0.0) {
        for (int i = 0; i < dd; ++i) dc[i] = dc[i + 1];
        --dd;
    }
    if (dd <= 0) return {};
    auto z = aberth<double>(dc, dd, 120);
    double best_re = 0, best_im = 0, best_res = 1e300;
    for (const auto& r : z) {
        Cplx<double> pv = poly_eval_c<double>(Pdesc, 4, r);
        double res = std::sqrt(pv.re * pv.re + pv.im * pv.im) / scale;
        if (res < best_res) {
            best_res = res;
            best_re = r.re;
            best_im = r.im;
        }
    }
    PhysicalRootProbe out;
    out.stationary_t = best_re;
    out.normalized_residual = best_res;
    out.stationary_valid = std::isfinite(best_res) && std::isfinite(best_re) &&
                            std::isfinite(best_im);
    out.physically_real = out.stationary_valid && best_res < tol &&
                          std::fabs(best_im) <
                              1e-4 * (std::fabs(best_re) + 1.0);
    return out;
}

inline bool double_root_is_real(double R, const PrimaryFrame& pf,
                                double tol = 1e-6) {
    return probe_double_root(R, pf, tol).physically_real;
}

// Process-wide opt-out: HOLO_D14_LEGACY_SOLVE=1 forces the original
// __float128-only warm polish.  Default: the Phase-B1 compensated
// (double-double) polish, which reproduces the same roots at
// residual <= the __float128 path but without libquadmath in the hot
// loop (evidence/holonomic/d14_compensated_bench.txt).  The kept flag is
// the A/B escape hatch.
inline bool holo_d14_compensated_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_LEGACY_SOLVE");
        return !(e && e[0] == '1');
    }();
    return on;
}

// Mixed-interaction production path.  Keep the same separation certificate
// and D14Real D/D' evaluation, but promote only dangerous pair terms instead
// of recomputing an entire interaction row.  HOLO_D14_LOCAL_PAIRS=0 is the
// retained A/B escape hatch.
inline bool holo_d14_local_pairs_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_LOCAL_PAIRS");
        return !(e && e[0] == '0');
    }();
    return on;
}

#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
inline bool holo_d14_eager_dd_block_research_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_EAGER_DD_BLOCK");
        return e && e[0] == '1';
    }();
    return on;
}
#endif

inline double holo_d14_schedule_env_double(const char* name, double fallback,
                                           double maximum) {
    const char* e = std::getenv(name);
    if (!e || !*e) return fallback;
    char* end = nullptr;
    const double value = std::strtod(e, &end);
    if (end == e || *end != '\0' || !std::isfinite(value) || value < 0.0 ||
        value > maximum)
        return fallback;
    return value;
}

inline D14RealScheduleConfig holo_d14_real_schedule_config() {
    static const D14RealScheduleConfig config = [] {
        D14RealScheduleConfig value;
        const char* active = std::getenv("HOLO_D14_REAL_ACTIVE");
        if (!active || active[0] != '1') return value;
        value.absolute_correction_tolerance = holo_d14_schedule_env_double(
            "HOLO_D14_REAL_ABS_TOL", 0.0, 1.0);
        value.relative_correction_separation_tolerance =
            holo_d14_schedule_env_double("HOLO_D14_REAL_REL_TOL", 0.0, 1.0);
        value.patience = std::clamp(std::atoi(
            std::getenv("HOLO_D14_REAL_PATIENCE")
                ? std::getenv("HOLO_D14_REAL_PATIENCE") : "2"), 1, 8);
        value.cluster_relative_separation = holo_d14_schedule_env_double(
            "HOLO_D14_REAL_CLUSTER_REL", 64.0 *
                std::sqrt(std::numeric_limits<double>::epsilon()), 1.0);
        value.reactivate_step_separation = holo_d14_schedule_env_double(
            "HOLO_D14_REAL_REACTIVATE_RATIO", 0.25, 1e6);
        const char* role = std::getenv("HOLO_D14_ROLE_PRECISION");
        value.consumer_precision = role && role[0] == '1';
        if (value.consumer_precision) {
            value.physical_position_rtol = holo_d14_schedule_env_double(
                "HOLO_D14_ROLE_PHYSICAL_RTOL", 1e-10, 1.0);
            value.soft_cut_position_rtol = holo_d14_schedule_env_double(
                "HOLO_D14_ROLE_SOFT_RTOL", 1e-8, 1.0);
            value.other_correction_separation_rtol = holo_d14_schedule_env_double(
                "HOLO_D14_ROLE_OTHER_RATIO", 1e-12, 1.0);
        }
        const bool has_budget = value.absolute_correction_tolerance > 0.0 ||
            value.relative_correction_separation_tolerance > 0.0 ||
            value.physical_position_rtol > 0.0 ||
            value.soft_cut_position_rtol > 0.0 ||
            value.other_correction_separation_rtol > 0.0;
        value.enabled = has_budget;
        return value;
    }();
    return config;
}

// Process-wide opt-out: HOLO_D14_STRUCT_LEGACY=1 forces Horner on the
// expanded degree-14 coefficient vector.  Default: run the Aberth polish
// against the C3/G4/Z3 block-form D/D' evaluator (d14_structure.hpp), and
// build the (still expanded, for the residual / Newton-sum gates)
// coefficient vector from the same exact blocks.  The blocks carry no
// high-order cancellation, so the polish sees ~1e-32 coefficients where
// the expansion route carries ~1e-28 -- the difference that forced the
// rand028-class near-multiple clusters to escalate to __float128.  The
// block form is ~1.8x the flops per Aberth node (isolated solve_d14
// median 0.83x) but collapses the latency tail: epoch p90 1.14x / p99
// 1.64x / max 1.72x, all percentiles non-regressing; identity exact to
// 3.2e-28, parity 0/115, no new cold fallbacks
// (evidence/holonomic/d14_structure_bench.txt, checkpoint 34).  Isolated
// holonomic build only -- header-only, no shared .so wiring.
inline bool holo_d14_struct_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_STRUCT_LEGACY");
        return !(e && e[0] == '1');
    }();
    return on;
}

// Research selector.  The shipped default is the proven compensated
// Aberth path.  `HOLO_D14_METHOD=lifted` only permits the cubic-discriminant
// candidate below after its independent global certificate passes; a failed
// candidate immediately continues through the incumbent ladder.
inline bool holo_d14_lifted_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_METHOD");
        return e && e[0] == 'l';
    }();
    return on;
}

inline bool holo_d14_hybrid_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_METHOD");
        return e && std::strncmp(e, "hybrid", 6) == 0;
    }();
    return on;
}

inline bool holo_d14_horner_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_METHOD");
        return e && std::strncmp(e, "horner", 6) == 0;
    }();
    return on;
}

// Research A/B knob for the balanced double basin search.  The shipped
// defaults remain 200 cold / 60 warm.  Any shorter search is still followed
// by the existing DD/qf residual and global completeness gates, so a missed
// basin can only escalate or fail closed.
inline int holo_d14_presearch_max(bool warm) {
    static const int requested = [] {
        const char* e = std::getenv("HOLO_D14_PRESEARCH_MAX");
        if (!e || !*e) return 0;
        const int n = std::atoi(e);
        return (n >= 20 && n <= 200) ? n : 0;
    }();
    const int default_max = warm ? 60 : 200;
    if (requested == 0) return default_max;
    return warm ? std::min(default_max, requested) : requested;
}

// Research-only early-stop knob for the double basin locator.  This pass is
// not the accuracy owner: DD/qf polishing and the global certificates below
// remain unchanged.  The default is zero, which preserves the incumbent
// fixed-work presearch exactly; a positive value is an A/B experiment for
// whether a certified downstream polish can safely absorb an earlier stop.
inline double holo_d14_presearch_tol() {
    static const double tol = [] {
        const char* e = std::getenv("HOLO_D14_PRESEARCH_TOL");
        if (!e || !*e) return 0.0;
        char* end = nullptr;
        const double x = std::strtod(e, &end);
        if (end == e || *end != '\0' || !std::isfinite(x) ||
            !(x > 0.0) || x > 1e-12)
            return 0.0;
        return x;
    }();
    return tol;
}

inline bool holo_d14_skip_warm_presearch() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_SKIP_WARM_PRESEARCH");
        return e && e[0] == '1';
    }();
    return on;
}

// Root-wise early-stop for the balanced double basin search.
// The usual all-root Aberth interaction is retained for active roots and
// inactive roots remain in every interaction sum.  A later D14Real/qf
// residual and global root-set certificate still owns acceptance; an active
// candidate can therefore only save work or fall back.  The 1e-12/patience=2
// defaults have full 14,432-row trajectory parity; setting
// HOLO_D14_ACTIVE_PRESEARCH=0 restores the fixed-work A/B path.
inline bool holo_d14_active_presearch_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_ACTIVE_PRESEARCH");
        return !(e && e[0] == '0');
    }();
    return on;
}

inline double holo_d14_active_presearch_tol() {
    static const double tol = [] {
        const char* e = std::getenv("HOLO_D14_ACTIVE_TOL");
        if (!e || !*e) return 1e-12;
        char* end = nullptr;
        const double x = std::strtod(e, &end);
        if (end == e || *end != '\0' || !std::isfinite(x) ||
            !(x > 0.0) || x > 1e-8)
            return 1e-12;
        return x;
    }();
    return tol;
}

inline int holo_d14_active_presearch_patience() {
    static const int patience = [] {
        const char* e = std::getenv("HOLO_D14_ACTIVE_PATIENCE");
        if (!e || !*e) return 2;
        const int x = std::atoi(e);
        return x >= 1 && x <= 8 ? x : 2;
    }();
    return patience;
}

// Research-only recovery for a non-finite double presearch update.  The
// default remains the incumbent retry/cold behavior; when enabled, a fully
// finite checkpoint is promoted to the existing D14Real all-root polish.
inline bool holo_d14_last_finite_handoff_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_LAST_FINITE_HANDOFF");
        return e && e[0] == '1';
    }();
    return on;
}

// Emits per-root Aberth quantities only for a one-case diagnostic replay.
inline bool holo_d14_presearch_trace_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_PRESEARCH_TRACE");
        return e && e[0] == '1';
    }();
    return on;
}

// Dedicated D14Real polish is the selected isolated D14 kernel after the
// matched root-set and whole-epoch A/B.  `HOLO_D14_REAL=0` retains the
// incumbent DD path for regression comparisons.
inline bool holo_d14_real_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_REAL");
        return !(e && e[0] == '0');
    }();
    return on;
}

inline D14Real holo_d14_real_tol() {
    static const D14Real tol = [] {
        const char* e = std::getenv("HOLO_D14_REAL_TOL");
        if (!e || !*e) return D14Real(1e-14);
        char* end = nullptr;
        const double x = std::strtod(e, &end);
        if (end == e || *end != '\0' || !std::isfinite(x) ||
            !(x > 0.0) || x > 1e-10 || x < 1e-30)
            return D14Real(1e-14);
        return D14Real(x);
    }();
    return tol;
}

inline bool holo_d14_direct_warm_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_D14_DIRECT_WARM");
        return e && e[0] == '1';
    }();
    return on;
}

inline qf holo_d14_warm_screen_rel() {
    static const qf rel = [] {
        const char* e = std::getenv("HOLO_D14_WARM_SCREEN_REL");
        if (!e || !*e) return (qf)1e-5;
        char* end = nullptr;
        const double x = std::strtod(e, &end);
        if (end == e || *end != '\0' || !std::isfinite(x) ||
            !(x > 0.0) || x > 1e-2)
            return (qf)1e-5;
        return (qf)x;
    }();
    return rel;
}

// A warm seed is a basin locator only.  This screen is a numerical residual
// and finiteness check against the current coefficients; it is never the
// final acceptance condition.  A rejected seed simply uses the incumbent
// balanced double presearch, while a direct candidate still has the exact
// qf residual, conjugacy and completeness gates below.
inline bool d14_warm_seed_screen(const D14StructQf& sc,
                                 const std::vector<Cplx<qf>>& warm,
                                 int deg, qf dscale) {
    if (deg != 14 || static_cast<int>(warm.size()) != deg) return false;
    const qf limit = holo_d14_warm_screen_rel() * (dscale + (qf)1e-300);
    for (const auto& root : warm) {
        if (!finiteq(root.re) || !finiteq(root.im)) return false;
        Cplx<qf> value, deriv;
        d14_struct_eval(sc, root, value, deriv);
        if (!finiteq(value.re) || !finiteq(value.im)) return false;
        if (cabs(value) > limit) return false;
    }
    return true;
}

// Result of the multi-tier D14 root solve.
struct D14Solve {
    std::vector<Cplx<qf>> roots;
    std::array<D14RootWorkRecord, 14> root_work{};
    std::array<Cplx<qf>, 14> d14real_roots{};
    bool has_d14real_roots = false;
    qf worst_res = 0;
    int tier = 0;  // 0 dd-sufficed, 1 qf-warm escalation, 2 cold quad,
                   // 3 legacy qf-warm, -1 cold (seed non-finite)
    bool warm_seeded = false;  // the double presearch was seeded by a
                               // previous epoch's root set (Phase B2)
#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
    std::vector<D14EventContractCandidate> event_contract_candidates;
    bool event_contract_accepted = false;
#endif
};

#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
inline void d14_capture_event_contract_candidate(
    D14Solve& out, const char* stage,
    const std::vector<Cplx<qf>>& roots, qf residual, bool converged) {
    D14EventContractCandidate candidate;
    candidate.stage = stage;
    candidate.roots = roots;
    candidate.worst_residual = static_cast<double>(residual);
    candidate.converged = converged;
    candidate.finite = roots.size() == 14;
    for (const auto& root : roots)
        candidate.finite = candidate.finite && finiteq(root.re) && finiteq(root.im);
    out.event_contract_candidates.push_back(std::move(candidate));
}
#endif

// Minimum-cost bijection used only by the opt-in per-root diagnostic when a
// cold qf restart has changed Aberth root ordering.  The assignment does not
// affect solver acceptance or ordering.  A separate separation test marks
// ambiguous matches invalid for displacement reporting.
inline std::array<int, 14> d14_diagnostic_root_assignment(
    const std::vector<Cplx<qf>>& final_roots,
    const std::array<Cplx<qf>, 14>& source_roots) {
    constexpr int n = 14;
    std::array<int, n> assignment{};
    assignment.fill(-1);
    if (final_roots.size() != n) return assignment;

    std::array<std::array<double, n>, n> cost{};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const qf scale = std::max((qf)1, cabs(final_roots[i]));
            cost[i][j] = static_cast<double>(
                cabs(final_roots[i] - source_roots[j]) / scale);
        }
    }
    // Hungarian assignment, 1-based internal indices.
    std::array<double, n + 1> u{}, v{}, minv{};
    std::array<int, n + 1> p{}, way{};
    std::array<bool, n + 1> used{};
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        minv.fill(std::numeric_limits<double>::infinity());
        used.fill(false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    for (int j = 1; j <= n; ++j)
        if (p[j] > 0) assignment[p[j] - 1] = j - 1;
    return assignment;
}

// worst relative residual of `roots` against the degree-`deg` descending
// __float128 polynomial `desc`.
inline qf d14_worst_res(const qf* desc, int deg,
                        const std::vector<Cplx<qf>>& roots, qf dscale) {
    V2Profile* prof = v2_profile_current();
    const auto residual_begin = prof ? V2Clock::now() : V2Clock::time_point{};
    qf worst = 0;
    for (const auto& r : roots) {
        qf res = cabs(poly_eval_c(desc, deg, r)) / (dscale + (qf)1e-300);
        if (res > worst) worst = res;
    }
    if (prof)
        v2_profile_add_ms(&V2Profile::d14_residual_eval_ms, residual_begin,
                          V2Clock::now());
    return worst;
}

struct D14ActivePresearchResult {
    std::vector<Cplx<double>> roots;
    std::vector<Cplx<double>> last_finite_roots;
    int iterations = 0;
    int skipped_updates = 0;
    int failure_code = 0;
    int failure_iteration = 0;
    int failure_root = -1;
    int failure_partner = -1;
    double failure_abs_p = std::numeric_limits<double>::quiet_NaN();
    double failure_abs_dp = std::numeric_limits<double>::quiet_NaN();
    double failure_nearest_separation = std::numeric_limits<double>::quiet_NaN();
    double failure_abs_sum = std::numeric_limits<double>::quiet_NaN();
    double failure_abs_denominator = std::numeric_limits<double>::quiet_NaN();
    double failure_abs_correction = std::numeric_limits<double>::quiet_NaN();
    bool has_last_finite_roots = false;
    bool finite = true;
};

// Fixed-degree research variant of the balanced double Aberth basin search.
// A root whose correction is small for `patience` consecutive sweeps is
// temporarily made inactive.  It remains in every other root's interaction
// sum, so this is a work scheduler rather than deflation.  The inactive roots
// are not a proof of convergence; solve_d14's D14Real polish and independent
// qf residual/completeness checks decide whether the candidate is accepted.
inline D14ActivePresearchResult d14_active_presearch(
    const double* coeffs, int deg, int max_iter,
    const Cplx<double>* seed, double tol, int patience,
    bool preserve_last_finite = false) {
    D14ActivePresearchResult out;
    const bool trace = holo_d14_presearch_trace_enabled();
    preserve_last_finite = preserve_last_finite || trace;
    if (!coeffs || deg <= 0 || max_iter <= 0 || !(tol > 0.0) || patience <= 0) {
        out.finite = false;
        out.failure_code = 1;
        return out;
    }
    out.roots.resize(deg);

    double bound = 1.0;
    if (!seed) {
        const double an = std::fabs(coeffs[0]);
        if (!(an > 0.0) || !std::isfinite(an)) {
            out.finite = false;
            out.failure_code = 2;
            return out;
        }
        for (int i = 1; i <= deg; ++i)
            bound = std::max(bound, 1.0 + std::fabs(coeffs[i]) / an);
    }
    const double pi = 3.14159265358979323846264338327950288;
    for (int i = 0; i < deg; ++i) {
        if (seed) {
            out.roots[i] = seed[i];
        } else {
            const double angle = 2.0 * pi * static_cast<double>(i) /
                                     static_cast<double>(deg) + 0.4;
            const double radius = bound * (0.5 + 0.5 * static_cast<double>(i) /
                                                   static_cast<double>(deg));
            out.roots[i] = Cplx<double>(
                radius * std::cos(angle), radius * std::sin(angle));
        }
        if (!std::isfinite(out.roots[i].re) ||
            !std::isfinite(out.roots[i].im)) {
            out.finite = false;
            out.failure_code = 2;
            out.failure_root = i;
            return out;
        }
    }
    if (preserve_last_finite) {
        out.last_finite_roots = out.roots;
        out.has_last_finite_roots = true;
    }

    std::vector<unsigned char> active(deg, 1), stable(deg, 0);
    const double tol2 = tol * tol;
    const bool legacy = holo_legacy_complex_ops();
    auto abs_complex = [](const Cplx<double>& z) {
        return std::hypot(z.re, z.im);
    };
    auto fail = [&](int code, int it, int root, int partner,
                    const Cplx<double>& p, const Cplx<double>& dp,
                    double nearest, const Cplx<double>& sum,
                    const Cplx<double>& denominator,
                    const Cplx<double>& correction) {
        out.finite = false;
        out.failure_code = code;
        out.failure_iteration = it + 1;
        out.failure_root = root;
        out.failure_partner = partner;
        out.failure_abs_p = abs_complex(p);
        out.failure_abs_dp = abs_complex(dp);
        out.failure_nearest_separation = nearest;
        out.failure_abs_sum = abs_complex(sum);
        out.failure_abs_denominator = abs_complex(denominator);
        out.failure_abs_correction = abs_complex(correction);
        if (preserve_last_finite && out.has_last_finite_roots)
            out.roots = out.last_finite_roots;
        if (trace) {
            const Cplx<double> z = root >= 0 && root < deg
                ? out.roots[root] : Cplx<double>(0.0, 0.0);
            const Cplx<double> scaled_correction = p * crecip(denominator);
            const double p_scale = std::max(std::fabs(p.re), std::fabs(p.im));
            const double b_scale = std::max(std::fabs(denominator.re),
                                            std::fabs(denominator.im));
            const int product_exponent = p_scale > 0.0 && b_scale > 0.0 &&
                                         std::isfinite(p_scale) &&
                                         std::isfinite(b_scale)
                ? std::ilogb(p_scale) + std::ilogb(b_scale) : -9999;
            std::fprintf(stderr,
                "D14PRE_FAIL\tcode=%d\tsweep=%d\troot=%d\tpartner=%d"
                "\tzRe=%.17g\tzIm=%.17g\tabsZ=%.17g"
                "\tabsP=%.17g\tabsDP=%.17g\tminSep=%.17g\tabsS=%.17g"
                "\tabsB=%.17g\tabsW=%.17g\tscaledWRe=%.17g"
                "\tscaledWIm=%.17g\tproductExponent=%d\tcheckpoint=%d\n",
                code, it + 1, root, partner, z.re, z.im, abs_complex(z),
                out.failure_abs_p,
                out.failure_abs_dp, nearest, out.failure_abs_sum,
                out.failure_abs_denominator, out.failure_abs_correction,
                scaled_correction.re, scaled_correction.im,
                product_exponent, int(out.has_last_finite_roots));
        }
    };
    for (int it = 0; it < max_iter; ++it) {
        out.iterations = it + 1;
        if (preserve_last_finite) {
            // This snapshot is the last complete finite sweep.  An in-place
            // update that fails halfway through must not leak a partial root
            // set into the higher-precision interaction sum.
            out.last_finite_roots = out.roots;
            out.has_last_finite_roots = true;
        }
        double max_step2 = 0.0;
        int active_count = 0;
        for (int i = 0; i < deg; ++i) {
            if (!active[i]) {
                ++out.skipped_updates;
                continue;
            }
            ++active_count;
            const Cplx<double> p = poly_eval_c(coeffs, deg, out.roots[i]);
            const Cplx<double> dp = polyder_eval_c(coeffs, deg, out.roots[i]);
            if (!std::isfinite(p.re) || !std::isfinite(p.im) ||
                !std::isfinite(dp.re) || !std::isfinite(dp.im)) {
                fail(3, it, i, -1, p, dp,
                     std::numeric_limits<double>::quiet_NaN(),
                     Cplx<double>(0.0, 0.0), Cplx<double>(0.0, 0.0),
                     Cplx<double>(0.0, 0.0));
                return out;
            }
            Cplx<double> sum(0.0, 0.0);
            double nearest_separation = std::numeric_limits<double>::infinity();
            int nearest_partner = -1;
            for (int j = 0; j < deg; ++j) {
                if (j == i) continue;
                const Cplx<double> d = out.roots[i] - out.roots[j];
                const double dn2 = d.re * d.re + d.im * d.im;
                if (trace) {
                    const double separation = std::hypot(d.re, d.im);
                    if (separation < nearest_separation) {
                        nearest_separation = separation;
                        nearest_partner = j;
                    }
                }
                if (!(dn2 > 0.0) || !std::isfinite(dn2)) {
                    fail(4, it, i, j, p, dp, trace ? nearest_separation :
                         std::numeric_limits<double>::quiet_NaN(), sum,
                         Cplx<double>(0.0, 0.0), Cplx<double>(0.0, 0.0));
                    return out;
                }
                sum = sum + Cplx<double>(1.0, 0.0) / d;
                if (!std::isfinite(sum.re) || !std::isfinite(sum.im)) {
                    fail(5, it, i, j, p, dp, trace ? nearest_separation :
                         std::numeric_limits<double>::quiet_NaN(), sum,
                         Cplx<double>(0.0, 0.0), Cplx<double>(0.0, 0.0));
                    return out;
                }
            }
            const Cplx<double> denominator = dp - p * sum;
            const Cplx<double> w = [&] {
                if (legacy) {
                    const Cplx<double> ratio = p / dp;
                    return ratio / (Cplx<double>(1.0, 0.0) - ratio * sum);
                }
                return p / (dp - p * sum);
            }();
            if (!std::isfinite(w.re) || !std::isfinite(w.im)) {
                fail(6, it, i, nearest_partner, p, dp,
                     trace ? nearest_separation :
                         std::numeric_limits<double>::quiet_NaN(),
                     sum, denominator, w);
                return out;
            }
            const double step2 = cabs2(w);
            const Cplx<double> candidate = out.roots[i] - w;
            if (!std::isfinite(candidate.re) || !std::isfinite(candidate.im)) {
                fail(7, it, i, nearest_partner, p, dp,
                     trace ? nearest_separation :
                         std::numeric_limits<double>::quiet_NaN(),
                     sum, denominator, w);
                return out;
            }
            if (!std::isfinite(step2)) {
                fail(8, it, i, nearest_partner, p, dp,
                     trace ? nearest_separation :
                         std::numeric_limits<double>::quiet_NaN(),
                     sum, denominator, w);
                return out;
            }
            if (trace) {
                std::fprintf(stderr,
                    "D14PRE\tstatus=commit\tsweep=%d\troot=%d"
                    "\tzRe=%.17g\tzIm=%.17g\tabsP=%.17g\tabsDP=%.17g"
                    "\tminSep=%.17g\tnearest=%d\tabsS=%.17g\tabsB=%.17g"
                    "\tabsW=%.17g\tstepSep=%.17g\tcandidateFinite=1\n",
                    it + 1, i, out.roots[i].re, out.roots[i].im,
                    abs_complex(p), abs_complex(dp), nearest_separation,
                    nearest_partner, abs_complex(sum), abs_complex(denominator),
                    abs_complex(w), nearest_separation > 0.0
                        ? abs_complex(w) / nearest_separation
                        : std::numeric_limits<double>::infinity());
            }
            out.roots[i] = candidate;
            max_step2 = std::max(max_step2, step2);
            if (step2 < tol2) {
                if (stable[i] < 255) ++stable[i];
                if (stable[i] >= patience) active[i] = 0;
            } else {
                stable[i] = 0;
            }
        }
        if (preserve_last_finite) out.last_finite_roots = out.roots;
        if (active_count == 0 || max_step2 < tol2) break;
    }
    if (trace)
        std::fprintf(stderr,
            "D14PRE_DONE\tfinite=%d\tsweeps=%d\tlastFinite=%d\tfailureCode=%d\n",
            int(out.finite), out.iterations, int(out.has_last_finite_roots),
            out.failure_code);
    return out;
}

// Two/three-tier D14 solve.  `desc` descending __float128, length deg+1.
//
// Shared prefix: balance the monomial basis (v = s w, decision 21) and run
// a `double` Aberth-Ehrlich presearch to locate every root basin.
//
// Polish:
//   compensated == true  -> double-double Aberth warm-started from the
//     presearch (tier 0).  If its __float128 residual still exceeds 1e-12
//     (a near-multiple cluster beyond ~106 bits) escalate to an
//     __float128 warm polish seeded by the dd result (tier 1).
//   compensated == false -> __float128 warm polish directly (tier 3).
//
// Backstop (either path): non-finite presearch or a failed residual gate
// -> cold __float128 Aberth (tier 2 / -1).
// `warm_seed` (optional, Phase B2): a previous epoch's D14 root set in the
// same v = R^2 space, length deg.  When present and finite it seeds the
// `double` root-basin presearch instead of a cold Aberth-Ehrlich spread --
// on a smoothly drifting trajectory the roots move O(drift), so a short
// warm refine locates every basin and the compensated dd polish then
// converges in a handful of sweeps.  Every downstream gate (residual,
// Newton-sum completeness, cold __float128 backstop) is unchanged, so a
// stale or wrong seed still fails closed to the cold solve.
#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
inline bool d14_event_contract_screen(
    const PrimaryFrame& pf,const D14StructQf& sc,
    const std::vector<qf>& desc_v,
    std::vector<Cplx<qf>>& roots) {
    if(!d14_rouche_fast_screen(desc_v,roots))return false;
    const auto certificate=d14_rouche_certificate(pf,roots);
    return d14_rouche_refine(sc,certificate,roots);
}
#endif

inline D14Solve solve_d14(const std::vector<qf>& desc_v, int deg,
                          bool compensated,
                          const std::vector<Cplx<qf>>* warm_seed = nullptr,
                          const D14StructQf* sc = nullptr,
                          const PrimaryFrame* event_pf = nullptr,
                          double event_rmax = 0.0) {
#if defined(HOLO_D14_QF_WARM_MAX_ITER_OVERRIDE)
    constexpr int qf_warm_max_iter = HOLO_D14_QF_WARM_MAX_ITER_OVERRIDE;
#else
    constexpr int qf_warm_max_iter = 24;
#endif
#if defined(HOLO_D14_REAL_MAX_ITER_OVERRIDE)
    constexpr int d14_real_max_iter = HOLO_D14_REAL_MAX_ITER_OVERRIDE;
#else
    constexpr int d14_real_max_iter = 25;
#endif
    const qf* desc = desc_v.data();
    D14Solve out;
    V2Profile* prof = v2_profile_current();
    if (prof) ++prof->d14_solve_calls;
    V2ProfileTimer solve_timer(&V2Profile::d14_solve_ms);

    qf dscale = 0;
    for (int i = 0; i <= deg; ++i) {
        qf av = fabsq(desc[i]);
        if (av > dscale) dscale = av;
    }

    double sscale = 1.0;
    if ((double)fabsq(desc[deg]) > 0.0 && (double)fabsq(desc[0]) > 0.0) {
        double ratio = (double)(fabsq(desc[deg]) / fabsq(desc[0]));
        sscale = std::pow(ratio, 1.0 / deg);
        if (!(sscale > 0.0) || !std::isfinite(sscale)) sscale = 1.0;
    }
    bool direct_warm = false;
    if (warm_seed && sc && compensated && holo_d14_direct_warm_enabled()) {
        if (prof) ++prof->d14_direct_warm_attempts;
        const auto screen_begin = V2Clock::now();
        direct_warm = d14_warm_seed_screen(*sc, *warm_seed, deg, dscale);
        if (prof)
            v2_profile_add_ms(&V2Profile::d14_warm_screen_ms, screen_begin,
                              V2Clock::now());
        if (!direct_warm && prof) ++prof->d14_direct_warm_reject;
    }
    std::vector<double> descd(deg + 1);
    {
        double sp = 1.0;
        for (int i = deg; i >= 0; --i) {
            descd[i] = (double)desc[i] * sp;
            sp *= sscale;
        }
    }
    // Root-basin presearch in balanced `double`.  Cold by default; when a
    // previous epoch's roots are supplied, refine from them (balanced the
    // same way, w = v / sscale) in a few iterations instead.
    std::vector<Cplx<double>> presearch_seed;
    if (warm_seed && (int)warm_seed->size() == deg && sscale > 0.0) {
        presearch_seed.resize(deg);
        bool ok = true;
        for (int i = 0; i < deg; ++i) {
            double wr = (double)(*warm_seed)[i].re / sscale;
            double wi = (double)(*warm_seed)[i].im / sscale;
            if (!std::isfinite(wr) || !std::isfinite(wi)) { ok = false; break; }
            presearch_seed[i] = Cplx<double>(wr, wi);
        }
        if (!ok) presearch_seed.clear();
    }
    int pre_iters = 0;
    auto pre_begin = V2Clock::now();
    const int pre_max = holo_d14_presearch_max(!presearch_seed.empty());
    const double pre_tol = !presearch_seed.empty()
                               ? holo_d14_presearch_tol()
                               : 0.0;
    std::vector<Cplx<double>> zd;
    if (direct_warm) {
        // Keep a rounded copy for optional research candidates.  The
        // production D14Real path below uses the original qf seed so its
        // low limb is not discarded.
        zd.resize(deg);
        for (int i = 0; i < deg; ++i)
            zd[i] = Cplx<double>(static_cast<double>((*warm_seed)[i].re) / sscale,
                                 static_cast<double>((*warm_seed)[i].im) / sscale);
    } else if (!presearch_seed.empty() && holo_d14_skip_warm_presearch()) {
        // The seed was produced by a previously certified D14 solve.  This
        // is a research-only warm trajectory experiment: the DD/qf polish,
        // residual gate, and global certificate still own correctness.
        zd = presearch_seed;
    } else if (holo_d14_active_presearch_enabled()) {
        const bool preserve_last_finite =
            holo_d14_last_finite_handoff_enabled() ||
            holo_d14_presearch_trace_enabled();
        const auto active = d14_active_presearch(
            descd.data(), deg, pre_max,
            presearch_seed.empty() ? nullptr : presearch_seed.data(),
            holo_d14_active_presearch_tol(),
            holo_d14_active_presearch_patience(), preserve_last_finite);
        zd = active.roots;
        pre_iters = active.iterations;
        if (prof) {
            ++prof->d14_presearch_active_calls;
            prof->d14_presearch_active_sweeps +=
                static_cast<V2Profile::u64>(active.iterations);
            prof->d14_presearch_active_skips +=
                static_cast<V2Profile::u64>(active.skipped_updates);
            if (active.failure_code != 0) {
                ++prof->d14_presearch_nonfinite_failures;
                if (prof->d14_presearch_failure_code == 0) {
                    prof->d14_presearch_failure_code = active.failure_code;
                    prof->d14_presearch_failure_iteration =
                        active.failure_iteration;
                    prof->d14_presearch_failure_root = active.failure_root;
                }
            }
        }
        bool active_roots_finite = static_cast<int>(zd.size()) == deg;
        for (int i = 0; active_roots_finite && i < deg; ++i)
            active_roots_finite = std::isfinite(zd[i].re) &&
                                  std::isfinite(zd[i].im);
        if (!active.finite || !active_roots_finite) {
            bool checkpoint_finite = active.has_last_finite_roots &&
                static_cast<int>(active.last_finite_roots.size()) == deg;
            for (int i = 0; checkpoint_finite && i < deg; ++i)
                checkpoint_finite = std::isfinite(active.last_finite_roots[i].re) &&
                                    std::isfinite(active.last_finite_roots[i].im);
            if (holo_d14_last_finite_handoff_enabled() && checkpoint_finite) {
                zd = active.last_finite_roots;
                if (prof) ++prof->d14_presearch_last_finite_handoffs;
            } else {
            // Scheduler failure is not a reason to jump directly to the
            // expensive qf cold solve.  Re-run the incumbent double basin
            // search from the same certified warm seed (or cold spread),
            // then keep all existing D14Real/qf and root-set certificates.
                int fallback_iters = 0;
                zd = !presearch_seed.empty()
                         ? aberth<double>(descd.data(), deg, pre_max,
                                          presearch_seed.data(), pre_tol, nullptr,
                                          &fallback_iters)
                         : aberth<double>(descd.data(), deg, pre_max, nullptr,
                                          0.0, nullptr, &fallback_iters);
                pre_iters += fallback_iters;
                if (prof) ++prof->d14_presearch_active_fallbacks;
            }
        }
    } else if (presearch_seed.empty()) {
        zd = aberth<double>(descd.data(), deg, pre_max, nullptr,
                            pre_tol, nullptr, &pre_iters);
    } else {
        zd = aberth<double>(descd.data(), deg, pre_max, presearch_seed.data(),
                            pre_tol, nullptr, &pre_iters);
    }
    if (prof) {
        prof->d14_presearch_sweeps += (V2Profile::u64)pre_iters;
        v2_profile_add_ms(&V2Profile::d14_presearch_ms, pre_begin, V2Clock::now());
    }
    out.warm_seeded = !presearch_seed.empty();
    if (prof && prof->capture_d14_root_work &&
        static_cast<int>(zd.size()) == deg) {
        for (int i = 0; i < deg; ++i) {
            auto& work = out.root_work[i];
            work.double_seed_index = i;
            work.double_seed_v_re = zd[i].re * sscale;
            work.double_seed_v_im = zd[i].im * sscale;
            work.double_seed_valid = std::isfinite(work.double_seed_v_re) &&
                                     std::isfinite(work.double_seed_v_im);
        }
    }
    bool seed_ok = static_cast<int>(zd.size()) == deg;
    for (int i = 0; seed_ok && i < deg; ++i)
        if (!std::isfinite(zd[i].re) || !std::isfinite(zd[i].im)) {
            seed_ok = false;
            break;
        }
#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
    // Diagnostic snapshot of the balanced binary64 basin locator before any
    // D14Real/qf polish.  This candidate is never accepted here; the event
    // contract harness compares its downstream events and adaptive value with
    // the fully certified oracle produced below.
    if (d14_event_contract_capture && seed_ok) {
        std::vector<Cplx<qf>> presearch_roots(deg);
        for (int i = 0; i < deg; ++i)
            presearch_roots[i] = Cplx<qf>((qf)zd[i].re * (qf)sscale,
                                          (qf)zd[i].im * (qf)sscale);
        d14_capture_event_contract_candidate(
            out, "presearch", presearch_roots,
            (qf)d14_worst_res(desc, deg, presearch_roots, dscale), false);
    }
#endif
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
    const char* seed_reject_reason = seed_ok ? "" : "presearch_nonfinite_seed";
#endif
    // Block-form D/D' polish (memo section 3): exact C3/G4/Z3 evaluation
    // in place of Horner on the cancellation-carrying expanded vector.
    const bool use_struct = sc && holo_d14_struct_enabled() && deg == 14 &&
                            !holo_d14_horner_enabled();
    if (prof && use_struct) ++prof->d14_struct_calls;
    if (prof && holo_d14_horner_enabled()) ++prof->d14_horner_calls;
    D14StructC<DD> scdd;
    D14StructC<D14Real> screal;
    D14StructQf scqf;
    if (use_struct) {
        // Only one compensated kernel is entered below.  Constructing both
        // converted coefficient blocks charged every normal epoch for a DD
        // block that the default D14Real path never reads.
        if (holo_d14_real_enabled()) {
            screal = d14_struct_cast<D14Real>(*sc);
#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
            // A/B oracle for the removed zero work.  This branch is absent
            // from ordinary builds and never changes the selected solver.
            if (holo_d14_eager_dd_block_research_enabled())
                scdd = d14_struct_cast<DD>(*sc);
#endif
        } else
            scdd = d14_struct_cast<DD>(*sc);
        scqf = *sc;
    }

    bool used_lifted = false;
    bool used_hybrid = false;
    if (seed_ok && use_struct && holo_d14_hybrid_enabled()) {
        if (prof) ++prof->d14_hybrid_attempts;
        auto hybrid_begin = V2Clock::now();
        auto hybrid = d14_hybrid_solve(*sc, desc_v, zd, sscale);
        qf hybrid_res = 0;
        const bool cert = d14_scalar_certificate(sc, desc_v, hybrid.roots,
                                                 &hybrid_res);
        if (prof) {
            prof->d14_hybrid_cheap_calls +=
                static_cast<V2Profile::u64>(hybrid.cheap_calls);
            prof->d14_hybrid_structural_calls +=
                static_cast<V2Profile::u64>(hybrid.structural_calls);
            prof->d14_hybrid_unsafe_calls +=
                static_cast<V2Profile::u64>(hybrid.unsafe_calls);
            v2_profile_add_ms(&V2Profile::d14_hybrid_ms, hybrid_begin,
                              V2Clock::now());
        }
        if (hybrid.converged && cert) {
            out.roots = std::move(hybrid.roots);
            out.worst_res = hybrid_res;
            out.tier = 0;
            used_hybrid = true;
            if (prof) ++prof->d14_hybrid_success;
        } else if (prof) {
            ++prof->d14_hybrid_certificate_fail;
        }
    }
    if (seed_ok && use_struct && holo_d14_lifted_enabled()) {
        if (prof) ++prof->d14_lifted_attempts;
        auto lift_begin = V2Clock::now();
        auto lifted = d14_lifted_solve(*sc, zd, sscale);
        qf lifted_res = 0;
        qf lifted_rec = 0;
        int lifted_reason = 0;
        const bool cert = d14_lifted_certificate(*sc, desc_v, lifted,
                                                 &lifted_res, &lifted_reason,
                                                 &lifted_rec);
        if (prof)
            prof->d14_lifted_max_reconstruct =
                std::max(prof->d14_lifted_max_reconstruct, (double)lifted_rec);
        if (prof)
            v2_profile_add_ms(&V2Profile::d14_qf_ms, lift_begin, V2Clock::now());
        if (cert) {
            out.roots = std::move(lifted.roots);
            out.worst_res = lifted_res;
            out.tier = 0;
            used_lifted = true;
            if (prof) ++prof->d14_lifted_success;
        } else if (prof) {
            ++prof->d14_lifted_certificate_fail;
            if (lifted.failure_code == 1) ++prof->d14_lifted_fail_seed;
            else if (lifted.failure_code == 2) ++prof->d14_lifted_fail_newton;
            else if (lifted.failure_code == 3) ++prof->d14_lifted_fail_scalar;
            else if (lifted_reason == 4) ++prof->d14_lifted_fail_scalar;
            else if (lifted_reason == 3) ++prof->d14_lifted_fail_lift;
            else if (lifted_reason == 5) ++prof->d14_lifted_fail_conjugacy;
            else if (lifted_reason == 6) ++prof->d14_lifted_fail_vieta;
            else if (lifted_reason == 7) ++prof->d14_lifted_fail_reconstruct;
        }
    }

    bool used_real = false;
    if (!used_hybrid && !used_lifted && seed_ok && compensated &&
        use_struct && holo_d14_real_enabled()) {
        ScopedNoFlushDenormals _eft;
        std::array<Cplx<D14Real>, 14> seed{};
        std::array<int, 14> role_hints{};
        D14RealScheduleConfig schedule = holo_d14_real_schedule_config();
        const bool capture_root_work = prof && prof->capture_d14_root_work;
        schedule.capture_diagnostics = capture_root_work;
        if ((schedule.enabled && schedule.consumer_precision) || capture_root_work) {
            for (int i = 0; i < deg; ++i) {
                const double vr = zd[i].re * sscale;
                const double vi = zd[i].im * sscale;
                const double real_cut = 1e-8 * (1.0 + std::fabs(vr));
                if (vr > 0.0 && std::fabs(vi) <= real_cut)
                    role_hints[i] = static_cast<int>(D14RootUse::PositiveRealCandidate);
                else if (vr > 0.0 && std::fabs(vi) > real_cut)
                    role_hints[i] = static_cast<int>(D14RootUse::ComplexSoftCut);
            }
        }
        for (int i = 0; i < deg; ++i) {
            if (direct_warm) {
                seed[i] = Cplx<D14Real>(d14_from_qf((*warm_seed)[i].re),
                                        d14_from_qf((*warm_seed)[i].im));
            } else {
                seed[i] = Cplx<D14Real>(D14Real(zd[i].re * sscale),
                                        D14Real(zd[i].im * sscale));
            }
        }
        auto real_begin = V2Clock::now();
        D14RealAberthResult real;
        if (direct_warm) {
            const D14RealNewtonResult warm =
                d14_real_warm_newton(screal, seed, 16, holo_d14_real_tol());
            real.roots = warm.roots;
            real.iterations = warm.iterations;
            real.finite = warm.finite;
            real.converged = warm.converged;
            if (capture_root_work && real.finite) {
                for (int i = 0; i < deg; ++i) {
                    real.root_updates[i] = warm.iterations;
                    Cplx<D14Real> p, dp;
                    d14_struct_eval(screal, real.roots[i], p, dp);
                    if (qfinite_(p.re) && qfinite_(p.im) &&
                        qfinite_(dp.re) && qfinite_(dp.im) &&
                        !(dp.re == D14Real(0.0) && dp.im == D14Real(0.0))) {
                        const Cplx<D14Real> correction = p / dp;
                        real.final_newton_correction[i] = std::hypot(
                            static_cast<double>(correction.re),
                            static_cast<double>(correction.im));
                    }
                    double nearest = std::numeric_limits<double>::infinity();
                    for (int j = 0; j < deg; ++j) {
                        if (i == j) continue;
                        nearest = std::min(nearest, std::hypot(
                            static_cast<double>(real.roots[i].re) -
                                static_cast<double>(real.roots[j].re),
                            static_cast<double>(real.roots[i].im) -
                                static_cast<double>(real.roots[j].im)));
                    }
                    real.nearest_separation[i] = nearest;
                    real.relative_newton_correction[i] = nearest > 0.0
                        ? real.final_newton_correction[i] / nearest
                        : std::numeric_limits<double>::infinity();
                    real.cluster_id[i] = i;
                    real.cluster_size[i] = 1;
                }
            }
        } else {
            real = aberth_d14_real_mixed(
                screal, seed, d14_real_max_iter, holo_d14_real_tol(),
                holo_d14_local_pairs_enabled(),
                (schedule.enabled || capture_root_work) ? &schedule : nullptr,
                &role_hints);
        }
        if (prof) {
            ++prof->d14_real_calls;
            if (holo_d14_local_pairs_enabled()) {
                ++prof->d14_real_local_pair_calls;
                prof->d14_real_local_pair_rows +=
                    static_cast<V2Profile::u64>(real.dangerous_rows);
            }
            prof->d14_real_mixed_pairs +=
                static_cast<V2Profile::u64>(real.mixed_pairs);
            prof->d14_real_dangerous_pairs +=
                static_cast<V2Profile::u64>(real.dangerous_pairs);
            prof->d14_real_full_recompute_rows +=
                static_cast<V2Profile::u64>(real.full_recompute_rows);
            prof->d14_real_root_updates += static_cast<V2Profile::u64>(
                std::accumulate(real.root_updates.begin(), real.root_updates.end(), 0));
            prof->d14_real_root_skips +=
                static_cast<V2Profile::u64>(real.skipped_updates);
            prof->d14_real_root_reactivations +=
                static_cast<V2Profile::u64>(real.reactivations);
            prof->d14_real_cluster_wakeups +=
                static_cast<V2Profile::u64>(real.cluster_wakeups);
            for (int i = 0; i < deg; ++i)
                prof->d14_real_root_freezes +=
                    static_cast<V2Profile::u64>(real.root_freezes[i]);
            if (real.finite) ++prof->d14_real_finite_calls;
            if (!real.converged) ++prof->d14_real_nonconverged;
            if (direct_warm && real.converged)
                ++prof->d14_direct_newton_converged;
            if (direct_warm && !real.finite)
                ++prof->d14_direct_newton_nonfinite;
            v2_profile_add_ms(&V2Profile::d14_real_ms, real_begin,
                              V2Clock::now());
        }
        if (real.finite) {
            out.roots.resize(deg);
            for (int i = 0; i < deg; ++i) {
                out.roots[i] = Cplx<qf>(d14_to_qf(real.roots[i].re),
                                        d14_to_qf(real.roots[i].im));
                if (capture_root_work) {
                    out.d14real_roots[i] = out.roots[i];
                    out.has_d14real_roots = true;
                    auto& work = out.root_work[i];
                    work.root_index = i;
                    work.role_hint = role_hints[i];
                    work.d14real_updates = real.root_updates[i];
                    work.cluster_id = real.cluster_id[i];
                    work.cluster_size = real.cluster_size[i];
                    work.freeze_count = real.root_freezes[i];
                    work.reactivation_count = real.root_reactivations[i];
                    work.v_re = static_cast<double>(out.roots[i].re);
                    work.v_im = static_cast<double>(out.roots[i].im);
                    work.d14real_newton_correction =
                        real.final_newton_correction[i];
                    work.d14real_nearest_separation = real.nearest_separation[i];
                    work.d14real_relative_correction =
                        real.relative_newton_correction[i];
                    const double projected_r = std::sqrt(std::max(0.0, work.v_re));
                    work.d14real_position_error = projected_r > 0.0
                        ? work.d14real_newton_correction / (2.0 * projected_r)
                        : -1.0;
                }
            }
            out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
            out.tier = 0;
            used_real = true;
#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
            d14_capture_event_contract_candidate(
                out, "d14real", out.roots, out.worst_res, real.converged);
#endif
            // A finite D14Real iterate is not necessarily a usable complete
            // root set. In particular, ill-conditioned small roots can have
            // tiny polynomial residuals far from their correct locations.
            // Keep the finite roots as a seed and try the bounded qf warm
            // polish whenever D14Real failed its convergence test; only then
            // use the existing qf cold solve.
            if (!real.converged || !(out.worst_res <= (qf)1e-13)) {
                int qf_iters = 0;
                bool qf_converged = false;
                auto qf_begin = V2Clock::now();
                out.roots = aberth_d14_struct<qf>(
                    scqf, nullptr, qf_warm_max_iter, out.roots.data(), (qf)1e-20,
                    &qf_iters, &qf_converged, role_hints.data());
                out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
                out.tier = 1;
#if defined(HOLO_D14_QF_PAIR_POLISH_RESEARCH)
                int pair_verify_iters = 0;
                bool pair_attempted = false;
                bool pair_accepted = false;
                bool pair_scalar_certificate = false;
                int pair_i = -1, pair_j = -1;
                qf pair_driver_step = 0;
                qf pair_check_residual = out.worst_res;
                D14QfPairPolishResearchResult pair_result;
                const char* pair_switch = std::getenv(
                    "HOLO_D14_QF_PAIR_POLISH_RESEARCH");
                const bool pair_research_enabled =
                    pair_switch && pair_switch[0] == '1';
                const char* pair_probe_switch = std::getenv(
                    "HOLO_D14_QF_PAIR_PROBE");
                const bool pair_probe_enabled =
                    pair_probe_switch && pair_probe_switch[0] == '1';
                if (!qf_converged && out.worst_res <= (qf)1e-12 &&
                    pair_research_enabled &&
                    d14_qf_aberth_driver_pair(scqf, out.roots, &pair_i,
                                              &pair_j, &pair_driver_step)) {
                    pair_attempted = true;
                    pair_result = d14_qf_pair_polish_research(
                        scqf, out.roots, pair_i, pair_j, 8, (qf)1e-20,
                        pair_probe_enabled);
                    // A non-converged local pair trial is not a useful seed
                    // for the global verification sweep, except when the
                    // one-step probe certifies that the complete post-probe
                    // Aberth correction is already within the incumbent
                    // global step tolerance. The probe itself never accepts
                    // a root set; all existing global gates still run below.
                    if (pair_result.finite &&
                        (pair_result.converged ||
                         pair_result.probe_global_ready)) {
                        bool verify_converged = false;
                        std::vector<Cplx<qf>> verified =
                            aberth_d14_struct<qf>(
                                scqf, nullptr, 1, pair_result.roots.data(),
                                (qf)1e-20, &pair_verify_iters,
                                &verify_converged, role_hints.data());
                        pair_check_residual =
                            d14_worst_res(desc, deg, verified, dscale);
                        qf scalar_residual = 0;
                        int scalar_reason = 0;
                        pair_scalar_certificate = d14_scalar_certificate(
                            &scqf, desc_v, verified, &scalar_residual,
                            &scalar_reason);
                        if (verify_converged &&
                            pair_check_residual <= (qf)1e-12 &&
                            pair_scalar_certificate) {
                            out.roots = std::move(verified);
                            out.worst_res = pair_check_residual;
                            qf_converged = true;
                            pair_accepted = true;
                        }
                    }
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
                    if (re_detail::d14_qf_trace_enabled()) {
                        std::fprintf(stderr,
                            "D14QF_PAIRPOLISH\tpair=%d,%d\tattempted=%d"
                            "\taccepted=%d\tpair_finite=%d\tpair_converged=%d"
                            "\tpair_iters=%d\tbacktracks=%d\tverify_iters=%d"
                            "\tverify_converged=%d\tscalar_certificate=%d"
                            "\tprobe_available=%d\tprobe_finite=%d"
                            "\tprobe_backtracks=%d\tprobe_driver_stable=%d"
                            "\tprobe_viable=%d\tprobe_global_ready=%d"
                            "\tprobe_driver=%d,%d",
                            pair_i, pair_j, int(pair_attempted),
                            int(pair_accepted), int(pair_result.finite),
                            int(pair_result.converged), pair_result.iterations,
                            pair_result.backtracks, pair_verify_iters,
                            int(qf_converged), int(pair_scalar_certificate),
                            int(pair_result.probe_available),
                            int(pair_result.probe_finite),
                            pair_result.probe_backtracks,
                            int(pair_result.probe_driver_stable),
                            int(pair_result.probe_viable),
                            int(pair_result.probe_global_ready),
                            pair_result.probe_driver_i,
                            pair_result.probe_driver_j);
                        re_detail::d14_qf_trace_value("driver_step",
                                                      pair_driver_step);
                        re_detail::d14_qf_trace_value(
                            "start_pair_residual", pair_result.start_residual);
                        re_detail::d14_qf_trace_value(
                            "end_pair_residual", pair_result.end_residual);
                        re_detail::d14_qf_trace_value(
                            "pair_step", pair_result.last_pair_step);
                        re_detail::d14_qf_trace_value(
                            "probe_full_ratio",
                            pair_result.probe_full_step_ratio);
                        re_detail::d14_qf_trace_value(
                            "probe_accepted_ratio",
                            pair_result.probe_accepted_ratio);
                        re_detail::d14_qf_trace_value(
                            "probe_step_over_sep",
                            pair_result.probe_step_over_separation);
                        re_detail::d14_qf_trace_value(
                            "probe_det_fraction",
                            pair_result.probe_det_fraction);
                        re_detail::d14_qf_trace_value(
                            "probe_post_driver_step",
                            pair_result.probe_post_driver_step);
                        re_detail::d14_qf_trace_value(
                            "probe_post_outer_step",
                            pair_result.probe_post_outer_step);
                        re_detail::d14_qf_trace_value(
                            "check_worst_residual", pair_check_residual);
                        std::fputc('\n', stderr);
                    }
#endif
                }
#endif
#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
                d14_capture_event_contract_candidate(
                    out, "qf_warm", out.roots, out.worst_res, qf_converged);
                bool event_contract_accept=false;
                const char* event_switch=std::getenv("HOLO_D14_EVENT_CONTRACT_ACCEPT");
                if(!qf_converged&&out.worst_res<=(qf)1e-12&&event_pf&&
                   event_rmax>0&&event_switch&&event_switch[0]=='1') {
                    if(prof)++prof->d14_event_contract_attempts;
                    qf scalar_residual=0;int scalar_reason=0;
                    const bool scalar=d14_scalar_certificate(
                        &scqf,desc_v,out.roots,&scalar_residual,&scalar_reason);
                    if(scalar) {
                        if(d14_event_contract_screen(*event_pf,scqf,desc_v,out.roots)) {
                            out.event_contract_accepted=true;
                            event_contract_accept=true;
                            if(prof)++prof->d14_event_contract_accepts;
                        }
                    }
                }
#endif
                if (prof) {
                    ++prof->d14_qf_warm_calls;
                    prof->d14_qf_warm_sweeps +=
                        static_cast<V2Profile::u64>(qf_iters
#if defined(HOLO_D14_QF_PAIR_POLISH_RESEARCH)
                                                    + pair_verify_iters
#endif
                                                    );
                    const auto qf_end = V2Clock::now();
                    v2_profile_add_ms(&V2Profile::d14_qf_ms, qf_begin, qf_end);
                    v2_profile_add_ms(&V2Profile::d14_qf_polish_ms, qf_begin,
                                      qf_end);
                }
                if ((!qf_converged
#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
                     && !event_contract_accept
#endif
                    ) || !(out.worst_res <= (qf)1e-12)) {
                    seed_ok = false;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
                    seed_reject_reason = !qf_converged
                        ? "qf_warm_step_tolerance_not_met"
                        : "qf_warm_residual_gate_failed";
#endif
                }
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
                if (re_detail::d14_qf_trace_enabled()) {
                    int finite_roots = 0;
                    for (const auto& root : out.roots)
                        if (qfinite_(root.re) && qfinite_(root.im)) ++finite_roots;
                    qf certificate_residual = 0;
                    int certificate_reason = 0;
                    const bool scalar_certificate = d14_scalar_certificate(
                        use_struct ? &scqf : nullptr, desc_v, out.roots,
                        &certificate_residual, &certificate_reason);
                    std::fprintf(stderr,
                        "D14QF_STAGE\tkind=warm\tconverged=%d\tfinite_roots=%d"
                        "\titers=%d\tresidual=",
                        int(qf_converged), finite_roots, qf_iters);
                    re_detail::d14_qf_trace_value("worst", out.worst_res);
                    std::fprintf(stderr, "\tgate=1e-12\trejected=%d"
                        "\tscalar_certificate=%d\tcertificate_reason=%d",
                        int(!qf_converged || !(out.worst_res <= (qf)1e-12)),
                        int(scalar_certificate), certificate_reason);
                    re_detail::d14_qf_trace_value("certificate_residual",
                                                  certificate_residual);
                    std::fputc('\n', stderr);
                }
#endif
            }
        } else {
            seed_ok = false;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
            seed_reject_reason = "d14real_nonfinite";
#endif
        }
    }
    if (!used_hybrid && !used_lifted && !used_real && seed_ok && compensated) {
        // FTZ/DAZ off so the error-free transforms keep their lo limbs.
        ScopedNoFlushDenormals _eft;
        std::vector<DD> descdd(deg + 1);
        for (int i = 0; i <= deg; ++i) descdd[i] = dd_from_qf(desc[i]);
        std::vector<Cplx<DD>> seed(deg);
        for (int i = 0; i < deg; ++i) {
            if (direct_warm) {
                seed[i] = Cplx<DD>(dd_from_qf((*warm_seed)[i].re),
                                   dd_from_qf((*warm_seed)[i].im));
            } else {
                seed[i] = Cplx<DD>(DD((double)zd[i].re * sscale),
                                   DD((double)zd[i].im * sscale));
            }
        }
        // The balanced double presearch seeds every basin to ~1e-13 rel;
        // dd Aberth is locally cubic, so ~3 sweeps reach the ~1e-30 dd
        // floor.  tol 1e-26 is inside that floor (a step norm below it for
        // O(10) roots is dd round-off); 25 sweeps is headroom for a
        // poorly-seeded root (the residual gate escalates it if not).
        int dd_iters = 0;
        auto dd_begin = V2Clock::now();
        auto zdd =
            use_struct
                ? aberth_d14_struct<DD>(scdd, nullptr, 25, seed.data(), DD(1e-26),
                                        &dd_iters)
                : aberth<DD>(descdd.data(), deg, 25, seed.data(), DD(1e-26),
                             nullptr, &dd_iters);
        if (prof) {
            ++prof->d14_dd_calls;
            prof->d14_dd_sweeps += (V2Profile::u64)dd_iters;
            v2_profile_add_ms(&V2Profile::d14_dd_ms, dd_begin, V2Clock::now());
        }
        out.roots.resize(deg);
        for (int i = 0; i < deg; ++i)
            out.roots[i] = Cplx<qf>(qf_from_dd(zdd[i].re), qf_from_dd(zdd[i].im));
        out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
        out.tier = 0;
        // dd normally reaches ~1e-15 rel residual; a worse result signals a
        // near-multiple cluster past ~106 bits -> escalate that solve to an
        // __float128 warm polish seeded by the dd roots.
        if (!(out.worst_res <= (qf)1e-13)) {
            int qf_iters = 0;
            auto qf_begin = V2Clock::now();
            out.roots =
                use_struct
                    ? aberth_d14_struct<qf>(scqf, nullptr, 24, out.roots.data(),
                                            (qf)1e-20, &qf_iters)
                    : aberth<qf>(desc, deg, 24, out.roots.data(), (qf)1e-20,
                                 nullptr, &qf_iters);
            if (prof) {
                ++prof->d14_qf_warm_calls;
                prof->d14_qf_warm_sweeps += (V2Profile::u64)qf_iters;
                const auto qf_end = V2Clock::now();
                v2_profile_add_ms(&V2Profile::d14_qf_ms, qf_begin, qf_end);
                v2_profile_add_ms(&V2Profile::d14_qf_polish_ms, qf_begin, qf_end);
            }
            out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
            out.tier = 1;
            if (!(out.worst_res <= (qf)1e-12)) {
                seed_ok = false;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
                seed_reject_reason = "dd_qf_warm_residual_gate_failed";
#endif
            }
        }
    } else if (!used_hybrid && !used_lifted && !used_real && seed_ok) {
        std::vector<Cplx<qf>> seed(deg);
        for (int i = 0; i < deg; ++i)
            seed[i] = Cplx<qf>((qf)zd[i].re * (qf)sscale,
                               (qf)zd[i].im * (qf)sscale);
        int qf_iters = 0;
        auto qf_begin = V2Clock::now();
        out.roots =
            use_struct
                ? aberth_d14_struct<qf>(scqf, nullptr, 24, seed.data(), (qf)1e-20,
                                        &qf_iters)
                : aberth<qf>(desc, deg, 24, seed.data(), (qf)1e-20, nullptr,
                             &qf_iters);
        if (prof) {
            ++prof->d14_qf_warm_calls;
            prof->d14_qf_warm_sweeps += (V2Profile::u64)qf_iters;
            const auto qf_end = V2Clock::now();
            v2_profile_add_ms(&V2Profile::d14_qf_ms, qf_begin, qf_end);
            v2_profile_add_ms(&V2Profile::d14_qf_polish_ms, qf_begin, qf_end);
        }
        out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
        out.tier = 3;
        if (!(out.worst_res <= (qf)1e-12)) {
            seed_ok = false;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
            seed_reject_reason = "qf_warm_residual_gate_failed";
#endif
        }
    }

    // A direct warm corrector is a performance shortcut, not a reason to
    // jump straight to the expensive qf cold solve.  If it leaves the
    // incumbent residual gate, retry the normal balanced warm/cold basin
    // search once.  This keeps the warm path fail-closed while avoiding a
    // pathological "shortcut failed -> full qf" latency spike.
    if (!seed_ok && direct_warm) {
        if (prof) ++prof->d14_direct_warm_reject;
        return solve_d14(desc_v, deg, compensated, nullptr, sc,event_pf,event_rmax);
    }

    if (!seed_ok) {
        int qf_iters = 0;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
        bool qf_converged = false;
#endif
        auto qf_begin = V2Clock::now();
        out.roots =
            use_struct
                ? aberth_d14_struct<qf>(scqf, desc, 400, nullptr, (qf)1e-22,
                                        &qf_iters
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
                                        , &qf_converged
#endif
                                        )
                : aberth<qf>(desc, deg, 400, nullptr, (qf)1e-22, nullptr,
                             &qf_iters);
        if (prof) {
            ++prof->d14_qf_cold_calls;
            prof->d14_qf_cold_sweeps += (V2Profile::u64)qf_iters;
            const auto qf_end = V2Clock::now();
            v2_profile_add_ms(&V2Profile::d14_qf_ms, qf_begin, qf_end);
            v2_profile_add_ms(&V2Profile::d14_qf_polish_ms, qf_begin, qf_end);
        }
        out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
        out.tier = (out.tier == 0 && !compensated) ? -1 : 2;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
        if (re_detail::d14_qf_trace_enabled()) {
            int finite_roots = 0;
            for (const auto& root : out.roots)
                if (qfinite_(root.re) && qfinite_(root.im)) ++finite_roots;
            qf certificate_residual = 0;
            int certificate_reason = 0;
            const bool scalar_certificate = d14_scalar_certificate(
                use_struct ? &scqf : nullptr, desc_v, out.roots,
                &certificate_residual, &certificate_reason);
            std::fprintf(stderr,
                "D14QF_STAGE\tkind=cold\tseed_source=cauchy\treason=%s\tconverged=%d"
                "\tfinite_roots=%d\titers=%d\tresidual=",
                seed_reject_reason, int(qf_converged), finite_roots, qf_iters);
            re_detail::d14_qf_trace_value("worst", out.worst_res);
            std::fprintf(stderr, "\tscalar_certificate=%d\tcertificate_reason=%d",
                         int(scalar_certificate), certificate_reason);
            re_detail::d14_qf_trace_value("certificate_residual",
                                          certificate_residual);
            std::fputc('\n', stderr);
        }
#endif
    }

    // Completeness sanity check (separate from the per-root residual gate,
    // which bounds backward error but not root *count*).  Newton's first
    // identity: sum of the deg roots == -c[1]/c[0].  A dropped or doubled
    // basin -- the way an under-resolved Aberth actually fails -- shifts
    // the power sum well outside rounding.  The residual is scale-free;
    // the roots are O(1..10) so an absolute 1e-6 slack on a deg-14 sum is
    // ~1e5 x the honest error.  On failure redo cold (once).
    auto validation_begin = prof ? V2Clock::now() : V2Clock::time_point{};
    bool completeness_failure = false;
    const auto completeness_begin = prof ? V2Clock::now() : V2Clock::time_point{};
    if (out.tier != 2 && out.tier != -1 && deg >= 1 &&
        (double)fabsq(desc[0]) > 0.0) {
        Cplx<qf> s{(qf)0, (qf)0};
        for (const auto& r : out.roots) s = s + r;
        qf want = -desc[1] / desc[0];
        qf err = fabsq(s.re - want) + fabsq(s.im);
        qf tolsum = (qf)1e-6 * ((qf)1 + fabsq(want));
        completeness_failure = !(err <= tolsum);
    }
    if (prof)
        v2_profile_add_ms(&V2Profile::d14_completeness_check_ms,
                          completeness_begin, V2Clock::now());
    if (completeness_failure) {
        if (prof) ++prof->d14_completeness_fails;
        int qf_iters = 0;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
        bool qf_converged = false;
#endif
        auto qf_begin = V2Clock::now();
        out.roots =
            use_struct
                ? aberth_d14_struct<qf>(scqf, desc, 400, nullptr, (qf)1e-22,
                                        &qf_iters
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
                                        , &qf_converged
#endif
                                        )
                : aberth<qf>(desc, deg, 400, nullptr, (qf)1e-22, nullptr,
                             &qf_iters);
        if (prof) {
            ++prof->d14_qf_cold_calls;
            prof->d14_qf_cold_sweeps += (V2Profile::u64)qf_iters;
            const auto qf_end = V2Clock::now();
            v2_profile_add_ms(&V2Profile::d14_qf_ms, qf_begin, qf_end);
            v2_profile_add_ms(&V2Profile::d14_qf_polish_ms, qf_begin, qf_end);
        }
        out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
        out.tier = 2;
#if defined(HOLO_D14_TRACE_QF_ITERATIONS)
        if (re_detail::d14_qf_trace_enabled()) {
            std::fprintf(stderr,
                "D14QF_STAGE\tkind=cold\treason=completeness_gate\tconverged=%d"
                "\titers=%d\tresidual=",
                int(qf_converged), qf_iters);
            re_detail::d14_qf_trace_value("worst", out.worst_res);
            std::fprintf(stderr, "\n");
        }
#endif
    }
    if (prof) v2_profile_add_ms(&V2Profile::d14_validate_ms, validation_begin,
                                V2Clock::now());

    if (direct_warm && out.tier == 2) {
        if (prof) ++prof->d14_direct_warm_reject;
        return solve_d14(desc_v, deg, compensated, nullptr, sc,event_pf,event_rmax);
    }

    if (prof && direct_warm) {
        if (out.tier == 0 || out.tier == 1)
            ++prof->d14_direct_warm_success;
        else
            ++prof->d14_direct_warm_reject;
    }

    // D14 has real coefficients: its non-real roots are exact conjugate
    // pairs, so a pair's two real parts are mathematically identical.  A
    // finite-precision Aberth leaves them a little apart -- ~1e-11 rel in
    // __float128, ~1e-8 rel in double-double.  Downstream, radial_events
    // dedups the complex-Re soft-boundary list at a fixed gap, so the
    // wider double-double split would post one conjugate pair as *two*
    // soft boundaries.  Snap every near-conjugate pair (clearly complex,
    // conjugate distance < 1e-6 rel) onto its common real part and mean
    // imaginary magnitude.  A no-op for the __float128 paths (the shift is
    // far below the dedup gap); it makes the compensated path's event
    // list identical.
    {
        std::vector<char> done(deg, 0);
        for (int i = 0; i < deg; ++i) {
            if (done[i]) continue;
            qf ai = fabsq(out.roots[i].im);
            qf scale_i = (qf)1 + fabsq(out.roots[i].re);
            if (ai < (qf)1e-9 * scale_i) continue;  // real root: leave it
            int best = -1;
            qf best_d = (qf)1e-6 * scale_i;
            for (int j = i + 1; j < deg; ++j) {
                if (done[j]) continue;
                qf dre = fabsq(out.roots[i].re - out.roots[j].re);
                qf dim = fabsq(out.roots[i].im + out.roots[j].im);
                qf d = dre + dim;
                if (d < best_d) { best_d = d; best = j; }
            }
            if (best < 0) continue;
            int j = best;
            qf re_avg = (qf)0.5 * (out.roots[i].re + out.roots[j].re);
            qf im_mag = (qf)0.5 * (fabsq(out.roots[i].im) + fabsq(out.roots[j].im));
            qf si = out.roots[i].im >= 0 ? (qf)1 : (qf)-1;
            out.roots[i] = Cplx<qf>(re_avg, si * im_mag);
            out.roots[j] = Cplx<qf>(re_avg, -si * im_mag);
            done[i] = done[j] = 1;
        }
    }

    // Per-root consumer evidence is diagnostic only. qf escalation remains
    // an all-root operation and every existing residual/completeness gate is
    // evaluated before this block. A cold qf restart can change root order,
    // so the opt-in report first obtains a minimum-distance bijection and
    // marks close/ambiguous matches explicitly.
    if (prof && prof->capture_d14_root_work) {
        std::array<int, 14> final_to_real{};
        std::array<bool, 14> match_valid{};
        final_to_real.fill(-1);
        if (out.has_d14real_roots && out.roots.size() == 14) {
            if (out.tier == 2)
                final_to_real = d14_diagnostic_root_assignment(
                    out.roots, out.d14real_roots);
            else
                for (int i = 0; i < deg; ++i) final_to_real[i] = i;

            for (int i = 0; i < deg; ++i) {
                const int source = final_to_real[i];
                if (source < 0) continue;
                if (out.tier != 2) {
                    match_valid[i] = true;
                    continue;
                }
                const qf distance = cabs(out.roots[i] - out.d14real_roots[source]);
                qf final_nearest = HUGE_VALQ, source_nearest = HUGE_VALQ;
                for (int j = 0; j < deg; ++j) {
                    if (j != i)
                        final_nearest = std::min(final_nearest,
                                                cabs(out.roots[i] - out.roots[j]));
                    if (j != source)
                        source_nearest = std::min(
                            source_nearest,
                            cabs(out.d14real_roots[source] - out.d14real_roots[j]));
                }
                const qf local_scale = std::min(final_nearest, source_nearest);
                match_valid[i] = finiteq(distance) && finiteq(local_scale) &&
                                 local_scale > 0 && distance <= (qf)0.25 * local_scale;
            }

            const auto source_work = out.root_work;
            for (int i = 0; i < deg; ++i) {
                const int source = final_to_real[i];
                if (source >= 0) out.root_work[i] = source_work[source];
                out.root_work[i].root_index = i;
                out.root_work[i].source_root_index = source;
                out.root_work[i].qf_source_match_valid = match_valid[i];
                if (source >= 0) {
                    out.root_work[i].qf_displacement = static_cast<double>(
                        cabs(out.roots[i] - out.d14real_roots[source]));
                    out.root_work[i].qf_displacement_valid = match_valid[i] &&
                        std::isfinite(out.root_work[i].qf_displacement);
                }
            }
        }
        if (out.roots.size() == deg) {
            std::array<Cplx<qf>, 14> expanded_seed_roots{};
            bool all_seeds_valid = true;
            for (int i = 0; i < deg; ++i) {
                const auto& work = out.root_work[i];
                all_seeds_valid = all_seeds_valid && work.double_seed_valid;
                expanded_seed_roots[i] = Cplx<qf>(
                    static_cast<qf>(work.double_seed_v_re),
                    static_cast<qf>(work.double_seed_v_im));
            }
            if (all_seeds_valid) {
                const auto final_to_seed = d14_diagnostic_root_assignment(
                    out.roots, expanded_seed_roots);
                for (int i = 0; i < deg; ++i) {
                    const int source = final_to_seed[i];
                    if (source < 0) continue;
                    auto& work = out.root_work[i];
                    const qf distance = cabs(
                        out.roots[i] - expanded_seed_roots[source]);
                    qf final_nearest = HUGE_VALQ, source_nearest = HUGE_VALQ;
                    for (int j = 0; j < deg; ++j) {
                        if (j == i)
                            continue;
                        final_nearest = std::min(
                            final_nearest, cabs(out.roots[i] - out.roots[j]));
                    }
                    for (int j = 0; j < deg; ++j) {
                        if (j == source)
                            continue;
                        source_nearest = std::min(
                            source_nearest,
                            cabs(expanded_seed_roots[source] -
                                 expanded_seed_roots[j]));
                    }
                    const qf local_scale = std::min(final_nearest, source_nearest);
                    work.expanded_seed_source_index =
                        out.root_work[source].double_seed_index;
                    work.expanded_seed_displacement = static_cast<double>(distance);
                    work.expanded_seed_match_valid = finiteq(distance) &&
                        finiteq(local_scale) && local_scale > 0 &&
                        distance <= (qf)0.25 * local_scale;
                }
            }
        }
        for (int i = 0; i < deg && i < static_cast<int>(out.roots.size()); ++i) {
            auto& work = out.root_work[i];
            work.root_index = i;
            work.final_v_re = static_cast<double>(out.roots[i].re);
            work.final_v_im = static_cast<double>(out.roots[i].im);
            work.qf_escalated = out.tier > 0;
            Cplx<qf> p, dp;
            if (use_struct) d14_struct_eval(scqf, out.roots[i], p, dp);
            else {
                p = poly_eval_c(desc, deg, out.roots[i]);
                dp = polyder_eval_c(desc, deg, out.roots[i]);
            }
            const qf dp_abs = cabs(dp);
            if (finiteq(dp_abs) && dp_abs > 0 && finiteq(cabs(p))) {
                const qf correction = cabs(p / dp);
                work.qf_newton_correction = static_cast<double>(correction);
                if (!std::isfinite(work.qf_newton_correction))
                    work.qf_newton_correction = -1.0;
            } else {
                work.qf_newton_correction = -1.0;
            }
            const Cplx<qf> expanded_value =
                poly_eval_c(desc, deg, out.roots[i]);
            const qf expanded_residual = cabs(expanded_value) /
                (dscale + (qf)1e-300);
            work.qf_expanded_relative_residual = finiteq(expanded_residual)
                ? static_cast<double>(expanded_residual) : -1.0;
            qf nearest = HUGE_VALQ;
            for (int j = 0; j < deg && j < static_cast<int>(out.roots.size()); ++j)
                if (i != j)
                    nearest = std::min(nearest, cabs(out.roots[i] - out.roots[j]));
            work.qf_nearest_separation = finiteq(nearest)
                ? static_cast<double>(nearest) : -1.0;
            const double projected_r = std::sqrt(std::max(0.0, work.final_v_re));
            work.qf_position_error = projected_r > 0.0 &&
                                     work.qf_newton_correction >= 0.0
                ? work.qf_newton_correction / (2.0 * projected_r) : -1.0;
            if (!out.has_d14real_roots) {
                work.v_re = work.final_v_re;
                work.v_im = work.final_v_im;
                work.source_root_index = -1;
            }
        }
    }

    // Diagnostic only: count suspiciously close root pairs after the final
    // conjugacy snap.  This is deliberately outside all solver decisions;
    // it identifies the clusters that drive the DD/qf ladder without adding
    // another heuristic to the production root path.
    if (prof) {
        if ((int)out.roots.size() != deg) ++prof->d14_root_count_bad;
        for (size_t i = 0; i < out.roots.size(); ++i) {
            const auto& a = out.roots[i];
            if (fabsq(a.im) <= (qf)1e-10 * ((qf)1 + fabsq(a.re))) continue;
            bool found = false;
            for (size_t j = 0; j < out.roots.size(); ++j) {
                if (i == j) continue;
                const auto& b = out.roots[j];
                const qf e = fabsq(a.re - b.re) + fabsq(a.im + b.im);
                if (e <= (qf)1e-7 * ((qf)1 + fabsq(a.re) + fabsq(a.im))) {
                    found = true;
                    break;
                }
            }
            if (!found) ++prof->d14_conjugacy_bad;
        }
        if ((int)out.roots.size() == deg && desc[0] != 0) {
            for (int power = 1; power <= 3; ++power) {
                Cplx<qf> sum(0, 0);
                for (const auto& r : out.roots) {
                    Cplx<qf> x(1, 0);
                    for (int k = 0; k < power; ++k) x = x * r;
                    sum = sum + x;
                }
                const qf c1 = desc[1] / desc[0];
                const qf c2 = desc[2] / desc[0];
                const qf c3 = desc[3] / desc[0];
                const qf want = power == 1 ? -c1
                                  : power == 2 ? c1 * c1 - 2 * c2
                                               : -c1 * c1 * c1 + 3 * c1 * c2 - 3 * c3;
                if (!(cabs(sum - Cplx<qf>(want, 0)) <=
                      (qf)2e-8 * ((qf)1 + fabsq(want))))
                    ++prof->d14_vieta_bad;
            }
        }
        for (int i = 0; i < deg; ++i) {
            for (int j = i + 1; j < deg; ++j) {
                const qf d = fabsq(out.roots[i].re - out.roots[j].re) +
                             fabsq(out.roots[i].im - out.roots[j].im);
                const qf sc = (qf)1 + fabsq(out.roots[i].re) +
                              fabsq(out.roots[i].im) + fabsq(out.roots[j].re) +
                              fabsq(out.roots[j].im);
                if (d <= (qf)1e-8 * sc) ++prof->d14_root_clusters;
            }
        }
    }

    // The Horner candidate is allowed to reach event classification only
    // after validation through the exact structural evaluator.  A failure is
    // a hard qf escalation; it never returns an unchecked root set.
    if (holo_d14_horner_enabled() && sc && deg == 14) {
        qf cert_res = 0;
        if (!d14_scalar_certificate(sc, desc_v, out.roots, &cert_res)) {
            int qf_iters = 0;
            auto qf_begin = V2Clock::now();
            out.roots = aberth_d14_struct<qf>(*sc, desc, 400, nullptr,
                                              (qf)1e-22, &qf_iters);
            out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
            out.tier = 2;
            if (prof) {
                ++prof->d14_qf_cold_calls;
                prof->d14_qf_cold_sweeps += (V2Profile::u64)qf_iters;
                const auto qf_end = V2Clock::now();
                v2_profile_add_ms(&V2Profile::d14_qf_ms, qf_begin, qf_end);
                v2_profile_add_ms(&V2Profile::d14_qf_polish_ms, qf_begin, qf_end);
            }
        }
    }
    return out;
}

}  // namespace re_detail

// Port of radial_events.radial_events.  Returns events sorted by radius and
// merged within `merge_tol` per kind; `r_max` via the same formula.
// `d14_warm` (optional, Phase B2): a previous epoch's full D14 root set
// (v = R^2 space, all 14 incl. complex) used to warm-seed the solve.
// `d14_roots_out` (optional): receives this epoch's full post-symmetrised
// D14 root set for the next epoch / a prepared-geometry cache (Phase E).
// Explicit isolated research scope; fixed-n_r callers never consult this policy.
enum class D14EventPolicy { AllComplexSoft, NoProjectedComplexSoft, PositiveReal };
inline thread_local D14EventPolicy d14_event_policy = D14EventPolicy::AllComplexSoft;
struct D14EventPolicyScope {
    D14EventPolicy previous;
    explicit D14EventPolicyScope(D14EventPolicy p) : previous(d14_event_policy) { d14_event_policy = p; }
    ~D14EventPolicyScope() { d14_event_policy = previous; }
    D14EventPolicyScope(const D14EventPolicyScope&) = delete;
};

inline std::vector<RadialEvent> radial_events(
    const PrimaryFrame& pf, double* r_max_out, double merge_tol = 1e-7,
    const std::vector<Cplx<__float128>>* d14_warm = nullptr,
    std::vector<Cplx<__float128>>* d14_roots_out = nullptr,
    bool retain_adaptive_metadata = false,
    const PositiveD14Cache* positive_cache = nullptr,
    PositiveD14Result* positive_out = nullptr) {
    using namespace re_detail;
    V2Profile* prof = v2_profile_current();
    if (prof) ++prof->radial_event_calls;
    V2ProfileTimer radial_timer(&V2Profile::radial_events_ms);
    const double a = pf.a, m0 = pf.m0, X = pf.X, Y = pf.Y, rho = pf.rho;
    const double W = std::hypot(X, Y) + rho;
    const double Rmax = 0.5 * (a + W + std::hypot(a - W, 2.0)) + 1e-12;
    if (r_max_out) *r_max_out = Rmax;

    std::vector<RadialEvent> ev;

    PositiveD14Result positive;
    bool certified = false;
    if (retain_adaptive_metadata && d14_event_policy == D14EventPolicy::PositiveReal) {
        positive = positive_d14_roots(pf, Rmax, positive_cache);
        certified = positive.assurance == PositiveRootAssurance::PositiveRealCertified;
        if (certified) {
            positive_detail::StageTimer classification_timer{&positive.stats.event_classification_ms};
            double previous = -1;
            for (unsigned i=0;i<positive.root_count;++i) {
                const auto& bracket=positive.roots[i];
                if(bracket.v_lo >= (qf)Rmax*Rmax) continue;
                if(bracket.v_hi >= (qf)Rmax*Rmax || bracket.v_lo<=0) { certified=false; break; }
                qf low=nextafterq(sqrtq(bracket.v_lo),-HUGE_VALQ);
                qf high=nextafterq(sqrtq(bracket.v_hi),HUGE_VALQ);
                qf Rq=(low+high)/2;double R=(double)Rq;
                if(!(R>previous)) {certified=false;break;} previous=R;
                const auto probe_begin = prof ? V2Clock::now() : V2Clock::time_point{};
                auto probe=probe_double_root(R,pf);
                if (prof) v2_profile_add_ms(&V2Profile::d14_event_classify_ms,
                                            probe_begin, V2Clock::now());
                RadialEvent event{R,probe.physically_real?"physical_real":"physical_complex",
                    probe.physically_real,"D14 certified positive real root"};
                event.radius_lo=(double)(Rq-(qf)R);
                event.radius_uncertainty=std::nextafter((double)((high-low)/2),INFINITY);
                event.precision_tier=positive.stats.chain_tier;
                event.fold_t_seed=probe.stationary_t;event.fold_t_seed_valid=probe.stationary_valid;
                event.positive_certified=true;event.positive_root_id=(int)i;
                event.certified_radius_lo=low;event.certified_radius_hi=high;
                ev.push_back(event);
            }
            if(!certified){ev.clear();positive.stats.reason="RepresentationLimited";}
        }
        if(!certified){positive.assurance=PositiveRootAssurance::LegacyValidated;
            ++positive.stats.legacy_backend_calls;}
        if(positive_out)*positive_out=positive;
        if(certified && d14_roots_out)d14_roots_out->clear();
    }
    // The structured D14 block and exact chart_p4 factorization do not need
    // the generic polynomial family.  Construct it only for the explicitly
    // requested legacy lanes; this removes a duplicate generic-series build
    // from the default production route without changing either certificate.
    const bool need_poly_family = !holo_d14_struct_enabled() ||
                                  !holo_chart_p4_factor_enabled();
    PolyFamilyR fam{};
    if (need_poly_family) {
        auto coeff_begin = V2Clock::now();
        fam = p_coeffs_in_R((qf)a, (qf)m0, (qf)X, (qf)Y, (qf)rho);
        if (prof) v2_profile_add_ms(&V2Profile::d14_coeff_ms, coeff_begin,
                                    V2Clock::now());
    }
    if (!certified) {
    // D14 low-degree block form (memo sec 2): exact, cancellation-free.
    // Used for the polish evaluator and -- unless HOLO_D14_STRUCT_LEGACY=1
    // -- as the source of the expanded coefficient vector too.
    auto struct_begin = V2Clock::now();
    D14StructQf d14s =
        d14_struct_build((qf)a, (qf)m0, (qf)X, (qf)Y, (qf)rho);
    if (prof) v2_profile_add_ms(&V2Profile::d14_struct_build_ms, struct_begin,
                                V2Clock::now());
    std::vector<qf> d14;
    if (holo_d14_struct_enabled()) {
        auto expand_begin = V2Clock::now();
        d14 = d14_expanded_from_struct(d14s);
        if (prof) v2_profile_add_ms(&V2Profile::d14_expand_ms, expand_begin,
                                    V2Clock::now());
    } else {
        auto expand_begin = V2Clock::now();
        d14 = d14_coeffs(fam);  // ascending in v
        if (prof) v2_profile_add_ms(&V2Profile::d14_expand_ms, expand_begin,
                                    V2Clock::now());
    }
    if (!d14.empty()) {
        int deg = (int)d14.size() - 1;
        std::vector<qf> desc(deg + 1);
        for (int i = 0; i <= deg; ++i) desc[i] = d14[deg - i];

        // Multi-tier solve.  The D14 monomial basis is ill-conditioned
        // (kappa ~ 1e9), so the *polish* must exceed `double` -- but the
        // *search* need not.  A balanced `double` Aberth-Ehrlich pass
        // locates every root basin; a compensated double-double polish
        // (default) or an __float128 polish (HOLO_D14_LEGACY_SOLVE=1)
        // warm-started from those guesses then converges in a handful of
        // iterations.  A failed __float128 residual gate -> cold
        // __float128 solve.  See re_detail::solve_d14.
        D14Solve sol = solve_d14(desc, deg, holo_d14_compensated_enabled(),
                                 (d14_warm && (int)d14_warm->size() == deg)
                                     ? d14_warm
                                     : nullptr,
                                 &d14s
#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
                                 , &pf, Rmax
#endif
                                 );
#if defined(HOLO_D14_EVENT_CONTRACT_RESEARCH)
        if (d14_event_contract_capture) {
            d14_event_contract_capture->candidates =
                sol.event_contract_candidates;
            d14_event_contract_capture->oracle_roots = sol.roots;
            d14_event_contract_capture->oracle_tier = sol.tier;
            d14_event_contract_capture->r_max = Rmax;
        }
#endif
        if (prof) {
            if (sol.warm_seeded) ++prof->d14_warm_seeded;
        }
        std::vector<Cplx<qf>>& roots = sol.roots;
        if (d14_roots_out) *d14_roots_out = roots;
        // The fixed-resolution V2 path needs the same event radii and
        // physical classification as before.  Adaptive callers opt in to the
        // extra qf radius split, stationary-root seed, and D/D' metadata;
        // keeping that work optional prevents the adaptive hand-off from
        // adding cost to fixed-n_r production consumers of classify_cells().
        if (!retain_adaptive_metadata) {
            auto rv = positive_real_roots(roots, 1e-8, 1e-9);
            for (double v : rv) {
                const double R = std::sqrt(v);
                if (!(R > 0.0 && R < Rmax)) continue;
                const auto probe_begin = prof ? V2Clock::now() : V2Clock::time_point{};
                const bool is_real = double_root_is_real(R, pf);
                if (prof) v2_profile_add_ms(&V2Profile::d14_event_classify_ms,
                                            probe_begin, V2Clock::now());
                ev.push_back({R, is_real ? "physical_real" : "physical_complex",
                              is_real, "D14 real root"});
            }
        } else {
            // Keep the qf D14 root through the event hand-off.  Casting to a
            // double before forming R loses precisely the small event
            // remainder needed by adaptive near-fold maps.  The stationary-
            // root probe is still the physical/soft classifier, but its t
            // seed is retained.
            for (std::size_t root_index = 0; root_index < roots.size(); ++root_index) {
                const auto& z = roots[root_index];
                const qf av = fabsq(z.re), ai = fabsq(z.im);
                if (!(z.re > 0) || ai > (qf)1e-8 * (qf(1) + av)) continue;
                const qf vq = z.re;
                const qf Rq = sqrtq(vq);
                const double R = (double)Rq;
                if (!(R > 0.0 && R < Rmax)) continue;
                const auto probe_begin = prof ? V2Clock::now() : V2Clock::time_point{};
                const PhysicalRootProbe probe = probe_double_root(R, pf);
                if (prof) v2_profile_add_ms(&V2Profile::d14_event_classify_ms,
                                            probe_begin, V2Clock::now());
                if (prof && prof->capture_d14_root_work &&
                    root_index < sol.root_work.size()) {
                    auto& work = sol.root_work[root_index];
                    work.role = static_cast<int>(D14RootUse::PositiveRealCandidate);
                    work.physical_real = probe.physically_real ? 1 : 0;
                }
                RadialEvent event{R,
                                  probe.physically_real ? "physical_real"
                                                         : "physical_complex",
                                  probe.physically_real, "D14 real root"};
                event.radius_lo = (double)(Rq - (qf)R);
                event.precision_tier = 2;
                if (probe.stationary_valid) {
                    event.fold_t_seed = probe.stationary_t;
                    event.fold_t_seed_valid =
                        std::isfinite(probe.stationary_t);
                }
                // A simple D14 root supplies a cheap radius-conditioning
                // estimate. It is only metadata: the adaptive consumer still
                // verifies the coupled P=P_t equations before accepting it.
                Cplx<qf> D{}, Dp{};
                d14_struct_eval(d14s, Cplx<qf>(vq, qf(0)), D, Dp);
                const qf adp = fabsq(Dp.re), ad = fabsq(D.re);
                if (adp > 0 && finiteq(adp) && finiteq(ad)) {
                    const qf vshift = ad / adp;
                    event.d14_condition = (double)(adp / (qf(1) + ad));
                    event.radius_uncertainty =
                        (double)(vshift / (qf(2) * Rq));
                }
                ev.push_back(event);
            }
        }
        // complex roots (Re v > 0) -> soft boundaries
        const auto soft_begin = prof ? V2Clock::now() : V2Clock::time_point{};
        std::vector<double> cv;
        for (std::size_t root_index = 0; root_index < roots.size(); ++root_index) {
            const auto& r = roots[root_index];
            if (retain_adaptive_metadata && d14_event_policy != D14EventPolicy::AllComplexSoft) break;
            double re = (double)r.re, im = (double)r.im;
            if (re > 0.0 && std::fabs(im) >= 1e-8 * (1.0 + std::fabs(re))) {
                cv.push_back(re);
                if (prof && prof->capture_d14_root_work &&
                    retain_adaptive_metadata && d14_event_policy == D14EventPolicy::AllComplexSoft &&
                    std::sqrt(re) < Rmax && root_index < sol.root_work.size())
                    sol.root_work[root_index].role =
                        static_cast<int>(D14RootUse::ComplexSoftCut);
            }
        }
        std::sort(cv.begin(), cv.end());
        std::vector<double> cvd;
        for (double x : cv)
            if (cvd.empty() || x - cvd.back() > 1e-9) cvd.push_back(x);
        for (double v : cvd) {
            double R = std::sqrt(v);
            if (R > 0.0 && R < Rmax)
                ev.push_back({R, "physical_complex", false,
                              "D14 complex root (Re v)"});
        }
        if (prof) v2_profile_add_ms(&V2Profile::d14_soft_event_ms,
                                    soft_begin, V2Clock::now());
        if (prof && prof->capture_d14_root_work && retain_adaptive_metadata) {
            const std::size_t count = std::min<std::size_t>(
                roots.size(), sol.root_work.size());
            prof->d14_root_work.insert(prof->d14_root_work.end(),
                                       sol.root_work.begin(),
                                       sol.root_work.begin() + count);
        }
    }

    } // incumbent backend only when positive isolation was not certified

    if (a > 0.0 && a < Rmax)
        ev.push_back({a, "R_eq_a", false, "P<->B factor clash"});
    const double rm0 = std::sqrt(m0);
    if (rm0 > 0.0 && rm0 < Rmax)
        ev.push_back({rm0, "R_eq_sqrt_m0", false, "P acquires factor A"});

    if (std::fabs(Y) < 1e-14) {
        // L(v) = (a-X) v^2 + (-(a-X) a^2 - a) v + a^3 m0
        double c2 = a - X;
        double c1 = -(a - X) * a * a - a;
        double c0 = a * a * a * m0;
        std::array<double, 5> lp{c0, c1, c2, 0.0, 0.0};
        double desc[3] = {c2, c1, c0};
        int ld = 2;
        while (ld > 0 && desc[0] == 0.0) {
            for (int i = 0; i < ld; ++i) desc[i] = desc[i + 1];
            --ld;
        }
        if (ld > 0) {
            auto z = aberth<double>(desc, ld, 80);
            auto rv = positive_real_roots(z, 1e-8, 1e-9);
            for (double v : rv) {
                double R = std::sqrt(v);
                if (R > 0.0 && R < Rmax)
                    ev.push_back({R, "L_root", false, "Res(P,B) axis root"});
            }
        }
        (void)lp;
    }

    // chart_p4 : p4(R) = 0.  The factored cubic route is independent of the
    // D14 parameter regime; the old degree-6 Aberth solve remains the A/B
    // oracle under HOLO_CHART_P4_LEGACY=1.
    {
        std::vector<double> rv;
        if (holo_chart_p4_factor_enabled()) {
            rv = chart_p4_factor_roots(pf);
        } else {
            int deg = fam.p[4].deg;
            while (deg > 0 && fabsq(fam.p[4].c[deg]) == 0) --deg;
            if (deg > 0) {
                std::vector<double> desc(deg + 1);
                for (int i = 0; i <= deg; ++i)
                    desc[i] = (double)fam.p[4].c[deg - i];
                auto z = aberth<double>(desc.data(), deg, 200);
                rv = positive_real_roots(z, 1e-8, 1e-9);
            }
        }
        for (double R : rv)
            if (R > 0.0 && R < Rmax)
                ev.push_back({R, "chart_p4", false,
                              "p4(R)=0 : boundary point at theta=pi"});
    }

    std::sort(ev.begin(), ev.end(),
              [](const RadialEvent& x, const RadialEvent& y) {
                  return x.radius < y.radius;
              });
    std::vector<RadialEvent> merged;
    for (const auto& e : ev) {
        // The fixed-resolution route keeps its historical same-kind merge
        // policy.  Adaptive topology, however, needs every distinct
        // physical fold: two physical D14 roots can be closer than the old
        // absolute merge tolerance while still enclosing a real, thin
        // four-crossing band.  Collapsing that pair makes a supposedly open
        // cell contain a topology change and causes the adaptive evaluator to
        // fail closed at the first node in the band.  The D14 root solve and
        // downstream quartic probe remain the certificates; retaining the
        // roots here does not introduce a sampling heuristic.
        const bool preserve_adaptive_physical_pair =
            retain_adaptive_metadata &&
            merged.size() > 0 &&
            merged.back().kind == "physical_real" &&
            e.kind == "physical_real";
        if (!e.positive_certified && !preserve_adaptive_physical_pair && !merged.empty() && !merged.back().positive_certified &&
            merged.back().kind == e.kind &&
            std::fabs(e.radius - merged.back().radius) < merge_tol)
            continue;
        merged.push_back(e);
    }
    if (prof) {
        for (const auto& e : merged) {
            if (e.kind == "physical_real") ++prof->physical_real_events;
            else if (e.kind == "physical_complex") {
                ++prof->physical_complex_events;
                if (e.detail.find("complex") != std::string::npos)
                    ++prof->d14_soft_events;
            } else if (e.kind == "chart_p4") ++prof->chart_p4_events;
            else if (e.kind == "R_eq_a" || e.kind == "R_eq_sqrt_m0")
                ++prof->radial_eq_events;
            else if (e.kind == "L_root") ++prof->l_root_events;
            else ++prof->representation_events;
        }
    }
    return merged;
}

}  // namespace lcbinint::holonomic
