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
#include <cstdlib>
#include <string>
#include <vector>

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/d14_structure.hpp"
#include "lcbinint/magnification/holonomic/dd_real.hpp"
#include "lcbinint/magnification/holonomic/fp_env.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/poly_roots.hpp"

namespace lcbinint::holonomic {

struct RadialEvent {
    double radius;
    std::string kind;
    bool physically_real;
    std::string detail;
};

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

// classify: at a discriminant zero P has a double root t*; real?
// Port of radial_events._double_root_is_real.
inline bool double_root_is_real(double R, const PrimaryFrame& pf,
                                double tol = 1e-6) {
    QuarticCoeffs q = boundary_quartic(R, pf);  // ascending
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
    if (dd <= 0) return false;
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
    return best_res < tol &&
           std::fabs(best_im) < 1e-4 * (std::fabs(best_re) + 1.0);
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

// Result of the multi-tier D14 root solve.
struct D14Solve {
    std::vector<Cplx<qf>> roots;
    qf worst_res = 0;
    int tier = 0;  // 0 dd-sufficed, 1 qf-warm escalation, 2 cold quad,
                   // 3 legacy qf-warm, -1 cold (seed non-finite)
    bool warm_seeded = false;  // the double presearch was seeded by a
                               // previous epoch's root set (Phase B2)
};

// worst relative residual of `roots` against the degree-`deg` descending
// __float128 polynomial `desc`.
inline qf d14_worst_res(const qf* desc, int deg,
                        const std::vector<Cplx<qf>>& roots, qf dscale) {
    qf worst = 0;
    for (const auto& r : roots) {
        qf res = cabs(poly_eval_c(desc, deg, r)) / (dscale + (qf)1e-300);
        if (res > worst) worst = res;
    }
    return worst;
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
inline D14Solve solve_d14(const std::vector<qf>& desc_v, int deg,
                          bool compensated,
                          const std::vector<Cplx<qf>>* warm_seed = nullptr,
                          const D14StructQf* sc = nullptr) {
    const qf* desc = desc_v.data();
    D14Solve out;

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
    auto zd = presearch_seed.empty()
                  ? aberth<double>(descd.data(), deg, 200)
                  : aberth<double>(descd.data(), deg, 60, presearch_seed.data());
    out.warm_seeded = !presearch_seed.empty();
    bool seed_ok = true;
    for (int i = 0; i < deg; ++i)
        if (!std::isfinite(zd[i].re) || !std::isfinite(zd[i].im)) {
            seed_ok = false;
            break;
        }

    // Block-form D/D' polish (memo section 3): exact C3/G4/Z3 evaluation
    // in place of Horner on the cancellation-carrying expanded vector.
    const bool use_struct = sc && holo_d14_struct_enabled() && deg == 14;
    D14StructC<DD> scdd;
    D14StructQf scqf;
    if (use_struct) {
        scdd = d14_struct_cast<DD>(*sc);
        scqf = *sc;
    }

    if (seed_ok && compensated) {
        // FTZ/DAZ off so the error-free transforms keep their lo limbs.
        ScopedNoFlushDenormals _eft;
        std::vector<DD> descdd(deg + 1);
        for (int i = 0; i <= deg; ++i) descdd[i] = dd_from_qf(desc[i]);
        std::vector<Cplx<DD>> seed(deg);
        for (int i = 0; i < deg; ++i)
            seed[i] = Cplx<DD>(DD((double)zd[i].re * sscale),
                               DD((double)zd[i].im * sscale));
        // The balanced double presearch seeds every basin to ~1e-13 rel;
        // dd Aberth is locally cubic, so ~3 sweeps reach the ~1e-30 dd
        // floor.  tol 1e-26 is inside that floor (a step norm below it for
        // O(10) roots is dd round-off); 25 sweeps is headroom for a
        // poorly-seeded root (the residual gate escalates it if not).
        auto zdd =
            use_struct
                ? aberth_d14_struct<DD>(scdd, nullptr, 25, seed.data(), DD(1e-26))
                : aberth<DD>(descdd.data(), deg, 25, seed.data(), DD(1e-26));
        out.roots.resize(deg);
        for (int i = 0; i < deg; ++i)
            out.roots[i] = Cplx<qf>(qf_from_dd(zdd[i].re), qf_from_dd(zdd[i].im));
        out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
        out.tier = 0;
        // dd normally reaches ~1e-15 rel residual; a worse result signals a
        // near-multiple cluster past ~106 bits -> escalate that solve to an
        // __float128 warm polish seeded by the dd roots.
        if (!(out.worst_res <= (qf)1e-13)) {
            out.roots =
                use_struct
                    ? aberth_d14_struct<qf>(scqf, nullptr, 24, out.roots.data(),
                                            (qf)1e-20)
                    : aberth<qf>(desc, deg, 24, out.roots.data(), (qf)1e-20);
            out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
            out.tier = 1;
            if (!(out.worst_res <= (qf)1e-12)) seed_ok = false;
        }
    } else if (seed_ok) {
        std::vector<Cplx<qf>> seed(deg);
        for (int i = 0; i < deg; ++i)
            seed[i] = Cplx<qf>((qf)zd[i].re * (qf)sscale,
                               (qf)zd[i].im * (qf)sscale);
        out.roots =
            use_struct
                ? aberth_d14_struct<qf>(scqf, nullptr, 24, seed.data(), (qf)1e-20)
                : aberth<qf>(desc, deg, 24, seed.data(), (qf)1e-20);
        out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
        out.tier = 3;
        if (!(out.worst_res <= (qf)1e-12)) seed_ok = false;
    }

    if (!seed_ok) {
        out.roots =
            use_struct
                ? aberth_d14_struct<qf>(scqf, desc, 400, nullptr, (qf)1e-22)
                : aberth<qf>(desc, deg, 400, nullptr, (qf)1e-22);
        out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
        out.tier = (out.tier == 0 && !compensated) ? -1 : 2;
    }

    // Completeness sanity check (separate from the per-root residual gate,
    // which bounds backward error but not root *count*).  Newton's first
    // identity: sum of the deg roots == -c[1]/c[0].  A dropped or doubled
    // basin -- the way an under-resolved Aberth actually fails -- shifts
    // the power sum well outside rounding.  The residual is scale-free;
    // the roots are O(1..10) so an absolute 1e-6 slack on a deg-14 sum is
    // ~1e5 x the honest error.  On failure redo cold (once).
    if (out.tier != 2 && out.tier != -1 && deg >= 1 &&
        (double)fabsq(desc[0]) > 0.0) {
        Cplx<qf> s{(qf)0, (qf)0};
        for (const auto& r : out.roots) s = s + r;
        qf want = -desc[1] / desc[0];
        qf err = fabsq(s.re - want) + fabsq(s.im);
        qf tolsum = (qf)1e-6 * ((qf)1 + fabsq(want));
        if (!(err <= tolsum)) {
            out.roots =
                use_struct
                    ? aberth_d14_struct<qf>(scqf, desc, 400, nullptr, (qf)1e-22)
                    : aberth<qf>(desc, deg, 400, nullptr, (qf)1e-22);
            out.worst_res = d14_worst_res(desc, deg, out.roots, dscale);
            out.tier = 2;
        }
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
    return out;
}

}  // namespace re_detail

// Port of radial_events.radial_events.  Returns events sorted by radius and
// merged within `merge_tol` per kind; `r_max` via the same formula.
// `d14_warm` (optional, Phase B2): a previous epoch's full D14 root set
// (v = R^2 space, all 14 incl. complex) used to warm-seed the solve.
// `d14_roots_out` (optional): receives this epoch's full post-symmetrised
// D14 root set for the next epoch / a prepared-geometry cache (Phase E).
inline std::vector<RadialEvent> radial_events(
    const PrimaryFrame& pf, double* r_max_out, double merge_tol = 1e-7,
    const std::vector<Cplx<__float128>>* d14_warm = nullptr,
    std::vector<Cplx<__float128>>* d14_roots_out = nullptr) {
    using namespace re_detail;
    const double a = pf.a, m0 = pf.m0, X = pf.X, Y = pf.Y, rho = pf.rho;
    const double W = std::hypot(X, Y) + rho;
    const double Rmax = 0.5 * (a + W + std::hypot(a - W, 2.0)) + 1e-12;
    if (r_max_out) *r_max_out = Rmax;

    std::vector<RadialEvent> ev;

    PolyFamilyR fam = p_coeffs_in_R((qf)a, (qf)m0, (qf)X, (qf)Y, (qf)rho);
    // D14 low-degree block form (memo sec 2): exact, cancellation-free.
    // Used for the polish evaluator and -- unless HOLO_D14_STRUCT_LEGACY=1
    // -- as the source of the expanded coefficient vector too.
    D14StructQf d14s =
        d14_struct_build((qf)a, (qf)m0, (qf)X, (qf)Y, (qf)rho);
    std::vector<qf> d14 = holo_d14_struct_enabled()
                              ? d14_expanded_from_struct(d14s)
                              : d14_coeffs(fam);  // ascending in v
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
                                 &d14s);
        std::vector<Cplx<qf>>& roots = sol.roots;
        if (d14_roots_out) *d14_roots_out = roots;
        auto rv = positive_real_roots(roots, 1e-8, 1e-9);
        for (double v : rv) {
            double R = std::sqrt(v);
            if (!(R > 0.0 && R < Rmax)) continue;
            bool is_real = double_root_is_real(R, pf);
            ev.push_back({R, is_real ? "physical_real" : "physical_complex",
                          is_real, "D14 real root"});
        }
        // complex roots (Re v > 0) -> soft boundaries
        std::vector<double> cv;
        for (const auto& r : roots) {
            double re = (double)r.re, im = (double)r.im;
            if (re > 0.0 && std::fabs(im) >= 1e-8 * (1.0 + std::fabs(re)))
                cv.push_back(re);
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
    }

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

    // chart_p4 : p4(R) = 0
    {
        int deg = fam.p[4].deg;
        while (deg > 0 && fabsq(fam.p[4].c[deg]) == 0) --deg;
        if (deg > 0) {
            std::vector<double> desc(deg + 1);
            for (int i = 0; i <= deg; ++i) desc[i] = (double)fam.p[4].c[deg - i];
            auto z = aberth<double>(desc.data(), deg, 200);
            auto rv = positive_real_roots(z, 1e-8, 1e-9);
            for (double R : rv)
                if (R > 0.0 && R < Rmax)
                    ev.push_back({R, "chart_p4", false,
                                  "p4(R)=0 : boundary point at theta=pi"});
        }
    }

    std::sort(ev.begin(), ev.end(),
              [](const RadialEvent& x, const RadialEvent& y) {
                  return x.radius < y.radius;
              });
    std::vector<RadialEvent> merged;
    for (const auto& e : ev) {
        if (!merged.empty() && merged.back().kind == e.kind &&
            std::fabs(e.radius - merged.back().radius) < merge_tol)
            continue;
        merged.push_back(e);
    }
    return merged;
}

}  // namespace lcbinint::holonomic
