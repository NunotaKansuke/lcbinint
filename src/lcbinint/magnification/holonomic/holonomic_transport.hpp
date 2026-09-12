#pragma once

// ATPT holonomic solver -- regularized flux transport (rescue spikes a/b/c).
//
// Replaces the per-arc 64-point Gauss-Chebyshev-1 angular sqrt(phi) sweep in
// radius_terms (the F_half value + Jacobian integrand) with the deflated
// x-chart identity
//
//     (2/rho) Phi_arc(R)  =  v * K ,
//     K = (2/rho) sum_{i=1..16} W_i w(t_i),   t_i = m + sqrt(v) x_i
//     w(t) = sqrt(S2) / (A^{3/2} sqrt(B)),
//     S2 = -(d0 + t (d1 + t p4)),  A = 1 + t^2,  B = (R-a)^2 + (R+a)^2 t^2
//
// where (p4, d1, d0) deflate the boundary quartic P(.; R) by its two real
// arc-boundary roots t_-, t_+ -- needs ONLY the symmetric pair
//   m = (t_- + t_+) / 2 ,  v = ((t_+ - t_-) / 2)^2
// and NO root solve (checkpoint sec. 26: J_{1/2} = v K exact; K finite and
// smooth as v -> 0; feasibility GO sec. 28,
// evidence/holonomic/coupled_transport_feasibility.txt).
//
// Nodes are Gauss-Chebyshev of the *2nd* kind (sin^2 weight) -- a different
// rule from angular_rule.hpp's Cheb1 (sqrt(1-x^2) weight).
//
// The analytic Jacobian d/dP_j[v K] below matches phi.hpp's PhiValDP::dP and
// boundary_polynomial.hpp's QuarticParamJac::dp param order (X, Y, rho, m0, a).
//
// Fail-closed: any arc that straddles theta = pi (endpoint |t| > kHoloTMax or
// t_hi <= t_lo), a near-tangency (v <= kHoloVFloor), a negative S2, a
// non-finite result, OR an 8-vs-16-node quadrature disagreement above
// kHoloKRelTol (wide arcs whose endpoints approach theta = pi -- the (m,v)
// x-chart integrand sqrt(S2)/(A^{3/2} sqrt(B)) then develops structure the
// 8-node rule cannot resolve; an isolated reciprocal-chart retry is available
// for A/B, while the production default remains the t-chart)
// returns ok == false and the caller keeps the incumbent angular sweep for
// that arc -- never a silent approximation.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/v2_profile.hpp"

namespace lcbinint::holonomic {

constexpr double kHoloPi = 3.14159265358979323846;
constexpr double kHoloTMax = 12.0;     // max endpoint |t| = |m| + sqrt(v);
                                       // t = tan(theta/2), |t| > 12 is within
                                       // ~0.17 rad of theta = pi where the
                                       // (m, v) chart is unusable; an isolated
                                       // reciprocal retry is available for A/B.
constexpr double kHoloVFloor = 1e-10;  // v <= this: near-tangency, the x-chart
                                       // node derivative dt/dv ~ x/(2 sqrt(v))
                                       // blows up; such arcs are already
                                       // rt.reliable = false via tan_thresh.
constexpr double kHoloKRelTol = 1e-8;  // max |K16 - K8| / |K16|; above this the
                                       // 8-node rule is under-resolving (arc
                                       // endpoint near theta = pi) -> fail
                                       // closed to the angular sweep.  K16's
                                       // own error is then <~ 1e-10.

// Runtime override for tests / the 3-solver benchmark (they toggle variants in
// one process): -1 = follow the environment variable (production), 0 = force
// off, 1 = force on.  Production code never touches this.
inline int& holo_holonomic_transport_override() {
    static int v = -1;
    return v;
}

inline bool holo_holonomic_transport_enabled() {
    const int o = holo_holonomic_transport_override();
    if (o >= 0) return o != 0;
    static const bool on = [] {
        const char* e = std::getenv("HOLO_HOLONOMIC_TRANSPORT");
        return e && e[0] == '1';
    }();
    return on;
}

// Research-only condition-driven chart switch.  The incumbent t-chart is
// unchanged unless an isolated A/B process explicitly enables this flag.
inline bool holo_reciprocal_chart_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_RECIPROCAL_CHART");
        return e && e[0] == '1';
    }();
    return on;
}

// Gauss-Chebyshev, 2nd kind, on (-1, 1):
//   x_k = cos(k pi / (n+1)),  W_k = (pi / (n+1)) sin^2(k pi / (n+1)),  k = 1..n
// exact for  int_{-1}^{1} p(x) sqrt(1 - x^2) dx ,  deg p <= 2n - 1.
// The primary rule is n = 16 (the transported value + Jacobian); an 8-node
// subrule is evaluated alongside it purely as a fail-closed error estimate.
constexpr int kHoloNK = 16;
constexpr int kHoloNKLo = 8;
struct GC2Rule {
    double x[32];
    double w[32];
};
inline void gc2_fill(GC2Rule& t, int n) {
    for (int k = 1; k <= n; ++k) {
        const double a = k * kHoloPi / (n + 1);
        const double s = std::sin(a);
        t.x[k - 1] = std::cos(a);
        t.w[k - 1] = (kHoloPi / (n + 1)) * s * s;
    }
}
inline const GC2Rule& gc2_rule16() {
    static const GC2Rule r = [] { GC2Rule t{}; gc2_fill(t, kHoloNK); return t; }();
    return r;
}
inline const GC2Rule& gc2_rule8() {
    static const GC2Rule r = [] { GC2Rule t{}; gc2_fill(t, kHoloNKLo); return t; }();
    return r;
}
inline const GC2Rule& gc2_rule24() {
    static const GC2Rule r = [] { GC2Rule t{}; gc2_fill(t, 24); return t; }();
    return r;
}

// Research-only three-level gate.  The incumbent 8-vs-16 test remains the
// first certificate.  On its disagreement, 24 nodes are evaluated and the
// 16-node value is accepted only when the independent 16-vs-24 difference is
// within the same tolerance.  Default is off so all existing callers retain
// the exact V2 gate unless the isolated A/B process opts in.
inline bool holo_k_three_gate_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_K_GATE");
        return e && (e[0] == 't' || e[0] == 's');
    }();
    return on;
}

inline bool holo_k_three_gate_strict() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_K_GATE");
        return e && e[0] == 's';
    }();
    return on;
}

enum class KRejectReason {
    none,
    nonfinite,
    vfloor,
    tmax,
    s2,
    b,
    disagreement,
};

inline void profile_k_reject(KRejectReason why) {
    V2Profile* p = v2_profile_current();
    if (!p) return;
    ++p->k_reject;
    switch (why) {
        case KRejectReason::vfloor: ++p->k_reject_vfloor; break;
        case KRejectReason::tmax: ++p->k_reject_tmax; break;
        case KRejectReason::s2: ++p->k_reject_s2; break;
        case KRejectReason::b: ++p->k_reject_b; break;
        case KRejectReason::disagreement: ++p->k_reject_disagreement; break;
        default: ++p->k_reject_nonfinite; break;
    }
}

// Deflate the ascending quartic `pc` by (t - (m-s))(t - (m+s)), s = sqrt(v):
//   pc(t) = (t^2 - 2 m t + (m^2 - v)) (p4 t^2 + d1 t + d0)
// (ports period_reduction._deflated_quadratic with t1,t2 = m -/+ sqrt(v)).
struct DeflatedQuad {
    double p4, d1, d0;
};
inline DeflatedQuad deflate_mv(const std::array<double, 5>& pc, double m,
                               double v) {
    const double p4 = pc[4];
    const double beta = -2.0 * m;               // -(t1 + t2)
    const double d1 = pc[3] - beta * p4;        // pc[3] + 2 m p4
    const double d0 = pc[2] - beta * d1 - (m * m - v) * p4;
    return {p4, d1, d0};
}

// G = sum_i w_i sqrt(S2(t_i)) / (A(t_i)^{3/2} sqrt(B(t_i))),  t_i = m + s x_i,
// over the first `n` nodes of `gc`.  Returns false on S2 < 0 or B <= 0.
inline bool gc2_accum_G(const GC2Rule& gc, int n, double m, double s,
                        const DeflatedQuad& q, double Rma, double Rpa2,
                        double& G, KRejectReason* why = nullptr) {
    G = 0.0;
    for (int i = 0; i < n; ++i) {
        const double t = m + s * gc.x[i];
        const double S2 = -(q.d0 + t * (q.d1 + t * q.p4));
        if (!(S2 >= 0.0)) {
            if (why) *why = KRejectReason::s2;
            return false;
        }
        const double A = 1.0 + t * t;
        const double B = Rma * Rma + Rpa2 * t * t;
        if (!(B > 0.0)) {
            if (why) *why = KRejectReason::b;
            return false;
        }
#if !defined(HOLO_K_DISABLE_SINGLE_SQRT)
        const double ab = A * B;
        const double ratio = S2 / ab;
        // Preserve the original arithmetic at the exponent-range edges.
        if (std::isnormal(ab) && std::isnormal(ratio))
            G += gc.w[i] * std::sqrt(ratio) / A;
        else
#endif
        G += gc.w[i] * std::sqrt(S2) / (A * std::sqrt(A) * std::sqrt(B));
    }
    return true;
}

// 8-vs-16-node relative disagreement of v K for one arc; used as a fail-closed
// quadrature-error estimate.  Returns false (reject) if either subrule hits an
// invalid node or the two disagree by more than kHoloKRelTol.  On success
// `vK` carries the 16-node value.
inline bool k_rule_converged(double m, double v, double s, const DeflatedQuad& q,
                             double Rma, double Rpa2, double two_over_rho,
                             double& vK, KRejectReason* why = nullptr,
                             double* estimated_error = nullptr) {
    if (why) *why = KRejectReason::none;
    double G16 = 0.0, G8 = 0.0;
    if (!gc2_accum_G(gc2_rule16(), kHoloNK, m, s, q, Rma, Rpa2, G16, why)) return false;
    if (!gc2_accum_G(gc2_rule8(), kHoloNKLo, m, s, q, Rma, Rpa2, G8, why)) return false;
    const double vK16 = two_over_rho * v * G16;
    const double vK8 = two_over_rho * v * G8;
    if (!std::isfinite(vK16) || !std::isfinite(vK8)) {
        if (why) *why = KRejectReason::nonfinite;
        return false;
    }
    if (estimated_error) *estimated_error = std::fabs(vK16-vK8);
    const double denom = std::fabs(vK16) > 0.0 ? std::fabs(vK16) : 1.0;
    if (std::fabs(vK16 - vK8) > kHoloKRelTol * denom) {
        if (holo_k_three_gate_enabled()) {
            V2Profile* prof = v2_profile_current();
            if (prof) ++prof->k_mid_attempts;
            double G24 = 0.0;
            auto mid_begin = V2Clock::now();
            const bool ok24 = gc2_accum_G(gc2_rule24(), 24, m, s, q, Rma,
                                          Rpa2, G24, nullptr);
            const double vK24 = two_over_rho * v * G24;
            const double mid_tol = holo_k_three_gate_strict()
                                       ? 1e-12
                                       : kHoloKRelTol;
            const bool agree24 = ok24 && std::isfinite(vK24) &&
                std::fabs(vK16 - vK24) <= mid_tol * denom;
            if (prof)
                v2_profile_add_ms(&V2Profile::k_mid_ms, mid_begin,
                                  V2Clock::now());
            if (agree24) {
                if (estimated_error) *estimated_error=std::fabs(vK16-vK24);
                if (prof) ++prof->k_mid_success;
                vK = vK16;
                return true;
            }
            if (prof) ++prof->k_mid_reject;
        }
        if (why) *why = KRejectReason::disagreement;
        return false;
    }
    vK = vK16;
    return true;
}

// (m, v) of one arc and its exact parameter derivatives, built from the two
// polished angular endpoints te < tl (theta) and their IFT-theta sensitivities
// dte_j = -dphi/dP_j / (dphi/dtheta), dtl_j likewise -- the same quantities
// radius_terms already forms for df0.  Param order (X, Y, rho, m0, a).
struct ArcPairJac {
    double m, v, s;                  // s = sqrt(v) = (t_hi - t_lo) / 2
    std::array<double, 5> dm, dv;
    bool ok;                         // false: straddles theta = pi / tangency
    int reject_reason = 0;           // 1 nonfinite, 2 order, 3 TMax, 4 VFloor
};
inline ArcPairJac arc_pair_jac(double te, double tl,
                               const std::array<double, 5>& dte,
                               const std::array<double, 5>& dtl) {
    ArcPairJac out{};
    const double t_lo = std::tan(0.5 * te);
    const double t_hi = std::tan(0.5 * tl);
    if (!std::isfinite(t_lo) || !std::isfinite(t_hi) || t_hi <= t_lo) {
        out.ok = false;
        out.reject_reason = (!std::isfinite(t_lo) || !std::isfinite(t_hi)) ? 1 : 2;
        return out;
    }
    const double half = 0.5 * (t_hi - t_lo);
    out.m = 0.5 * (t_lo + t_hi);
    out.v = half * half;
    out.s = half;
    if (std::fabs(out.m) + out.s > kHoloTMax || out.v <= kHoloVFloor) {
        out.ok = false;
        out.reject_reason = std::fabs(out.m) + out.s > kHoloTMax ? 3 : 4;
        return out;
    }
    // dt/dP_j = 0.5 (1 + t^2) dtheta/dP_j     (t = tan(theta/2))
    const double j_lo = 0.5 * (1.0 + t_lo * t_lo);
    const double j_hi = 0.5 * (1.0 + t_hi * t_hi);
    for (int j = 0; j < 5; ++j) {
        const double dlo = j_lo * dte[j];
        const double dhi = j_hi * dtl[j];
        out.dm[j] = 0.5 * (dlo + dhi);
        out.dv[j] = out.s * (dhi - dlo);
    }
    out.ok = true;
    return out;
}

// The reciprocal chart u = -1/t = tan((theta-pi)/2) is continuous across
// theta=pi, where the incumbent t chart has its pole.  It has the other pole
// at theta=0, so an arc that crosses a 2*pi boundary is rejected here and
// remains on the incumbent angular rescue.  The derivative is
// du/dP = (1/t^2) dt/dP = (1+u^2) dtheta/dP / 2.
inline ArcPairJac arc_pair_jac_reciprocal(
    double te, double tl, const std::array<double, 5>& dte,
    const std::array<double, 5>& dtl) {
    ArcPairJac out{};
    const double two_pi = 2.0 * kHoloPi;
    if (!std::isfinite(te) || !std::isfinite(tl) ||
        std::floor(te / two_pi) != std::floor(tl / two_pi)) {
        out.ok = false;
        out.reject_reason = 2;
        return out;
    }
    const double t_e = std::tan(0.5 * te);
    const double t_l = std::tan(0.5 * tl);
    if (!std::isfinite(t_e) || !std::isfinite(t_l) ||
        t_e == 0.0 || t_l == 0.0) {
        out.ok = false;
        out.reject_reason = 1;
        return out;
    }
    const double u_e = -1.0 / t_e;
    const double u_l = -1.0 / t_l;
    if (!std::isfinite(u_e) || !std::isfinite(u_l) || u_e == u_l) {
        out.ok = false;
        out.reject_reason = !std::isfinite(u_e) || !std::isfinite(u_l) ? 1 : 2;
        return out;
    }

    const double ju_e = 0.5 * (1.0 + t_e * t_e) / (t_e * t_e);
    const double ju_l = 0.5 * (1.0 + t_l * t_l) / (t_l * t_l);
    std::array<double, 5> du_e{}, du_l{};
    for (int j = 0; j < 5; ++j) {
        du_e[j] = ju_e * dte[j];
        du_l[j] = ju_l * dtl[j];
    }

    const double u_lo = std::min(u_e, u_l);
    const double u_hi = std::max(u_e, u_l);
    const std::array<double, 5>& d_lo = u_e < u_l ? du_e : du_l;
    const std::array<double, 5>& d_hi = u_e < u_l ? du_l : du_e;
    const double half = 0.5 * (u_hi - u_lo);
    out.m = 0.5 * (u_lo + u_hi);
    out.v = half * half;
    out.s = half;
    if (std::fabs(out.m) + out.s > kHoloTMax || out.v <= kHoloVFloor) {
        out.ok = false;
        out.reject_reason = std::fabs(out.m) + out.s > kHoloTMax ? 3 : 4;
        return out;
    }
    for (int j = 0; j < 5; ++j) {
        out.dm[j] = 0.5 * (d_lo[j] + d_hi[j]);
        out.dv[j] = out.s * (d_hi[j] - d_lo[j]);
    }
    out.ok = true;
    return out;
}

// (2/rho) Phi_arc(R) = v K, the F_half value integrand for one arc.
struct VKValue {
    double vK;
    bool ok;
    double estimated_error=0;
};
inline VKValue v_times_K(double m, double v, double R, const PrimaryFrame& pf,
                         const std::array<double, 5>& pc,
                         bool reciprocal = false) {
    V2Profile* prof = v2_profile_current();
    if (prof) ++prof->k_attempts;
    if (!std::isfinite(v) || !std::isfinite(m)) {
        profile_k_reject(KRejectReason::nonfinite);
        return {0.0, false};
    }
    if (!(v > kHoloVFloor)) {
        profile_k_reject(KRejectReason::vfloor);
        return {0.0, false};
    }
    const double s = std::sqrt(v);
    if (std::fabs(m) + s > kHoloTMax) {
        profile_k_reject(KRejectReason::tmax);
        return {0.0, false};
    }
    const DeflatedQuad q = deflate_mv(pc, m, v);
    const double Rma = R - pf.a, Rpa = R + pf.a;
    const double b0 = reciprocal ? Rpa : Rma;
    const double b2 = reciprocal ? Rma * Rma : Rpa * Rpa;
    double vK = 0.0, estimated_error=0.0;
    KRejectReason why = KRejectReason::none;
    auto k_begin = V2Clock::now();
    if (!k_rule_converged(m, v, s, q, b0, b2, 2.0 / pf.rho, vK, &why, &estimated_error)) {
        if (prof) v2_profile_add_ms(&V2Profile::k_ms, k_begin, V2Clock::now());
        profile_k_reject(why);
        return {0.0, false};
    }
    if (prof) {
        ++prof->k_success;
        v2_profile_add_ms(&V2Profile::k_ms, k_begin, V2Clock::now());
    }
    return {vK, true, estimated_error};
}

// v K and its analytic d/dP_j (param order X, Y, rho, m0, a).  `dpc` is
// boundary_quartic_dp(R, pf).dp (dpc[j] ascending [dp0..dp4]).  Requires
// S2 > 0 strictly at every node (the derivative divides by S2); any miss or
// non-finite output returns ok == false -> caller keeps the angular sweep.
struct VKJacobian {
    double vK;
    std::array<double, 5> dvK;
    bool ok;
};
inline VKJacobian v_times_K_jac(
    double m, double v, const std::array<double, 5>& dm,
    const std::array<double, 5>& dv, double R, const PrimaryFrame& pf,
    const std::array<double, 5>& pc,
    const std::array<std::array<double, 5>, 5>& dpc,
    bool reciprocal = false, int error_rule_nodes = kHoloNK) {
    VKJacobian out{};
    out.ok = false;
    if(error_rule_nodes!=8 && error_rule_nodes!=16) return out;
    V2Profile* prof = v2_profile_current();
    if (prof) ++prof->k_attempts;
    if (!std::isfinite(v) || !std::isfinite(m)) {
        profile_k_reject(KRejectReason::nonfinite);
        return out;
    }
    if (!(v > kHoloVFloor)) {
        profile_k_reject(KRejectReason::vfloor);
        return out;
    }
    const double s = std::sqrt(v);
    if (std::fabs(m) + s > kHoloTMax) {
        profile_k_reject(KRejectReason::tmax);
        return out;
    }

    const DeflatedQuad q = deflate_mv(pc, m, v);
    const double p4 = q.p4, d1 = q.d1, d0 = q.d0;
    const double Rma = R - pf.a, Rpa = R + pf.a;
    const double b0 = reciprocal ? Rpa : Rma;
    const double b2 = reciprocal ? Rma * Rma : Rpa * Rpa;
    const double inv_rho = 1.0 / pf.rho;
    const double two_over_rho = 2.0 * inv_rho;
    const double inv_2s = 0.5 / s;

    // fail closed on an 8-vs-16-node quadrature disagreement (wide arc, endpoint
    // near theta = pi) -- same gate v_times_K applies to the value.
    double vK_chk = 0.0;
    KRejectReason why = KRejectReason::none;
    auto k_begin = V2Clock::now();
    if (!k_rule_converged(m, v, s, q, b0, b2, two_over_rho, vK_chk, &why)) {
        if (prof) v2_profile_add_ms(&V2Profile::k_ms, k_begin, V2Clock::now());
        profile_k_reject(why);
        return out;
    }
    if (prof) v2_profile_add_ms(&V2Profile::k_ms, k_begin, V2Clock::now());

    // deflated-coefficient parameter derivatives
    //   p4 = pc[4]
    //   d1 = pc[3] + 2 m p4
    //   d0 = pc[2] + 2 m d1 - (m^2 - v) p4
    std::array<double, 5> dp4{}, dd1{}, dd0{};
    for (int j = 0; j < 5; ++j) {
        dp4[j] = dpc[j][4];
        dd1[j] = dpc[j][3] + 2.0 * (dm[j] * p4 + m * dp4[j]);
        dd0[j] = dpc[j][2] + 2.0 * (dm[j] * d1 + m * dd1[j]) -
                 (2.0 * m * dm[j] - dv[j]) * p4 - (m * m - v) * dp4[j];
    }

    const GC2Rule& gc = error_rule_nodes==8 ? gc2_rule8() : gc2_rule16();
    double G = 0.0;
    std::array<double, 5> dG{};
    for (int i = 0; i < error_rule_nodes; ++i) {
        const double xi = gc.x[i], wi = gc.w[i];
        const double t = m + s * xi;
        const double S2 = -(d0 + t * (d1 + t * p4));
        if (!(S2 > 0.0)) {
            profile_k_reject(KRejectReason::s2);
            return out;
        }
        const double A = 1.0 + t * t;
        const double B = b0 * b0 + b2 * t * t;
        if (!(B > 0.0)) {
            profile_k_reject(KRejectReason::b);
            return out;
        }
        const double wv = std::sqrt(S2) / (A * std::sqrt(A) * std::sqrt(B));
        G += wi * wv;

        const double dS2_dt = -(d1 + 2.0 * t * p4);
        const double dA_dt = 2.0 * t;
        const double dB_dt = 2.0 * b2 * t;
        const double c_S2 = 0.5 / S2, c_A = -1.5 / A, c_B = -0.5 / B;
        for (int j = 0; j < 5; ++j) {
            const double dt = dm[j] + xi * inv_2s * dv[j];
            double dS2 = -(dd0[j] + t * (dd1[j] + t * dp4[j])) + dS2_dt * dt;
            double dA = dA_dt * dt;
            double dB = dB_dt * dt;
            if (j == 4) {
                dB += reciprocal ? 2.0 * Rpa - 2.0 * Rma * t * t
                                  : -2.0 * Rma + 2.0 * Rpa * t * t;
            }  // explicit d/da B
            const double dw = wv * (c_S2 * dS2 + c_A * dA + c_B * dB);
            dG[j] += wi * dw;
        }
    }

    const double vK = two_over_rho * v * G;
    for (int j = 0; j < 5; ++j) {
        double d = two_over_rho * (dv[j] * G + v * dG[j]);
        if (j == 2) d += (-2.0 * inv_rho * inv_rho) * v * G;  // d/drho of 2/rho
        out.dvK[j] = d;
    }
    out.vK = vK;
    out.ok = std::isfinite(vK);
    for (double d : out.dvK) out.ok = out.ok && std::isfinite(d);
    if (prof) {
        if (out.ok) ++prof->k_success;
        else profile_k_reject(KRejectReason::nonfinite);
    }
    return out;
}

}  // namespace lcbinint::holonomic
