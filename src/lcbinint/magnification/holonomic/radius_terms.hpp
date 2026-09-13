#pragma once

// ATPT holonomic solver (M7) -- one radius: value + derivative integrands.
// Ports python/lcbinint/holonomic_ref/jacobian.py:
//   polish_endpoint, _real_root_thetas, arc_intervals, _grid_intervals,
//   _full_circle_terms, radius_terms.  The fixed-grid helper remains only as
// an independent diagnostic; production arc discovery is quartic-based.
//
//   f0    = R * sum_arcs (theta_leave - theta_enter)
//   fh    = R * sum_arcs int_arc sqrt(phi) dtheta        (GC-1(64) angular)
//   df0   = R * sum_arcs ( d theta_leave/dP - d theta_enter/dP )   (IFT-theta)
//   dfh   = R * sum_arcs int_arc  dphi/dP / (2 sqrt phi) dtheta
//
// `reliable` is false on a near-tangency, a degenerate quartic, or a
// full-circle radius -- the caller then marks the whole Jacobian
// GRADIENT_UNRELIABLE (fail closed).

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <limits>

#include "lcbinint/magnification/holonomic/angular_rule.hpp"
#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/holonomic_transport.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/phi.hpp"
#include "lcbinint/magnification/holonomic/poly_roots.hpp"
#include "lcbinint/magnification/holonomic/quartic_sturm.hpp"
#include "lcbinint/magnification/holonomic/quartic_local_bracket.hpp"
#include "lcbinint/magnification/holonomic/root_pair.hpp"
#include "lcbinint/magnification/holonomic/v2_profile.hpp"

namespace lcbinint::holonomic {

constexpr double kTwoPi = 6.28318530717958647692;
constexpr double kTanRel = 1e-4;       // jacobian._TAN_REL
constexpr double kRootImRel = 1e-8;    // jacobian._ROOT_IM_REL
constexpr double kP4DegenRel = 1e-11;  // jacobian._P4_DEGEN_REL

struct RadiusTerms {
    double f0 = 0.0;
    double fh = 0.0;
    std::array<double, 5> df0{};
    std::array<double, 5> dfh{};
    bool reliable = true;
};

// ---- Newton on phi(R, theta; P) = 0 -----------------------------------
struct PolishResult {
    double theta;
    double dphi_dtheta;
    bool reliable;
};
inline PolishResult polish_endpoint(double R, double theta,
                                    const PrimaryFrame& pf, int iters = 6,
                                    PhiValDtheta* final_phi = nullptr) {
    V2Profile* prof = v2_profile_current();
    if (prof) ++prof->endpoint_calls;
    V2ProfileTimer endpoint_timer(&V2Profile::endpoint_ms);
    double th = theta, dth = 1.0;
#ifdef HOLO_ENDPOINT_WORK_PROBE
    double visited[7]{}; int nvisited=0;
    auto record=[&](double arg) {
        if(!prof)return;
        ++prof->endpoint_evaluations;
        for(int k=0;k<nvisited;++k)if(arg==visited[k] && std::signbit(arg)==std::signbit(visited[k])) {
            ++prof->endpoint_repeated_arguments;break;
        }
        if(nvisited<7)visited[nvisited++]=arg;
    };
#endif
    for (int i = 0; i < iters; ++i) {
#ifdef HOLO_ENDPOINT_WORK_PROBE
        record(th);
#endif
        PhiValDtheta g = phi_val_dtheta(R, th, pf);
        dth = g.dphi_dtheta;
        if (std::fabs(dth) < 1e-13) {
            if (prof) ++prof->endpoint_unreliable;
            return {th, dth, false};
        }
        double step = g.phi / dth;
#ifdef HOLO_ADAPTIVE_STATIONARY_ENDPOINT
        const double previous = th;
#endif
        th -= step;
#ifdef HOLO_ADAPTIVE_STATIONARY_ENDPOINT
        // Adaptive caller requests the final phi. Reuse only an identical
        // argument, including the sign of zero; this is not a looser gate.
        if (final_phi && th == previous &&
            std::signbit(th) == std::signbit(previous)) {
            if(prof)++prof->endpoint_stationary_stops;
            *final_phi = g;
            const bool reliable = std::fabs(g.dphi_dtheta) >= 1e-13;
            if (prof && !reliable) ++prof->endpoint_unreliable;
            return {th, g.dphi_dtheta, reliable};
        }
#endif
        if (std::fabs(step) < 1e-15) {
            if(prof)++prof->endpoint_small_step_stops;
            break;
        }
        if(i+1==iters && prof)++prof->endpoint_limit_stops;
    }
#ifdef HOLO_ENDPOINT_WORK_PROBE
    record(th);
#endif
    PhiValDtheta g = phi_val_dtheta(R, th, pf);
    if (final_phi) *final_phi = g;
    const bool reliable = std::fabs(g.dphi_dtheta) >= 1e-13;
    if (prof && !reliable) ++prof->endpoint_unreliable;
    return {th, g.dphi_dtheta, reliable};
}

// ---- sorted theta of the real roots of P(t), t = tan(theta/2) --------
// descending, leading-zero-deflated quartic coeffs -> working degree.
inline int quartic_descending(const std::array<double, 5>& pc, double c[5]) {
    c[0] = pc[4]; c[1] = pc[3]; c[2] = pc[2]; c[3] = pc[1]; c[4] = pc[0];
    int deg = 4;
    while (deg > 0 && c[0] == 0.0) {
        for (int i = 0; i < deg; ++i) c[i] = c[i + 1];
        --deg;
    }
    return deg;
}

// complex roots -> sorted, near-double-merged theta list.  Real filter
// matches np.roots + |Im| <= _ROOT_IM_REL (1 + |Re|); merge at 1e-11 in t.
inline std::vector<double> thetas_from_complex(
    const std::vector<Cplx<double>>& z) {
    std::vector<double> out;
    for (const auto& r : z)
        if (std::fabs(r.im) <= kRootImRel * (1.0 + std::fabs(r.re))) {
            double t = std::fmod(2.0 * std::atan(r.re), kTwoPi);
            if (t < 0.0) t += kTwoPi;
            out.push_back(t);
        }
    std::sort(out.begin(), out.end());
    std::vector<double> merged;
    for (double x : out)
        if (merged.empty() || x - merged.back() > 1e-11) merged.push_back(x);
    return merged;
}

inline std::vector<double> real_root_thetas(const std::array<double, 5>& pc) {
    double c[5];
    int deg = quartic_descending(pc, c);
    if (deg <= 0) return {};
    return thetas_from_complex(aberth<double>(c, deg, 40));
}

// Roots in the reflected projective chart u = -1/t = tan((theta-pi)/2).
// Mapping through theta = pi + 2 atan(u) keeps roots near theta=pi finite,
// including the exact u=0 representative of the t=+/-infinity root.  This
// path is used only when the t-chart leading coefficient is numerically small;
// it is an algebraic chart change, never an angular-grid rescue.
inline std::vector<double> thetas_from_reciprocal_complex(
    const std::vector<Cplx<double>>& z) {
    std::vector<double> out;
    out.reserve(z.size());
    for (const auto& r : z) {
        if (std::fabs(r.im) > kRootImRel * (1.0 + std::fabs(r.re))) continue;
        double theta = 0.5 * kTwoPi + 2.0 * std::atan(r.re);
        theta = std::fmod(theta, kTwoPi);
        if (theta < 0.0) theta += kTwoPi;
        out.push_back(theta);
    }
    std::sort(out.begin(), out.end());
    std::vector<double> merged;
    for (double x : out)
        if (merged.empty() || x - merged.back() > 1e-11)
            merged.push_back(x);
    return merged;
}

inline std::vector<double> real_root_thetas_reciprocal(
    const std::array<double, 5>& pc) {
    double c[5];
    int deg = quartic_descending(pc, c);
    if (deg <= 0) return {};
    return thetas_from_reciprocal_complex(aberth<double>(c, deg, 40));
}

// Warm-start state for the per-radial-node quartic solve inside one cell.
// The Chebyshev nodes are monotone in R inside a cell and the boundary
// quartic's roots move smoothly with R away from a tangency, so node k
// seeds node k-1's converged roots into a short Aberth pass instead of a
// cold circle start.
//
// The warm result is adopted only when the final Aberth correction is
// <= kWarmStepTol (roots then good well past what the downstream
// 6-iteration polish_endpoint Newton needs -- the quartic roots are arc
// seeds, not final endpoints) AND the real-root count is unchanged from
// the seed.  Any failure -> exact cold 40-iteration solve, identical to
// real_root_thetas().  So an adopted set is never a degraded guess and the
// arc topology can never silently change.  After kColdStreak consecutive
// warm misses the warm path is disabled for the rest of the cell (a moving
// tangency -- stop paying warm+cold).  Reset (valid=false) at each cell
// boundary and at a chart (p4~0) radius.
constexpr int kWarmIters = 20;
constexpr double kWarmStepTol = 1e-7;
constexpr int kColdStreak = 3;

struct QuarticWarm {
    Cplx<double> z[4];
    int deg = 0;
    int n_real = -1;
    bool valid = false;    // z[] holds a usable previous-node root set
    int cold_streak = 0;   // consecutive warm misses -> disable at kColdStreak
    long warm_hits = 0, cold_falls = 0;
};

inline std::vector<double> real_root_thetas_warm(const std::array<double, 5>& pc,
                                                 QuarticWarm& w) {
    V2Profile* prof = v2_profile_current();
    double c[5];
    int deg = quartic_descending(pc, c);
    if (deg <= 0) { w.valid = false; return {}; }

    if (w.valid && w.deg == deg && w.cold_streak < kColdStreak) {
        if (prof) ++prof->quartic_warm_calls;
        double step = 1.0;
        auto z = aberth<double>(c, deg, kWarmIters, w.z, 0.0, &step);
        if (step <= kWarmStepTol) {
            auto th = thetas_from_complex(z);
            if (w.n_real < 0 || (int)th.size() == w.n_real) {
                for (int i = 0; i < deg; ++i) w.z[i] = z[i];
                w.n_real = (int)th.size();
                w.cold_streak = 0;
                ++w.warm_hits;
                if (prof) ++prof->quartic_warm_hits;
                return th;
            }
        }
        ++w.cold_streak;
    }

    // cold path -- identical roots to real_root_thetas()
    auto z = aberth<double>(c, deg, 40);
    auto th = thetas_from_complex(z);
    for (int i = 0; i < deg && i < 4; ++i) w.z[i] = z[i];
    w.deg = deg;
    w.n_real = (int)th.size();
    w.valid = (deg <= 4);
    ++w.cold_falls;
    if (prof) {
        ++prof->quartic_cold_calls;
        ++prof->quartic_cold_falls;
    }
    return th;
}

// ===================================================================
// (m, v) root-pair radial transport  --  flag-gated alternative to the
// warm Aberth quartic solve inside one integration cell.
//
// Instead of re-solving the boundary quartic at every radial node the
// integrator carries the symmetric pair (m, v) = ((t_+ + t_-)/2,
// ((t_+ - t_-)/2)^2) of each image arc (root_pair.hpp) and advances it
// with a predictor (root_pair_dR, IFT in R using the previous node's
// coefficients) + a few Newton steps on (E, O) = 0 at the new node.
//
// Fail-closed: any singular 2x2, near-tangency (v below kTransportVFloor),
// non-converged corrector, arc-count change, or non-finite state drops to
// the exact 40-iteration cold Aberth solve -- byte-identical roots to
// real_root_thetas() -- and re-seeds.  After kColdStreak consecutive
// misses the caller abandons transport for the rest of the cell.
//
// Scope limit (t = tan(theta/2)): an arc that crosses theta = pi maps to
// the unbounded t-interval (t_last, +inf) u (-inf, t_first); (m, v) has no
// finite midpoint there.  Such a cell is detected at seed time (the phi>0
// arcs are the "odd" gaps) and left un-seeded -> always cold.
// ===================================================================
// Runtime override for tests / the 3-solver benchmark (variants toggled in one
// process): -1 = follow the environment variable (production), 0 = force off,
// 1 = force on.  Production code never touches this.
inline int& holo_mv_transport_override() {
    static int v = -1;
    return v;
}

// Process-wide opt-out: HOLO_MV_TRANSPORT_LEGACY=1 restores the pre-flip
// behaviour (arc endpoints re-solved cold with real_root_thetas_warm at
// every radial node).  Default: the (m,v) root-pair transport -- proven
// bit-parity vs the cold path (max |dmu| 2.53e-14, 0 status downgrades,
// checkpoint sec. 25.7) at -8.7% epoch median / all percentiles
// non-regressing (checkpoint sec. 29,
// evidence/holonomic/holonomic_3solver_benchmark.txt).  The kept flag is
// the A/B escape hatch; the arc-birth disc-sign guard, branch-aware
// acceptance and from_warm_d14 thin-arc cross-check make the transport
// fail closed to the cold solve on every topology change.
inline bool holo_mv_transport_enabled() {
    const int o = holo_mv_transport_override();
    if (o >= 0) return o != 0;
    static const bool on = [] {
        const char* e = std::getenv("HOLO_MV_TRANSPORT_LEGACY");
        return !(e && e[0] == '1');
    }();
    return on;
}

inline bool holo_mv_noseed() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_MV_NOSEED");
        return e && e[0] == '1';
    }();
    return on;
}

inline bool holo_mv_debug() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_MV_DEBUG");
        return e && e[0] == '1';
    }();
    return on;
}

constexpr int kTransportNewton = 5;
constexpr double kTransportStepTol = 1e-13;    // step-relative corrector gate
constexpr double kTransportRootRel = 1e-11;    // |P(t_pm)| / running-abssum: is
                                               // each transported endpoint an
                                               // actual root of P (scale-correct
                                               // at any |t|, unlike an (E,O)
                                               // residual scaled by cmax*m^4).
constexpr double kTransportVFloor = 1e-10;      // v <= this: near-tangency, the
                                               // 2x2 (E,O) solve is ill-cond
                                               // (det ~ -1/2 P''^2, corrector
                                               // step ~ 1/sqrt(v)) -> cold.
                                               // Genuine physical tangencies
                                               // are additionally caught by
                                               // radius_terms' tan_thresh.
constexpr int kTransportMaxTrips = 4;          // total misses/cell -> abandon
                                               // (guards an oscillating cell
                                               //  from paying transport+cold)
constexpr double kTransportTMax = 12.0;        // max endpoint |t| = |m| + sqrt(v)
                                               // for a transported arc.  t =
                                               // tan(theta/2): |t| > 12 is an
                                               // arc within ~0.17 rad of
                                               // theta = pi, where the (m, v)
                                               // chart is catastrophically
                                               // ill-conditioned (dt/dtheta ~
                                               // (1+t^2)/2 ~ 70; v = ((t+-t-)/
                                               // 2)^2 worse still) and the
                                               // predictor/corrector wander
                                               // basins on a ~1e-13 seed
                                               // perturbation.  Such an arc
                                               // can use the exact reciprocal
                                               // chart in the isolated A/B
                                               // retry; the production default
                                               // stays on the t-chart.
constexpr double kTransportVJumpRel = 4.0;     // predictor sanity: reject the
                                               // step if the linear IFT
                                               // extrapolation changes v by
                                               // more than this factor (a fold
                                               // / near-tangency the linear
                                               // predictor cannot see) -> cold.
constexpr double kTransportCertifyV = 1e-5;    // on a warm-D14-reused cell plan
                                               // (RootPairWarm::certify), any
                                               // transported arc whose v drops
                                               // below this is thin enough that
                                               // the ~1e-8 seed perturbation of
                                               // the reused cell boundary is
                                               // amplified by the near-fold
                                               // conditioning -> cross-check the
                                               // whole theta set against a cold
                                               // quartic solve at this node.
constexpr double kTransportCertifyRel = 1e-7;  // reject the continuation (fall
                                               // closed to the cold solve) if a
                                               // cross-checked theta differs
                                               // from the cold root by more
                                               // than this * (1 + |theta|):
                                               // loose enough to pass ordinary
                                               // continuation truncation, tight
                                               // enough to catch a wrong-arc /
                                               // basin-flipped pair.
constexpr double kTransportGapRel = 0.25;      // pair-ambiguity gate: reject the
                                               // whole set if the real t-gap
                                               // between two consecutive arcs is
                                               // below this fraction of the
                                               // narrower arc's full width
                                               // (2 sqrt(v)).  Two arcs about to
                                               // merge = an approaching caustic,
                                               // where dmu/d(endpoint) ~ 1e3 and
                                               // the (E, O) corrector can swap an
                                               // endpoint between the near-merged
                                               // arcs -- both assignments satisfy
                                               // E = O = 0 to a tiny residual, so
                                               // the ~1e-13 L2 warm-D14 coeff
                                               // jitter alone flips which arcs
                                               // pass.  -> cold quartic.

// Sign of the ascending quartic's discriminant (coeffs pc[0..4]).  The
// discriminant changes sign exactly when the real-root count crosses between
// {2} and {0 or 4} -- i.e. an image-arc pair is born or dies.  Pre-scaled by
// the max coeff magnitude (the scale enters as a positive power, sign-safe).
// ~30 flops, no root solve.  +1 disc > 0 / -1 disc < 0 / 0 disc == 0.
// (Same formula as fast_topo_detail::quartic_disc_sign; kept local so this
// low-level header carries no upward dependency.)
inline int transport_disc_sign(const std::array<double, 5>& pc) {
    double sc = 0.0;
    for (double v : pc) sc = std::max(sc, std::fabs(v));
    if (!(sc > 0.0)) return 0;
    const double a = pc[4] / sc, b = pc[3] / sc, c = pc[2] / sc, d = pc[1] / sc,
                 e = pc[0] / sc;
    const double D =
        256.0 * a * a * a * e * e * e - 192.0 * a * a * b * d * e * e -
        128.0 * a * a * c * c * e * e + 144.0 * a * a * c * d * d * e -
        27.0 * a * a * d * d * d * d + 144.0 * a * b * b * c * e * e -
        6.0 * a * b * b * d * d * e - 80.0 * a * b * c * c * d * e +
        18.0 * a * b * c * d * d * d + 16.0 * a * c * c * c * c * e -
        4.0 * a * c * c * c * d * d - 27.0 * b * b * b * b * e * e +
        18.0 * b * b * b * c * d * e - 4.0 * b * b * b * d * d * d -
        4.0 * b * b * c * c * c * e + b * b * c * c * d * d;
    return D > 0.0 ? 1 : (D < 0.0 ? -1 : 0);
}

// Is `t` a genuine root of the ascending quartic `pc`?  Compares |P(t)| to
// the running abs-sum of the Horner accumulation -- the standard evaluation
// error scale, correct at any |t| (t = tan(theta/2) can be large near
// theta = pi where an (|E|+|O|) < tol * cmax * m^4 gate is near-vacuous).
inline bool transport_is_root(const std::array<double, 5>& pc, double t) {
    double P = 0.0, S = 0.0;
    const double at = std::fabs(t);
    for (int k = 4; k >= 0; --k) {
        P = P * t + pc[k];
        S = S * at + std::fabs(pc[k]);
    }
    return std::fabs(P) <= kTransportRootRel * (S > 0.0 ? S : 1.0);
}

// A quartic can contribute at most two bounded (m,v) pairs.  Keep the
// transport state inline so adding a radial sample cannot allocate/copy a
// heap-backed vector just to carry two roots.
struct RootPairSet {
    std::array<RootPair, 2> data{};
    std::uint8_t count = 0;

    std::size_t size() const { return count; }
    bool empty() const { return count == 0; }
    void clear() { count = 0; }
    bool push_back(const RootPair& p) {
        if (count >= data.size()) return false;
        data[count++] = p;
        return true;
    }
    RootPair& operator[](std::size_t i) { return data[i]; }
    const RootPair& operator[](std::size_t i) const { return data[i]; }
    auto begin() { return data.begin(); }
    auto end() { return data.begin() + count; }
    auto begin() const { return data.begin(); }
    auto end() const { return data.begin() + count; }
};

// Branch-aware acceptance for a transported pair set: the pairs must still be
// the same ascending, non-overlapping family of *real inside* arcs they were
// seeded as.  Rejects a corrector basin-flip that converges to a different
// (E, O) root -- a neighbouring arc, or an outside gap -- whose (E, O)
// residual is nonetheless tiny.  Checks, per pair: v > 0 and finite; strict
// ascending order with a real t-gap to the previous arc; both endpoints
// genuine roots of P; phi > 0 at the arc's t-midpoint (a real inside arc,
// not an outside gap).
template <class PairSet>
inline bool transport_pairs_valid(const PairSet& ps,
                                  size_t n_expect,
                                  const std::array<double, 5>& pc, double R,
                                  const PrimaryFrame& pf) {
    if (ps.size() != n_expect) return false;
    double prev_hi = -std::numeric_limits<double>::infinity();
    double prev_w = std::numeric_limits<double>::infinity();
    for (const auto& rp : ps) {
        if (!(rp.v > 0.0) || !std::isfinite(rp.v) || !std::isfinite(rp.m))
            return false;
        const double s = std::sqrt(rp.v);
        if (std::fabs(rp.m) + s > kTransportTMax)
            return false;  // arc too close to theta = pi: (m, v) chart unusable
        const double tlo = rp.m - s, thi = rp.m + s;
        if (!(tlo > prev_hi)) return false;  // overlap / reorder / collapse
        const double w = 2.0 * s;            // full arc width
        if (tlo - prev_hi < kTransportGapRel * std::fmin(w, prev_w))
            return false;  // arcs near-merged: pair assignment ambiguous (caustic)
        prev_hi = thi;
        prev_w = w;
        if (!transport_is_root(pc, tlo) || !transport_is_root(pc, thi))
            return false;
        double thm = std::fmod(2.0 * std::atan(rp.m), kTwoPi);
        if (thm < 0.0) thm += kTwoPi;
        if (!(phi_lens(R, thm, pf) > 0.0)) return false;
    }
    return true;
}

struct RootPairWarm {
    RootPairSet pairs;                  // tracked t-bounded pairs, ascending in t
    std::array<double, 5> pc_prev{};    // boundary_quartic(R_prev) coeffs
    std::array<double, 5> pcR_prev{};   // boundary_quartic_dR(R_prev) coeffs
    double R_prev = 0.0;
    bool valid = false;                 // pairs[] holds a usable previous-node set
    int disc_sign = 0;                  // sign(quartic discriminant) at the seed
                                       // node: a flip = an arc pair born/died,
                                       // which the continuation cannot see
    bool certify = false;               // cell plan came from a warm-D14 reuse:
                                       // cross-check the continuation on thin
                                       // near-caustic arcs against a cold solve
    int cold_streak = 0;                // consecutive transport misses
    int trips = 0;                      // total misses this cell
    long warm_hits = 0, cold_falls = 0, certify_falls = 0;
};

// Sorted theta list of the real roots of P(t), via (m, v) transport when
// `w` carries a live seed, else a cold solve that (re)seeds `w`.  The
// returned list is structurally identical to real_root_thetas() /
// real_root_thetas_warm() so the downstream arc formation is unchanged.
inline std::vector<double> real_root_thetas_transport(
    const std::array<double, 5>& pc, double R, const PrimaryFrame& pf,
    RootPairWarm& w, QuarticWarm* qw = nullptr) {
    V2Profile* prof = v2_profile_current();
    if (prof) ++prof->rootpair_calls;
    V2ProfileTimer rootpair_timer(&V2Profile::rootpair_ms);
    // ---- warm transport -------------------------------------------------
    // A discriminant sign flip since the seed means an arc pair was born or
    // died between nodes -- a topology change the (m, v) continuation cannot
    // detect (it would keep tracking its old pair count and silently drop the
    // new arc, e.g. the HOLO_MV_TRANSPORT x L2 mu ~ 2e-3 on extreme-q-planet).
    // Fall closed: the cold solve below re-seeds with the full root set.
    const int disc_now = transport_disc_sign(pc);
    if (w.valid && !w.pairs.empty() && w.disc_sign != 0 && disc_now != 0 &&
        disc_now != w.disc_sign) {
        if (prof) ++prof->rootpair_disc_mismatch;
        w.valid = false;
        ++w.cold_streak;
        ++w.trips;
    }
    if (w.valid && !w.pairs.empty()) {
        const double dR = R - w.R_prev;
        RootPairSet next = w.pairs;
        const bool dbg = holo_mv_debug();
        bool ok = true;
        for (auto& rp : next) {
            const double m_seed = rp.m, v_seed = rp.v;
            // predictor: IFT in R on the *previous* node's coefficients
            const RootPairDR pd = root_pair_dR(rp, w.pc_prev, w.pcR_prev);
            if (pd.ok && std::isfinite(pd.dm_dR) && std::isfinite(pd.dv_dR)) {
                rp.m += pd.dm_dR * dR;
                rp.v += pd.dv_dR * dR;
            }
            const double m_pred = rp.m, v_pred = rp.v;
            // predictor sanity: the linear IFT step must not blow v up (a fold
            // the linear model cannot see) nor push the arc toward theta = pi.
            const bool pred_wild =
                !std::isfinite(v_pred) || !std::isfinite(m_pred) ||
                v_pred <= 0.0 ||
                std::fabs(v_pred - v_seed) >
                    kTransportVJumpRel * v_seed + kTransportVFloor ||
                std::fabs(m_pred) + std::sqrt(std::fabs(v_pred)) >
                    kTransportTMax;
            if(prof && v_pred<=0.0)++prof->rootpair_predictor_nonpositive;
            if (pred_wild && prof) {
                ++prof->rootpair_predictor_reject;
                if (v_pred <= kTransportVFloor) ++prof->rootpair_vfloor_reject;
                if (std::isfinite(v_pred) && std::isfinite(m_pred) &&
                    std::fabs(m_pred) + std::sqrt(std::fabs(v_pred)) > kTransportTMax)
                    ++prof->rootpair_tmax_reject;
            }
            // corrector: Newton on (E, O) = 0 at the new node
            bool conv = false;
            double det_lo = std::numeric_limits<double>::infinity();
            double detsc_lo = std::numeric_limits<double>::infinity();
            for (int it = 0; it < kTransportNewton; ++it) {
                if (prof) ++prof->rootpair_newton_iterations;
                const EO f = eo_residuals(rp, pc);
                const EOJac J = eo_jacobian(rp, pc);
                const double det = J.E_m * J.O_v - J.E_v * J.O_m;
                if (dbg && std::fabs(det) < det_lo) {
                    det_lo = std::fabs(det);
                    detsc_lo = (std::fabs(J.E_m) + std::fabs(J.E_v)) *
                               (std::fabs(J.O_m) + std::fabs(J.O_v));
                }
                if (!(std::fabs(det) > 0.0) || !std::isfinite(det)) break;
                const double dm = -(J.O_v * f.E - J.E_v * f.O) / det;
                const double dv = -(J.E_m * f.O - J.O_m * f.E) / det;
                rp.m += dm;
                rp.v += dv;
                const double sc =
                    std::fabs(rp.m) + std::sqrt(std::fabs(rp.v)) + 1.0;
                if (std::fabs(dm) + std::fabs(dv) <= kTransportStepTol * sc) {
                    conv = true;
                    break;
                }
            }
            if(prof && rp.v<=0.0)++prof->rootpair_corrected_nonpositive;
            if(prof && rp.v>0.0 && rp.v<=kTransportVFloor)++prof->rootpair_corrected_tiny_positive;
            if (dbg) {
                std::fprintf(stderr,
                    "[mvT] R=%.15g dR=%.3e seed(m=%.12g v=%.6e) "
                    "pred(m=%.12g v=%.6e) fin(m=%.12g v=%.6e) conv=%d "
                    "dPred=%.3e dCorr=%.3e detEO=%.3e detEO/sc=%.3e\n",
                    R, dR, m_seed, v_seed, m_pred, v_pred, rp.m, rp.v, (int)conv,
                    std::fabs(m_pred - m_seed) + std::fabs(v_pred - v_seed),
                    std::fabs(rp.m - m_pred) + std::fabs(rp.v - v_pred),
                    det_lo, detsc_lo > 0.0 ? det_lo / detsc_lo : 0.0);
            }
#ifdef HOLO_MV_TINY_BRACKET_PROBE
            if(prof && !pred_wild && conv && std::isfinite(rp.m) &&
               rp.v>0.0 && rp.v<=kTransportVFloor) {
                ++prof->rootpair_tiny_eligible;
                std::array<double,4> endpoints{rp.t_minus(),rp.t_plus(),0,0};
                if(local_bracket_detail::certify(pc,endpoints,2))
                    ++prof->rootpair_tiny_bracket_pass;
            }
#endif
            if (pred_wild || !conv || !std::isfinite(rp.m) ||
                !std::isfinite(rp.v) || rp.v <= kTransportVFloor) {
                if (prof && !pred_wild) {
                    ++prof->rootpair_newton_reject;
                    if (!conv) ++prof->rootpair_residual_reject;
                    if (rp.v <= kTransportVFloor) ++prof->rootpair_vfloor_reject;
                }
                ok = false;
                break;
            }
            if (prof) ++prof->rootpair_newton_pairs_accepted;
        }
        // branch-aware acceptance: still the same ascending, non-overlapping
        // family of real inside arcs (rejects a corrector basin-flip whose
        // (E, O) residual is tiny but which tracks the wrong arc / an
        // outside gap).
        if (ok && transport_pairs_valid(next, w.pairs.size(), pc, R, pf)) {
            std::vector<double> out;
            out.reserve(next.size() * 2);
            for (const auto& rp : next)
                for (double t : {rp.t_minus(), rp.t_plus()}) {
                    double th = std::fmod(2.0 * std::atan(t), kTwoPi);
                    if (th < 0.0) th += kTwoPi;
                    out.push_back(th);
                }
            std::sort(out.begin(), out.end());
#ifdef HOLO_ARC_COMPACT_STORAGE
            size_t count=0;
            for(double x:out)
                if(count==0 || x-out[count-1]>1e-11)out[count++]=x;
            out.resize(count);
            auto merged=std::move(out);
#else
            std::vector<double> merged;
            for (double x : out)
                if (merged.empty() || x - merged.back() > 1e-11)
                    merged.push_back(x);
#endif
            if ((int)merged.size() == 2 * (int)next.size()) {
                // Certify the continuation on a warm-D14-reused cell plan when
                // an arc is thin enough for the reused-boundary seed
                // perturbation to be amplified: cross-check every transported
                // theta against a cold quartic solve at this node, fall closed
                // (cold path below) on any mismatch.  This is the
                // HOLO_MV_TRANSPORT x L2-warm-D14 interaction (checkpoint 25.4).
                bool cert_ok = true;
                if (w.certify) {
                    if (prof) ++prof->rootpair_certify_calls;
                    double v_min = std::numeric_limits<double>::infinity();
                    for (const auto& rp : next) v_min = std::fmin(v_min, rp.v);
                    if (v_min < kTransportCertifyV) {
                        bool local_certified=false;
#if !defined(HOLO_WARM_DISABLE_LOCAL_BRACKET)
                        if(prof)++prof->local_bracket_attempts;
                        std::array<double,4> tracked{};int count=0;
                        for(const auto& pair:next){tracked[count++]=pair.t_minus();tracked[count++]=pair.t_plus();}
                        std::sort(tracked.begin(),tracked.begin()+count);
                        local_certified=local_bracket_detail::certify(pc,tracked,count);
                        if(prof&&local_certified)++prof->local_bracket_successes;
#endif
                        if(!local_certified){
                        std::vector<double> cold = real_root_thetas(pc);
                        std::sort(cold.begin(), cold.end());
                        for (double thc : merged) {
                            double best = std::numeric_limits<double>::infinity();
                            for (double x : cold)
                                best = std::fmin(best, std::fabs(x - thc));
                            if (best > kTransportCertifyRel * (1.0 + std::fabs(thc))) {
                                cert_ok = false;
                                break;
                            }
                        }
                        if (!cert_ok) {
                            ++w.certify_falls;
                            if (prof) ++prof->rootpair_certify_falls;
                        }
                        }
                    }
                }
                if (cert_ok) {
                    w.pairs = next;
                    w.pc_prev = pc;
                    w.pcR_prev = boundary_quartic_dR(R, pf).p;
                    w.R_prev = R;
                    ++w.warm_hits;
                    if (prof) ++prof->rootpair_warm_success;
                    // Keep the warm Aberth seed (qw) live: it is only a few
                    // nodes stale and real_root_thetas_warm re-polishes it,
                    // whereas forcing a *cold* 40-iter Aberth on the next
                    // fallback is condition ~ 1/sqrt(disc) near the fold that
                    // made transport fall in the first place.
                    return merged;
                }
            }
        }
        w.valid = false;
        if (prof) ++prof->rootpair_branch_reject;
        ++w.cold_streak;
        ++w.trips;
    }

    // ---- cold solve + (re)seed ----------------------------------------
    // Delegate to the warm Aberth quartic (QuarticWarm) rather than a bare
    // cold solve.  A warm continuation solver stays in the right basin under
    // the ~1e-13 coefficient perturbation an L2 warm-D14 cell boundary
    // carries; a *cold* global Aberth near a folding (near-double) root has
    // condition ~ 1/sqrt(disc) and can shift a root enough to move mu by
    // ~1e-3 on a near-caustic cell -- the HOLO_MV_TRANSPORT x L2 interaction.
    double c[5];
    int deg = quartic_descending(pc, c);
    if (deg <= 0) {
        w.valid = false;
        ++w.cold_streak;
        return {};
    }
    std::vector<double> th;
    Cplx<double> zbuf[4];
    if (qw) {
        th = real_root_thetas_warm(pc, *qw);
        for (int i = 0; i < deg && i < 4; ++i) zbuf[i] = qw->z[i];
    } else {
        auto z = aberth<double>(c, deg, 40);
        th = thetas_from_complex(z);
        for (int i = 0; i < deg && i < 4; ++i) zbuf[i] = z[i];
    }
    ++w.cold_falls;
    if (prof) ++prof->rootpair_cold_falls;

    std::vector<double> tr;  // real t-roots, same filter thetas_from_complex uses
#ifdef HOLO_ARC_COMPACT_STORAGE
    tr.reserve(4);
#endif
    for (int i = 0; i < deg && i < 4; ++i)
        if (std::fabs(zbuf[i].im) <= kRootImRel * (1.0 + std::fabs(zbuf[i].re)))
            tr.push_back(zbuf[i].re);
    std::sort(tr.begin(), tr.end());

    bool seeded = false;
    if (!holo_mv_noseed() && !tr.empty() && tr.size() % 2 == 0 &&
        tr.size() == th.size()) {
        // inside arcs are the non-wrapping "even" gaps (t0,t1),(t2,t3),...
        // iff phi > 0 at the (t0,t1) t-midpoint; the "odd" set owns the
        // theta = pi (t = +-inf) arc, which (m, v) cannot represent.
        RootPairSet cand;
        bool capacity_ok = true;
        for (size_t i = 0; i + 1 < tr.size(); i += 2)
            capacity_ok = cand.push_back(
                root_pair_from_endpoints(tr[i], tr[i + 1])) && capacity_ok;
        // Accept the seed only if EVERY candidate arc is a real inside arc
        // (phi > 0 at its t-midpoint) and the family is ascending / non-
        // overlapping -- same predicate the warm step must keep satisfying.
        if (capacity_ok && transport_pairs_valid(cand, cand.size(), pc, R, pf)) {
            w.pairs = cand;
            w.pc_prev = pc;
            w.pcR_prev = boundary_quartic_dR(R, pf).p;
            w.R_prev = R;
            w.disc_sign = disc_now;
            w.valid = true;
            w.cold_streak = 0;
            seeded = true;
        }
    }
    if (!seeded) {
        w.valid = false;
        ++w.cold_streak;
        ++w.trips;
    }
    return th;
}

enum class ArcKind { kArcs, kFull, kEmpty, kDegenerate };
struct ArcSet {
    ArcKind kind;
    std::vector<std::array<double, 2>> arcs;  // (theta_enter, theta_leave)
};

// Build the physical positive-phi intervals from a certified/solved angular
// root list.  This is shared by the ordinary t chart and the reciprocal chart
// so the chart switch cannot alter the interval convention.  The fixed-grid
// sampler remains a diagnostic API and is deliberately absent here.
inline ArcSet arc_set_from_root_thetas(double R, const PrimaryFrame& pf,
                                       const std::vector<double>& th) {
    V2Profile* prof = v2_profile_current();
    if (th.empty()) {
        const double ph0 = phi_lens(R, 0.0, pf);
        if (!std::isfinite(ph0) || ph0 == 0.0) {
            if (prof) ++prof->arc_degenerate;
            return {ArcKind::kDegenerate, {}};
        }
        if (ph0 > 0.0) {
            if (prof) ++prof->arc_full;
            return {ArcKind::kFull, {}};
        }
        if (prof) ++prof->arc_empty;
        return {ArcKind::kEmpty, {}};
    }
    if (th.size() % 2 != 0) {
        if (prof) ++prof->arc_degenerate;
        return {ArcKind::kDegenerate, {}};
    }
    ArcSet out{ArcKind::kArcs, {}};
#ifdef HOLO_ARC_COMPACT_STORAGE
    out.arcs.reserve(th.size()/2);
#endif
    const int n = static_cast<int>(th.size());
    for (int i = 0; i < n; ++i) {
        const double lo = th[i];
        double hi = th[(i + 1) % n];
        if (hi <= lo) hi += kTwoPi;
        if (phi_lens(R, 0.5 * (lo + hi), pf) > 0.0)
            out.arcs.push_back({lo, hi});
    }
    if (prof) {
        ++prof->arc_sets;
        prof->arc_count += static_cast<V2Profile::u64>(out.arcs.size());
    }
    return out;
}

// ---- topology.arcs_at : grid sign-scan + bisection refine ------------
struct GridArcs {
    ArcKind kind;  // kArcs / kFull / kEmpty
    int n_crossings;
    std::vector<std::array<double, 2>> arcs;
    // These fields describe quartic_topology results.  arcs_at() remains a
    // diagnostic angular sampler and leaves them false/zero.
    bool certified = false;
    int sturm_tier = -1;  // 0=double, 1=DD, 2=__float128
    bool reciprocal_chart = false;
};
inline GridArcs arcs_at(double R, const PrimaryFrame& pf, int n_grid = 3072) {
    V2Profile* prof = v2_profile_current();
    if (prof) {
        if (n_grid == 512) ++prof->grid512_calls;
        else if (n_grid == 3072) ++prof->grid3072_calls;
        else if (n_grid == 4096) ++prof->grid4096_calls;
        prof->grid_total_nodes += (V2Profile::u64)n_grid;
    }
    V2ProfileTimer grid_timer(&V2Profile::topology_grid_ms);
    std::vector<double> val(n_grid);
    bool all_pos = true, all_neg = true;
    for (int i = 0; i < n_grid; ++i) {
        double th = kTwoPi * i / n_grid;
        double v = phi_lens(R, th, pf);
        val[i] = v;
        if (v >= 0.0)
            all_neg = false;
        else
            all_pos = false;
    }
    if (all_pos) return {ArcKind::kFull, 0, {}};
    if (all_neg) return {ArcKind::kEmpty, 0, {}};

    const double step = kTwoPi / n_grid;
    std::vector<double> roots;
    std::vector<char> rising;
    for (int i = 0; i < n_grid; ++i) {
        int j = (i + 1) % n_grid;
        bool pi = val[i] >= 0.0, pj = val[j] >= 0.0;
        if (pi == pj) continue;
        double t0 = kTwoPi * i / n_grid, t1 = t0 + step;
        double f0 = val[i], f1 = (j == 0) ? phi_lens(R, t1, pf) : val[j];
        for (int it = 0; it < 80; ++it) {
            double tm = 0.5 * (t0 + t1);
            double fm = phi_lens(R, tm, pf);
            if ((fm >= 0.0) == (f0 >= 0.0)) {
                t0 = tm;
                f0 = fm;
            } else {
                t1 = tm;
                f1 = fm;
            }
        }
        double r = std::fmod(0.5 * (t0 + t1), kTwoPi);
        if (r < 0.0) r += kTwoPi;
        roots.push_back(r);
        rising.push_back(!pi);  // neg -> pos == entering
        (void)f1;
    }
    // sort by root, keep rising flag aligned
    std::vector<int> order(roots.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return roots[a] < roots[b]; });
    std::vector<double> rs;
    std::vector<char> ri;
    for (int k : order) {
        rs.push_back(roots[k]);
        ri.push_back(rising[k]);
    }
    int n = (int)rs.size();
    if (n % 2 != 0) return {ArcKind::kArcs, n, {}};
    int start = ri.empty() || ri[0] ? 0 : -1;
    std::vector<std::array<double, 2>> arcs;
    for (int k = 0; k < n; k += 2) {
        double te = rs[((start + k) % n + n) % n];
        double tl = rs[((start + k + 1) % n + n) % n];
        arcs.push_back({te, tl});
    }
    std::sort(arcs.begin(), arcs.end(),
              [](const std::array<double, 2>& a, const std::array<double, 2>& b) {
                  return a[0] < b[0];
              });
    return {ArcKind::kArcs, n, arcs};
}

// ---- arc_intervals : from the boundary quartic ----------------------
// `w`   (optional): per-cell warm-start state for the warm Aberth solve.
// `rpw` (optional): per-cell (m, v) root-pair transport state; used by
//                   default, `w` / cold with HOLO_MV_TRANSPORT_LEGACY=1.
inline ArcSet arc_intervals(double R, const PrimaryFrame& pf,
                            QuarticWarm* w = nullptr,
                            RootPairWarm* rpw = nullptr,
                            QuarticCoeffs* out_pc = nullptr) {
    V2Profile* prof = v2_profile_current();
    if (prof) ++prof->arc_interval_calls;
    V2ProfileTimer arc_timer(&V2Profile::arc_ms);
    QuarticCoeffs q = boundary_quartic(R, pf);
    if (out_pc) *out_pc = q;
    double amax = 0.0;
    for (double c : q.p) amax = std::max(amax, std::fabs(c));
    if (amax == 0.0 || std::fabs(q.p[4]) < kP4DegenRel * amax) {
        if (w) w->valid = false;  // chart radius -> break the warm chain
        if (rpw) rpw->valid = false;
        // p4 is the coefficient of the t^4 term.  Near p4=0 the missing
        // finite t root is the projective point theta=pi, so the t chart is
        // the wrong numerical representation.  Re-solve the same quartic in
        // u=-1/t and map u roots with theta=pi+2 atan(u).  The reciprocal
        // polynomial has no angular sampling and remains valid at the exact
        // p4=0 event whenever p0 (the reciprocal leading coefficient) is
        // nonzero.
        if (q.p[0] != 0.0 && std::isfinite(q.p[0])) {
            if (prof) ++prof->arc_reciprocal_attempts;
            const QuarticCoeffs uq = boundary_quartic_reciprocal(q);
            const std::vector<double> uth = real_root_thetas_reciprocal(uq.p);
            const ArcSet reciprocal = arc_set_from_root_thetas(R, pf, uth);
            if (reciprocal.kind != ArcKind::kDegenerate) {
                if (prof) ++prof->arc_reciprocal_success;
                return reciprocal;
            }
            if (prof) ++prof->arc_reciprocal_failures;
        }
        if (prof) ++prof->arc_degenerate;
        return {ArcKind::kDegenerate, {}};
    }
    const bool use_transport = rpw && holo_mv_transport_enabled() &&
                               rpw->cold_streak < kColdStreak &&
                               rpw->trips < kTransportMaxTrips;
    auto th = use_transport ? real_root_thetas_transport(q.p, R, pf, *rpw, w)
              : w           ? real_root_thetas_warm(q.p, *w)
                            : real_root_thetas(q.p);
    return arc_set_from_root_thetas(R, pf, th);
}

// Certified cell-topology probe.  The fixed angular sampler above is kept for
// independent diagnostics only; it is never consulted here.  The D14 event
// partition removes interior discriminant crossings, so one certified
// Sturm count at the cell probe determines the angular crossing count.
inline GridArcs quartic_topology(double R, const PrimaryFrame& pf) {
    V2Profile* prof = v2_profile_current();
    if (prof) {
        ++prof->quartic_probe_calls;
        ++prof->sturm_calls;
    }
    QuarticCoeffs q = boundary_quartic(R, pf);
    QuarticSturmCertificate cert = certify_quartic(q.p);
    if (prof) {
        if (cert.precision_tier == 0) ++prof->sturm_double_accepts;
        else if (cert.precision_tier == 1) ++prof->sturm_dd_accepts;
        else if (cert.precision_tier == 2) ++prof->sturm_qf_accepts;
        if (!cert.certified) ++prof->sturm_ambiguous;
    }
    if (!cert.certified)
        return {ArcKind::kDegenerate, 0, {}, false, cert.precision_tier,
                cert.reciprocal};

#if defined(HOLO_TOPOLOGY_VERIFY_UNUSED_ROOTS)
    // Optional legacy A/B cross-check. This count-only consumer does not
    // use or return endpoint coordinates; actual radial evaluation still
    // solves its own certified arc boundaries. Keep the old redundant
    // root solve/isolation available for diagnostics.
    auto th = real_root_thetas(q.p);
    if (static_cast<int>(th.size()) != cert.root_count) {
        if (prof) ++prof->sturm_root_count_mismatch;
        const auto isolated =
            sturm_isolate_real_roots(cert.chart_coeffs, cert.root_count);
        if (static_cast<int>(isolated.size()) != cert.root_count) {
            if (prof) ++prof->sturm_isolation_failures;
            return {ArcKind::kDegenerate, 0, {}, false,
                    cert.precision_tier, cert.reciprocal};
        }
        if (prof) ++prof->sturm_isolation_repairs;
    }
#endif
    if (cert.root_count == 0) {
        const double ph0 = phi_lens(R, 0.0, pf);
        if (!std::isfinite(ph0) || ph0 == 0.0)
            return {ArcKind::kDegenerate, 0, {}, false,
                    cert.precision_tier, cert.reciprocal};
        return {ph0 > 0.0 ? ArcKind::kFull : ArcKind::kEmpty, 0, {}, true,
                cert.precision_tier, cert.reciprocal};
    }
    return {ArcKind::kArcs, cert.root_count, {}, true, cert.precision_tier,
            cert.reciprocal};
}

inline ArcSet grid_intervals(double R, const PrimaryFrame& pf) {
    GridArcs g = arcs_at(R, pf, 4096);
    if (g.kind == ArcKind::kFull) return {ArcKind::kFull, {}};
    if (g.kind == ArcKind::kEmpty) return {ArcKind::kEmpty, {}};
    ArcSet out{ArcKind::kArcs, {}};
    for (auto& arc : g.arcs) {
        double tl = arc[1];
        if (tl <= arc[0]) tl += kTwoPi;
        out.arcs.push_back({arc[0], tl});
    }
    return out;
}

// ---- full phi>0 circle : 256-point periodic rule --------------------
inline RadiusTerms full_circle_terms(double R, const PrimaryFrame& pf) {
    constexpr int M = 256;
    const double dth = kTwoPi / M;
    RadiusTerms rt;
    rt.reliable = false;
    double fh = 0.0;
    std::array<double, 5> dfh{};
    for (int i = 0; i < M; ++i) {
        double t = kTwoPi * i / M;
        PhiValDP g = phi_val_dP(R, t, pf);
        if (g.phi <= 0.0) continue;
        double sq = std::sqrt(g.phi);
        fh += sq;
        for (int j = 0; j < 5; ++j) dfh[j] += g.dP[j] / (2.0 * sq);
    }
    rt.f0 = R * kTwoPi;
    rt.fh = R * dth * fh;
    for (int j = 0; j < 5; ++j) rt.dfh[j] = R * dth * dfh[j];
    return rt;
}

// ---- radius_terms --------------------------------------------------
inline RadiusTerms radius_terms(double R, const PrimaryFrame& pf,
                                double tan_rel = kTanRel,
                                QuarticWarm* w = nullptr,
                                RootPairWarm* rpw = nullptr) {
    V2Profile* prof = v2_profile_current();
    if (prof) {
        ++prof->radius_terms_calls;
        ++prof->radial_nodes;
        ++prof->jacobian_nodes;
    }
    V2ProfileTimer jac_timer(&V2Profile::jacobian_ms);
    RadiusTerms rt;
    QuarticCoeffs arc_pc{};
    ArcSet as = arc_intervals(R, pf, w, rpw, &arc_pc);
    if (as.kind == ArcKind::kEmpty) return rt;
    if (as.kind == ArcKind::kFull) {
        if (prof) ++prof->full_circle_calls;
        return full_circle_terms(R, pf);
    }
    if (as.kind == ArcKind::kDegenerate) {
        if (prof) ++prof->arc_degenerate;
        // A chart/root ambiguity is a numerical failure, not permission to
        // replace the quartic with a fixed angular sampler.  The caller keeps
        // the result but receives reliable=false, so the epoch fails closed.
        rt.reliable = false;
        return rt;
    }

    const double tan_thresh = tan_rel * pf.rho / std::max(R, 1e-9);
    const auto& AR = ang_rule();
    const bool holo_on = holo_holonomic_transport_enabled();
    // holonomic transport works on the raw primary frame quartic P(.; R)
    const QuarticCoeffs holo_pc = holo_on ? arc_pc : QuarticCoeffs{};
    const QuarticParamJac holo_dpc =
        holo_on ? boundary_quartic_dp(R, pf) : QuarticParamJac{};
    const bool reciprocal_on = holo_on && holo_reciprocal_chart_enabled();
    const QuarticCoeffs holo_rpc =
        reciprocal_on ? boundary_quartic_reciprocal(holo_pc)
                       : QuarticCoeffs{};
    const QuarticParamJac holo_rdpc =
        reciprocal_on ? boundary_quartic_reciprocal_dp(holo_dpc)
                       : QuarticParamJac{};

    for (const auto& arc : as.arcs) {
        if (prof) {
            ++prof->f0_arcs;
        }
        auto f0_begin = V2Clock::now();
        PolishResult pe = polish_endpoint(R, arc[0], pf);
        PolishResult pl = polish_endpoint(R, arc[1], pf);
        double te = pe.theta, tl = pl.theta;
        double td = pe.dphi_dtheta, tdl = pl.dphi_dtheta;
        if (tl <= te) tl += kTwoPi;
        rt.reliable = rt.reliable && pe.reliable && pl.reliable;
        const bool arc_clean = std::fabs(td) >= tan_thresh &&
                               std::fabs(tdl) >= tan_thresh;
        if (!arc_clean) rt.reliable = false;

        rt.f0 += R * (tl - te);
        PhiValDP ge = phi_val_dP(R, te, pf);
        PhiValDP gl = phi_val_dP(R, tl, pf);
        std::array<double, 5> dte_arr{}, dtl_arr{};
        for (int j = 0; j < 5; ++j) {
            double dte = (td != 0.0) ? -ge.dP[j] / td : 0.0;
            double dtl = (tdl != 0.0) ? -gl.dP[j] / tdl : 0.0;
            dte_arr[j] = dte;
            dtl_arr[j] = dtl;
            rt.df0[j] += R * (dtl - dte);
        }
        if (prof) v2_profile_add_ms(&V2Profile::f0_ms, f0_begin,
                                    V2Clock::now());

        // ---- F_half via regularized holonomic transport ------------------
        // (2/rho) Phi_arc(R) = v K, deflated x-chart, no root solve / no
        // 64-point angular sweep.  Per-arc fail-closed: any theta = pi
        // straddle, near-tangency, negative S2, or non-finite result keeps
        // the incumbent sweep for that arc.
        if (holo_on) {
            if (!arc_clean) {
                if (prof) {
                    ++prof->k_reject_arc;
                    ++prof->k_arc_endpoint;
                }
            } else {
                ArcPairJac ap = arc_pair_jac(te, tl, dte_arr, dtl_arr);
                if (ap.ok) {
                    VKJacobian vkj =
                        v_times_K_jac(ap.m, ap.v, ap.dm, ap.dv, R, pf, holo_pc.p,
                                      holo_dpc.dp);
                    if (vkj.ok) {
                        rt.fh += vkj.vK;
                        for (int j = 0; j < 5; ++j) rt.dfh[j] += vkj.dvK[j];
                        continue;
                    }
                } else if (prof) {
                    ++prof->k_reject_arc;
                    if (ap.reject_reason == 1) ++prof->k_arc_nonfinite;
                    else if (ap.reject_reason == 2) ++prof->k_arc_order;
                    else if (ap.reject_reason == 3) ++prof->k_arc_tmax;
                    else if (ap.reject_reason == 4) ++prof->k_arc_vfloor;
                }

                // A t-chart rejection or quadrature disagreement can be a
                // representation problem at theta ~= pi.  Retry the same
                // physical arc in the exact reciprocal chart only in the
                // isolated research A/B lane.  The normal V2 path has this
                // flag off and therefore follows the byte-identical rescue.
                if (reciprocal_on) {
                    if (prof) ++prof->k_reciprocal_attempts;
                    auto reciprocal_begin = V2Clock::now();
                    ArcPairJac rap = arc_pair_jac_reciprocal(
                        te, tl, dte_arr, dtl_arr);
                    bool reciprocal_ok = false;
                    if (rap.ok) {
                        VKJacobian rvkj = v_times_K_jac(
                            rap.m, rap.v, rap.dm, rap.dv, R, pf, holo_rpc.p,
                            holo_rdpc.dp, true);
                        if (rvkj.ok) {
                            rt.fh += rvkj.vK;
                            for (int j = 0; j < 5; ++j)
                                rt.dfh[j] += rvkj.dvK[j];
                            reciprocal_ok = true;
                        }
                    }
                    if (prof) {
                        if (reciprocal_ok) ++prof->k_reciprocal_success;
                        else ++prof->k_reciprocal_reject;
                        v2_profile_add_ms(&V2Profile::k_reciprocal_ms,
                                          reciprocal_begin, V2Clock::now());
                    }
                    if (reciprocal_ok) continue;
                }
            }
        }

        double half = 0.5 * (tl - te);
        double mid = 0.5 * (te + tl);
        double acc_val = 0.0;
        std::array<double, 5> acc_der{};
        auto rescue_begin = V2Clock::now();
        for (int k = 0; k < 64; ++k) {
            double thn = mid + half * AR.x[k];
            PhiValDP g = phi_val_dP(R, thn, pf);
            if (g.phi <= 0.0) continue;
            double sq = std::sqrt(g.phi);
            double wk = AR.w[k];
            acc_val += wk * sq;
            double inv = wk / (2.0 * sq);
            for (int j = 0; j < 5; ++j) acc_der[j] += inv * g.dP[j];
        }
        if (prof) {
            ++prof->angular_rescue_calls;
            prof->angular_rescue_nodes += 64;
            v2_profile_add_ms(&V2Profile::angular_rescue_ms, rescue_begin,
                              V2Clock::now());
        }
        rt.fh += R * half * acc_val;
        for (int j = 0; j < 5; ++j) rt.dfh[j] += R * half * acc_der[j];
    }
    if (prof && !rt.reliable) ++prof->radius_unreliable;
    return rt;
}

// ---- radius_value : Phase D value-only sibling of radius_terms -----------
//
// F0 always; F_half only when `want_fh` (the linear-LD blend needs it iff
// u != 0).  No derivative accumulators, no per-endpoint IFT dtheta/dP, no
// dP evaluation at the angular nodes, no internal->user chain rule.
//
// Arc discovery is byte-identical to radius_terms: the same arc_intervals
// with the same QuarticWarm / RootPairWarm ((m,v) transport) state, so F0
// matches the fused pass to the last bit.  With transport disabled, F_half
// uses the same angular rule.  With HOLO_HOLONOMIC_TRANSPORT enabled, this
// sibling uses the same v*K evaluator as radius_terms but omits endpoint
// derivatives; the public finite-source router remains conservative and is
// unchanged.
struct RadiusValue {
    double f0 = 0.0;
    double fh = 0.0;
    bool reliable = true;
};

namespace radius_value_detail {
// Value-only mirror of full_circle_terms: F0 = R*2pi, F_half from the same
// 256-point periodic sqrt(phi) rule (phi_val == phi_val_dP().phi).
inline void full_circle_value(double R, const PrimaryFrame& pf, bool want_fh,
                              RadiusValue* rt) {
    rt->f0 = R * kTwoPi;
    rt->reliable = false;
    if (!want_fh) return;
    constexpr int M = 256;
    const double dth = kTwoPi / M;
    double fh = 0.0;
    for (int i = 0; i < M; ++i) {
        double t = kTwoPi * i / M;
        double ph = phi_val(R, t, pf);
        if (ph <= 0.0) continue;
        fh += std::sqrt(ph);
    }
    rt->fh = R * dth * fh;
}
}  // namespace radius_value_detail

inline RadiusValue radius_value(double R, const PrimaryFrame& pf, bool want_fh,
                                double tan_rel = kTanRel,
                                QuarticWarm* w = nullptr,
                                RootPairWarm* rpw = nullptr) {
    V2Profile* prof = v2_profile_current();
    if (prof) {
        ++prof->radius_value_calls;
        ++prof->radial_nodes;
    }
    V2ProfileTimer value_timer(&V2Profile::value_angular_ms);
    RadiusValue rt;
    QuarticCoeffs arc_pc{};
    ArcSet as = arc_intervals(R, pf, w, rpw, &arc_pc);
    if (as.kind == ArcKind::kEmpty) return rt;
    if (as.kind == ArcKind::kFull) {
        if (prof) ++prof->full_circle_calls;
        radius_value_detail::full_circle_value(R, pf, want_fh, &rt);
        return rt;
    }
    if (as.kind == ArcKind::kDegenerate) {
        // Keep the value/status contract fail-closed.  A fixed angular grid
        // may be retained through grid_intervals() for independent diagnostics
        // but cannot manufacture a production value at a chart ambiguity.
        rt.reliable = false;
        return rt;
    }

    const double tan_thresh = tan_rel * pf.rho / std::max(R, 1e-9);
    const auto& AR = ang_rule();
    const bool holo_on = holo_holonomic_transport_enabled();
    const bool reciprocal_on = holo_on && holo_reciprocal_chart_enabled();
    const QuarticCoeffs holo_pc = holo_on ? arc_pc : QuarticCoeffs{};
    const QuarticCoeffs holo_rpc =
        reciprocal_on ? boundary_quartic_reciprocal(holo_pc)
                       : QuarticCoeffs{};

    for (const auto& arc : as.arcs) {
        PolishResult pe = polish_endpoint(R, arc[0], pf);
        PolishResult pl = polish_endpoint(R, arc[1], pf);
        double te = pe.theta, tl = pl.theta;
        double td = pe.dphi_dtheta, tdl = pl.dphi_dtheta;
        if (tl <= te) tl += kTwoPi;
        rt.reliable = rt.reliable && pe.reliable && pl.reliable;
        const bool arc_clean = std::fabs(td) >= tan_thresh &&
                               std::fabs(tdl) >= tan_thresh;
        if (!arc_clean) rt.reliable = false;

        rt.f0 += R * (tl - te);
        if (!want_fh) continue;

        // Value-only V2 uses the same regularized v*K evaluator as the fused
        // path, but it does not form endpoint IFT derivatives or any angular
        // dP values.  The zero derivative arrays are intentional: v_times_K
        // consumes only (m, v).  A reciprocal retry is the same isolated
        // condition-driven chart experiment used by radius_terms.
        bool holo_value_ok = false;
        if (holo_on) {
            if (!arc_clean) {
                if (prof) {
                    ++prof->k_reject_arc;
                    ++prof->k_arc_endpoint;
                }
            } else {
                const std::array<double, 5> zero{};
                ArcPairJac ap = arc_pair_jac(te, tl, zero, zero);
                if (ap.ok) {
                    VKValue vk =
                        v_times_K(ap.m, ap.v, R, pf, holo_pc.p);
                    if (vk.ok) {
                        rt.fh += vk.vK;
                        holo_value_ok = true;
                    }
                } else if (prof) {
                    ++prof->k_reject_arc;
                    if (ap.reject_reason == 1) ++prof->k_arc_nonfinite;
                    else if (ap.reject_reason == 2) ++prof->k_arc_order;
                    else if (ap.reject_reason == 3) ++prof->k_arc_tmax;
                    else if (ap.reject_reason == 4) ++prof->k_arc_vfloor;
                }
                if (!holo_value_ok && reciprocal_on) {
                    if (prof) ++prof->k_reciprocal_attempts;
                    auto reciprocal_begin = V2Clock::now();
                    ArcPairJac rap = arc_pair_jac_reciprocal(
                        te, tl, zero, zero);
                    if (rap.ok) {
                        VKValue rvk = v_times_K(
                            rap.m, rap.v, R, pf, holo_rpc.p, true);
                        if (rvk.ok) {
                            rt.fh += rvk.vK;
                            holo_value_ok = true;
                        }
                    }
                    if (prof) {
                        if (holo_value_ok) ++prof->k_reciprocal_success;
                        else ++prof->k_reciprocal_reject;
                        v2_profile_add_ms(&V2Profile::k_reciprocal_ms,
                                          reciprocal_begin, V2Clock::now());
                    }
                }
            }
        }
        if (holo_value_ok) continue;

        double half = 0.5 * (tl - te);
        double mid = 0.5 * (te + tl);
        double acc_val = 0.0;
        auto rescue_begin = V2Clock::now();
        for (int k = 0; k < 64; ++k) {
            double thn = mid + half * AR.x[k];
            double ph = phi_val(R, thn, pf);
            if (ph <= 0.0) continue;
            acc_val += AR.w[k] * std::sqrt(ph);
        }
        if (prof) {
            ++prof->angular_rescue_calls;
            prof->angular_rescue_nodes += 64;
            prof->value_angular_nodes += 64;
            v2_profile_add_ms(&V2Profile::angular_rescue_ms, rescue_begin,
                              V2Clock::now());
        }
        rt.fh += R * half * acc_val;
    }
    return rt;
}

}  // namespace lcbinint::holonomic
