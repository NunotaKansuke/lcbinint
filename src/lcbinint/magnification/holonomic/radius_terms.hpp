#pragma once

// ATPT holonomic solver (M7) -- one radius: value + derivative integrands.
// Ports python/lcbinint/holonomic_ref/jacobian.py:
//   polish_endpoint, _real_root_thetas, arc_intervals, _grid_intervals,
//   _full_circle_terms, radius_terms
// and topology.arcs_at (grid fallback / cell kind).
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
#include "lcbinint/magnification/holonomic/root_pair.hpp"

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
                                    const PrimaryFrame& pf, int iters = 6) {
    double th = theta, dth = 1.0;
    for (int i = 0; i < iters; ++i) {
        PhiValDtheta g = phi_val_dtheta(R, th, pf);
        dth = g.dphi_dtheta;
        if (std::fabs(dth) < 1e-13) return {th, dth, false};
        double step = g.phi / dth;
        th -= step;
        if (std::fabs(step) < 1e-15) break;
    }
    PhiValDtheta g = phi_val_dtheta(R, th, pf);
    return {th, g.dphi_dtheta, std::fabs(g.dphi_dtheta) >= 1e-13};
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
    double c[5];
    int deg = quartic_descending(pc, c);
    if (deg <= 0) { w.valid = false; return {}; }

    if (w.valid && w.deg == deg && w.cold_streak < kColdStreak) {
        double step = 1.0;
        auto z = aberth<double>(c, deg, kWarmIters, w.z, 0.0, &step);
        if (step <= kWarmStepTol) {
            auto th = thetas_from_complex(z);
            if (w.n_real < 0 || (int)th.size() == w.n_real) {
                for (int i = 0; i < deg; ++i) w.z[i] = z[i];
                w.n_real = (int)th.size();
                w.cold_streak = 0;
                ++w.warm_hits;
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
                                               // needs the Moebius chart change
                                               // t' = (t-c)/(1+ct) (not yet
                                               // implemented) -> until then the
                                               // whole seed is refused and the
                                               // cell stays on the cold quartic.
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

// Branch-aware acceptance for a transported pair set: the pairs must still be
// the same ascending, non-overlapping family of *real inside* arcs they were
// seeded as.  Rejects a corrector basin-flip that converges to a different
// (E, O) root -- a neighbouring arc, or an outside gap -- whose (E, O)
// residual is nonetheless tiny.  Checks, per pair: v > 0 and finite; strict
// ascending order with a real t-gap to the previous arc; both endpoints
// genuine roots of P; phi > 0 at the arc's t-midpoint (a real inside arc,
// not an outside gap).
inline bool transport_pairs_valid(const std::vector<RootPair>& ps,
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
    std::vector<RootPair> pairs;        // tracked t-bounded pairs, ascending in t
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
    // ---- warm transport -------------------------------------------------
    // A discriminant sign flip since the seed means an arc pair was born or
    // died between nodes -- a topology change the (m, v) continuation cannot
    // detect (it would keep tracking its old pair count and silently drop the
    // new arc, e.g. the HOLO_MV_TRANSPORT x L2 mu ~ 2e-3 on extreme-q-planet).
    // Fall closed: the cold solve below re-seeds with the full root set.
    const int disc_now = transport_disc_sign(pc);
    if (w.valid && !w.pairs.empty() && w.disc_sign != 0 && disc_now != 0 &&
        disc_now != w.disc_sign) {
        w.valid = false;
        ++w.cold_streak;
        ++w.trips;
    }
    if (w.valid && !w.pairs.empty()) {
        const double dR = R - w.R_prev;
        std::vector<RootPair> next = w.pairs;
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
            // corrector: Newton on (E, O) = 0 at the new node
            bool conv = false;
            double det_lo = std::numeric_limits<double>::infinity();
            double detsc_lo = std::numeric_limits<double>::infinity();
            for (int it = 0; it < kTransportNewton; ++it) {
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
            if (pred_wild || !conv || !std::isfinite(rp.m) ||
                !std::isfinite(rp.v) || rp.v <= kTransportVFloor) {
                ok = false;
                break;
            }
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
            std::vector<double> merged;
            for (double x : out)
                if (merged.empty() || x - merged.back() > 1e-11)
                    merged.push_back(x);
            if ((int)merged.size() == 2 * (int)next.size()) {
                // Certify the continuation on a warm-D14-reused cell plan when
                // an arc is thin enough for the reused-boundary seed
                // perturbation to be amplified: cross-check every transported
                // theta against a cold quartic solve at this node, fall closed
                // (cold path below) on any mismatch.  This is the
                // HOLO_MV_TRANSPORT x L2-warm-D14 interaction (checkpoint 25.4).
                bool cert_ok = true;
                if (w.certify) {
                    double v_min = std::numeric_limits<double>::infinity();
                    for (const auto& rp : next) v_min = std::fmin(v_min, rp.v);
                    if (v_min < kTransportCertifyV) {
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
                        if (!cert_ok) ++w.certify_falls;
                    }
                }
                if (cert_ok) {
                    w.pairs = next;
                    w.pc_prev = pc;
                    w.pcR_prev = boundary_quartic_dR(R, pf).p;
                    w.R_prev = R;
                    ++w.warm_hits;
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

    std::vector<double> tr;  // real t-roots, same filter thetas_from_complex uses
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
        std::vector<RootPair> cand;
        cand.reserve(tr.size() / 2);
        for (size_t i = 0; i + 1 < tr.size(); i += 2)
            cand.push_back(root_pair_from_endpoints(tr[i], tr[i + 1]));
        // Accept the seed only if EVERY candidate arc is a real inside arc
        // (phi > 0 at its t-midpoint) and the family is ascending / non-
        // overlapping -- same predicate the warm step must keep satisfying.
        if (transport_pairs_valid(cand, cand.size(), pc, R, pf)) {
            w.pairs = std::move(cand);
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

// ---- topology.arcs_at : grid sign-scan + bisection refine ------------
struct GridArcs {
    ArcKind kind;  // kArcs / kFull / kEmpty
    int n_crossings;
    std::vector<std::array<double, 2>> arcs;
};
inline GridArcs arcs_at(double R, const PrimaryFrame& pf, int n_grid = 3072) {
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
                            RootPairWarm* rpw = nullptr) {
    QuarticCoeffs q = boundary_quartic(R, pf);
    double amax = 0.0;
    for (double c : q.p) amax = std::max(amax, std::fabs(c));
    if (amax == 0.0 || std::fabs(q.p[4]) < kP4DegenRel * amax) {
        if (w) w->valid = false;  // chart radius -> break the warm chain
        if (rpw) rpw->valid = false;
        return {ArcKind::kDegenerate, {}};
    }
    const bool use_transport = rpw && holo_mv_transport_enabled() &&
                               rpw->cold_streak < kColdStreak &&
                               rpw->trips < kTransportMaxTrips;
    auto th = use_transport ? real_root_thetas_transport(q.p, R, pf, *rpw, w)
              : w           ? real_root_thetas_warm(q.p, *w)
                            : real_root_thetas(q.p);
    if (th.empty())
        return {q.p[4] > 0.0 ? ArcKind::kFull : ArcKind::kEmpty, {}};
    if (th.size() % 2 != 0) return {ArcKind::kDegenerate, {}};
    ArcSet out{ArcKind::kArcs, {}};
    int n = (int)th.size();
    for (int i = 0; i < n; ++i) {
        double lo = th[i];
        double hi = th[(i + 1) % n];
        if (hi <= lo) hi += kTwoPi;
        if (phi_lens(R, 0.5 * (lo + hi), pf) > 0.0) out.arcs.push_back({lo, hi});
    }
    return out;
}

// Cheap cell-topology probe: (kind, crossing count) straight from the
// boundary quartic's real roots -- no 3072-point grid.  Falls back to the
// grid only at a p4~0 (degenerate) probe radius.  Used by classify_cells.
inline GridArcs quartic_topology(double R, const PrimaryFrame& pf) {
    QuarticCoeffs q = boundary_quartic(R, pf);
    double amax = 0.0;
    for (double c : q.p) amax = std::max(amax, std::fabs(c));
    if (amax == 0.0 || std::fabs(q.p[4]) < kP4DegenRel * amax)
        return arcs_at(R, pf, 3072);
    auto th = real_root_thetas(q.p);
    if (th.empty())
        return {q.p[4] > 0.0 ? ArcKind::kFull : ArcKind::kEmpty, 0, {}};
    return {ArcKind::kArcs, (int)th.size(), {}};
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
    RadiusTerms rt;
    ArcSet as = arc_intervals(R, pf, w, rpw);
    if (as.kind == ArcKind::kEmpty) return rt;
    if (as.kind == ArcKind::kFull) return full_circle_terms(R, pf);
    if (as.kind == ArcKind::kDegenerate) {
        ArcSet g = grid_intervals(R, pf);
        rt.reliable = false;
        if (g.kind == ArcKind::kEmpty) return rt;
        if (g.kind == ArcKind::kFull) {
            RadiusTerms f = full_circle_terms(R, pf);
            f.reliable = false;
            return f;
        }
        as = g;
    }

    const double tan_thresh = tan_rel * pf.rho / std::max(R, 1e-9);
    const auto& AR = ang_rule();
    const bool holo_on = holo_holonomic_transport_enabled();
    // holonomic transport works on the raw primary frame quartic P(.; R)
    const QuarticCoeffs holo_pc =
        holo_on ? boundary_quartic(R, pf) : QuarticCoeffs{};
    const QuarticParamJac holo_dpc =
        holo_on ? boundary_quartic_dp(R, pf) : QuarticParamJac{};

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

        // ---- F_half via regularized holonomic transport ------------------
        // (2/rho) Phi_arc(R) = v K, deflated x-chart, no root solve / no
        // 64-point angular sweep.  Per-arc fail-closed: any theta = pi
        // straddle, near-tangency, negative S2, or non-finite result keeps
        // the incumbent sweep for that arc.
        if (holo_on && arc_clean) {
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
            }
        }

        double half = 0.5 * (tl - te);
        double mid = 0.5 * (te + tl);
        double acc_val = 0.0;
        std::array<double, 5> acc_der{};
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
        rt.fh += R * half * acc_val;
        for (int j = 0; j < 5; ++j) rt.dfh[j] += R * half * acc_der[j];
    }
    return rt;
}

}  // namespace lcbinint::holonomic
