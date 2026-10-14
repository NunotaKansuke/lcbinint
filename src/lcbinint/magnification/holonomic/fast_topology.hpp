#pragma once

// ATPT holonomic solver -- fast band-discovery front end for classify_cells.
//
// Phase A step 4 (unified-engine implementation order, checkpoint sec.0.7 /
// sec.16.8).  Produces a TopologyResult with the SAME semantics classify_cells
// gives -- the only fields epoch_jacobian's integrator reads are per-cell
// [r_lo, r_hi] and `kind` (kEmpty -> skip, kFull -> GRADIENT_UNRELIABLE); the
// moment integrator re-derives the arc set at every quadrature radius, so a
// cell only needs valid bounds and a coarse kind.
//
// Path:
//   seeds = pimg_solve_verify<double>(pf)       (DOUBLE tier only)
//   if the double tier is not a clean odd set   ->  classify_cells(pf)
//       (extreme-q / near-caustic -- D14 authority, no quad rescue, no planner)
//   fb    = fast_bands(pf, seeds)               (seed-anchored radial march)
//   scr   = fast_bands_screen(pf, fb, seeds)    (sec.5.2 confidence screen)
//   if fb unreliable OR screen does not pass    ->  classify_cells(pf)
//       (the D14 completeness oracle -- unchanged, still the authority)
//   else  ->  subdivide each planner band at the cheaply reconstructible
//             radial_events (fold radii via the quartic discriminant sign,
//             chart_p4 radii via the p4 sign, plus R = sqrt(m0) and R = a) so
//             the fixed radial rule is not under-resolved across the merged
//             envelope (checkpoint sec.17); one CellPlan per sub-interval,
//             kind from its midpoint quartic_topology probe; a kEmpty midpoint
//             (should not happen post-screen) forces the oracle fallback.
//             The D14-complex soft boundaries are NOT reconstructible from the
//             real root structure -- with the flag on, a few geometries keep a
//             ~1e-6..1e-8 mu residual (the deferred cheaper-than-D14
//             certificate, design point 3).
//
// This never returns a result the screen did not clear; every uncertain
// geometry falls through to classify_cells, which fails closed on its own
// (TOPOLOGY_UNCERTAIN).  See checkpoint sec.5.4 for the epistemics and
// sec.16.6 for the residual (a thin fully-seed-less band the linear
// complement sweep can miss) -- the caller is expected to run the oracle as
// authority on the first epoch of a trajectory / a low-rate audit.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/fast_bands.hpp"
#include "lcbinint/magnification/holonomic/fast_bands_screen.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/point_images.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"
#include "lcbinint/magnification/holonomic/status.hpp"

namespace lcbinint::holonomic {

namespace fast_topo_detail {

// Sign of the quartic discriminant of the boundary polynomial at radius R
// (coeffs p0..p4 ascending).  The discriminant changes sign exactly when the
// count of real roots crosses between {2} and {0 or 4} -- i.e. at every
// boundary-quartic fold (a physical_real radial_event: an arc pair is born or
// dies).  Coeffs are pre-scaled by their max magnitude so the formula neither
// overflows nor loses the sign; the scale factor enters the discriminant as a
// positive power and cannot flip it.  ~30 flops, no root solve.
//   +1 disc > 0   -1 disc < 0   0 disc == 0 / all-zero coeffs
inline int quartic_disc_sign(double R, const PrimaryFrame& pf) {
    const std::array<double, 5>& p = boundary_quartic(R, pf).p;
    double sc = 0.0;
    for (double v : p) sc = std::max(sc, std::fabs(v));
    if (!(sc > 0.0)) return 0;
    const double a = p[4] / sc, b = p[3] / sc, c = p[2] / sc, d = p[1] / sc,
                 e = p[0] / sc;
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

// Sign of the boundary quartic's leading coefficient p4 at radius R.  p4
// changes sign at a chart_p4 radial_event (a boundary point crosses theta = pi;
// the t chart is ill-conditioned there, while the reciprocal chart handles a
// simple projective root and leaves a multiple contact fail-closed).  One
// boundary_quartic evaluation, no root solve.
inline int quartic_p4_sign(double R, const PrimaryFrame& pf) {
    const double p4 = boundary_quartic(R, pf).p[4];
    return p4 > 0.0 ? 1 : (p4 < 0.0 ? -1 : 0);
}

// Bisect [a, b] (a and b straddle a sign change of f) to ~1e-13 relative and
// return the crossing radius.  f is quartic_disc_sign or quartic_p4_sign.
template <class F>
inline double bisect_sign(double a, double b, int sa, F f,
                          const PrimaryFrame& pf) {
    for (int it = 0; it < 48 && (b - a) > 1e-13 * (1.0 + b); ++it) {
        const double m = 0.5 * (a + b);
        const int sm = f(m, pf);
        if (sm == 0) return m;
        if (sm == sa)
            a = m;
        else
            b = m;
    }
    return 0.5 * (a + b);
}

// classify_cells places a quadrature-panel boundary at every radial_event
// inside an image band (folds, chart crossings, R = sqrt(m0), R = a, and the
// D14-complex soft boundaries).  The fast planner emits only the outer
// image-region ENVELOPE, so a single 64-node Gauss-Chebyshev panel is then
// spread across many classify_cells panels and the radial quadrature is
// under-resolved (~1e-3 relative on mu; checkpoint sec.17).  This recovers the
// cheaply-reconstructible cuts -- fold radii (disc sign), chart_p4 radii (p4
// sign), and R = sqrt(m0), R = a -- WITHOUT a D14 solve.  The D14-complex soft
// boundaries are not reconstructible from the real root structure and remain
// the deferred "cheaper-than-D14 root-structure certificate" (design point 3):
// a handful of geometries keep a ~1e-6..1e-8 residual with the flag on.
inline void panel_cuts(double lo, double hi, const PrimaryFrame& pf,
                       std::vector<double>& cuts) {
    const double span = hi - lo;
    if (!(span > 0.0)) return;
    const double eps = std::max(1e-12, 1e-7 * (1.0 + hi));

    for (double rr : {std::sqrt(pf.m0), pf.a})
        if (rr > lo + eps && rr < hi - eps) cuts.push_back(rr);

    // Cheap scan: disc sign (folds) and p4 sign (chart crossings).  Start a
    // hair inside each end -- the envelope endpoints are folds themselves and
    // land on disc == 0.
    const int K = 128;
    const double s0 = lo + 1e-6 * span, s1 = hi - 1e-6 * span;
    int pds = quartic_disc_sign(s0, pf), pps = quartic_p4_sign(s0, pf);
    double pr = s0;
    for (int i = 1; i <= K; ++i) {
        const double R = s0 + (s1 - s0) * (double)i / K;
        const int ds = quartic_disc_sign(R, pf);
        const int ps = quartic_p4_sign(R, pf);
        if (ds != 0 && pds != 0 && ds != pds) {
            const double cut = bisect_sign(pr, R, pds, quartic_disc_sign, pf);
            if (cut > lo + eps && cut < hi - eps) cuts.push_back(cut);
        }
        if (ps != 0 && pps != 0 && ps != pps) {
            const double cut = bisect_sign(pr, R, pps, quartic_p4_sign, pf);
            if (cut > lo + eps && cut < hi - eps) cuts.push_back(cut);
        }
        if (ds != 0) pds = ds;
        if (ps != 0) pps = ps;
        pr = R;
    }

    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end(),
                           [&](double x, double y) {
                               return std::fabs(x - y) <
                                      std::max(1e-10, 1e-8 * (1.0 + hi));
                           }),
               cuts.end());
}

}  // namespace fast_topo_detail

struct FastTopologyStats {
    bool used_fast = false;     // fast band set was adopted
    bool fell_back = false;     // dropped to classify_cells (bail or screen fail)
    const char* reason = "";    // why the fallback fired
    int n_bands = 0;            // planner bands adopted (fast path only)
    int screen_probe_calls = 0;
};

// classify_cells-compatible topology via the fast planner + sec.5.2 screen,
// with a full classify_cells fallback on any uncertainty.
inline TopologyResult classify_cells_fast(const PrimaryFrame& pf,
                                          FastTopologyStats* st = nullptr) {
    FastTopologyStats local;
    FastTopologyStats& s = st ? *st : local;

    // Seed solve, DOUBLE tier only.  A clean odd (3/5) image set with worst
    // verified residual inside 1e-9 is the two-tier solve's own reliability
    // predicate (point_images.hpp).  When the double tier does NOT clear it
    // -- always because it dropped the planet-image pair, count -> 2 (probe
    // dbg_cnt: every escalated case on the 108-bench) -- the geometry is an
    // extreme-q / near-caustic configuration and D14 is the authority
    // (checkpoint sec.16.4 constraint a / sec.16.5).  Route straight to the
    // oracle WITHOUT the __float128 rescue and WITHOUT the planner: the quad
    // Aberth pass is pure overhead in front of a full classify_cells call
    // (that overhead was the sec.16.4 blended-tail regression).
    PointImages seeds;
    const double w_d =
        pimg_detail::pimg_solve_verify<double>(pf, 1e-9, &seeds);
    if (!(w_d <= 1e-9)) {
        s.fell_back = true;
        s.reason = "point-source double tier not a clean odd set (fragile geometry)";
        return classify_cells(pf);
    }
    seeds.reliable = true;

    FastBands fb = fast_bands(pf, seeds);
    if (!fb.reliable) {
        s.fell_back = true;
        s.reason = fb.reason;
        return classify_cells(pf);
    }

    FastBandsScreen scr = fast_bands_screen(pf, fb, seeds);
    s.screen_probe_calls = scr.probe_calls;
    if (!scr.pass) {
        s.fell_back = true;
        s.reason = scr.reason;
        return classify_cells(pf);
    }

    // Build the cell list from the planner bands.  Each band is the outer
    // image-region envelope; classify_cells would subdivide it at every
    // radial_event to place quadrature panels, so re-introduce the cheaply
    // reconstructible cuts (folds, chart crossings, sqrt(m0), a) here --
    // otherwise the fixed 64-node radial rule is under-resolved across the
    // merged span (checkpoint sec.17).  One quartic_topology probe per
    // resulting sub-cell fixes the coarse kind (kFull -> the integrator flags
    // GRADIENT_UNRELIABLE, matching classify_cells; kEmpty -> bail).
    TopologyResult out;
    double r_max = 0.0;
    std::vector<double> edges;
    for (const auto& b : fb.bands) {
        edges.clear();
        edges.push_back(b.r_lo);
        fast_topo_detail::panel_cuts(b.r_lo, b.r_hi, pf, edges);
        edges.push_back(b.r_hi);
        for (size_t i = 1; i < edges.size(); ++i) {
            const double lo = edges[i - 1], hi = edges[i];
            if (!(hi > lo)) continue;

            // Per-sub-cell validation: the boundary quartic topology at
            // fractions 0.18 / 0.50 / 0.82 must be uniform, and a second
            // certified midpoint probe must agree.  A sub-cell that fails is
            // one the planner + cheap-cut partition cannot certify.  Fail
            // closed -- hand the WHOLE epoch to the D14 oracle, never integrate
            // a lone unchecked quartic guess.  No fixed angular sampler is
            // part of this authority path.
            const double fr[3] = {0.18, 0.50, 0.82};
            GridArcs pr[3];
            for (int k = 0; k < 3; ++k)
                pr[k] = quartic_topology(lo + fr[k] * (hi - lo), pf);
            const bool uniform = pr[0].certified && pr[1].certified &&
                pr[2].certified &&
                pr[1].kind == pr[0].kind && pr[1].kind == pr[2].kind &&
                pr[1].n_crossings == pr[0].n_crossings &&
                pr[1].n_crossings == pr[2].n_crossings;
            const GridArcs mg = quartic_topology(0.5 * (lo + hi), pf);
            const bool mid_ok = mg.certified &&
                                mg.kind == pr[1].kind &&
                                mg.n_crossings == pr[1].n_crossings;
            if (!uniform || !mid_ok) {
                s.fell_back = true;
                s.reason = "fast sub-cell topology not certified (razor / fold)";
                return classify_cells(pf);
            }

            const double rmid = 0.5 * (lo + hi);
            ArcKind kind = pr[1].kind;
            if (kind == ArcKind::kEmpty) {
                s.fell_back = true;
                s.reason = "planner band probed kEmpty at a sub-cell midpoint";
                return classify_cells(pf);
            }
            // kDegenerate (theta=pi chart singularity) integrates like kArcs --
            // radius_terms handles it per node; only kEmpty/kFull are special.
            if (kind == ArcKind::kDegenerate) kind = ArcKind::kArcs;
            out.cells.push_back(CellPlan{(int)out.cells.size(), lo, hi, rmid,
                                         kind, pr[1].n_crossings, Status::OK});
        }
        if (b.r_hi > r_max) r_max = b.r_hi;
    }
    out.r_max = r_max;
    out.status = Status::OK;
    s.used_fast = true;
    s.n_bands = (int)out.cells.size();
    return out;
}

// Process-wide opt-in: HOLO_FAST_PLANNER=1 routes flux_jacobian's band
// discovery through classify_cells_fast.  Read once.
inline bool holo_fast_planner_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HOLO_FAST_PLANNER");
        return e && e[0] == '1';
    }();
    return on;
}

}  // namespace lcbinint::holonomic
