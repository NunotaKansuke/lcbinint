#pragma once

// ATPT holonomic solver -- prepared-epoch geometry cache (Phase E) with the
// Phase B2 all-root D14 warm-start layered on top.
//
// The user's final architecture (checkpoint sec.0.7 reorder + sec.6-8):
// pay the D14 oracle ONCE at the first epoch of a trajectory / proposal,
// freeze the result as an immutable `PreparedEpochGeometry`, and on every
// later epoch decide -- cheaply, fail closed -- between
//
//     cached cells  ->  compensated dd warm polish  ->  dd cold solve
//                   ->  legacy __float128 fallback
//
//   L1  reuse the cached cell plan verbatim               (no D14 at all)
//   L2  rebuild D14 coeffs, solve WARM-SEEDED by the cached 14 roots
//   L3  cold recompute (classify_cells, the authority)
//
// The decision is the sec.8 ladder: a cheap drift pre-filter selects a
// reuse *candidate*, then a mandatory re-screen (per-cached-cell quartic
// topology at three interior fractions + a coarse global fold-count + r_max
// invariance) must certify it before L1 is taken.  Any disagreement or
// uncertainty drops to L2 (small drift) or L3.  This is NOT a completeness
// proof of the pre-filter -- it is an independent cheap re-derivation of
// the cached topology at the new geometry.
//
// No new public API: this is an internal header threaded through
// epoch_jacobian via a caller-owned rolling `PreparedEpochGeometry`
// (the QuarticWarm workspace precedent).  The eventual mapping onto
// MagnificationExecutionPlan / warmup() is a thin additive adapter
// (checkpoint sec.6 correspondence table).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/fast_topology.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/poly_roots.hpp"
#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"
#include "lcbinint/magnification/holonomic/status.hpp"

namespace lcbinint::holonomic {

// Conditioning / validity information captured with the frozen plan.
struct ConditioningMargins {
    double min_cell_width = 0.0;    // narrowest cached cell
    double min_abs_p4_mid = 0.0;    // min |p4(r_mid)| over cached cells
    int n_cells = 0;
    int coarse_fold_count = 0;      // disc-sign changes on the build-time scan
    double r_max = 0.0;
};

// Immutable once built (Layer 1).  Safe to share by const reference across
// evaluations; the only mutation is the trajectory driver swapping in a
// freshly built instance when a recompute happens.
struct PreparedEpochGeometry {
    PrimaryFrame anchor{};                    // geometry the plan was built at
    double r_max = 0.0;
    std::vector<CellPlan> cells;              // the quadrature panel plan
    std::vector<Cplx<__float128>> d14_roots;  // all 14 (incl. complex) -- B2 seed
    int d14_deg = 0;
    ConditioningMargins margins;
    Status status = Status::OK;

    enum Provenance {
        kColdOracle,      // built cold by classify_cells (the authority)
        kTopologyReused,  // L1: cached cells reused verbatim
        kWarmRecomputed,  // L2: D14 rebuilt, warm-seeded by the cached roots
        kColdRecomputed,  // L3: cold recompute
    } provenance = kColdOracle;

    bool valid = false;  // false => first epoch, must build cold
};

struct PreparedReuseConfig {
    // L1 (verbatim cached cell-plan reuse) is OFF by default.  The
    // bench_holonomic_trajectory audit shows the quartic-only re-screen
    // cannot certify D14's complex-root panel boundaries -- near-caustic
    // epochs leak false reuse (evidence/holonomic/prepared_geometry_trajectory.txt).
    // A cheaper-than-D14 topology / panel certificate is designated future
    // research (checkpoint sec.0 policy point 3); until it exists L1 stays
    // opt-in for experiments only.
    bool allow_topology_reuse = false;
    // L2 (classify_cells every epoch, D14 root solve warm-seeded by the
    // previous epoch's 14 roots -- Phase B2) is ON by default: classify_cells
    // stays the authority, every completeness / residual gate is intact, and
    // a stale seed fails closed to the cold solve inside solve_d14.
    bool allow_warm_d14 = true;
    double l1_drift = 0.5;   // pre-filter: attempt an L1 re-screen below this
    // Purely a perf guard -- solve_d14 fails closed on a bad seed, so there is
    // no correctness reason to bound this and the seed setup cost is a few
    // double ops.  Large default => the warm seed is used whenever roots exist.
    double l2_drift = 1e9;
};

struct PreparedReuseStats {
    long l1_topology_reuse = 0;
    long l2_warm_recompute = 0;
    long l3_cold_recompute = 0;
    long rescreen_fail = 0;    // pre-filter said "candidate", re-screen rejected
    long warm_solve_used = 0;  // D14 solve actually consumed the warm seed
};

namespace prep_detail {

// radial_events' closed-form r_max -- no root solve.
inline double prepared_rmax(const PrimaryFrame& pf) {
    const double W = std::hypot(pf.X, pf.Y) + pf.rho;
    return 0.5 * (pf.a + W + std::hypot(pf.a - W, 2.0)) + 1e-12;
}

// Coarse count of boundary-quartic discriminant sign changes over
// (eps, r_max] -- one integer that moves whenever a fold (arc pair birth /
// death) enters or leaves the radial range.  Cheap: K quartic evaluations,
// no root solve.  Used as a global invariant for the reuse re-screen.
inline int coarse_fold_count(const PrimaryFrame& pf, double r_max) {
    const int K = 192;
    const double lo = 1e-4 * r_max, hi = r_max;
    int prev = 0, changes = 0;
    for (int i = 0; i <= K; ++i) {
        const double R = lo + (hi - lo) * (double)i / K;
        const int s = fast_topo_detail::quartic_disc_sign(R, pf);
        if (s == 0) continue;
        if (prev != 0 && s != prev) ++changes;
        prev = s;
    }
    return changes;
}

inline double drift_norm(const PrimaryFrame& a, const PrimaryFrame& b) {
    const double s = 1.0 / std::max(b.rho, 1e-12);
    return s * (std::fabs(a.X - b.X) + std::fabs(a.Y - b.Y) +
                std::fabs(a.a - b.a) + std::fabs(a.m0 - b.m0) +
                std::fabs(a.rho - b.rho));
}

}  // namespace prep_detail

// Cheap mandatory re-screen of an L1 reuse candidate at the new geometry
// `now` (checkpoint sec.8 step 2).  Independent re-derivation of the cached
// topology -- not a trust of the drift pre-filter.
inline bool prepared_rescreen(const PreparedEpochGeometry& s,
                              const PrimaryFrame& now) {
    if (!s.valid || s.cells.empty()) return false;

    const double rmax_now = prep_detail::prepared_rmax(now);
    if (std::fabs(rmax_now - s.r_max) > 5e-4 * (1.0 + s.r_max)) return false;

    // Global invariant: the fold count must not have moved.
    if (prep_detail::coarse_fold_count(now, rmax_now) !=
        s.margins.coarse_fold_count)
        return false;

    // Per-cached-cell: quartic topology at three interior fractions must
    // still match the cached (kind, crossing count), and stay uniform.
    const double fr[3] = {0.2, 0.5, 0.8};
    for (const auto& c : s.cells) {
        const double w = c.r_hi - c.r_lo;
        if (!(w > 0.0)) return false;
        for (double f : fr) {
            const GridArcs g = quartic_topology(c.r_lo + f * w, now);
            ArcKind ck = c.kind;
            // classify_cells stores kDegenerate as kArcs already; a cached
            // kArcs cell may probe kDegenerate at a chart radius -- that is
            // still integrable, accept it.
            const bool kind_ok =
                g.kind == ck ||
                (ck == ArcKind::kArcs && g.kind == ArcKind::kDegenerate);
            if (!kind_ok) return false;
            if (g.kind == ArcKind::kEmpty || g.kind == ArcKind::kFull) {
                // a band that was kArcs must not have collapsed
                if (ck == ArcKind::kArcs) return false;
            }
            if (g.kind == ArcKind::kArcs && g.n_crossings != c.n_crossings)
                return false;
        }
    }
    return true;
}

// Freeze a solved (topo, root set) at geometry `pf` into a prepared cache
// entry -- fills r_max / cells / status / the B2 warm-seed root set and the
// conditioning margins the re-screen reads.
inline PreparedEpochGeometry finalize_prepared(
    const PrimaryFrame& pf, const TopologyResult& topo,
    const std::vector<Cplx<__float128>>& roots) {
    PreparedEpochGeometry s;
    s.anchor = pf;
    s.r_max = topo.r_max;
    s.cells = topo.cells;
    s.status = topo.status;
    s.d14_roots = roots;
    s.d14_deg = (int)roots.size();

    ConditioningMargins m;
    m.n_cells = (int)s.cells.size();
    m.r_max = s.r_max;
    m.min_cell_width = 1e300;
    m.min_abs_p4_mid = 1e300;
    for (const auto& c : s.cells) {
        m.min_cell_width = std::min(m.min_cell_width, c.r_hi - c.r_lo);
        const double p4 = std::fabs(boundary_quartic(c.r_mid, pf).p[4]);
        m.min_abs_p4_mid = std::min(m.min_abs_p4_mid, p4);
    }
    if (s.cells.empty()) { m.min_cell_width = 0.0; m.min_abs_p4_mid = 0.0; }
    m.coarse_fold_count = prep_detail::coarse_fold_count(pf, s.r_max);
    s.margins = m;

    s.valid = true;
    return s;
}

// Build the frozen plan cold -- classify_cells is the authority.
inline PreparedEpochGeometry build_prepared_geometry(const PrimaryFrame& pf) {
    std::vector<Cplx<__float128>> roots;
    TopologyResult topo = classify_cells(pf, nullptr, &roots);
    PreparedEpochGeometry s = finalize_prepared(pf, topo, roots);
    s.provenance = PreparedEpochGeometry::kColdOracle;
    return s;
}

// The sec.8 fail-closed ladder for one epoch.  `state` is the caller-owned
// rolling prepared geometry (in/out); returns the TopologyResult the
// integrator consumes.
inline TopologyResult prepared_topology(const PrimaryFrame& pf,
                                        PreparedEpochGeometry& state,
                                        const PreparedReuseConfig& cfg,
                                        PreparedReuseStats* st) {
    auto bump = [&](long PreparedReuseStats::*f) {
        if (st) (st->*f)++;
    };

    // First epoch / no valid cache -> cold build.
    if (!state.valid) {
        state = build_prepared_geometry(pf);
        state.provenance = PreparedEpochGeometry::kColdOracle;
        bump(&PreparedReuseStats::l3_cold_recompute);
        return TopologyResult{state.r_max, state.cells, state.status};
    }

    const double drift = prep_detail::drift_norm(state.anchor, pf);

    // L1: drift pre-filter + mandatory cheap re-screen.
    if (cfg.allow_topology_reuse && drift <= cfg.l1_drift &&
        prepared_rescreen(state, pf)) {
        // Reuse the cached cell plan verbatim; the cache (and its warm-seed
        // roots) is unchanged, still anchored at the original geometry.
        state.provenance = PreparedEpochGeometry::kTopologyReused;
        bump(&PreparedReuseStats::l1_topology_reuse);
        TopologyResult r{state.r_max, state.cells, state.status};
        r.from_warm_d14 = true;
        return r;
    }
    if (cfg.allow_topology_reuse && drift <= cfg.l1_drift)
        bump(&PreparedReuseStats::rescreen_fail);

    // L2: warm-seeded D14 recompute (Phase B2).  classify_cells is still
    // the classifier / authority; only the D14 root solve is warm-started.
    if (cfg.allow_warm_d14 && drift <= cfg.l2_drift &&
        (int)state.d14_roots.size() > 0) {
        std::vector<Cplx<__float128>> fresh;
        TopologyResult topo = classify_cells(pf, &state.d14_roots, &fresh);
        PreparedEpochGeometry next = finalize_prepared(pf, topo, fresh);
        next.provenance = PreparedEpochGeometry::kWarmRecomputed;
        state = next;
        bump(&PreparedReuseStats::l2_warm_recompute);
        if (!fresh.empty()) bump(&PreparedReuseStats::warm_solve_used);
        topo.from_warm_d14 = true;
        return topo;
    }

    // L3: cold recompute (the authority).
    {
        std::vector<Cplx<__float128>> fresh;
        TopologyResult topo = classify_cells(pf, nullptr, &fresh);
        PreparedEpochGeometry next = finalize_prepared(pf, topo, fresh);
        next.provenance = PreparedEpochGeometry::kColdRecomputed;
        state = next;
        bump(&PreparedReuseStats::l3_cold_recompute);
        return topo;
    }
}

}  // namespace lcbinint::holonomic
