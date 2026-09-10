#pragma once

// ATPT holonomic solver -- radial cell classification.
// Ports python/lcbinint/holonomic_ref/topology.py:classify_cells.
//
// The radial events partition (0, r_max] into cells of constant arc
// topology.  Each cell is probed at three interior radii (fractions
// 0.18, 0.50, 0.82).
//
// M7 probed all three with the 3072-point grid scanner (topology.arcs_at)
// for bit-parity with the M6 reference.  Profiling (checkpoint_M7 sec.7,
// checkpoint_algebraic_vs_holonomic sec.6) showed those probes are
// 58-83% of the solver wall-clock -- ~0.05 ms x 3 x (13..16 cells).
//
// M8 hybrid: the probe topology (kind, crossing count) comes from the
// boundary quartic's real roots (radius_terms.quartic_topology, ~85x
// cheaper -- the same polynomial the integrator itself root-finds).  It
// is cross-checked against a 512-point grid at the mid radius; on ANY
// disagreement -- crossing count, kind, non-uniformity across the three
// fractions, odd mid count -- the cell escalates to the full M7 path
// (arcs_at(3072) at all three fractions).  A cell whose two independent
// methods still disagree after escalation is marked TOPOLOGY_UNCERTAIN
// (fail closed).  So the adopted classification is either (a) confirmed
// by two independent methods, or (b) exactly the M7 3072-grid result, or
// (c) fail-closed -- never a lone unchecked quartic guess.

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"
#include "lcbinint/magnification/holonomic/status.hpp"
#include "lcbinint/magnification/holonomic/v2_profile.hpp"

namespace lcbinint::holonomic {

struct CellPlan {
    int index;
    double r_lo, r_hi, r_mid;
    ArcKind kind;       // kArcs / kFull / kEmpty
    int n_crossings;
    Status status = Status::OK;
};

struct TopologyResult {
    double r_max = 0.0;
    std::vector<CellPlan> cells;
    Status status = Status::OK;
    // The cell plan / D14 roots came from a warm-seeded reuse (L1 verbatim or
    // L2 warm-D14 recompute), so cell boundaries carry a ~1e-8..1e-13 seed
    // perturbation vs a cold solve.  Tells the integrator to cross-certify the
    // (m, v) transport continuation on thin near-caustic arcs, where that
    // perturbation is amplified ((m, v) transport, i.e. not
    // HOLO_MV_TRANSPORT_LEGACY=1).
    bool from_warm_d14 = false;
    // Retain the already computed radial event list for isolated diagnostic
    // consumers.  This avoids a second D14/event solve when they classify
    // their local work by the nearest fold, chart, or soft divisor.
    std::vector<RadialEvent> events;
};

namespace cells_detail {
inline bool same_sig(const GridArcs& a, const GridArcs& b) {
    return a.kind == b.kind && a.n_crossings == b.n_crossings;
}
}  // namespace cells_detail

// `d14_warm` / `d14_roots_out` (optional, Phase B2 / E): forwarded to
// `radial_events` -- a previous epoch's D14 root set to warm-seed the solve,
// and an out-slot for this epoch's full root set for a prepared-geometry cache.
inline TopologyResult classify_cells(
    const PrimaryFrame& pf,
    const std::vector<Cplx<__float128>>* d14_warm = nullptr,
    std::vector<Cplx<__float128>>* d14_roots_out = nullptr) {
    V2Profile* prof = v2_profile_current();
    if (prof) ++prof->classify_calls;
    V2ProfileTimer topology_timer(&V2Profile::topology_ms);
    double r_max = 0.0;
    auto events = radial_events(pf, &r_max, 1e-7, d14_warm, d14_roots_out);

    // merge events sharing a radius (tol max(1e-9, 1e-7 * radius))
    std::vector<double> radii;
    for (const auto& e : events) radii.push_back(e.radius);
    std::sort(radii.begin(), radii.end());
    std::vector<double> merged;
    for (double r : radii) {
        double tol = std::max(1e-9, 1e-7 * r);
        if (merged.empty() || r - merged.back() > tol) merged.push_back(r);
    }

    std::vector<double> bounds;
    bounds.push_back(0.0);
    for (double r : merged)
        if (r > 0.0 && r < r_max) bounds.push_back(r);
    bounds.push_back(r_max);

    TopologyResult out;
    out.r_max = r_max;
    out.events = std::move(events);
    Status worst = Status::OK;

    const double fr[3] = {0.18, 0.50, 0.82};
    for (size_t i = 0; i + 1 < bounds.size(); ++i) {
        double lo = bounds[i], hi = bounds[i + 1];
        if (hi - lo < 1e-11) continue;
        double Rp[3];
        for (int k = 0; k < 3; ++k) Rp[k] = lo + fr[k] * (hi - lo);
        auto probe_begin = V2Clock::now();

        // --- fast path: boundary-quartic topology at the 3 fractions ----
        GridArcs probe[3];
        bool quartic_ok = true;
        for (int k = 0; k < 3; ++k) {
            probe[k] = quartic_topology(Rp[k], pf);
            // quartic_topology self-escalates to arcs_at(...,3072) at a
            // p4~0 (chart) radius; such a probe already carries a grid
            // signature and needs no further check.
        }
        bool uniform = cells_detail::same_sig(probe[1], probe[0]) &&
                       cells_detail::same_sig(probe[1], probe[2]);

        // --- cross-check the mid probe against a coarse independent grid -
        GridArcs mid_grid = arcs_at(Rp[1], pf, 512);
        bool mid_agrees = cells_detail::same_sig(probe[1], mid_grid);

        if (!uniform || !mid_agrees) {
            if (prof) ++prof->topology_escalations;
            // escalate exactly to the M7 path: 3072-grid at all three.
            for (int k = 0; k < 3; ++k) probe[k] = arcs_at(Rp[k], pf, 3072);
            uniform = cells_detail::same_sig(probe[1], probe[0]) &&
                      cells_detail::same_sig(probe[1], probe[2]);
            quartic_ok = false;
        }
        (void)quartic_ok;

        const GridArcs& mid = probe[1];
        Status cs = Status::OK;
        if (!uniform || (mid.n_crossings % 2 != 0)) {
            cs = Status::TOPOLOGY_UNCERTAIN;
            worst = Status::TOPOLOGY_UNCERTAIN;
            if (prof) ++prof->topology_uncertain;
        }
        out.cells.push_back(CellPlan{(int)out.cells.size(), lo, hi,
                                     0.5 * (lo + hi), mid.kind,
                                     mid.n_crossings, cs});
        if (prof) {
            ++prof->classified_cells;
            v2_profile_add_ms(&V2Profile::topology_probe_ms, probe_begin,
                              V2Clock::now());
        }
    }
    out.status = worst;
    return out;
}

}  // namespace lcbinint::holonomic
