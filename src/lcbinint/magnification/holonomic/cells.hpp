#pragma once

// ATPT holonomic solver (M7) -- radial cell classification.
// Ports python/lcbinint/holonomic_ref/topology.py:classify_cells.
//
// The radial events partition (0, r_max] into cells of constant arc
// topology.  Each cell is probed at three interior radii (fractions
// 0.18, 0.50, 0.82) with the grid arc scanner (topology.arcs_at); if the
// (kind, n_crossings) signature is not constant across the three probes,
// or the mid probe has an odd crossing count, the cell -- and the whole
// result -- is marked TOPOLOGY_UNCERTAIN (fail closed).

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"
#include "lcbinint/magnification/holonomic/status.hpp"

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
};

inline TopologyResult classify_cells(const PrimaryFrame& pf) {
    double r_max = 0.0;
    auto events = radial_events(pf, &r_max);

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
    Status worst = Status::OK;

    const double fr[3] = {0.18, 0.50, 0.82};
    for (size_t i = 0; i + 1 < bounds.size(); ++i) {
        double lo = bounds[i], hi = bounds[i + 1];
        if (hi - lo < 1e-11) continue;
        // Grid arc scan at 3 interior radii -- parity with the M6
        // reference (topology.classify_cells).  A cheaper quartic probe
        // (quartic_topology) shifts thin-arc classification on extreme-q
        // planetary cells and is deferred to M8 pending an independent
        // gradient cross-check.
        GridArcs probe[3];
        for (int k = 0; k < 3; ++k)
            probe[k] = arcs_at(lo + fr[k] * (hi - lo), pf, 3072);
        const GridArcs& mid = probe[1];

        bool uniform = true;
        for (int k = 1; k < 3; ++k)
            if (probe[k].kind != probe[0].kind ||
                probe[k].n_crossings != probe[0].n_crossings)
                uniform = false;
        Status cs = Status::OK;
        if (!uniform || (mid.n_crossings % 2 != 0)) {
            cs = Status::TOPOLOGY_UNCERTAIN;
            worst = Status::TOPOLOGY_UNCERTAIN;
        }
        out.cells.push_back(CellPlan{(int)out.cells.size(), lo, hi,
                                     0.5 * (lo + hi), mid.kind,
                                     mid.n_crossings, cs});
    }
    out.status = worst;
    return out;
}

}  // namespace lcbinint::holonomic
