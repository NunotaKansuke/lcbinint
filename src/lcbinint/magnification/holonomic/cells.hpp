#pragma once

// ATPT holonomic solver -- radial cell classification.
// Ports python/lcbinint/holonomic_ref/topology.py:classify_cells.
//
// The radial events partition (0, r_max] into cells of constant arc
// topology.  Each cell is probed at one interior radius.
//
// M7/M8 used fixed angular grids as a cross-check and, on disagreement,
// allowed the grid to overwrite the quartic result.  That is not a complete
// topology test: a positive arc can be narrower than every fixed grid.
//
// The cell authority is now the boundary quartic plus a certified degree-4
// Sturm real-root count at one interior radius.  D14 events already cut the
// radial discriminant crossings, so the count is constant on the open cell.
// `arcs_at` remains available as an independent diagnostic sampler, but it is
// not used for classification or as a fallback.

#include <algorithm>
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
    PositiveD14Result positive_roots{};
};

// `d14_warm` / `d14_roots_out` (optional, Phase B2 / E): forwarded to
// `radial_events` -- a previous epoch's D14 root set to warm-seed the solve,
// and an out-slot for this epoch's full root set for a prepared-geometry cache.
inline TopologyResult classify_cells(
    const PrimaryFrame& pf,
    const std::vector<Cplx<__float128>>* d14_warm = nullptr,
    std::vector<Cplx<__float128>>* d14_roots_out = nullptr,
    bool retain_adaptive_metadata = false,
    const PositiveD14Cache* positive_cache = nullptr) {
    V2Profile* prof = v2_profile_current();
    if (prof) ++prof->classify_calls;
    V2ProfileTimer topology_timer(&V2Profile::topology_ms);
    double r_max = 0.0;
    PositiveD14Result positive;
    auto events = radial_events(pf, &r_max, 1e-7, d14_warm, d14_roots_out,
                                retain_adaptive_metadata,positive_cache,&positive);

    const bool strict_boundaries=positive.assurance==PositiveRootAssurance::PositiveRealCertified;
    // merge events sharing a radius (tol max(1e-9, 1e-7 * radius))
    std::vector<double> radii;
    for (const auto& e : events) radii.push_back(e.radius);
    std::sort(radii.begin(), radii.end());
    std::vector<double> merged;
    for (double r : radii) {
        double tol = strict_boundaries ? 0.0 : std::max(1e-9, 1e-7 * r);
        if (merged.empty() || r - merged.back() > tol) merged.push_back(r);
    }

    std::vector<double> bounds;
    bounds.push_back(0.0);
    for (double r : merged)
        if (r > 0.0 && r < r_max) bounds.push_back(r);
    bounds.push_back(r_max);

    TopologyResult out;
    out.r_max = r_max;
    out.positive_roots=positive;
    out.events = std::move(events);
    Status worst = Status::OK;

    for (size_t i = 0; i + 1 < bounds.size(); ++i) {
        double lo = bounds[i], hi = bounds[i + 1];
        if (!strict_boundaries && hi - lo < 1e-11) continue;
        bool interior_certified=0.5*(lo+hi)>lo && 0.5*(lo+hi)<hi;
        if(strict_boundaries)for(const auto& e:out.events)if(e.positive_certified) {
            const __float128 mid=0.5*(lo+hi);
            if(e.radius==lo && !(mid>e.certified_radius_hi))interior_certified=false;
            if(e.radius==hi && !(mid<e.certified_radius_lo))interior_certified=false;
        }
        if (strict_boundaries && !interior_certified) {
            worst=Status::TOPOLOGY_UNCERTAIN;
            out.cells.push_back(CellPlan{(int)out.cells.size(),lo,hi,lo,ArcKind::kDegenerate,0,Status::TOPOLOGY_UNCERTAIN});
            continue;
        }
        auto probe_begin = V2Clock::now();

        // D14 has removed every radial event from the open cell.  One
        // certified midpoint Sturm count is therefore the topology oracle.
        const GridArcs mid = quartic_topology(0.5 * (lo + hi), pf);
        Status cs = Status::OK;
        if (!mid.certified || mid.n_crossings < 0 ||
            (mid.n_crossings & 1) != 0) {
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
