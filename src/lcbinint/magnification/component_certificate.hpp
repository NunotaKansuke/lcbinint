#pragma once

#include "lcbinint/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lcbinint::magnification {

// Certified support for the finite-source problem.
//
// Every connected component of the image set f^-1(D) of the source disk D has
// its boundary in f^-1(dD), so an image component can only be discovered from
// a source-plane point that lies in the component of D \ K (K = caustic) it
// belongs to.  Conversely, if C is a component of D \ K then either K does not
// meet the interior of D (and C is all of D), or dC contains a caustic arc.
// On any such arc the distance to the disk centre attains a local extremum
// inside D.
//
// At such an extremum the local frame of the caustic spans the plane, so C is
// entered along one of the four axis directions:
//
//   * Where K is smooth the caustic is tangent to the circle about the centre,
//     so the normal ray leaves the caustic on one side only and one of the two
//     normal directions enters C.
//   * Where K has a cusp the two branches leave the cusp tangent to a common
//     direction, so the narrow wedge between them opens along the *tangent*.
//     A normal probe misses it however fine the offset is, which is why both
//     tangent directions are probed as well.
//
// Probing all four directions at every local extremum of |zeta - centre| along
// K \cap D therefore reaches every component of D \ K.  That criterion is
// derived from the caustic geometry alone: it does not depend on any source
// limb raster, integration grid, or refinement level, which is what makes it a
// completeness certificate rather than a sampling heuristic.
//
// Nothing here is specific to a lens topology.  The argument uses only that K
// is a closed curve set and D a disk, so the binary and triple paths share one
// implementation.  The module also deliberately performs no lens-equation
// solves: it returns source-plane probe positions only, so the native double
// kernel and the JAX Jet kernel can consume one identical support descriptor
// while each keeps its own root solver.

// The four probe directions of an extremum's local frame.
enum class ProbeDirection : int {
    normal_plus = 0,
    normal_minus = 1,
    tangent_plus = 2,
    tangent_minus = 3,
};
inline constexpr int kProbeDirections = 4;

// A local extremum of the distance from the source-disk centre to the caustic,
// restricted to the part of the caustic inside (or within the polyline error
// of) the disk.
struct DiskExtremum {
    SourcePosition position {};      // point on the caustic
    SourcePosition normal {};        // unit caustic normal at `position`
    SourcePosition tangent {};       // unit caustic tangent at `position`
    double distance = 0.0;           // |position - source centre|
    double polyline_margin = 0.0;    // transverse uncertainty of the polyline
    bool inside_disk = false;        // distance < source_radius
};

// A source-plane point that the certificate claims lies strictly inside one
// component of D \ K.  `extremum` indexes DiskSupport::extrema.
struct DiskProbe {
    SourcePosition position {};
    int extremum = -1;
    ProbeDirection direction = ProbeDirection::normal_plus;
    double offset = 0.0;  // along that direction, always positive
};

struct DiskSupport {
    std::vector<DiskExtremum> extrema;
    std::vector<DiskProbe> probes;
    double min_caustic_distance = std::numeric_limits<double>::infinity();
    bool caustic_touches_disk = false;
    // The polyline resolution cannot decide whether some segment enters the
    // disk, and the probe ladder bottomed out before resolving it.  Callers
    // must not report convergence while this is set.
    bool unresolved = false;
    // Stable over evaluations with identical geometry; changes when the set of
    // certified extrema changes.  Used to assert that the value and the
    // derivative paths consumed the same support.
    std::uint64_t fingerprint = 0;
};

DiskSupport certify_disk_support(
    const std::vector<std::vector<SourcePosition>>& caustic_branches,
    SourcePosition source,
    double source_radius);

// Complete the sort/merge/probe half of the certificate when a caller has
// already found the raw radial extrema while performing another mandatory
// walk over the same caustic polylines.
DiskSupport finalize_disk_support(
    DiskSupport support,
    SourcePosition source,
    double source_radius);

// Walks the certified probes and reports whether the support was proven.
//
// `probe` is called with a source-plane position and must return the number of
// physical images there, having consumed them as seeds; a negative return
// means the position was unusable and is ignored.
//
// A caustic arc separates an n-image region from an (n+2)-image one, so an
// extremum is proven exactly when its probes observed two different image
// counts: that, and only that, witnesses that the arc was straddled and both
// adjacent components were entered.  Counting images rather than testing them
// against a fixed floor is what makes this correct for a lens whose caustics
// nest -- a triple lens can have an "outside" of six or eight images, where a
// floor test would accept a probe that never left the region it started in.
//
// Probes arrive grouped per extremum and per direction, ordered coarse to
// fine.  The count seen by an extremum's first usable probe is its baseline;
// a direction stops as soon as it departs from that baseline, because further
// offsets there can only re-enter the same component or one that owns an
// extremum of its own.  Every direction is walked, though: at a cusp the
// normals and the tangents reach different components, and the flood fill
// wants a seed in each.
//
// An extremum inside the disk whose four directions all stay on the baseline
// means the component on one side is thinner than the finest offset, and the
// support is unproven.  The double kernel and the Jet kernel share this
// traversal so that a support descriptor can never mean two different things
// to them.

// Resolve a local geometric scale only when the ordinary ladder needs it.
inline double local_support_probe_scale(const DiskSupport& support, std::size_t i)
{
    const auto& extremum = support.extrema[i];
    double local_scale2 = std::numeric_limits<double>::infinity();
    for (std::size_t other = 0; other < support.extrema.size(); ++other) {
        if (other == i) continue;
        const auto& neighbour = support.extrema[other];
        const double dx = extremum.position.x - neighbour.position.x;
        const double dy = extremum.position.y - neighbour.position.y;
        const double uncertainty = 32.0 * std::max(
            extremum.polyline_margin, neighbour.polyline_margin);
        const double separation2 = dx * dx + dy * dy;
        if (separation2 > uncertainty * uncertainty)
            local_scale2 = std::min(local_scale2, separation2);
    }
    return std::sqrt(local_scale2);
}

template <typename ProbeFn>
bool resolve_certified_probes(const DiskSupport& support, ProbeFn&& probe)
{
    if (support.extrema.empty()) {
        return !support.unresolved;
    }
    constexpr int kUnseen = -1;
    std::vector<int> baseline(support.extrema.size(), kUnseen);
    std::vector<unsigned char> departed(kProbeDirections * support.extrema.size(), 0);
    std::size_t previous_slot = std::numeric_limits<std::size_t>::max();
    int ray_step = 0;
    for (std::size_t probe_index = 0; probe_index < support.probes.size(); ++probe_index) {
        const auto& candidate = support.probes[probe_index];
        if (candidate.extremum < 0 ||
            static_cast<std::size_t>(candidate.extremum) >= support.extrema.size()) {
            continue;
        }
        const auto index = static_cast<std::size_t>(candidate.extremum);
        const std::size_t slot =
            kProbeDirections * index + static_cast<std::size_t>(candidate.direction);
        ray_step = slot == previous_slot ? ray_step + 1 : 0;
        previous_slot = slot;
        if (departed[slot]) {
            continue;
        }
        // Before spending the fine tail of a complete legacy ray, try one
        // local-scale point if even the legacy minimum can jump over a tiny
        // caustic. A witnessed transition uses the existing per-ray early-out.
        // If it fails, every original point is still evaluated unchanged.
        if (ray_step == 3 && support.extrema[index].inside_disk &&
            !support.unresolved && probe_index + 4 < support.probes.size()) {
            const auto& last = support.probes[probe_index + 4];
            if (last.extremum == candidate.extremum && last.direction == candidate.direction) {
                const auto& extremum = support.extrema[index];
                const double offset = std::max(
                    0.1 * local_support_probe_scale(support, index),
                    4.0 * extremum.polyline_margin);
                if (offset < last.offset) {
                    const int d = static_cast<int>(candidate.direction);
                    const SourcePosition direction = d < 2 ? extremum.normal : extremum.tangent;
                    const double sign = (d % 2) == 0 ? 1.0 : -1.0;
                    const int local_count = probe({
                        extremum.position.x + sign * offset * direction.x,
                        extremum.position.y + sign * offset * direction.y});
                    if (local_count >= 0 && baseline[index] != kUnseen &&
                        baseline[index] != local_count) {
                        departed[slot] = 1;
                        continue;
                    }
                }
            }
        }
        const int count = probe(candidate.position);
        if (count < 0) {
            continue;
        }
        if (baseline[index] == kUnseen) {
            baseline[index] = count;
        } else if (baseline[index] != count) {
            departed[slot] = 1;
        }
    }
    bool proven = !support.unresolved;
    for (std::size_t i = 0; i < support.extrema.size(); ++i) {
        if (!support.extrema[i].inside_disk) {
            continue;
        }
        bool straddled = false;
        for (int d = 0; d < kProbeDirections; ++d) {
            straddled =
                straddled || departed[kProbeDirections * i + static_cast<std::size_t>(d)] != 0;
        }
        if (!straddled && !support.unresolved) {
            // If neither the legacy ladder nor the single local lookahead
            // witnessed a transition, try the remaining local scales. This
            // fallback is confined to still-unproven extrema; the acceptance
            // criterion remains an observed image-count change.
            const auto& extremum = support.extrema[i];
            const double local_scale = local_support_probe_scale(support, i);
            std::array<int, kProbeDirections> offered {};
            std::array<double, kProbeDirections> safe_offset;
            safe_offset.fill(std::numeric_limits<double>::infinity());
            for (const auto& candidate : support.probes) {
                if (candidate.extremum == static_cast<int>(i)) {
                    const auto d = static_cast<std::size_t>(candidate.direction);
                    if (d < safe_offset.size()) {
                        ++offered[d];
                        safe_offset[d] = std::min(safe_offset[d], candidate.offset);
                    }
                }
            }
            const std::array<SourcePosition, kProbeDirections> directions {{
                extremum.normal, {-extremum.normal.x, -extremum.normal.y},
                extremum.tangent, {-extremum.tangent.x, -extremum.tangent.y}}};
            constexpr std::array<double, 5> fractions {{0.1, 0.03, 0.01, 0.003, 0.001}};
            // The initial pass has already supplied its seeds. The retry
            // needs only one additional local image-count witness for this
            // extremum's existing predicate. Interleave the four rays so an
            // unproductive ray cannot exhaust the fine ladder first.
            std::array<double, kProbeDirections> previous_offset = safe_offset;
            for (double fraction : fractions) {
                for (int d = 0; d < kProbeDirections; ++d) {
                    // A shorter positive step along an already offered ray
                    // stays inside the convex source disk.
                    if (offered[d] != 8 || !std::isfinite(safe_offset[d]) ||
                        !std::isfinite(local_scale))
                        continue;
                    const double offset = std::max(
                        fraction * local_scale, 4.0 * extremum.polyline_margin);
                    if (!(offset < previous_offset[d])) continue;
                    previous_offset[d] = offset;
                    const SourcePosition position {
                        extremum.position.x + offset * directions[d].x,
                        extremum.position.y + offset * directions[d].y};
                    const int count = probe(position);
                    if (count < 0) continue;
                    if (baseline[i] == kUnseen) baseline[i] = count;
                    else if (baseline[i] != count) {
                        straddled = true;
                        break;
                    }
                }
                if (straddled) break;
            }
        }
        if (!straddled) {
            proven = false;
        }
    }
    return proven;
}

} // namespace lcbinint::magnification
