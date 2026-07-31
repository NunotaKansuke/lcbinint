#pragma once

#include "lcbinint/types.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace lcbinint::magnification {

// Certified support for the finite-source problem of a binary lens.
//
// Every connected component of the image set f^-1(D) of the source disk D has
// its boundary in f^-1(dD), so an image component can only be discovered from
// a source-plane point that lies in the component of D \ K (K = caustic) it
// belongs to.  Conversely, if C is a component of D \ K then either K does not
// meet the interior of D (and C is all of D), or dC contains a caustic arc.
// On any such arc the distance to the disk centre attains a local extremum
// inside D, and at that extremum the caustic is tangent to a circle about the
// centre, so the normal ray leaves the caustic on one side only.
//
// Probing both normals of every local extremum of |zeta - centre| along
// K \cap D therefore reaches every component of D \ K.  That criterion is
// derived from the caustic geometry alone: it does not depend on any source
// limb raster, integration grid, or refinement level, which is what makes it a
// completeness certificate rather than a sampling heuristic.
//
// This module deliberately performs no lens-equation solves.  It returns
// source-plane probe positions only, so the native double kernel and the JAX
// Jet kernel can consume one identical support descriptor while each keeps its
// own root solver.

// A local extremum of the distance from the source-disk centre to the caustic,
// restricted to the part of the caustic inside (or within the polyline error
// of) the disk.
struct BinaryDiskExtremum {
    SourcePosition position {};      // point on the caustic
    SourcePosition normal {};        // unit caustic normal at `position`
    double distance = 0.0;           // |position - source centre|
    double polyline_margin = 0.0;    // transverse uncertainty of the polyline
    bool inside_disk = false;        // distance < source_radius
};

// A source-plane point that the certificate claims lies strictly inside one
// component of D \ K.  `extremum` indexes BinaryDiskSupport::extrema.
struct BinaryDiskProbe {
    SourcePosition position {};
    int extremum = -1;
    double signed_offset = 0.0;  // along BinaryDiskExtremum::normal
};

struct BinaryDiskSupport {
    std::vector<BinaryDiskExtremum> extrema;
    std::vector<BinaryDiskProbe> probes;
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

BinaryDiskSupport certify_binary_disk_support(
    const std::vector<std::vector<SourcePosition>>& caustic_branches,
    SourcePosition source,
    double source_radius);

// Walks the certified probes and reports whether the support was proven.
//
// `probe` is called with a source-plane position and must return true when
// that position reached a five-image region, having consumed its images.
// Probes arrive grouped per extremum and per side, ordered coarse to fine, so
// the first success on a side ends that side: the remaining offsets can only
// re-enter the same component or one that owns a certified extremum of its
// own.
//
// A caustic arc always separates a three-image from a five-image region, so an
// extremum inside the disk with no success on either side means the component
// is thinner than the finest offset and the support is unproven.  The double
// kernel and the Jet kernel share this traversal so that a support descriptor
// can never mean two different things to them.
template <typename ProbeFn>
bool resolve_certified_probes(const BinaryDiskSupport& support, ProbeFn&& probe)
{
    if (support.extrema.empty()) {
        return !support.unresolved;
    }
    std::vector<unsigned char> resolved(2 * support.extrema.size(), 0);
    for (const auto& candidate : support.probes) {
        if (candidate.extremum < 0 ||
            static_cast<std::size_t>(candidate.extremum) >= support.extrema.size()) {
            continue;
        }
        const std::size_t slot = 2 * static_cast<std::size_t>(candidate.extremum) +
            (candidate.signed_offset >= 0.0 ? 0 : 1);
        if (resolved[slot]) {
            continue;
        }
        if (probe(candidate.position)) {
            resolved[slot] = 1;
        }
    }
    bool proven = !support.unresolved;
    for (std::size_t i = 0; i < support.extrema.size(); ++i) {
        if (support.extrema[i].inside_disk && !resolved[2 * i] && !resolved[2 * i + 1]) {
            proven = false;
        }
    }
    return proven;
}

} // namespace lcbinint::magnification
