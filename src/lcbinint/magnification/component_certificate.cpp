#include "lcbinint/magnification/component_certificate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace lcbinint::magnification {
namespace {

// Offsets are expressed as fractions of the room available along the probe
// direction before the ray leaves the source disk, ordered coarse to fine.
// The coarse entries land in the middle of a fat component; the fine tail
// resolves a component whose thickness is a small fraction of that room, which
// is what a caustic nearly tangent to the source limb produces.  Callers may
// stop at the first offset of a given direction that yields extra images:
// further offsets there can only reach the same component or one that owns an
// extremum of its own.
constexpr std::array<double, 8> kOffsetFractions {{
    0.5, 0.25, 0.75, 0.1, 0.03, 0.01, 0.003, 0.001,
}};

// Bound on how many distance extrema a physical caustic can contribute.  A
// binary caustic has at most six cusps and a triple caustic at most ten, so a
// correct polyline cannot produce anywhere near this many; exceeding it means
// the polyline is degenerate and the support must be reported as unproven
// rather than silently truncated.
constexpr std::size_t kMaxExtrema = 64;

double distance2(SourcePosition a, SourcePosition b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Distance from `origin` along the unit vector `direction` to the circle of
// radius `radius` about `centre`.  Returns 0 when `origin` is outside.
double ray_exit_distance(
    SourcePosition origin, SourcePosition direction, SourcePosition centre, double radius)
{
    const double ox = origin.x - centre.x;
    const double oy = origin.y - centre.y;
    const double projection = ox * direction.x + oy * direction.y;
    const double discriminant =
        projection * projection - (ox * ox + oy * oy) + radius * radius;
    if (!(discriminant > 0.0)) {
        return 0.0;
    }
    return std::max(0.0, std::sqrt(discriminant) - projection);
}

void mix(std::uint64_t& state, double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    state ^= bits + 0x9e3779b97f4a7c15ULL + (state << 6) + (state >> 2);
}

} // namespace

DiskSupport certify_disk_support(
    const std::vector<std::vector<SourcePosition>>& caustic_branches,
    SourcePosition source,
    double source_radius)
{
    DiskSupport support;
    if (!(source_radius > 0.0)) {
        return support;
    }

    std::vector<double> radius;
    for (const auto& branch : caustic_branches) {
        const std::size_t count = branch.size();
        if (count < 3) {
            continue;
        }
        radius.assign(count, 0.0);
        for (std::size_t i = 0; i < count; ++i) {
            radius[i] = std::sqrt(distance2(branch[i], source));
            support.min_caustic_distance =
                std::min(support.min_caustic_distance, radius[i]);
        }

        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t prev = (i + count - 1) % count;
            const std::size_t next = (i + 1) % count;
            // Half-open comparisons keep exactly one representative of a
            // plateau; the merge pass below removes near-duplicates left by
            // polyline jitter.
            const bool minimum = radius[i] <= radius[prev] && radius[i] < radius[next];
            const bool maximum = radius[i] >= radius[prev] && radius[i] > radius[next];
            if (!minimum && !maximum) {
                continue;
            }

            // Transverse uncertainty of the chord representation.  The second
            // difference is the sagitta scale of the local arc and stays a
            // valid bound where the curvature blows up at a cusp, which is
            // exactly where a chord is least trustworthy.
            const SourcePosition curvature {
                branch[prev].x - 2.0 * branch[i].x + branch[next].x,
                branch[prev].y - 2.0 * branch[i].y + branch[next].y,
            };
            const double margin = std::hypot(curvature.x, curvature.y);
            if (radius[i] >= source_radius + margin) {
                continue;
            }

            const SourcePosition chord {
                branch[next].x - branch[prev].x,
                branch[next].y - branch[prev].y,
            };
            const double chord_length = std::hypot(chord.x, chord.y);
            if (!(chord_length > 0.0)) {
                continue;
            }

            DiskExtremum extremum;
            extremum.position = branch[i];
            extremum.tangent = {chord.x / chord_length, chord.y / chord_length};
            extremum.normal = {-extremum.tangent.y, extremum.tangent.x};
            extremum.distance = radius[i];
            extremum.polyline_margin = margin;
            extremum.inside_disk = radius[i] < source_radius;
            support.extrema.push_back(extremum);
        }
    }

    if (support.extrema.empty()) {
        return support;
    }
    support.caustic_touches_disk = true;

    // Merge extrema that the polyline resolution cannot separate.  Two
    // genuinely distinct components are always separated by a caustic arc
    // longer than the local sampling step, so merging within one step cannot
    // discard a component.
    std::sort(support.extrema.begin(), support.extrema.end(),
              [](const DiskExtremum& a, const DiskExtremum& b) {
                  return a.distance < b.distance;
              });
    std::vector<DiskExtremum> merged;
    for (const auto& extremum : support.extrema) {
        bool duplicate = false;
        const double tolerance = std::max(extremum.polyline_margin, 1.0e-14);
        for (const auto& kept : merged) {
            if (distance2(kept.position, extremum.position) < tolerance * tolerance) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            merged.push_back(extremum);
        }
    }
    if (merged.size() > kMaxExtrema) {
        merged.resize(kMaxExtrema);
        support.unresolved = true;
    }
    support.extrema = std::move(merged);

    for (std::size_t index = 0; index < support.extrema.size(); ++index) {
        const auto& extremum = support.extrema[index];
        const std::array<std::pair<ProbeDirection, SourcePosition>, kProbeDirections>
            directions {{
                {ProbeDirection::normal_plus, extremum.normal},
                {ProbeDirection::normal_minus,
                 {-extremum.normal.x, -extremum.normal.y}},
                // A cusp's wedge opens along the tangent, so a normal probe
                // never enters it.  See the header for why the four axis
                // directions of the local frame are exhaustive.
                {ProbeDirection::tangent_plus, extremum.tangent},
                {ProbeDirection::tangent_minus,
                 {-extremum.tangent.x, -extremum.tangent.y}},
            }};
        for (const auto& [label, direction] : directions) {
            SourcePosition origin = extremum.position;
            if (!extremum.inside_disk) {
                // The chord says the extremum is outside, but only by less
                // than the polyline can resolve.  Probe inwards from the limb
                // point facing it: if the true caustic does enter the disk it
                // does so within `polyline_margin` of that point.
                const double norm = std::sqrt(distance2(extremum.position, source));
                if (!(norm > 0.0)) {
                    continue;
                }
                origin = {
                    source.x + (extremum.position.x - source.x) / norm * source_radius,
                    source.y + (extremum.position.y - source.y) / norm * source_radius,
                };
            }
            const double room = extremum.inside_disk
                ? ray_exit_distance(origin, direction, source, source_radius)
                : extremum.polyline_margin;
            if (!(room > 0.0)) {
                continue;
            }
            for (const double fraction : kOffsetFractions) {
                const double offset = fraction * room;
                SourcePosition probe {
                    origin.x + offset * direction.x,
                    origin.y + offset * direction.y,
                };
                if (!extremum.inside_disk) {
                    // Step off the limb towards the centre as well, so the
                    // probe is strictly interior whichever way the direction
                    // points.
                    probe.x -= (probe.x - source.x) * (offset / source_radius);
                    probe.y -= (probe.y - source.y) * (offset / source_radius);
                }
                if (distance2(probe, source) >= source_radius * source_radius) {
                    continue;
                }
                support.probes.push_back(
                    {probe, static_cast<int>(index), label, offset});
            }
        }
    }

    std::uint64_t fingerprint = 0x243f6a8885a308d3ULL;
    mix(fingerprint, source_radius);
    for (const auto& extremum : support.extrema) {
        mix(fingerprint, extremum.position.x);
        mix(fingerprint, extremum.position.y);
        mix(fingerprint, extremum.distance);
        mix(fingerprint, extremum.inside_disk ? 1.0 : 0.0);
    }
    support.fingerprint = fingerprint;
    return support;
}

} // namespace lcbinint::magnification
