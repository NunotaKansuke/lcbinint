#pragma once

#include "lcbinint/types.hpp"
#include "lcbinint/math/filtered_ball.hpp"
#include <array>

namespace lcbinint::magnification {

enum class DiscriminantStatus {
    resolved,
    invalid_geometry,
    symmetry_axis,
    coordinate_degeneracy,
    uncertain_determinant,
    unsupported_arithmetic,
};

struct BinaryImageCount {
    int image_count = -1;
    DiscriminantStatus status = DiscriminantStatus::invalid_geometry;
    bool filtered = false;
};

// Count only, never root positions or a completeness certificate. Coordinates
// match PointSourceMagnifier's native binary input (after trajectory transforms).
// A positive/negative discriminant of the real quintic means five/three images
// only off axis and away from the coordinate reduction's W=0 degeneracy.
// A center-radius sign filter precedes endpoint intervals. Uncertain signs
// are unresolved; callers must fall back. allow_filter=false is an audit path.
BinaryImageCount binary_image_count_discriminant(
    double separation, double mass_ratio, SourcePosition source, bool allow_filter = true);

// One inexpensive enclosure only. No interval/elimination retry: an
// unresolved result is intended to go straight to the existing root solver.
BinaryImageCount binary_image_count_caustic_filter(
    double separation,double mass_ratio,SourcePosition source);
// Uniform count over a rounding enclosure of masses and source origin.
// This closes the exact-input/native-mapper distinction for seed acceptance.
BinaryImageCount binary_image_count_rounded_geometry(
    double separation,double mass_ratio,SourcePosition source);
struct BinaryCausticGradient {
    math::FilteredBall x{},y{};
    bool valid=false;
};

// K restricted to a source ray, with a constant rectangular residual tube.
// All coefficients include geometry, substitution, and tube rounding errors.
// A nonzero enclosure over [first,last] certifies the whole segment family.
struct BinaryCausticRay {
    std::array<math::FilteredBall,17> coefficients{};
    bool valid=false;
    BinaryImageCount count(math::FilteredBall parameter) const;
};
BinaryCausticRay binary_caustic_ray(double separation,double mass_ratio,
    SourcePosition first,SourcePosition last,SourcePosition residual_radius);

struct BinarySourceBox {
    SourcePosition lower{},upper{};
    int image_count=-1;
    int cache_hits=0,enclosure_evaluations=0;
};
// Dyadic source cells; cached proofs and unresolved cells are keyed by exact
// geometry and cell coordinates. At most one new enclosure per call. A miss
// preserves the cache so later probes continue the same refinement work.
BinarySourceBox binary_source_box(double separation,double mass_ratio,
    SourcePosition source,double source_radius);
// Derivative enclosure in native source coordinates over the whole rectangle.
// Together with equal nonzero endpoint signs, a definite directional derivative
// proves a caustic-free segment, not discrete image-grid coverage.
BinaryCausticGradient binary_caustic_gradient_box(
    double separation,double mass_ratio,SourcePosition lower,SourcePosition upper);

// Enclose a complete source rectangle. A resolved result excludes both a
// physical caustic and the coordinate degeneracies throughout the rectangle.
// It certifies continuous image branches, not discrete integration coverage.
BinaryImageCount binary_image_count_box(
    double separation, double mass_ratio, SourcePosition lower, SourcePosition upper);

// Bounded subdivision of the actual segment parameter; false is unresolved.
bool binary_image_link_verified(double separation, double mass_ratio,
    SourcePosition first, SourcePosition last, int expected_count, int* boxes_used);

const char* discriminant_status_name(DiscriminantStatus status);

} // namespace lcbinint::magnification
