#pragma once

#include "lcbinint/types.hpp"

#include <array>

namespace lcbinint::model {

struct TripleLensGeometry {
    std::array<SourcePosition, 3> lens_positions {};
    std::array<double, 3> masses {};
};

TripleLensGeometry make_triple_lens_geometry(
    double separation,
    double mass_ratio,
    double secondary_mass_ratio,
    double secondary_separation,
    double secondary_angle);

SourcePosition triple_lens_equation(
    const TripleLensGeometry& geometry,
    SourcePosition image);

} // namespace lcbinint::model
