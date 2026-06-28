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

// VBM-compatible triple lens geometry.
// Accepts VBM's TripleLightCurve geometry parameters directly:
//   d12  = |v12| (binary separation, 2-body COM of lens1+lens2 at origin)
//   q    = m2/m1
//   d13  = |v13| (tertiary distance from primary lens)
//   psi  = angle of v13 from binary axis (measured at primary)
//   q2   = m3/m1
// Lens positions are the negation of VBM's positions so that the source position
// computed by lcbinint's vbm_compatible formula (which equals -VBM_source) gives
// the correct magnification: mag(-VBM_source, -VBM_lenses) = mag(VBM_source, VBM_lenses).
TripleLensGeometry make_triple_lens_geometry_vbm(
    double d12,
    double q,
    double d13,
    double psi,
    double q2);

SourcePosition triple_lens_equation(
    const TripleLensGeometry& geometry,
    SourcePosition image);

} // namespace lcbinint::model
