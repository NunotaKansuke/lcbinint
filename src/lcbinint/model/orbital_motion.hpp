#pragma once

#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/types.hpp"

namespace lcbinint::model {

struct OrbitalState {
    double separation = 1.0;
    double angle = 0.0;
    double line_of_sight_separation = 0.0;
};

OrbitalState circular_orbital_motion_3d(
    double time,
    double separation,
    double angle,
    double w1,
    double w2,
    double w3,
    double reference_time);

OrbitalState kepler_orbital_motion_3d(
    double time,
    double separation,
    double angle,
    double w1,
    double w2,
    double w3,
    double szs,
    double ar,
    double reference_time);

OrbitalState orbital_state(const LensParameters& params, double time);
SourcePosition rotate_source_to_orbital_frame(SourcePosition source, double angle_delta);

} // namespace lcbinint::model
