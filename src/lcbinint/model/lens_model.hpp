#pragma once

#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/model/trajectory.hpp"
#include "lcbinint/types.hpp"

namespace lcbinint::model {

class LensModel {
public:
    LensModel(LensParameters params, ComputationOptions options);

    MagnificationResult magnification(double time) const;

private:
    LensParameters params_;
    ComputationOptions options_;
    Trajectory trajectory_;
    double cos_theta_ = 1.0;
    double sin_theta_ = 0.0;
    magnification::FiniteSourceMagnifier finite_magnifier_;
    mutable magnification::PointSourceMagnifier point_magnifier_;
};

} // namespace lcbinint::model
