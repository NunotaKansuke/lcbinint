#pragma once

#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/model/trajectory.hpp"
#include "lcbinint/obs/coordinates.hpp"
#include "lcbinint/types.hpp"

#include <memory>

namespace lcbinint::model {

struct MagnificationExecutionPlan {
    magnification::FiniteSourceMethod method =
        magnification::FiniteSourceMethod::point_source;
    int resolution = 0;
};

class LensModel {
public:
    LensModel(
        LensParameters params,
        ComputationOptions options,
        std::shared_ptr<const obs::Site> site = nullptr);

    MagnificationResult magnification(double time) const;
    MagnificationResult magnification(
        double time, const MagnificationExecutionPlan& plan) const;

    // Evaluate a preplanned static-binary epoch from source coordinates that
    // are already in the internal lens frame.  This deliberately bypasses
    // the trajectory/orbit coordinate path used by magnification(time,...).
    MagnificationResult magnification_source(
        SourcePosition source, const MagnificationExecutionPlan& plan) const;

    // Produce engine-neutral finite-source geometry after all trajectory and
    // orbital transformations.  This is intentionally separate from the
    // magnification result so an external caller can consume a whole
    // trajectory's geometry in one call without root-solving.
    bool finite_source_geometry(
        double time,
        magnification::FiniteSourceGeometry& output) const;

private:
    MagnificationResult magnification_impl(
        double time, const MagnificationExecutionPlan* plan) const;
    LensParameters params_;
    ComputationOptions options_;
    Trajectory trajectory_;
    std::shared_ptr<const obs::Site> site_;
    double cos_theta_ = 1.0;
    double sin_theta_ = 0.0;
    std::shared_ptr<magnification::FiniteSourceMagnifier> finite_magnifier_;
    std::shared_ptr<magnification::PointSourceMagnifier> point_magnifier_;
};

} // namespace lcbinint::model
