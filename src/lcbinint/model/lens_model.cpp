#include "lcbinint/model/lens_model.hpp"

#include "lcbinint/model/lens_system.hpp"
#include "lcbinint/model/trajectory.hpp"

#include <limits>

namespace lcbinint::model {

LensModel::LensModel(LensParameters params, ComputationOptions options)
    : params_(params), options_(options)
{
}

MagnificationResult LensModel::magnification(double time) const
{
    const auto source = Trajectory(params_).source_position(time);
    const auto system = LensSystem::from_parameters(params_);
    (void)system;
    (void)options_;

    const double nan = std::numeric_limits<double>::quiet_NaN();
    MagnificationResult result;
    result.magnification = nan;
    result.point_source_magnification = nan;
    result.finite_source_magnification = nan;
    result.source = source;
    result.image_count = 0;
    return result;
}

} // namespace lcbinint::model
