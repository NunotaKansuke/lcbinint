#include "lcbinint/model/lens_model.hpp"

#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/model/lens_system.hpp"
#include "lcbinint/model/trajectory.hpp"

#include <cmath>
#include <limits>

namespace lcbinint::model {
namespace {

bool has_dynamic_effects(const LensParameters& params)
{
    return params.piEN != 0.0 || params.piEE != 0.0 || params.piEN_xa != 0.0 ||
           params.piEE_xa != 0.0 || params.omega != 0.0 || params.v_sep != 0.0;
}

bool requires_legacy_wide_binary_offset(const LensParameters& params, const ComputationOptions& options)
{
    return options.center_of_mass == 0 && std::abs(params.sep) > 1.0;
}

bool supports_binary_point_source(const LensParameters& params, const ComputationOptions& options)
{
    return options.is_point_source() && !params.is_triple() && !has_dynamic_effects(params) &&
           !requires_legacy_wide_binary_offset(params, options);
}

} // namespace

LensModel::LensModel(LensParameters params, ComputationOptions options)
    : params_(params), options_(options)
{
}

MagnificationResult LensModel::magnification(double time) const
{
    const auto source = Trajectory(params_).source_position(time);
    const auto system = LensSystem::from_parameters(params_);
    (void)system;

    const double nan = std::numeric_limits<double>::quiet_NaN();
    MagnificationResult result;
    result.magnification = nan;
    result.point_source_magnification = nan;
    result.finite_source_magnification = nan;
    result.source = source;
    result.image_count = 0;

    if (supports_binary_point_source(params_, options_)) {
        const magnification::PointSourceMagnifier magnifier;
        const auto point_result = magnifier.binary_mag0(params_.sep, params_.q, source);
        result.magnification = point_result.magnification;
        result.point_source_magnification = point_result.magnification;
        result.image_count = point_result.image_count;
    }

    return result;
}

} // namespace lcbinint::model
