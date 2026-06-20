#include "lcbinint/model/lens_model.hpp"

#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/model/lens_system.hpp"
#include "lcbinint/model/orbital_motion.hpp"
#include "lcbinint/model/trajectory.hpp"

#include <cmath>
#include <limits>

namespace lcbinint::model {
namespace {

bool has_unsupported_dynamic_effects(const LensParameters& params)
{
    return params.piEN_xa != 0.0 || params.piEE_xa != 0.0 ||
           params.omega != 0.0 || params.v_sep != 0.0;
}

double legacy_wide_binary_offset(
    double separation,
    const LensParameters& params,
    const ComputationOptions& options)
{
    const double projected_separation = std::abs(separation);
    if (options.center_of_mass != 0 || projected_separation <= 1.0) {
        return 0.0;
    }
    const double q = std::abs(params.q);
    const double m2 = q / (1.0 + q);
    return m2 * projected_separation - m2 / projected_separation;
}

bool supports_binary_point_source(const LensParameters& params, const ComputationOptions& /*options*/)
{
    return !params.is_triple() && !has_unsupported_dynamic_effects(params) && params.rho == 0.0;
}

magnification::FiniteSourceSettings finite_source_settings(
    const LensParameters& params,
    const ComputationOptions& options)
{
    magnification::FiniteSourceSettings settings;
    settings.source_bins = options.source_bins;
    settings.caustic_bins = options.caustic_bins;
    settings.grid_ratio = options.grid_ratio;
    settings.finite_mode = options.mode;
    settings.kinji_threshold = options.point_source_threshold;
    settings.hex_threshold = options.hexadecapole_threshold;
    settings.adaptive_hex_threshold = options.adaptive_hex_threshold;
    settings.limb_darkening_c = params.limb_darkening_c;
    settings.limb_darkening_d = params.limb_darkening_d;
    return settings;
}

} // namespace

LensModel::LensModel(LensParameters params, ComputationOptions options)
    : params_(params), options_(options), finite_magnifier_(finite_source_settings(params_, options_))
{
}

MagnificationResult LensModel::magnification(double time) const
{
    const auto source = Trajectory(params_).source_position(time, options_.vbbl_compatible != 0);
    const auto system = LensSystem::from_parameters(params_);
    (void)system;
    const auto orbit = orbital_state(params_, time);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    MagnificationResult result;
    result.magnification = nan;
    result.point_source_magnification = nan;
    result.finite_source_magnification = nan;
    result.source = source;
    result.image_count = 0;

    if (!std::isfinite(orbit.separation) || !std::isfinite(orbit.angle)) {
        result.status = EvaluationStatus::numerical_error;
        return result;
    }

    if (!params_.is_triple() && !has_unsupported_dynamic_effects(params_)) {
        const magnification::PointSourceMagnifier point_magnifier;
        auto source_for_magnification =
            rotate_source_to_orbital_frame(source, orbit.angle - params_.theta);
        result.source = source_for_magnification;
        if (options_.vbbl_compatible == 0) {
            source_for_magnification.x -= legacy_wide_binary_offset(orbit.separation, params_, options_);
        }
        const double effective_q = (options_.vbbl_compatible != 0 && params_.q != 0.0)
            ? 1.0 / params_.q
            : params_.q;
        const auto point_result =
            point_magnifier.binary_mag0(orbit.separation, effective_q, source_for_magnification);
        result.point_source_magnification = point_result.magnification;
        result.image_count = point_result.image_count;

        if (supports_binary_point_source(params_, options_)) {
            result.magnification = point_result.magnification;
            result.status = std::isfinite(result.magnification)
                ? EvaluationStatus::ok
                : EvaluationStatus::numerical_error;
            return result;
        }

        const auto finite_result = finite_magnifier_.binary_mag(
            orbit.separation,
            effective_q,
            source_for_magnification, std::abs(params_.rho), point_result.magnification);
        result.magnification = finite_result.magnification;
        result.finite_source_magnification = finite_result.magnification;
        if (!finite_result.converged || !std::isfinite(result.magnification)) {
            result.status = EvaluationStatus::numerical_error;
            return result;
        }
        result.status = EvaluationStatus::ok;
    }

    return result;
}

} // namespace lcbinint::model
