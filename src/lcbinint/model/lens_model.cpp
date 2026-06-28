#include "lcbinint/model/lens_model.hpp"

#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/model/orbital_motion.hpp"
#include "lcbinint/model/trajectory.hpp"
#include "lcbinint/model/triple_lens_geometry.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace lcbinint::model {
namespace {

bool has_unsupported_dynamic_effects(
    const LensParameters& params, const ComputationOptions& options)
{
    const bool unhandled_piEN_xa =
        (params.piEN_xa != 0.0 || params.piEE_xa != 0.0) &&
        options.xallarap_param_type != LCBI_XALLARAP_ORBITAL_ELEMENTS;
    const bool unhandled_xi =
        (params.xi_1 != 0.0 || params.xi_2 != 0.0) &&
        options.xallarap_param_type != LCBI_XALLARAP_ANGULAR_VELOCITY;
    return unhandled_piEN_xa || unhandled_xi ||
           params.omega != 0.0 || params.v_sep != 0.0;
}

double wide_binary_offset(
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

bool supports_binary_point_source(const LensParameters& params, const ComputationOptions& options)
{
    return !params.is_triple() && !has_unsupported_dynamic_effects(params, options) && params.rho == 0.0;
}

magnification::FiniteSourceSettings finite_source_settings(
    const LensParameters& params,
    const ComputationOptions& options)
{
    magnification::FiniteSourceSettings settings;
    settings.source_bins = options.source_bins;
    settings.caustic_bins = options.caustic_bins;
    settings.grid_ratio = options.grid_ratio;
    settings.polar_source_bins = options.polar_source_bins;
    settings.polar_grid_ratio = options.polar_grid_ratio;
    settings.finite_mode = options.mode;
    settings.kinji_threshold = options.point_source_threshold;
    settings.hex_threshold = options.hexadecapole_threshold;
    settings.adaptive_hex_threshold = options.adaptive_hex_threshold;
    settings.limb_darkening_c = params.limb_darkening_c;
    settings.limb_darkening_d = params.limb_darkening_d;
    settings.adaptive_source_bins = options.adaptive_source_bins;
    settings.max_source_bins = options.max_source_bins;
    settings.finite_source_tol = options.finite_source_tol;
    settings.finite_source_reltol = options.finite_source_reltol;
    return settings;
}

} // namespace

LensModel::LensModel(LensParameters params, ComputationOptions options)
    : params_(params)
    , options_(options)
    , trajectory_(params_)
    , cos_theta_(std::cos(params_.theta))
    , sin_theta_(std::sin(params_.theta))
    , finite_magnifier_(finite_source_settings(params_, options_))
{
}

MagnificationResult LensModel::magnification(double time) const
{
    SourcePosition source;
    const bool has_xallarap = options_.xallarap_param_type != LCBI_XALLARAP_NONE &&
        (options_.xallarap_param_type == LCBI_XALLARAP_ANGULAR_VELOCITY
            ? params_.has_xallarap_angular_velocity()
            : params_.has_xallarap_orbital_elements());
    if (params_.piEN == 0.0 && params_.piEE == 0.0 && !has_xallarap) {
        const double tn = (time - params_.t0) / params_.tE;
        const double beta = params_.umin;
        if (options_.vbm_compatible != 0) {
            source.x = tn * cos_theta_ - beta * sin_theta_;
            source.y = tn * sin_theta_ + beta * cos_theta_;
        } else {
            source.x = beta * sin_theta_ + tn * cos_theta_;
            source.y = beta * cos_theta_ - tn * sin_theta_;
        }
    } else {
        source = trajectory_.source_position(
            time, options_.vbm_compatible != 0, options_.xallarap_param_type);
    }
    const bool static_orbit = params_.orbital_motion_mode == LCBI_ORBIT_STATIC;
    const auto orbit = static_orbit ? OrbitalState {params_.sep, params_.theta, 0.0}
                                    : orbital_state(params_, time);

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

    if (params_.is_triple()) {
        if (has_unsupported_dynamic_effects(params_, options_) || !static_orbit) {
            result.status = EvaluationStatus::unsupported;
            return result;
        }
        const auto geometry = options_.vbm_compatible != 0
            ? make_triple_lens_geometry_vbm(
                params_.sep, params_.q, params_.sep2, params_.ang, params_.q2)
            : make_triple_lens_geometry(
                params_.sep, params_.q, params_.q2, params_.sep2, params_.ang);
        const auto point = point_magnifier_.triple_mag0(geometry, source);
        result.point_source_magnification = point.magnification;
        result.image_count = point.image_count;
        if (params_.rho == 0.0) {
            result.magnification = point.magnification;
            result.finite_source_magnification = point.magnification;
            result.status = std::isfinite(result.magnification)
                ? EvaluationStatus::ok
                : EvaluationStatus::numerical_error;
            return result;
        }

        const auto finite = finite_magnifier_.triple_mag(
            geometry,
            source,
            std::abs(params_.rho),
            point.magnification,
            &point_magnifier_);
        result.magnification = finite.magnification;
        result.finite_source_magnification = finite.magnification;
        result.finite_source_error_estimate = finite.error_estimate;
        result.finite_source_method = static_cast<int>(finite.decision.method);
        result.finite_source_refinement_level = finite.refinement_level;
        result.finite_source_converged = finite.converged;
        result.image_count = finite.image_count;
        result.status = std::isfinite(result.magnification)
            ? EvaluationStatus::ok
            : EvaluationStatus::numerical_error;
        return result;
    }

    if (!params_.is_triple() && !has_unsupported_dynamic_effects(params_, options_)) {
        auto source_for_magnification = source;
        if (!static_orbit) {
            if (options_.vbm_compatible != 0) {
                double tau = 0.0;
                double beta = 0.0;
                if (params_.piEN == 0.0 && params_.piEE == 0.0) {
                    tau = (time - params_.t0) / params_.tE;
                    beta = params_.umin;
                } else {
                    tau = source.x * cos_theta_ + source.y * sin_theta_;
                    beta = -source.x * sin_theta_ + source.y * cos_theta_;
                }
                source_for_magnification = {
                    tau * std::cos(orbit.angle) - beta * std::sin(orbit.angle),
                    beta * std::cos(orbit.angle) + tau * std::sin(orbit.angle),
                };
            } else {
                source_for_magnification =
                    rotate_source_to_orbital_frame(source, orbit.angle - params_.theta);
            }
        }
        result.source = source_for_magnification;
        if (options_.vbm_compatible == 0) {
            source_for_magnification.x -= wide_binary_offset(orbit.separation, params_, options_);
        }
        const double effective_q = (options_.vbm_compatible != 0 && params_.q != 0.0)
            ? 1.0 / params_.q
            : params_.q;

        if (supports_binary_point_source(params_, options_)) {
            const auto point =
                point_magnifier_.binary_mag0(orbit.separation, effective_q, source_for_magnification);
            result.point_source_magnification = point.magnification;
            result.image_count = point.image_count;
            result.magnification = point.magnification;
            result.status = std::isfinite(result.magnification)
                ? EvaluationStatus::ok
                : EvaluationStatus::numerical_error;
            return result;
        }

        const auto point_images =
            point_magnifier_.binary_images(orbit.separation, effective_q, source_for_magnification);
        double point_source_magnification = 0.0;
        std::vector<SourcePosition> center_image_seeds;
        center_image_seeds.reserve(point_images.size());
        for (const auto& image : point_images) {
            point_source_magnification += 1.0 / std::abs(image.jacobian_determinant);
            center_image_seeds.push_back(image.position);
        }
        result.point_source_magnification = point_source_magnification;
        result.image_count = static_cast<int>(point_images.size());

        const auto finite_result = finite_magnifier_.binary_mag(
            orbit.separation,
            effective_q,
            source_for_magnification, std::abs(params_.rho), point_source_magnification,
            &center_image_seeds,
            true);
        result.magnification = finite_result.magnification;
        result.finite_source_magnification = finite_result.magnification;
        result.finite_source_error_estimate = finite_result.error_estimate;
        result.finite_source_method = static_cast<int>(finite_result.decision.method);
        result.finite_source_refinement_level = finite_result.refinement_level;
        result.finite_source_converged = finite_result.converged;
        if (!std::isfinite(result.magnification)) {
            result.status = EvaluationStatus::numerical_error;
            return result;
        }
        result.status = EvaluationStatus::ok;
    }

    return result;
}

} // namespace lcbinint::model
