#include "lcbinint/lcbinint.h"

#include "lcbinint/model/lens_model.hpp"
#include "lcbinint/model/lens_parameters.hpp"

#include <cmath>

namespace {

void copy_result(const lcbinint::MagnificationResult &from, lcbi_result &to)
{
    to.magnification = from.magnification;
    to.point_source_magnification = from.point_source_magnification;
    to.finite_source_magnification = from.finite_source_magnification;
    to.source_x = from.source.x;
    to.source_y = from.source.y;
    to.image_count = from.image_count;
    to.finite_source_error_estimate = from.finite_source_error_estimate;
    to.finite_source_refinement_level = from.finite_source_refinement_level;
    to.finite_source_converged = from.finite_source_converged ? 1 : 0;
}

} // namespace

lcbi_params lcbi_default_params(void)
{
    lcbi_params params = {};
    params.tE = 1.0;
    params.q = 1.0;
    params.sep = 1.0;
    params.orbital_motion_mode = LCBI_ORBIT_STATIC;
    params.lom_ar = 1.0;
    return params;
}

lcbi_options lcbi_default_options(void)
{
    lcbi_options options = {};
    options.orbit_pair = 23;
    options.caustic_bins = 1400;
    options.source_bins = 64;
    options.mode = 1;
    options.grid_ratio = 4.0;
    options.point_source_threshold = 20.0;
    options.hexadecapole_threshold = 3.0;
    options.adaptive_hex_threshold = 0.001;
    options.adaptive_source_bins = 1;
    options.max_source_bins = 400;
    options.finite_source_reltol = 1.0e-3;
    return options;
}

lcbi_status lcbi_magnification(
    double time,
    const lcbi_params *params,
    const lcbi_options *options,
    lcbi_result *result)
{
    (void)time;
    (void)options;

    if (params == nullptr || result == nullptr) {
        return LCBI_INVALID_ARGUMENT;
    }

    const auto cpp_params = lcbinint::model::from_c_params(*params);
    if (!cpp_params.is_valid()) {
        return LCBI_INVALID_ARGUMENT;
    }

    const auto cpp_options = lcbinint::model::from_c_options(options);
    const lcbinint::model::LensModel model(cpp_params, cpp_options);
    const auto magnification_result = model.magnification(time);
    copy_result(magnification_result, *result);
    if (magnification_result.status == lcbinint::EvaluationStatus::unsupported) {
        return LCBI_UNSUPPORTED;
    }
    if (magnification_result.status == lcbinint::EvaluationStatus::numerical_error ||
        !std::isfinite(result->magnification)) {
        return LCBI_NUMERICAL_ERROR;
    }
    return LCBI_OK;
}

lcbi_status lcbi_magnification_array(
    const double *times,
    int count,
    const lcbi_params *params,
    const lcbi_options *options,
    lcbi_result *results)
{
    if (count < 0) {
        return LCBI_INVALID_ARGUMENT;
    }
    if (count == 0) {
        return LCBI_OK;
    }
    if (times == nullptr || params == nullptr || results == nullptr) {
        return LCBI_INVALID_ARGUMENT;
    }

    const auto cpp_params = lcbinint::model::from_c_params(*params);
    if (!cpp_params.is_valid()) {
        return LCBI_INVALID_ARGUMENT;
    }
    const auto cpp_options = lcbinint::model::from_c_options(options);
    const lcbinint::model::LensModel model(cpp_params, cpp_options);
    for (int i = 0; i < count; ++i) {
        const auto magnification_result = model.magnification(times[i]);
        copy_result(magnification_result, results[i]);
        if (magnification_result.status == lcbinint::EvaluationStatus::unsupported) {
            return LCBI_UNSUPPORTED;
        }
        if (magnification_result.status == lcbinint::EvaluationStatus::numerical_error ||
            !std::isfinite(results[i].magnification)) {
            return LCBI_NUMERICAL_ERROR;
        }
    }
    return LCBI_OK;
}

const char *lcbi_status_string(lcbi_status status)
{
    switch (status) {
    case LCBI_OK:
        return "ok";
    case LCBI_INVALID_ARGUMENT:
        return "invalid argument";
    case LCBI_NUMERICAL_ERROR:
        return "numerical error";
    case LCBI_UNSUPPORTED:
        return "unsupported";
    default:
        return "unknown status";
    }
}
