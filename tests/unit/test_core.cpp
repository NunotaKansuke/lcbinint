#include "lcbinint/lcbinint.h"
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/math/polynomial_roots.hpp"

#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

namespace {

bool close_to_zero(lcbinint::Complex value)
{
    return std::abs(value) < 1e-9;
}

bool all_roots_satisfy(
    const std::vector<lcbinint::Complex>& coefficients,
    const std::vector<lcbinint::Complex>& roots)
{
    for (const auto& root : roots) {
        if (!close_to_zero(lcbinint::math::PolynomialRootSolver::evaluate(coefficients, root))) {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    lcbi_params params = lcbi_default_params();
    lcbi_options options = lcbi_default_options();
    lcbi_result result = {};

    if (params.tE != 1.0) {
        return 1;
    }
    if (params.orbital_motion_mode != LCBI_ORBIT_STATIC || std::abs(params.lom_ar - 1.0) > 1e-12) {
        return 37;
    }
    if (options.finite_source_mode != LCBI_SOURCE_AUTO) {
        return 2;
    }
    if (options.inverse_ray_method != LCBI_INVERSE_RAY_AUTO) {
        return 28;
    }
    if (options.caustic_bins != 1400 || options.legacy_finite_mode != 4 ||
        std::abs(options.legacy_kinji - 9.0) > 1e-12 ||
        std::abs(options.legacy_hex - 2.0) > 1e-12) {
        return 31;
    }
    if (lcbi_magnification(0.0, &params, &options, &result) != LCBI_OK) {
        return 3;
    }
    if (std::abs(result.source_x) > 1e-12 || std::abs(result.source_y) > 1e-12) {
        return 4;
    }
    params.umin = 0.1;
    params.theta = 0.0;
    params.q = 0.1;
    params.sep = 1.0;
    if (lcbi_magnification(0.2, &params, &options, &result) != LCBI_OK) {
        return 5;
    }
    if (std::abs(result.source_x - 0.2) > 1e-12 || std::abs(result.source_y - 0.1) > 1e-12) {
        return 6;
    }
    if (std::abs(result.magnification - 5.871444912771214) > 1e-10) {
        return 7;
    }
    params.sep = 1.5;
    params.q = 1.0;
    if (lcbi_magnification(0.2, &params, &options, &result) != LCBI_OK) {
        return 17;
    }
    if (std::abs(result.magnification - 3.5659775904852786) > 1e-10) {
        return 18;
    }
    params.sep = 1.0;
    params.q = 0.1;
    params.umin = 1.1;
    params.rho = 0.001;
    options.center_of_mass = 1;
    if (lcbi_magnification(1.2, &params, &options, &result) != LCBI_OK) {
        return 19;
    }
    if (!std::isfinite(result.finite_source_magnification)) {
        return 20;
    }
    if (std::strcmp(lcbi_status_string(LCBI_UNSUPPORTED), "unsupported") != 0) {
        return 23;
    }

    lcbinint::math::PolynomialRootSolver solver;
    auto linear = solver.solve({-2.0, 1.0});
    if (linear.status != lcbinint::math::RootSolverStatus::ok || linear.roots.size() != 1) {
        return 8;
    }
    if (std::abs(linear.roots[0] - lcbinint::Complex(2.0, 0.0)) > 1e-12) {
        return 9;
    }

    auto quadratic = solver.solve({-1.0, 0.0, 1.0});
    if (quadratic.status != lcbinint::math::RootSolverStatus::ok || quadratic.roots.size() != 2) {
        return 10;
    }
    if (!close_to_zero(lcbinint::math::PolynomialRootSolver::evaluate({-1.0, 0.0, 1.0}, quadratic.roots[0]))) {
        return 11;
    }
    if (!close_to_zero(lcbinint::math::PolynomialRootSolver::evaluate({-1.0, 0.0, 1.0}, quadratic.roots[1]))) {
        return 12;
    }

    const std::vector<lcbinint::Complex> cubic_coefficients = {-1.0, 0.0, 0.0, 1.0};
    auto cubic = solver.solve(cubic_coefficients);
    if (cubic.status != lcbinint::math::RootSolverStatus::ok || cubic.roots.size() != 3) {
        return 13;
    }
    if (!all_roots_satisfy(cubic_coefficients, cubic.roots)) {
        return 14;
    }

    const std::vector<lcbinint::Complex> fifth_coefficients = {
        lcbinint::Complex(-1.0, 0.25),
        lcbinint::Complex(0.5, -0.75),
        lcbinint::Complex(-1.0, 0.0),
        lcbinint::Complex(0.25, 0.5),
        lcbinint::Complex(-0.25, 0.0),
        lcbinint::Complex(1.0, 0.0),
    };
    auto fifth = solver.solve(fifth_coefficients);
    if (fifth.status != lcbinint::math::RootSolverStatus::ok || fifth.roots.size() != 5) {
        return 15;
    }
    if (!all_roots_satisfy(fifth_coefficients, fifth.roots)) {
        return 16;
    }

    lcbinint::magnification::FiniteSourceSettings finite_settings;
    finite_settings.tolerance = 1.0e-3;
    finite_settings.source_bins = 20;
    finite_settings.grid_ratio = 4.0;
    lcbinint::magnification::FiniteSourceMagnifier finite_magnifier(finite_settings);
    auto finite_decision = finite_magnifier.choose_binary_method({0.01, 0.0}, 0.01, 20.0);
    if (finite_decision.method != lcbinint::magnification::FiniteSourceMethod::inverse_ray_polar) {
        return 21;
    }
    if (finite_decision.estimated_evaluations <= 0) {
        return 22;
    }
    lcbinint::magnification::FiniteSourceMagnifier loose_finite_magnifier(finite_settings);
    auto finite_result = loose_finite_magnifier.binary_mag(1.0, 0.1, {1.2, 1.1}, 0.001, 1.0);
    if (finite_result.decision.method != lcbinint::magnification::FiniteSourceMethod::point_source) {
        return 24;
    }
    if (!std::isfinite(finite_result.error_estimate) || finite_result.error_estimate > 1.0e-3) {
        return 25;
    }
    if (!finite_result.converged) {
        return 26;
    }
    lcbi_params strict_params = lcbi_default_params();
    lcbi_options strict_options = lcbi_default_options();
    strict_params.q = 0.1;
    strict_params.sep = 1.0;
    strict_params.umin = 0.0;
    strict_params.theta = 0.0;
    strict_params.rho = 0.02;
    strict_options.center_of_mass = 1;
    strict_options.tolerance = 1.0e-14;
    strict_options.relative_tolerance = 0.0;
    strict_options.source_bins = 4;
    if (lcbi_magnification(0.45, &strict_params, &strict_options, &result) != LCBI_NUMERICAL_ERROR) {
        return 27;
    }
    strict_options.inverse_ray_method = LCBI_INVERSE_RAY_POLAR;
    if (lcbi_magnification(0.45, &strict_params, &strict_options, &result) != LCBI_NUMERICAL_ERROR) {
        return 29;
    }
    strict_options.inverse_ray_method = LCBI_INVERSE_RAY_AUTO;
    strict_options.relative_tolerance = 1.0;
    if (lcbi_magnification(0.45, &strict_params, &strict_options, &result) != LCBI_OK) {
        return 30;
    }
    lcbinint::magnification::FiniteSourceSettings legacy_settings;
    legacy_settings.tolerance = 1.0e-3;
    legacy_settings.source_bins = 20;
    legacy_settings.caustic_bins = 128;
    legacy_settings.legacy_mode = true;
    legacy_settings.legacy_finite_mode = 4;
    legacy_settings.legacy_kinji = 9.0;
    legacy_settings.legacy_hex = 2.0;
    lcbinint::magnification::FiniteSourceMagnifier legacy_finite_magnifier(legacy_settings);
    auto legacy_hex_result = legacy_finite_magnifier.binary_mag(1.0, 0.1, {0.2, 0.2}, 0.02, 1.0);
    if (legacy_hex_result.decision.method != lcbinint::magnification::FiniteSourceMethod::hexadecapole) {
        return 32;
    }
    if (!std::isfinite(legacy_hex_result.magnification) || !legacy_hex_result.converged) {
        return 33;
    }
    auto limb_darkened_settings = legacy_settings;
    limb_darkened_settings.limb_darkening_c = 0.5;
    limb_darkened_settings.limb_darkening_d = 0.2;
    lcbinint::magnification::FiniteSourceMagnifier limb_darkened_finite_magnifier(limb_darkened_settings);
    auto limb_darkened_hex_result =
        limb_darkened_finite_magnifier.binary_mag(1.0, 0.1, {0.2, 0.2}, 0.02, 1.0);
    if (limb_darkened_hex_result.decision.method != lcbinint::magnification::FiniteSourceMethod::hexadecapole) {
        return 34;
    }
    if (!std::isfinite(limb_darkened_hex_result.magnification)) {
        return 35;
    }
    if (std::abs(limb_darkened_hex_result.magnification - legacy_hex_result.magnification) < 1.0e-12) {
        return 36;
    }
    return 0;
}
