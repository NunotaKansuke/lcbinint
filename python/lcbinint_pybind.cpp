#include "lcbinint/lcbinint.h"
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/math/polynomial_roots.hpp"
#include "lcbinint/model/lens_model.hpp"
#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/model/orbital_motion.hpp"
#include "lcbinint/model/trajectory.hpp"

#include "bind_obs.hpp"
#include "bind_bayes.hpp"
#include "bind_optimize.hpp"
#include "bind_sample.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

struct PyLightCurve {
    std::vector<double> times;
    std::vector<double> magnifications;
    std::vector<double> point_source_magnifications;
    std::vector<double> finite_source_magnifications;
    std::vector<double> finite_source_error_estimates;
    std::vector<double> source_x;
    std::vector<double> source_y;
    std::vector<int> image_counts;
    std::vector<int> finite_source_methods;
    std::vector<std::string> finite_source_method_names;
    std::vector<int> finite_source_refinement_levels;
    std::vector<bool> finite_source_converged;
};

struct PyGeometryCurve {
    std::vector<double> times;
    std::vector<double> x;
    std::vector<double> y;
};

struct PyGeometryBranches {
    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;
};

bool all_converged(const PyLightCurve& curve)
{
    return std::all_of(
        curve.finite_source_converged.begin(),
        curve.finite_source_converged.end(),
        [](bool value) { return value; });
}

std::vector<int> unconverged_indices(const PyLightCurve& curve)
{
    std::vector<int> indices;
    for (std::size_t i = 0; i < curve.finite_source_converged.size(); ++i) {
        if (!curve.finite_source_converged[i]) {
            indices.push_back(static_cast<int>(i));
        }
    }
    return indices;
}

struct PySourceBinCandidate {
    int source_bins = 0;
    double max_absolute_difference = 0.0;
    double max_relative_difference = 0.0;
    double rms_relative_difference = 0.0;
    bool accepted = false;
};

struct PySourceBinEstimate {
    int reference_source_bins = 0;
    int recommended_source_bins = 0;
    std::vector<double> sampled_times;
    std::vector<PySourceBinCandidate> candidates;
};

struct PyLimbDarkening {
    double c = 0.0;
    double d = 0.0;
};

struct PyEventCoordinates {
    double ra = 0.0;
    double dec = 0.0;
    double tfix = 0.0;
    double obs_lat = 0.0;
    double obs_lon = 0.0;
};

struct PyBinaryParams {
    double t0 = 0.0;
    double tE = 1.0;
    double u0 = 0.0;
    double alpha = 0.0;
    double s = 1.0;
    double q = 1.0;
    double q2 = 0.0;
    double sep2 = 0.0;
    double ang = 0.0;
    double rho = 0.0;
    double piEN = 0.0;
    double piEE = 0.0;
    double g1 = 0.0;
    double g2 = 0.0;
    double g3 = 0.0;
    double lom_szs = 0.0;
    double lom_ar = 1.0;
    double u0_2 = 0.0;
    double t0_2 = 0.0;
    double flux_ratio = 0.0;
    double xi_1 = 0.0;
    double xi_2 = 0.0;
    double omega_xa = 0.0;
    double inc_xa = 0.0;
    double phi_xa = 0.0;
    double piEN_xa = 0.0;
    double piEE_xa = 0.0;
    double period_xa = 0.0;
    double ecc_xa = 0.0;
    double peri_xa = 0.0;
};

double dict_get_double(const py::dict& params, const char* key, double default_value)
{
    const py::str py_key(key);
    if (!params.contains(py_key)) {
        return default_value;
    }
    return py::cast<double>(params[py_key]);
}

PyBinaryParams binary_params_from_dict(const py::dict& params)
{
    PyBinaryParams out;
    out.t0 = dict_get_double(params, "t0", out.t0);
    out.tE = dict_get_double(params, "tE", out.tE);
    out.u0 = dict_get_double(params, "u0", out.u0);
    out.alpha = dict_get_double(params, "alpha", out.alpha);
    out.s = dict_get_double(params, "s", out.s);
    out.q = dict_get_double(params, "q", out.q);
    out.q2 = dict_get_double(params, "q2", out.q2);
    out.sep2 = dict_get_double(params, "sep2", out.sep2);
    out.ang = dict_get_double(params, "ang", out.ang);
    out.rho = dict_get_double(params, "rho", out.rho);
    out.piEN = dict_get_double(params, "piEN", out.piEN);
    out.piEE = dict_get_double(params, "piEE", out.piEE);
    out.g1 = dict_get_double(params, "g1", out.g1);
    out.g2 = dict_get_double(params, "g2", out.g2);
    out.g3 = dict_get_double(params, "g3", out.g3);
    out.lom_szs = dict_get_double(params, "lom_szs", out.lom_szs);
    out.lom_ar = dict_get_double(params, "lom_ar", out.lom_ar);
    out.u0_2 = dict_get_double(params, "u0_2", out.u0_2);
    out.t0_2 = dict_get_double(params, "t0_2", out.t0_2);
    out.flux_ratio = dict_get_double(params, "flux_ratio", out.flux_ratio);
    out.xi_1 = dict_get_double(params, "xi_1", out.xi_1);
    out.xi_2 = dict_get_double(params, "xi_2", out.xi_2);
    out.omega_xa = dict_get_double(params, "omega_xa", out.omega_xa);
    out.inc_xa = dict_get_double(params, "inc_xa", out.inc_xa);
    out.phi_xa = dict_get_double(params, "phi_xa", out.phi_xa);
    out.piEN_xa = dict_get_double(params, "piEN_xa", out.piEN_xa);
    out.piEE_xa = dict_get_double(params, "piEE_xa", out.piEE_xa);
    out.period_xa = dict_get_double(params, "period_xa", out.period_xa);
    out.ecc_xa = dict_get_double(params, "ecc_xa", out.ecc_xa);
    out.peri_xa = dict_get_double(params, "peri_xa", out.peri_xa);
    return out;
}

int optional_int_or(const py::object& value, int default_value)
{
    if (value.is_none()) {
        return default_value;
    }
    return py::cast<int>(value);
}

double optional_double_or(const py::object& value, double default_value)
{
    if (value.is_none()) {
        return default_value;
    }
    return py::cast<double>(value);
}

py::object optional_positive_int(int value)
{
    if (value <= 0) {
        return py::none();
    }
    return py::int_(value);
}

py::object optional_positive_double(double value)
{
    if (value <= 0.0) {
        return py::none();
    }
    return py::float_(value);
}

lcbi_options public_default_options()
{
    auto options = lcbi_default_options();
    options.center_of_mass = 0;
    options.vbm_compatible = 1;
    options.mode = 4;
    return options;
}

void apply_xallarap_param_type(lcbi_options& options, const std::string& xallarap_param_type)
{
    if (xallarap_param_type.empty() || xallarap_param_type == "none") {
        options.xallarap_param_type = LCBI_XALLARAP_NONE;
        return;
    }
    if (xallarap_param_type == "angular_velocity") {
        options.xallarap_param_type = LCBI_XALLARAP_ANGULAR_VELOCITY;
        return;
    }
    if (xallarap_param_type == "orbital_elements") {
        options.xallarap_param_type = LCBI_XALLARAP_ORBITAL_ELEMENTS;
        return;
    }
    throw std::invalid_argument(
        "xallarap_param_type must be 'none', 'angular_velocity', or 'orbital_elements'");
}

std::string xallarap_param_type_name(const lcbi_options& options)
{
    switch (options.xallarap_param_type) {
    case LCBI_XALLARAP_ANGULAR_VELOCITY:
        return "angular_velocity";
    case LCBI_XALLARAP_ORBITAL_ELEMENTS:
        return "orbital_elements";
    default:
        return "none";
    }
}

void apply_param_type(lcbi_options& options, const std::string& param_type)
{
    if (param_type == "auto" || param_type.empty()) {
        options.center_of_mass = 0;
        options.vbm_compatible = 1;
        return;
    }
    if (param_type == "vbm" || param_type == "vbbl" || param_type == "standard") {
        options.center_of_mass = 0;
        options.vbm_compatible = 1;
        return;
    }
    if (param_type == "lcbinint" || param_type == "original" || param_type == "legacy") {
        options.center_of_mass = 0;
        options.vbm_compatible = 0;
        return;
    }
    if (param_type == "center_of_mass") {
        options.center_of_mass = 1;
        options.vbm_compatible = 0;
        return;
    }
    if (param_type == "vbm_center_of_mass") {
        options.center_of_mass = 1;
        options.vbm_compatible = 1;
        return;
    }
    throw std::invalid_argument(
        "param_type must be 'vbm', 'lcbinint', 'original', or 'center_of_mass'");
}

// Backwards-compatible alias.
void apply_coordinate_system(lcbi_options& options, const std::string& coordinates)
{
    apply_param_type(options, coordinates);
}

std::string coordinate_system_name(const lcbi_options& options)
{
    if (options.vbm_compatible != 0 && options.center_of_mass == 0) {
        return "vbm";
    }
    if (options.center_of_mass != 0 && options.vbm_compatible != 0) {
        return "vbm_center_of_mass";
    }
    if (options.center_of_mass != 0) {
        return "center_of_mass";
    }
    if (options.vbm_compatible == 0 && options.center_of_mass == 0) {
        return "lcbinint";
    }
    return "custom";
}

void apply_inverse_ray_grid(lcbi_options& options, const std::string& grid)
{
    if (grid == "auto" || grid.empty()) {
        options.mode = 4;
        return;
    }
    if (grid == "cartesian" || grid == "inverse_ray_cartesian") {
        options.mode = 1;
        return;
    }
    if (grid == "polar" || grid == "inverse_ray_polar") {
        options.mode = 2;
        return;
    }
    throw std::invalid_argument(
        "inverse_ray_grid must be 'auto', 'cartesian', or 'polar'");
}

std::string inverse_ray_grid_name(const lcbi_options& options)
{
    if (options.mode == 4) {
        return "auto";
    }
    if (options.mode == 1) {
        return "cartesian";
    }
    if (options.mode == 2) {
        return "polar";
    }
    return "internal";
}

lcbi_params make_binary_params(
    double t0,
    double tE,
    double u0,
    double alpha,
    double separation,
    double mass_ratio,
    double source_radius,
    const PyLimbDarkening& limb_darkening,
    const PyEventCoordinates& event,
    double piEN,
    double piEE,
    double g1,
    double g2,
    double g3,
    lcbi_orbital_motion_mode orbital_motion_mode,
    double lom_szs,
    double lom_ar,
    double xi_1 = 0.0,
    double xi_2 = 0.0,
    double omega_xa = 0.0,
    double inc_xa = 0.0,
    double phi_xa = 0.0,
    double piEN_xa = 0.0,
    double piEE_xa = 0.0,
    double period_xa = 0.0,
    double ecc_xa = 0.0,
    double peri_xa = 0.0)
{
    auto params = lcbi_default_params();
    params.t0 = t0;
    params.tE = tE;
    params.umin = u0;
    params.theta = alpha;
    params.sep = separation;
    params.q = mass_ratio;
    params.rho = source_radius;
    params.limb_darkening_c = limb_darkening.c;
    params.limb_darkening_d = limb_darkening.d;
    params.ra = event.ra;
    params.dec = event.dec;
    params.tfix = event.tfix;
    params.obs_lat = event.obs_lat;
    params.obs_lon = event.obs_lon;
    params.piEN = piEN;
    params.piEE = piEE;
    params.g1 = g1;
    params.g2 = g2;
    params.g3 = g3;
    params.orbital_motion_mode = orbital_motion_mode;
    params.lom_szs = lom_szs;
    params.lom_ar = lom_ar;
    params.xi_1 = xi_1;
    params.xi_2 = xi_2;
    params.omega_xa = omega_xa;
    params.inc_xa = inc_xa;
    params.phi_xa = phi_xa;
    params.piEN_xa = piEN_xa;
    params.piEE_xa = piEE_xa;
    params.period_xa = period_xa;
    params.ecc_xa = ecc_xa;
    params.peri_xa = peri_xa;
    return params;
}

lcbi_params make_triple_params(
    double t0,
    double tE,
    double u0,
    double alpha,
    double separation,
    double mass_ratio,
    double secondary_mass_ratio,
    double secondary_separation,
    double secondary_angle,
    double source_radius,
    const PyLimbDarkening& limb_darkening,
    const PyEventCoordinates& event)
{
    auto params = make_binary_params(
        t0, tE, u0, alpha, separation, mass_ratio, source_radius,
        limb_darkening, event, 0.0, 0.0, 0.0, 0.0, 0.0,
        LCBI_ORBIT_STATIC, 0.0, 1.0);
    params.q2 = secondary_mass_ratio;
    params.sep2 = secondary_separation;
    params.ang = secondary_angle;
    return params;
}

bool same_finite_source_settings(
    const lcbinint::magnification::FiniteSourceSettings& a,
    const lcbinint::magnification::FiniteSourceSettings& b)
{
    return a.source_bins == b.source_bins &&
           a.caustic_bins == b.caustic_bins &&
           a.grid_ratio == b.grid_ratio &&
           a.polar_source_bins == b.polar_source_bins &&
           a.polar_grid_ratio == b.polar_grid_ratio &&
           a.finite_mode == b.finite_mode &&
           a.kinji_threshold == b.kinji_threshold &&
           a.hex_threshold == b.hex_threshold &&
           a.adaptive_hex_threshold == b.adaptive_hex_threshold &&
           a.limb_darkening_c == b.limb_darkening_c &&
           a.limb_darkening_d == b.limb_darkening_d &&
           a.adaptive_source_bins == b.adaptive_source_bins &&
           a.max_source_bins == b.max_source_bins &&
           a.finite_source_tol == b.finite_source_tol &&
           a.finite_source_reltol == b.finite_source_reltol;
}

lcbinint::magnification::FiniteSourceSettings make_finite_source_settings(
    const lcbinint::model::LensParameters& model_params,
    const lcbinint::model::ComputationOptions& model_options)
{
    lcbinint::magnification::FiniteSourceSettings settings;
    settings.source_bins = model_options.source_bins;
    settings.caustic_bins = model_options.caustic_bins;
    settings.grid_ratio = model_options.grid_ratio;
    settings.polar_source_bins = model_options.polar_source_bins;
    settings.polar_grid_ratio = model_options.polar_grid_ratio;
    settings.finite_mode = model_options.mode;
    settings.kinji_threshold = model_options.point_source_threshold;
    settings.hex_threshold = model_options.hexadecapole_threshold;
    settings.adaptive_hex_threshold = model_options.adaptive_hex_threshold;
    settings.limb_darkening_c = model_params.limb_darkening_c;
    settings.limb_darkening_d = model_params.limb_darkening_d;
    settings.adaptive_source_bins = model_options.adaptive_source_bins;
    settings.max_source_bins = model_options.max_source_bins;
    settings.finite_source_tol = model_options.finite_source_tol;
    settings.finite_source_reltol = model_options.finite_source_reltol;
    return settings;
}

lcbinint::magnification::FiniteSourceSettings make_finite_source_settings(
    const PyLimbDarkening& limb_darkening,
    const lcbinint::model::ComputationOptions& model_options)
{
    lcbinint::magnification::FiniteSourceSettings settings;
    settings.source_bins = model_options.source_bins;
    settings.caustic_bins = model_options.caustic_bins;
    settings.grid_ratio = model_options.grid_ratio;
    settings.polar_source_bins = model_options.polar_source_bins;
    settings.polar_grid_ratio = model_options.polar_grid_ratio;
    settings.finite_mode = model_options.mode;
    settings.kinji_threshold = model_options.point_source_threshold;
    settings.hex_threshold = model_options.hexadecapole_threshold;
    settings.adaptive_hex_threshold = model_options.adaptive_hex_threshold;
    settings.limb_darkening_c = limb_darkening.c;
    settings.limb_darkening_d = limb_darkening.d;
    settings.adaptive_source_bins = model_options.adaptive_source_bins;
    settings.max_source_bins = model_options.max_source_bins;
    settings.finite_source_tol = model_options.finite_source_tol;
    settings.finite_source_reltol = model_options.finite_source_reltol;
    return settings;
}

bool can_use_direct_static_binary(
    const lcbinint::model::ComputationOptions& model_options,
    lcbi_orbital_motion_mode orbital_motion_mode,
    bool parallax)
{
    return !parallax &&
           orbital_motion_mode == LCBI_ORBIT_STATIC &&
           model_options.center_of_mass == 0 &&
           model_options.vbm_compatible != 0;
}

double py_safe_sqrt(double value)
{
    return std::sqrt(std::max(0.0, value));
}

double py_dot(std::array<double, 3> lhs, std::array<double, 3> rhs)
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

std::array<double, 3> py_scale(std::array<double, 3> value, double factor)
{
    return {value[0] * factor, value[1] * factor, value[2] * factor};
}

std::array<double, 3> py_cross(std::array<double, 3> lhs, std::array<double, 3> rhs)
{
    return {
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    };
}

std::array<double, 3> py_normalize(std::array<double, 3> value)
{
    const double norm = py_safe_sqrt(py_dot(value, value));
    if (norm == 0.0 || !std::isfinite(norm)) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan, nan};
    }
    return py_scale(value, 1.0 / norm);
}

double py_solve_kepler_newton(double mean_anomaly, double eccentricity)
{
    double eccentric_anomaly = mean_anomaly + eccentricity * std::sin(mean_anomaly);
    for (int i = 0; i < 10; ++i) {
        const double f = eccentric_anomaly - eccentricity * std::sin(eccentric_anomaly) -
                         mean_anomaly;
        const double fp = 1.0 - eccentricity * std::cos(eccentric_anomaly);
        const double delta = f / fp;
        eccentric_anomaly -= delta;
        if (std::abs(delta) <= 1.0e-13) {
            break;
        }
    }
    return eccentric_anomaly;
}

double py_distance_squared(lcbinint::SourcePosition lhs, lcbinint::SourcePosition rhs)
{
    const double dx = lhs.x - rhs.x;
    const double dy = lhs.y - rhs.y;
    return dx * dx + dy * dy;
}

std::vector<lcbinint::Complex> py_critical_curve_polynomial_coefficients(
    double separation,
    double mass_ratio,
    lcbinint::Complex phase)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const lcbinint::Complex a =
        q_input < 1.0 ? lcbinint::Complex(-s, 0.0) : lcbinint::Complex(s, 0.0);
    const lcbinint::Complex m1 = 1.0 / (1.0 + q);
    const lcbinint::Complex m2 = q * m1;
    const lcbinint::Complex a2 = a * a;

    return {
        m2 * a2,
        -2.0 * m2 * a,
        m1 + m2 - phase * a2,
        2.0 * phase * a,
        -phase,
    };
}

std::vector<lcbinint::SourcePosition> py_critical_points_at_phase(
    double separation,
    double mass_ratio,
    double phase_angle)
{
    const auto phase = std::polar(1.0, phase_angle);
    lcbinint::math::PolynomialRootSolver solver;
    const auto roots =
        solver.solve(py_critical_curve_polynomial_coefficients(separation, mass_ratio, phase));
    if (roots.status != lcbinint::math::RootSolverStatus::ok) {
        return {};
    }

    std::vector<lcbinint::SourcePosition> points;
    points.reserve(roots.roots.size());
    for (const auto& root : roots.roots) {
        points.push_back({root.real(), root.imag()});
    }
    return points;
}

std::vector<lcbinint::SourcePosition> py_caustic_points_at_phase(
    const lcbinint::magnification::PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    double phase_angle)
{
    const auto roots = py_critical_points_at_phase(separation, mass_ratio, phase_angle);
    std::vector<lcbinint::SourcePosition> points;
    points.reserve(roots.size());
    for (const auto& root : roots) {
        points.push_back(point_magnifier.binary_lens_equation(separation, mass_ratio, root));
    }
    return points;
}

void py_append_tracked_points(
    std::vector<std::vector<lcbinint::SourcePosition>>& branches,
    std::vector<lcbinint::SourcePosition> points)
{
    const std::size_t n = branches.size();
    if (points.size() != n) {
        return;
    }

    if (branches[0].empty()) {
        std::sort(points.begin(), points.end(), [](const auto& lhs, const auto& rhs) {
            return std::atan2(lhs.y, lhs.x) < std::atan2(rhs.y, rhs.x);
        });
        for (std::size_t i = 0; i < n; ++i) {
            branches[i].push_back(points[i]);
        }
        return;
    }

    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<std::size_t> best_perm = perm;
    double best_cost = std::numeric_limits<double>::infinity();
    do {
        double cost = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            cost += py_distance_squared(branches[i].back(), points[perm[i]]);
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_perm = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    for (std::size_t i = 0; i < n; ++i) {
        branches[i].push_back(points[best_perm[i]]);
    }
}

PyGeometryBranches py_pack_geometry_branches(
    const std::vector<std::vector<lcbinint::SourcePosition>>& branches)
{
    PyGeometryBranches packed;
    packed.x.reserve(branches.size());
    packed.y.reserve(branches.size());
    for (const auto& branch : branches) {
        auto& xs = packed.x.emplace_back();
        auto& ys = packed.y.emplace_back();
        xs.reserve(branch.size());
        ys.reserve(branch.size());
        for (const auto& point : branch) {
            xs.push_back(point.x);
            ys.push_back(point.y);
        }
    }
    return packed;
}

// Multiply two polynomials with coefficients in constant-first order.
std::vector<lcbinint::Complex> py_poly_mul(
    const std::vector<lcbinint::Complex>& a,
    const std::vector<lcbinint::Complex>& b)
{
    std::vector<lcbinint::Complex> result(a.size() + b.size() - 1, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j)
            result[i + j] += a[i] * b[j];
    return result;
}

// Build the degree-6 critical curve polynomial for the triple lens.
// The polynomial is in w = conj(z_image), constant-first coefficients.
// Critical condition: sum_i m_i / (w - conj(z_i))^2 = e^{i*phi}
// Rearranged: e^{i*phi} * P0*P1*P2 - m0*P1*P2 - m1*P0*P2 - m2*P0*P1 = 0
// where Pi = (w - conj(z_i))^2.
std::vector<lcbinint::Complex> py_triple_critical_polynomial(
    const lcbinint::model::TripleLensGeometry& geometry,
    lcbinint::Complex phase)
{
    std::array<lcbinint::Complex, 3> w;
    for (int i = 0; i < 3; ++i) {
        w[i] = std::conj(lcbinint::Complex(
            geometry.lens_positions[i].x, geometry.lens_positions[i].y));
    }

    // Quadratic (w - wi)^2 = wi^2 - 2*wi*w + w^2  (constant-first)
    auto quad = [](lcbinint::Complex wi) -> std::vector<lcbinint::Complex> {
        return {wi * wi, -2.0 * wi, 1.0};
    };

    const auto P0  = quad(w[0]);
    const auto P1  = quad(w[1]);
    const auto P2  = quad(w[2]);
    const auto P01 = py_poly_mul(P0, P1);
    const auto P02 = py_poly_mul(P0, P2);
    const auto P12 = py_poly_mul(P1, P2);
    const auto P012 = py_poly_mul(P01, P2);

    // phase * P012 - m0*P12 - m1*P02 - m2*P01
    std::vector<lcbinint::Complex> poly(P012.size(), 0.0);
    for (std::size_t i = 0; i < P012.size(); ++i) poly[i] = phase * P012[i];
    for (std::size_t i = 0; i < P12.size(); ++i)  poly[i] -= geometry.masses[0] * P12[i];
    for (std::size_t i = 0; i < P02.size(); ++i)  poly[i] -= geometry.masses[1] * P02[i];
    for (std::size_t i = 0; i < P01.size(); ++i)  poly[i] -= geometry.masses[2] * P01[i];
    return poly;
}

// Returns the 6 caustic points (source plane) for a given phase angle.
std::vector<lcbinint::SourcePosition> py_triple_caustic_points_at_phase(
    const lcbinint::model::TripleLensGeometry& geometry,
    double phase_angle)
{
    const auto phase = std::polar(1.0, phase_angle);
    const auto poly  = py_triple_critical_polynomial(geometry, phase);

    lcbinint::math::PolynomialRootSolver solver;
    const auto result = solver.solve(poly);
    if (result.status != lcbinint::math::RootSolverStatus::ok) {
        return {};
    }

    // w_root is conj(z_image); map to source plane via the triple lens equation:
    // zeta = conj(w) - sum_i m_i / (w - conj(z_i))
    std::array<lcbinint::Complex, 3> w_lenses;
    for (int i = 0; i < 3; ++i) {
        w_lenses[i] = std::conj(lcbinint::Complex(
            geometry.lens_positions[i].x, geometry.lens_positions[i].y));
    }

    std::vector<lcbinint::SourcePosition> points;
    points.reserve(result.roots.size());
    for (const auto& w : result.roots) {
        lcbinint::Complex source = std::conj(w);
        for (int i = 0; i < 3; ++i) {
            source -= geometry.masses[i] / (w - w_lenses[i]);
        }
        points.push_back({source.real(), source.imag()});
    }
    return points;
}

PyGeometryBranches py_triple_caustics(
    const lcbinint::model::TripleLensGeometry& geometry,
    int n_points)
{
    constexpr int kBranches = 6;
    std::vector<std::vector<lcbinint::SourcePosition>> branches(kBranches);
    for (int k = 0; k < n_points; ++k) {
        const double phase_angle = 2.0 * M_PI * k / n_points;
        py_append_tracked_points(branches, py_triple_caustic_points_at_phase(geometry, phase_angle));
    }
    return py_pack_geometry_branches(branches);
}

struct PyOrbitCache {
    lcbi_orbital_motion_mode mode = LCBI_ORBIT_STATIC;
    double reference_time = 0.0;
    double separation = 1.0;
    double angle = 0.0;

    double w_orb = 0.0;
    double phi0 = 0.0;
    double c_inc = 1.0;
    double s_inc = 0.0;
    double s_true = 1.0;
    double c_om = 1.0;
    double s_om = 0.0;

    double mean_motion = 0.0;
    double eccentricity = 0.0;
    double t_peri = 0.0;
    double semi_major_axis = 1.0;
    double one_minus_e2_sqrt = 1.0;
    std::array<double, 3> x_axis = {};
    std::array<double, 3> y_axis = {};
};

double py_orbital_reference_time(const lcbinint::model::LensParameters& params)
{
    return params.tfix != 0.0 ? params.tfix : params.t0;
}

PyOrbitCache make_orbit_cache(const lcbinint::model::LensParameters& params)
{
    constexpr double branch_eps = 1.0e-8;
    PyOrbitCache cache;
    cache.mode = params.orbital_motion_mode;
    cache.reference_time = py_orbital_reference_time(params);
    cache.separation = params.sep;
    cache.angle = params.theta;

    if (params.orbital_motion_mode == LCBI_ORBIT_CIRCULAR) {
        const double w13_sq = params.g1 * params.g1 + params.g3 * params.g3;
        const double w13 = std::sqrt(w13_sq);
        const double w123 = std::sqrt(w13_sq + params.g2 * params.g2);

        cache.w_orb = params.g2;
        cache.phi0 = 0.0;
        double inc = 0.0;
        if (w13 > branch_eps) {
            const double w3_eff = params.g3 > branch_eps ? params.g3 : branch_eps;
            cache.w_orb = w3_eff * w123 / w13;
            const double cos_inc_arg =
                std::clamp(params.g2 * w3_eff / (w13 * w123), -1.0, 1.0);
            inc = std::acos(cos_inc_arg);
            cache.phi0 = std::atan2(-params.g1 * w123, w3_eff * w13);
        }

        const double c_phi0 = std::cos(cache.phi0);
        const double s_phi0 = std::sin(cache.phi0);
        cache.c_inc = std::cos(inc);
        cache.s_inc = std::sin(inc);
        const double den0 =
            std::sqrt(c_phi0 * c_phi0 + cache.c_inc * cache.c_inc * s_phi0 * s_phi0);
        cache.s_true = params.sep / den0;
        const double c_angle0 = std::cos(params.theta);
        const double s_angle0 = std::sin(params.theta);
        cache.c_om = (c_phi0 * c_angle0 + cache.c_inc * s_angle0 * s_phi0) / den0;
        cache.s_om = (c_phi0 * s_angle0 - cache.c_inc * c_angle0 * s_phi0) / den0;
        return cache;
    }

    if (params.orbital_motion_mode == LCBI_ORBIT_KEPLER) {
        const double ar = params.lom_ar + branch_eps;
        const double smix = 1.0 + params.lom_szs * params.lom_szs;
        const double sqsmix = std::sqrt(smix);
        const double w11 = params.g1 * params.g1;
        const double w22 = params.g2 * params.g2;
        const double w33 = params.g3 * params.g3;
        const double w12 = w11 + w22;
        const double wt2 = w12 + w33;
        const double arm1 = ar - 1.0;
        const double arm2 = 2.0 * ar - 1.0;
        cache.mean_motion = std::sqrt(wt2 / arm2 / smix) / ar;

        const auto z_axis = py_normalize(
            {-params.lom_szs * params.g2, params.lom_szs * params.g1 - params.g3, params.g2});
        cache.x_axis = {
            -ar * w11 + arm1 * w22 - arm2 * params.lom_szs * params.g1 * params.g3 +
                arm1 * w33,
            -arm2 * params.g2 * (params.g1 + params.lom_szs * params.g3),
            arm1 * params.lom_szs * w12 - arm2 * params.g1 * params.g3 -
                ar * params.lom_szs * w33,
        };
        const double x_norm = py_safe_sqrt(py_dot(cache.x_axis, cache.x_axis));
        cache.x_axis = py_scale(cache.x_axis, 1.0 / x_norm);
        cache.eccentricity = x_norm / (ar * sqsmix * wt2);
        cache.y_axis = py_cross(z_axis, cache.x_axis);

        const double conu = (cache.x_axis[0] + cache.x_axis[2] * params.lom_szs) / sqsmix;
        const double cos_e0 =
            std::clamp((conu + cache.eccentricity) /
                           (1.0 + cache.eccentricity * conu),
                -1.0,
                1.0);
        const double sign =
            (cache.y_axis[0] + cache.y_axis[2] * params.lom_szs) > 0.0 ? 1.0 : -1.0;
        const double e0 = std::acos(cos_e0) * sign;
        const double sin_e0 = py_safe_sqrt(1.0 - cos_e0 * cos_e0) * sign;
        cache.t_peri =
            cache.reference_time - (e0 - cache.eccentricity * sin_e0) / cache.mean_motion;
        cache.semi_major_axis = ar * params.sep * sqsmix;
        cache.one_minus_e2_sqrt = py_safe_sqrt(1.0 - cache.eccentricity * cache.eccentricity);
        return cache;
    }

    return cache;
}

lcbinint::model::OrbitalState orbit_from_cache(const PyOrbitCache& cache, double time)
{
    if (cache.mode == LCBI_ORBIT_CIRCULAR) {
        const double phi = cache.w_orb * (time - cache.reference_time) + cache.phi0;
        const double c_phi = std::cos(phi);
        const double s_phi = std::sin(phi);
        const double den = std::sqrt(c_phi * c_phi + cache.c_inc * cache.c_inc * s_phi * s_phi);
        const double sin_angle = (c_phi * cache.s_om + cache.c_inc * s_phi * cache.c_om) / den;
        const double cos_angle = (c_phi * cache.c_om - cache.c_inc * s_phi * cache.s_om) / den;
        return {
            cache.s_true * den,
            std::atan2(sin_angle, cos_angle),
            cache.s_true * cache.s_inc * s_phi,
        };
    }

    if (cache.mode == LCBI_ORBIT_KEPLER) {
        const double mean_anomaly = cache.mean_motion * (time - cache.t_peri);
        const double eccentric_anomaly =
            py_solve_kepler_newton(mean_anomaly, cache.eccentricity);
        const double cos_e = std::cos(eccentric_anomaly);
        const double sin_e = std::sin(eccentric_anomaly);
        const double r0 = cache.semi_major_axis * (cos_e - cache.eccentricity);
        const double r1 = cache.semi_major_axis * cache.one_minus_e2_sqrt * sin_e;
        const double x0 = r0 * cache.x_axis[0] + r1 * cache.y_axis[0];
        const double x1 = r0 * cache.x_axis[1] + r1 * cache.y_axis[1];
        const double x2 = r0 * cache.x_axis[2] + r1 * cache.y_axis[2];
        return {
            std::sqrt(x0 * x0 + x1 * x1),
            cache.angle + std::atan2(x1, x0),
            x2,
        };
    }

    return {cache.separation, cache.angle, 0.0};
}

lcbinint::magnification::FiniteSourceMagnifier& cached_finite_source_magnifier(
    const lcbinint::magnification::FiniteSourceSettings& settings)
{
    struct Cache {
        bool valid = false;
        lcbinint::magnification::FiniteSourceSettings settings;
        std::unique_ptr<lcbinint::magnification::FiniteSourceMagnifier> magnifier;
    };
    thread_local Cache cache;
    if (!cache.valid || !same_finite_source_settings(cache.settings, settings)) {
        cache.settings = settings;
        cache.magnifier =
            std::make_unique<lcbinint::magnification::FiniteSourceMagnifier>(settings);
        cache.valid = true;
    }
    return *cache.magnifier;
}

std::vector<double> evaluate_binary_light_curve(
    const std::vector<double>& times,
    const lcbi_params& params,
    const lcbi_options& options)
{
    const auto model_params = lcbinint::model::from_c_params(params);
    const auto model_options = lcbinint::model::from_c_options(&options);
    const bool direct_static_binary =
        !model_params.is_triple() &&
        !model_params.has_orbital_motion() &&
        model_params.piEN == 0.0 &&
        model_params.piEE == 0.0 &&
        model_params.piEN_xa == 0.0 &&
        model_params.piEE_xa == 0.0 &&
        model_params.xi_1 == 0.0 &&
        model_params.xi_2 == 0.0 &&
        model_params.omega == 0.0 &&
        model_params.v_sep == 0.0 &&
        model_options.center_of_mass == 0 &&
        model_options.vbm_compatible != 0;
    py::gil_scoped_release release;
    if (direct_static_binary) {
        lcbinint::magnification::PointSourceMagnifier point_magnifier;
        const double cos_alpha = std::cos(model_params.theta);
        const double sin_alpha = std::sin(model_params.theta);
        const double inverse_tE = 1.0 / model_params.tE;
        const double effective_q = model_params.q != 0.0 ? 1.0 / model_params.q : model_params.q;

        std::vector<double> values;
        values.reserve(times.size());
        if (model_params.rho == 0.0) {
            for (const double time : times) {
                const double tau = (time - model_params.t0) * inverse_tE;
                const lcbinint::SourcePosition source {
                    tau * cos_alpha - model_params.umin * sin_alpha,
                    tau * sin_alpha + model_params.umin * cos_alpha,
                };
                values.push_back(
                    point_magnifier.binary_mag0(model_params.sep, effective_q, source).magnification);
            }
            return values;
        }

        const auto settings = make_finite_source_settings(model_params, model_options);
        auto& finite_magnifier = cached_finite_source_magnifier(settings);
        for (const double time : times) {
            const double tau = (time - model_params.t0) * inverse_tE;
            const lcbinint::SourcePosition source {
                tau * cos_alpha - model_params.umin * sin_alpha,
                tau * sin_alpha + model_params.umin * cos_alpha,
            };
            const auto point =
                point_magnifier.binary_mag0_cached(model_params.sep, effective_q, source);
            const auto finite = finite_magnifier.binary_mag(
                model_params.sep, effective_q, source, std::abs(model_params.rho),
                point.magnification, nullptr, true, &point_magnifier);
            if (!std::isfinite(finite.magnification)) {
                throw std::runtime_error("numerical error");
            }
            values.push_back(finite.magnification);
        }
        return values;
    }

    const lcbinint::model::LensModel model(
        model_params,
        model_options);
    std::vector<double> values;
    values.reserve(times.size());
    for (const double time : times) {
        const auto result = model.magnification(time);
        if (result.status == lcbinint::EvaluationStatus::unsupported) {
            throw std::runtime_error("unsupported");
        }
        if (result.status == lcbinint::EvaluationStatus::numerical_error ||
            !std::isfinite(result.magnification)) {
            throw std::runtime_error("numerical error");
        }
        values.push_back(result.magnification);
    }
    return values;
}

double evaluate_binary_magnification(
    double time,
    const lcbi_params& params,
    const lcbi_options& options)
{
    const auto model_params = lcbinint::model::from_c_params(params);
    const auto model_options = lcbinint::model::from_c_options(&options);
    const bool direct_static_binary =
        !model_params.is_triple() &&
        !model_params.has_orbital_motion() &&
        model_params.piEN == 0.0 &&
        model_params.piEE == 0.0 &&
        model_params.piEN_xa == 0.0 &&
        model_params.piEE_xa == 0.0 &&
        model_params.xi_1 == 0.0 &&
        model_params.xi_2 == 0.0 &&
        model_params.omega == 0.0 &&
        model_params.v_sep == 0.0 &&
        model_options.center_of_mass == 0 &&
        model_options.vbm_compatible != 0;
    py::gil_scoped_release release;
    if (direct_static_binary) {
        lcbinint::magnification::PointSourceMagnifier point_magnifier;
        const double tau = (time - model_params.t0) / model_params.tE;
        const double cos_alpha = std::cos(model_params.theta);
        const double sin_alpha = std::sin(model_params.theta);
        const lcbinint::SourcePosition source {
            tau * cos_alpha - model_params.umin * sin_alpha,
            tau * sin_alpha + model_params.umin * cos_alpha,
        };
        const double effective_q = model_params.q != 0.0 ? 1.0 / model_params.q : model_params.q;
        const auto point = point_magnifier.binary_mag0(model_params.sep, effective_q, source);
        if (model_params.rho == 0.0) {
            return point.magnification;
        }

        const auto settings = make_finite_source_settings(model_params, model_options);
        auto& finite_magnifier = cached_finite_source_magnifier(settings);
        const auto finite = finite_magnifier.binary_mag(
            model_params.sep, effective_q, source, std::abs(model_params.rho),
            point.magnification, nullptr, true);
        if (!std::isfinite(finite.magnification)) {
            throw std::runtime_error("numerical error");
        }
        return finite.magnification;
    }

    const lcbinint::model::LensModel model(model_params, model_options);
    const auto result = model.magnification(time);
    if (result.status == lcbinint::EvaluationStatus::unsupported) {
        throw std::runtime_error("unsupported");
    }
    if (result.status == lcbinint::EvaluationStatus::numerical_error ||
        !std::isfinite(result.magnification)) {
        throw std::runtime_error("numerical error");
    }
    return result.magnification;
}

py::array_t<double> evaluate_binary_light_curve_numpy(
    py::array_t<double, py::array::c_style | py::array::forcecast> times,
    const lcbi_params& params,
    const lcbi_options& options)
{
    const py::buffer_info input = times.request();
    if (input.ndim != 1) {
        throw std::invalid_argument("times must be one-dimensional");
    }
    const auto count = static_cast<py::ssize_t>(input.shape[0]);
    const auto* time_values = static_cast<const double*>(input.ptr);
    std::vector<double> time_copy(time_values, time_values + count);
    auto output = py::array_t<double>(
        {count},
        {static_cast<py::ssize_t>(sizeof(double))});
    auto* values = output.mutable_data();

    const auto model_params = lcbinint::model::from_c_params(params);
    const auto model_options = lcbinint::model::from_c_options(&options);
    const bool direct_static_binary =
        !model_params.is_triple() &&
        !model_params.has_orbital_motion() &&
        model_params.piEN == 0.0 &&
        model_params.piEE == 0.0 &&
        model_params.piEN_xa == 0.0 &&
        model_params.piEE_xa == 0.0 &&
        model_params.xi_1 == 0.0 &&
        model_params.xi_2 == 0.0 &&
        model_params.omega == 0.0 &&
        model_params.v_sep == 0.0 &&
        model_options.center_of_mass == 0 &&
        model_options.vbm_compatible != 0;
    py::gil_scoped_release release;
    if (direct_static_binary) {
        lcbinint::magnification::PointSourceMagnifier point_magnifier;
        const double cos_alpha = std::cos(model_params.theta);
        const double sin_alpha = std::sin(model_params.theta);
        const double inverse_tE = 1.0 / model_params.tE;
        const double effective_q = model_params.q != 0.0 ? 1.0 / model_params.q : model_params.q;
        if (model_params.rho == 0.0) {
            for (py::ssize_t i = 0; i < count; ++i) {
                const double tau = (time_copy[static_cast<std::size_t>(i)] - model_params.t0) * inverse_tE;
                const lcbinint::SourcePosition source {
                    tau * cos_alpha - model_params.umin * sin_alpha,
                    tau * sin_alpha + model_params.umin * cos_alpha,
                };
                values[i] = point_magnifier
                    .binary_mag0(model_params.sep, effective_q, source)
                    .magnification;
            }
            return output;
        }

        const auto settings = make_finite_source_settings(model_params, model_options);
        auto& finite_magnifier = cached_finite_source_magnifier(settings);

        for (py::ssize_t i = 0; i < count; ++i) {
            const double tau = (time_copy[static_cast<std::size_t>(i)] - model_params.t0) * inverse_tE;
            const lcbinint::SourcePosition source {
                tau * cos_alpha - model_params.umin * sin_alpha,
                tau * sin_alpha + model_params.umin * cos_alpha,
            };
            const auto point = point_magnifier.binary_mag0(model_params.sep, effective_q, source);
            const auto finite = finite_magnifier.binary_mag(
                model_params.sep, effective_q, source, std::abs(model_params.rho),
                point.magnification, nullptr, true);
            if (!std::isfinite(finite.magnification)) {
                throw std::runtime_error("numerical error");
            }
            values[i] = finite.magnification;
        }
        return output;
    }

    const lcbinint::model::LensModel model(model_params, model_options);
    for (py::ssize_t i = 0; i < count; ++i) {
        const auto result = model.magnification(time_copy[static_cast<std::size_t>(i)]);
        if (result.status == lcbinint::EvaluationStatus::unsupported) {
            throw std::runtime_error("unsupported");
        }
        if (result.status == lcbinint::EvaluationStatus::numerical_error ||
            !std::isfinite(result.magnification)) {
            throw std::runtime_error("numerical error");
        }
        values[i] = result.magnification;
    }
    return output;
}

PyLightCurve evaluate_binary_light_curve_info(
    const std::vector<double>& times,
    const lcbi_params& params,
    const lcbi_options& options)
{
    PyLightCurve curve;
    curve.times = times;
    curve.magnifications.reserve(times.size());
    curve.point_source_magnifications.reserve(times.size());
    curve.finite_source_magnifications.reserve(times.size());
    curve.finite_source_error_estimates.reserve(times.size());
    curve.source_x.reserve(times.size());
    curve.source_y.reserve(times.size());
    curve.image_counts.reserve(times.size());
    curve.finite_source_methods.reserve(times.size());
    curve.finite_source_method_names.reserve(times.size());
    curve.finite_source_refinement_levels.reserve(times.size());
    curve.finite_source_converged.reserve(times.size());

    const auto model_params = lcbinint::model::from_c_params(params);
    const auto model_options = lcbinint::model::from_c_options(&options);
    const bool direct_static_binary =
        !model_params.is_triple() &&
        !model_params.has_orbital_motion() &&
        model_params.piEN == 0.0 &&
        model_params.piEE == 0.0 &&
        model_params.piEN_xa == 0.0 &&
        model_params.piEE_xa == 0.0 &&
        model_params.xi_1 == 0.0 &&
        model_params.xi_2 == 0.0 &&
        model_params.omega == 0.0 &&
        model_params.v_sep == 0.0 &&
        model_options.center_of_mass == 0 &&
        model_options.vbm_compatible != 0;

    py::gil_scoped_release release;
    if (direct_static_binary) {
        lcbinint::magnification::PointSourceMagnifier point_magnifier;
        const double cos_alpha = std::cos(model_params.theta);
        const double sin_alpha = std::sin(model_params.theta);
        const double inverse_tE = 1.0 / model_params.tE;
        const double effective_q = model_params.q != 0.0 ? 1.0 / model_params.q : model_params.q;

        const auto settings = make_finite_source_settings(model_params, model_options);
        auto& finite_magnifier = cached_finite_source_magnifier(settings);

        for (const double time : times) {
            const double tau = (time - model_params.t0) * inverse_tE;
            const lcbinint::SourcePosition source {
                tau * cos_alpha - model_params.umin * sin_alpha,
                tau * sin_alpha + model_params.umin * cos_alpha,
            };
            const auto point = point_magnifier.binary_mag0(model_params.sep, effective_q, source);
            curve.point_source_magnifications.push_back(point.magnification);
            curve.source_x.push_back(source.x);
            curve.source_y.push_back(source.y);
            curve.image_counts.push_back(point.image_count);

            if (model_params.rho == 0.0) {
                curve.magnifications.push_back(point.magnification);
                curve.finite_source_magnifications.push_back(point.magnification);
                curve.finite_source_error_estimates.push_back(0.0);
                curve.finite_source_methods.push_back(
                    static_cast<int>(lcbinint::magnification::FiniteSourceMethod::point_source));
                curve.finite_source_method_names.push_back("point_source");
                curve.finite_source_refinement_levels.push_back(0);
                curve.finite_source_converged.push_back(true);
                continue;
            }

            const auto finite = finite_magnifier.binary_mag(
                model_params.sep, effective_q, source, std::abs(model_params.rho),
                point.magnification, nullptr, true);
            if (!std::isfinite(finite.magnification)) {
                throw std::runtime_error("numerical error");
            }
            curve.magnifications.push_back(finite.magnification);
            curve.finite_source_magnifications.push_back(finite.magnification);
            curve.finite_source_error_estimates.push_back(finite.error_estimate);
            curve.finite_source_methods.push_back(static_cast<int>(finite.decision.method));
            curve.finite_source_method_names.push_back(
                lcbinint::magnification::finite_source_method_name(finite.decision.method));
            curve.finite_source_refinement_levels.push_back(finite.refinement_level);
            curve.finite_source_converged.push_back(finite.converged);
        }
        return curve;
    }

    const lcbinint::model::LensModel model(model_params, model_options);
    for (const double time : times) {
        const auto result = model.magnification(time);
        if (result.status == lcbinint::EvaluationStatus::unsupported) {
            throw std::runtime_error("unsupported");
        }
        if (result.status == lcbinint::EvaluationStatus::numerical_error ||
            !std::isfinite(result.magnification)) {
            throw std::runtime_error("numerical error");
        }
        curve.magnifications.push_back(result.magnification);
        curve.point_source_magnifications.push_back(result.point_source_magnification);
        curve.finite_source_magnifications.push_back(result.finite_source_magnification);
        curve.finite_source_error_estimates.push_back(result.finite_source_error_estimate);
        curve.source_x.push_back(result.source.x);
        curve.source_y.push_back(result.source.y);
        curve.image_counts.push_back(result.image_count);
        curve.finite_source_methods.push_back(result.finite_source_method);
        curve.finite_source_method_names.push_back(
            lcbinint::magnification::finite_source_method_name(
                static_cast<lcbinint::magnification::FiniteSourceMethod>(
                    result.finite_source_method)));
        curve.finite_source_refinement_levels.push_back(result.finite_source_refinement_level);
        curve.finite_source_converged.push_back(result.finite_source_converged);
    }
    return curve;
}

std::string normalize_lens_name(std::string lens)
{
    if (lens == "binary" || lens == "binary_lens") {
        return "binary_lens";
    }
    if (lens == "triple" || lens == "triple_lens") {
        return "triple_lens";
    }
    throw std::invalid_argument("lens must be 'binary_lens' or 'triple_lens'");
}

std::string normalize_source_name(const std::string& source)
{
    if (source.empty() || source == "single") {
        return "single";
    }
    if (source == "binary" || source == "binary_source") {
        return "binary";
    }
    throw std::invalid_argument("source must be 'single' or 'binary'");
}

class PyLightCurveFunc {
public:
    PyLightCurveFunc(
        std::string lens,
        PyEventCoordinates event,
        lcbi_options options,
        PyLimbDarkening limb_darkening,
        lcbi_orbital_motion_mode orbital_motion_mode)
        : lens_(normalize_lens_name(std::move(lens)))
        , event_(event)
        , options_(options)
        , limb_darkening_(limb_darkening)
        , orbital_motion_mode_(orbital_motion_mode)
        , model_options_(lcbinint::model::from_c_options(&options_))
        , finite_settings_(make_finite_source_settings(limb_darkening_, model_options_))
        , direct_static_binary_(lens_ == "binary_lens" && can_use_direct_static_binary(
              model_options_, orbital_motion_mode_, false))
    {
    }

    const std::string& lens() const { return lens_; }
    const PyEventCoordinates& event() const { return event_; }
    const lcbi_options& options() const { return options_; }
    const PyLimbDarkening& limb_darkening() const { return limb_darkening_; }
    lcbi_orbital_motion_mode orbital_motion_mode() const { return orbital_motion_mode_; }
    bool parallax() const { return false; }

    py::array_t<double> light_curve(
        py::array_t<double, py::array::c_style | py::array::forcecast> times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        if (direct_static_binary_) {
            return direct_light_curve(times, t0, tE, u0, alpha, s, q, rho);
        }
        return dynamic_light_curve(
            times, t0, tE, u0, alpha, s, q, rho,
            0.0, 0.0, g1, g2, g3, lom_szs, lom_ar);
    }

    py::array_t<double> dynamic_light_curve(
        py::array_t<double, py::array::c_style | py::array::forcecast> times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        const auto params = make_binary_params(
            t0, tE, u0, alpha, s, q, rho, limb_darkening_, event_,
            piEN, piEE, g1, g2, g3, orbital_motion_mode_, lom_szs, lom_ar);
        return direct_dynamic_light_curve(times, params);
    }

    std::vector<double> light_curve_list(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        if (direct_static_binary_) {
            return direct_light_curve_list(times, t0, tE, u0, alpha, s, q, rho);
        }
        return dynamic_light_curve_list(
            times, t0, tE, u0, alpha, s, q, rho,
            0.0, 0.0, g1, g2, g3, lom_szs, lom_ar);
    }

    std::vector<double> dynamic_light_curve_list(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        const auto params = make_binary_params(
            t0, tE, u0, alpha, s, q, rho, limb_darkening_, event_,
            piEN, piEE, g1, g2, g3, orbital_motion_mode_, lom_szs, lom_ar);
        std::vector<double> values(times.size());
        py::gil_scoped_release release;
        direct_dynamic_fill_values(
            times.data(), static_cast<py::ssize_t>(times.size()), values.data(), params);
        return values;
    }

    PyLightCurve info(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        if (direct_static_binary_) {
            return direct_info(times, t0, tE, u0, alpha, s, q, rho);
        }
        return dynamic_info(
            times, t0, tE, u0, alpha, s, q, rho,
            0.0, 0.0, g1, g2, g3, lom_szs, lom_ar);
    }

    PyLightCurve dynamic_info(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        const auto params = make_binary_params(
            t0, tE, u0, alpha, s, q, rho, limb_darkening_, event_,
            piEN, piEE, g1, g2, g3, orbital_motion_mode_, lom_szs, lom_ar);
        return direct_dynamic_info(times, params);
    }

    PyGeometryCurve source_trajectory(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        const auto params = make_binary_params(
            t0, tE, u0, alpha, s, q, 0.0, limb_darkening_, event_,
            0.0, 0.0, g1, g2, g3, orbital_motion_mode_, lom_szs, lom_ar);
        return dynamic_source_trajectory(times, params);
    }

    PyGeometryCurve dynamic_source_trajectory(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        const auto params = make_binary_params(
            t0, tE, u0, alpha, s, q, 0.0, limb_darkening_, event_,
            piEN, piEE, g1, g2, g3, orbital_motion_mode_, lom_szs, lom_ar);
        return dynamic_source_trajectory(times, params);
    }

    PyGeometryBranches caustics(
        double s,
        double q,
        int n_points,
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        return geometry_branches(
            s, q, n_points, time, t0, tE, u0, alpha, g1, g2, g3, lom_szs, lom_ar, true);
    }

    PyGeometryBranches critical_curves(
        double s,
        double q,
        int n_points,
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        return geometry_branches(
            s, q, n_points, time, t0, tE, u0, alpha, g1, g2, g3, lom_szs, lom_ar, false);
    }

    double magnification(
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        if (direct_static_binary_) {
            return direct_magnification(time, t0, tE, u0, alpha, s, q, rho);
        }
        return dynamic_magnification(
            time, t0, tE, u0, alpha, s, q, rho,
            0.0, 0.0, g1, g2, g3, lom_szs, lom_ar);
    }

    double dynamic_magnification(
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        const auto params = make_binary_params(
            t0, tE, u0, alpha, s, q, rho, limb_darkening_, event_,
            piEN, piEE, g1, g2, g3, orbital_motion_mode_, lom_szs, lom_ar);
        double value = 0.0;
        py::gil_scoped_release release;
        direct_dynamic_fill_values(&time, 1, &value, params);
        return value;
    }

private:
    py::array_t<double> direct_light_curve(
        py::array_t<double, py::array::c_style | py::array::forcecast> times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho) const
    {
        const py::buffer_info input = times.request();
        if (input.ndim != 1) {
            throw std::invalid_argument("times must be one-dimensional");
        }
        const auto count = static_cast<py::ssize_t>(input.shape[0]);
        const auto* time_values = static_cast<const double*>(input.ptr);
        std::vector<double> time_copy(time_values, time_values + count);
        auto output = py::array_t<double>(
            {count},
            {static_cast<py::ssize_t>(sizeof(double))});
        auto* values = output.mutable_data();

        py::gil_scoped_release release;
        direct_fill_values(time_copy.data(), count, values, t0, tE, u0, alpha, s, q, rho);
        return output;
    }

    std::vector<double> direct_light_curve_list(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho) const
    {
        std::vector<double> values(times.size());
        py::gil_scoped_release release;
        direct_fill_values(
            times.data(), static_cast<py::ssize_t>(times.size()), values.data(),
            t0, tE, u0, alpha, s, q, rho);
        return values;
    }

    double direct_magnification(
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho) const
    {
        double value = 0.0;
        py::gil_scoped_release release;
        direct_fill_values(&time, 1, &value, t0, tE, u0, alpha, s, q, rho);
        return value;
    }

    PyLightCurve direct_info(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho) const
    {
        PyLightCurve curve;
        curve.times = times;
        curve.magnifications.reserve(times.size());
        curve.point_source_magnifications.reserve(times.size());
        curve.finite_source_magnifications.reserve(times.size());
        curve.finite_source_error_estimates.reserve(times.size());
        curve.source_x.reserve(times.size());
        curve.source_y.reserve(times.size());
        curve.image_counts.reserve(times.size());
        curve.finite_source_methods.reserve(times.size());
        curve.finite_source_method_names.reserve(times.size());
        curve.finite_source_refinement_levels.reserve(times.size());
        curve.finite_source_converged.reserve(times.size());

        py::gil_scoped_release release;
        lcbinint::magnification::PointSourceMagnifier point_magnifier;
        const double cos_alpha = std::cos(alpha);
        const double sin_alpha = std::sin(alpha);
        const double inverse_tE = 1.0 / tE;
        const double effective_q = q != 0.0 ? 1.0 / q : q;
        auto& finite_magnifier = cached_finite_source_magnifier(finite_settings_);

        for (const double time : times) {
            const double tau = (time - t0) * inverse_tE;
            const lcbinint::SourcePosition source {
                tau * cos_alpha - u0 * sin_alpha,
                tau * sin_alpha + u0 * cos_alpha,
            };
            const auto point = point_magnifier.binary_mag0(s, effective_q, source);
            curve.point_source_magnifications.push_back(point.magnification);
            curve.source_x.push_back(source.x);
            curve.source_y.push_back(source.y);
            curve.image_counts.push_back(point.image_count);

            if (rho == 0.0) {
                curve.magnifications.push_back(point.magnification);
                curve.finite_source_magnifications.push_back(point.magnification);
                curve.finite_source_error_estimates.push_back(0.0);
                curve.finite_source_methods.push_back(
                    static_cast<int>(lcbinint::magnification::FiniteSourceMethod::point_source));
                curve.finite_source_method_names.push_back("point_source");
                curve.finite_source_refinement_levels.push_back(0);
                curve.finite_source_converged.push_back(true);
                continue;
            }

            const auto finite = finite_magnifier.binary_mag(
                s, effective_q, source, std::abs(rho), point.magnification, nullptr, true);
            if (!std::isfinite(finite.magnification)) {
                throw std::runtime_error("numerical error");
            }
            curve.magnifications.push_back(finite.magnification);
            curve.finite_source_magnifications.push_back(finite.magnification);
            curve.finite_source_error_estimates.push_back(finite.error_estimate);
            curve.finite_source_methods.push_back(static_cast<int>(finite.decision.method));
            curve.finite_source_method_names.push_back(
                lcbinint::magnification::finite_source_method_name(finite.decision.method));
            curve.finite_source_refinement_levels.push_back(finite.refinement_level);
            curve.finite_source_converged.push_back(finite.converged);
        }
        return curve;
    }

    py::array_t<double> direct_dynamic_light_curve(
        py::array_t<double, py::array::c_style | py::array::forcecast> times,
        const lcbi_params& c_params) const
    {
        const py::buffer_info input = times.request();
        if (input.ndim != 1) {
            throw std::invalid_argument("times must be one-dimensional");
        }
        const auto count = static_cast<py::ssize_t>(input.shape[0]);
        const auto* time_values = static_cast<const double*>(input.ptr);
        std::vector<double> time_copy(time_values, time_values + count);
        auto output = py::array_t<double>(
            {count},
            {static_cast<py::ssize_t>(sizeof(double))});
        auto* values = output.mutable_data();

        py::gil_scoped_release release;
        direct_dynamic_fill_values(time_copy.data(), count, values, c_params);
        return output;
    }

    PyLightCurve direct_dynamic_info(
        const std::vector<double>& times,
        const lcbi_params& c_params) const
    {
        PyLightCurve curve;
        curve.times = times;
        curve.magnifications.reserve(times.size());
        curve.point_source_magnifications.reserve(times.size());
        curve.finite_source_magnifications.reserve(times.size());
        curve.finite_source_error_estimates.reserve(times.size());
        curve.source_x.reserve(times.size());
        curve.source_y.reserve(times.size());
        curve.image_counts.reserve(times.size());
        curve.finite_source_methods.reserve(times.size());
        curve.finite_source_method_names.reserve(times.size());
        curve.finite_source_refinement_levels.reserve(times.size());
        curve.finite_source_converged.reserve(times.size());

        const auto params = lcbinint::model::from_c_params(c_params);
        const lcbinint::model::Trajectory trajectory(params);
        const auto orbit_cache = make_orbit_cache(params);
        py::gil_scoped_release release;
        lcbinint::magnification::PointSourceMagnifier point_magnifier;
        for (const double time : times) {
            const auto result = direct_dynamic_magnification_result(
                time, params, trajectory, orbit_cache, point_magnifier);
            if (result.status == lcbinint::EvaluationStatus::unsupported) {
                throw std::runtime_error("unsupported");
            }
            if (result.status == lcbinint::EvaluationStatus::numerical_error ||
                !std::isfinite(result.magnification)) {
                throw std::runtime_error("numerical error");
            }
            curve.magnifications.push_back(result.magnification);
            curve.point_source_magnifications.push_back(result.point_source_magnification);
            curve.finite_source_magnifications.push_back(result.finite_source_magnification);
            curve.finite_source_error_estimates.push_back(result.finite_source_error_estimate);
            curve.source_x.push_back(result.source.x);
            curve.source_y.push_back(result.source.y);
            curve.image_counts.push_back(result.image_count);
            curve.finite_source_methods.push_back(result.finite_source_method);
            curve.finite_source_method_names.push_back(
                lcbinint::magnification::finite_source_method_name(
                    static_cast<lcbinint::magnification::FiniteSourceMethod>(
                        result.finite_source_method)));
            curve.finite_source_refinement_levels.push_back(result.finite_source_refinement_level);
            curve.finite_source_converged.push_back(result.finite_source_converged);
        }
        return curve;
    }

    PyGeometryCurve dynamic_source_trajectory(
        const std::vector<double>& times,
        const lcbi_params& c_params) const
    {
        PyGeometryCurve curve;
        curve.times = times;
        curve.x.reserve(times.size());
        curve.y.reserve(times.size());

        const auto params = lcbinint::model::from_c_params(c_params);
        const lcbinint::model::Trajectory trajectory(params);
        const auto orbit_cache = make_orbit_cache(params);
        py::gil_scoped_release release;
        for (const double time : times) {
            const auto source = source_for_geometry(time, params, trajectory, orbit_cache);
            curve.x.push_back(source.x);
            curve.y.push_back(source.y);
        }
        return curve;
    }

    lcbinint::SourcePosition source_for_geometry(
        double time,
        const lcbinint::model::LensParameters& params,
        const lcbinint::model::Trajectory& trajectory,
        const PyOrbitCache& orbit_cache) const
    {
        lcbinint::SourcePosition source;
        if (params.piEN == 0.0 && params.piEE == 0.0) {
            const double tau = (time - params.t0) / params.tE;
            const double beta = params.umin;
            const double cos_theta = std::cos(params.theta);
            const double sin_theta = std::sin(params.theta);
            if (model_options_.vbm_compatible != 0) {
                source.x = tau * cos_theta - beta * sin_theta;
                source.y = tau * sin_theta + beta * cos_theta;
            } else {
                source.x = beta * sin_theta + tau * cos_theta;
                source.y = beta * cos_theta - tau * sin_theta;
            }
        } else {
            source = trajectory.source_position(time, model_options_.vbm_compatible != 0);
        }

        if (params.orbital_motion_mode == LCBI_ORBIT_STATIC) {
            return source;
        }

        const auto orbit = orbit_from_cache(orbit_cache, time);
        if (model_options_.vbm_compatible != 0) {
            const double cos_theta = std::cos(params.theta);
            const double sin_theta = std::sin(params.theta);
            const double tau = source.x * cos_theta + source.y * sin_theta;
            const double beta = -source.x * sin_theta + source.y * cos_theta;
            return {
                tau * std::cos(orbit.angle) - beta * std::sin(orbit.angle),
                beta * std::cos(orbit.angle) + tau * std::sin(orbit.angle),
            };
        }
        return lcbinint::model::rotate_source_to_orbital_frame(
            source, orbit.angle - params.theta);
    }

    PyGeometryBranches geometry_branches(
        double s,
        double q,
        int n_points,
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar,
        bool map_to_source_plane) const
    {
        if (n_points < 8) {
            throw std::invalid_argument("n_points must be at least 8");
        }

        double separation = s;
        if (orbital_motion_mode_ != LCBI_ORBIT_STATIC && std::isfinite(time)) {
            const auto c_params = make_binary_params(
                t0, tE, u0, alpha, s, q, 0.0, limb_darkening_, event_,
                0.0, 0.0, g1, g2, g3, orbital_motion_mode_, lom_szs, lom_ar);
            const auto params = lcbinint::model::from_c_params(c_params);
            const auto orbit = orbit_from_cache(make_orbit_cache(params), time);
            separation = orbit.separation;
        }

        const double effective_q =
            (model_options_.vbm_compatible != 0 && q != 0.0) ? 1.0 / q : q;
        std::vector<std::vector<lcbinint::SourcePosition>> branches(4);
        lcbinint::magnification::PointSourceMagnifier point_magnifier;
        py::gil_scoped_release release;
        for (int i = 0; i < n_points; ++i) {
            const double phase_angle =
                2.0 * std::acos(-1.0) * static_cast<double>(i) / static_cast<double>(n_points);
            auto points = map_to_source_plane
                ? py_caustic_points_at_phase(point_magnifier, separation, effective_q, phase_angle)
                : py_critical_points_at_phase(separation, effective_q, phase_angle);
            py_append_tracked_points(branches, std::move(points));
        }
        return py_pack_geometry_branches(branches);
    }

    void direct_dynamic_fill_values(
        const double* times,
        py::ssize_t count,
        double* values,
        const lcbi_params& c_params) const
    {
        const auto params = lcbinint::model::from_c_params(c_params);
        const lcbinint::model::Trajectory trajectory(params);
        const auto orbit_cache = make_orbit_cache(params);
        lcbinint::magnification::PointSourceMagnifier point_magnifier;
        for (py::ssize_t i = 0; i < count; ++i) {
            const auto result = direct_dynamic_magnification_result(
                times[i], params, trajectory, orbit_cache, point_magnifier);
            if (result.status == lcbinint::EvaluationStatus::unsupported) {
                throw std::runtime_error("unsupported");
            }
            if (result.status == lcbinint::EvaluationStatus::numerical_error ||
                !std::isfinite(result.magnification)) {
                throw std::runtime_error("numerical error");
            }
            values[i] = result.magnification;
        }
    }

    lcbinint::MagnificationResult direct_dynamic_magnification_result(
        double time,
        const lcbinint::model::LensParameters& params,
        const lcbinint::model::Trajectory& trajectory,
        const PyOrbitCache& orbit_cache,
        lcbinint::magnification::PointSourceMagnifier& point_magnifier) const
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        lcbinint::MagnificationResult result;
        result.magnification = nan;
        result.point_source_magnification = nan;
        result.finite_source_magnification = nan;
        result.image_count = 0;

        lcbinint::SourcePosition source;
        if (params.piEN == 0.0 && params.piEE == 0.0) {
            const double tau = (time - params.t0) / params.tE;
            const double beta = params.umin;
            const double cos_theta = std::cos(params.theta);
            const double sin_theta = std::sin(params.theta);
            if (model_options_.vbm_compatible != 0) {
                source.x = tau * cos_theta - beta * sin_theta;
                source.y = tau * sin_theta + beta * cos_theta;
            } else {
                source.x = beta * sin_theta + tau * cos_theta;
                source.y = beta * cos_theta - tau * sin_theta;
            }
        } else {
            source = trajectory.source_position(time, model_options_.vbm_compatible != 0);
        }

        const bool static_orbit = params.orbital_motion_mode == LCBI_ORBIT_STATIC;
        const auto orbit = static_orbit
            ? lcbinint::model::OrbitalState {params.sep, params.theta, 0.0}
            : orbit_from_cache(orbit_cache, time);
        if (!std::isfinite(orbit.separation) || !std::isfinite(orbit.angle)) {
            result.status = lcbinint::EvaluationStatus::numerical_error;
            return result;
        }

        auto source_for_magnification = source;
        if (!static_orbit) {
            if (model_options_.vbm_compatible != 0) {
                const double cos_theta = std::cos(params.theta);
                const double sin_theta = std::sin(params.theta);
                const double tau = source.x * cos_theta + source.y * sin_theta;
                const double beta = -source.x * sin_theta + source.y * cos_theta;
                source_for_magnification = {
                    tau * std::cos(orbit.angle) - beta * std::sin(orbit.angle),
                    beta * std::cos(orbit.angle) + tau * std::sin(orbit.angle),
                };
            } else {
                source_for_magnification =
                    lcbinint::model::rotate_source_to_orbital_frame(
                        source, orbit.angle - params.theta);
            }
        }
        result.source = source_for_magnification;
        if (model_options_.vbm_compatible == 0) {
            source_for_magnification.x -= wide_binary_offset(orbit.separation, params.q);
        }
        const double effective_q =
            (model_options_.vbm_compatible != 0 && params.q != 0.0) ? 1.0 / params.q : params.q;

        if (params.rho == 0.0) {
            const auto point =
                point_magnifier.binary_mag0(orbit.separation, effective_q, source_for_magnification);
            result.point_source_magnification = point.magnification;
            result.finite_source_magnification = point.magnification;
            result.magnification = point.magnification;
            result.image_count = point.image_count;
            result.finite_source_error_estimate = 0.0;
            result.finite_source_method =
                static_cast<int>(lcbinint::magnification::FiniteSourceMethod::point_source);
            result.finite_source_refinement_level = 0;
            result.finite_source_converged = true;
            result.status = std::isfinite(result.magnification)
                ? lcbinint::EvaluationStatus::ok
                : lcbinint::EvaluationStatus::numerical_error;
            return result;
        }

        const auto point_images =
            point_magnifier.binary_images(orbit.separation, effective_q, source_for_magnification);
        double point_source_magnification = 0.0;
        std::vector<lcbinint::SourcePosition> center_image_seeds;
        center_image_seeds.reserve(point_images.size());
        for (const auto& image : point_images) {
            point_source_magnification += 1.0 / std::abs(image.jacobian_determinant);
            center_image_seeds.push_back(image.position);
        }
        result.point_source_magnification = point_source_magnification;
        result.image_count = static_cast<int>(point_images.size());

        auto& finite_magnifier = cached_finite_source_magnifier(finite_settings_);
        const auto finite = finite_magnifier.binary_mag(
            orbit.separation, effective_q, source_for_magnification, std::abs(params.rho),
            point_source_magnification, &center_image_seeds, true);
        result.magnification = finite.magnification;
        result.finite_source_magnification = finite.magnification;
        result.finite_source_error_estimate = finite.error_estimate;
        result.finite_source_method = static_cast<int>(finite.decision.method);
        result.finite_source_refinement_level = finite.refinement_level;
        result.finite_source_converged = finite.converged;
        result.status = std::isfinite(result.magnification)
            ? lcbinint::EvaluationStatus::ok
            : lcbinint::EvaluationStatus::numerical_error;
        return result;
    }

    double wide_binary_offset(double separation, double q) const
    {
        const double projected_separation = std::abs(separation);
        if (model_options_.center_of_mass != 0 || projected_separation <= 1.0) {
            return 0.0;
        }
        const double mass_ratio = std::abs(q);
        const double m2 = mass_ratio / (1.0 + mass_ratio);
        return m2 * projected_separation - m2 / projected_separation;
    }

    void direct_fill_values(
        const double* times,
        py::ssize_t count,
        double* values,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho) const
    {
        lcbinint::magnification::PointSourceMagnifier point_magnifier;
        const double cos_alpha = std::cos(alpha);
        const double sin_alpha = std::sin(alpha);
        const double inverse_tE = 1.0 / tE;
        const double effective_q = q != 0.0 ? 1.0 / q : q;
        if (rho == 0.0) {
            for (py::ssize_t i = 0; i < count; ++i) {
                const double tau = (times[i] - t0) * inverse_tE;
                const lcbinint::SourcePosition source {
                    tau * cos_alpha - u0 * sin_alpha,
                    tau * sin_alpha + u0 * cos_alpha,
                };
                values[i] = point_magnifier.binary_mag0(s, effective_q, source).magnification;
            }
            return;
        }

        auto& finite_magnifier = cached_finite_source_magnifier(finite_settings_);
        for (py::ssize_t i = 0; i < count; ++i) {
            const double tau = (times[i] - t0) * inverse_tE;
            const lcbinint::SourcePosition source {
                tau * cos_alpha - u0 * sin_alpha,
                tau * sin_alpha + u0 * cos_alpha,
            };
            const auto point = point_magnifier.binary_mag0_cached(s, effective_q, source);
            const auto finite = finite_magnifier.binary_mag(
                s, effective_q, source, std::abs(rho), point.magnification, nullptr, true,
                &point_magnifier);
            if (!std::isfinite(finite.magnification)) {
                throw std::runtime_error("numerical error");
            }
            values[i] = finite.magnification;
        }
    }

    std::string lens_;
    PyEventCoordinates event_;
    lcbi_options options_;
    PyLimbDarkening limb_darkening_;
    lcbi_orbital_motion_mode orbital_motion_mode_;
    lcbinint::model::ComputationOptions model_options_;
    lcbinint::magnification::FiniteSourceSettings finite_settings_;
    bool direct_static_binary_;
};

class PyParallaxLightCurveFunc {
public:
    PyParallaxLightCurveFunc(
        std::string lens,
        PyEventCoordinates event,
        lcbi_options options,
        PyLimbDarkening limb_darkening,
        lcbi_orbital_motion_mode orbital_motion_mode)
        : base_(std::move(lens), event, options, limb_darkening, orbital_motion_mode)
    {
    }

    const std::string& lens() const { return base_.lens(); }
    const PyEventCoordinates& event() const { return base_.event(); }
    const lcbi_options& options() const { return base_.options(); }
    const PyLimbDarkening& limb_darkening() const { return base_.limb_darkening(); }
    lcbi_orbital_motion_mode orbital_motion_mode() const { return base_.orbital_motion_mode(); }
    bool parallax() const { return true; }

    py::array_t<double> light_curve(
        py::array_t<double, py::array::c_style | py::array::forcecast> times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        return base_.dynamic_light_curve(
            times, t0, tE, u0, alpha, s, q, rho,
            piEN, piEE, g1, g2, g3, lom_szs, lom_ar);
    }

    std::vector<double> light_curve_list(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        return base_.dynamic_light_curve_list(
            times, t0, tE, u0, alpha, s, q, rho,
            piEN, piEE, g1, g2, g3, lom_szs, lom_ar);
    }

    PyLightCurve info(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        return base_.dynamic_info(
            times, t0, tE, u0, alpha, s, q, rho,
            piEN, piEE, g1, g2, g3, lom_szs, lom_ar);
    }

    PyGeometryCurve source_trajectory(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        return base_.dynamic_source_trajectory(
            times, t0, tE, u0, alpha, s, q, piEN, piEE, g1, g2, g3, lom_szs, lom_ar);
    }

    PyGeometryBranches caustics(
        double s,
        double q,
        int n_points,
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        return base_.caustics(s, q, n_points, time, t0, tE, u0, alpha, g1, g2, g3, lom_szs, lom_ar);
    }

    PyGeometryBranches critical_curves(
        double s,
        double q,
        int n_points,
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        return base_.critical_curves(
            s, q, n_points, time, t0, tE, u0, alpha, g1, g2, g3, lom_szs, lom_ar);
    }

    double magnification(
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        return base_.dynamic_magnification(
            time, t0, tE, u0, alpha, s, q, rho,
            piEN, piEE, g1, g2, g3, lom_szs, lom_ar);
    }

private:
    PyLightCurveFunc base_;
};

class PyLightCurveEvaluator {
public:
    PyLightCurveEvaluator(
        std::string lens,
        std::string source,
        PyEventCoordinates event,
        lcbi_options options,
        PyLimbDarkening limb_darkening,
        lcbi_orbital_motion_mode orbital_motion_mode,
        bool parallax,
        bool terrestrial_parallax)
        : base_(std::move(lens), event, options, limb_darkening, orbital_motion_mode)
        , source_(normalize_source_name(source))
        , parallax_(parallax)
        , terrestrial_parallax_(terrestrial_parallax)
    {
    }

    const std::string& lens() const { return base_.lens(); }
    const std::string& source() const { return source_; }
    const PyEventCoordinates& event() const { return base_.event(); }
    const lcbi_options& options() const { return base_.options(); }
    const PyLimbDarkening& limb_darkening() const { return base_.limb_darkening(); }
    lcbi_orbital_motion_mode orbital_motion_mode() const { return base_.orbital_motion_mode(); }
    bool parallax() const { return parallax_; }
    bool terrestrial_parallax() const { return terrestrial_parallax_; }

    py::array_t<double> light_curve_from_dict(
        py::array_t<double, py::array::c_style | py::array::forcecast> times,
        const py::dict& params) const
    {
        const auto p = binary_params_from_dict(params);

        if (source_ == "binary") {
            const py::buffer_info input = times.request();
            if (input.ndim != 1) {
                throw std::invalid_argument("times must be one-dimensional");
            }
            const auto count = static_cast<py::ssize_t>(input.shape[0]);
            const std::vector<double> times_vec(
                static_cast<const double*>(input.ptr),
                static_cast<const double*>(input.ptr) + count);
            auto v1 = single_source_list(times_vec, p);
            auto p2 = p;
            p2.u0 = p.u0_2;
            p2.t0 = p.t0_2;
            const auto v2 = single_source_list(times_vec, p2);
            const double fr = p.flux_ratio;
            const double inv = 1.0 / (1.0 + fr);
            auto output = py::array_t<double>(
                {count}, {static_cast<py::ssize_t>(sizeof(double))});
            auto* out_data = output.mutable_data();
            for (std::size_t i = 0; i < v1.size(); ++i) {
                out_data[i] = (v1[i] + fr * v2[i]) * inv;
            }
            return output;
        }

        return evaluate_binary_light_curve_numpy(
            times, make_c_params_for_single_source(p), base_.options());
    }

    py::array_t<double> light_curve_static(
        py::array_t<double, py::array::c_style | py::array::forcecast> times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        return parallax_
            ? base_.dynamic_light_curve(
                  times, t0, tE, u0, alpha, s, q, rho,
                  0.0, 0.0, g1, g2, g3, lom_szs, lom_ar)
            : base_.light_curve(times, t0, tE, u0, alpha, s, q, rho, g1, g2, g3, lom_szs, lom_ar);
    }

    py::array_t<double> light_curve(
        py::array_t<double, py::array::c_style | py::array::forcecast> times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        require_parallax();
        return base_.dynamic_light_curve(
            times, t0, tE, u0, alpha, s, q, rho,
            piEN, piEE, g1, g2, g3, lom_szs, lom_ar);
    }

    std::vector<double> light_curve_list_static(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        return parallax_
            ? base_.dynamic_light_curve_list(
                  times, t0, tE, u0, alpha, s, q, rho,
                  0.0, 0.0, g1, g2, g3, lom_szs, lom_ar)
            : base_.light_curve_list(
                  times, t0, tE, u0, alpha, s, q, rho, g1, g2, g3, lom_szs, lom_ar);
    }

    std::vector<double> light_curve_list(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        require_parallax();
        return base_.dynamic_light_curve_list(
            times, t0, tE, u0, alpha, s, q, rho,
            piEN, piEE, g1, g2, g3, lom_szs, lom_ar);
    }

    PyLightCurve info_from_dict(const std::vector<double>& times, const py::dict& params) const
    {
        const auto p = binary_params_from_dict(params);

        auto blend_info = [&](PyLightCurve result) -> PyLightCurve {
            if (source_ == "binary") {
                auto p2 = p;
                p2.u0 = p.u0_2;
                p2.t0 = p.t0_2;
                const auto v2 = single_source_list(times, p2);
                const double fr = p.flux_ratio;
                const double inv = 1.0 / (1.0 + fr);
                for (std::size_t i = 0; i < result.magnifications.size(); ++i) {
                    result.magnifications[i] = (result.magnifications[i] + fr * v2[i]) * inv;
                }
                for (std::size_t i = 0; i < result.finite_source_magnifications.size(); ++i) {
                    result.finite_source_magnifications[i] =
                        (result.finite_source_magnifications[i] + fr * v2[i]) * inv;
                }
            }
            return result;
        };

        return blend_info(evaluate_binary_light_curve_info(
            times, make_c_params_for_single_source(p), base_.options()));
    }

    PyLightCurve info_static(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        return parallax_
            ? base_.dynamic_info(
                  times, t0, tE, u0, alpha, s, q, rho,
                  0.0, 0.0, g1, g2, g3, lom_szs, lom_ar)
            : base_.info(times, t0, tE, u0, alpha, s, q, rho, g1, g2, g3, lom_szs, lom_ar);
    }

    PyLightCurve info(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        require_parallax();
        return base_.dynamic_info(
            times, t0, tE, u0, alpha, s, q, rho,
            piEN, piEE, g1, g2, g3, lom_szs, lom_ar);
    }

    PyGeometryCurve source_trajectory_from_dict(
        const std::vector<double>& times,
        const py::dict& params) const
    {
        const auto p = binary_params_from_dict(params);
        return base_.dynamic_source_trajectory(
            times, p.t0, p.tE, p.u0, p.alpha, p.s, p.q,
            p.piEN, p.piEE, p.g1, p.g2, p.g3, p.lom_szs, p.lom_ar);
    }

    PyGeometryBranches caustics_from_dict(
        const py::dict& params,
        int n_points) const
    {
        if (lens() == "triple_lens") {
            const auto p = binary_params_from_dict(params);
            const auto model_opts = lcbinint::model::from_c_options(&base_.options());
            const auto geometry = model_opts.vbm_compatible != 0
                ? lcbinint::model::make_triple_lens_geometry_vbm(
                    p.s, p.q, p.sep2, p.ang, p.q2)
                : lcbinint::model::make_triple_lens_geometry(
                    p.s, p.q, p.q2, p.sep2, p.ang);
            return py_triple_caustics(geometry, n_points);
        }
        const auto p = binary_params_from_dict(params);
        return base_.caustics(p.s, p.q, n_points, std::numeric_limits<double>::quiet_NaN(),
                              0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    }

    PyGeometryCurve source_trajectory_static(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        return parallax_
            ? base_.dynamic_source_trajectory(
                  times, t0, tE, u0, alpha, s, q,
                  0.0, 0.0, g1, g2, g3, lom_szs, lom_ar)
            : base_.source_trajectory(times, t0, tE, u0, alpha, s, q, g1, g2, g3, lom_szs, lom_ar);
    }

    PyGeometryCurve source_trajectory(
        const std::vector<double>& times,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        require_parallax();
        return base_.dynamic_source_trajectory(
            times, t0, tE, u0, alpha, s, q, piEN, piEE, g1, g2, g3, lom_szs, lom_ar);
    }

    PyGeometryBranches caustics(
        double s,
        double q,
        int n_points,
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        return base_.caustics(s, q, n_points, time, t0, tE, u0, alpha, g1, g2, g3, lom_szs, lom_ar);
    }

    PyGeometryBranches critical_curves(
        double s,
        double q,
        int n_points,
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        return base_.critical_curves(
            s, q, n_points, time, t0, tE, u0, alpha, g1, g2, g3, lom_szs, lom_ar);
    }

    double magnification_from_dict(double time, const py::dict& params) const
    {
        const auto p = binary_params_from_dict(params);
        const double mag1 = single_source_magnification(time, p);
        if (source_ == "binary") {
            auto p2 = p;
            p2.u0 = p.u0_2;
            p2.t0 = p.t0_2;
            const double mag2 = single_source_magnification(time, p2);
            const double fr = p.flux_ratio;
            return (mag1 + fr * mag2) / (1.0 + fr);
        }
        return mag1;
    }

    double magnification(
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double piEN,
        double piEE,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        require_parallax();
        return base_.dynamic_magnification(
            time, t0, tE, u0, alpha, s, q, rho,
            piEN, piEE, g1, g2, g3, lom_szs, lom_ar);
    }

    double magnification_static(
        double time,
        double t0,
        double tE,
        double u0,
        double alpha,
        double s,
        double q,
        double rho,
        double g1,
        double g2,
        double g3,
        double lom_szs,
        double lom_ar) const
    {
        require_binary_positional_api();
        return parallax_
            ? base_.dynamic_magnification(
                  time, t0, tE, u0, alpha, s, q, rho,
                  0.0, 0.0, g1, g2, g3, lom_szs, lom_ar)
            : base_.magnification(time, t0, tE, u0, alpha, s, q, rho, g1, g2, g3, lom_szs, lom_ar);
    }

private:
    void require_parallax() const
    {
        if (!parallax_) {
            throw std::invalid_argument(
                "piEN/piEE require LightCurve(parallax=True); "
                "the current trajectory model is static");
        }
    }

    void require_binary_positional_api() const
    {
        if (lens() != "binary_lens") {
            throw std::invalid_argument(
                "triple_lens requires a parameter dictionary with q2, sep2, and ang");
        }
    }

    lcbi_params make_c_params_for_single_source(const PyBinaryParams& p) const
    {
        if (lens() == "triple_lens") {
            if (p.q2 <= 0.0) {
                throw py::value_error("triple_lens requires q2 > 0");
            }
            if (parallax_) {
                throw std::runtime_error("unsupported");
            }
            return make_triple_params(
                p.t0, p.tE, p.u0, p.alpha, p.s, p.q, p.q2, p.sep2, p.ang,
                p.rho, base_.limb_darkening(), base_.event());
        }
        const double piEN = parallax_ ? p.piEN : 0.0;
        const double piEE = parallax_ ? p.piEE : 0.0;
        auto c_params = make_binary_params(
            p.t0, p.tE, p.u0, p.alpha, p.s, p.q, p.rho,
            base_.limb_darkening(), base_.event(),
            piEN, piEE, p.g1, p.g2, p.g3,
            base_.orbital_motion_mode(), p.lom_szs, p.lom_ar,
            p.xi_1, p.xi_2, p.omega_xa, p.inc_xa, p.phi_xa,
            p.piEN_xa, p.piEE_xa, p.period_xa, p.ecc_xa, p.peri_xa);
        if (!terrestrial_parallax_) {
            c_params.obs_lat = 0.0;
            c_params.obs_lon = 0.0;
        }
        return c_params;
    }

    std::vector<double> single_source_list(
        const std::vector<double>& times,
        const PyBinaryParams& p) const
    {
        return evaluate_binary_light_curve(times, make_c_params_for_single_source(p), base_.options());
    }

    double single_source_magnification(double time, const PyBinaryParams& p) const
    {
        return evaluate_binary_magnification(time, make_c_params_for_single_source(p), base_.options());
    }

    PyLightCurveFunc base_;
    std::string source_;
    bool parallax_ = false;
    bool terrestrial_parallax_ = false;
};

} // namespace

PYBIND11_MODULE(lcbinint, m)
{
    m.doc() = "Python bindings for the lcbinint C++ core";

    py::enum_<lcbi_status>(m, "Status")
        .value("OK", LCBI_OK)
        .value("INVALID_ARGUMENT", LCBI_INVALID_ARGUMENT)
        .value("NUMERICAL_ERROR", LCBI_NUMERICAL_ERROR)
        .value("UNSUPPORTED", LCBI_UNSUPPORTED);

    py::enum_<lcbi_orbital_motion_mode>(m, "OrbitalMotionMode")
        .value("STATIC", LCBI_ORBIT_STATIC)
        .value("CIRCULAR", LCBI_ORBIT_CIRCULAR)
        .value("KEPLER", LCBI_ORBIT_KEPLER)
        .export_values();

    m.def("status_string", [](lcbi_status status) {
        return lcbi_status_string(status);
    });

    py::class_<PyLightCurve>(m, "LightCurveInfo")
        .def_readonly("times", &PyLightCurve::times)
        .def_readonly("magnifications", &PyLightCurve::magnifications)
        .def_readonly("point_source_magnifications", &PyLightCurve::point_source_magnifications)
        .def_readonly("finite_source_magnifications", &PyLightCurve::finite_source_magnifications)
        .def_readonly("finite_source_error_estimates", &PyLightCurve::finite_source_error_estimates)
        .def_readonly("source_x", &PyLightCurve::source_x)
        .def_readonly("source_y", &PyLightCurve::source_y)
        .def_readonly("image_counts", &PyLightCurve::image_counts)
        .def_readonly("finite_source_methods", &PyLightCurve::finite_source_methods)
        .def_readonly("finite_source_method_names", &PyLightCurve::finite_source_method_names)
        .def_readonly("finite_source_refinement_levels", &PyLightCurve::finite_source_refinement_levels)
        .def_readonly("finite_source_converged", &PyLightCurve::finite_source_converged)
        .def_property_readonly("all_converged", &all_converged)
        .def_property_readonly("unconverged_indices", &unconverged_indices);

    py::class_<PyGeometryCurve>(m, "GeometryCurve")
        .def_readonly("times", &PyGeometryCurve::times)
        .def_readonly("x", &PyGeometryCurve::x)
        .def_readonly("y", &PyGeometryCurve::y);

    py::class_<PyGeometryBranches>(m, "GeometryBranches")
        .def_readonly("x", &PyGeometryBranches::x)
        .def_readonly("y", &PyGeometryBranches::y);

    py::class_<PySourceBinCandidate>(m, "SourceBinCandidate")
        .def_readonly("source_bins", &PySourceBinCandidate::source_bins)
        .def_readonly("max_absolute_difference", &PySourceBinCandidate::max_absolute_difference)
        .def_readonly("max_relative_difference", &PySourceBinCandidate::max_relative_difference)
        .def_readonly("rms_relative_difference", &PySourceBinCandidate::rms_relative_difference)
        .def_readonly("accepted", &PySourceBinCandidate::accepted);

    py::class_<PySourceBinEstimate>(m, "SourceBinEstimate")
        .def_readonly("reference_source_bins", &PySourceBinEstimate::reference_source_bins)
        .def_readonly("recommended_source_bins", &PySourceBinEstimate::recommended_source_bins)
        .def_readonly("sampled_times", &PySourceBinEstimate::sampled_times)
        .def_readonly("candidates", &PySourceBinEstimate::candidates);

    py::class_<PyLimbDarkening>(m, "LimbDarkening")
        .def("__repr__", [](const PyLimbDarkening& ld) {
            return "LimbDarkening(c=" + std::to_string(ld.c) +
                ", d=" + std::to_string(ld.d) + ")";
        })
        .def(py::init([](double c, double d) {
            return PyLimbDarkening {c, d};
        }), py::arg("c") = 0.0, py::arg("d") = 0.0)
        .def_static("none", []() {
            return PyLimbDarkening {};
        })
        .def_static("linear", [](double c) {
            return PyLimbDarkening {c, 0.0};
        }, py::arg("c"))
        .def_static("quadratic", [](double c, double d) {
            return PyLimbDarkening {c, d};
        }, py::arg("c"), py::arg("d"))
        .def_readwrite("c", &PyLimbDarkening::c)
        .def_readwrite("d", &PyLimbDarkening::d);

    py::class_<PyEventCoordinates>(m, "EventCoordinates")
        .def("__repr__", [](const PyEventCoordinates& event) {
            return "EventCoordinates(ra=" + std::to_string(event.ra) +
                ", dec=" + std::to_string(event.dec) +
                ", tfix=" + std::to_string(event.tfix) +
                ", obs_lat=" + std::to_string(event.obs_lat) +
                ", obs_lon=" + std::to_string(event.obs_lon) + ")";
        })
        .def(py::init([](double ra, double dec, double tfix, double obs_lat, double obs_lon) {
            return PyEventCoordinates {ra, dec, tfix, obs_lat, obs_lon};
        }), py::arg("ra") = 0.0, py::arg("dec") = 0.0, py::arg("tfix") = 0.0,
            py::arg("obs_lat") = 0.0, py::arg("obs_lon") = 0.0)
        .def_readwrite("ra", &PyEventCoordinates::ra)
        .def_readwrite("dec", &PyEventCoordinates::dec)
        .def_readwrite("tfix", &PyEventCoordinates::tfix)
        .def_readwrite("obs_lat", &PyEventCoordinates::obs_lat)
        .def_readwrite("obs_lon", &PyEventCoordinates::obs_lon);

    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    py::class_<lcbi_params>(m, "LensParams")
        .def(py::init([](double t0,
                         double tE,
                         double umin,
                         double q,
                         double sep,
                         double theta,
                         double rho,
                         double omega,
                         double piEN,
                         double piEE,
                         double v_sep,
                         double q2,
                         double sep2,
                         double ang,
                         double ra,
                         double dec,
                         double tfix,
                         double limb_darkening_c,
                         double limb_darkening_d,
                         lcbi_orbital_motion_mode orbital_motion_mode,
                         double g1,
                         double g2,
                         double g3,
                         double lom_szs,
                         double lom_ar,
                         double u0,
                         double alpha) {
                 auto params = lcbi_default_params();
                 params.t0 = t0;
                 params.tE = tE;
                 params.umin = std::isnan(u0) ? umin : u0;
                 params.q = q;
                 params.sep = sep;
                 params.theta = std::isnan(alpha) ? theta : alpha;
                 params.rho = rho;
                 params.omega = omega;
                 params.piEN = piEN;
                 params.piEE = piEE;
                 params.v_sep = v_sep;
                 params.q2 = q2;
                 params.sep2 = sep2;
                 params.ang = ang;
                 params.ra = ra;
                 params.dec = dec;
                 params.tfix = tfix;
                 params.limb_darkening_c = limb_darkening_c;
                 params.limb_darkening_d = limb_darkening_d;
                 params.orbital_motion_mode = orbital_motion_mode;
                 params.g1 = g1;
                 params.g2 = g2;
                 params.g3 = g3;
                 params.lom_szs = lom_szs;
                 params.lom_ar = lom_ar;
                 return params;
             }),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("umin") = 0.0,
            py::arg("q") = 1.0,
            py::arg("sep") = 1.0,
            py::arg("theta") = 0.0,
            py::arg("rho") = 0.0,
            py::arg("omega") = 0.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("v_sep") = 0.0,
            py::arg("q2") = 0.0,
            py::arg("sep2") = 0.0,
            py::arg("ang") = 0.0,
            py::arg("ra") = 0.0,
            py::arg("dec") = 0.0,
            py::arg("tfix") = 0.0,
            py::arg("limb_darkening_c") = 0.0,
            py::arg("limb_darkening_d") = 0.0,
            py::arg("orbital_motion_mode") = LCBI_ORBIT_STATIC,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0,
            py::arg("u0") = kNaN,    // VBM alias for umin; takes priority when set
            py::arg("alpha") = kNaN) // VBM alias for theta; takes priority when set
        .def_readwrite("t0", &lcbi_params::t0)
        .def_readwrite("tE", &lcbi_params::tE)
        .def_readwrite("umin", &lcbi_params::umin)
        .def_readwrite("q", &lcbi_params::q)
        .def_readwrite("sep", &lcbi_params::sep)
        .def_readwrite("theta", &lcbi_params::theta)
        .def_readwrite("rho", &lcbi_params::rho)
        .def_readwrite("omega", &lcbi_params::omega)
        .def_readwrite("piEN", &lcbi_params::piEN)
        .def_readwrite("piEE", &lcbi_params::piEE)
        .def_readwrite("v_sep", &lcbi_params::v_sep)
        .def_readwrite("q2", &lcbi_params::q2)
        .def_readwrite("sep2", &lcbi_params::sep2)
        .def_readwrite("ang", &lcbi_params::ang)
        .def_readwrite("ra", &lcbi_params::ra)
        .def_readwrite("dec", &lcbi_params::dec)
        .def_readwrite("tfix", &lcbi_params::tfix)
        .def_readwrite("obs_lat", &lcbi_params::obs_lat)
        .def_readwrite("obs_lon", &lcbi_params::obs_lon)
        .def_readwrite("limb_darkening_c", &lcbi_params::limb_darkening_c)
        .def_readwrite("limb_darkening_d", &lcbi_params::limb_darkening_d)
        .def_readwrite("orbital_motion_mode", &lcbi_params::orbital_motion_mode)
        .def_readwrite("g1", &lcbi_params::g1)
        .def_readwrite("g2", &lcbi_params::g2)
        .def_readwrite("g3", &lcbi_params::g3)
        .def_readwrite("lom_szs", &lcbi_params::lom_szs)
        .def_readwrite("lom_ar", &lcbi_params::lom_ar)
        .def_property("u0",
            [](const lcbi_params& p) { return p.umin; },
            [](lcbi_params& p, double v) { p.umin = v; })
        .def_property("alpha",
            [](const lcbi_params& p) { return p.theta; },
            [](lcbi_params& p, double v) { p.theta = v; });

    py::class_<lcbi_options>(m, "Options")
        .def(py::init([](int source_bins,
                         const py::object& nbin,
                         const std::string& inverse_ray_grid,
                         int caustic_bins,
                         double grid_ratio,
                         const py::object& polar_nbin,
                         const py::object& polar_source_bins,
                         const py::object& polar_grid_ratio,
                         double point_source_threshold,
                         double hexadecapole_threshold,
                         double adaptive_hex_threshold,
                         int adaptive_source_bins,
                         int max_source_bins,
                         double finite_source_tol,
                         double finite_source_reltol,
                         double tol,
                         double reltol,
                         const std::string& coordinates,
                         double hex_tol,
                         const std::string& param_type,
                         const std::string& xallarap_param_type) {
                 auto options = public_default_options();
                 // param_type takes precedence over coordinates when explicitly set
                 if (!param_type.empty() && param_type != "auto") {
                     apply_param_type(options, param_type);
                 } else {
                     apply_coordinate_system(options, coordinates);
                 }
                 if (!xallarap_param_type.empty() && xallarap_param_type != "none") {
                     apply_xallarap_param_type(options, xallarap_param_type);
                 }
                 apply_inverse_ray_grid(options, inverse_ray_grid);
                 options.source_bins = optional_int_or(nbin, source_bins);
                 options.caustic_bins = caustic_bins;
                 options.grid_ratio = grid_ratio;
                 options.polar_source_bins = optional_int_or(
                     polar_nbin, optional_int_or(polar_source_bins, 0));
                 options.polar_grid_ratio = optional_double_or(polar_grid_ratio, 0.0);
                 options.point_source_threshold = point_source_threshold;
                 options.hexadecapole_threshold = hexadecapole_threshold;
                 options.adaptive_hex_threshold =
                     std::isnan(hex_tol) ? adaptive_hex_threshold : hex_tol;
                 options.max_source_bins = max_source_bins;
                 options.finite_source_tol = std::isnan(tol) ? finite_source_tol : tol;
                 options.finite_source_reltol = std::isnan(reltol) ? finite_source_reltol : reltol;
                 if (adaptive_source_bins < 0) {
                     options.adaptive_source_bins = 0;
                 } else {
                     options.adaptive_source_bins = adaptive_source_bins;
                 }
                 return options;
            }),
            py::arg("source_bins") = 50,
            py::arg("nbin") = py::none(),
            py::arg("inverse_ray_grid") = "auto",
            py::arg("caustic_bins") = 1400,
            py::arg("grid_ratio") = 4.0,
            py::arg("polar_nbin") = py::none(),
            py::arg("polar_source_bins") = py::none(),
            py::arg("polar_grid_ratio") = py::none(),
            py::arg("point_source_threshold") = 20.0,
            py::arg("hexadecapole_threshold") = 3.0,
            py::arg("adaptive_hex_threshold") = 0.001,
            py::arg("adaptive_source_bins") = -1,
            py::arg("max_source_bins") = 400,
            py::arg("finite_source_tol") = 0.0,
            py::arg("finite_source_reltol") = 0.0,
            py::arg("tol") = kNaN,
            py::arg("reltol") = kNaN,
            py::arg("coordinates") = "auto",
            py::arg("hex_tol") = kNaN,
            py::arg("param_type") = "auto",
            py::arg("xallarap_param_type") = "none")
        .def_readwrite("source_bins", &lcbi_options::source_bins)
        .def_property("nbin",
            [](const lcbi_options& o) { return o.source_bins; },
            [](lcbi_options& o, int value) { o.source_bins = value; })
        .def_property("inverse_ray_grid",
            &inverse_ray_grid_name,
            [](lcbi_options& o, const std::string& value) {
                apply_inverse_ray_grid(o, value);
            })
        .def_property_readonly("_mode", [](const lcbi_options& o) { return o.mode; })
        .def_readwrite("caustic_bins", &lcbi_options::caustic_bins)
        .def_readwrite("grid_ratio", &lcbi_options::grid_ratio)
        .def_property("polar_nbin",
            [](const lcbi_options& o) { return optional_positive_int(o.polar_source_bins); },
            [](lcbi_options& o, const py::object& value) {
                o.polar_source_bins = optional_int_or(value, 0);
            })
        .def_property("polar_source_bins",
            [](const lcbi_options& o) { return optional_positive_int(o.polar_source_bins); },
            [](lcbi_options& o, const py::object& value) {
                o.polar_source_bins = optional_int_or(value, 0);
            })
        .def_property("polar_grid_ratio",
            [](const lcbi_options& o) { return optional_positive_double(o.polar_grid_ratio); },
            [](lcbi_options& o, const py::object& value) {
                o.polar_grid_ratio = optional_double_or(value, 0.0);
            })
        .def_readwrite("point_source_threshold", &lcbi_options::point_source_threshold)
        .def_readwrite("hexadecapole_threshold", &lcbi_options::hexadecapole_threshold)
        .def_readwrite("adaptive_hex_threshold", &lcbi_options::adaptive_hex_threshold)
        .def_property("hex_tol",
            [](const lcbi_options& o) { return o.adaptive_hex_threshold; },
            [](lcbi_options& o, double v) { o.adaptive_hex_threshold = v; })
        .def_readwrite("adaptive_source_bins", &lcbi_options::adaptive_source_bins)
        .def_readwrite("max_source_bins", &lcbi_options::max_source_bins)
        .def_readwrite("finite_source_tol", &lcbi_options::finite_source_tol)
        .def_readwrite("finite_source_reltol", &lcbi_options::finite_source_reltol)
        .def_property("tol",
            [](const lcbi_options& o) { return o.finite_source_tol; },
            [](lcbi_options& o, double v) { o.finite_source_tol = v; })
        .def_property("reltol",
            [](const lcbi_options& o) { return o.finite_source_reltol; },
            [](lcbi_options& o, double v) { o.finite_source_reltol = v; })
        .def_property("coordinates",
            &coordinate_system_name,
            [](lcbi_options& o, const std::string& value) { apply_coordinate_system(o, value); })
        .def_property("param_type",
            &coordinate_system_name,
            [](lcbi_options& o, const std::string& value) { apply_param_type(o, value); })
        .def_property("xallarap_param_type",
            &xallarap_param_type_name,
            [](lcbi_options& o, const std::string& value) {
                apply_xallarap_param_type(o, value);
            });

    py::class_<PyLightCurveEvaluator>(m, "LightCurve")
        .def(py::init([](const std::string& lens,
                         const std::string& source,
                         const PyEventCoordinates& event,
                         const lcbi_options& options,
                         const PyLimbDarkening& limb_darkening,
                         lcbi_orbital_motion_mode orbital_motion_mode,
                         bool parallax,
                         bool terrestrial_parallax) {
                 return PyLightCurveEvaluator(
                     lens, source, event, options, limb_darkening, orbital_motion_mode,
                     parallax, terrestrial_parallax);
             }),
            py::kw_only(),
            py::arg("lens") = "binary_lens",
            py::arg("source") = "single",
            py::arg("event") = PyEventCoordinates {},
            py::arg("options") = public_default_options(),
            py::arg("limb_darkening") = PyLimbDarkening {},
            py::arg("orbital_motion_mode") = LCBI_ORBIT_STATIC,
            py::arg("parallax") = false,
            py::arg("terrestrial_parallax") = false)
        .def_property_readonly("lens", &PyLightCurveEvaluator::lens)
        .def_property_readonly("source", &PyLightCurveEvaluator::source)
        .def_property_readonly("event", &PyLightCurveEvaluator::event)
        .def_property_readonly("options", &PyLightCurveEvaluator::options)
        .def_property_readonly("limb_darkening", &PyLightCurveEvaluator::limb_darkening)
        .def_property_readonly("orbital_motion_mode", &PyLightCurveEvaluator::orbital_motion_mode)
        .def_property_readonly("parallax", &PyLightCurveEvaluator::parallax)
        .def_property_readonly("terrestrial_parallax", &PyLightCurveEvaluator::terrestrial_parallax)
        .def("__call__",
            &PyLightCurveEvaluator::light_curve_from_dict,
            py::arg("times"),
            py::arg("params"))
        .def("__call__",
            &PyLightCurveEvaluator::light_curve_static,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("__call__",
            &PyLightCurveEvaluator::light_curve,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("light_curve",
            &PyLightCurveEvaluator::light_curve_from_dict,
            py::arg("times"),
            py::arg("params"))
        .def("light_curve",
            &PyLightCurveEvaluator::light_curve_static,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("light_curve",
            &PyLightCurveEvaluator::light_curve,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("list",
            &PyLightCurveEvaluator::light_curve_list_static,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("list",
            &PyLightCurveEvaluator::light_curve_list,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("info",
            &PyLightCurveEvaluator::info_from_dict,
            py::arg("times"),
            py::arg("params"))
        .def("info",
            &PyLightCurveEvaluator::info_static,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("info",
            &PyLightCurveEvaluator::info,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("source_trajectory",
            &PyLightCurveEvaluator::source_trajectory_from_dict,
            py::arg("times"),
            py::arg("params"))
        .def("source_trajectory",
            &PyLightCurveEvaluator::source_trajectory_static,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("source_trajectory",
            &PyLightCurveEvaluator::source_trajectory,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("caustics",
            &PyLightCurveEvaluator::caustics_from_dict,
            py::arg("params"),
            py::arg("n_points") = 1000,
            "Compute caustics from a parameter dict. For triple_lens, pass "
            "params containing s, q, q2, sep2, ang.")
        .def("caustics",
            &PyLightCurveEvaluator::caustics,
            py::kw_only(),
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("n_points") = 1400,
            py::arg("time") = std::numeric_limits<double>::quiet_NaN(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("critical_curves",
            &PyLightCurveEvaluator::critical_curves,
            py::kw_only(),
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("n_points") = 1400,
            py::arg("time") = std::numeric_limits<double>::quiet_NaN(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("magnification",
            &PyLightCurveEvaluator::magnification_from_dict,
            py::arg("time"),
            py::arg("params"))
        .def("magnification",
            &PyLightCurveEvaluator::magnification_static,
            py::arg("time"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("magnification",
            &PyLightCurveEvaluator::magnification,
            py::arg("time"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0);

    m.attr("LightCurveFunc") = m.attr("LightCurve");

    py::class_<PyParallaxLightCurveFunc>(m, "_ParallaxLightCurveKernel")
        .def(py::init([](const std::string& lens,
                         const PyEventCoordinates& event,
                         const lcbi_options& options,
                         const PyLimbDarkening& limb_darkening,
                         lcbi_orbital_motion_mode orbital_motion_mode) {
                 return PyParallaxLightCurveFunc(
                     lens, event, options, limb_darkening, orbital_motion_mode);
             }),
            py::kw_only(),
            py::arg("lens") = "binary_lens",
            py::arg("event") = PyEventCoordinates {},
            py::arg("options") = public_default_options(),
            py::arg("limb_darkening") = PyLimbDarkening {},
            py::arg("orbital_motion_mode") = LCBI_ORBIT_STATIC)
        .def_property_readonly("lens", &PyParallaxLightCurveFunc::lens)
        .def_property_readonly("event", &PyParallaxLightCurveFunc::event)
        .def_property_readonly("options", &PyParallaxLightCurveFunc::options)
        .def_property_readonly("limb_darkening", &PyParallaxLightCurveFunc::limb_darkening)
        .def_property_readonly("orbital_motion_mode", &PyParallaxLightCurveFunc::orbital_motion_mode)
        .def_property_readonly("parallax", &PyParallaxLightCurveFunc::parallax)
        .def("__call__",
            &PyParallaxLightCurveFunc::light_curve,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("light_curve",
            &PyParallaxLightCurveFunc::light_curve,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("list",
            &PyParallaxLightCurveFunc::light_curve_list,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("info",
            &PyParallaxLightCurveFunc::info,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("source_trajectory",
            &PyParallaxLightCurveFunc::source_trajectory,
            py::arg("times"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("caustics",
            &PyParallaxLightCurveFunc::caustics,
            py::kw_only(),
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("n_points") = 1400,
            py::arg("time") = std::numeric_limits<double>::quiet_NaN(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("critical_curves",
            &PyParallaxLightCurveFunc::critical_curves,
            py::kw_only(),
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("n_points") = 1400,
            py::arg("time") = std::numeric_limits<double>::quiet_NaN(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0)
        .def("magnification",
            &PyParallaxLightCurveFunc::magnification,
            py::arg("time"),
            py::kw_only(),
            py::arg("t0") = 0.0,
            py::arg("tE") = 1.0,
            py::arg("u0") = 0.0,
            py::arg("alpha") = 0.0,
            py::arg("s") = 1.0,
            py::arg("q") = 1.0,
            py::arg("rho") = 0.0,
            py::arg("piEN") = 0.0,
            py::arg("piEE") = 0.0,
            py::arg("g1") = 0.0,
            py::arg("g2") = 0.0,
            py::arg("g3") = 0.0,
            py::arg("lom_szs") = 0.0,
            py::arg("lom_ar") = 1.0);

    m.attr("ParallaxLightCurve") = m.attr("LightCurve");
    m.attr("ParallaxLightCurveFunc") = m.attr("LightCurve");

    auto light_curve_array = [](py::array_t<double, py::array::c_style | py::array::forcecast> times,
           double t0,
           double tE,
           double u0,
           double alpha,
           double s,
           double q,
           double rho,
           const PyLimbDarkening& limb_darkening,
           const PyEventCoordinates& event,
           const lcbi_options& options,
           double piEN,
           double piEE,
           double g1,
           double g2,
           double g3,
           lcbi_orbital_motion_mode orbital_motion_mode,
           double lom_szs,
           double lom_ar) {
            const auto params = make_binary_params(
                t0, tE, u0, alpha, s, q, rho, limb_darkening, event,
                piEN, piEE, g1, g2, g3, orbital_motion_mode, lom_szs, lom_ar);
            return evaluate_binary_light_curve_numpy(times, params, options);
        };

    auto light_curve_list = [](const std::vector<double>& times,
           double t0,
           double tE,
           double u0,
           double alpha,
           double s,
           double q,
           double rho,
           const PyLimbDarkening& limb_darkening,
           const PyEventCoordinates& event,
           const lcbi_options& options,
           double piEN,
           double piEE,
           double g1,
           double g2,
           double g3,
           lcbi_orbital_motion_mode orbital_motion_mode,
           double lom_szs,
           double lom_ar) {
            const auto params = make_binary_params(
                t0, tE, u0, alpha, s, q, rho, limb_darkening, event,
                piEN, piEE, g1, g2, g3, orbital_motion_mode, lom_szs, lom_ar);
            return evaluate_binary_light_curve(times, params, options);
        };

    auto light_curve_info = [](const std::vector<double>& times,
           double t0,
           double tE,
           double u0,
           double alpha,
           double s,
           double q,
           double rho,
           const PyLimbDarkening& limb_darkening,
           const PyEventCoordinates& event,
           const lcbi_options& options,
           double piEN,
           double piEE,
           double g1,
           double g2,
           double g3,
           lcbi_orbital_motion_mode orbital_motion_mode,
           double lom_szs,
           double lom_ar) {
            const auto params = make_binary_params(
                t0, tE, u0, alpha, s, q, rho, limb_darkening, event,
                piEN, piEE, g1, g2, g3, orbital_motion_mode, lom_szs, lom_ar);
            return evaluate_binary_light_curve_info(times, params, options);
        };

    auto point_magnification = [](double time,
           double t0,
           double tE,
           double u0,
           double alpha,
           double s,
           double q,
           double rho,
           const PyLimbDarkening& limb_darkening,
           const PyEventCoordinates& event,
           const lcbi_options& options,
           double piEN,
           double piEE,
           double g1,
           double g2,
           double g3,
           lcbi_orbital_motion_mode orbital_motion_mode,
           double lom_szs,
           double lom_ar) {
            const auto params = make_binary_params(
                t0, tE, u0, alpha, s, q, rho, limb_darkening, event,
                piEN, piEE, g1, g2, g3, orbital_motion_mode, lom_szs, lom_ar);
            return evaluate_binary_magnification(time, params, options);
        };

    m.def("light_curve",
        light_curve_array,
        py::arg("times"),
        py::kw_only(),
        py::arg("t0") = 0.0,
        py::arg("tE") = 1.0,
        py::arg("u0") = 0.0,
        py::arg("alpha") = 0.0,
        py::arg("s") = 1.0,
        py::arg("q") = 1.0,
        py::arg("rho") = 0.0,
        py::arg("limb_darkening") = PyLimbDarkening {},
        py::arg("event") = PyEventCoordinates {},
        py::arg("options") = public_default_options(),
        py::arg("piEN") = 0.0,
        py::arg("piEE") = 0.0,
        py::arg("g1") = 0.0,
        py::arg("g2") = 0.0,
        py::arg("g3") = 0.0,
        py::arg("orbital_motion_mode") = LCBI_ORBIT_STATIC,
        py::arg("lom_szs") = 0.0,
        py::arg("lom_ar") = 1.0,
        R"pbdoc(
Evaluate a binary-lens light curve with named physical parameters.

Returns a NumPy array. This is the recommended high-level API for ordinary use.
)pbdoc");

    m.def("binary_light_curve",
        light_curve_list,
        py::arg("times"),
        py::kw_only(),
        py::arg("t0") = 0.0,
        py::arg("tE") = 1.0,
        py::arg("u0") = 0.0,
        py::arg("alpha") = 0.0,
        py::arg("s") = 1.0,
        py::arg("q") = 1.0,
        py::arg("rho") = 0.0,
        py::arg("limb_darkening") = PyLimbDarkening {},
        py::arg("event") = PyEventCoordinates {},
        py::arg("options") = public_default_options(),
        py::arg("piEN") = 0.0,
        py::arg("piEE") = 0.0,
        py::arg("g1") = 0.0,
        py::arg("g2") = 0.0,
        py::arg("g3") = 0.0,
        py::arg("orbital_motion_mode") = LCBI_ORBIT_STATIC,
        py::arg("lom_szs") = 0.0,
        py::arg("lom_ar") = 1.0,
        R"pbdoc(
Compatibility wrapper returning a Python list. Prefer light_curve for new code.
)pbdoc");

    m.def("light_curve_info",
        light_curve_info,
        py::arg("times"),
        py::kw_only(),
        py::arg("t0") = 0.0,
        py::arg("tE") = 1.0,
        py::arg("u0") = 0.0,
        py::arg("alpha") = 0.0,
        py::arg("s") = 1.0,
        py::arg("q") = 1.0,
        py::arg("rho") = 0.0,
        py::arg("limb_darkening") = PyLimbDarkening {},
        py::arg("event") = PyEventCoordinates {},
        py::arg("options") = public_default_options(),
        py::arg("piEN") = 0.0,
        py::arg("piEE") = 0.0,
        py::arg("g1") = 0.0,
        py::arg("g2") = 0.0,
        py::arg("g3") = 0.0,
        py::arg("orbital_motion_mode") = LCBI_ORBIT_STATIC,
        py::arg("lom_szs") = 0.0,
        py::arg("lom_ar") = 1.0,
        R"pbdoc(
Evaluate a light curve and return diagnostic metadata.

For ordinary fitting, prefer light_curve, which returns only the magnification array.
)pbdoc");

    m.def("magnification",
        point_magnification,
        py::arg("time"),
        py::kw_only(),
        py::arg("t0") = 0.0,
        py::arg("tE") = 1.0,
        py::arg("u0") = 0.0,
        py::arg("alpha") = 0.0,
        py::arg("s") = 1.0,
        py::arg("q") = 1.0,
        py::arg("rho") = 0.0,
        py::arg("limb_darkening") = PyLimbDarkening {},
        py::arg("event") = PyEventCoordinates {},
        py::arg("options") = public_default_options(),
        py::arg("piEN") = 0.0,
        py::arg("piEE") = 0.0,
        py::arg("g1") = 0.0,
        py::arg("g2") = 0.0,
        py::arg("g3") = 0.0,
        py::arg("orbital_motion_mode") = LCBI_ORBIT_STATIC,
        py::arg("lom_szs") = 0.0,
        py::arg("lom_ar") = 1.0,
        R"pbdoc(
Evaluate one binary-lens magnification with named physical parameters.

For a full light curve, use light_curve so
the C++ side can keep per-curve state local to the loop.
)pbdoc");

    m.def("binary_magnification",
        point_magnification,
        py::arg("time"),
        py::kw_only(),
        py::arg("t0") = 0.0,
        py::arg("tE") = 1.0,
        py::arg("u0") = 0.0,
        py::arg("alpha") = 0.0,
        py::arg("s") = 1.0,
        py::arg("q") = 1.0,
        py::arg("rho") = 0.0,
        py::arg("limb_darkening") = PyLimbDarkening {},
        py::arg("event") = PyEventCoordinates {},
        py::arg("options") = public_default_options(),
        py::arg("piEN") = 0.0,
        py::arg("piEE") = 0.0,
        py::arg("g1") = 0.0,
        py::arg("g2") = 0.0,
        py::arg("g3") = 0.0,
        py::arg("orbital_motion_mode") = LCBI_ORBIT_STATIC,
        py::arg("lom_szs") = 0.0,
        py::arg("lom_ar") = 1.0,
        "Compatibility wrapper for magnification.");

    m.def("polynomial_roots", [](const std::vector<std::complex<double>>& coefficients) {
        lcbinint::math::PolynomialRootSolver solver;
        auto result = solver.solve(coefficients);
        if (result.status == lcbinint::math::RootSolverStatus::invalid_polynomial) {
            throw py::value_error("invalid polynomial");
        }
        if (result.status == lcbinint::math::RootSolverStatus::unsupported_degree) {
            throw py::value_error("unsupported polynomial degree");
        }
        return result.roots;
    }, py::arg("coefficients"),
        "Roots of a complex polynomial with constant-first coefficients.");

    m.def("binary_mag0", [](double separation, double mass_ratio, double y1, double y2) {
        lcbinint::magnification::PointSourceMagnifier magnifier;
        const auto result = magnifier.binary_mag0(separation, mass_ratio, {y1, y2});
        return result.magnification;
    }, py::arg("separation"), py::arg("mass_ratio"), py::arg("y1"), py::arg("y2"),
        "Binary point-source magnification matching VBMicrolensing BinaryMag0 coordinates.");

    m.def("binary_inverse_ray",
        [](double s, double q, double x, double y, double rho,
           const PyLimbDarkening& ld, const lcbi_options& options) -> double {
            const auto model_options = lcbinint::model::from_c_options(&options);
            auto settings = make_finite_source_settings(ld, model_options);
            settings.kinji_threshold = 1e18;
            settings.hex_threshold = 0.0;
            settings.adaptive_hex_threshold = 0.0;
            lcbinint::magnification::PointSourceMagnifier point_magnifier;
            lcbinint::magnification::FiniteSourceMagnifier finite_magnifier(settings);
            const lcbinint::SourcePosition source {x, y};
            py::gil_scoped_release release;
            const auto point = point_magnifier.binary_mag0(s, q, source);
            const auto result = finite_magnifier.binary_mag(
                s, q, source, std::abs(rho), point.magnification, nullptr, true);
            return result.magnification;
        },
        py::arg("s"), py::arg("q"), py::arg("x"), py::arg("y"), py::arg("rho"),
        py::arg("limb_darkening") = PyLimbDarkening {},
        py::arg("options") = public_default_options(),
        "Binary finite-source magnification at source position (x, y) using inverse-ray shooting.\n\n"
        "Same (s, q, x, y, rho) coordinate convention as VBMicrolensing BinaryMag2.");

    m.def("triple_inverse_ray",
        [](double s, double q, double q2, double sep2, double ang,
           double x, double y, double rho,
           const std::string& frame,
           const PyLimbDarkening& ld, const lcbi_options& options) -> double {
            const auto model_options = lcbinint::model::from_c_options(&options);
            auto settings = make_finite_source_settings(ld, model_options);
            settings.kinji_threshold = 1e18;
            settings.hex_threshold = 0.0;
            settings.adaptive_hex_threshold = 0.0;
            lcbinint::magnification::PointSourceMagnifier point_magnifier;
            lcbinint::magnification::FiniteSourceMagnifier finite_magnifier(settings);
            const auto geometry =
                lcbinint::model::make_triple_lens_geometry(s, q, q2, sep2, ang);
            lcbinint::SourcePosition source {x, y};
            if (frame == "vbm") {
                // VBM frame: origin at 2-body COM of lens1+lens2, x-axis along lens1→lens2
                const auto& z1 = geometry.lens_positions[0];
                const auto& z2 = geometry.lens_positions[1];
                const double e1 = geometry.masses[0];
                const double e2 = geometry.masses[1];
                const double com12x = (e1 * z1.x + e2 * z2.x) / (e1 + e2);
                const double com12y = (e1 * z1.y + e2 * z2.y) / (e1 + e2);
                const double a12 = std::atan2(z2.y - z1.y, z2.x - z1.x);
                const double ca = std::cos(a12);
                const double sa = std::sin(a12);
                source.x = com12x + x * ca - y * sa;
                source.y = com12y + x * sa + y * ca;
            } else if (frame != "lcbi" && frame != "lcbinint") {
                throw py::value_error("frame must be 'vbm' or 'lcbi'");
            }
            py::gil_scoped_release release;
            const auto point = point_magnifier.triple_mag0(geometry, source);
            const auto result = finite_magnifier.triple_mag(
                geometry, source, std::abs(rho), point.magnification, &point_magnifier);
            return result.magnification;
        },
        py::arg("s"), py::arg("q"), py::arg("q2"), py::arg("sep2"), py::arg("ang"),
        py::arg("x"), py::arg("y"), py::arg("rho"),
        py::arg("frame") = "vbm",
        py::arg("limb_darkening") = PyLimbDarkening {},
        py::arg("options") = public_default_options(),
        "Triple-lens finite-source magnification at source position (x, y) using inverse-ray shooting.\n\n"
        "frame='vbm' (default): (x,y) in VBM frame — origin at 2-body COM of lens1+lens2, "
        "x-axis along lens1→lens2.\n"
        "frame='lcbi': (x,y) in 3-body center-of-mass frame.");

    m.def("circular_orbital_motion", [](double time,
                                         double separation,
                                         double angle,
                                         double g1,
                                         double g2,
                                         double g3,
                                         double reference_time) {
        const auto state = lcbinint::model::circular_orbital_motion_3d(
            time, separation, angle, g1, g2, g3, reference_time);
        return py::make_tuple(state.separation, state.angle, state.line_of_sight_separation);
    }, py::arg("time"), py::arg("separation"), py::arg("angle"), py::arg("g1") = 0.0,
        py::arg("g2") = 0.0, py::arg("g3") = 0.0, py::arg("reference_time") = 0.0,
        "VBM-compatible circular 3D orbital-motion state.");

    m.def("kepler_orbital_motion", [](double time,
                                       double separation,
                                       double angle,
                                       double g1,
                                       double g2,
                                       double g3,
                                       double szs,
                                       double ar,
                                       double reference_time) {
        const auto state = lcbinint::model::kepler_orbital_motion_3d(
            time, separation, angle, g1, g2, g3, szs, ar, reference_time);
        return py::make_tuple(state.separation, state.angle, state.line_of_sight_separation);
    }, py::arg("time"), py::arg("separation"), py::arg("angle"), py::arg("g1") = 0.0,
        py::arg("g2") = 0.0, py::arg("g3") = 0.0, py::arg("szs") = 0.0,
        py::arg("ar") = 1.0, py::arg("reference_time") = 0.0,
        "VBM-compatible Keplerian 3D orbital-motion state.");

    // --- New architecture submodules ---
    register_obs_submodule(m);
    register_bayes_submodule(m);
    register_optimize_submodule(m);
    register_sample_submodule(m);
}
