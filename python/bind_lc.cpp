#include "bind_lc.hpp"
#include "lcbinint/lcbinint.h"
#include "lcbinint/lc/model.hpp"
#include "lcbinint/lc/light_curve.hpp"
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/obs/coordinates.hpp"
#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/model/orbital_motion.hpp"
#include "lcbinint/model/trajectory.hpp"
#include "lcbinint/model/triple_lens_geometry.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <optional>
#include <vector>

namespace py = pybind11;

struct PyLimbDarkening {
    double c = 0.0;
    double d = 0.0;
};

struct PyLightCurveInfo {
    std::vector<double> times;
    std::vector<double> magnifications;
    std::vector<double> point_source_magnifications;
    std::vector<double> finite_source_magnifications;
    std::vector<double> source_x;
    std::vector<double> source_y;
    std::vector<int> image_counts;
    std::vector<int> finite_source_methods;
    std::vector<std::string> finite_source_method_names;
    std::vector<double> finite_source_error_estimates;
    std::vector<int> finite_source_refinement_levels;
    std::vector<bool> finite_source_converged;
    std::vector<std::string> statuses;
    std::vector<int> root_candidate_counts;
    std::vector<int> root_duplicate_counts;
    std::vector<int> root_polish_failure_counts;
    std::vector<int> root_used_warm_start;
    std::vector<int> root_used_cold_retry;
    std::vector<int> root_used_high_precision;
    std::vector<int> root_needs_high_precision;
    std::vector<double> root_max_residuals;
    std::vector<double> point_source_quadrupole_indicators;
    std::vector<double> point_source_cusp_indicators;
    std::vector<double> point_source_ghost_indicators;
    std::vector<double> point_source_planetary_distances2;
    std::vector<double> point_source_safety_tolerances;
    std::vector<int> point_source_ghost_counts;
    std::vector<int> point_source_safety_flags;
    std::vector<double> separations;
    std::vector<double> mass_ratios;
    std::vector<double> caustic_distances;
    bool all_converged = true;
    std::vector<int> unconverged_indices;
};

struct PySourceTrajectory {
    std::vector<double> times;
    std::vector<double> x;
    std::vector<double> y;
};

struct PyBinarySourceComponent {
    std::vector<double> magnification;
    PySourceTrajectory trajectory;
};

struct PyBinarySourceComponents {
    PyBinarySourceComponent source1;
    PyBinarySourceComponent source2;
    std::vector<double> total;
    double flux_ratio = 1.0;
};

struct PyFiniteSourceGeometry {
    std::vector<double> separation;
    std::vector<double> mass_ratio;
    std::vector<double> source_x;
    std::vector<double> source_y;
    std::vector<double> source_radius;
    std::vector<double> limb_darkening_c;
    std::vector<double> limb_darkening_d;
    std::vector<double> absolute_tolerance;
    std::vector<double> relative_tolerance;
};

struct PyGeometryBranches {
    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;
};

struct PyImagePoint {
    double x = 0.0;
    double y = 0.0;
    double jacobian_determinant = 0.0;
    double magnification = 0.0;
    int parity = 0;
};

namespace {

// Map param_type string → vbm_compatible + center_of_mass (same logic as old API).
void apply_param_type(lcbi_options& o, const std::string& pt)
{
    if (pt == "auto" || pt == "vbm" || pt == "vbbl" || pt == "standard") {
        o.vbm_compatible = 1; o.center_of_mass = 0;
    } else if (pt == "lcbinint" || pt == "original" || pt == "legacy") {
        o.vbm_compatible = 0; o.center_of_mass = 0;
    } else if (pt == "center_of_mass") {
        o.vbm_compatible = 0; o.center_of_mass = 1;
    } else if (pt == "vbm_center_of_mass") {
        o.vbm_compatible = 1; o.center_of_mass = 1;
    } else {
        throw std::invalid_argument(
            "param_type must be 'vbm', 'lcbinint', 'center_of_mass', or 'vbm_center_of_mass'");
    }
}

void apply_inverse_ray_grid(lcbi_options& o, const std::string& grid)
{
    if (grid == "cartesian") {
        o.mode = 1;
    } else if (grid == "polar") {
        o.mode = 2;
    } else if (grid == "auto") {
        o.mode = 4;
    } else {
        throw std::invalid_argument("inverse_ray_grid must be 'cartesian', 'polar', or 'auto'");
    }
}

void apply_nbin(lcbi_options& o, const py::object& value)
{
    if (py::isinstance<py::str>(value)) {
        const auto mode = value.cast<std::string>();
        if (mode != "auto") {
            throw std::invalid_argument("nbin must be a positive integer or 'auto'");
        }
        o.automatic_source_bins = 1;
        return;
    }
    const int bins = value.cast<int>();
    if (bins <= 0) {
        throw std::invalid_argument("nbin must be a positive integer or 'auto'");
    }
    o.source_bins = bins;
    o.automatic_source_bins = 0;
}

std::string inverse_ray_grid_from_mode(int mode)
{
    if (mode == 1) return "cartesian";
    if (mode == 2) return "polar";
    if (mode == 4) return "auto";
    return "mode_" + std::to_string(mode);
}

// Build lcbi_params from a Python dict (or py::kwargs).
// Supports both canonical names (umin, theta, sep) and friendly aliases (u0, alpha, s).
lcbi_params params_from_dict(const py::dict& d)
{
    auto p = lcbi_default_params();
	    for (auto& item : d) {
	        const std::string key = item.first.cast<std::string>();
	        if (key == "orbital_motion_mode") {
	            p.orbital_motion_mode = item.second.cast<lcbi_orbital_motion_mode>();
	            continue;
	        }
	        // Most params are double; handle int-typed ones below if needed
	        const double val = item.second.cast<double>();
        if      (key == "t0"   || key == "t_0")  p.t0    = val;
        else if (key == "tE"   || key == "t_E")  p.tE    = val;
        else if (key == "u0"   || key == "umin")  p.umin  = val;
        else if (key == "alpha"|| key == "theta") p.theta = val;
        else if (key == "s"    || key == "sep")   p.sep   = val;
        else if (key == "q")                       p.q     = val;
        else if (key == "rho" || key == "rho1")  p.rho   = val;
        else if (key == "piEN")                    p.piEN  = val;
        else if (key == "piEE")                    p.piEE  = val;
        else if (key == "q2")                      p.q2    = val;
        else if (key == "sep2")                    p.sep2  = val;
        else if (key == "ang")                     p.ang   = val;
        else if (key == "ra")                      p.ra    = val;
        else if (key == "dec")                     p.dec   = val;
        else if (key == "tfix")                    p.tfix  = val;
        else if (key == "obs_lat")                 p.obs_lat = val;
        else if (key == "obs_lon")                 p.obs_lon = val;
        else if (key == "limb_darkening_c")        p.limb_darkening_c = val;
        else if (key == "limb_darkening_d")        p.limb_darkening_d = val;
        else if (key == "g1")                      p.g1    = val;
        else if (key == "g2")                      p.g2    = val;
        else if (key == "g3")                      p.g3    = val;
        else if (key == "lom_szs")                 p.lom_szs = val;
        else if (key == "lom_ar")                  p.lom_ar  = val;
        else if (key == "v_sep")                   p.v_sep = val;
        // xallarap amplitude/position (all modes)
        else if (key == "xi_1")                    p.xi_1  = val;
        else if (key == "xi_2")                    p.xi_2  = val;
        // orbital_elements / circular_elements: period-based orbit params
        else if (key == "period_xa")               p.period_xa = val;
        else if (key == "ecc_xa")                  p.ecc_xa    = val;
        else if (key == "peri_xa")                 p.peri_xa   = val;
        else if (key == "inc_xa")                  p.inc_xa    = val;
        // circular_velocity / kepler_velocity: w1/w2/w3 (mapped to omega/inc/phi fields)
        else if (key == "w1" || key == "omega_xa") p.omega_xa = val;
        else if (key == "w2" || key == "inc_xa")   p.inc_xa   = val;
        else if (key == "w3" || key == "phi_xa")   p.phi_xa   = val;
        // kepler_velocity: xa_szs/xa_ar (mapped to piEN_xa/piEE_xa fields)
        else if (key == "xa_szs")                  p.piEN_xa = val;
        else if (key == "xa_ar")                   p.piEE_xa = val;
        // Binary-source parameters are handled by binary_source_parameter_row.
        else if (key == "t0_2" || key == "u0_2" || key == "rho2" ||
                 key == "flux_ratio" || key == "source_mass_ratio") { /* skip */ }
        else {
            throw py::key_error("lcbinint: unknown parameter '" + key + "'");
        }
    }
    return p;
}

// Return a numpy array from a vector, transferring ownership via capsule.
py::array_t<double> vec_to_numpy(std::vector<double> mags)
{
    auto* heap = new std::vector<double>(std::move(mags));
    py::capsule cap(heap, [](void* p) {
        delete static_cast<std::vector<double>*>(p);
    });
    return py::array_t<double>(
        {static_cast<py::ssize_t>(heap->size())},
        {sizeof(double)},
        heap->data(),
        cap);
}

py::array_t<double> matrix_to_numpy(
    std::vector<double> values, py::ssize_t rows, py::ssize_t columns)
{
    auto* heap = new std::vector<double>(std::move(values));
    py::capsule cap(heap, [](void* p) {
        delete static_cast<std::vector<double>*>(p);
    });
    return py::array_t<double>(
        {rows, columns},
        {static_cast<py::ssize_t>(sizeof(double) * columns),
         static_cast<py::ssize_t>(sizeof(double))},
        heap->data(), cap);
}

// Core dispatch: build times vector, release GIL, compute, return numpy array.
using TimesArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<double> times_from_array(const TimesArray& times)
{
    const auto buffer = times.request();
    if (buffer.ndim == 0) {
        return {*static_cast<const double*>(buffer.ptr)};
    }
    if (buffer.ndim != 1) {
        throw std::invalid_argument("times must be one-dimensional");
    }
    const auto* pointer = static_cast<const double*>(buffer.ptr);
    return {pointer, pointer + buffer.size};
}

py::array_t<double> compute(
    const lcbinint::lc::LightCurve& lc,
    const TimesArray&               times,
    const lcbi_params&              params)
{
    const std::vector<double> tv = times_from_array(times);
    std::vector<double> mags;
    {
        py::gil_scoped_release release;
        mags = lc.magnification(tv, params);
    }
    return vec_to_numpy(std::move(mags));
}

std::vector<lcbinint::model::MagnificationExecutionPlan> execution_plan_from(
    const std::vector<int>& methods,
    const std::vector<int>& resolutions)
{
    if (methods.size() != resolutions.size()) {
        throw std::invalid_argument(
            "warm-up methods and resolutions must have equal length");
    }
    std::vector<lcbinint::model::MagnificationExecutionPlan> plan;
    plan.reserve(methods.size());
    for (std::size_t i = 0; i < methods.size(); ++i) {
        if (methods[i] < 0 || methods[i] > 5) {
            throw std::invalid_argument("invalid warm-up finite-source method");
        }
        plan.push_back({
            static_cast<lcbinint::magnification::FiniteSourceMethod>(methods[i]),
            resolutions[i],
        });
    }
    return plan;
}

py::array_t<double> compute_preplanned(
    const lcbinint::lc::LightCurve& lc,
    const TimesArray& times,
    const lcbi_params& params,
    const std::vector<int>& methods,
    const std::vector<int>& resolutions)
{
    const std::vector<double> tv = times_from_array(times);
    const auto plan = execution_plan_from(methods, resolutions);
    std::vector<double> magnifications;
    {
        py::gil_scoped_release release;
        magnifications = lc.magnification_preplanned(tv, params, plan);
    }
    return vec_to_numpy(std::move(magnifications));
}

py::dict compute_preplanned_diagnostic(
    const lcbinint::lc::LightCurve& lc,
    const TimesArray& times,
    const lcbi_params& params,
    const std::vector<int>& methods,
    const std::vector<int>& resolutions)
{
    const std::vector<double> tv = times_from_array(times);
    const auto plan = execution_plan_from(methods, resolutions);
    std::vector<lcbinint::MagnificationResult> results;
    std::vector<double> epoch_seconds;
    {
        py::gil_scoped_release release;
        results = lc.evaluate_preplanned_diagnostic(
            tv, params, plan, &epoch_seconds);
    }
    std::vector<double> magnifications;
    std::vector<double> errors;
    std::vector<int> converged;
    std::vector<int> actual_methods;
    magnifications.reserve(results.size());
    errors.reserve(results.size());
    converged.reserve(results.size());
    actual_methods.reserve(results.size());
    for (const auto& result : results) {
        magnifications.push_back(result.finite_source_magnification);
        errors.push_back(result.finite_source_error_estimate);
        converged.push_back(
            result.status == lcbinint::EvaluationStatus::ok &&
            result.finite_source_converged);
        actual_methods.push_back(result.finite_source_method);
    }
    py::dict output;
    output["magnification"] = vec_to_numpy(std::move(magnifications));
    output["error_estimate"] = vec_to_numpy(std::move(errors));
    output["converged"] = py::cast(std::move(converged));
    output["method"] = py::cast(std::move(actual_methods));
    output["seconds"] = vec_to_numpy(std::move(epoch_seconds));
    return output;
}

lcbi_params inference_params_from_object(const py::handle& item)
{
    const auto object = py::reinterpret_borrow<py::object>(item);
    if (!py::isinstance<py::dict>(object)) {
        return object.cast<lcbi_params>();
    }
    const py::dict input = object.cast<py::dict>();
    py::dict filtered;
    for (const auto& entry : input) {
        const std::string key = entry.first.cast<std::string>();
        if (key.rfind("Fs_", 0) == 0 || key.rfind("Fb_", 0) == 0) continue;
        filtered[entry.first] = entry.second;
    }
    return params_from_dict(filtered);
}

struct InferenceParameterRow {
    lcbi_params primary;
    lcbi_params secondary;
    double source_ratio = 1.0;
};

InferenceParameterRow binary_source_parameter_row(
    const py::dict& values,
    const lcbinint::lc::Model& model)
{
    InferenceParameterRow row;
    row.primary = params_from_dict(values);
    row.secondary = row.primary;

    bool has_rho1 = false;
    bool has_t0_2 = false;
    bool has_u0_2 = false;
    bool has_rho2 = false;
    bool has_flux_ratio = false;
    bool has_mass_ratio = false;
    double source_mass_ratio = 0.0;
    double rho1 = 0.0;
    double rho2 = 0.0;
    for (const auto& entry : values) {
        const std::string key = entry.first.cast<std::string>();
        if (key == "rho") {
            throw std::invalid_argument("binary source uses rho1, not rho");
        }
        if (key == "rho1") {
            rho1 = entry.second.cast<double>();
            has_rho1 = true;
        }
        else if (key == "t0_2") {
            row.secondary.t0 = entry.second.cast<double>();
            has_t0_2 = true;
        } else if (key == "u0_2") {
            row.secondary.umin = entry.second.cast<double>();
            has_u0_2 = true;
        } else if (key == "rho2") {
            rho2 = entry.second.cast<double>();
            row.secondary.rho = rho2;
            has_rho2 = true;
        } else if (key == "flux_ratio") {
            row.source_ratio = entry.second.cast<double>();
            has_flux_ratio = true;
        } else if (key == "source_mass_ratio") {
            source_mass_ratio = entry.second.cast<double>();
            has_mass_ratio = true;
        }
    }
    const bool has_xallarap = model.xallarap != LCBI_XALLARAP_NONE;
    const bool offset_coordinates =
        model.source_orbit_coordinates == lcbinint::lc::SourceOrbitCoordinates::trajectory_offset;
    const bool needs_secondary_track = !has_xallarap || offset_coordinates;
    if (!(has_rho1 && has_rho2 && has_flux_ratio) ||
        (needs_secondary_track && !(has_t0_2 && has_u0_2)) ||
        (has_xallarap && !has_mass_ratio)) {
        throw std::invalid_argument(
            "binary source requires rho1, rho2, flux_ratio, and the parameters for its selected orbit coordinates");
    }
    if (!(std::isfinite(row.primary.rho) && row.primary.rho >= 0.0 &&
          std::isfinite(row.secondary.rho) && row.secondary.rho >= 0.0 &&
          std::isfinite(row.source_ratio) && row.source_ratio >= 0.0)) {
        throw std::invalid_argument(
            "binary-source rho1, rho2, and flux_ratio must be finite and non-negative");
    }
    if (has_xallarap && !(std::isfinite(source_mass_ratio) && source_mass_ratio > 0.0)) {
        throw std::invalid_argument("source_mass_ratio must be finite and positive");
    }
    if (!has_xallarap) return row;

    if (offset_coordinates) {
        // t0/u0 and t0_2/u0_2 specify the two tangent trajectories. Convert
        // their projected separation at t_ref into the CoM state required by
        // the velocity xallarap propagators.
        const double q = source_mass_ratio;
        const double inv_total_mass = 1.0 / (1.0 + q);
        const double relative_tau = (row.primary.t0 - row.secondary.t0) / row.primary.tE;
        const double relative_beta = row.secondary.umin - row.primary.umin;
        row.primary.t0 = (row.primary.t0 + q * row.secondary.t0) * inv_total_mass;
        row.primary.umin = (row.primary.umin + q * row.secondary.umin) * inv_total_mass;
        row.secondary = row.primary;
        row.primary.rho = rho1;
        row.secondary.rho = rho2;
        row.primary.xi_1 = -q * inv_total_mass * relative_tau;
        row.primary.xi_2 = -q * inv_total_mass * relative_beta;
        row.secondary.xi_1 = -row.primary.xi_1 / q;
        row.secondary.xi_2 = -row.primary.xi_2 / q;
    } else {
        // xallarap coordinates use the supplied source-one state around the
        // CoM; source two is fixed by the dynamical mass ratio.
        row.secondary.xi_1 = -row.primary.xi_1 / source_mass_ratio;
        row.secondary.xi_2 = -row.primary.xi_2 / source_mass_ratio;
    }
    return row;
}

InferenceParameterRow inference_parameter_row(
    const py::handle& item, const lcbinint::lc::Model& model)
{
    const bool binary_source = model.source == lcbinint::lc::SourceKind::binary;
    if (!binary_source) {
        InferenceParameterRow row;
        row.primary = inference_params_from_object(item);
        row.secondary = row.primary;
        return row;
    }
    const auto object = py::reinterpret_borrow<py::object>(item);
    if (!py::isinstance<py::dict>(object)) {
        throw std::invalid_argument(
            "binary-source batch rows must be parameter dictionaries");
    }
    return binary_source_parameter_row(object.cast<py::dict>(), model);
}

py::array_t<double> compute_batch(
    const lcbinint::lc::LightCurve& lc,
    const TimesArray& times,
    const py::sequence& parameter_rows)
{
    const std::vector<double> tv = times_from_array(times);
    const bool binary_source =
        lc.source_kind() == lcbinint::lc::SourceKind::binary;
    std::vector<lcbi_params> parameters;
    std::vector<lcbi_params> secondary_parameters;
    std::vector<double> source_ratios;
    const auto count = static_cast<std::size_t>(py::len(parameter_rows));
    parameters.reserve(count);
    if (binary_source) {
        secondary_parameters.reserve(count);
        source_ratios.reserve(count);
    }
    for (const auto& item : parameter_rows) {
        const auto row = inference_parameter_row(item, lc.model());
        parameters.push_back(row.primary);
        if (binary_source) {
            secondary_parameters.push_back(row.secondary);
            source_ratios.push_back(row.source_ratio);
        }
    }
    std::vector<double> values;
    {
        py::gil_scoped_release release;
        values = lc.magnification_batch(
            tv, parameters, secondary_parameters, source_ratios);
    }
    return matrix_to_numpy(
        std::move(values), static_cast<py::ssize_t>(parameters.size()),
        static_cast<py::ssize_t>(tv.size()));
}

py::dict compute_light_curve_log_likelihood_batch(
    const lcbinint::lc::LightCurve& lc,
    py::array_t<double, py::array::c_style | py::array::forcecast> times,
    py::array_t<double, py::array::c_style | py::array::forcecast> flux,
    py::array_t<double, py::array::c_style | py::array::forcecast> error,
    const py::sequence& parameter_rows,
    const std::string& distribution_name,
    const std::string& flux_mode_name,
    double nu,
    py::object sampled_source_object,
    py::object sampled_blend_object)
{
    using Distribution = lcbinint::lc::LikelihoodDistribution;
    using FluxMode = lcbinint::lc::FluxMode;
    const bool binary_source =
        lc.source_kind() == lcbinint::lc::SourceKind::binary;
    const Distribution distribution = distribution_name == "gaussian"
        ? Distribution::gaussian
        : distribution_name == "student_t" ? Distribution::student_t
        : throw std::invalid_argument(
            "distribution must be 'gaussian' or 'student_t'");
    const FluxMode flux_mode = flux_mode_name == "fit" ? FluxMode::fit
        : flux_mode_name == "sample" ? FluxMode::sample
        : flux_mode_name == "marginalize" ? FluxMode::marginalize
        : throw std::invalid_argument(
            "flux_mode must be 'fit', 'sample', or 'marginalize'");

    const auto times_buffer = times.request();
    const auto flux_buffer = flux.request();
    const auto error_buffer = error.request();
    if (times_buffer.ndim != 1 || flux_buffer.ndim != 1 ||
        error_buffer.ndim != 1) {
        throw std::invalid_argument("times, flux, and error must be one-dimensional");
    }
    const auto* times_ptr = static_cast<const double*>(times_buffer.ptr);
    const auto* flux_ptr = static_cast<const double*>(flux_buffer.ptr);
    const auto* error_ptr = static_cast<const double*>(error_buffer.ptr);
    std::vector<double> times_vector(times_ptr, times_ptr + times_buffer.size);
    std::vector<double> flux_vector(flux_ptr, flux_ptr + flux_buffer.size);
    std::vector<double> error_vector(error_ptr, error_ptr + error_buffer.size);
    std::vector<lcbi_params> parameters;
    std::vector<lcbi_params> secondary_parameters;
    std::vector<double> source_ratios;
    parameters.reserve(static_cast<std::size_t>(py::len(parameter_rows)));
    if (binary_source) {
        secondary_parameters.reserve(
            static_cast<std::size_t>(py::len(parameter_rows)));
        source_ratios.reserve(
            static_cast<std::size_t>(py::len(parameter_rows)));
    }
    for (const auto& item : parameter_rows) {
        const auto row = inference_parameter_row(item, lc.model());
        parameters.push_back(row.primary);
        if (binary_source) {
            secondary_parameters.push_back(row.secondary);
            source_ratios.push_back(row.source_ratio);
        }
    }

    auto optional_vector = [](py::object value) {
        std::vector<double> output;
        if (value.is_none()) return output;
        py::array_t<double, py::array::c_style | py::array::forcecast> array(value);
        const auto buffer = array.request();
        if (buffer.ndim != 1) {
            throw std::invalid_argument("sampled flux arrays must be one-dimensional");
        }
        const auto* ptr = static_cast<const double*>(buffer.ptr);
        output.assign(ptr, ptr + buffer.size);
        return output;
    };
    std::vector<double> sampled_source = optional_vector(sampled_source_object);
    std::vector<double> sampled_blend = optional_vector(sampled_blend_object);
    lcbinint::lc::LikelihoodBatchResult result;
    {
        py::gil_scoped_release release;
        result = lc.light_curve_log_likelihood_batch(
            times_vector, flux_vector, error_vector, parameters,
            distribution, flux_mode, nu, sampled_source, sampled_blend,
            secondary_parameters, source_ratios);
    }
    py::dict output;
    output["log_likelihood"] = vec_to_numpy(std::move(result.log_likelihood));
    output["source_flux"] = vec_to_numpy(std::move(result.source_flux));
    output["blend_flux"] = vec_to_numpy(std::move(result.blend_flux));
    output["conditional_scale"] =
        vec_to_numpy(std::move(result.conditional_scale));
    output["conditional_df"] = vec_to_numpy(std::move(result.conditional_df));
    return output;
}

std::string finite_source_method_name_from_int(int method)
{
    return lcbinint::magnification::finite_source_method_name(
        static_cast<lcbinint::magnification::FiniteSourceMethod>(method));
}

PyLightCurveInfo compute_info(
    const lcbinint::lc::LightCurve& lc,
    const TimesArray&               times,
    const lcbi_params&              params)
{
    std::vector<double> tv = times_from_array(times);
    const int n = static_cast<int>(tv.size());
    std::vector<lcbinint::MagnificationResult> results;
    {
        py::gil_scoped_release release;
        // Diagnostics deliberately retain the finite last iterate when the
        // solver exhausts its refinement capacity.  Ordinary magnification,
        // likelihood, and C API paths fail closed on the same status.
        results = lc.evaluate_diagnostic(tv, params);
    }

    PyLightCurveInfo info;
    info.times = std::move(tv);
    info.magnifications.reserve(static_cast<std::size_t>(n));
    info.point_source_magnifications.reserve(static_cast<std::size_t>(n));
    info.finite_source_magnifications.reserve(static_cast<std::size_t>(n));
    info.source_x.reserve(static_cast<std::size_t>(n));
    info.source_y.reserve(static_cast<std::size_t>(n));
    info.image_counts.reserve(static_cast<std::size_t>(n));
    info.finite_source_methods.reserve(static_cast<std::size_t>(n));
    info.finite_source_method_names.reserve(static_cast<std::size_t>(n));
    info.finite_source_error_estimates.reserve(static_cast<std::size_t>(n));
    info.finite_source_refinement_levels.reserve(static_cast<std::size_t>(n));
    info.finite_source_converged.reserve(static_cast<std::size_t>(n));
    info.statuses.reserve(static_cast<std::size_t>(n));
    info.root_candidate_counts.reserve(static_cast<std::size_t>(n));
    info.root_duplicate_counts.reserve(static_cast<std::size_t>(n));
    info.root_polish_failure_counts.reserve(static_cast<std::size_t>(n));
    info.root_used_warm_start.reserve(static_cast<std::size_t>(n));
    info.root_used_cold_retry.reserve(static_cast<std::size_t>(n));
    info.root_used_high_precision.reserve(static_cast<std::size_t>(n));
    info.root_needs_high_precision.reserve(static_cast<std::size_t>(n));
    info.root_max_residuals.reserve(static_cast<std::size_t>(n));
    info.point_source_quadrupole_indicators.reserve(static_cast<std::size_t>(n));
    info.point_source_cusp_indicators.reserve(static_cast<std::size_t>(n));
    info.point_source_ghost_indicators.reserve(static_cast<std::size_t>(n));
    info.point_source_planetary_distances2.reserve(static_cast<std::size_t>(n));
    info.point_source_safety_tolerances.reserve(static_cast<std::size_t>(n));
    info.point_source_ghost_counts.reserve(static_cast<std::size_t>(n));
    info.point_source_safety_flags.reserve(static_cast<std::size_t>(n));
    info.separations.reserve(static_cast<std::size_t>(n));
    info.mass_ratios.reserve(static_cast<std::size_t>(n));
    info.caustic_distances.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto& result = results[static_cast<std::size_t>(i)];
        info.magnifications.push_back(
            result.status == lcbinint::EvaluationStatus::ok
                ? result.magnification
                : std::numeric_limits<double>::quiet_NaN());
        info.point_source_magnifications.push_back(result.point_source_magnification);
        info.finite_source_magnifications.push_back(result.finite_source_magnification);
        info.source_x.push_back(result.source.x);
        info.source_y.push_back(result.source.y);
        info.image_counts.push_back(result.image_count);
        info.finite_source_methods.push_back(result.finite_source_method);
        info.finite_source_method_names.push_back(
            finite_source_method_name_from_int(result.finite_source_method));
        info.finite_source_error_estimates.push_back(result.finite_source_error_estimate);
        info.finite_source_refinement_levels.push_back(result.finite_source_refinement_level);
        info.root_candidate_counts.push_back(result.root_candidate_count);
        info.root_duplicate_counts.push_back(result.root_duplicate_count);
        info.root_polish_failure_counts.push_back(result.root_polish_failure_count);
        info.root_used_warm_start.push_back(result.root_used_warm_start);
        info.root_used_cold_retry.push_back(result.root_used_cold_retry);
        info.root_used_high_precision.push_back(result.root_used_high_precision);
        info.root_needs_high_precision.push_back(result.root_needs_high_precision);
        info.root_max_residuals.push_back(result.root_max_residual);
        info.point_source_quadrupole_indicators.push_back(
            result.point_source_quadrupole_indicator);
        info.point_source_cusp_indicators.push_back(result.point_source_cusp_indicator);
        info.point_source_ghost_indicators.push_back(result.point_source_ghost_indicator);
        info.point_source_planetary_distances2.push_back(
            result.point_source_planetary_distance2);
        info.point_source_safety_tolerances.push_back(
            result.point_source_safety_tolerance);
        info.point_source_ghost_counts.push_back(result.point_source_ghost_count);
        info.point_source_safety_flags.push_back(result.point_source_safety_flags);
        info.separations.push_back(result.separation);
        info.mass_ratios.push_back(result.mass_ratio);
        info.caustic_distances.push_back(result.caustic_distance);
        info.statuses.push_back(lcbinint::evaluation_status_name(result.status));
        const bool converged =
            result.status != lcbinint::EvaluationStatus::unsupported_tolerance &&
            result.finite_source_converged;
        info.finite_source_converged.push_back(converged);
        if (!converged) {
            info.all_converged = false;
            info.unconverged_indices.push_back(i);
        }
	    }
	    return info;
	}

lcbinint::magnification::FiniteSourceSettings finite_source_settings_from(
    const lcbi_params& params,
    const lcbi_options& options)
{
    lcbinint::magnification::FiniteSourceSettings settings;
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
    settings.automatic_source_bins = options.automatic_source_bins != 0;
    settings.max_source_bins = options.max_source_bins;
    settings.finite_source_tol = options.finite_source_tol;
    settings.finite_source_reltol = options.finite_source_reltol;
    return settings;
}

PyGeometryBranches branches_to_python(
    const std::vector<std::vector<lcbinint::SourcePosition>>& branches)
{
    PyGeometryBranches out;
    out.x.reserve(branches.size());
    out.y.reserve(branches.size());
    for (const auto& branch : branches) {
        std::vector<double> xs;
        std::vector<double> ys;
        xs.reserve(branch.size());
        ys.reserve(branch.size());
        for (const auto& point : branch) {
            xs.push_back(point.x);
            ys.push_back(point.y);
        }
        out.x.push_back(std::move(xs));
        out.y.push_back(std::move(ys));
    }
    return out;
}

std::vector<PyImagePoint> binary_images_to_python(
    double separation,
    double mass_ratio,
    double source_x,
    double source_y)
{
    const lcbinint::magnification::PointSourceMagnifier magnifier;
    const auto images = magnifier.binary_images(
        separation, mass_ratio, {source_x, source_y});
    std::vector<PyImagePoint> out;
    out.reserve(images.size());
    for (const auto& image : images) {
        const double j = image.jacobian_determinant;
        PyImagePoint point;
        point.x = image.position.x;
        point.y = image.position.y;
        point.jacobian_determinant = j;
        point.magnification = 1.0 / std::abs(j);
        point.parity = j >= 0.0 ? 1 : -1;
        out.push_back(point);
    }
    return out;
}

struct GeometryRequest {
    std::optional<double> time;
    int n_points = 0;
    lcbi_params params;
};

GeometryRequest parse_geometry_request(
    const lcbinint::lc::LightCurve& lc,
    const py::args& args,
    const py::kwargs& kw)
{
    if (args.size() > 2) {
        throw py::type_error("expected at most time and params positional arguments");
    }

    std::optional<double> time;
    py::dict params_dict;
    if (kw) {
        params_dict = py::dict(kw);
    }

    if (args.size() >= 1) {
        py::object first = py::reinterpret_borrow<py::object>(args[0]);
        if (py::isinstance<py::dict>(first)) {
            params_dict = py::dict(first);
        } else if (!first.is_none()) {
            time = first.cast<double>();
        }
    }
    if (args.size() == 2) {
        py::object second = py::reinterpret_borrow<py::object>(args[1]);
        if (!second.is_none()) {
            params_dict = py::dict(second);
        }
    }

    GeometryRequest request;
    request.time = time;
    if (params_dict.contains("n_points")) {
        request.n_points = params_dict["n_points"].cast<int>();
        params_dict.attr("pop")("n_points");
    }
    request.params = lc.apply_coords(params_from_dict(params_dict));
    return request;
}

double effective_binary_separation(
    const lcbinint::lc::LightCurve& lc,
    const GeometryRequest& request)
{
    if (lc.orbital_motion() == LCBI_ORBIT_STATIC) {
        return request.params.sep;
    }
    if (!request.time.has_value()) {
        throw std::runtime_error("time is required when orbital motion is enabled");
    }
    const lcbinint::model::LensParameters lp =
        lcbinint::model::from_c_params(request.params);
    return lcbinint::model::orbital_state(lp, *request.time).separation;
}

double effective_binary_mass_ratio(
    const lcbinint::lc::LightCurve& lc,
    const lcbi_params& params)
{
    if (lc.options().vbm_compatible != 0 && params.q != 0.0) {
        return 1.0 / params.q;
    }
    return params.q;
}

lcbinint::model::TripleLensGeometry effective_triple_geometry(
    const lcbinint::lc::LightCurve& lc,
    const lcbi_params& params)
{
    if (params.orbital_motion_mode != LCBI_ORBIT_STATIC) {
        throw std::runtime_error("triple-lens caustics with orbital motion are not supported");
    }
    if (lc.options().vbm_compatible != 0) {
        return lcbinint::model::make_triple_lens_geometry_vbm(
            params.sep, params.q, params.sep2, params.ang, params.q2);
    }
    return lcbinint::model::make_triple_lens_geometry(
        params.sep, params.q, params.q2, params.sep2, params.ang);
}

PyGeometryBranches geometry_branches(
    const lcbinint::lc::LightCurve& lc,
    const GeometryRequest& request,
    bool critical_curves)
{
    auto settings = finite_source_settings_from(request.params, lc.options());
    if (request.n_points > 0) {
        settings.caustic_bins = request.n_points;
    }
    const lcbinint::magnification::FiniteSourceMagnifier magnifier(settings);
    if (request.params.q2 > 0.0) {
        const auto geometry = effective_triple_geometry(lc, request.params);
        return branches_to_python(critical_curves
            ? magnifier.triple_critical_curve_branches(geometry)
            : magnifier.triple_caustic_branches(geometry));
    }

    const double separation = effective_binary_separation(lc, request);
    const double mass_ratio = effective_binary_mass_ratio(lc, request.params);
    return branches_to_python(critical_curves
        ? magnifier.binary_critical_curve_branches(separation, mass_ratio)
        : magnifier.binary_caustic_branches(separation, mass_ratio));
}

lcbinint::SourcePosition lens_frame_source_position(
    const lcbinint::lc::LightCurve& lc,
    const lcbi_params& params,
    double time)
{
    const lcbi_params resolved_params = lc.apply_coords(params);
    const lcbinint::model::LensParameters lp = lcbinint::model::from_c_params(resolved_params);
    const lcbinint::model::Trajectory traj(lp);
    const bool vbm = lc.options().vbm_compatible != 0;
    const bool parallax_enabled = lc.model().parallax &&
        (resolved_params.piEN != 0.0 || resolved_params.piEE != 0.0);
    lcbinint::SourcePosition source =
        traj.source_position(
            time, vbm, lc.model().xallarap, parallax_enabled,
            lc.site() ? lc.site().get() : nullptr);

    if (params.orbital_motion_mode != LCBI_ORBIT_STATIC) {
        const auto orbit = lcbinint::model::orbital_state(lp, time);
        if (vbm) {
            const double costheta = std::cos(params.theta);
            const double sintheta = std::sin(params.theta);
            double tau = 0.0;
            double beta = 0.0;
            if (!parallax_enabled) {
                tau = (time - params.t0) / params.tE;
                beta = params.umin;
            } else {
                tau = source.x * costheta + source.y * sintheta;
                beta = -source.x * sintheta + source.y * costheta;
            }
            source = {
                tau * std::cos(orbit.angle) - beta * std::sin(orbit.angle),
                beta * std::cos(orbit.angle) + tau * std::sin(orbit.angle),
            };
        } else {
            source = lcbinint::model::rotate_source_to_orbital_frame(
                source, orbit.angle - params.theta);
        }
    }
    return source;
}

	} // namespace

void register_lc_submodule(py::module_& parent)
{
	// Keep the complete LightCurve API, but expose it directly as lcbinint.*.
	py::module_ lc = parent;

    py::class_<lcbinint::magnification::PointSourceSafetyDiagnostic>(
        lc, "PointSourceSafetyDiagnostic")
        .def_readonly("magnification",
            &lcbinint::magnification::PointSourceSafetyDiagnostic::magnification)
        .def_readonly("image_count",
            &lcbinint::magnification::PointSourceSafetyDiagnostic::image_count)
        .def_readonly("quadrupole_indicator",
            &lcbinint::magnification::PointSourceSafetyDiagnostic::quadrupole_indicator)
        .def_readonly("cusp_indicator",
            &lcbinint::magnification::PointSourceSafetyDiagnostic::cusp_indicator)
        .def_readonly("ghost_indicator",
            &lcbinint::magnification::PointSourceSafetyDiagnostic::ghost_indicator)
        .def_readonly("ghost_count",
            &lcbinint::magnification::PointSourceSafetyDiagnostic::ghost_count);

    // --- Enums ---
    py::enum_<lcbi_orbital_motion_mode>(lc, "OrbitalMotionMode")
        .value("STATIC",   LCBI_ORBIT_STATIC)
        .value("CIRCULAR", LCBI_ORBIT_CIRCULAR)
        .value("KEPLER",   LCBI_ORBIT_KEPLER)
        .export_values();

    py::enum_<lcbi_xallarap_param_type>(lc, "XallarapParamType")
        .value("NONE",              LCBI_XALLARAP_NONE)
        .value("ORBITAL_ELEMENTS",  LCBI_XALLARAP_ORBITAL_ELEMENTS)
        .value("CIRCULAR_ELEMENTS", LCBI_XALLARAP_CIRCULAR_ELEMENTS)
        .value("CIRCULAR_VEL",      LCBI_XALLARAP_CIRCULAR_VEL)
        .value("KEPLER_VEL",        LCBI_XALLARAP_KEPLER_VEL)
        .export_values();

	// --- Options: lcbi_options exposed directly ---
	    py::class_<lcbi_options>(lc, "Options")
	        .def(py::init([](
	                std::string param_type,
	                std::string coordinates,
	                py::object source_bins,
	                py::object nbin,
	                int    caustic_bins,
	                double grid_ratio,
	                int    mode,
	                py::object inverse_ray_grid,
	                double adaptive_hex_threshold,
	                double hex_tol,
	                double point_source_threshold,
	                double hexadecapole_threshold,
	                int    polar_source_bins,
	                py::object polar_nbin,
	                double polar_grid_ratio,
	                lcbi_xallarap_param_type xallarap_param_type,
	                int    parallax_mode,
	                int    max_source_bins,
	                double finite_source_tol,
	                double finite_source_reltol,
	                double tol,
	                double reltol) {
	            auto o = lcbi_default_options();
	            apply_param_type(o, coordinates.empty() ? param_type : coordinates);
	            if (!nbin.is_none()) {
	                apply_nbin(o, nbin);
	            } else if (!source_bins.is_none()) {
	                apply_nbin(o, source_bins);
	            } else {
	                o.automatic_source_bins = 1;
	            }
	            o.caustic_bins           = caustic_bins;
	            o.grid_ratio             = grid_ratio;
	            o.mode                   = mode;
	            if (!inverse_ray_grid.is_none()) {
	                if (py::isinstance<py::str>(inverse_ray_grid)) {
	                    apply_inverse_ray_grid(o, inverse_ray_grid.cast<std::string>());
	                } else {
	                    o.mode = inverse_ray_grid.cast<int>();
	                }
	            }
	            o.adaptive_hex_threshold = hex_tol >= 0.0 ? hex_tol : adaptive_hex_threshold;
	            o.point_source_threshold = point_source_threshold;
	            o.hexadecapole_threshold = hexadecapole_threshold;
	            o.polar_source_bins      = polar_source_bins;
	            if (!polar_nbin.is_none()) {
	                o.polar_source_bins = polar_nbin.cast<int>();
	            }
	            o.polar_grid_ratio       = polar_grid_ratio;
	            o.xallarap_param_type    = xallarap_param_type;
	            o.parallax_mode          = parallax_mode;
	            o.max_source_bins        = max_source_bins;
	            o.finite_source_tol      = finite_source_tol;
	            o.finite_source_reltol   = finite_source_reltol;
	            if (tol > 0.0) {
	                o.finite_source_tol = tol;
	            }
	            if (reltol > 0.0) {
	                o.finite_source_reltol = reltol;
	            }
	            return o;
	        }),
	            py::arg("param_type")             = "vbm",
	            py::arg("coordinates")            = "",
	            py::arg("source_bins")            = py::none(),
	            py::arg("nbin")                   = py::none(),
	            py::arg("caustic_bins")           = lcbi_default_options().caustic_bins,
	            py::arg("grid_ratio")             = lcbi_default_options().grid_ratio,
	            py::arg("mode")                   = lcbi_default_options().mode,
	            py::arg("inverse_ray_grid")       = py::none(),
	            py::arg("adaptive_hex_threshold") = lcbi_default_options().adaptive_hex_threshold,
	            py::arg("hex_tol")                = -1.0,
	            py::arg("point_source_threshold") = lcbi_default_options().point_source_threshold,
	            py::arg("hexadecapole_threshold") = lcbi_default_options().hexadecapole_threshold,
	            py::arg("polar_source_bins")      = 0,
	            py::arg("polar_nbin")             = py::none(),
	            py::arg("polar_grid_ratio")       = 0.0,
	            py::arg("xallarap_param_type")    = LCBI_XALLARAP_NONE,
	            py::arg("parallax_mode")          = 0,
	            py::arg("max_source_bins")        = lcbi_default_options().max_source_bins,
	            py::arg("finite_source_tol")      = lcbi_default_options().finite_source_tol,
	            py::arg("finite_source_reltol")   = lcbi_default_options().finite_source_reltol,
	            py::arg("tol")                    = 0.0,
	            py::arg("reltol")                 = 0.0)
	        .def_property("source_bins",
	            [](const lcbi_options& o) { return o.source_bins; },
	            [](lcbi_options& o, int value) { apply_nbin(o, py::int_(value)); })
	        .def_property("nbin",
	            [](const lcbi_options& o) -> py::object {
	                if (o.automatic_source_bins != 0) return py::str("auto");
	                return py::int_(o.source_bins);
	            },
	            [](lcbi_options& o, py::object value) { apply_nbin(o, value); })
	        .def_readwrite("caustic_bins",           &lcbi_options::caustic_bins)
	        .def_readwrite("grid_ratio",             &lcbi_options::grid_ratio)
	        .def_readwrite("mode",                   &lcbi_options::mode)
	        .def_property("_mode",
	            [](const lcbi_options& o) { return o.mode; },
	            [](lcbi_options& o, int value) { o.mode = value; })
        .def_property("param_type",
            [](const lcbi_options& o) -> std::string {
                if (o.vbm_compatible != 0 && o.center_of_mass == 0) return "vbm";
                if (o.vbm_compatible != 0 && o.center_of_mass != 0) return "vbm_center_of_mass";
                if (o.vbm_compatible == 0 && o.center_of_mass != 0) return "center_of_mass";
                return "lcbinint";
	            },
	            [](lcbi_options& o, const std::string& pt) { apply_param_type(o, pt); })
	        .def_property("coordinates",
	            [](const lcbi_options& o) -> std::string {
	                if (o.vbm_compatible != 0 && o.center_of_mass == 0) return "vbm";
	                if (o.vbm_compatible != 0 && o.center_of_mass != 0) return "vbm_center_of_mass";
	                if (o.vbm_compatible == 0 && o.center_of_mass != 0) return "center_of_mass";
	                return "lcbinint";
	            },
	            [](lcbi_options& o, const std::string& pt) { apply_param_type(o, pt); })
	        .def_property("inverse_ray_grid",
	            [](const lcbi_options& o) { return inverse_ray_grid_from_mode(o.mode); },
	            [](lcbi_options& o, const std::string& grid) { apply_inverse_ray_grid(o, grid); })
	        .def_readwrite("adaptive_hex_threshold", &lcbi_options::adaptive_hex_threshold)
	        .def_property("hex_tol",
	            [](const lcbi_options& o) { return o.adaptive_hex_threshold; },
	            [](lcbi_options& o, double value) { o.adaptive_hex_threshold = value; })
	        .def_readwrite("point_source_threshold", &lcbi_options::point_source_threshold)
	        .def_readwrite("hexadecapole_threshold", &lcbi_options::hexadecapole_threshold)
	        .def_property("polar_source_bins",
	            [](const lcbi_options& o) -> py::object {
	                if (o.polar_source_bins <= 0) return py::none();
	                return py::int_(o.polar_source_bins);
	            },
	            [](lcbi_options& o, py::object value) {
	                o.polar_source_bins = value.is_none() ? 0 : value.cast<int>();
	            })
	        .def_property("polar_nbin",
	            [](const lcbi_options& o) -> py::object {
	                if (o.polar_source_bins <= 0) return py::none();
	                return py::int_(o.polar_source_bins);
	            },
	            [](lcbi_options& o, py::object value) {
	                o.polar_source_bins = value.is_none() ? 0 : value.cast<int>();
	            })
	        .def_property("polar_grid_ratio",
	            [](const lcbi_options& o) -> py::object {
	                if (o.polar_grid_ratio <= 0.0) return py::none();
	                return py::float_(o.polar_grid_ratio);
	            },
	            [](lcbi_options& o, py::object value) {
	                o.polar_grid_ratio = value.is_none() ? 0.0 : value.cast<double>();
	            })
	        .def_readwrite("parallax_mode",          &lcbi_options::parallax_mode)
	        .def_readwrite("xallarap_param_type",    &lcbi_options::xallarap_param_type)
	        .def_readwrite("max_source_bins",        &lcbi_options::max_source_bins)
	        .def_readwrite("finite_source_tol",      &lcbi_options::finite_source_tol)
	        .def_readwrite("finite_source_reltol",   &lcbi_options::finite_source_reltol)
	        .def_property("tol",
	            [](const lcbi_options& o) { return o.finite_source_tol; },
	            [](lcbi_options& o, double value) { o.finite_source_tol = value; })
	        .def_property("reltol",
	            [](const lcbi_options& o) { return o.finite_source_reltol; },
	            [](lcbi_options& o, double value) { o.finite_source_reltol = value; })
        .def("__repr__", [](const lcbi_options& o) {
            std::string pt;
            if (o.vbm_compatible != 0 && o.center_of_mass == 0) pt = "vbm";
            else if (o.vbm_compatible != 0)                       pt = "vbm_center_of_mass";
            else if (o.center_of_mass != 0)                       pt = "center_of_mass";
            else                                                   pt = "lcbinint";
            const std::string bins = o.automatic_source_bins != 0
                ? "'auto'" : std::to_string(o.source_bins);
            return "<lc.Options param_type='" + pt + "' nbin=" + bins + ">";
        });

    // --- LimbDarkening ---
    // Limb darkening profile: I(μ) = 1 - c*(1-μ) - d*(1-√μ)
    // c=0, d=0 → uniform disk (point source limit).
    py::class_<PyLimbDarkening>(lc, "LimbDarkening")
        .def(py::init<double, double>(), py::arg("c") = 0.0, py::arg("d") = 0.0)
        .def_readwrite("c", &PyLimbDarkening::c)
        .def_readwrite("d", &PyLimbDarkening::d)
        .def_static("none", []() { return PyLimbDarkening{}; },
            "Uniform source profile.")
        .def_static("linear",      [](double u) { return PyLimbDarkening{u, 0.0}; },
            py::arg("u"),
            "Linear profile: I(μ) = 1 - u*(1-μ).  c=u, d=0.")
        .def_static("square_root", [](double c, double d) { return PyLimbDarkening{c, d}; },
            py::arg("c"), py::arg("d"),
            "Square-root profile: I(μ) = 1 - c*(1-μ) - d*(1-√μ).")
        .def("__repr__", [](const PyLimbDarkening& ld) {
            return "<lc.LimbDarkening c=" + std::to_string(ld.c)
                + " d=" + std::to_string(ld.d) + ">";
        });

	    py::class_<PyLightCurveInfo>(lc, "LightCurveInfo")
	        .def_readonly("times", &PyLightCurveInfo::times)
	        .def_readonly("magnifications", &PyLightCurveInfo::magnifications)
        .def_readonly("point_source_magnifications", &PyLightCurveInfo::point_source_magnifications)
        .def_readonly("finite_source_magnifications", &PyLightCurveInfo::finite_source_magnifications)
        .def_readonly("source_x", &PyLightCurveInfo::source_x)
        .def_readonly("source_y", &PyLightCurveInfo::source_y)
        .def_readonly("image_counts", &PyLightCurveInfo::image_counts)
        .def_readonly("finite_source_methods", &PyLightCurveInfo::finite_source_methods)
        .def_readonly("finite_source_method_names", &PyLightCurveInfo::finite_source_method_names)
        .def_readonly("finite_source_error_estimates", &PyLightCurveInfo::finite_source_error_estimates)
        .def_readonly("finite_source_refinement_levels", &PyLightCurveInfo::finite_source_refinement_levels)
        .def_readonly("finite_source_converged", &PyLightCurveInfo::finite_source_converged)
        .def_readonly("statuses", &PyLightCurveInfo::statuses)
        .def_readonly("root_candidate_counts", &PyLightCurveInfo::root_candidate_counts)
        .def_readonly("root_duplicate_counts", &PyLightCurveInfo::root_duplicate_counts)
        .def_readonly("root_polish_failure_counts", &PyLightCurveInfo::root_polish_failure_counts)
        .def_readonly("root_used_warm_start", &PyLightCurveInfo::root_used_warm_start)
        .def_readonly("root_used_cold_retry", &PyLightCurveInfo::root_used_cold_retry)
        .def_readonly("root_used_high_precision", &PyLightCurveInfo::root_used_high_precision)
        .def_readonly("root_needs_high_precision", &PyLightCurveInfo::root_needs_high_precision)
        .def_readonly("root_max_residuals", &PyLightCurveInfo::root_max_residuals)
        .def_readonly("point_source_quadrupole_indicators",
            &PyLightCurveInfo::point_source_quadrupole_indicators)
        .def_readonly("point_source_cusp_indicators",
            &PyLightCurveInfo::point_source_cusp_indicators)
        .def_readonly("point_source_ghost_indicators",
            &PyLightCurveInfo::point_source_ghost_indicators)
        .def_readonly("point_source_planetary_distances2",
            &PyLightCurveInfo::point_source_planetary_distances2)
        .def_readonly("point_source_safety_tolerances",
            &PyLightCurveInfo::point_source_safety_tolerances)
        .def_readonly("point_source_ghost_counts",
            &PyLightCurveInfo::point_source_ghost_counts)
        .def_readonly("point_source_safety_flags",
            &PyLightCurveInfo::point_source_safety_flags)
        .def_readonly("separations", &PyLightCurveInfo::separations)
        .def_readonly("mass_ratios", &PyLightCurveInfo::mass_ratios)
        .def_readonly("caustic_distances", &PyLightCurveInfo::caustic_distances)
	        .def_readonly("all_converged", &PyLightCurveInfo::all_converged)
	        .def_readonly("unconverged_indices", &PyLightCurveInfo::unconverged_indices);

	    py::class_<PySourceTrajectory>(lc, "SourceTrajectory")
	        .def_readonly("times", &PySourceTrajectory::times)
	        .def_readonly("x", &PySourceTrajectory::x)
	        .def_readonly("y", &PySourceTrajectory::y)
	        .def("__getitem__", [](const PySourceTrajectory& t, const std::string& key) -> py::object {
	            if (key == "times") return py::cast(t.times);
	            if (key == "x") return py::cast(t.x);
	            if (key == "y") return py::cast(t.y);
	            throw py::key_error("SourceTrajectory: unknown key '" + key + "'");
	        });

    py::class_<PyBinarySourceComponent>(lc, "BinarySourceComponent")
        .def_readonly("magnification", &PyBinarySourceComponent::magnification)
        .def_readonly("trajectory", &PyBinarySourceComponent::trajectory);

    py::class_<PyBinarySourceComponents>(lc, "BinarySourceComponents")
        .def_readonly("source1", &PyBinarySourceComponents::source1)
        .def_readonly("source2", &PyBinarySourceComponents::source2)
        .def_readonly("total", &PyBinarySourceComponents::total)
        .def_readonly("flux_ratio", &PyBinarySourceComponents::flux_ratio);

    py::class_<PyFiniteSourceGeometry>(lc, "FiniteSourceGeometry")
        .def_readonly("separation", &PyFiniteSourceGeometry::separation)
        .def_readonly("mass_ratio", &PyFiniteSourceGeometry::mass_ratio)
        .def_readonly("source_x", &PyFiniteSourceGeometry::source_x)
        .def_readonly("source_y", &PyFiniteSourceGeometry::source_y)
        .def_readonly("source_radius", &PyFiniteSourceGeometry::source_radius)
        .def_readonly("limb_darkening_c", &PyFiniteSourceGeometry::limb_darkening_c)
        .def_readonly("limb_darkening_d", &PyFiniteSourceGeometry::limb_darkening_d)
        .def_readonly("absolute_tolerance", &PyFiniteSourceGeometry::absolute_tolerance)
        .def_readonly("relative_tolerance", &PyFiniteSourceGeometry::relative_tolerance);

	    py::class_<PyGeometryBranches>(lc, "GeometryBranches")
	        .def_readonly("x", &PyGeometryBranches::x)
	        .def_readonly("y", &PyGeometryBranches::y)
	        .def("__getitem__", [](const PyGeometryBranches& b, const std::string& key) -> py::object {
	            if (key == "x") return py::cast(b.x);
	            if (key == "y") return py::cast(b.y);
	            throw py::key_error("GeometryBranches: unknown key '" + key + "'");
	        });

	    py::class_<PyImagePoint>(lc, "ImagePoint")
	        .def_readonly("x", &PyImagePoint::x)
	        .def_readonly("y", &PyImagePoint::y)
	        .def_readonly("jacobian_determinant", &PyImagePoint::jacobian_determinant)
	        .def_readonly("magnification", &PyImagePoint::magnification)
	        .def_readonly("parity", &PyImagePoint::parity)
	        .def("__getitem__", [](const PyImagePoint& p, const std::string& key) -> py::object {
	            if (key == "x") return py::float_(p.x);
	            if (key == "y") return py::float_(p.y);
	            if (key == "jacobian_determinant") return py::float_(p.jacobian_determinant);
	            if (key == "magnification") return py::float_(p.magnification);
	            if (key == "parity") return py::int_(p.parity);
	            throw py::key_error("ImagePoint: unknown key '" + key + "'");
	        })
	        .def("__repr__", [](const PyImagePoint& p) {
	            return "<lc.ImagePoint x=" + std::to_string(p.x)
	                + " y=" + std::to_string(p.y)
	                + " magnification=" + std::to_string(p.magnification)
	                + " parity=" + std::to_string(p.parity) + ">";
	        });

	    // --- Parameters: lcbi_params exposed directly ---
	    py::class_<lcbi_params>(lc, "Parameters")
	        .def(py::init([]() { return lcbi_default_params(); }))
	        .def(py::init([](py::kwargs kw) {
	            return params_from_dict(py::dict(kw));
	        }))
	        .def(py::init([](
	                double t0, double tE, double u0, double alpha,
                double s, double q, double rho,
                double piEN, double piEE,
                double q2, double sep2, double ang,
                double g1, double g2, double g3,
                double lom_szs, double lom_ar) {
            auto p = lcbi_default_params();
            p.t0 = t0; p.tE = tE; p.umin = u0; p.theta = alpha;
            p.sep = s; p.q = q; p.rho = rho;
            p.piEN = piEN; p.piEE = piEE;
            p.q2 = q2; p.sep2 = sep2; p.ang = ang;
            p.g1 = g1; p.g2 = g2; p.g3 = g3;
            p.lom_szs = lom_szs; p.lom_ar = lom_ar;
            return p;
        }),
            py::arg("t0")    = 0.0,  py::arg("tE")    = 1.0,
            py::arg("u0")    = 0.0,  py::arg("alpha")  = 0.0,
            py::arg("s")     = 1.0,  py::arg("q")      = 1.0,
            py::arg("rho")   = 0.0,
            py::arg("piEN")  = 0.0,  py::arg("piEE")   = 0.0,
            py::arg("q2")    = 0.0,  py::arg("sep2")   = 0.0, py::arg("ang")    = 0.0,
            py::arg("g1")    = 0.0,  py::arg("g2")     = 0.0, py::arg("g3")     = 0.0,
            py::arg("lom_szs") = 0.0, py::arg("lom_ar") = 1.0)
        .def_property("t0",    [](const lcbi_params& p){ return p.t0; },    [](lcbi_params& p, double v){ p.t0 = v; })
        .def_property("tE",    [](const lcbi_params& p){ return p.tE; },    [](lcbi_params& p, double v){ p.tE = v; })
	        .def_property("u0",    [](const lcbi_params& p){ return p.umin; },  [](lcbi_params& p, double v){ p.umin = v; })
	        .def_property("alpha", [](const lcbi_params& p){ return p.theta; }, [](lcbi_params& p, double v){ p.theta = v; })
	        .def_property("s",     [](const lcbi_params& p){ return p.sep; },   [](lcbi_params& p, double v){ p.sep = v; })
	        .def_property("umin",  [](const lcbi_params& p){ return p.umin; },  [](lcbi_params& p, double v){ p.umin = v; })
	        .def_property("theta", [](const lcbi_params& p){ return p.theta; }, [](lcbi_params& p, double v){ p.theta = v; })
	        .def_property("sep",   [](const lcbi_params& p){ return p.sep; },   [](lcbi_params& p, double v){ p.sep = v; })
        .def_readwrite("q",    &lcbi_params::q)
        .def_readwrite("rho",  &lcbi_params::rho)
        .def_readwrite("q2",   &lcbi_params::q2)
        .def_readwrite("sep2", &lcbi_params::sep2)
        .def_readwrite("ang",  &lcbi_params::ang)
        .def_readwrite("piEN",    &lcbi_params::piEN)
        .def_readwrite("piEE",    &lcbi_params::piEE)
        .def_readwrite("ra",      &lcbi_params::ra)
        .def_readwrite("dec",     &lcbi_params::dec)
        .def_readwrite("tfix",    &lcbi_params::tfix)
        .def_readwrite("obs_lat", &lcbi_params::obs_lat)
        .def_readwrite("obs_lon", &lcbi_params::obs_lon)
        .def_readwrite("orbital_motion_mode", &lcbi_params::orbital_motion_mode)
        .def_readwrite("g1",      &lcbi_params::g1)
        .def_readwrite("g2",      &lcbi_params::g2)
        .def_readwrite("g3",      &lcbi_params::g3)
        .def_readwrite("lom_szs", &lcbi_params::lom_szs)
        .def_readwrite("lom_ar",  &lcbi_params::lom_ar)
        .def_readwrite("v_sep",   &lcbi_params::v_sep)
        .def_readwrite("xi_1",      &lcbi_params::xi_1)
        .def_readwrite("xi_2",      &lcbi_params::xi_2)
        .def_readwrite("omega_xa",  &lcbi_params::omega_xa)
        .def_readwrite("inc_xa",    &lcbi_params::inc_xa)
        .def_readwrite("phi_xa",    &lcbi_params::phi_xa)
        .def_readwrite("limb_darkening_c", &lcbi_params::limb_darkening_c)
        .def_readwrite("limb_darkening_d", &lcbi_params::limb_darkening_d)
        .def("__repr__", [](const lcbi_params& p) {
            return "<lc.Parameters t0=" + std::to_string(p.t0)
                + " tE=" + std::to_string(p.tE)
                + " u0=" + std::to_string(p.umin)
                + " s=" + std::to_string(p.sep)
                + " q=" + std::to_string(p.q) + ">";
        });

    // --- Model ---
    using LC       = lcbinint::lc::LightCurve;
    using Model     = lcbinint::lc::Model;
    using LKind    = lcbinint::lc::LensKind;
    using SKind    = lcbinint::lc::SourceKind;
    using SkyCoord = lcbinint::obs::SkyCoord;
    using Site     = lcbinint::obs::Site;

    // Parse lens string.
    auto parse_lens = [](const std::string& s) -> LKind {
        if (s == "binary" || s.empty()) return LKind::binary;
        if (s == "triple")              return LKind::triple;
        throw std::invalid_argument("lens must be 'binary' or 'triple'");
    };

    // Parse "binary"/"single" source string.
    auto parse_source = [](const std::string& s) -> SKind {
        if (s == "single" || s.empty()) return SKind::single;
        if (s == "binary" || s == "binary_source") return SKind::binary;
        throw std::invalid_argument("source must be 'single' or 'binary'");
    };

    // Parse orbital motion mode string.
    auto parse_orbital = [](const std::string& s) -> lcbi_orbital_motion_mode {
        if (s == "static"  || s.empty()) return LCBI_ORBIT_STATIC;
        if (s == "circular")             return LCBI_ORBIT_CIRCULAR;
        if (s == "kepler")               return LCBI_ORBIT_KEPLER;
        throw std::invalid_argument("orbital_motion must be 'static', 'circular', or 'kepler'");
    };

    // Parse xallarap mode string.
    auto parse_xallarap = [](const std::string& s) -> lcbi_xallarap_param_type {
        if (s.empty() || s == "none")                        return LCBI_XALLARAP_NONE;
        if (s == "orbital_elements" || s == "kepler")        return LCBI_XALLARAP_ORBITAL_ELEMENTS;
        if (s == "circular_elements" || s == "circular")     return LCBI_XALLARAP_CIRCULAR_ELEMENTS;
        if (s == "circular_velocity" || s == "circular_vel") return LCBI_XALLARAP_CIRCULAR_VEL;
        if (s == "kepler_velocity"   || s == "kepler_vel")   return LCBI_XALLARAP_KEPLER_VEL;
        throw std::invalid_argument(
            "xallarap must be 'none', 'orbital_elements', 'circular_elements', "
            "'circular_velocity', or 'kepler_velocity'");
    };

    auto parse_source_orbit_coordinates = [](const std::string& s)
        -> lcbinint::lc::SourceOrbitCoordinates {
        using Coordinates = lcbinint::lc::SourceOrbitCoordinates;
        if (s.empty() || s == "none") return Coordinates::none;
        if (s == "xallarap" || s == "xi") return Coordinates::xallarap;
        if (s == "trajectory_offset" || s == "offset")
            return Coordinates::trajectory_offset;
        throw std::invalid_argument(
            "source_orbit_coordinates must be 'xallarap' or 'trajectory_offset'");
    };

    py::class_<Model, std::shared_ptr<Model>>(lc, "Model",
        R"(Physical model configuration for a LightCurve.

Separates physics configuration from numerical
Options (source_bins, grid_ratio, etc.).

terrestrial=False by default. When it is enabled on a shared Model, only
LightCurves with a ground Site apply the terrestrial correction.)")
        .def(py::init([&](
                const std::string& source,
                const std::string& orbital_motion,
                const std::string& xallarap,
                const std::string& source_orbit_coordinates,
                bool               parallax,
                bool               terrestrial,
                py::object         sky,
                py::object         t_ref,
                const std::string& lens,
                bool               finite_source) {
            Model model;
            model.lens           = parse_lens(lens);
            model.source         = parse_source(source);
            model.finite_source  = finite_source;
            model.orbital_motion = parse_orbital(orbital_motion);
            model.xallarap       = parse_xallarap(xallarap);
            model.source_orbit_coordinates =
                parse_source_orbit_coordinates(source_orbit_coordinates);
            model.parallax       = parallax;
            model.terrestrial    = terrestrial;
            if (!sky.is_none())   model.sky  = sky.cast<std::shared_ptr<SkyCoord>>();
            if (!t_ref.is_none()) model.t_ref = t_ref.cast<double>();
            return model;
        }),
            py::arg("source")         = "single",
            py::arg("orbital_motion") = "static",
            py::arg("xallarap")       = "none",
            py::arg("source_orbit_coordinates") = "none",
            py::arg("parallax")       = false,
            py::arg("terrestrial")    = false,
            py::arg("sky")            = py::none(),
            py::arg("t_ref")          = py::none(),
            py::arg("lens")           = "binary",
            py::arg("finite_source")  = true)
        .def_property("lens",
            [](const Model& model) { return model.lens == LKind::triple ? "triple" : "binary"; },
            [&](Model& model, const std::string& s) { model.lens = parse_lens(s); })
        .def_property("source",
            [](const Model& model) { return model.source == SKind::binary ? "binary" : "single"; },
            [&](Model& model, const std::string& s) { model.source = parse_source(s); })
        .def_readwrite("finite_source", &Model::finite_source)
        .def_property("orbital_motion",
            [](const Model& model) -> std::string {
                if (model.orbital_motion == LCBI_ORBIT_CIRCULAR) return "circular";
                if (model.orbital_motion == LCBI_ORBIT_KEPLER)   return "kepler";
                return "static";
            },
            [&](Model& model, const std::string& s) { model.orbital_motion = parse_orbital(s); })
        .def_property("xallarap",
            [](const Model& model) -> std::string {
                switch (model.xallarap) {
                case LCBI_XALLARAP_ORBITAL_ELEMENTS:  return "orbital_elements";
                case LCBI_XALLARAP_CIRCULAR_ELEMENTS: return "circular_elements";
                case LCBI_XALLARAP_CIRCULAR_VEL:      return "circular_velocity";
                case LCBI_XALLARAP_KEPLER_VEL:        return "kepler_velocity";
                default:                               return "none";
                }
            },
            [&](Model& model, const std::string& s) { model.xallarap = parse_xallarap(s); })
        .def_property("source_orbit_coordinates",
            [](const Model& model) -> std::string {
                using Coordinates = lcbinint::lc::SourceOrbitCoordinates;
                switch (model.source_orbit_coordinates) {
                case Coordinates::xallarap: return "xallarap";
                case Coordinates::trajectory_offset: return "trajectory_offset";
                default: return "none";
                }
            },
            [&](Model& model, const std::string& s) {
                model.source_orbit_coordinates = parse_source_orbit_coordinates(s);
            })
        .def_readwrite("parallax",    &Model::parallax)
        .def_readwrite("terrestrial", &Model::terrestrial)
        .def_property("sky",
            [](const Model& model) -> py::object {
                if (!model.sky) return py::none();
                return py::cast(model.sky);
            },
            [](Model& model, py::object obj) {
                if (obj.is_none()) model.sky = nullptr;
                else model.sky = obj.cast<std::shared_ptr<SkyCoord>>();
            })
        .def_property("t_ref",
            [](const Model& model) -> py::object {
                if (!model.t_ref.has_value()) return py::none();
                return py::float_(*model.t_ref);
            },
            [](Model& model, py::object obj) {
                if (obj.is_none()) model.t_ref = std::nullopt;
                else model.t_ref = obj.cast<double>();
            })
        .def("__repr__", [](const Model& model) {
            std::string s = "<lc.Model";
            if (model.lens == LKind::triple) s += " lens=triple";
            if (model.parallax)    s += " parallax";
            if (model.terrestrial) s += " terrestrial";
            if (model.orbital_motion != LCBI_ORBIT_STATIC)
                s += model.orbital_motion == LCBI_ORBIT_CIRCULAR ? " orbital_motion=circular" : " orbital_motion=kepler";
            if (model.xallarap != LCBI_XALLARAP_NONE) {
                switch (model.xallarap) {
                case LCBI_XALLARAP_ORBITAL_ELEMENTS:  s += " xallarap=orbital_elements";  break;
                case LCBI_XALLARAP_CIRCULAR_ELEMENTS: s += " xallarap=circular_elements"; break;
                case LCBI_XALLARAP_CIRCULAR_VEL:      s += " xallarap=circular_velocity"; break;
                case LCBI_XALLARAP_KEPLER_VEL:        s += " xallarap=kepler_velocity";   break;
                default: break;
                }
            }
            if (model.source == SKind::binary) s += " source=binary";
            using Coordinates = lcbinint::lc::SourceOrbitCoordinates;
            if (model.source_orbit_coordinates == Coordinates::xallarap)
                s += " source_orbit_coordinates=xallarap";
            else if (model.source_orbit_coordinates == Coordinates::trajectory_offset)
                s += " source_orbit_coordinates=trajectory_offset";
            if (model.sky)  s += " sky=set";
            if (model.t_ref.has_value()) s += " t_ref=" + std::to_string(*model.t_ref);
            s += ">";
            return s;
        });

    // --- LightCurve ---
    // Optimized for magnification-only use: construct once, call many times.
    // Accepts params as: lcbi_params object, dict, or **kwargs.

    // Default options with vbm_compatible=1 (same as lc.Options() default).
    static const lcbi_options kDefaultOpts = []{ auto o = lcbi_default_options(); o.vbm_compatible = 1; return o; }();

    // Dispatch __call__ based on source kind.
    auto compute_dispatch = [](const LC& lc,
                                const TimesArray& times,
                                const lcbi_params& base_params,
                                py::dict extra) -> py::array_t<double> {
        if (lc.source_kind() == SKind::single) {
            return compute(lc, times, base_params);
        }
        const auto row = binary_source_parameter_row(extra, lc.model());
        const std::vector<double> tv = times_from_array(times);
        std::vector<double> mags;
        {
            py::gil_scoped_release release;
            mags = lc.magnification_binary(
                tv, row.primary, row.source_ratio, row.secondary);
        }
        return vec_to_numpy(std::move(mags));
    };

    auto binary_source_components = [](const LC& lc,
                                       const TimesArray& times,
                                       py::dict values) -> PyBinarySourceComponents {
        if (lc.source_kind() != SKind::binary) {
            throw std::invalid_argument(
                "binary_source_components requires LightCurve(source='binary')");
        }
        const auto row = binary_source_parameter_row(values, lc.model());
        const std::vector<double> tv = times_from_array(times);
        PyBinarySourceComponents result;
        result.flux_ratio = row.source_ratio;
        {
            py::gil_scoped_release release;
            result.source1.magnification = lc.magnification(tv, row.primary);
            result.source2.magnification = lc.magnification(tv, row.secondary);
        }
        result.total.resize(tv.size());
        const double denominator = 1.0 + row.source_ratio;
        for (std::size_t i = 0; i < tv.size(); ++i) {
            result.total[i] = (result.source1.magnification[i]
                + row.source_ratio * result.source2.magnification[i]) / denominator;
        }

        const auto primary = lc.apply_coords(row.primary);
        const auto secondary = lc.apply_coords(row.secondary);
        result.source1.trajectory.times = tv;
        result.source2.trajectory.times = tv;
        result.source1.trajectory.x.resize(tv.size());
        result.source1.trajectory.y.resize(tv.size());
        result.source2.trajectory.x.resize(tv.size());
        result.source2.trajectory.y.resize(tv.size());
        {
            py::gil_scoped_release release;
            for (std::size_t i = 0; i < tv.size(); ++i) {
                const auto source1 = lens_frame_source_position(lc, primary, tv[i]);
                const auto source2 = lens_frame_source_position(lc, secondary, tv[i]);
                result.source1.trajectory.x[i] = source1.x;
                result.source1.trajectory.y[i] = source1.y;
                result.source2.trajectory.x[i] = source2.x;
                result.source2.trajectory.y[i] = source2.y;
            }
        }
        return result;
    };

    py::class_<LC, std::shared_ptr<LC>>(lc, "LightCurve")
        // Constructor 1: explicit lc.Options + lc.Model objects
        .def(py::init([&](const lcbi_options& opts,
                           py::object model,
                           const PyLimbDarkening& ld,
                           py::object site) {
            auto model_ptr = model.is_none()
                ? std::make_shared<Model>()
                : model.cast<std::shared_ptr<Model>>();
            std::shared_ptr<Site> site_ptr;
            if (!site.is_none()) site_ptr = site.cast<std::shared_ptr<Site>>();
            return std::make_shared<LC>(
                opts, ld.c, ld.d, std::move(model_ptr), std::move(site_ptr));
        }),
            py::arg("options")        = kDefaultOpts,
            py::arg("model")          = py::none(),
            py::arg("limb_darkening") = PyLimbDarkening{}, py::arg("site") = py::none())
        // Constructor 2: kwargs directly (convenience, backward-compatible)
        .def(py::init([&](py::kwargs kw) {
            auto o = kDefaultOpts;
            PyLimbDarkening ld{};
            auto model = std::make_shared<Model>();
            std::shared_ptr<Site> site;
            for (auto& item : kw) {
                const std::string key = item.first.cast<std::string>();
                // --- Options (numerics) ---
	                if      (key == "source_bins")            apply_nbin(o, py::reinterpret_borrow<py::object>(item.second));
	                else if (key == "nbin")                   apply_nbin(o, py::reinterpret_borrow<py::object>(item.second));
	                else if (key == "caustic_bins")           o.caustic_bins           = item.second.cast<int>();
	                else if (key == "param_type")             apply_param_type(o, item.second.cast<std::string>());
	                else if (key == "coordinates")            apply_param_type(o, item.second.cast<std::string>());
	                else if (key == "adaptive_hex_threshold") o.adaptive_hex_threshold = item.second.cast<double>();
	                else if (key == "hex_tol")                o.adaptive_hex_threshold = item.second.cast<double>();
	                else if (key == "parallax_mode")          o.parallax_mode          = item.second.cast<int>();
	                else if (key == "mode")                   o.mode                   = item.second.cast<int>();
	                else if (key == "_mode")                  o.mode                   = item.second.cast<int>();
	                else if (key == "inverse_ray_grid") {
	                    if (py::isinstance<py::str>(item.second)) {
	                        apply_inverse_ray_grid(o, item.second.cast<std::string>());
	                    } else {
	                        o.mode = item.second.cast<int>();
	                    }
	                }
	                else if (key == "grid_ratio")             o.grid_ratio             = item.second.cast<double>();
	                else if (key == "point_source_threshold") o.point_source_threshold = item.second.cast<double>();
	                else if (key == "hexadecapole_threshold") o.hexadecapole_threshold = item.second.cast<double>();
	                else if (key == "polar_source_bins" || key == "polar_nbin") {
	                    auto obj = py::reinterpret_borrow<py::object>(item.second);
	                    o.polar_source_bins = obj.is_none() ? 0 : obj.cast<int>();
	                }
	                else if (key == "polar_grid_ratio") {
	                    auto obj = py::reinterpret_borrow<py::object>(item.second);
	                    o.polar_grid_ratio = obj.is_none() ? 0.0 : obj.cast<double>();
	                }
	                else if (key == "max_source_bins")        o.max_source_bins        = item.second.cast<int>();
	                else if (key == "finite_source_tol")      o.finite_source_tol      = item.second.cast<double>();
	                else if (key == "finite_source_reltol")   o.finite_source_reltol   = item.second.cast<double>();
	                else if (key == "tol")                    o.finite_source_tol      = item.second.cast<double>();
	                else if (key == "reltol")                 o.finite_source_reltol   = item.second.cast<double>();
	                else if (key == "options")                o = item.second.cast<lcbi_options>();
                // --- LimbDarkening ---
                else if (key == "ld_c" || key == "limb_darkening_c") ld.c = item.second.cast<double>();
                else if (key == "ld_d" || key == "limb_darkening_d") ld.d = item.second.cast<double>();
                else if (key == "limb_darkening") ld = item.second.cast<PyLimbDarkening>();
                // --- Model (physics) ---
                else if (key == "source")         model->source        = parse_source(item.second.cast<std::string>());
                else if (key == "orbital_motion") model->orbital_motion = parse_orbital(item.second.cast<std::string>());
                else if (key == "xallarap")       model->xallarap      = parse_xallarap(item.second.cast<std::string>());
                else if (key == "source_orbit_coordinates") {
                    model->source_orbit_coordinates =
                        parse_source_orbit_coordinates(item.second.cast<std::string>());
                }
                else if (key == "parallax")       model->parallax      = item.second.cast<bool>();
                else if (key == "terrestrial")    model->terrestrial   = item.second.cast<bool>();
                else if (key == "lens") {
                    model->lens = parse_lens(item.second.cast<std::string>());
                }
                else if (key == "sky") {
                    auto obj = py::reinterpret_borrow<py::object>(item.second);
                    if (!obj.is_none()) model->sky = obj.cast<std::shared_ptr<SkyCoord>>();
                }
                else if (key == "site") {
                    auto obj = py::reinterpret_borrow<py::object>(item.second);
                    if (!obj.is_none()) site = obj.cast<std::shared_ptr<Site>>();
                }
                else if (key == "t_ref") {
                    auto obj = py::reinterpret_borrow<py::object>(item.second);
                    if (!obj.is_none()) model->t_ref = obj.cast<double>();
                }
                else throw py::key_error("LightCurve: unknown option '" + key + "'");
            }
            return std::make_shared<LC>(
                o, ld.c, ld.d, model, site);
        }))
        .def_property_readonly("options", &LC::options)
        .def_property_readonly("model", &LC::model_ptr)
        .def_property_readonly("ld_c",    &LC::ld_c)
        .def_property_readonly("ld_d",    &LC::ld_d)
        // Convenience shortcuts (delegate to Model)
        .def_property_readonly("lens", [](const LC& lc) -> std::string {
            return lc.lens_kind() == LKind::triple ? "triple" : "binary";
        })
        .def_property_readonly("source", [](const LC& lc) -> std::string {
            return lc.source_kind() == SKind::binary ? "binary" : "single";
        })
        .def_property_readonly("orbital_motion", [](const LC& lc) -> std::string {
            if (lc.orbital_motion() == LCBI_ORBIT_CIRCULAR) return "circular";
            if (lc.orbital_motion() == LCBI_ORBIT_KEPLER)   return "kepler";
            return "static";
        })
        .def_property_readonly("sky", [](const LC& lc) -> py::object {
            if (!lc.sky_coord()) return py::none();
            return py::cast(lc.sky_coord());
        })
        .def_property_readonly("site", [](const LC& lc) -> py::object {
            if (!lc.site()) return py::none();
            return py::cast(lc.site());
        })
        .def_property_readonly("t_ref", [](const LC& lc) -> py::object {
            if (!lc.t_ref()) return py::none();
            return py::float_(*lc.t_ref());
        })
        .def_property_readonly("terrestrial", [](const LC& lc) {
            return lc.model().terrestrial;
        })

        // __call__ overload 1: lcbi_params object
        .def("__call__",
            [&](const LC& lc, const TimesArray& times, const lcbi_params& params) {
                if (lc.source_kind() == SKind::single) return compute(lc, times, params);
                throw std::runtime_error(
                    "source='binary': pass a dictionary with rho1, t0_2, u0_2, rho2, and flux_ratio");
                return py::array_t<double>{};  // unreachable
            },
            py::arg("times"), py::arg("params"))

        // __call__ overload 2: dict
        .def("__call__",
            [&](const LC& lc, const TimesArray& times, py::dict d) {
                return compute_dispatch(lc, times, params_from_dict(d), d);
            },
            py::arg("times"), py::arg("params"))

        // __call__ overload 3: **kwargs
        .def("__call__",
            [&](const LC& lc, const TimesArray& times, py::kwargs kw) {
                py::dict d(kw);
                return compute_dispatch(lc, times, params_from_dict(d), d);
            },
            py::arg("times"))

        // .magnification() alias
        .def("magnification",
            [&](const LC& lc, const TimesArray& times, const lcbi_params& params) {
                if (lc.source_kind() == SKind::single) return compute(lc, times, params);
                throw std::runtime_error(
                    "source='binary': pass a dictionary with rho1, t0_2, u0_2, rho2, and flux_ratio");
                return py::array_t<double>{};
            },
            py::arg("times"), py::arg("params"))
        .def("magnification",
            [&](const LC& lc, const TimesArray& times, py::dict d) {
                return compute_dispatch(lc, times, params_from_dict(d), d);
            },
            py::arg("times"), py::arg("params"))
        .def("magnification",
            [&](const LC& lc, const TimesArray& times, py::kwargs kw) {
                py::dict d(kw);
                return compute_dispatch(lc, times, params_from_dict(d), d);
            },
            py::arg("times"))

        // Internal route-specialized path populated by Python warmup().
        .def("_magnification_preplanned",
            [&](const LC& lc, const TimesArray& times, py::dict d,
                const std::vector<int>& methods,
                const std::vector<int>& resolutions) {
                return compute_preplanned(
                    lc, times, params_from_dict(d), methods, resolutions);
            },
            py::arg("times"), py::arg("params"), py::arg("methods"),
            py::arg("resolutions"))
        .def("_evaluate_preplanned",
            [&](const LC& lc, const TimesArray& times, py::dict d,
                const std::vector<int>& methods,
                const std::vector<int>& resolutions) {
                return compute_preplanned_diagnostic(
                    lc, times, params_from_dict(d), methods, resolutions);
            },
            py::arg("times"), py::arg("params"), py::arg("methods"),
            py::arg("resolutions"))

        .def("binary_source_components", binary_source_components,
            py::arg("times"), py::arg("params"),
            "Return source-1/source-2 magnifications and trajectories plus the flux-weighted total.\n"
            "Requires LightCurve(source='binary').")

        // Internal inference fast path. Existing scalar call/magnification
        // overloads are unchanged.
        .def("magnification_batch", &compute_batch,
            py::arg("times"), py::arg("parameter_rows"))

        .def("light_curve_log_likelihood_batch",
            &compute_light_curve_log_likelihood_batch,
            py::arg("times"), py::arg("flux"), py::arg("flux_err"),
            py::arg("parameter_rows"), py::arg("distribution") = "gaussian",
            py::arg("flux_mode") = "fit", py::arg("nu") = 4.0,
            py::arg("source_flux") = py::none(),
            py::arg("blend_flux") = py::none())

        // .info() returns diagnostics in addition to magnification.
        .def("info",
            [&](const LC& lc, const TimesArray& times, const lcbi_params& params) {
                if (lc.source_kind() == SKind::single) return compute_info(lc, times, params);
                throw std::runtime_error(
                    "source='binary': LightCurve.info currently supports single-source only");
                return PyLightCurveInfo{};
            },
            py::arg("times"), py::arg("params"))
        .def("info",
            [&](const LC& lc, const TimesArray& times, py::dict d) {
                if (lc.source_kind() == SKind::single) {
                    return compute_info(lc, times, params_from_dict(d));
                }
                throw std::runtime_error(
                    "source='binary': LightCurve.info currently supports single-source only");
                return PyLightCurveInfo{};
            },
            py::arg("times"), py::arg("params"))
	        .def("info",
	            [&](const LC& lc, const TimesArray& times, py::kwargs kw) {
	                if (lc.source_kind() == SKind::single) {
	                    py::dict d(kw);
	                    return compute_info(lc, times, params_from_dict(d));
                }
                throw std::runtime_error(
                    "source='binary': LightCurve.info currently supports single-source only");
                return PyLightCurveInfo{};
	            },
	            py::arg("times"))

	        .def("separation",
	            [](const LC& lc, py::args args, py::kwargs kw) {
	                return effective_binary_separation(
	                    lc, parse_geometry_request(lc, args, kw));
	            },
	            "Effective binary separation. time is required only when orbital motion is enabled.")

	        .def("caustics",
	            [](const LC& lc, py::args args, py::kwargs kw) {
	                return geometry_branches(
	                    lc, parse_geometry_request(lc, args, kw), false);
	            },
	            "Return caustic branch polylines. time is required only with orbital motion.")

	        .def("critical_curves",
	            [](const LC& lc, py::args args, py::kwargs kw) {
	                return geometry_branches(
	                    lc, parse_geometry_request(lc, args, kw), true);
	            },
	            "Return critical-curve branch polylines. time is required only with orbital motion.")

	        // source_trajectory(times, **params)
	        // Returns SourceTrajectory in the lens-plane frame.
	        // Active model terms (parallax, xallarap) are applied.
	        .def("source_trajectory",
	            [](const LC& lc, py::object times_obj, py::kwargs kw) -> PySourceTrajectory {
	                const lcbi_params p = lc.apply_coords(params_from_dict(py::dict(kw)));
	                TimesArray times(times_obj);
	                const std::vector<double> values = times_from_array(times);
	                const int n = static_cast<int>(values.size());

	                PySourceTrajectory result;
	                result.times = values;
	                result.x.resize(n);
	                result.y.resize(n);
	                {
	                    py::gil_scoped_release release;
	                    for (int i = 0; i < n; ++i) {
	                        const auto pos = lens_frame_source_position(lc, p, values[i]);
	                        result.x[i] = pos.x;
	                        result.y[i] = pos.y;
	                    }
	                }
	                return result;
	            }, py::arg("times"),
	            "Compute source trajectory in the lens-plane frame.\n"
	            "Applies all active model terms (parallax, xallarap).\n"
	            "Returns SourceTrajectory with times, x, and y lists (Einstein ring units).")

            .def("finite_source_geometry",
                [](const LC& curve, py::object times_obj, py::kwargs kw) {
                    TimesArray times(times_obj);
                    const std::vector<double> values = times_from_array(times);
                    const lcbi_params parameters = params_from_dict(py::dict(kw));
                    std::vector<lcbinint::magnification::FiniteSourceGeometry> geometry;
                    {
                        py::gil_scoped_release release;
                        geometry = curve.finite_source_geometry(
                            values, parameters);
                    }
                    PyFiniteSourceGeometry result;
                    for (const auto& item : geometry) {
                        result.separation.push_back(item.separation);
                        result.mass_ratio.push_back(item.mass_ratio);
                        result.source_x.push_back(item.source.x);
                        result.source_y.push_back(item.source.y);
                        result.source_radius.push_back(item.source_radius);
                        result.limb_darkening_c.push_back(item.limb_darkening_c);
                        result.limb_darkening_d.push_back(item.limb_darkening_d);
                        result.absolute_tolerance.push_back(item.absolute_tolerance);
                        result.relative_tolerance.push_back(item.relative_tolerance);
                    }
                    return result;
                }, py::arg("times"),
                "Return root-solve-free finite-source geometry using this curve's sky/site/model.")

        .def("__repr__", [](const LC& lc) {
            const auto& o = lc.options();
            std::string pt;
            if (o.vbm_compatible != 0 && o.center_of_mass == 0) pt = "vbm";
            else if (o.vbm_compatible != 0)                      pt = "vbm_center_of_mass";
            else if (o.center_of_mass != 0)                      pt = "center_of_mass";
            else                                                  pt = "lcbinint";
            const std::string lens = lc.lens_kind() == LKind::triple ? " lens='triple'" : "";
            const std::string src = lc.source_kind() == SKind::binary ? " source='binary'" : "";
            return "<lc.LightCurve param_type='" + pt
                + "' source_bins=" + std::to_string(o.source_bins) + lens + src + ">";
        });

    lc.def("_binary_images", &binary_images_to_python,
        py::arg("s"), py::arg("q"), py::arg("x"), py::arg("y"),
        "Return point-source binary-lens image positions in the VBM-compatible lens frame.");

    lc.def("binary_ray_shooting",
        [](double x, double y, double s, double q, double rho,
           const PyLimbDarkening& limb_darkening, lcbi_options options) {
            if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(s) &&
                  std::isfinite(q) && std::isfinite(rho)) ||
                q <= 0.0 || rho <= 0.0) {
                throw std::invalid_argument(
                    "binary_ray_shooting requires finite x, y, s, q and positive rho");
            }

            // This is the direct analogue of VBM's BinaryMag2(s, q, y1, y2,
            // rho): source coordinates are center-of-mass coordinates and the
            // finite-source calculation is forced to an inverse-ray method.
            options.vbm_compatible = 0;
            options.center_of_mass = 1;
            options.parallax_mode = 0;
            options.xallarap_param_type = LCBI_XALLARAP_NONE;
            if (options.mode == 4) options.mode = 1;

            auto params = lcbi_default_params();
            params.t0 = -x;
            params.tE = 1.0;
            params.umin = y;
            params.theta = 0.0;
            params.sep = s;
            params.q = q;
            params.rho = rho;
            params.limb_darkening_c = limb_darkening.c;
            params.limb_darkening_d = limb_darkening.d;

            const auto model_params = lcbinint::model::from_c_params(params);
            lcbinint::model::LensModel lens_model(
                model_params, lcbinint::model::from_c_options(&options));
            const auto result = lens_model.magnification(0.0);
            if (result.status != lcbinint::EvaluationStatus::ok ||
                !std::isfinite(result.magnification)) {
                throw std::runtime_error("binary_ray_shooting failed numerically");
            }
            return result.magnification;
        },
        py::arg("x"), py::arg("y"), py::kw_only(),
        py::arg("s"), py::arg("q"), py::arg("rho"),
        py::arg("limb_darkening") = PyLimbDarkening{},
        py::arg("options") = kDefaultOpts,
        R"(Evaluate a finite binary-lens source at direct lens-plane coordinates.

This is the low-level counterpart of VBMicrolensing's BinaryMag2: ``x`` and
``y`` are center-of-mass source coordinates, and ``rho`` must be positive.
The calculation uses inverse-ray integration; pass ``Options`` to choose the
Cartesian or polar grid and its resolution.)");

    lc.def("_binary_safety_diagnostic",
        [](double separation, double mass_ratio, double source_x, double source_y) {
            lcbinint::magnification::PointSourceMagnifier magnifier;
            return magnifier.binary_safety_diagnostic_cached(
                separation, mass_ratio, {source_x, source_y});
        },
        py::arg("s"), py::arg("q"), py::arg("x"), py::arg("y"),
        "Return local point-source safety indicators for a binary lens.");
}
