#include "lcbinint/lcbinint.h"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/math/polynomial_roots.hpp"
#include "lcbinint/model/lens_model.hpp"
#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/model/orbital_motion.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

#include <pybind11/complex.h>
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
    std::vector<int> finite_source_refinement_levels;
    std::vector<bool> finite_source_converged;
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

class PyLensModel {
public:
    PyLensModel(lcbi_params params, lcbi_options options)
        : params_(params)
        , options_(options)
        , model_(lcbinint::model::from_c_params(params_),
                 lcbinint::model::from_c_options(&options_))
    {
    }

    double magnification(double time) const
    {
        const auto result = model_.magnification(time);
        if (result.status == lcbinint::EvaluationStatus::unsupported) {
            throw std::runtime_error("unsupported");
        }
        if (result.status == lcbinint::EvaluationStatus::numerical_error ||
            !std::isfinite(result.magnification)) {
            throw std::runtime_error("numerical error");
        }
        return result.magnification;
    }

    std::vector<double> magnifications(const std::vector<double>& times) const
    {
        std::vector<double> values;
        values.reserve(times.size());
        for (const double t : times) {
            values.push_back(magnification(t));
        }
        return values;
    }

    std::pair<double, double> source_position(double time) const
    {
        const auto result = model_.magnification(time);
        return {result.source.x, result.source.y};
    }

    std::vector<std::pair<double, double>> source_positions(const std::vector<double>& times) const
    {
        std::vector<std::pair<double, double>> positions;
        positions.reserve(times.size());
        for (const double t : times) {
            positions.push_back(source_position(t));
        }
        return positions;
    }

    PyLightCurve light_curve(const std::vector<double>& times) const
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
        curve.finite_source_refinement_levels.reserve(times.size());
        curve.finite_source_converged.reserve(times.size());
        for (const double t : times) {
            const auto result = model_.magnification(t);
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
            curve.finite_source_refinement_levels.push_back(result.finite_source_refinement_level);
            curve.finite_source_converged.push_back(result.finite_source_converged);
        }
        return curve;
    }

    PySourceBinEstimate estimate_source_bins(
        const std::vector<double>& times,
        std::vector<int> candidate_bins,
        int max_sample_points) const
    {
        if (max_sample_points <= 0) {
            throw std::invalid_argument("max_sample_points must be positive");
        }

        PySourceBinEstimate estimate;
        if (times.empty()) {
            return estimate;
        }

        if (candidate_bins.empty()) {
            candidate_bins = {20, 30, 40, 50, 60, 80};
        }
        std::sort(candidate_bins.begin(), candidate_bins.end());
        candidate_bins.erase(std::remove_if(candidate_bins.begin(), candidate_bins.end(), [](int bins) {
            return bins <= 0;
        }), candidate_bins.end());
        candidate_bins.erase(std::unique(candidate_bins.begin(), candidate_bins.end()), candidate_bins.end());
        if (candidate_bins.empty()) {
            throw std::invalid_argument("candidate_bins must contain at least one positive value");
        }

        const int sample_count = std::min<int>(static_cast<int>(times.size()), max_sample_points);
        estimate.sampled_times.reserve(static_cast<std::size_t>(sample_count));
        if (sample_count == static_cast<int>(times.size())) {
            estimate.sampled_times = times;
        } else if (sample_count == 1) {
            estimate.sampled_times.push_back(times[times.size() / 2]);
        } else {
            for (int i = 0; i < sample_count; ++i) {
                const double position =
                    static_cast<double>(i) * static_cast<double>(times.size() - 1) /
                    static_cast<double>(sample_count - 1);
                const auto index = static_cast<std::size_t>(std::llround(position));
                estimate.sampled_times.push_back(times[index]);
            }
        }

        estimate.reference_source_bins = candidate_bins.back();
        estimate.recommended_source_bins = estimate.reference_source_bins;

        auto reference_options = options_;
        reference_options.source_bins = estimate.reference_source_bins;
        const auto reference = magnifications_with_options(estimate.sampled_times, reference_options);

        estimate.candidates.reserve(candidate_bins.size());
        bool found_recommendation = false;
        for (const int bins : candidate_bins) {
            auto test_options = options_;
            test_options.source_bins = bins;
            const auto values = bins == estimate.reference_source_bins ?
                reference :
                magnifications_with_options(estimate.sampled_times, test_options);

            PySourceBinCandidate candidate;
            candidate.source_bins = bins;
            double squared_relative = 0.0;
            bool accepted = bins == estimate.reference_source_bins;
            for (std::size_t i = 0; i < values.size(); ++i) {
                const double difference = std::abs(values[i] - reference[i]);
                const double relative_denominator = std::max(std::abs(reference[i]), 1.0e-300);
                const double relative_difference = difference / relative_denominator;
                candidate.max_absolute_difference =
                    std::max(candidate.max_absolute_difference, difference);
                candidate.max_relative_difference =
                    std::max(candidate.max_relative_difference, relative_difference);
                squared_relative += relative_difference * relative_difference;
            }
            candidate.rms_relative_difference =
                values.empty() ? 0.0 : std::sqrt(squared_relative / static_cast<double>(values.size()));
            candidate.accepted = accepted;
            if (accepted && !found_recommendation) {
                estimate.recommended_source_bins = bins;
                found_recommendation = true;
            }
            estimate.candidates.push_back(candidate);
        }

        return estimate;
    }

private:
    std::vector<double> magnifications_with_options(
        const std::vector<double>& times,
        const lcbi_options& options) const
    {
        std::vector<lcbi_result> results(times.size());
        const lcbi_status status = lcbi_magnification_array(
            times.data(), static_cast<int>(times.size()), &params_, &options, results.data());
        if (status != LCBI_OK) {
            throw std::runtime_error(lcbi_status_string(status));
        }

        std::vector<double> values;
        values.reserve(times.size());
        for (const auto& result : results) {
            values.push_back(result.magnification);
        }
        return values;
    }

    lcbi_params params_;
    lcbi_options options_;
    lcbinint::model::LensModel model_;
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

    py::class_<PyLightCurve>(m, "LightCurve")
        .def_readonly("times", &PyLightCurve::times)
        .def_readonly("magnifications", &PyLightCurve::magnifications)
        .def_readonly("point_source_magnifications", &PyLightCurve::point_source_magnifications)
        .def_readonly("finite_source_magnifications", &PyLightCurve::finite_source_magnifications)
        .def_readonly("finite_source_error_estimates", &PyLightCurve::finite_source_error_estimates)
        .def_readonly("source_x", &PyLightCurve::source_x)
        .def_readonly("source_y", &PyLightCurve::source_y)
        .def_readonly("image_counts", &PyLightCurve::image_counts)
        .def_readonly("finite_source_refinement_levels", &PyLightCurve::finite_source_refinement_levels)
        .def_readonly("finite_source_converged", &PyLightCurve::finite_source_converged)
        .def_property_readonly("all_converged", &all_converged)
        .def_property_readonly("unconverged_indices", &unconverged_indices);

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
            py::arg("u0") = kNaN,    // VBBL alias for umin; takes priority when set
            py::arg("alpha") = kNaN) // VBBL alias for theta; takes priority when set
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
        .def(py::init([](int center_of_mass,
                         int source_bins,
                         int mode,
                         int caustic_bins,
                         double grid_ratio,
                         double point_source_threshold,
                         double hexadecapole_threshold,
                         double adaptive_hex_threshold,
                         int vbbl_compatible,
                         int adaptive_source_bins,
                         int max_source_bins,
                         double finite_source_tol,
                         double finite_source_reltol,
                         double tol,
                         double reltol) {
                 auto options = lcbi_default_options();
                 options.center_of_mass = center_of_mass;
                 options.source_bins = source_bins;
                 options.mode = mode;
                 options.caustic_bins = caustic_bins;
                 options.grid_ratio = grid_ratio;
                 options.point_source_threshold = point_source_threshold;
                 options.hexadecapole_threshold = hexadecapole_threshold;
                 options.adaptive_hex_threshold = adaptive_hex_threshold;
                 options.vbbl_compatible = vbbl_compatible;
                 options.max_source_bins = max_source_bins;
                 options.finite_source_tol = std::isnan(tol) ? finite_source_tol : tol;
                 options.finite_source_reltol = std::isnan(reltol) ? finite_source_reltol : reltol;
                 if (adaptive_source_bins < 0) {
                     options.adaptive_source_bins =
                         (options.finite_source_tol > 0.0 || options.finite_source_reltol > 0.0) ? 1 : 0;
                 } else {
                     options.adaptive_source_bins = adaptive_source_bins;
                 }
                 return options;
             }),
            py::arg("center_of_mass") = 0,
            py::arg("source_bins") = 64,
            py::arg("mode") = 1,
            py::arg("caustic_bins") = 1400,
            py::arg("grid_ratio") = 4.0,
            py::arg("point_source_threshold") = 20.0,
            py::arg("hexadecapole_threshold") = 3.0,
            py::arg("adaptive_hex_threshold") = 0.001,
            py::arg("vbbl_compatible") = 0,
            py::arg("adaptive_source_bins") = -1,
            py::arg("max_source_bins") = 400,
            py::arg("finite_source_tol") = 0.0,
            py::arg("finite_source_reltol") = 1.0e-3,
            py::arg("tol") = kNaN,
            py::arg("reltol") = kNaN)
        .def_readwrite("center_of_mass", &lcbi_options::center_of_mass)
        .def_readwrite("source_bins", &lcbi_options::source_bins)
        .def_readwrite("mode", &lcbi_options::mode)
        .def_readwrite("caustic_bins", &lcbi_options::caustic_bins)
        .def_readwrite("grid_ratio", &lcbi_options::grid_ratio)
        .def_readwrite("point_source_threshold", &lcbi_options::point_source_threshold)
        .def_readwrite("hexadecapole_threshold", &lcbi_options::hexadecapole_threshold)
        .def_readwrite("adaptive_hex_threshold", &lcbi_options::adaptive_hex_threshold)
        .def_readwrite("vbbl_compatible", &lcbi_options::vbbl_compatible)
        .def_readwrite("adaptive_source_bins", &lcbi_options::adaptive_source_bins)
        .def_readwrite("max_source_bins", &lcbi_options::max_source_bins)
        .def_readwrite("finite_source_tol", &lcbi_options::finite_source_tol)
        .def_readwrite("finite_source_reltol", &lcbi_options::finite_source_reltol)
        .def_property("tol",
            [](const lcbi_options& o) { return o.finite_source_tol; },
            [](lcbi_options& o, double v) { o.finite_source_tol = v; })
        .def_property("reltol",
            [](const lcbi_options& o) { return o.finite_source_reltol; },
            [](lcbi_options& o, double v) { o.finite_source_reltol = v; });

    py::class_<PyLensModel>(m, "LensModel")
        .def(py::init<lcbi_params, lcbi_options>(),
            py::arg("params"),
            py::arg("options") = lcbi_default_options())
        .def("magnification", &PyLensModel::magnification, py::arg("time"))
        .def("magnifications", &PyLensModel::magnifications, py::arg("times"))
        .def("source_position", &PyLensModel::source_position, py::arg("time"))
        .def("source_positions", &PyLensModel::source_positions, py::arg("times"))
        .def("light_curve", &PyLensModel::light_curve, py::arg("times"))
        .def("estimate_source_bins",
            &PyLensModel::estimate_source_bins,
            py::arg("times"),
            py::arg("candidate_bins") = std::vector<int> {20, 30, 40, 50, 60, 80},
            py::arg("max_sample_points") = 64);

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
        "Binary point-source magnification matching VBBinaryLensing BinaryMag0 coordinates.");

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
        "VBBinaryLensing-compatible circular 3D orbital-motion state.");

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
        "VBBinaryLensing-compatible Keplerian 3D orbital-motion state.");
}
