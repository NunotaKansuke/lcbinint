#include "lcbinint/lcbinint.h"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/math/polynomial_roots.hpp"
#include "lcbinint/model/orbital_motion.hpp"

#include <complex>
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
    std::vector<double> source_x;
    std::vector<double> source_y;
    std::vector<int> image_counts;
};

class PyLensModel {
public:
    PyLensModel(lcbi_params params, lcbi_options options)
        : params_(params), options_(options)
    {
    }

    double magnification(double time) const
    {
        lcbi_result result = {};
        const lcbi_status status = lcbi_magnification(time, &params_, &options_, &result);
        if (status != LCBI_OK) {
            throw std::runtime_error(lcbi_status_string(status));
        }
        return result.magnification;
    }

    std::vector<double> magnifications(const std::vector<double>& times) const
    {
        return light_curve(times).magnifications;
    }

    std::pair<double, double> source_position(double time) const
    {
        lcbi_result result = {};
        const lcbi_status status = lcbi_magnification(time, &params_, &options_, &result);
        if (status != LCBI_OK) {
            throw std::runtime_error(lcbi_status_string(status));
        }
        return {result.source_x, result.source_y};
    }

    std::vector<std::pair<double, double>> source_positions(const std::vector<double>& times) const
    {
        const auto curve = light_curve(times);
        std::vector<std::pair<double, double>> positions;
        positions.reserve(times.size());
        for (std::size_t i = 0; i < times.size(); ++i) {
            positions.push_back({curve.source_x[i], curve.source_y[i]});
        }
        return positions;
    }

    PyLightCurve light_curve(const std::vector<double>& times) const
    {
        std::vector<lcbi_result> results(times.size());
        const lcbi_status status = lcbi_magnification_array(
            times.data(), static_cast<int>(times.size()), &params_, &options_, results.data());
        if (status != LCBI_OK) {
            throw std::runtime_error(lcbi_status_string(status));
        }

        PyLightCurve curve;
        curve.times = times;
        curve.magnifications.reserve(times.size());
        curve.point_source_magnifications.reserve(times.size());
        curve.finite_source_magnifications.reserve(times.size());
        curve.source_x.reserve(times.size());
        curve.source_y.reserve(times.size());
        curve.image_counts.reserve(times.size());
        for (const auto& result : results) {
            curve.magnifications.push_back(result.magnification);
            curve.point_source_magnifications.push_back(result.point_source_magnification);
            curve.finite_source_magnifications.push_back(result.finite_source_magnification);
            curve.source_x.push_back(result.source_x);
            curve.source_y.push_back(result.source_y);
            curve.image_counts.push_back(result.image_count);
        }
        return curve;
    }

private:
    lcbi_params params_;
    lcbi_options options_;
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

    py::enum_<lcbi_finite_source_mode>(m, "FiniteSourceMode")
        .value("AUTO", LCBI_SOURCE_AUTO)
        .value("POINT_SOURCE", LCBI_POINT_SOURCE)
        .value("LEGACY", LCBI_SOURCE_LEGACY)
        .export_values();

    py::enum_<lcbi_inverse_ray_method>(m, "InverseRayMethod")
        .value("AUTO", LCBI_INVERSE_RAY_AUTO)
        .value("CARTESIAN", LCBI_INVERSE_RAY_CARTESIAN)
        .value("POLAR", LCBI_INVERSE_RAY_POLAR)
        .export_values();

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
        .def_readonly("source_x", &PyLightCurve::source_x)
        .def_readonly("source_y", &PyLightCurve::source_y)
        .def_readonly("image_counts", &PyLightCurve::image_counts);

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
                         double lom_ar) {
                 auto params = lcbi_default_params();
                 params.t0 = t0;
                 params.tE = tE;
                 params.umin = umin;
                 params.q = q;
                 params.sep = sep;
                 params.theta = theta;
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
            py::arg("lom_ar") = 1.0)
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
        .def_readwrite("lom_ar", &lcbi_params::lom_ar);

    py::class_<lcbi_options>(m, "Options")
        .def(py::init([](lcbi_finite_source_mode finite_source_mode,
                         lcbi_inverse_ray_method inverse_ray_method,
                         int center_of_mass,
                         double tolerance,
                         double relative_tolerance,
                         int caustic_bins,
                         int source_bins,
                         int legacy_finite_mode,
                         double grid_ratio,
                         double legacy_kinji,
                         double legacy_hex) {
                 auto options = lcbi_default_options();
                 options.finite_source_mode = finite_source_mode;
                 options.inverse_ray_method = inverse_ray_method;
                 options.center_of_mass = center_of_mass;
                 options.tolerance = tolerance;
                 options.relative_tolerance = relative_tolerance;
                 options.caustic_bins = caustic_bins;
                 options.source_bins = source_bins;
                 options.legacy_finite_mode = legacy_finite_mode;
                 options.grid_ratio = grid_ratio;
                 options.legacy_kinji = legacy_kinji;
                 options.legacy_hex = legacy_hex;
                 return options;
             }),
            py::arg("finite_source_mode") = LCBI_SOURCE_AUTO,
            py::arg("inverse_ray_method") = LCBI_INVERSE_RAY_AUTO,
            py::arg("center_of_mass") = 0,
            py::arg("tolerance") = 1.0e-3,
            py::arg("relative_tolerance") = 0.0,
            py::arg("caustic_bins") = 1400,
            py::arg("source_bins") = 20,
            py::arg("legacy_finite_mode") = 4,
            py::arg("grid_ratio") = 4.0,
            py::arg("legacy_kinji") = 9.0,
            py::arg("legacy_hex") = 2.0)
        .def_readwrite("finite_source_mode", &lcbi_options::finite_source_mode)
        .def_readwrite("inverse_ray_method", &lcbi_options::inverse_ray_method)
        .def_readwrite("center_of_mass", &lcbi_options::center_of_mass)
        .def_readwrite("tolerance", &lcbi_options::tolerance)
        .def_readwrite("relative_tolerance", &lcbi_options::relative_tolerance)
        .def_readwrite("caustic_bins", &lcbi_options::caustic_bins)
        .def_readwrite("source_bins", &lcbi_options::source_bins)
        .def_readwrite("legacy_finite_mode", &lcbi_options::legacy_finite_mode)
        .def_readwrite("grid_ratio", &lcbi_options::grid_ratio)
        .def_readwrite("legacy_kinji", &lcbi_options::legacy_kinji)
        .def_readwrite("legacy_hex", &lcbi_options::legacy_hex);

    py::class_<PyLensModel>(m, "LensModel")
        .def(py::init<lcbi_params, lcbi_options>(),
            py::arg("params"),
            py::arg("options") = lcbi_default_options())
        .def("magnification", &PyLensModel::magnification, py::arg("time"))
        .def("magnifications", &PyLensModel::magnifications, py::arg("times"))
        .def("source_position", &PyLensModel::source_position, py::arg("time"))
        .def("source_positions", &PyLensModel::source_positions, py::arg("times"))
        .def("light_curve", &PyLensModel::light_curve, py::arg("times"));

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
