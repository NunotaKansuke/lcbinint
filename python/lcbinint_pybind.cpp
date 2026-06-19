#include "lcbinint/lcbinint.h"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/math/polynomial_roots.hpp"

#include <complex>
#include <stdexcept>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

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
        std::vector<double> values;
        values.reserve(times.size());
        for (double time : times) {
            values.push_back(magnification(time));
        }
        return values;
    }

private:
    lcbi_params params_;
    lcbi_options options_;
};

} // namespace

PYBIND11_MODULE(lcbinint, m)
{
    m.doc() = "Python bindings for the lcbinint C core";

    py::enum_<lcbi_status>(m, "Status")
        .value("OK", LCBI_OK)
        .value("INVALID_ARGUMENT", LCBI_INVALID_ARGUMENT)
        .value("NUMERICAL_ERROR", LCBI_NUMERICAL_ERROR)
        .value("UNSUPPORTED", LCBI_UNSUPPORTED);

    m.def("status_string", [](lcbi_status status) {
        return lcbi_status_string(status);
    });

    py::class_<lcbi_params>(m, "LensParams")
        .def(py::init([]() { return lcbi_default_params(); }))
        .def_readwrite("t0", &lcbi_params::t0)
        .def_readwrite("tE", &lcbi_params::tE)
        .def_readwrite("umin", &lcbi_params::umin)
        .def_readwrite("q", &lcbi_params::q)
        .def_readwrite("sep", &lcbi_params::sep)
        .def_readwrite("theta", &lcbi_params::theta)
        .def_readwrite("rho", &lcbi_params::rho)
        .def_readwrite("omega", &lcbi_params::omega)
        .def_readwrite("v_sep", &lcbi_params::v_sep)
        .def_readwrite("q2", &lcbi_params::q2)
        .def_readwrite("sep2", &lcbi_params::sep2)
        .def_readwrite("ang", &lcbi_params::ang);

    py::class_<lcbi_options>(m, "Options")
        .def(py::init([]() { return lcbi_default_options(); }))
        .def_readwrite("finite_source_mode", &lcbi_options::finite_source_mode)
        .def_readwrite("center_of_mass", &lcbi_options::center_of_mass);

    py::class_<PyLensModel>(m, "LensModel")
        .def(py::init<lcbi_params, lcbi_options>(),
            py::arg("params"),
            py::arg("options") = lcbi_default_options())
        .def("magnification", &PyLensModel::magnification, py::arg("time"))
        .def("magnifications", &PyLensModel::magnifications, py::arg("times"));

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
}
