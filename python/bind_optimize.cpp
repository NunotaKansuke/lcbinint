#include "bind_optimize.hpp"
#include "lcbinint/optimize/result.hpp"
#include "lcbinint/optimize/differential_evolution.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace lcbinint::optimize;

void register_optimize_submodule(py::module_& parent)
{
    auto opt = parent.def_submodule("optimize", "Optimization algorithms");

    py::class_<Result>(opt, "Result")
        .def_readonly("position",       &Result::position)
        .def_readonly("parameters",     &Result::parameters)
        .def_readonly("chi2",           &Result::chi2)
        .def_readonly("log_likelihood", &Result::log_likelihood)
        .def_readonly("log_prob",       &Result::log_prob)
        .def_readonly("success",        &Result::success)
        .def_readonly("message",        &Result::message)
        .def_readonly("n_eval",         &Result::n_eval)
        .def("__repr__", [](const Result& r) {
            return "<optimize.Result chi2=" + std::to_string(r.chi2)
                + " success=" + (r.success ? "True" : "False") + ">";
        });

    py::class_<DifferentialEvolution>(opt, "DifferentialEvolution")
        .def(py::init<int, int, unsigned int>(),
            py::arg("population_size") = 64,
            py::arg("max_iter")        = 2000,
            py::arg("seed")            = 0u)
        .def("minimize", &DifferentialEvolution::minimize,
            py::arg("model"), py::arg("target") = "chi2");
}
