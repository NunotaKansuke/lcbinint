#include "bind_optimize.hpp"
#include "lcbinint/optimize/result.hpp"
#include "lcbinint/optimize/differential_evolution.hpp"
#include "lcbinint/optimize/levenberg_marquardt.hpp"
#include "lcbinint/bayes/model.hpp"

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
        .def_readonly("n_iter",         &Result::n_iter)
        .def("__repr__", [](const Result& r) {
            return "<optimize.Result chi2=" + std::to_string(r.chi2)
                + " n_eval=" + std::to_string(r.n_eval)
                + " success=" + (r.success ? "True" : "False") + ">";
        });

    // GIL released during minimize() — entire optimization loop runs in C++.
    py::class_<DifferentialEvolution>(opt, "DifferentialEvolution")
        .def(py::init<int, int, double, double, unsigned int, std::string>(),
            py::arg("population_size") = 64,
            py::arg("max_iter")        = 2000,
            py::arg("mutation_factor") = 0.8,
            py::arg("crossover_prob")  = 0.9,
            py::arg("seed")            = 0u,
            py::arg("strategy")        = "rand1bin")
        .def("minimize",
            [](DifferentialEvolution& de, lcbinint::bayes::Model& model,
               const std::string& target) {
                py::gil_scoped_release release;
                return de.minimize(model, target);
            },
            py::arg("model"), py::arg("target") = "chi2");

    // GIL released during minimize() — Jacobian build and linear solve in C++.
    py::class_<LevenbergMarquardt>(opt, "LevenbergMarquardt")
        .def(py::init<int, double, double, double, double, double, double, double>(),
            py::arg("max_iter")    = 200,
            py::arg("ftol")        = 1e-8,
            py::arg("xtol")        = 1e-8,
            py::arg("gtol")        = 1e-8,
            py::arg("fd_step")     = 1e-5,
            py::arg("lambda_init") = 1e-3,
            py::arg("lambda_up")   = 3.0,
            py::arg("lambda_down") = 3.0)
        // start can be optimize.Result or list/array of theta values.
        // Python processing (cast) happens before GIL release.
        .def("minimize",
            [](LevenbergMarquardt& lm, lcbinint::bayes::Model& model,
               py::object start) {
                std::vector<double> theta;
                if (py::isinstance<Result>(start))
                    theta = start.cast<const Result&>().position;
                else
                    theta = start.cast<std::vector<double>>();
                py::gil_scoped_release release;
                return lm.minimize(model, theta);
            },
            py::arg("model"), py::arg("start"));
}
