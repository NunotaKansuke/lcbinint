#include "bind_sample.hpp"
#include "lcbinint/sample/chain.hpp"
#include "lcbinint/sample/ensemble_sampler.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace lcbinint::sample;

void register_sample_submodule(py::module_& parent)
{
    auto spl = parent.def_submodule("sample", "Posterior sampling");

    py::class_<Chain>(spl, "Chain")
        .def_property_readonly("size",       &Chain::size)
        .def_property_readonly("n_params",   &Chain::n_params)
        .def_property_readonly("acceptance", &Chain::acceptance)
        .def_property_readonly("samples",    &Chain::samples)
        .def_property_readonly("log_prob",   &Chain::log_prob)
        .def("__repr__", [](const Chain& c) {
            return "<sample.Chain steps=" + std::to_string(c.size())
                + " acceptance=" + std::to_string(c.acceptance()) + ">";
        });

    py::class_<EnsembleSampler>(spl, "EnsembleSampler")
        .def(py::init<int, unsigned int>(),
            py::arg("nwalkers") = 64,
            py::arg("seed")     = 0u)
        .def("run", [](EnsembleSampler& s, lcbinint::bayes::Model& model,
                       int nsteps, int burnin) {
            return s.run(model, nsteps, burnin, nullptr);
        }, py::arg("model"), py::arg("nsteps") = 1000, py::arg("burnin") = 0);
}
