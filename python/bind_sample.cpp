#include "bind_sample.hpp"
#include "lcbinint/sample/chain.hpp"
#include "lcbinint/sample/ensemble_sampler.hpp"
#include "lcbinint/optimize/result.hpp"
#include "lcbinint/bayes/model.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace lcbinint::sample;

namespace {

// Zero-copy numpy view of a Chain's flat data with Chain object as base.
py::array_t<double> chain_view_1d(const Chain& chain,
                                   const std::vector<double>& data,
                                   py::object owner)
{
    return py::array_t<double>(
        {static_cast<py::ssize_t>(data.size())},
        {sizeof(double)},
        data.data(), owner);
}

py::array_t<double> chain_view_2d(const Chain& chain,
                                   const std::vector<double>& data,
                                   py::ssize_t rows, py::ssize_t cols,
                                   py::object owner)
{
    return py::array_t<double>(
        {rows, cols},
        {cols * static_cast<py::ssize_t>(sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        data.data(), owner);
}

} // namespace

void register_sample_submodule(py::module_& parent)
{
    auto spl = parent.def_submodule("sample", "Posterior sampling");

    // --- Chain ---
    py::class_<Chain>(spl, "Chain")
        .def_property_readonly("nsteps",   &Chain::nsteps)
        .def_property_readonly("nwalkers", &Chain::nwalkers)
        .def_property_readonly("ndim",     &Chain::ndim)
        .def_property_readonly("acceptance_fraction", &Chain::acceptance)
        .def_property_readonly("param_names", &Chain::param_names)

        // flat_samples: shape (nsteps*nwalkers, ndim) — zero-copy numpy view
        .def_property_readonly("flat_samples", [](py::object self) {
            const Chain& c = self.cast<const Chain&>();
            return chain_view_2d(c, c.flat_samples(),
                static_cast<py::ssize_t>(c.nsteps() * c.nwalkers()),
                static_cast<py::ssize_t>(c.ndim()), self);
        })

        // get_chain(): shape (nsteps, nwalkers, ndim) — zero-copy numpy view
        .def("get_chain", [](py::object self) {
            const Chain& c = self.cast<const Chain&>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(c.nsteps()),
                 static_cast<py::ssize_t>(c.nwalkers()),
                 static_cast<py::ssize_t>(c.ndim())},
                {static_cast<py::ssize_t>(c.nwalkers() * c.ndim() * sizeof(double)),
                 static_cast<py::ssize_t>(c.ndim() * sizeof(double)),
                 static_cast<py::ssize_t>(sizeof(double))},
                c.flat_samples().data(), self);
        })

        // flat_log_prob: shape (nsteps*nwalkers,) — zero-copy
        .def_property_readonly("flat_log_prob", [](py::object self) {
            const Chain& c = self.cast<const Chain&>();
            return chain_view_1d(c, c.flat_log_prob(), self);
        })

        .def("__repr__", [](const Chain& c) {
            return "<sample.Chain nsteps=" + std::to_string(c.nsteps())
                + " nwalkers=" + std::to_string(c.nwalkers())
                + " ndim=" + std::to_string(c.ndim())
                + " acceptance=" + std::to_string(c.acceptance()).substr(0, 5) + ">";
        });

    // --- EnsembleSampler ---
    // GIL released during run() — entire sampling loop runs in C++.
    py::class_<EnsembleSampler>(spl, "EnsembleSampler")
        .def(py::init<int, unsigned int, double>(),
            py::arg("nwalkers") = 64,
            py::arg("seed")     = 0u,
            py::arg("a")        = 2.0)

        // run(model, nsteps, burnin) — start from prior bounds
        .def("run",
            [](EnsembleSampler& s, lcbinint::bayes::Model& model,
               int nsteps, int burnin) {
                py::gil_scoped_release release;
                return s.run(model, nsteps, burnin);
            },
            py::arg("model"), py::arg("nsteps") = 1000, py::arg("burnin") = 0)

        // run(model, start=..., nsteps, burnin)
        // start can be optimize.Result or list-of-lists (nwalkers × ndim).
        // Python processing before GIL release to avoid pybind11 overload ambiguity.
        .def("run",
            [](EnsembleSampler& s, lcbinint::bayes::Model& model,
               py::object start, int nsteps, int burnin) {
                if (py::isinstance<lcbinint::optimize::Result>(start)) {
                    const auto& r = start.cast<const lcbinint::optimize::Result&>();
                    py::gil_scoped_release release;
                    return s.run(model, r, nsteps, burnin);
                } else {
                    auto pos = start.cast<std::vector<std::vector<double>>>();
                    py::gil_scoped_release release;
                    return s.run(model, pos, nsteps, burnin);
                }
            },
            py::arg("model"), py::arg("start"),
            py::arg("nsteps") = 1000, py::arg("burnin") = 0);
}
