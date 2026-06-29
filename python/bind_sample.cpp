#include "bind_sample.hpp"
#include "lcbinint/sample/chain.hpp"
#include "lcbinint/sample/move.hpp"
#include "lcbinint/sample/stretch_move.hpp"
#include "lcbinint/sample/ensemble_sampler.hpp"
#include "lcbinint/optimize/result.hpp"
#include "lcbinint/bayes/model.hpp"

#include <cmath>
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

        // flat_fluxes: shape (nsteps*nwalkers, n_datasets*2) — [Fs0, Fb0, Fs1, Fb1, ...]
        .def_property_readonly("flat_fluxes", [](py::object self) {
            const Chain& c = self.cast<const Chain&>();
            const auto& fl = c.flat_fluxes();
            if (fl.empty() || c.n_fluxes() == 0) {
                std::vector<py::ssize_t> shape = {0, 0};
                return py::array_t<double>(shape);
            }
            return chain_view_2d(c, fl,
                static_cast<py::ssize_t>(c.nsteps() * c.nwalkers()),
                static_cast<py::ssize_t>(c.n_fluxes()), self);
        })

        .def_property_readonly("transforms",    &Chain::transforms)
        .def_property_readonly("dataset_names", &Chain::dataset_names)

        // samples: physical-space flat samples — applies exp() for log-transformed params
        .def_property_readonly("samples", [](py::object self) {
            const Chain& c = self.cast<const Chain&>();
            const auto& ts = c.transforms();
            const int ndim = c.ndim();
            const int ntot = c.nsteps() * c.nwalkers();
            const auto& raw = c.flat_samples();
            py::array_t<double> out({py::ssize_t(ntot), py::ssize_t(ndim)});
            auto buf = out.mutable_unchecked<2>();
            for (int i = 0; i < ntot; ++i) {
                for (int j = 0; j < ndim; ++j) {
                    const double v = raw[static_cast<std::size_t>(i * ndim + j)];
                    buf(i, j) = (!ts.empty() && ts[j] == "log") ? std::exp(v) : v;
                }
            }
            return out;
        })

        // summary(): dict {param_name: {'median': ..., 'lo': ..., 'hi': ..., 'std': ...}}
        // where lo/hi are 16th/84th percentile.
        .def("summary", [](const Chain& c) {
            const int ndim = c.ndim();
            const int ntot = c.nsteps() * c.nwalkers();
            const auto& ts   = c.transforms();
            const auto& names = c.param_names();
            const auto& raw  = c.flat_samples();

            py::dict result;
            std::vector<double> col(ntot);

            for (int j = 0; j < ndim; ++j) {
                for (int i = 0; i < ntot; ++i) {
                    const double v = raw[static_cast<std::size_t>(i * ndim + j)];
                    col[i] = (!ts.empty() && ts[j] == "log") ? std::exp(v) : v;
                }
                auto sorted = col;
                std::sort(sorted.begin(), sorted.end());
                const double med = sorted[ntot / 2];
                const double lo  = sorted[ntot * 16 / 100];
                const double hi  = sorted[ntot * 84 / 100];
                double mean = 0.0;
                for (double v : col) mean += v;
                mean /= ntot;
                double var = 0.0;
                for (double v : col) var += (v - mean) * (v - mean);
                var /= ntot;

                const std::string& nm = (j < static_cast<int>(names.size())) ? names[j] : ("p" + std::to_string(j));
                result[py::str(nm)] = py::dict(
                    py::arg("median") = med,
                    py::arg("lo")     = lo,
                    py::arg("hi")     = hi,
                    py::arg("std")    = std::sqrt(var));
            }
            return result;
        })

        // tau(c=5.0): integrated autocorrelation time per parameter
        // Returns numpy array of length ndim; NaN where chain is too short.
        .def("tau", [](py::object self, double window) {
            py::gil_scoped_release release;
            return self.cast<const Chain&>().tau(window);
        }, py::arg("c") = 5.0,
        "Integrated autocorrelation time per parameter (Sokal auto-window).\n"
        "Returns array of length ndim. NaN where chain is too short to converge.")

        // ess(): effective sample size = nsteps * nwalkers / tau
        .def("ess", [](py::object self) {
            py::gil_scoped_release release;
            return self.cast<const Chain&>().ess();
        }, "Effective sample size per parameter.")

        .def("__repr__", [](const Chain& c) {
            return "<sample.Chain nsteps=" + std::to_string(c.nsteps())
                + " nwalkers=" + std::to_string(c.nwalkers())
                + " ndim=" + std::to_string(c.ndim())
                + " acceptance=" + std::to_string(c.acceptance()).substr(0, 5) + ">";
        });

    // --- Move hierarchy ---
    py::class_<Move, std::shared_ptr<Move>>(spl, "Move");

    py::class_<StretchMove, Move, std::shared_ptr<StretchMove>>(spl, "StretchMove")
        .def(py::init<double>(), py::arg("a") = 2.0)
        .def_property_readonly("a", &StretchMove::a)
        .def("__repr__", [](const StretchMove& m) {
            return "<sample.StretchMove a=" + std::to_string(m.a()) + ">";
        });

    // --- EnsembleSampler ---
    // GIL released during run() — entire sampling loop runs in C++.
    py::class_<EnsembleSampler>(spl, "EnsembleSampler")
        .def(py::init<int, unsigned int, std::shared_ptr<Move>>(),
            py::arg("nwalkers") = 64,
            py::arg("seed")     = 0u,
            py::arg("move")     = std::make_shared<StretchMove>(2.0))

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
