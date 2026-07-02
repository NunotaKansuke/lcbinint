#include "bind_sample.hpp"
#include "lcbinint/sample/chain.hpp"
#include "lcbinint/sample/move.hpp"
#include "lcbinint/sample/sampler_state.hpp"
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
            if (ntot == 0) return result;
            std::vector<double> col(ntot);

            for (int j = 0; j < ndim; ++j) {
                for (int i = 0; i < ntot; ++i) {
                    const double v = raw[static_cast<std::size_t>(i * ndim + j)];
                    col[i] = (!ts.empty() && ts[j] == "log") ? std::exp(v) : v;
                }
                auto sorted = col;
                std::sort(sorted.begin(), sorted.end());
                // Average two middle elements for even ntot (matches scipy.median).
                const double med = (ntot % 2 == 1)
                    ? sorted[ntot / 2]
                    : 0.5 * (sorted[(ntot - 1) / 2] + sorted[ntot / 2]);
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
            const Chain& c = self.cast<const Chain&>();
            py::gil_scoped_release release;
            return c.tau(window);
        }, py::arg("c") = 5.0,
        "Integrated autocorrelation time per parameter (Sokal auto-window).\n"
        "Returns array of length ndim. NaN where chain is too short to converge.")

        // ess(): effective sample size = nsteps * nwalkers / tau
        .def("ess", [](py::object self) {
            const Chain& c = self.cast<const Chain&>();
            py::gil_scoped_release release;
            return c.ess();
        }, "Effective sample size per parameter.")

        .def("__repr__", [](const Chain& c) {
            return "<sample.Chain nsteps=" + std::to_string(c.nsteps())
                + " nwalkers=" + std::to_string(c.nwalkers())
                + " ndim=" + std::to_string(c.ndim())
                + " acceptance=" + std::to_string(c.acceptance()).substr(0, 5) + ">";
        });

    // --- SamplerState ---
    // Mutable state of the ensemble sampler.
    // pos / log_prob / fluxes are zero-copy numpy views into C++ memory;
    // they update in-place after each sampler.step() call.
    using SS = lcbinint::sample::SamplerState;
    py::class_<SS>(spl, "SamplerState")
        .def_property_readonly("nwalkers", [](const SS& s){ return s.nwalkers; })
        .def_property_readonly("ndim",     [](const SS& s){ return s.ndim; })
        .def_property_readonly("n_step",   [](const SS& s){ return s.n_step; })
        .def_property_readonly("acceptance_fraction", &SS::acceptance_fraction)
        // pos: (nwalkers, ndim) zero-copy view
        .def_property_readonly("pos", [](py::object self) {
            const SS& s = self.cast<const SS&>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(s.nwalkers),
                 static_cast<py::ssize_t>(s.ndim)},
                {static_cast<py::ssize_t>(s.ndim * sizeof(double)),
                 static_cast<py::ssize_t>(sizeof(double))},
                s.pos.data(), self);
        })
        // log_prob: (nwalkers,) zero-copy view
        .def_property_readonly("log_prob", [](py::object self) {
            const SS& s = self.cast<const SS&>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(s.nwalkers)},
                {static_cast<py::ssize_t>(sizeof(double))},
                s.log_prob.data(), self);
        })
        // fluxes: (nwalkers, n_fluxes) zero-copy view — [Fs0,Fb0,Fs1,Fb1,...]
        .def_property_readonly("fluxes", [](py::object self) {
            const SS& s = self.cast<const SS&>();
            if (s.n_fluxes == 0) {
                std::vector<py::ssize_t> shape = {0, 0};
                return py::array_t<double>(shape);
            }
            return py::array_t<double>(
                {static_cast<py::ssize_t>(s.nwalkers),
                 static_cast<py::ssize_t>(s.n_fluxes)},
                {static_cast<py::ssize_t>(s.n_fluxes * sizeof(double)),
                 static_cast<py::ssize_t>(sizeof(double))},
                s.fluxes.data(), self);
        })
        // get_chain(): zero-copy view of accumulated history — shape (n_step, nwalkers, ndim)
        .def_property_readonly("get_chain", [](py::object self) {
            const SS& s = self.cast<const SS&>();
            const int hist = s.nwalkers > 0 && s.ndim > 0
                ? static_cast<int>(s.history.size()) / (s.nwalkers * s.ndim) : 0;
            return py::array_t<double>(
                {static_cast<py::ssize_t>(hist),
                 static_cast<py::ssize_t>(s.nwalkers),
                 static_cast<py::ssize_t>(s.ndim)},
                {static_cast<py::ssize_t>(s.nwalkers * s.ndim * sizeof(double)),
                 static_cast<py::ssize_t>(s.ndim * sizeof(double)),
                 static_cast<py::ssize_t>(sizeof(double))},
                s.history.data(), self);
        })
        // flat_log_prob of history — shape (n_step * nwalkers,)
        .def_property_readonly("get_log_prob", [](py::object self) {
            const SS& s = self.cast<const SS&>();
            return py::array_t<double>(
                {static_cast<py::ssize_t>(s.hist_lp.size())},
                {static_cast<py::ssize_t>(sizeof(double))},
                s.hist_lp.data(), self);
        })
        // Free accumulated history (e.g. after flushing a chunk to disk).
        .def("reset_history", &SS::reset_history)
        .def("__repr__", [](const SS& s) {
            return "<sample.SamplerState nwalkers=" + std::to_string(s.nwalkers)
                + " ndim=" + std::to_string(s.ndim)
                + " n_step=" + std::to_string(s.n_step)
                + " accept=" + std::to_string(s.acceptance_fraction()).substr(0,5) + ">";
        });

    // --- chain_from_arrays: build Chain from numpy arrays (used by load_chain) ---
    spl.def("_chain_from_arrays",
        [](py::array_t<double, py::array::c_style> samples,   // (nsteps*nwalkers, ndim)
           py::array_t<double, py::array::c_style> log_prob,  // (nsteps*nwalkers,)
           int nwalkers,
           std::vector<std::string> param_names,
           std::vector<std::string> transforms,
           double acceptance) -> Chain
        {
            auto sb = samples.unchecked<2>();
            auto lb = log_prob.unchecked<1>();
            const int ntot   = static_cast<int>(sb.shape(0));
            const int ndim   = static_cast<int>(sb.shape(1));
            const int nsteps = nwalkers > 0 ? ntot / nwalkers : 0;
            Chain chain;
            chain.init(nsteps, nwalkers, ndim);
            chain.set_param_names(std::move(param_names));
            chain.set_transforms(std::move(transforms));
            if (nsteps > 0)
                chain.assign_flat(sb.data(0, 0), lb.data(0), nullptr);
            chain.set_acceptance(acceptance);
            return chain;
        },
        py::arg("samples"), py::arg("log_prob"), py::arg("nwalkers"),
        py::arg("param_names") = std::vector<std::string>{},
        py::arg("transforms")  = std::vector<std::string>{},
        py::arg("acceptance")  = 0.0);

    // --- _make_state: build SamplerState from numpy arrays (Python init path) ---
    spl.def("_make_state",
        [](int nw, int ndim, unsigned int seed,
           py::array_t<double, py::array::c_style> pos,  // (nw, ndim)
           py::array_t<double, py::array::c_style> lp)   // (nw,)
            -> lcbinint::sample::SamplerState
        {
            auto pb = pos.unchecked<2>();
            auto lb = lp.unchecked<1>();
            lcbinint::sample::SamplerState st;
            st.nwalkers = nw; st.ndim = ndim; st.n_fluxes = 0;
            st.pos.resize(nw * ndim);
            st.log_prob.resize(nw);
            for (int w = 0; w < nw; ++w) {
                for (int j = 0; j < ndim; ++j)
                    st.pos[w * ndim + j] = pb(w, j);
                st.log_prob[w] = lb(w);
            }
            st.rng.seed(seed);
            return st;
        },
        py::arg("nwalkers"), py::arg("ndim"), py::arg("seed"),
        py::arg("pos"), py::arg("log_prob"));

    // --- Move hierarchy ---
    py::class_<Move, std::shared_ptr<Move>>(spl, "Move");

    py::class_<StretchMove, Move, std::shared_ptr<StretchMove>>(spl, "StretchMove")
        .def(py::init<double>(), py::arg("a") = 2.0)
        .def_property_readonly("a", &StretchMove::a)
        .def("__repr__", [](const StretchMove& m) {
            return "<sample.StretchMove a=" + std::to_string(m.a()) + ">";
        });

    // --- EnsembleSampler ---
    py::class_<EnsembleSampler>(spl, "EnsembleSampler")
        .def(py::init<int, unsigned int, std::shared_ptr<Move>>(),
            py::arg("nwalkers") = 64,
            py::arg("seed")     = 0u,
            py::arg("move")     = std::make_shared<StretchMove>(2.0))

        // ----- Step-by-step API -----

        // init_state(model) — random init within prior bounds
        .def("init_state",
            [](EnsembleSampler& s, lcbinint::bayes::Model& model) {
                py::gil_scoped_release release;
                return s.init_state(model);
            },
            py::arg("model"))

        // init_state(model, start, hessian_init=False)
        // start: optimize.Result or list-of-lists
        .def("init_state",
            [](EnsembleSampler& s, lcbinint::bayes::Model& model,
               py::object start, bool hessian_init) {
                if (py::isinstance<lcbinint::optimize::Result>(start)) {
                    const auto& r = start.cast<const lcbinint::optimize::Result&>();
                    py::gil_scoped_release release;
                    return s.init_state(model, r, hessian_init);
                } else {
                    auto pos = start.cast<std::vector<std::vector<double>>>();
                    py::gil_scoped_release release;
                    return s.init_state(model, pos);
                }
            },
            py::arg("model"), py::arg("start"), py::arg("hessian_init") = false)

        // step(model, state) — one ensemble step, in-place
        // GIL must be re-acquired for the call but released during the C++ work.
        .def("step",
            [](EnsembleSampler& s, lcbinint::bayes::Model& model,
               lcbinint::sample::SamplerState& state) {
                py::gil_scoped_release release;
                s.step(model, state);
            },
            py::arg("model"), py::arg("state"))

        // step(py_model, state) — duck-typed overload for Reparameterization.
        // Releases GIL for the stretch-move loop; re-acquires only for log_prob.
        .def("step",
            [](EnsembleSampler& s, py::object model,
               lcbinint::sample::SamplerState& state) {
                auto log_prob_fn = [&model](const std::vector<double>& t) -> double {
                    py::gil_scoped_acquire g;
                    return model.attr("log_prob")(t).cast<double>();
                };
                py::gil_scoped_release release;
                s.step(log_prob_fn, state);
            },
            py::arg("model"), py::arg("state"))

        // collect(model, state) — snapshot current walker positions as a Chain
        .def("collect",
            [](const EnsembleSampler& s, lcbinint::bayes::Model& model,
               const lcbinint::sample::SamplerState& state, int discard) {
                return s.collect(model, state, discard);
            },
            py::arg("model"), py::arg("state"), py::arg("discard") = 0)

        // ----- Batch run() convenience wrappers -----

        // run(model, nsteps, burnin) — random init
        .def("run",
            [](EnsembleSampler& s, lcbinint::bayes::Model& model,
               int nsteps, int burnin) {
                py::gil_scoped_release release;
                return s.run(model, nsteps, burnin);
            },
            py::arg("model"), py::arg("nsteps") = 1000, py::arg("burnin") = 0)

        // run(model, start, nsteps, burnin, hessian_init)
        .def("run",
            [](EnsembleSampler& s, lcbinint::bayes::Model& model,
               py::object start, int nsteps, int burnin, bool hessian_init) {
                if (py::isinstance<lcbinint::optimize::Result>(start)) {
                    const auto& r = start.cast<const lcbinint::optimize::Result&>();
                    py::gil_scoped_release release;
                    return s.run(model, r, nsteps, burnin, hessian_init);
                } else {
                    auto pos = start.cast<std::vector<std::vector<double>>>();
                    py::gil_scoped_release release;
                    return s.run(model, pos, nsteps, burnin);
                }
            },
            py::arg("model"), py::arg("start"),
            py::arg("nsteps") = 1000, py::arg("burnin") = 0,
            py::arg("hessian_init") = false);
}
