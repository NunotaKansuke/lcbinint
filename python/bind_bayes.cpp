#include "bind_bayes.hpp"
#include "lcbinint/lcbinint.h"
#include "lcbinint/bayes/prior.hpp"
#include "lcbinint/bayes/model.hpp"
#include "lcbinint/lc/light_curve.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace lcbinint::bayes;

void register_bayes_submodule(py::module_& parent)
{
    auto bayes = parent.def_submodule("bayes", "Bayesian model definition");

    // --- Prior base class ---
    py::class_<Prior, std::shared_ptr<Prior>>(bayes, "Prior")
        .def("log_prob", &Prior::log_prob, py::arg("x"))
        .def("bounds", [](const Prior& p) {
            auto b = p.bounds();
            return py::make_tuple(b.lo, b.hi);
        })
        .def("__repr__", &Prior::name);

    // --- Concrete priors ---
    py::class_<Uniform, Prior, std::shared_ptr<Uniform>>(bayes, "Uniform")
        .def(py::init<double, double>(), py::arg("lo"), py::arg("hi"))
        .def("__repr__", [](const Uniform& u) {
            auto b = u.bounds();
            return "Uniform(" + std::to_string(b.lo) + ", " + std::to_string(b.hi) + ")";
        });

    py::class_<Normal, Prior, std::shared_ptr<Normal>>(bayes, "Normal")
        .def(py::init<double, double>(), py::arg("mu"), py::arg("sigma"))
        .def("__repr__", [](const Normal& n) {
            auto b = n.bounds();
            return "Normal(mu=" + std::to_string(b.lo) + ", sigma=...)";
        });

    py::class_<LogUniform, Prior, std::shared_ptr<LogUniform>>(bayes, "LogUniform")
        .def(py::init<double, double>(), py::arg("lo"), py::arg("hi"))
        .def("__repr__", [](const LogUniform& u) {
            auto b = u.bounds();
            return "LogUniform(" + std::to_string(b.lo) + ", " + std::to_string(b.hi) + ")";
        });

    py::class_<Flat, Prior, std::shared_ptr<Flat>>(bayes, "Flat")
        .def(py::init<>())
        .def("__repr__", [](const Flat&) { return "Flat()"; });

    // --- Model (internal C++ base; Python subclass is bayes.Model) ---
    // Takes lc.LightCurve — source kind, orbital motion mode, and options come from it.
    using LC = lcbinint::lc::LightCurve;
    py::class_<Model>(bayes, "_Model")
        .def(py::init<std::shared_ptr<LC>, std::shared_ptr<lcbinint::obs::Event>>(),
            py::arg("light_curve"), py::arg("event"))
        .def(py::init<std::shared_ptr<LC>, std::shared_ptr<lcbinint::obs::LightCurveData>>(),
            py::arg("light_curve"), py::arg("data"))
        .def("param", [](Model& m, std::string name, std::shared_ptr<Prior> prior) {
            m.param(std::move(name), std::move(prior));
        }, py::arg("name"), py::arg("prior"))
        .def("flux",           &Model::flux,          py::arg("mode") = "linear_blend")
        .def("likelihood",     &Model::likelihood,    py::arg("mode") = "gaussian")
        .def("n_params",       &Model::n_params)
        .def_property_readonly("param_names", [](const Model& m) {
            std::vector<std::string> names;
            for (const auto& d : m.param_defs())
                if (!d.fixed) names.push_back(d.name);
            return names;
        })
        .def_property_readonly("optimizer_bounds", [](const Model& m) {
            auto bs = m.optimizer_bounds();
            py::list out;
            for (const auto& b : bs)
                out.append(py::make_tuple(b.lo, b.hi));
            return out;
        })
        .def_property_readonly("n_data",      &Model::n_data)
        .def_property_readonly("options",     [](const Model& m) { return m.options(); })
        .def_property_readonly("light_curve", [](const Model& m) -> const lcbinint::lc::LightCurve& {
            return m.light_curve();
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("t_ref", [](const Model& m) -> py::object {
            auto tr = m.light_curve().t_ref();
            if (!tr) return py::none();
            return py::float_(*tr);
        })
        .def("log_prior",      &Model::log_prior,      py::arg("theta"))
        .def("log_likelihood", &Model::log_likelihood,  py::arg("theta"))
        .def("log_prob",       &Model::log_prob,        py::arg("theta"))
        .def("chi2",           &Model::chi2,            py::arg("theta"))
        .def("residuals",      &Model::residuals,       py::arg("theta"))
        .def("fluxes", [](const Model& m, const std::vector<double>& theta) {
            const auto sols = m.fluxes(theta);
            const auto& ev  = m.event();
            py::dict d;
            for (std::size_t k = 0; k < sols.size(); ++k) {
                const std::string name = (k < ev.size())
                    ? ev.at(k).name() : ("ds" + std::to_string(k));
                d[py::str(name)] = py::dict(
                    py::arg("Fs") = sols[k].Fs,
                    py::arg("Fb") = sols[k].Fb);
            }
            return d;
        }, py::arg("theta"),
        "Return {dataset_name: {'Fs': ..., 'Fb': ...}} for each dataset.");
}
