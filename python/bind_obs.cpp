#include "bind_obs.hpp"
#include "lcbinint/obs/light_curve_data.hpp"
#include "lcbinint/obs/event.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace lcbinint::obs;

void register_obs_submodule(py::module_& parent)
{
    auto obs = parent.def_submodule("obs", "Observational data containers");

    py::class_<LightCurveData, std::shared_ptr<LightCurveData>>(obs, "LightCurveData")
        .def(py::init([](
                py::array_t<double> time,
                py::array_t<double> flux,
                py::array_t<double> flux_err,
                std::string name,
                std::string band,
                std::string observatory)
            {
                auto t = time.unchecked<1>();
                auto f = flux.unchecked<1>();
                auto e = flux_err.unchecked<1>();
                std::vector<double> tv(t.data(0), t.data(0) + t.size());
                std::vector<double> fv(f.data(0), f.data(0) + f.size());
                std::vector<double> ev(e.data(0), e.data(0) + e.size());
                return std::make_shared<LightCurveData>(
                    std::move(tv), std::move(fv), std::move(ev),
                    std::move(name), std::move(band), std::move(observatory));
            }),
            py::arg("time"), py::arg("flux"), py::arg("flux_err"),
            py::arg("name") = "", py::arg("band") = "", py::arg("observatory") = "")
        .def_property_readonly("size",        &LightCurveData::size)
        .def_property_readonly("name",        &LightCurveData::name)
        .def_property_readonly("band",        &LightCurveData::band)
        .def_property_readonly("observatory", &LightCurveData::observatory)
        .def("__repr__", [](const LightCurveData& d) {
            return "<LightCurveData name='" + d.name() + "' n=" + std::to_string(d.size()) + ">";
        });

    py::class_<Event, std::shared_ptr<Event>>(obs, "Event")
        .def(py::init<std::string, double, double, double>(),
            py::arg("name") = "", py::arg("ra") = 0.0,
            py::arg("dec")  = 0.0, py::arg("t_ref") = 0.0)
        .def("add",               &Event::add, py::arg("data"))
        .def_property_readonly("size",  &Event::size)
        .def_property_readonly("name",  &Event::name)
        .def_property_readonly("ra",    &Event::ra)
        .def_property_readonly("dec",   &Event::dec)
        .def_property_readonly("t_ref", &Event::t_ref)
        .def("__repr__", [](const Event& e) {
            return "<Event name='" + e.name() + "' datasets=" + std::to_string(e.size()) + ">";
        });
}
