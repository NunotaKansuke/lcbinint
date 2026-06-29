#include "bind_obs.hpp"
#include "lcbinint/obs/light_curve_data.hpp"
#include "lcbinint/obs/event.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace lcbinint::obs;

namespace {

// Zero-copy view of a C++ vector as a read-only numpy array.
// `owner` (the Python LightCurveData object) is kept alive via the array's base.
py::array_t<double> vec_view(const std::vector<double>& v, py::object owner)
{
    return py::array_t<double>(
        {static_cast<py::ssize_t>(v.size())},
        {sizeof(double)},
        v.data(),
        owner);
}

} // namespace

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
                std::string observatory) {
            auto t = time.unchecked<1>();
            auto f = flux.unchecked<1>();
            auto e = flux_err.unchecked<1>();
            return std::make_shared<LightCurveData>(
                std::vector<double>(t.data(0), t.data(0) + t.size()),
                std::vector<double>(f.data(0), f.data(0) + f.size()),
                std::vector<double>(e.data(0), e.data(0) + e.size()),
                std::move(name), std::move(band), std::move(observatory));
        }),
            py::arg("time"), py::arg("flux"), py::arg("flux_err"),
            py::arg("name") = "", py::arg("band") = "", py::arg("observatory") = "")
        // Zero-copy numpy views — array borrows C++ memory; LightCurveData stays alive via base
        .def_property_readonly("time", [](py::object self) {
            return vec_view(self.cast<const LightCurveData&>().time(), self);
        })
        .def_property_readonly("flux", [](py::object self) {
            return vec_view(self.cast<const LightCurveData&>().flux(), self);
        })
        .def_property_readonly("flux_err", [](py::object self) {
            return vec_view(self.cast<const LightCurveData&>().flux_err(), self);
        })
        .def_property_readonly("weight", [](py::object self) {
            return vec_view(self.cast<const LightCurveData&>().weight(), self);
        })
        .def("__len__",   &LightCurveData::size)
        .def_property_readonly("size",        &LightCurveData::size)
        .def_property_readonly("name",        &LightCurveData::name)
        .def_property_readonly("band",        &LightCurveData::band)
        .def_property_readonly("observatory", &LightCurveData::observatory)
        .def("__repr__", [](const LightCurveData& d) {
            return "<LightCurveData name='" + d.name()
                + "' n=" + std::to_string(d.size()) + ">";
        });

    py::class_<Event, std::shared_ptr<Event>>(obs, "Event")
        .def(py::init<std::string, double, double, double>(),
            py::arg("name") = "", py::arg("ra") = 0.0,
            py::arg("dec")  = 0.0, py::arg("t_ref") = 0.0)
        .def("add", &Event::add, py::arg("data"))
        .def("__len__", &Event::size)
        .def("__getitem__", [](const Event& e, std::size_t i) -> const LightCurveData& {
            if (i >= e.size())
                throw py::index_error("Event index out of range");
            return e.at(i);
        }, py::return_value_policy::reference_internal)
        .def("__iter__", [](const Event& e) {
            return py::make_iterator<
                py::return_value_policy::reference_internal>(e.begin(), e.end());
        }, py::keep_alive<0, 1>())
        .def_property_readonly("size",  &Event::size)
        .def_property_readonly("name",  &Event::name)
        .def_property_readonly("ra",    &Event::ra)
        .def_property_readonly("dec",   &Event::dec)
        .def_property_readonly("t_ref", &Event::t_ref)
        .def("__repr__", [](const Event& e) {
            return "<Event name='" + e.name()
                + "' datasets=" + std::to_string(e.size()) + ">";
        });
}
