#include "bind_obs.hpp"
#include "lcbinint/obs/coordinates.hpp"
#include "lcbinint/obs/light_curve_data.hpp"
#include "lcbinint/obs/event.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

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

// Parse "hh:mm:ss.ss" or "hh mm ss.ss" → hours → degrees (*15).
// Also accepts plain float (always interpreted as degrees).
double parse_ra(py::object obj, const std::string& unit)
{
    if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
        double val = obj.cast<double>();
        if (unit == "hours") val *= 15.0;
        return val;
    }
    // String: hh:mm:ss.ss
    std::string s = obj.cast<std::string>();
    double h = 0, m = 0, sec = 0;
    char sep1, sep2;
    std::istringstream ss(s);
    if (!(ss >> h >> sep1 >> m >> sep2 >> sec))
        throw std::invalid_argument("Cannot parse RA: '" + s + "'. Expected hh:mm:ss.ss");
    return (h + m / 60.0 + sec / 3600.0) * 15.0;
}

// Parse "±dd:mm:ss.ss" → degrees. Also accepts plain float.
double parse_dec(py::object obj)
{
    if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj))
        return obj.cast<double>();
    std::string s = obj.cast<std::string>();
    bool neg = (!s.empty() && s[0] == '-');
    if (!s.empty() && (s[0] == '+' || s[0] == '-')) s = s.substr(1);
    double d = 0, m = 0, sec = 0;
    char sep1, sep2;
    std::istringstream ss(s);
    if (!(ss >> d >> sep1 >> m >> sep2 >> sec))
        throw std::invalid_argument("Cannot parse Dec: '" + s + "'. Expected ±dd:mm:ss.ss");
    double val = d + m / 60.0 + sec / 3600.0;
    return neg ? -val : val;
}

// Parse "±dd:mm:ss.ss" or plain float for lat/lon (always degrees).
double parse_angle_deg(py::object obj)
{
    if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj))
        return obj.cast<double>();
    return parse_dec(obj);  // same format
}

} // namespace

void register_obs_submodule(py::module_& parent)
{
    auto obs = parent.def_submodule("obs", "Observational data containers");

    // --- SkyCoord ---
    py::class_<SkyCoord, std::shared_ptr<SkyCoord>>(obs, "SkyCoord")
        .def(py::init([](py::object ra, py::object dec, std::string unit) {
            return std::make_shared<SkyCoord>(parse_ra(ra, unit), parse_dec(dec));
        }),
            py::arg("ra"), py::arg("dec"), py::arg("unit") = "deg",
            R"(Sky coordinates of a microlensing event.

ra  : float (degrees) or str "hh:mm:ss.ss"  [unit='deg'/'hours']
dec : float (degrees) or str "±dd:mm:ss.ss")")
        .def_property_readonly("ra_deg",  &SkyCoord::ra_deg)
        .def_property_readonly("dec_deg", &SkyCoord::dec_deg)
        .def_property_readonly("ra_hms", [](const SkyCoord& sc) {
            double h_total = sc.ra_deg() / 15.0;
            int h = static_cast<int>(h_total);
            double rem = (h_total - h) * 60.0;
            int m = static_cast<int>(rem);
            double s = (rem - m) * 60.0;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%02d:%02d:%06.3f", h, m, s);
            return std::string(buf);
        })
        .def_property_readonly("dec_dms", [](const SkyCoord& sc) {
            double d_total = std::abs(sc.dec_deg());
            int d = static_cast<int>(d_total);
            double rem = (d_total - d) * 60.0;
            int m = static_cast<int>(rem);
            double s = (rem - m) * 60.0;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s%02d:%02d:%05.2f",
                          sc.dec_deg() < 0 ? "-" : "+", d, m, s);
            return std::string(buf);
        })
        .def("__repr__", [](const SkyCoord& sc) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "<SkyCoord ra=%.5f dec=%.5f [deg]>",
                          sc.ra_deg(), sc.dec_deg());
            return std::string(buf);
        });

    // --- Site ---
    py::class_<Site, std::shared_ptr<Site>>(obs, "Site")
        .def(py::init([](py::object lat, py::object lon) {
            return std::make_shared<Site>(parse_angle_deg(lat), parse_angle_deg(lon));
        }),
            py::arg("lat"), py::arg("lon"),
            R"(Observatory/telescope position.

lat : float or str "±dd:mm:ss.ss"  [degrees, positive = North]
lon : float or str "±dd:mm:ss.ss"  [degrees East])")
        .def_property_readonly("lat_deg", &Site::lat_deg)
        .def_property_readonly("lon_deg", &Site::lon_deg)
        .def("__repr__", [](const Site& s) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "<Site lat=%.4f lon=%.4f [deg]>",
                          s.lat_deg(), s.lon_deg());
            return std::string(buf);
        });

    // --- LightCurveData ---
    py::class_<LightCurveData, std::shared_ptr<LightCurveData>>(obs, "LightCurveData")
        .def(py::init([](
                py::array_t<double>         time,
                py::array_t<double>         flux,
                py::array_t<double>         flux_err,
                std::string                 name,
                std::string                 band,
                std::string                 observatory,
                std::shared_ptr<Site>       site) {
            auto t = time.unchecked<1>();
            auto f = flux.unchecked<1>();
            auto e = flux_err.unchecked<1>();
            return std::make_shared<LightCurveData>(
                std::vector<double>(t.data(0), t.data(0) + t.size()),
                std::vector<double>(f.data(0), f.data(0) + f.size()),
                std::vector<double>(e.data(0), e.data(0) + e.size()),
                std::move(name), std::move(band), std::move(observatory),
                std::move(site));
        }),
            py::arg("time"), py::arg("flux"), py::arg("flux_err"),
            py::arg("name") = "", py::arg("band") = "", py::arg("observatory") = "",
            py::arg("site") = py::none())
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
        .def_property_readonly("site", [](const LightCurveData& d) -> py::object {
            if (!d.site()) return py::none();
            return py::cast(d.site());
        })
        .def("__repr__", [](const LightCurveData& d) {
            return "<LightCurveData name='" + d.name()
                + "' n=" + std::to_string(d.size()) + ">";
        });

    // --- Event ---
    py::class_<Event, std::shared_ptr<Event>>(obs, "Event")
        .def(py::init([](std::string name, py::object sky) {
            std::shared_ptr<SkyCoord> sc;
            if (!sky.is_none()) sc = sky.cast<std::shared_ptr<SkyCoord>>();
            return std::make_shared<Event>(std::move(name), std::move(sc));
        }),
            py::arg("name") = "", py::arg("sky") = py::none())
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
        .def_property_readonly("sky", [](const Event& e) -> py::object {
            if (!e.sky_coord()) return py::none();
            return py::cast(e.sky_coord());
        })
        // Convenience: plain ra/dec in degrees (None if not set)
        .def_property_readonly("ra",  [](const Event& e) -> py::object {
            if (!e.sky_coord()) return py::none();
            return py::float_(e.ra());
        })
        .def_property_readonly("dec", [](const Event& e) -> py::object {
            if (!e.sky_coord()) return py::none();
            return py::float_(e.dec());
        })
        .def("__repr__", [](const Event& e) {
            return "<Event name='" + e.name()
                + "' datasets=" + std::to_string(e.size()) + ">";
        });
}
