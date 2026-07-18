#include "bind_obs.hpp"
#include "lcbinint/obs/coordinates.hpp"

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
    auto obs = parent.def_submodule("obs", "Site/coordinate geometry for parallax calculations");

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
        .def(py::init([](py::args args) {
            if (args.empty() || !py::isinstance<py::str>(args[0])) {
                throw py::type_error("Site requires 'ground' or 'space' as its first argument");
            }
            const std::string kind = args[0].cast<std::string>();
            if (kind == "ground") {
                if (args.size() == 1) {
                    return std::make_shared<Site>();
                }
                if (args.size() != 3) {
                    throw py::type_error("Site('ground') accepts optional (lat, lon)");
                }
                return std::make_shared<Site>(
                    parse_angle_deg(args[1]), parse_angle_deg(args[2]));
            }
            if (kind != "space") {
                throw std::invalid_argument("site kind must be 'ground' or 'space'");
            }
            if (args.size() == 1) {
                return std::make_shared<Site>(
                    std::vector<double>{}, std::vector<std::array<double, 3>>{});
            }
            if (args.size() != 2) {
                throw py::type_error("Site('space') accepts an optional table");
            }
            py::array_t<double, py::array::c_style | py::array::forcecast> values(args[1]);
            const auto rows = values.unchecked<2>();
            if (rows.shape(0) < 2 || rows.shape(1) != 4) {
                throw std::invalid_argument(
                    "space Site table must have shape (N, 4): JD, RA_deg, Dec_deg, distance_AU");
            }
            std::vector<double> times;
            std::vector<std::array<double, 3>> positions;
            times.reserve(static_cast<std::size_t>(rows.shape(0)));
            positions.reserve(static_cast<std::size_t>(rows.shape(0)));
            for (py::ssize_t i = 0; i < rows.shape(0); ++i) {
                const double time = rows(i, 0);
                const double ra = rows(i, 1) * M_PI / 180.0;
                const double dec = rows(i, 2) * M_PI / 180.0;
                const double distance = rows(i, 3);
                if (!std::isfinite(time) || !std::isfinite(ra) || !std::isfinite(dec) ||
                    !std::isfinite(distance) || (!times.empty() && time <= times.back())) {
                    throw std::invalid_argument(
                        "space Site table must contain finite, strictly increasing times");
                }
                // VBMicrolensing satellite tables express RA/Dec in its
                // ecliptic-reference basis. Convert with VBM's J2000
                // Eq2000/Quad2000/North2000 vectors.
                constexpr double cos_obliquity = 0.9174820003578725;
                constexpr double sin_obliquity = 0.3977772982704228;
                const double cos_dec = std::cos(dec);
                const double along_equator = distance * cos_dec * std::cos(ra);
                const double along_quad = distance * cos_dec * std::sin(ra);
                const double along_north = distance * std::sin(dec);
                times.push_back(time);
                positions.push_back({
                    along_equator,
                    along_quad * cos_obliquity + along_north * sin_obliquity,
                    -along_quad * sin_obliquity + along_north * cos_obliquity,
                });
            }
            return std::make_shared<Site>(std::move(times), std::move(positions));
        }),
            R"(Observation site.

Site('ground', lat, lon) creates a ground site in degrees (North/East).
Site('space', table) creates a space site; table columns are [JD, RA_deg, Dec_deg, distance_AU].
The space table follows VBMicrolensing's geocentric satellite convention.)")
        .def_property_readonly("kind", [](const Site& s) {
            return s.kind() == SiteKind::space ? "space" : "ground";
        })
        .def_property_readonly("has_position", [](const Site& s) {
            return s.kind() == SiteKind::space ? s.has_space_ephemeris() : s.has_ground_position();
        })
        .def_property_readonly("lat_deg", &Site::lat_deg)
        .def_property_readonly("lon_deg", &Site::lon_deg)
        .def("__repr__", [](const Site& s) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "<Site lat=%.4f lon=%.4f [deg]>",
                          s.lat_deg(), s.lon_deg());
            return std::string(buf);
        });

}
