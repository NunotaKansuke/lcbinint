#include "bind_lc.hpp"
#include "lcbinint/lcbinint.h"
#include "lcbinint/lc/light_curve.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

namespace {

struct PyLimbDarkening {
    double c = 0.0;
    double d = 0.0;
};

} // namespace

void register_lc_submodule(py::module_& parent)
{
    auto lc = parent.def_submodule("lc", "Light-curve magnification evaluation");

    // --- Enums ---
    py::enum_<lcbi_orbital_motion_mode>(lc, "OrbitalMotionMode")
        .value("STATIC",   LCBI_ORBIT_STATIC)
        .value("CIRCULAR", LCBI_ORBIT_CIRCULAR)
        .value("KEPLER",   LCBI_ORBIT_KEPLER)
        .export_values();

    py::enum_<lcbi_xallarap_param_type>(lc, "XallarapParamType")
        .value("NONE",             LCBI_XALLARAP_NONE)
        .value("ANGULAR_VELOCITY", LCBI_XALLARAP_ANGULAR_VELOCITY)
        .value("ORBITAL_ELEMENTS", LCBI_XALLARAP_ORBITAL_ELEMENTS)
        .export_values();

    // --- LimbDarkening ---
    py::class_<PyLimbDarkening>(lc, "LimbDarkening")
        .def(py::init([](double c, double d) { return PyLimbDarkening{c, d}; }),
            py::arg("c") = 0.0, py::arg("d") = 0.0)
        .def_readwrite("c", &PyLimbDarkening::c)
        .def_readwrite("d", &PyLimbDarkening::d)
        .def_static("linear", [](double c) { return PyLimbDarkening{c, 0.0}; },
            py::arg("c"))
        .def_static("square_root", [](double c, double d) { return PyLimbDarkening{c, d}; },
            py::arg("c"), py::arg("d"))
        .def("__repr__", [](const PyLimbDarkening& ld) {
            return "LimbDarkening(c=" + std::to_string(ld.c)
                + ", d=" + std::to_string(ld.d) + ")";
        });

    // --- Options: lcbi_options exposed directly ---
    py::class_<lcbi_options>(lc, "Options")
        .def(py::init([]() { return lcbi_default_options(); }))
        .def(py::init([](
                int    source_bins,
                int    caustic_bins,
                double grid_ratio,
                int    mode,
                bool   vbm_compatible,
                double adaptive_hex_threshold,
                bool   adaptive_source_bins,
                int    max_source_bins,
                double finite_source_tol,
                double finite_source_reltol,
                double point_source_threshold,
                int    polar_source_bins,
                double polar_grid_ratio,
                lcbi_xallarap_param_type xallarap_param_type,
                int    parallax_mode) {
            auto o = lcbi_default_options();
            o.source_bins            = source_bins;
            o.caustic_bins           = caustic_bins;
            o.grid_ratio             = grid_ratio;
            o.mode                   = mode;
            o.vbm_compatible         = vbm_compatible ? 1 : 0;
            o.adaptive_hex_threshold = adaptive_hex_threshold;
            o.adaptive_source_bins   = adaptive_source_bins ? 1 : 0;
            o.max_source_bins        = max_source_bins;
            o.finite_source_tol      = finite_source_tol;
            o.finite_source_reltol   = finite_source_reltol;
            o.point_source_threshold = point_source_threshold;
            o.polar_source_bins      = polar_source_bins;
            o.polar_grid_ratio       = polar_grid_ratio;
            o.xallarap_param_type    = xallarap_param_type;
            o.parallax_mode          = parallax_mode;
            return o;
        }),
            py::arg("source_bins")            = lcbi_default_options().source_bins,
            py::arg("caustic_bins")           = lcbi_default_options().caustic_bins,
            py::arg("grid_ratio")             = lcbi_default_options().grid_ratio,
            py::arg("mode")                   = lcbi_default_options().mode,
            py::arg("vbm_compatible")         = false,
            py::arg("adaptive_hex_threshold") = lcbi_default_options().adaptive_hex_threshold,
            py::arg("adaptive_source_bins")   = false,
            py::arg("max_source_bins")        = lcbi_default_options().max_source_bins,
            py::arg("finite_source_tol")      = lcbi_default_options().finite_source_tol,
            py::arg("finite_source_reltol")   = lcbi_default_options().finite_source_reltol,
            py::arg("point_source_threshold") = lcbi_default_options().point_source_threshold,
            py::arg("polar_source_bins")      = 0,
            py::arg("polar_grid_ratio")       = 0.0,
            py::arg("xallarap_param_type")    = LCBI_XALLARAP_NONE,
            py::arg("parallax_mode")          = 0)
        .def_readwrite("source_bins",            &lcbi_options::source_bins)
        .def_readwrite("caustic_bins",           &lcbi_options::caustic_bins)
        .def_readwrite("grid_ratio",             &lcbi_options::grid_ratio)
        .def_readwrite("mode",                   &lcbi_options::mode)
        .def_property("vbm_compatible",
            [](const lcbi_options& o) { return o.vbm_compatible != 0; },
            [](lcbi_options& o, bool v) { o.vbm_compatible = v ? 1 : 0; })
        .def_readwrite("adaptive_hex_threshold", &lcbi_options::adaptive_hex_threshold)
        .def_property("adaptive_source_bins",
            [](const lcbi_options& o) { return o.adaptive_source_bins != 0; },
            [](lcbi_options& o, bool v) { o.adaptive_source_bins = v ? 1 : 0; })
        .def_readwrite("max_source_bins",        &lcbi_options::max_source_bins)
        .def_readwrite("finite_source_tol",      &lcbi_options::finite_source_tol)
        .def_readwrite("finite_source_reltol",   &lcbi_options::finite_source_reltol)
        .def_readwrite("point_source_threshold", &lcbi_options::point_source_threshold)
        .def_readwrite("polar_source_bins",      &lcbi_options::polar_source_bins)
        .def_readwrite("polar_grid_ratio",       &lcbi_options::polar_grid_ratio)
        .def_readwrite("parallax_mode",          &lcbi_options::parallax_mode)
        .def_readwrite("xallarap_param_type",    &lcbi_options::xallarap_param_type)
        .def("__repr__", [](const lcbi_options& o) {
            return "<lc.Options source_bins=" + std::to_string(o.source_bins)
                + " vbm_compatible=" + (o.vbm_compatible ? "True" : "False") + ">";
        });

    // --- Parameters: lcbi_params exposed directly (no wrapper class) ---
    // Python-friendly aliases: u0=umin, alpha=theta, s=sep
    py::class_<lcbi_params>(lc, "Parameters")
        .def(py::init([]() { return lcbi_default_params(); }))
        .def(py::init([](
                double t0, double tE, double u0, double alpha,
                double s, double q, double rho,
                double piEN, double piEE,
                double q2, double sep2, double ang,
                double g1, double g2, double g3,
                double lom_szs, double lom_ar) {
            auto p = lcbi_default_params();
            p.t0 = t0; p.tE = tE; p.umin = u0; p.theta = alpha;
            p.sep = s; p.q = q; p.rho = rho;
            p.piEN = piEN; p.piEE = piEE;
            p.q2 = q2; p.sep2 = sep2; p.ang = ang;
            p.g1 = g1; p.g2 = g2; p.g3 = g3;
            p.lom_szs = lom_szs; p.lom_ar = lom_ar;
            return p;
        }),
            py::arg("t0")    = 0.0,  py::arg("tE")    = 1.0,
            py::arg("u0")    = 0.0,  py::arg("alpha")  = 0.0,
            py::arg("s")     = 1.0,  py::arg("q")      = 1.0,
            py::arg("rho")   = 0.0,
            py::arg("piEN")  = 0.0,  py::arg("piEE")   = 0.0,
            py::arg("q2")    = 0.0,  py::arg("sep2")   = 0.0, py::arg("ang")    = 0.0,
            py::arg("g1")    = 0.0,  py::arg("g2")     = 0.0, py::arg("g3")     = 0.0,
            py::arg("lom_szs") = 0.0, py::arg("lom_ar") = 1.0)
        // Core (aliases to friendly names)
        .def_property("t0",    [](const lcbi_params& p){ return p.t0; },    [](lcbi_params& p, double v){ p.t0 = v; })
        .def_property("tE",    [](const lcbi_params& p){ return p.tE; },    [](lcbi_params& p, double v){ p.tE = v; })
        .def_property("u0",    [](const lcbi_params& p){ return p.umin; },  [](lcbi_params& p, double v){ p.umin = v; })
        .def_property("alpha", [](const lcbi_params& p){ return p.theta; }, [](lcbi_params& p, double v){ p.theta = v; })
        .def_property("s",     [](const lcbi_params& p){ return p.sep; },   [](lcbi_params& p, double v){ p.sep = v; })
        .def_readwrite("q",    &lcbi_params::q)
        .def_readwrite("rho",  &lcbi_params::rho)
        // Triple
        .def_readwrite("q2",   &lcbi_params::q2)
        .def_readwrite("sep2", &lcbi_params::sep2)
        .def_readwrite("ang",  &lcbi_params::ang)
        // Parallax
        .def_readwrite("piEN",    &lcbi_params::piEN)
        .def_readwrite("piEE",    &lcbi_params::piEE)
        .def_readwrite("ra",      &lcbi_params::ra)
        .def_readwrite("dec",     &lcbi_params::dec)
        .def_readwrite("tfix",    &lcbi_params::tfix)
        .def_readwrite("obs_lat", &lcbi_params::obs_lat)
        .def_readwrite("obs_lon", &lcbi_params::obs_lon)
        // Orbital motion
        .def_readwrite("orbital_motion_mode", &lcbi_params::orbital_motion_mode)
        .def_readwrite("g1",      &lcbi_params::g1)
        .def_readwrite("g2",      &lcbi_params::g2)
        .def_readwrite("g3",      &lcbi_params::g3)
        .def_readwrite("lom_szs", &lcbi_params::lom_szs)
        .def_readwrite("lom_ar",  &lcbi_params::lom_ar)
        .def_readwrite("v_sep",   &lcbi_params::v_sep)
        // Xallarap
        .def_readwrite("xi_1",      &lcbi_params::xi_1)
        .def_readwrite("xi_2",      &lcbi_params::xi_2)
        .def_readwrite("omega_xa",  &lcbi_params::omega_xa)
        .def_readwrite("inc_xa",    &lcbi_params::inc_xa)
        .def_readwrite("phi_xa",    &lcbi_params::phi_xa)
        .def_readwrite("piEN_xa",   &lcbi_params::piEN_xa)
        .def_readwrite("piEE_xa",   &lcbi_params::piEE_xa)
        .def_readwrite("period_xa", &lcbi_params::period_xa)
        .def_readwrite("ecc_xa",    &lcbi_params::ecc_xa)
        .def_readwrite("peri_xa",   &lcbi_params::peri_xa)
        // Limb darkening
        .def_readwrite("limb_darkening_c", &lcbi_params::limb_darkening_c)
        .def_readwrite("limb_darkening_d", &lcbi_params::limb_darkening_d)
        .def("__repr__", [](const lcbi_params& p) {
            return "<lc.Parameters t0=" + std::to_string(p.t0)
                + " tE=" + std::to_string(p.tE)
                + " u0=" + std::to_string(p.umin)
                + " s=" + std::to_string(p.sep)
                + " q=" + std::to_string(p.q) + ">";
        });

    // --- LightCurve ---
    py::class_<lcbinint::lc::LightCurve, std::shared_ptr<lcbinint::lc::LightCurve>>(lc, "LightCurve")
        .def(py::init([](const lcbi_options& opts, const PyLimbDarkening& ld) {
            return std::make_shared<lcbinint::lc::LightCurve>(opts, ld.c, ld.d);
        }),
            py::arg("options")        = lcbi_default_options(),
            py::arg("limb_darkening") = PyLimbDarkening{})
        .def_property_readonly("options", &lcbinint::lc::LightCurve::options)
        .def_property_readonly("ld_c",    &lcbinint::lc::LightCurve::ld_c)
        .def_property_readonly("ld_d",    &lcbinint::lc::LightCurve::ld_d)
        .def("__call__",
            [](const lcbinint::lc::LightCurve& lc,
               py::array_t<double> times,
               const lcbi_params& params) -> py::array_t<double> {
                auto buf = times.request();
                const double* ptr = static_cast<const double*>(buf.ptr);
                std::vector<double> tv(ptr, ptr + buf.size);
                py::gil_scoped_release release;
                auto mags = lc.magnification(tv, params);
                py::gil_scoped_acquire acquire;
                return py::array_t<double>(mags.size(), mags.data());
            },
            py::arg("times"), py::arg("params"))
        .def("magnification",
            [](const lcbinint::lc::LightCurve& lc,
               py::array_t<double> times,
               const lcbi_params& params) -> py::array_t<double> {
                auto buf = times.request();
                const double* ptr = static_cast<const double*>(buf.ptr);
                std::vector<double> tv(ptr, ptr + buf.size);
                py::gil_scoped_release release;
                auto mags = lc.magnification(tv, params);
                py::gil_scoped_acquire acquire;
                return py::array_t<double>(mags.size(), mags.data());
            },
            py::arg("times"), py::arg("params"))
        .def("__repr__", [](const lcbinint::lc::LightCurve& lc) {
            return "<lc.LightCurve source_bins="
                + std::to_string(lc.options().source_bins) + ">";
        });
}
