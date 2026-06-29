#include "bind_lc.hpp"
#include "lcbinint/lcbinint.h"
#include "lcbinint/lc/parameters.hpp"
#include "lcbinint/lc/light_curve.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;
using namespace lcbinint::lc;

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
        .def(py::init([](double c, double d) {
            return PyLimbDarkening{c, d};
        }), py::arg("c") = 0.0, py::arg("d") = 0.0)
        .def_readwrite("c", &PyLimbDarkening::c)
        .def_readwrite("d", &PyLimbDarkening::d)
        .def_static("linear", [](double c) {
            return PyLimbDarkening{c, 0.0};
        }, py::arg("c"), "Linear limb darkening: u = c")
        .def_static("square_root", [](double c, double d) {
            return PyLimbDarkening{c, d};
        }, py::arg("c"), py::arg("d"), "Square-root limb darkening: u = c + d*sqrt(1-mu)")
        .def("__repr__", [](const PyLimbDarkening& ld) {
            return "LimbDarkening(c=" + std::to_string(ld.c)
                + ", d=" + std::to_string(ld.d) + ")";
        });

    // --- Options ---
    py::class_<lcbi_options>(lc, "Options")
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
            o.source_bins              = source_bins;
            o.caustic_bins             = caustic_bins;
            o.grid_ratio               = grid_ratio;
            o.mode                     = mode;
            o.vbm_compatible           = vbm_compatible ? 1 : 0;
            o.adaptive_hex_threshold   = adaptive_hex_threshold;
            o.adaptive_source_bins     = adaptive_source_bins ? 1 : 0;
            o.max_source_bins          = max_source_bins;
            o.finite_source_tol        = finite_source_tol;
            o.finite_source_reltol     = finite_source_reltol;
            o.point_source_threshold   = point_source_threshold;
            o.polar_source_bins        = polar_source_bins;
            o.polar_grid_ratio         = polar_grid_ratio;
            o.xallarap_param_type      = xallarap_param_type;
            o.parallax_mode            = parallax_mode;
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

    // --- Parameters ---
    py::class_<Parameters>(lc, "Parameters")
        .def(py::init([](
                double t0, double tE, double u0, double alpha,
                double s, double q, double rho,
                double piEN, double piEE,
                double q2, double sep2, double ang,
                double g1, double g2, double g3,
                double lom_szs, double lom_ar) {
            Parameters p;
            p.set_t0(t0);  p.set_tE(tE);  p.set_u0(u0);  p.set_alpha(alpha);
            p.set_s(s);    p.set_q(q);    p.set_rho(rho);
            p.set_piEN(piEN); p.set_piEE(piEE);
            p.set_q2(q2);  p.set_sep2(sep2); p.set_ang(ang);
            p.set_g1(g1);  p.set_g2(g2);  p.set_g3(g3);
            p.set_lom_szs(lom_szs); p.set_lom_ar(lom_ar);
            return p;
        }),
            py::arg("t0")    = 0.0, py::arg("tE")    = 1.0,
            py::arg("u0")    = 0.0, py::arg("alpha")  = 0.0,
            py::arg("s")     = 1.0, py::arg("q")      = 1.0,
            py::arg("rho")   = 0.0,
            py::arg("piEN")  = 0.0, py::arg("piEE")   = 0.0,
            py::arg("q2")    = 0.0, py::arg("sep2")   = 0.0, py::arg("ang")  = 0.0,
            py::arg("g1")    = 0.0, py::arg("g2")     = 0.0, py::arg("g3")   = 0.0,
            py::arg("lom_szs") = 0.0, py::arg("lom_ar") = 1.0)
        // Core
        .def_property("t0",    &Parameters::t0,    &Parameters::set_t0)
        .def_property("tE",    &Parameters::tE,    &Parameters::set_tE)
        .def_property("u0",    &Parameters::u0,    &Parameters::set_u0)
        .def_property("alpha", &Parameters::alpha, &Parameters::set_alpha)
        .def_property("s",     &Parameters::s,     &Parameters::set_s)
        .def_property("q",     &Parameters::q,     &Parameters::set_q)
        .def_property("rho",   &Parameters::rho,   &Parameters::set_rho)
        // Triple
        .def_property("q2",   &Parameters::q2,   &Parameters::set_q2)
        .def_property("sep2", &Parameters::sep2, &Parameters::set_sep2)
        .def_property("ang",  &Parameters::ang,  &Parameters::set_ang)
        // Parallax
        .def_property("piEN", &Parameters::piEN, &Parameters::set_piEN)
        .def_property("piEE", &Parameters::piEE, &Parameters::set_piEE)
        .def_property("ra",   &Parameters::ra,   &Parameters::set_ra)
        .def_property("dec",  &Parameters::dec,  &Parameters::set_dec)
        .def_property("tfix", &Parameters::tfix, &Parameters::set_tfix)
        // Terrestrial parallax
        .def_property("obs_lat", &Parameters::obs_lat, &Parameters::set_obs_lat)
        .def_property("obs_lon", &Parameters::obs_lon, &Parameters::set_obs_lon)
        // Orbital motion
        .def_property("orbital_motion_mode",
            &Parameters::orbital_motion_mode, &Parameters::set_orbital_motion_mode)
        .def_property("g1",      &Parameters::g1,      &Parameters::set_g1)
        .def_property("g2",      &Parameters::g2,      &Parameters::set_g2)
        .def_property("g3",      &Parameters::g3,      &Parameters::set_g3)
        .def_property("lom_szs", &Parameters::lom_szs, &Parameters::set_lom_szs)
        .def_property("lom_ar",  &Parameters::lom_ar,  &Parameters::set_lom_ar)
        .def_property("v_sep",   &Parameters::v_sep,   &Parameters::set_v_sep)
        // Xallarap
        .def_property("xi_1",     &Parameters::xi_1,     &Parameters::set_xi_1)
        .def_property("xi_2",     &Parameters::xi_2,     &Parameters::set_xi_2)
        .def_property("omega_xa", &Parameters::omega_xa, &Parameters::set_omega_xa)
        .def_property("inc_xa",   &Parameters::inc_xa,   &Parameters::set_inc_xa)
        .def_property("phi_xa",   &Parameters::phi_xa,   &Parameters::set_phi_xa)
        .def_property("piEN_xa",  &Parameters::piEN_xa,  &Parameters::set_piEN_xa)
        .def_property("piEE_xa",  &Parameters::piEE_xa,  &Parameters::set_piEE_xa)
        .def_property("period_xa",&Parameters::period_xa,&Parameters::set_period_xa)
        .def_property("ecc_xa",   &Parameters::ecc_xa,   &Parameters::set_ecc_xa)
        .def_property("peri_xa",  &Parameters::peri_xa,  &Parameters::set_peri_xa)
        // Limb darkening (per-params override)
        .def_property("limb_darkening_c",
            &Parameters::limb_darkening_c, &Parameters::set_limb_darkening_c)
        .def_property("limb_darkening_d",
            &Parameters::limb_darkening_d, &Parameters::set_limb_darkening_d)
        .def("__repr__", [](const Parameters& p) {
            return "<lc.Parameters t0=" + std::to_string(p.t0())
                + " tE=" + std::to_string(p.tE())
                + " u0=" + std::to_string(p.u0())
                + " s=" + std::to_string(p.s())
                + " q=" + std::to_string(p.q()) + ">";
        });

    // --- LightCurve ---
    py::class_<LightCurve, std::shared_ptr<LightCurve>>(lc, "LightCurve")
        .def(py::init([](const lcbi_options& opts, const PyLimbDarkening& ld) {
            return std::make_shared<LightCurve>(opts, ld.c, ld.d);
        }),
            py::arg("options")         = lcbi_default_options(),
            py::arg("limb_darkening")  = PyLimbDarkening{})
        .def_property_readonly("options", &LightCurve::options)
        .def_property_readonly("ld_c",    &LightCurve::ld_c)
        .def_property_readonly("ld_d",    &LightCurve::ld_d)
        .def("__call__",
            [](const LightCurve& lc,
               py::array_t<double> times,
               const Parameters& params) -> py::array_t<double> {
                auto buf = times.request();
                const double* ptr = static_cast<const double*>(buf.ptr);
                std::vector<double> tv(ptr, ptr + buf.size);
                py::gil_scoped_release release;
                auto mags = lc.magnification(tv, params);
                py::gil_scoped_acquire acquire;
                return py::array_t<double>(mags.size(), mags.data());
            },
            py::arg("times"), py::arg("params"),
            "Compute magnification array. Returns np.ndarray of shape (N,).")
        .def("magnification",
            [](const LightCurve& lc,
               py::array_t<double> times,
               const Parameters& params) -> py::array_t<double> {
                auto buf = times.request();
                const double* ptr = static_cast<const double*>(buf.ptr);
                std::vector<double> tv(ptr, ptr + buf.size);
                py::gil_scoped_release release;
                auto mags = lc.magnification(tv, params);
                py::gil_scoped_acquire acquire;
                return py::array_t<double>(mags.size(), mags.data());
            },
            py::arg("times"), py::arg("params"))
        .def("__repr__", [](const LightCurve& lc) {
            return "<lc.LightCurve source_bins="
                + std::to_string(lc.options().source_bins) + ">";
        });
}
