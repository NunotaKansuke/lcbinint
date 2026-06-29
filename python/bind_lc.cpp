#include "bind_lc.hpp"
#include "lcbinint/lcbinint.h"
#include "lcbinint/lc/light_curve.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

struct PyLimbDarkening {
    double c = 0.0;
    double d = 0.0;
};

// Map param_type string → vbm_compatible + center_of_mass (same logic as old API).
void apply_param_type(lcbi_options& o, const std::string& pt)
{
    if (pt == "auto" || pt == "vbm" || pt == "vbbl" || pt == "standard") {
        o.vbm_compatible = 1; o.center_of_mass = 0;
    } else if (pt == "lcbinint" || pt == "original" || pt == "legacy") {
        o.vbm_compatible = 0; o.center_of_mass = 0;
    } else if (pt == "center_of_mass") {
        o.vbm_compatible = 0; o.center_of_mass = 1;
    } else if (pt == "vbm_center_of_mass") {
        o.vbm_compatible = 1; o.center_of_mass = 1;
    } else {
        throw std::invalid_argument(
            "param_type must be 'vbm', 'lcbinint', 'center_of_mass', or 'vbm_center_of_mass'");
    }
}

// Build lcbi_params from a Python dict (or py::kwargs).
// Supports both canonical names (umin, theta, sep) and friendly aliases (u0, alpha, s).
lcbi_params params_from_dict(const py::dict& d)
{
    auto p = lcbi_default_params();
    for (auto& item : d) {
        const std::string key = item.first.cast<std::string>();
        // Most params are double; handle int-typed ones below if needed
        const double val = item.second.cast<double>();
        if      (key == "t0"   || key == "t_0")  p.t0    = val;
        else if (key == "tE"   || key == "t_E")  p.tE    = val;
        else if (key == "u0"   || key == "umin")  p.umin  = val;
        else if (key == "alpha"|| key == "theta") p.theta = val;
        else if (key == "s"    || key == "sep")   p.sep   = val;
        else if (key == "q")                       p.q     = val;
        else if (key == "rho")                     p.rho   = val;
        else if (key == "piEN")                    p.piEN  = val;
        else if (key == "piEE")                    p.piEE  = val;
        else if (key == "q2")                      p.q2    = val;
        else if (key == "sep2")                    p.sep2  = val;
        else if (key == "ang")                     p.ang   = val;
        else if (key == "ra")                      p.ra    = val;
        else if (key == "dec")                     p.dec   = val;
        else if (key == "tfix")                    p.tfix  = val;
        else if (key == "obs_lat")                 p.obs_lat = val;
        else if (key == "obs_lon")                 p.obs_lon = val;
        else if (key == "limb_darkening_c")        p.limb_darkening_c = val;
        else if (key == "limb_darkening_d")        p.limb_darkening_d = val;
        else if (key == "g1")                      p.g1    = val;
        else if (key == "g2")                      p.g2    = val;
        else if (key == "g3")                      p.g3    = val;
        else if (key == "xi_1")                    p.xi_1  = val;
        else if (key == "xi_2")                    p.xi_2  = val;
        else if (key == "omega_xa")                p.omega_xa = val;
        else if (key == "inc_xa")                  p.inc_xa = val;
        else if (key == "phi_xa")                  p.phi_xa = val;
        else {
            throw py::key_error("lcbinint.lc: unknown parameter '" + key + "'");
        }
    }
    return p;
}

// Return a numpy array from a vector, transferring ownership via capsule.
py::array_t<double> vec_to_numpy(std::vector<double> mags)
{
    auto* heap = new std::vector<double>(std::move(mags));
    py::capsule cap(heap, [](void* p) {
        delete static_cast<std::vector<double>*>(p);
    });
    return py::array_t<double>(
        {static_cast<py::ssize_t>(heap->size())},
        {sizeof(double)},
        heap->data(),
        cap);
}

// Core dispatch: build times vector, release GIL, compute, return numpy array.
py::array_t<double> compute(
    const lcbinint::lc::LightCurve& lc,
    py::array_t<double>             times,
    const lcbi_params&              params)
{
    auto buf = times.request();
    const double* ptr = static_cast<const double*>(buf.ptr);
    std::vector<double> tv(ptr, ptr + buf.size);
    std::vector<double> mags;
    {
        py::gil_scoped_release release;
        mags = lc.magnification(tv, params);
    }
    return vec_to_numpy(std::move(mags));
}

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
        .def_static("linear",      [](double c) { return PyLimbDarkening{c, 0.0}; }, py::arg("c"))
        .def_static("square_root", [](double c, double d) { return PyLimbDarkening{c, d}; },
            py::arg("c"), py::arg("d"))
        .def("__repr__", [](const PyLimbDarkening& ld) {
            return "LimbDarkening(c=" + std::to_string(ld.c)
                + ", d=" + std::to_string(ld.d) + ")";
        });

    // --- Options: lcbi_options exposed directly (for power users / bayes module) ---
    py::class_<lcbi_options>(lc, "Options")
        .def(py::init([]() { return lcbi_default_options(); }))
        .def(py::init([](
                std::string param_type,
                int    source_bins,
                int    caustic_bins,
                double grid_ratio,
                int    mode,
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
            apply_param_type(o, param_type);
            o.source_bins            = source_bins;
            o.caustic_bins           = caustic_bins;
            o.grid_ratio             = grid_ratio;
            o.mode                   = mode;
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
            py::arg("param_type")             = "vbm",
            py::arg("source_bins")            = lcbi_default_options().source_bins,
            py::arg("caustic_bins")           = lcbi_default_options().caustic_bins,
            py::arg("grid_ratio")             = lcbi_default_options().grid_ratio,
            py::arg("mode")                   = lcbi_default_options().mode,
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
        .def_property("param_type",
            [](const lcbi_options& o) -> std::string {
                if (o.vbm_compatible != 0 && o.center_of_mass == 0) return "vbm";
                if (o.vbm_compatible != 0 && o.center_of_mass != 0) return "vbm_center_of_mass";
                if (o.vbm_compatible == 0 && o.center_of_mass != 0) return "center_of_mass";
                return "lcbinint";
            },
            [](lcbi_options& o, const std::string& pt) { apply_param_type(o, pt); })
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
            std::string pt;
            if (o.vbm_compatible != 0 && o.center_of_mass == 0) pt = "vbm";
            else if (o.vbm_compatible != 0)                       pt = "vbm_center_of_mass";
            else if (o.center_of_mass != 0)                       pt = "center_of_mass";
            else                                                   pt = "lcbinint";
            return "<lc.Options param_type='" + pt
                + "' source_bins=" + std::to_string(o.source_bins) + ">";
        });

    // --- Parameters: lcbi_params exposed directly (for power users / bayes module) ---
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
        .def_property("t0",    [](const lcbi_params& p){ return p.t0; },    [](lcbi_params& p, double v){ p.t0 = v; })
        .def_property("tE",    [](const lcbi_params& p){ return p.tE; },    [](lcbi_params& p, double v){ p.tE = v; })
        .def_property("u0",    [](const lcbi_params& p){ return p.umin; },  [](lcbi_params& p, double v){ p.umin = v; })
        .def_property("alpha", [](const lcbi_params& p){ return p.theta; }, [](lcbi_params& p, double v){ p.theta = v; })
        .def_property("s",     [](const lcbi_params& p){ return p.sep; },   [](lcbi_params& p, double v){ p.sep = v; })
        .def_readwrite("q",    &lcbi_params::q)
        .def_readwrite("rho",  &lcbi_params::rho)
        .def_readwrite("q2",   &lcbi_params::q2)
        .def_readwrite("sep2", &lcbi_params::sep2)
        .def_readwrite("ang",  &lcbi_params::ang)
        .def_readwrite("piEN",    &lcbi_params::piEN)
        .def_readwrite("piEE",    &lcbi_params::piEE)
        .def_readwrite("ra",      &lcbi_params::ra)
        .def_readwrite("dec",     &lcbi_params::dec)
        .def_readwrite("tfix",    &lcbi_params::tfix)
        .def_readwrite("obs_lat", &lcbi_params::obs_lat)
        .def_readwrite("obs_lon", &lcbi_params::obs_lon)
        .def_readwrite("orbital_motion_mode", &lcbi_params::orbital_motion_mode)
        .def_readwrite("g1",      &lcbi_params::g1)
        .def_readwrite("g2",      &lcbi_params::g2)
        .def_readwrite("g3",      &lcbi_params::g3)
        .def_readwrite("lom_szs", &lcbi_params::lom_szs)
        .def_readwrite("lom_ar",  &lcbi_params::lom_ar)
        .def_readwrite("v_sep",   &lcbi_params::v_sep)
        .def_readwrite("xi_1",      &lcbi_params::xi_1)
        .def_readwrite("xi_2",      &lcbi_params::xi_2)
        .def_readwrite("omega_xa",  &lcbi_params::omega_xa)
        .def_readwrite("inc_xa",    &lcbi_params::inc_xa)
        .def_readwrite("phi_xa",    &lcbi_params::phi_xa)
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
    // Optimized for magnification-only use: construct once, call many times.
    // Accepts params as: lcbi_params object, dict, or **kwargs.
    using LC = lcbinint::lc::LightCurve;

    py::class_<LC, std::shared_ptr<LC>>(lc, "LightCurve")
        // Constructor 1: explicit lc.Options object (for bayes module / power users)
        .def(py::init([](const lcbi_options& opts, const PyLimbDarkening& ld) {
            return std::make_shared<LC>(opts, ld.c, ld.d);
        }),
            py::arg("options")        = lcbi_default_options(),
            py::arg("limb_darkening") = PyLimbDarkening{})
        // Constructor 2: kwargs directly (VBM style)
        .def(py::init([](py::kwargs kw) {
            auto o = lcbi_default_options();
            PyLimbDarkening ld{};
            for (auto& item : kw) {
                const std::string key = item.first.cast<std::string>();
                if      (key == "source_bins")            o.source_bins            = item.second.cast<int>();
                else if (key == "caustic_bins")           o.caustic_bins           = item.second.cast<int>();
                else if (key == "param_type")             apply_param_type(o, item.second.cast<std::string>());
                else if (key == "adaptive_hex_threshold") o.adaptive_hex_threshold = item.second.cast<double>();
                else if (key == "parallax_mode")          o.parallax_mode          = item.second.cast<int>();
                else if (key == "mode")                   o.mode                   = item.second.cast<int>();
                else if (key == "grid_ratio")             o.grid_ratio             = item.second.cast<double>();
                else if (key == "finite_source_tol")      o.finite_source_tol      = item.second.cast<double>();
                else if (key == "finite_source_reltol")   o.finite_source_reltol   = item.second.cast<double>();
                else if (key == "point_source_threshold") o.point_source_threshold = item.second.cast<double>();
                else if (key == "adaptive_source_bins")   o.adaptive_source_bins   = item.second.cast<bool>() ? 1 : 0;
                else if (key == "max_source_bins")        o.max_source_bins        = item.second.cast<int>();
                else if (key == "ld_c" || key == "limb_darkening_c") ld.c = item.second.cast<double>();
                else if (key == "ld_d" || key == "limb_darkening_d") ld.d = item.second.cast<double>();
                else throw py::key_error("LightCurve: unknown option '" + key + "'");
            }
            return std::make_shared<LC>(o, ld.c, ld.d);
        }))
        .def_property_readonly("options", &LC::options)
        .def_property_readonly("ld_c",    &LC::ld_c)
        .def_property_readonly("ld_d",    &LC::ld_d)

        // __call__ overload 1: lcbi_params object (zero overhead fast path)
        .def("__call__",
            [](const LC& lc, py::array_t<double> times, const lcbi_params& params) {
                return compute(lc, times, params);
            },
            py::arg("times"), py::arg("params"))

        // __call__ overload 2: dict  e.g. lc_obj(times, {"t0": 9000, "tE": 30, ...})
        .def("__call__",
            [](const LC& lc, py::array_t<double> times, py::dict d) {
                return compute(lc, times, params_from_dict(d));
            },
            py::arg("times"), py::arg("params"))

        // __call__ overload 3: **kwargs  e.g. lc_obj(times, t0=9000, tE=30, ...)
        .def("__call__",
            [](const LC& lc, py::array_t<double> times, py::kwargs kw) {
                return compute(lc, times, params_from_dict(kw));
            },
            py::arg("times"))

        // .magnification() as alias (same overloads)
        .def("magnification",
            [](const LC& lc, py::array_t<double> times, const lcbi_params& params) {
                return compute(lc, times, params);
            },
            py::arg("times"), py::arg("params"))
        .def("magnification",
            [](const LC& lc, py::array_t<double> times, py::dict d) {
                return compute(lc, times, params_from_dict(d));
            },
            py::arg("times"), py::arg("params"))
        .def("magnification",
            [](const LC& lc, py::array_t<double> times, py::kwargs kw) {
                return compute(lc, times, params_from_dict(kw));
            },
            py::arg("times"))

        .def("__repr__", [](const LC& lc) {
            return "<lc.LightCurve source_bins="
                + std::to_string(lc.options().source_bins)
                + " vbm_compatible="
                + (lc.options().vbm_compatible ? "True" : "False") + ">";
        });
}
