#include "bind_lc.hpp"
#include "lcbinint/lcbinint.h"
#include "lcbinint/lc/effects.hpp"
#include "lcbinint/lc/light_curve.hpp"
#include "lcbinint/obs/coordinates.hpp"
#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/model/trajectory.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

namespace py = pybind11;

struct PyLimbDarkening {
    double c = 0.0;
    double d = 0.0;
};

namespace {

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
        else if (key == "lom_szs")                 p.lom_szs = val;
        else if (key == "lom_ar")                  p.lom_ar  = val;
        // xallarap amplitude/position (all modes)
        else if (key == "xi_1")                    p.xi_1  = val;
        else if (key == "xi_2")                    p.xi_2  = val;
        // orbital_elements / circular_elements: period-based orbit params
        else if (key == "period_xa")               p.period_xa = val;
        else if (key == "ecc_xa")                  p.ecc_xa    = val;
        else if (key == "peri_xa")                 p.peri_xa   = val;
        else if (key == "inc_xa")                  p.inc_xa    = val;
        // circular_velocity / kepler_velocity: w1/w2/w3 (mapped to omega/inc/phi fields)
        else if (key == "w1")                      p.omega_xa = val;
        else if (key == "w2")                      p.inc_xa   = val;
        else if (key == "w3")                      p.phi_xa   = val;
        // kepler_velocity: xa_szs/xa_ar (mapped to piEN_xa/piEE_xa fields)
        else if (key == "xa_szs")                  p.piEN_xa = val;
        else if (key == "xa_ar")                   p.piEE_xa = val;
        // Binary source params — handled by compute_dispatch, not lcbi_params.
        else if (key == "q_source" || key == "fluxratio" ||
                 key == "t0_2"     || key == "u0_2" || key == "q_mass") { /* skip */ }
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
        .value("NONE",              LCBI_XALLARAP_NONE)
        .value("ORBITAL_ELEMENTS",  LCBI_XALLARAP_ORBITAL_ELEMENTS)
        .value("CIRCULAR_ELEMENTS", LCBI_XALLARAP_CIRCULAR_ELEMENTS)
        .value("CIRCULAR_VEL",      LCBI_XALLARAP_CIRCULAR_VEL)
        .value("KEPLER_VEL",        LCBI_XALLARAP_KEPLER_VEL)
        .export_values();

    // --- Options: lcbi_options exposed directly (for power users / bayes module) ---
    py::class_<lcbi_options>(lc, "Options")
        .def(py::init([](
                std::string param_type,
                int    source_bins,
                int    caustic_bins,
                double grid_ratio,
                int    mode,
                double adaptive_hex_threshold,
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

    // --- LimbDarkening ---
    // Limb darkening profile: I(μ) = 1 - c*(1-μ) - d*(1-√μ)
    // c=0, d=0 → uniform disk (point source limit).
    py::class_<PyLimbDarkening>(lc, "LimbDarkening")
        .def(py::init<double, double>(), py::arg("c") = 0.0, py::arg("d") = 0.0)
        .def_readwrite("c", &PyLimbDarkening::c)
        .def_readwrite("d", &PyLimbDarkening::d)
        .def_static("none", []() { return PyLimbDarkening{}; },
            "Uniform source profile.")
        .def_static("linear",      [](double u) { return PyLimbDarkening{u, 0.0}; },
            py::arg("u"),
            "Linear profile: I(μ) = 1 - u*(1-μ).  c=u, d=0.")
        .def_static("square_root", [](double c, double d) { return PyLimbDarkening{c, d}; },
            py::arg("c"), py::arg("d"),
            "Square-root profile: I(μ) = 1 - c*(1-μ) - d*(1-√μ).")
        .def("__repr__", [](const PyLimbDarkening& ld) {
            return "<lc.LimbDarkening c=" + std::to_string(ld.c)
                + " d=" + std::to_string(ld.d) + ">";
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

    // --- Effects ---
    using LC       = lcbinint::lc::LightCurve;
    using Eff      = lcbinint::lc::Effects;
    using SKind    = lcbinint::lc::SourceKind;
    using SkyCoord = lcbinint::obs::SkyCoord;
    using Site     = lcbinint::obs::Site;

    // Parse "binary"/"single" source string.
    auto parse_source = [](const std::string& s) -> SKind {
        if (s == "single" || s.empty()) return SKind::single;
        if (s == "binary" || s == "binary_source") return SKind::binary;
        throw std::invalid_argument("source must be 'single' or 'binary'");
    };

    // Parse orbital motion mode string.
    auto parse_orbital = [](const std::string& s) -> lcbi_orbital_motion_mode {
        if (s == "static"  || s.empty()) return LCBI_ORBIT_STATIC;
        if (s == "circular")             return LCBI_ORBIT_CIRCULAR;
        if (s == "kepler")               return LCBI_ORBIT_KEPLER;
        throw std::invalid_argument("orbital_motion must be 'static', 'circular', or 'kepler'");
    };

    // Parse xallarap mode string.
    auto parse_xallarap = [](const std::string& s) -> lcbi_xallarap_param_type {
        if (s.empty() || s == "none")                        return LCBI_XALLARAP_NONE;
        if (s == "orbital_elements" || s == "kepler")        return LCBI_XALLARAP_ORBITAL_ELEMENTS;
        if (s == "circular_elements" || s == "circular")     return LCBI_XALLARAP_CIRCULAR_ELEMENTS;
        if (s == "circular_velocity" || s == "circular_vel") return LCBI_XALLARAP_CIRCULAR_VEL;
        if (s == "kepler_velocity"   || s == "kepler_vel")   return LCBI_XALLARAP_KEPLER_VEL;
        throw std::invalid_argument(
            "xallarap must be 'none', 'orbital_elements', 'circular_elements', "
            "'circular_velocity', or 'kepler_velocity'");
    };

    py::class_<Eff>(lc, "Effects",
        R"(Physical higher-order effect settings for a LightCurve.

Separates physics configuration (what effects are active) from numerical
Options (source_bins, grid_ratio, etc.).

terrestrial=False by default: site coordinates are stored but NOT applied
unless terrestrial is explicitly set to True.)")
        .def(py::init([&](
                const std::string& source,
                const std::string& orbital_motion,
                const std::string& xallarap,
                bool               parallax,
                bool               terrestrial,
                py::object         sky,
                py::object         site,
                py::object         t_ref) {
            Eff e;
            e.source         = parse_source(source);
            e.orbital_motion = parse_orbital(orbital_motion);
            e.xallarap       = parse_xallarap(xallarap);
            e.parallax       = parallax;
            e.terrestrial    = terrestrial;
            if (!sky.is_none())   e.sky  = sky.cast<std::shared_ptr<SkyCoord>>();
            if (!site.is_none())  e.site = site.cast<std::shared_ptr<Site>>();
            if (!t_ref.is_none()) e.t_ref = t_ref.cast<double>();
            return e;
        }),
            py::arg("source")         = "single",
            py::arg("orbital_motion") = "static",
            py::arg("xallarap")       = "none",
            py::arg("parallax")       = false,
            py::arg("terrestrial")    = false,
            py::arg("sky")            = py::none(),
            py::arg("site")           = py::none(),
            py::arg("t_ref")          = py::none())
        .def_property("source",
            [](const Eff& e) { return e.source == SKind::binary ? "binary" : "single"; },
            [&](Eff& e, const std::string& s) { e.source = parse_source(s); })
        .def_property("orbital_motion",
            [](const Eff& e) -> std::string {
                if (e.orbital_motion == LCBI_ORBIT_CIRCULAR) return "circular";
                if (e.orbital_motion == LCBI_ORBIT_KEPLER)   return "kepler";
                return "static";
            },
            [&](Eff& e, const std::string& s) { e.orbital_motion = parse_orbital(s); })
        .def_property("xallarap",
            [](const Eff& e) -> std::string {
                switch (e.xallarap) {
                case LCBI_XALLARAP_ORBITAL_ELEMENTS:  return "orbital_elements";
                case LCBI_XALLARAP_CIRCULAR_ELEMENTS: return "circular_elements";
                case LCBI_XALLARAP_CIRCULAR_VEL:      return "circular_velocity";
                case LCBI_XALLARAP_KEPLER_VEL:        return "kepler_velocity";
                default:                               return "none";
                }
            },
            [&](Eff& e, const std::string& s) { e.xallarap = parse_xallarap(s); })
        .def_readwrite("parallax",    &Eff::parallax)
        .def_readwrite("terrestrial", &Eff::terrestrial)
        .def_property("sky",
            [](const Eff& e) -> py::object {
                if (!e.sky) return py::none();
                return py::cast(e.sky);
            },
            [](Eff& e, py::object obj) {
                if (obj.is_none()) e.sky = nullptr;
                else e.sky = obj.cast<std::shared_ptr<SkyCoord>>();
            })
        .def_property("site",
            [](const Eff& e) -> py::object {
                if (!e.site) return py::none();
                return py::cast(e.site);
            },
            [](Eff& e, py::object obj) {
                if (obj.is_none()) e.site = nullptr;
                else e.site = obj.cast<std::shared_ptr<Site>>();
            })
        .def_property("t_ref",
            [](const Eff& e) -> py::object {
                if (!e.t_ref.has_value()) return py::none();
                return py::float_(*e.t_ref);
            },
            [](Eff& e, py::object obj) {
                if (obj.is_none()) e.t_ref = std::nullopt;
                else e.t_ref = obj.cast<double>();
            })
        .def("__repr__", [](const Eff& e) {
            std::string s = "<lc.Effects";
            if (e.parallax)    s += " parallax";
            if (e.terrestrial) s += " terrestrial";
            if (e.orbital_motion != LCBI_ORBIT_STATIC)
                s += e.orbital_motion == LCBI_ORBIT_CIRCULAR ? " orbital_motion=circular" : " orbital_motion=kepler";
            if (e.xallarap != LCBI_XALLARAP_NONE) {
                switch (e.xallarap) {
                case LCBI_XALLARAP_ORBITAL_ELEMENTS:  s += " xallarap=orbital_elements";  break;
                case LCBI_XALLARAP_CIRCULAR_ELEMENTS: s += " xallarap=circular_elements"; break;
                case LCBI_XALLARAP_CIRCULAR_VEL:      s += " xallarap=circular_velocity"; break;
                case LCBI_XALLARAP_KEPLER_VEL:        s += " xallarap=kepler_velocity";   break;
                default: break;
                }
            }
            if (e.source == SKind::binary) s += " source=binary";
            if (e.sky)  s += " sky=set";
            if (e.site) s += " site=set";
            if (e.t_ref.has_value()) s += " t_ref=" + std::to_string(*e.t_ref);
            s += ">";
            return s;
        });

    // --- LightCurve ---
    // Optimized for magnification-only use: construct once, call many times.
    // Accepts params as: lcbi_params object, dict, or **kwargs.

    // Default options with vbm_compatible=1 (same as lc.Options() default).
    static const lcbi_options kDefaultOpts = []{ auto o = lcbi_default_options(); o.vbm_compatible = 1; return o; }();

    // Dispatch __call__ based on source kind.
    // For binary source, dict/kwargs must contain q_source + t0_2 + u0_2.
    auto compute_dispatch = [](const LC& lc,
                                py::array_t<double> times,
                                const lcbi_params& base_params,
                                py::dict extra) -> py::array_t<double> {
        if (lc.source_kind() == SKind::single) {
            return compute(lc, times, base_params);
        }
        // Binary source: extract q_source, q_mass, t0_2, u0_2 from extra dict.
        double q_source = 1.0, q_mass = 0.0, t0_2 = base_params.t0, u0_2 = base_params.umin;
        for (auto& item : extra) {
            const std::string key = item.first.cast<std::string>();
            if      (key == "q_source" || key == "fluxratio") q_source = item.second.cast<double>();
            else if (key == "q_mass")                          q_mass   = item.second.cast<double>();
            else if (key == "t0_2")                            t0_2     = item.second.cast<double>();
            else if (key == "u0_2")                            u0_2     = item.second.cast<double>();
        }
        auto buf = times.request();
        const double* ptr = static_cast<const double*>(buf.ptr);
        std::vector<double> tv(ptr, ptr + buf.size);
        std::vector<double> mags;
        {
            py::gil_scoped_release release;
            if (q_mass > 0.0) {
                // Coupled xallarap: source 2 has xi scaled by -1/q_mass, same t0/u0 (CoM)
                lcbi_params p2 = base_params;
                p2.xi_1 = -base_params.xi_1 / q_mass;
                p2.xi_2 = -base_params.xi_2 / q_mass;
                mags = lc.magnification_binary(tv, base_params, q_source, p2);
            } else {
                mags = lc.magnification_binary(tv, base_params, q_source, t0_2, u0_2);
            }
        }
        return vec_to_numpy(std::move(mags));
    };

    py::class_<LC, std::shared_ptr<LC>>(lc, "LightCurve")
        // Constructor 1: explicit lc.Options + lc.Effects objects
        .def(py::init([&](const lcbi_options& opts,
                           const Eff&          effects,
                           const PyLimbDarkening& ld) {
            return std::make_shared<LC>(opts, ld.c, ld.d, effects);
        }),
            py::arg("options")        = kDefaultOpts,
            py::arg("effects")        = Eff{},
            py::arg("limb_darkening") = PyLimbDarkening{})
        // Constructor 2: kwargs directly (convenience, backward-compatible)
        .def(py::init([&](py::kwargs kw) {
            auto o = kDefaultOpts;
            PyLimbDarkening ld{};
            Eff eff{};
            for (auto& item : kw) {
                const std::string key = item.first.cast<std::string>();
                // --- Options (numerics) ---
                if      (key == "source_bins")            o.source_bins            = item.second.cast<int>();
                else if (key == "caustic_bins")           o.caustic_bins           = item.second.cast<int>();
                else if (key == "param_type")             apply_param_type(o, item.second.cast<std::string>());
                else if (key == "adaptive_hex_threshold") o.adaptive_hex_threshold = item.second.cast<double>();
                else if (key == "parallax_mode")          o.parallax_mode          = item.second.cast<int>();
                else if (key == "mode")                   o.mode                   = item.second.cast<int>();
                else if (key == "grid_ratio")             o.grid_ratio             = item.second.cast<double>();
                else if (key == "point_source_threshold") o.point_source_threshold = item.second.cast<double>();
                else if (key == "options")                o = item.second.cast<lcbi_options>();
                // --- LimbDarkening ---
                else if (key == "ld_c" || key == "limb_darkening_c") ld.c = item.second.cast<double>();
                else if (key == "ld_d" || key == "limb_darkening_d") ld.d = item.second.cast<double>();
                else if (key == "limb_darkening") ld = item.second.cast<PyLimbDarkening>();
                // --- Effects (physics) ---
                else if (key == "source")         eff.source        = parse_source(item.second.cast<std::string>());
                else if (key == "orbital_motion") eff.orbital_motion = parse_orbital(item.second.cast<std::string>());
                else if (key == "xallarap")       eff.xallarap      = parse_xallarap(item.second.cast<std::string>());
                else if (key == "parallax")       eff.parallax      = item.second.cast<bool>();
                else if (key == "terrestrial")    eff.terrestrial   = item.second.cast<bool>();
                else if (key == "lens") {
                    const std::string lens = item.second.cast<std::string>();
                    if (lens != "binary_lens" && lens != "triple_lens") {
                        throw py::key_error("LightCurve: unknown lens '" + lens + "'");
                    }
                }
                else if (key == "sky") {
                    auto obj = py::reinterpret_borrow<py::object>(item.second);
                    if (!obj.is_none()) eff.sky = obj.cast<std::shared_ptr<SkyCoord>>();
                }
                else if (key == "site") {
                    auto obj = py::reinterpret_borrow<py::object>(item.second);
                    if (!obj.is_none()) eff.site = obj.cast<std::shared_ptr<Site>>();
                }
                else if (key == "t_ref") {
                    auto obj = py::reinterpret_borrow<py::object>(item.second);
                    if (!obj.is_none()) eff.t_ref = obj.cast<double>();
                }
                else throw py::key_error("LightCurve: unknown option '" + key + "'");
            }
            return std::make_shared<LC>(o, ld.c, ld.d, eff);
        }))
        .def_property_readonly("options", &LC::options)
        .def_property_readonly("effects", &LC::effects,
            py::return_value_policy::reference_internal)
        .def_property_readonly("ld_c",    &LC::ld_c)
        .def_property_readonly("ld_d",    &LC::ld_d)
        // Convenience shortcuts (delegate to Effects)
        .def_property_readonly("source", [](const LC& lc) -> std::string {
            return lc.source_kind() == SKind::binary ? "binary" : "single";
        })
        .def_property_readonly("orbital_motion", [](const LC& lc) -> std::string {
            if (lc.orbital_motion() == LCBI_ORBIT_CIRCULAR) return "circular";
            if (lc.orbital_motion() == LCBI_ORBIT_KEPLER)   return "kepler";
            return "static";
        })
        .def_property_readonly("sky", [](const LC& lc) -> py::object {
            if (!lc.sky_coord()) return py::none();
            return py::cast(lc.sky_coord());
        })
        .def_property_readonly("site", [](const LC& lc) -> py::object {
            if (!lc.site()) return py::none();
            return py::cast(lc.site());
        })
        .def_property_readonly("t_ref", [](const LC& lc) -> py::object {
            if (!lc.t_ref()) return py::none();
            return py::float_(*lc.t_ref());
        })
        .def_property_readonly("terrestrial", [](const LC& lc) {
            return lc.effects().terrestrial;
        })

        // __call__ overload 1: lcbi_params object
        .def("__call__",
            [&](const LC& lc, py::array_t<double> times, const lcbi_params& params) {
                if (lc.source_kind() == SKind::single) return compute(lc, times, params);
                throw std::runtime_error(
                    "source='binary': pass params as dict/kwargs including q_source, t0_2, u0_2");
                return py::array_t<double>{};  // unreachable
            },
            py::arg("times"), py::arg("params"))

        // __call__ overload 2: dict
        .def("__call__",
            [&](const LC& lc, py::array_t<double> times, py::dict d) {
                return compute_dispatch(lc, times, params_from_dict(d), d);
            },
            py::arg("times"), py::arg("params"))

        // __call__ overload 3: **kwargs
        .def("__call__",
            [&](const LC& lc, py::array_t<double> times, py::kwargs kw) {
                py::dict d(kw);
                return compute_dispatch(lc, times, params_from_dict(d), d);
            },
            py::arg("times"))

        // .magnification() alias
        .def("magnification",
            [&](const LC& lc, py::array_t<double> times, const lcbi_params& params) {
                if (lc.source_kind() == SKind::single) return compute(lc, times, params);
                throw std::runtime_error(
                    "source='binary': pass params as dict/kwargs including q_source, t0_2, u0_2");
                return py::array_t<double>{};
            },
            py::arg("times"), py::arg("params"))
        .def("magnification",
            [&](const LC& lc, py::array_t<double> times, py::dict d) {
                return compute_dispatch(lc, times, params_from_dict(d), d);
            },
            py::arg("times"), py::arg("params"))
        .def("magnification",
            [&](const LC& lc, py::array_t<double> times, py::kwargs kw) {
                py::dict d(kw);
                return compute_dispatch(lc, times, params_from_dict(d), d);
            },
            py::arg("times"))

        // source_trajectory(times, **params)
        // Returns {"x": array, "y": array} in the lens-plane frame.
        // All active effects (parallax, xallarap) are applied.
        .def("source_trajectory",
            [](const LC& lc, py::array_t<double> times, py::kwargs kw) -> py::dict {
                const lcbi_params p = lc.apply_coords(params_from_dict(py::dict(kw)));
                const lcbinint::model::LensParameters lp = lcbinint::model::from_c_params(p);
                const lcbinint::model::Trajectory traj(lp);
                const bool vbm = lc.options().vbm_compatible != 0;
                const lcbi_xallarap_param_type xa = lc.options().xallarap_param_type;

                auto buf = times.request();
                const double* ptr = static_cast<const double*>(buf.ptr);
                const int n = static_cast<int>(buf.size);

                std::vector<double> xs(n), ys(n);
                {
                    py::gil_scoped_release release;
                    for (int i = 0; i < n; ++i) {
                        const auto pos = traj.source_position(ptr[i], vbm, xa);
                        xs[i] = pos.x;
                        ys[i] = pos.y;
                    }
                }
                py::dict result;
                result["x"] = vec_to_numpy(std::move(xs));
                result["y"] = vec_to_numpy(std::move(ys));
                return result;
            }, py::arg("times"),
            "Compute source trajectory in the lens-plane frame.\n"
            "Applies all active effects (parallax, xallarap).\n"
            "Returns dict with 'x' and 'y' numpy arrays (Einstein ring units).")

        .def("__repr__", [](const LC& lc) {
            const auto& o = lc.options();
            std::string pt;
            if (o.vbm_compatible != 0 && o.center_of_mass == 0) pt = "vbm";
            else if (o.vbm_compatible != 0)                      pt = "vbm_center_of_mass";
            else if (o.center_of_mass != 0)                      pt = "center_of_mass";
            else                                                  pt = "lcbinint";
            const std::string src = lc.source_kind() == SKind::binary ? " source='binary'" : "";
            return "<lc.LightCurve param_type='" + pt
                + "' source_bins=" + std::to_string(o.source_bins) + src + ">";
        });
}
