#include "bind_lc.hpp"
#include "bind_obs.hpp"
#include "bind_jax_ir.hpp"

#include "lcbinint/magnification/probe_diagnostics.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_lcbinint, m)
{
    m.doc() = "Python bindings for the lcbinint C++ core";

    register_lc_submodule(m);
    register_obs_submodule(m);
    register_jax_ir_submodule(m);

    // Calibration instrumentation for the seeding stage.  Inert unless
    // LCBININT_PROBE_STATS is set, and thread-local, so a harness that wants a
    // total runs the library single-threaded; see probe_diagnostics.hpp.
    m.def("probe_counters", []() {
        const auto& counters = lcbinint::magnification::probe_counters();
        py::dict out;
        out["enabled"] = lcbinint::magnification::probe_stats_enabled();
        out["policy"] = lcbinint::magnification::probe_policy_description();
        out["ring_solves"] = counters.ring_solves;
        out["heuristic_solves"] = counters.heuristic_solves;
        out["certified_solves"] = counters.certified_solves;
        out["certified_offered"] = counters.certified_offered;
        out["certified_extrema"] = counters.certified_extrema;
        out["certifications"] = counters.certifications;
        out["unproven"] = counters.unproven;
        out["ring_seconds"] = counters.ring_seconds;
        out["heuristic_seconds"] = counters.heuristic_seconds;
        out["certified_seconds"] = counters.certified_seconds;
        out["certify_seconds"] = counters.certify_seconds;
        return out;
    }, "Per-thread probe tallies since the last reset.");

    m.def("reset_probe_counters", &lcbinint::magnification::reset_probe_counters,
          "Zero the calling thread's probe tallies.");
}
