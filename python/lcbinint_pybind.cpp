#include "bind_lc.hpp"
#include "bind_obs.hpp"
#include "bind_bayes.hpp"
#include "bind_optimize.hpp"
#include "bind_sample.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(lcbinint, m)
{
    m.doc() = "Python bindings for the lcbinint C++ core";

    register_lc_submodule(m);
    register_obs_submodule(m);
    register_bayes_submodule(m);
    register_optimize_submodule(m);
    register_sample_submodule(m);
}
