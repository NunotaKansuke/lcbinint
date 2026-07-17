#include "bind_lc.hpp"
#include "bind_obs.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_lcbinint, m)
{
    m.doc() = "Python bindings for the lcbinint C++ core";

    register_lc_submodule(m);
    register_obs_submodule(m);
}
