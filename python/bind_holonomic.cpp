// ATPT holonomic solver (M7) -- isolated Python binding.
//
// Ships as a SEPARATELY NAMED module `_lcbinint_holonomic_m7` (decision 19).
// It shares no symbol and no file with `_lcbinint`; the shared
// `site-packages/_lcbinint*.so` is never rebuilt or touched.  Built only
// when `tests/holonomic_cpp/CMakeLists.txt` is configured explicitly into
// `build-holonomic-m7/` -- a normal configure of the main tree does not
// see this target.
//
// Purpose: keep the C++ value+Jacobian path A/B-comparable against the
// incumbent `binary_ray_shooting` from Python (benchmarks / audit).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"

namespace py = pybind11;
using namespace lcbinint::holonomic;

PYBIND11_MODULE(_lcbinint_holonomic_m7, m) {
    m.doc() =
        "ATPT holonomic solver (M7) -- C++ value+Jacobian path, isolated "
        "backend (decision 19). Not part of _lcbinint.";

    m.def(
        "epoch_jacobian",
        [](double xs, double ys, double rho, double q, double a, double u,
           bool barycentric, int n_r) {
            LensParams p{xs, ys, rho, q, a, barycentric};
            EpochJacobian ej = epoch_jacobian(p, u, n_r);
            py::dict out;
            out["mu"] = ej.mu;
            out["grad_mu"] = std::vector<double>(ej.grad_mu.begin(),
                                                 ej.grad_mu.end());
            out["dmu_du"] = ej.dmu_du;
            out["F0"] = ej.F0;
            out["F_half"] = ej.F_half;
            out["r_max"] = ej.r_max;
            out["status"] = std::string(to_string(ej.status));
            out["param_order"] =
                std::vector<std::string>{"xs", "ys", "rho", "q", "a"};
            return out;
        },
        py::arg("xs"), py::arg("ys"), py::arg("rho"), py::arg("q"),
        py::arg("a"), py::arg("u") = 0.0, py::arg("barycentric") = false,
        py::arg("n_r") = 64,
        "mu, grad_mu[5] d/d(xs,ys,rho,q,a), dmu_du, F0, F_half, r_max, "
        "status. Fail-closed: status != 'OK' means the Jacobian is not "
        "trustworthy (never a silent approximation).");
}
