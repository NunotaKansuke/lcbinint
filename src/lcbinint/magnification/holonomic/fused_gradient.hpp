#pragma once

#include "lens_frame.hpp"
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace lcbinint::holonomic {
struct FusedGradientOptions {
    double mu_atol=1e-8;
    double mu_rtol=1e-4;
};
struct FusedGradientResult {
    double mu=0;
    std::array<double,5> grad_mu{}; // xs, ys, rho, q, a; q is LensParams.q
    std::array<bool,5> gradient_valid{}; // usable finite result, not an accuracy bound
    bool value_converged=false;
    bool gradient_precision_checked=false;
    std::string stop;
    bool support_certified=false;
    std::size_t node_evaluations=0;
};

// Optional forward-only route, linked through liblcbinint_fused_gradient.a.
// Seeds are image positions in the primary frame (lenses at 0 and a), just
// like epoch_seeded_radial_frontier. Reuse seeds prepared by the value caller.
// Value tolerances select the mesh; no derivative-only refinement is done.
FusedGradientResult epoch_fused_gradient(const LensParams& p,double limb_darkening,
    const std::vector<std::array<double,2>>& primary_image_seeds,
    const FusedGradientOptions& options={});
}
