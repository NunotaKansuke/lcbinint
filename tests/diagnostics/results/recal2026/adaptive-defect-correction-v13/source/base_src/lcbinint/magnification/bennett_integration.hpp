#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace lcbinint::magnification::detail {

// Boundary coefficients for Bennett (2010), equations (11)--(14).
//
// The grid value nearest a boundary is f_1 and delta is the distance from
// that value to the boundary in units of the grid spacing.  The returned
// coefficients multiply the value at the limb and the nearest grid value in
// the one-dimensional boundary contribution, respectively.
struct BennettBoundaryWeights {
    double limb = 0.0;
    double node = 0.0;
};

inline BennettBoundaryWeights bennett_boundary_weights(
    double delta,
    double delta_c)
{
    const double d = std::isfinite(delta)
        ? std::clamp(delta, 0.0, 1.0)
        : 0.5;
    const double cutoff = std::isfinite(delta_c)
        ? std::clamp(delta_c, 0.0, 1.0)
        : 0.15;

    // At delta=0 the formal b(delta) expression is singular.  The boundary
    // is then exactly on the node, so the limiting finite rule is 1/2 f_1.
    // This also makes the formally second-order delta_c=0 option numerically
    // well-defined for an exactly aligned boundary.
    if (d <= 16.0 * std::numeric_limits<double>::epsilon()) {
        return {0.0, 0.5};
    }

    if (d >= cutoff) {
        const double b = (2.0 / 3.0) *
            std::sqrt((d + 0.5) / d);
        const double common = 0.5 + d;
        return {common * (1.0 - b), common * b};
    }

    // Bennett's guarded fallback for delta < delta_c.  It preserves the
    // ordinary second-order boundary moment while avoiding the large b(delta)
    // coefficient close to a grid point.
    return {d / 3.0, 0.5 + (2.0 * d) / 3.0};
}

// Convert one Bennett boundary contribution into a correction to the
// cell-centred/midpoint sum already accumulated by the caller.  For a run
// with at least two nodes, the nearest node has already been counted once;
// for a one-node run the caller must combine both sides and remove the single
// baseline node exactly once after adding both full boundary contributions.
inline double bennett_boundary_residual(
    double delta,
    double node_value,
    double limb_value,
    double delta_c)
{
    const auto weights = bennett_boundary_weights(delta, delta_c);
    return weights.limb * limb_value +
        (weights.node - 1.0) * node_value;
}

// Bennett (2010), equation (15), for the first two samples adjacent to an
// outer-coordinate boundary.  eta is the distance from the first interior
// sample to the boundary in grid spacings.
struct BennettOuterBoundaryWeights {
    double first = 0.0;
    double second = 0.0;
};

inline BennettOuterBoundaryWeights bennett_outer_boundary_weights(double eta)
{
    const double e = std::isfinite(eta)
        ? std::clamp(eta, 0.0, 1.0)
        : 0.5;
    return {
        3.0 / 8.0 + e + 0.5 * e * e,
        9.0 / 8.0 - 0.5 * e * e,
    };
}

} // namespace lcbinint::magnification::detail
