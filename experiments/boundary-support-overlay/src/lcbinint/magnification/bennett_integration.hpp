#pragma once

#include "same_grid_flux_error.hpp"
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

struct BennettBoundaryModelResidual {
    bool resolved = false;
    // Integral(model) - Bennett(model), in node-value times grid-spacing units.
    double signed_residual = 0.0;
    double sqrt_coefficient = 0.0;
    double linear_coefficient = 0.0;
    double three_half_coefficient=0;
    double model_remainder=0;
};

// Diagnostic only: fit a + b*sqrt(t) + c*t using the limb and the nearest two
// inside samples at t=delta and delta+1 (t in grid spacings). It measures the
// known model defect, including the delta_c guard. It does NOT bound the model
// remainder, interior quadrature, boundary-position or outer-coordinate errors.
inline BennettBoundaryModelResidual bennett_boundary_model_residual(
    double delta, double node, double next_node, double limb, double delta_c)
{
    if (!std::isfinite(delta) || delta < 0.0 || delta > 1.0 ||
        !std::isfinite(node) || !std::isfinite(next_node) || !std::isfinite(limb) ||
        !std::isfinite(delta_c) || delta_c < 0.0 || delta_c > 1.0) {
        return {};
    }
    if (node == limb && next_node == limb) return {true, 0.0, 0.0, 0.0};
    if (delta <= 64.0*std::numeric_limits<double>::epsilon()) return {};
    const double t1=delta, t2=delta+1.0;
    const double s1=std::sqrt(t1), s2=std::sqrt(t2);
    const double determinant=s1*t2-s2*t1;
    const double u=node-limb, v=next_node-limb;
    const double b=(u*t2-v*t1)/determinant;
    const double c=(s1*v-s2*u)/determinant;
    const auto weights=bennett_boundary_weights(delta,delta_c);
    const double length=delta+0.5;
    const double residual = b*((2.0/3.0)*length*std::sqrt(length)-weights.node*s1)
                          + c*(0.5*length*length-weights.node*t1);
    if (!std::isfinite(b) || !std::isfinite(c) || !std::isfinite(residual)) return {};
    return {true,residual,b,c};
}

// Fit and moment evaluation are shared. Unlike the legacy forensic model,
// the fast path never solves an unused sqrt(t)+t coefficient system.
inline BennettBoundaryModelResidual bennett_endpoint_from_coefficients(
 double delta,double leading,double next,const BennettBoundaryWeights& weights) {
    BennettBoundaryModelResidual model;
    model.sqrt_coefficient=leading;model.three_half_coefficient=next;
    const auto moments=power_endpoint_quadrature_defects(delta,weights.node);
    model.signed_residual=leading*moments[0]+next*moments[1];
    if(leading!=0)model.model_remainder=std::abs(next*next/(2*leading)*moments[2]);
    model.resolved=std::isfinite(model.signed_residual)&&std::isfinite(model.model_remainder);
    return model;
}
inline BennettBoundaryModelResidual bennett_endpoint_flux_residual(
 double delta,double node,double next_node,double limb,double delta_c) {
    if(!(std::isfinite(delta)&&delta>=0&&delta<=1&&std::isfinite(node)&&std::isfinite(next_node)&&std::isfinite(limb)&&std::isfinite(delta_c)&&delta_c>=0&&delta_c<=1))return {};
    if(node==limb&&next_node==limb)return {true};
    if(delta<=64*std::numeric_limits<double>::epsilon())return {};
    const double u=(node-limb)/std::sqrt(delta),v=(next_node-limb)/std::sqrt(delta+1),next=v-u;
    return bennett_endpoint_from_coefficients(delta,u-next*delta,next,bennett_boundary_weights(delta,delta_c));
}
// Linear LD brightness is (1-c)+c*sqrt(depth). Combine the depth and
// lattice-coordinate square roots before fitting: two sqrt evaluations,
// no repeated source-brightness construction or subtraction near the limb.
inline BennettBoundaryModelResidual bennett_linear_ld_endpoint_flux_residual(
 double delta,double depth,double next_depth,double c,const BennettBoundaryWeights& weights) {
    if(!(std::isfinite(delta)&&delta>=0&&delta<=1&&std::isfinite(depth)&&depth>=0&&depth<=1&&std::isfinite(next_depth)&&next_depth>=0&&next_depth<=1&&std::isfinite(c)&&c>=0&&c<=1))return {};
    if(c==0||(depth==0&&next_depth==0))return {true};
    if(delta<=64*std::numeric_limits<double>::epsilon())return {};
    const double u=c*std::sqrt(depth/delta),v=c*std::sqrt(next_depth/(delta+1)),next=v-u;
    return bennett_endpoint_from_coefficients(delta,u-next*delta,next,weights);
}

struct FiniteOneNodeLDModel {
    bool resolved=false;
    double integral=0,uncertainty=0;
};
// Coordinate t is in grid spacings, with roots -left and +right. The
// parabola interpolates source depth at both roots and the only interior
// node. depth_third bounds |D'''(t)| on the entire interval. With exact
// roots/arithmetic the positivity test and sqrt perturbation bound are
// rigorous; the native lens-bound caller is not an outward-rounding proof.
inline FiniteOneNodeLDModel finite_one_node_ld_model(
 double left,double right,double depth,double c,double depth_third) {
    if(!(std::isfinite(left)&&std::isfinite(right)&&left>0&&right>0&&
         std::isfinite(depth)&&depth>=0&&depth<=1&&std::isfinite(c)&&c>=0&&c<=1&&
         std::isfinite(depth_third)&&depth_third>=0))return {};
    const double extent=std::max(left,right),width=left+right;
    if(!(depth/(left*right)>depth_third*extent/6))return {};
    const double shape=3.141592653589793*width*width/8;
    return {true,(1-c)*width+c*std::sqrt(depth/(left*right))*shape,
        c*shape*std::sqrt(depth_third*extent/6)};
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
