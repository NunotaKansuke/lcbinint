#pragma once

// ATPT holonomic solver -- Phase C: the internal mode router.
//
// One engine-neutral entry point that the production finite-source layer
// (FiniteSourceMagnifier / MagnificationExecutionPlan) can dispatch to,
// without any new public API.  The router selects work by the REQUESTED
// OUTPUT, not by an algorithm name (checkpoint sec.0 policy point 2):
//
//   kValue          -- magnification only (linear-limb-darkening blend).
//                      Today this still runs the fused value+Jacobian pass
//                      and discards the Jacobian; the dedicated M1/M2 value
//                      lane (Phase D) will slot in here with no signature
//                      change.
//   kValueJacobian  -- magnification + d mu / d(xs,ys,rho,q,a) + d mu / d u.
//                      The M6/M7 fused analytic pass (the working path).
//   kValueJvp       -- magnification + one directional derivative (M3).
//                      NOT IMPLEMENTED yet -- fails closed.
//
// Two routes, same result contract:
//   finite_source_binary(req)           -- stateless / cold, one epoch.
//   finite_source_binary_prepared(req, state, cfg, stats)
//                                       -- trajectory / warmup route over a
//                      caller-owned rolling PreparedEpochGeometry (Phase E).
//                      state.valid == false on the first epoch (cold build),
//                      then reused / warm-recomputed / cold-recomputed per
//                      the sec.8 fail-closed ladder in prepared_topology.
//
// Failure is always a Status, never a silent approximation.  The standalone
// epoch_jacobian / epoch_jacobian_prepared entry points stay in the tree
// for A/B comparison; this router only forwards to them.

#include <array>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/prepared_geometry.hpp"
#include "lcbinint/magnification/holonomic/status.hpp"

namespace lcbinint::holonomic {

enum class RequestedOutput {
    kValue,          // mu only
    kValueJacobian,  // mu + 5-Jacobian + dmu/du
    kValueJvp,       // mu + one directional derivative -- not implemented
};

struct FiniteSourceRequest {
    LensParams params;
    double u = 0.0;                 // linear limb-darkening coefficient
    RequestedOutput output = RequestedOutput::kValueJacobian;
    int n_r = 64;                   // radial Gauss-Chebyshev nodes per cell
    // Direction for kValueJvp, in user (xs,ys,rho,q,a) space.  Ignored for
    // the other modes.
    std::array<double, 5> jvp_direction{};
};

struct FiniteSourceOutcome {
    double mu = 0.0;
    std::array<double, 5> grad_mu{};  // valid iff has_jacobian
    double dmu_du = 0.0;              // valid iff has_jacobian
    double mu_jvp = 0.0;             // valid iff has_jvp
    double F0 = 0.0, F_half = 0.0;
    double r_max = 0.0;
    Status status = Status::OK;
    PreparedEpochGeometry::Provenance provenance =
        PreparedEpochGeometry::kColdOracle;
    bool has_jacobian = false;
    bool has_jvp = false;
};

// Production trajectory-reuse defaults: L2 warm-seeded D14 ON (exact parity,
// fail-closed), L1 verbatim cell-plan reuse OFF (pending the deferred
// cheaper-than-D14 certificate -- checkpoint sec.21 / policy point 3).
inline PreparedReuseConfig default_reuse_config() {
    PreparedReuseConfig cfg;
    cfg.allow_topology_reuse = false;
    cfg.allow_warm_d14 = true;
    return cfg;
}

namespace fsb_detail {

inline FiniteSourceOutcome from_epoch(const EpochJacobian& ej,
                                      RequestedOutput output,
                                      PreparedEpochGeometry::Provenance prov) {
    FiniteSourceOutcome out;
    out.mu = ej.mu;
    out.F0 = ej.F0;
    out.F_half = ej.F_half;
    out.r_max = ej.r_max;
    out.status = ej.status;
    out.provenance = prov;
    if (output == RequestedOutput::kValueJacobian) {
        out.grad_mu = ej.grad_mu;
        out.dmu_du = ej.dmu_du;
        out.has_jacobian = true;
    }
    return out;
}

inline FiniteSourceOutcome from_epoch_value(
    const EpochValue& ev, PreparedEpochGeometry::Provenance prov) {
    FiniteSourceOutcome out;
    out.mu = ev.mu;
    out.F0 = ev.F0;
    out.F_half = ev.F_half;  // 0 when u == 0 (blend weight is 0)
    out.r_max = ev.r_max;
    out.status = ev.status;
    out.provenance = prov;
    return out;  // has_jacobian == false, has_jvp == false
}

// The Phase D value lane owns kValue only while the fused F_half rule is the
// 64-node angular sweep.  HOLO_HOLONOMIC_TRANSPORT swaps the fused F_half to
// the v*K rule; until that is wired into radius_value (Phase D step 2), a
// kValue request under that flag routes back to the fused pass so mu still
// matches the flag-on epoch_jacobian.
inline bool value_lane_active(const FiniteSourceRequest& req) {
    return req.output == RequestedOutput::kValue &&
           !holo_holonomic_transport_enabled();
}

}  // namespace fsb_detail

// ---- stateless (cold) route -----------------------------------------------
inline FiniteSourceOutcome finite_source_binary(const FiniteSourceRequest& req) {
    if (req.output == RequestedOutput::kValueJvp) {
        FiniteSourceOutcome out;
        out.status = Status::GRADIENT_UNRELIABLE;  // JVP lane not built (M3)
        return out;
    }
    if (fsb_detail::value_lane_active(req)) {
        EpochValue ev = epoch_value(req.params, req.u, req.n_r);
        return fsb_detail::from_epoch_value(ev,
                                            PreparedEpochGeometry::kColdOracle);
    }
    EpochJacobian ej = epoch_jacobian(req.params, req.u, req.n_r);
    return fsb_detail::from_epoch(ej, req.output,
                                 PreparedEpochGeometry::kColdOracle);
}

// ---- prepared / trajectory route ----------------------------------------
inline FiniteSourceOutcome finite_source_binary_prepared(
    const FiniteSourceRequest& req, PreparedEpochGeometry& state,
    const PreparedReuseConfig& cfg = default_reuse_config(),
    PreparedReuseStats* stats = nullptr) {
    if (req.output == RequestedOutput::kValueJvp) {
        FiniteSourceOutcome out;
        out.status = Status::GRADIENT_UNRELIABLE;
        return out;
    }
    if (fsb_detail::value_lane_active(req)) {
        EpochValue ev = epoch_value_prepared(req.params, req.u, req.n_r, state,
                                             cfg, stats);
        return fsb_detail::from_epoch_value(ev, state.provenance);
    }
    EpochJacobian ej =
        epoch_jacobian_prepared(req.params, req.u, req.n_r, state, cfg, stats);
    return fsb_detail::from_epoch(ej, req.output, state.provenance);
}

}  // namespace lcbinint::holonomic
