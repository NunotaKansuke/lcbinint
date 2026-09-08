#pragma once

// ATPT holonomic solver (M7) -- fused value + 5-component Jacobian.
// Ports python/lcbinint/holonomic_ref/jacobian.py:
//   _internal_to_user_jac, flux_jacobian, epoch_jacobian
// and singular.near_origin_source.
//
// One per-cell Gauss-Chebyshev radial pass (n_r nodes) produces BOTH the
// flux pair (F0, F_half) and its internal-parameter Jacobian; the chain
// rule maps (X, Y, rho, m0, a_pf) -> user (xs, ys, rho, q, a).  The blend
//   mu(u) = ((1-u) F0 + u F_half) / D,   D = pi rho^2 (1 - u/3)
// and dmu/du = pi rho^2 (F_half - 2 F0/3) / D^2  (numerator constant in u).
//
// Fail closed: any unreliable radius, a full cell, an uncertain topology,
// or a near-origin source downgrades status to GRADIENT_UNRELIABLE /
// TOPOLOGY_UNCERTAIN -- never a silent number.

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/angular_rule.hpp"
#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/fast_topology.hpp"
#include "lcbinint/magnification/holonomic/fp_env.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/prepared_geometry.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"
#include "lcbinint/magnification/holonomic/status.hpp"

namespace lcbinint::holonomic {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSingularZetaTol = 1.0e-3;  // singular.SINGULAR_ZETA_TOL

struct FluxJacobian {
    double F0 = 0.0, F_half = 0.0;
    std::array<double, 5> dF0{};       // user params (xs, ys, rho, q, a)
    std::array<double, 5> dF_half{};
    std::array<double, 5> dF0_internal{};
    std::array<double, 5> dF_half_internal{};
    double r_max = 0.0;
    Status status = Status::OK;
};

struct EpochJacobian {
    double mu = 0.0;
    std::array<double, 5> grad_mu{};   // d mu / d(xs, ys, rho, q, a)
    double dmu_du = 0.0;
    double F0 = 0.0, F_half = 0.0;
    double r_max = 0.0;
    Status status = Status::OK;
};

// dvec_internal order: (dX, dY, drho, dm0, da_pf).  Returns d/d(xs,ys,rho,q,a).
inline std::array<double, 5> internal_to_user_jac(
    const std::array<double, 5>& d, const LensParams& p) {
    const double q = p.q, a = p.a;
    const double inv = 1.0 / (1.0 + q);
    const double dm0_dq = -inv * inv;
    const double dm1_dq = inv * inv;
    std::array<double, 5> out{d[0], d[1], d[2], 0.0, 0.0};
    if (p.barycentric) {
        out[3] = d[3] * dm0_dq + d[0] * (a * dm1_dq);
        out[4] = d[4] + d[0] * (q * inv);
    } else {
        out[3] = d[3] * dm0_dq;
        out[4] = d[4];
    }
    return out;
}

inline bool near_origin_source(const PrimaryFrame& pf,
                               double tol = kSingularZetaTol) {
    return std::hypot(pf.X, pf.Y) < tol;
}

// The fused value + internal-Jacobian radial pass over an already-decided
// cell plan.  Split out of flux_jacobian so the prepared-geometry path
// (Phase E) reuses the identical integrator over a cached / warm-recomputed
// TopologyResult -- band discovery is the only thing that differs.
inline FluxJacobian flux_jacobian_integrate(const LensParams& p, int n_r,
                                            const PrimaryFrame& pf,
                                            const TopologyResult& topo) {
    FluxJacobian fj;
    fj.r_max = topo.r_max;
    // jacobian.flux_jacobian: OK iff the topology is clean, else
    // GRADIENT_UNRELIABLE (a single non-OK bucket -- never upgraded back).
    fj.status = (topo.status == Status::OK) ? Status::OK
                                            : Status::GRADIENT_UNRELIABLE;

    Cheb1Dyn rr(n_r);
    std::array<double, 5> dF0i{}, dFhi{};

    for (const auto& c : topo.cells) {
        double w = c.r_hi - c.r_lo;
        if (w <= 0.0 || c.kind == ArcKind::kEmpty) continue;
        if (c.kind == ArcKind::kFull)
            fj.status = Status::GRADIENT_UNRELIABLE;

        const double ins = 1e-9 * w;
        const double lo = c.r_lo + ins, hi = c.r_hi - ins;
        const double rmid = 0.5 * (lo + hi), rhalf = 0.5 * (hi - lo);
        // Chebyshev nodes are monotone in R within a cell -> warm-start the
        // per-node boundary-quartic solve from the previous node.  Fresh
        // (cold) start on the first node of every cell.
        QuarticWarm qw;
        RootPairWarm rpw;  // (m, v) transport state (HOLO_MV_TRANSPORT=1 only)
        rpw.certify = topo.from_warm_d14;  // cross-check thin arcs on reused plans
        for (int k = 0; k < n_r; ++k) {
            double R = rmid + rhalf * rr.x[k];
            RadiusTerms rt = radius_terms(R, pf, kTanRel, &qw, &rpw);
            if (!rt.reliable) fj.status = Status::GRADIENT_UNRELIABLE;
            double Wk = rhalf * rr.w[k];
            fj.F0 += Wk * rt.f0;
            fj.F_half += Wk * rt.fh;
            for (int j = 0; j < 5; ++j) {
                dF0i[j] += Wk * rt.df0[j];
                dFhi[j] += Wk * rt.dfh[j];
            }
        }
    }
    fj.dF0_internal = dF0i;
    fj.dF_half_internal = dFhi;
    fj.dF0 = internal_to_user_jac(dF0i, p);
    fj.dF_half = internal_to_user_jac(dFhi, p);
    return fj;
}

inline FluxJacobian flux_jacobian(const LensParams& p, int n_r,
                                  bool use_fast_planner) {
    const ScopedFlushDenormals _fp_guard;
    const PrimaryFrame pf = PrimaryFrame::from(p);
    // Band discovery: classify_cells (D14 oracle) is the default and the
    // authority.  use_fast_planner routes through the seed-anchored fast
    // planner + sec.5.2 screen, which itself falls back to classify_cells on
    // any bail / screen failure (fast_topology.hpp).  Phase A step 4.
    TopologyResult topo =
        use_fast_planner ? classify_cells_fast(pf) : classify_cells(pf);
    return flux_jacobian_integrate(p, n_r, pf, topo);
}

inline FluxJacobian flux_jacobian(const LensParams& p, int n_r = 64) {
    return flux_jacobian(p, n_r, holo_fast_planner_enabled());
}

// Prepared-geometry (Phase E) variant.  `state` is the caller-owned rolling
// PreparedEpochGeometry: invalid on the first epoch of a trajectory (cold
// build), then reused / warm-recomputed / cold-recomputed per the sec.8
// fail-closed ladder.  `cfg` toggles the L1 topology reuse and the L2 warm
// D14 seed independently (bench variants V1 / V2).
inline FluxJacobian flux_jacobian_prepared(const LensParams& p, int n_r,
                                           PreparedEpochGeometry& state,
                                           const PreparedReuseConfig& cfg,
                                           PreparedReuseStats* st) {
    const ScopedFlushDenormals _fp_guard;
    const PrimaryFrame pf = PrimaryFrame::from(p);
    TopologyResult topo = prepared_topology(pf, state, cfg, st);
    return flux_jacobian_integrate(p, n_r, pf, topo);
}

inline EpochJacobian epoch_jacobian(const LensParams& p, double u, int n_r,
                                    bool use_fast_planner) {
    const ScopedFlushDenormals _fp_guard;
    FluxJacobian fj = flux_jacobian(p, n_r, use_fast_planner);
    const PrimaryFrame pf = PrimaryFrame::from(p);

    EpochJacobian ej;
    ej.status = fj.status;
    ej.r_max = fj.r_max;
    ej.F0 = fj.F0;
    ej.F_half = fj.F_half;
    if (near_origin_source(pf)) ej.status = Status::GRADIENT_UNRELIABLE;

    const double rho = p.rho, r2 = rho * rho;
    const double D = kPi * r2 * (1.0 - u / 3.0);
    const double F0 = fj.F0, Fh = fj.F_half;
    ej.mu = ((1.0 - u) * F0 + u * Fh) / D;
    for (int j = 0; j < 5; ++j)
        ej.grad_mu[j] = ((1.0 - u) * fj.dF0[j] + u * fj.dF_half[j]) / D;
    ej.grad_mu[2] -= 2.0 * ej.mu / rho;
    ej.dmu_du = kPi * r2 * (Fh - 2.0 * F0 / 3.0) / (D * D);
    return ej;
}

inline EpochJacobian epoch_jacobian(const LensParams& p, double u = 0.0,
                                    int n_r = 64) {
    return epoch_jacobian(p, u, n_r, holo_fast_planner_enabled());
}

// Prepared-geometry (Phase E) epoch.  Identical result to
// epoch_jacobian(p, u, n_r, /*fast=*/false) when `state` is reused validly;
// the sec.8 cheap re-screen guarantees fail-closed to a cold recompute
// whenever the cached band topology is not certified at the new geometry.
inline EpochJacobian epoch_jacobian_prepared(const LensParams& p, double u,
                                             int n_r,
                                             PreparedEpochGeometry& state,
                                             const PreparedReuseConfig& cfg,
                                             PreparedReuseStats* st = nullptr) {
    const ScopedFlushDenormals _fp_guard;
    FluxJacobian fj = flux_jacobian_prepared(p, n_r, state, cfg, st);
    const PrimaryFrame pf = PrimaryFrame::from(p);

    EpochJacobian ej;
    ej.status = fj.status;
    ej.r_max = fj.r_max;
    ej.F0 = fj.F0;
    ej.F_half = fj.F_half;
    if (near_origin_source(pf)) ej.status = Status::GRADIENT_UNRELIABLE;

    const double rho = p.rho, r2 = rho * rho;
    const double D = kPi * r2 * (1.0 - u / 3.0);
    const double F0 = fj.F0, Fh = fj.F_half;
    ej.mu = ((1.0 - u) * F0 + u * Fh) / D;
    for (int j = 0; j < 5; ++j)
        ej.grad_mu[j] = ((1.0 - u) * fj.dF0[j] + u * fj.dF_half[j]) / D;
    ej.grad_mu[2] -= 2.0 * ej.mu / rho;
    ej.dmu_du = kPi * r2 * (Fh - 2.0 * F0 / 3.0) / (D * D);
    return ej;
}

}  // namespace lcbinint::holonomic
