#pragma once

// ATPT holonomic solver -- full coupled-state ODE flux transport.
//
// This is the "full transport" of the rescue spikes (a)-(c)
// (evidence/holonomic/{holonomic_rescue_feasibility,flux_priority_basis_
// feasibility,coupled_transport_feasibility}.txt), distinct from the
// current HOLO_HOLONOMIC_TRANSPORT path.  HOLO_HOLONOMIC_TRANSPORT still
// runs the per-radial-node machinery -- a boundary-quartic (re)solve /
// (m,v) Newton continuation, two 6-iteration polish_endpoint Newtons, and
// a fresh 16-node Gauss-Chebyshev-2 K-rule -- at every one of the 64 GC1
// radial nodes; it only swaps the F_half *value+Jacobian* integrand from
// the 64-node angular sqrt(phi) sweep to v*K.
//
// Here the state
//
//     X(R) = [ (m_a, v_a) per arc | F0 | F_half | dF0i[5] | dFhi[5] ]
//
// is transported as ONE coupled system across a whole radial cell:
//
//   * (m_a, v_a) : the autonomous 2x2 implicit-function ODE
//                  root_pair_dR (root_pair.hpp) -- no root solve.
//   * d(m,v)/dp  : solved ALGEBRAICALLY at each R from the same 2x2
//                  Jacobian J_EO and boundary_quartic_dp -- not an ODE.
//   * F0         : dF0/dR   = sum_a R * delta_theta_a(m,v)   (RootPair::
//                  delta_theta -- closed form in the (m,v) block alone).
//   * F_half     : dFh/dR   = sum_a v_a K_a  via v_times_K[_jac]
//                  (holonomic_transport.hpp -- the regularized deflated
//                  x-chart K-rule; K finite & smooth as v -> 0).
//   * dF0i/dFhi  : the internal-parameter (X,Y,rho,m0,a) derivative of the
//                  above, accumulated exactly like the per-node pass so
//                  flux_jacobian_integrate's internal->user chain rule is
//                  unchanged.
//
// so on the normal path the per-cell 64-node GC1 radial flux quadrature,
// the per-node quartic (re)solve, and the per-node angular sqrt(phi) sweep
// all disappear -- replaced by one RK4 march (fixed substep, seeded once
// per cell at an interior anchor by a single cold quartic solve).
//
// Fail closed, per cell: a seed that is not a clean 1-4 real-root arc set,
// any arc straddling theta = pi (|m| + sqrt(v) > kHoloTMax), a singular
// (m,v) Jacobian, a negative deflated S2, a non-finite state, or a
// pathologically wide cell -> the whole cell reverts to the incumbent
// per-node loop in flux_{jacobian,value}_integrate.  No silent number.
//
// Opt-in: HOLO_ODE_TRANSPORT=1 (default OFF -- flag OFF is bit-identical
// to the current build).  holo_ode_transport_override() forces it for the
// in-process 3-solver benchmark / tests.
//
// Regime 2 (spike (c) sec. 2): on a cell where every arc's fold is out of
// reach (dist_fold >= 2H) and Phi_arc(R) is provably analytic well past the
// cell (Chebyshev coefficient-decay rho >= kOdeJetDecayMin), the per-stage
// 16-node K-rule for v*K + d(v*K)/dp[5] is replaced by a degree-kOdeJetDeg
// Taylor packet in R, seeded ONCE per cell from kOdeJetSeedN K-rule samples
// near the anchor (ode_jet_build).  Per-stage F_half cost then drops to six
// Horner evaluations.  Any doubt reverts the cell to the universal K-rule
// (regime 1) and, failing that, to the incumbent per-node loop.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/holonomic_transport.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"
#include "lcbinint/magnification/holonomic/root_pair.hpp"

namespace lcbinint::holonomic {

// The boundary quartic of a binary lens has <= 4 real roots -> <= 2 arc
// pairs; 3 is a hard safety cap (a degenerate probe is fail-closed anyway).
constexpr int kOdeMaxArcs = 3;
// state = 2*na (m,v) + 2 (F0,Fh) + 10 (dF0i[5], dFhi[5]).
constexpr int kOdeStateDim = 2 * kOdeMaxArcs + 12;
constexpr double kOdeSubstepFrac = 6.0;      // substep <= cell_half_width / 6
constexpr double kOdeSubstepCapDefault = 3.0e-4;
constexpr long kOdeMaxSteps = 400000;        // wider -> fail closed to per-node

// The march stops when an arc's v reaches this working floor; the remaining
// [R_stop, fold_edge] sliver -- where d(delta_theta)/dv ~ 1/sqrt(v) is
// singular and the fixed-step RK4 cannot resolve the Jacobian integrand --
// is integrated by ode_fold_tail under the substitution u = sqrt(|R_edge -
// R|) (dR = 2u du regularizes 1/sqrt(v) ~ 1/u to a bounded integrand).  Set
// well above kHoloVFloor so the RHS v-guard (defence in depth) never trips.
constexpr double kOdeVWork = 1.0e-9;
// Inside ode_fold_tail the u = sqrt(|R_f - R|) substitution regularizes the
// 1/sqrt(v) integrand, so the tail can be evaluated arbitrarily close to the
// fold (v -> 0).  It therefore uses its OWN floor -- only guarding against
// exactly zero / negative / non-finite v -- instead of the shared
// near-tangency floor kHoloVFloor (1e-10) the semi-holonomic path applies.
// The F_half integrand v*K -> 0 as v -> 0, so tail nodes below kHoloVFloor
// simply contribute 0 to F_half (the K-rule is not evaluated there).
constexpr double kOdeTailVFloor = 1.0e-250;
// Adaptive step control near a shrinking arc: cap |dh| so no arc's v
// changes by more than this fraction per step -- keeps RK4 4th-order
// accurate on the d(delta_theta)/dv ~ 1/sqrt(v) integrand as v -> 0
// (whether v is heading to a fold edge or bottoming out mid-cell).
constexpr double kOdeVStepFrac = 0.10;
// regime-2 flux-priority jet (spike (c) sec. 2): see ode_jet_build below.
constexpr int kOdeJetDeg = 6;                  // Taylor packet degree d
constexpr int kOdeJetSeedN = kOdeJetDeg + 3;   // 9 Chebyshev-Gauss seed nodes
constexpr double kOdeJetSeedSpan = 0.85;       // seed spans anchor +/- span*H
constexpr double kOdeJetDecayMin = 6.0;        // coefficient-decay rho gate
// aggressive regime-2 packet: seed 9 Chebyshev-Gauss nodes across the WHOLE
// cell (anchor +/- span*H), fit degree-6 packets for all 12 integrand columns
// (R*dtheta, dF0/dp[5], v*K, dvK/dp[5]), and integrate each analytically over
// [lo,hi].  No RK4 march, no fold tail -- the cell closes on the packet alone.
constexpr int kOdeJetCols = 12;                // widest column set (r2 rule)

// 5-point Gauss-Legendre on [-1, 1] for the regularized fold tail in u.
constexpr int kOdeGLn = 5;
constexpr double kOdeGLX[5] = {-0.906179845938664, -0.5384693101056831, 0.0,
                               0.5384693101056831, 0.906179845938664};
constexpr double kOdeGLW[5] = {0.23692688505618908, 0.47862867049936647,
                               0.5688888888888889, 0.47862867049936647,
                               0.23692688505618908};

// ---- flag ----------------------------------------------------------------
inline int& holo_ode_transport_override() {
    static int v = -1;  // -1 env, 0 off, 1 on
    return v;
}
inline bool holo_ode_transport_enabled() {
    const int o = holo_ode_transport_override();
    if (o >= 0) return o != 0;
    static const bool on = [] {
        const char* e = std::getenv("HOLO_ODE_TRANSPORT");
        return e && e[0] == '1';
    }();
    return on;
}
// RK4 substep cap (HOLO_ODE_SUBSTEP, default 3e-4) -- lets the benchmark
// trade truncation error against march length without a rebuild.
inline double holo_ode_substep_cap() {
    static const double c = [] {
        const char* e = std::getenv("HOLO_ODE_SUBSTEP");
        const double v = e ? std::atof(e) : 0.0;
        return (v > 0.0) ? v : kOdeSubstepCapDefault;
    }();
    return c;
}
// v-work hand-off floor (HOLO_ODE_VWORK, default kOdeVWork = 1e-9) -- the v at
// which the RK4 march stops and hands [R_end, edge] to the u-sub tail.  Raising
// it ends the near-fold adaptive-clamp step crawl earlier (cheaper march, more
// work on the regularized tail); calibration knob for the step-count study.
inline double holo_ode_vwork() {
    static const double c = [] {
        const char* e = std::getenv("HOLO_ODE_VWORK");
        const double v = e ? std::atof(e) : 0.0;
        return (v > 0.0) ? v : kOdeVWork;
    }();
    return c;
}
// regime-2 jet coefficient-decay gate (HOLO_ODE_JET_DECAY) -- calibration
// knob for the decay_rho threshold below which a jet-eligible cell reverts
// to the per-stage K-rule.  Default kOdeJetDecayMin.
inline double holo_ode_jet_decay_min() {
    static const double c = [] {
        const char* e = std::getenv("HOLO_ODE_JET_DECAY");
        const double v = e ? std::atof(e) : 0.0;
        return (v > 0.0) ? v : kOdeJetDecayMin;
    }();
    return c;
}
// aggressive regime-2 packet (HOLO_ODE_R2_PACKET, default ON when the ODE
// path is enabled) -- when every arc in a cell is regime-2, close the cell on
// analytic packet integrals instead of an RK4 march.  Set to 0 to fall back
// to the flux-priority jet + march for A/B timing.
inline int& holo_ode_r2_packet_override() {
    static int v = -1;  // -1 env, 0 off, 1 on
    return v;
}
inline bool holo_ode_r2_packet_enabled() {
    const int o = holo_ode_r2_packet_override();
    if (o >= 0) return o != 0;
    static const bool on = [] {
        const char* e = std::getenv("HOLO_ODE_R2_PACKET");
        return !(e && e[0] == '0');  // default ON
    }();
    return on;
}
// r2 packet tolerance scale (HOLO_ODE_R2_TAIL, default 1e-6 -> scale 1.0):
// multiplies both the (m,v) geometry-packet fidelity gate in ode_jet_build
// and the F0 G7/K15 convergence gate in r2_value_cell.
inline double holo_ode_r2_packet_tail_scale() {
    static const double s = [] {
        const char* e = std::getenv("HOLO_ODE_R2_TAIL");
        const double v = e ? std::atof(e) : 0.0;
        return ((v > 0.0) ? v : 1.0e-6) / 1.0e-6;
    }();
    return s;
}

// ---- instrumentation (thread-local; the benchmark / tests audit this) ---
struct OdeCounters {
    long cells_seen = 0;            // kArcs cells offered to the ODE path
    long cells_ode = 0;            // completed fully on the ODE path
    long cells_fallback = 0;       // reverted to the per-node loop
    long seed_quartic_solves = 0;  // cold arc_intervals() at the anchor (1/cell)
    long rhs_evals = 0;            // RHS evaluations (4 per RK4 step)
    long rk4_steps = 0;
    long krule_gc2_evals = 0;      // v_times_K[_jac] calls (the 16-node GC2 rule)
    long angular_sweep_nodes = 0;  // MUST stay 0 on the ODE path (audit)
    long per_node_quartic_solves = 0;  // MUST stay 0 on the ODE path (audit)
    long regime1 = 0;              // dist_fold < 2H  (K-rule territory)
    long regime2 = 0;              // dist_fold >= 2H (jet territory, deferred)
    // fallback-reason breakdown (diagnostic)
    long fb_seed_notarcs = 0;      // anchor arc_intervals not a clean kArcs set
    long fb_seed_polish = 0;       // endpoint polish unreliable / near-tangency
    long fb_seed_tmax = 0;         // anchor arc straddles theta = pi
    long fb_route_det = 0;         // singular (m,v) Jacobian at the anchor
    long fb_march_mvdet = 0;       // singular (m,v) Jacobian mid-march
    long fb_march_vfloor = 0;      // an arc's v fell below kHoloVFloor mid-march
    long fb_march_rpdr = 0;        // root_pair_dR not ok mid-march
    long fb_march_krule = 0;       // K-rule S2<0 / 8-vs-16 disagreement mid-march
    long fb_march_dtheta = 0;      // delta_theta derivative singular mid-march
    long fb_march_steps = 0;       // cell too wide (> kOdeMaxSteps)
    long fb_nonfinite = 0;         // non-finite accumulated state
    long fb_march_tail = 0;        // fold tail (m,v) sub-march / node eval failed
    long fold_tails = 0;          // ode_fold_tail invocations (up + down)
    long rho_cancel_gated = 0;    // epochs whose ODE pass was discarded and
                                  // re-run per-node (rho-derivative cancellation
                                  // ratio > kOdeRhoCancelMax)
    // --- regime-2 flux-priority jet (spike (c) sec. 2) ---
    long jet_eligible_cells = 0;  // cells all-arcs dist_fold >= 2H (jet tried)
    long jet_cells = 0;           // cells whose F_half integrand took the jet
    long jet_seed_krule_evals = 0;  // K-rule evals spent seeding jet germs
    long jet_rhs_evals = 0;       // RHS arc-evals served by the Horner jet
    long jet_demote_decay = 0;    // jet-eligible but decay_rho < gate -> K-rule
    long jet_demote_seed = 0;     // jet-eligible but a seed sample failed
    // --- aggressive regime-2 packet (no RK4 march; analytic cell integral) ---
    long r2_packet_cells = 0;     // regime-2 cells closed by analytic packet integral
    long r2_packet_demote = 0;    // regime-2 packet built but a column failed the gate
    long r2_packet_seed_fail = 0;  // regime-2 packet seed sample failed -> jet/march
    long r2_jac_cells = 0;        // jac-lane regime-2 cells closed on the (m,v) packet
    long r2_jac_demote = 0;       // jac-lane packet built but F0 / dF0 gate failed
    void reset() { *this = OdeCounters{}; }
};
inline OdeCounters& ode_counters() {
    static thread_local OdeCounters c;
    return c;
}

// ======================================================================
// regime-2 flux-priority jet  (spike (c) sec. 2 / evidence
// coupled_transport_feasibility.txt).  On a cell where every arc's fold is
// out of reach (dist_fold >= 2H) AND Phi_arc(R) is provably analytic well
// past the cell (coefficient-decay rho >= kOdeJetDecayMin), the F_half
// value+Jacobian integrand v*K -- otherwise a fresh 16-node Gauss-Chebyshev
// K-rule with 5 parameter sensitivities at every RK4 stage -- is replaced
// by a degree-kOdeJetDeg Taylor packet in R, seeded ONCE per cell from
// kOdeJetSeedN K-rule samples near the anchor.  Per-stage cost then drops
// to six Horner evaluations (vK + dvK[5]) with no sqrt.
//
// The jet is a SPEED path only: v*K is already the correct number (spike
// (a)).  Any doubt -- a seed sample that will not evaluate, decay_rho below
// the gate, a non-finite germ, or a fold reached mid-march -- reverts the
// cell to the universal per-stage K-rule (regime 1) or, failing that, to
// the incumbent per-node loop.  Never a silent approximation.  Constants
// kOdeJet* are declared near the top of the file; the decay gate is
// runtime-overridable via holo_ode_jet_decay_min() (HOLO_ODE_JET_DECAY).

struct JetGerm {
    double R_c = 0.0;
    double r_half = 0.0;                    // |R - R_c| bound for jet use
    double c_val[kOdeJetDeg + 1] = {};      // Taylor coeff k of  v*K(R)
    double c_dp[5][kOdeJetDeg + 1] = {};    // ... of  d(v*K)/dp_j
    // (m,v) geometry packet: the same kOdeJetSeedN seed nodes that build the
    // flux germ already solve the quartic and the (m,v) implicit-function
    // Jacobian, so m(R), v(R) and their five parameter-derivatives fit into
    // Taylor packets too.  A regime-2 value cell then reconstructs the whole
    // root pair by Horner instead of an (m,v)-only RK4 march to every
    // quadrature node -- the last ODE march removed from the regime-2 path.
    double c_m[kOdeJetDeg + 1] = {};        // Taylor coeff k of  m(R)
    double c_v[kOdeJetDeg + 1] = {};        // ... of  v(R)
    double c_dm[5][kOdeJetDeg + 1] = {};    // ... of  dm/dp_j
    double c_dv[5][kOdeJetDeg + 1] = {};    // ... of  dv/dp_j
    double decay_rho = 0.0;
    double mv_decay_rho = 0.0;
    bool active = false;                    // v*K flux packet usable
    bool mv_active = false;                 // (m,v) geometry packet usable
};

namespace ode_detail {

// Square Chebyshev interpolation at kOdeJetSeedN abscissae xs[] in [-1,1]:
// solve  T_j(x_k) c_j = rhs_k  for up to `ncol` right-hand sides at once
// (Gauss-Jordan with partial pivot, kOdeJetSeedN x kOdeJetSeedN).  coef[k]
// holds the k-th Chebyshev coefficient for each column.
inline bool cheb_fit(const double* xs, const double rhs[kOdeJetSeedN][kOdeJetCols],
                     int ncol, double coef[kOdeJetSeedN][kOdeJetCols]) {
    constexpr int N = kOdeJetSeedN;
    double M[N][N], b[N][kOdeJetCols];
    for (int k = 0; k < N; ++k) {
        double tm1 = 1.0, t = xs[k];
        M[k][0] = 1.0;
        if (N > 1) M[k][1] = xs[k];
        for (int j = 2; j < N; ++j) {
            const double tp1 = 2.0 * xs[k] * t - tm1;
            M[k][j] = tp1;
            tm1 = t;
            t = tp1;
        }
        for (int c = 0; c < ncol; ++c) b[k][c] = rhs[k][c];
    }
    for (int col = 0; col < N; ++col) {
        int piv = col;
        double best = std::fabs(M[col][col]);
        for (int r = col + 1; r < N; ++r)
            if (std::fabs(M[r][col]) > best) { best = std::fabs(M[r][col]); piv = r; }
        if (!(best > 1e-300)) return false;
        if (piv != col) {
            for (int c = 0; c < N; ++c) std::swap(M[col][c], M[piv][c]);
            for (int c = 0; c < ncol; ++c) std::swap(b[col][c], b[piv][c]);
        }
        const double dinv = 1.0 / M[col][col];
        for (int r = 0; r < N; ++r) {
            if (r == col) continue;
            const double f = M[r][col] * dinv;
            if (f == 0.0) continue;
            for (int c = col; c < N; ++c) M[r][c] -= f * M[col][c];
            for (int c = 0; c < ncol; ++c) b[r][c] -= f * b[col][c];
        }
    }
    for (int col = 0; col < N; ++col) {
        const double dinv = 1.0 / M[col][col];
        for (int c = 0; c < ncol; ++c) coef[col][c] = b[col][c] * dinv;
    }
    return true;
}

// Coefficient-decay rho of a Chebyshev series cf[0..N-1] fitted over the R
// window [A, B]:  rho_x = exp(-slope) of log(|cf_k| / |cf_0|) vs k, r_a =
// rho_x (B - A)/2, decay_rho = r_a / H.  A clean geometric tail -> large
// rho.  The fit starts at k = 2, NOT k = 1: c_1 carries the arc's parity
// about the anchor and is routinely an order of magnitude out of line with
// the geometric trend (small when v*K is near-even, large when near-odd),
// which would corrupt a 3-point slope.  It ends at the last coefficient
// still clearly above the rounding floor (capped at k = 6, before the
// double-precision plateau of a well-resolved germ).
inline double cheb_decay_rho(const double* cf, double A, double B, double H) {
    constexpr int N = kOdeJetSeedN;
    const double c0 = std::fabs(cf[0]) + 1e-300;
    int kmax = 2;
    for (int k = 3; k < N; ++k)
        if (std::fabs(cf[k]) / c0 > 1e-9) kmax = k;
    if (kmax > 6) kmax = 6;
    if (kmax < 4) return 1.0e6;  // at the floor by k=3 -> extremely analytic
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int m = 0;
    for (int k = 2; k <= kmax; ++k) {
        const double y = std::log(std::max(std::fabs(cf[k]) / c0, 1e-16));
        sx += k; sy += y; sxx += (double)k * k; sxy += (double)k * y;
        ++m;
    }
    const double den = m * sxx - sx * sx;
    const double sl = (den != 0.0) ? (m * sxy - sx * sy) / den : 0.0;
    const double rho_x = (sl < 0.0) ? std::exp(-sl) : 40.0;
    const double r_a = rho_x * 0.5 * (B - A);
    return (H > 0.0) ? r_a / H : 0.0;
}

// Chebyshev series cf[0..N-1] over R in [A, B]  ->  Taylor coefficients
// tay[0..kOdeJetDeg] about R_c  (tay[k] = f^(k)(R_c)/k!).  Exact for the
// degree-(N-1) interpolant; the packet is then truncated to degree
// kOdeJetDeg.  Route: Cheb -> monomial in x (T_j recurrence) -> shift/scale
// x = xc + sc (R - R_c) -> collect powers of (R - R_c).
inline void cheb_to_taylor(const double* cf, double A, double B, double R_c,
                           double tay[kOdeJetDeg + 1]) {
    constexpr int N = kOdeJetSeedN;
    // T_j monomial rows: Tm[j][i] = coeff of x^i in T_j(x)
    double Tm[N][N];
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) Tm[j][i] = 0.0;
    Tm[0][0] = 1.0;
    if (N > 1) Tm[1][1] = 1.0;
    for (int j = 2; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            double val = -Tm[j - 2][i];
            if (i > 0) val += 2.0 * Tm[j - 1][i - 1];
            Tm[j][i] = val;
        }
    }
    double mono_x[N];
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) s += cf[j] * Tm[j][i];
        mono_x[i] = s;
    }
    // x(R) = xc + sc (R - R_c),  xc = (2 R_c - (A+B))/(B-A),  sc = 2/(B-A)
    const double sc = 2.0 / (B - A);
    const double xc = (2.0 * R_c - (A + B)) / (B - A);
    // binomials C(i, k), i,k < N
    double C[N][N];
    for (int i = 0; i < N; ++i) {
        C[i][0] = 1.0;
        for (int k = 1; k <= i; ++k)
            C[i][k] = C[i - 1][k - 1] + (k <= i - 1 ? C[i - 1][k] : 0.0);
        for (int k = i + 1; k < N; ++k) C[i][k] = 0.0;
    }
    // powers
    double xcpow[N], scpow[N];
    xcpow[0] = 1.0; scpow[0] = 1.0;
    for (int i = 1; i < N; ++i) { xcpow[i] = xcpow[i - 1] * xc; scpow[i] = scpow[i - 1] * sc; }
    for (int k = 0; k <= kOdeJetDeg; ++k) {
        double s = 0.0;
        for (int i = k; i < N; ++i)
            s += mono_x[i] * C[i][k] * xcpow[i - k] * scpow[k];
        tay[k] = s;
    }
}

// Horner evaluation of a degree-kOdeJetDeg Taylor packet about R_c.
inline double jet_eval(const double* c, double R_c, double R) {
    const double u = R - R_c;
    double acc = c[kOdeJetDeg];
    for (int k = kOdeJetDeg - 1; k >= 0; --k) acc = acc * u + c[k];
    return acc;
}

}  // namespace ode_detail

// ---- seed: one cold quartic solve at the cell's interior anchor ---------
struct OdeSeed {
    int na = 0;
    double mv[2 * kOdeMaxArcs] = {};
    bool ok = false;
};
inline OdeSeed ode_seed_anchor(const PrimaryFrame& pf, double R,
                               OdeCounters& ctr) {
    OdeSeed sd;
    QuarticWarm qw;
    const ArcSet as = arc_intervals(R, pf, &qw, /*rpw=*/nullptr);
    ++ctr.seed_quartic_solves;
    if (as.kind != ArcKind::kArcs) { ++ctr.fb_seed_notarcs; return sd; }
    if ((int)as.arcs.size() == 0 || (int)as.arcs.size() > kOdeMaxArcs) {
        ++ctr.fb_seed_notarcs;
        return sd;
    }

    const double tan_thresh = kTanRel * pf.rho / std::max(R, 1e-9);
    int na = 0;
    for (const auto& arc : as.arcs) {
        const PolishResult pe = polish_endpoint(R, arc[0], pf);
        const PolishResult pl = polish_endpoint(R, arc[1], pf);
        if (!pe.reliable || !pl.reliable) { ++ctr.fb_seed_polish; return sd; }
        double te = pe.theta, tl = pl.theta;
        if (tl <= te) tl += kTwoPi;
        if (std::fabs(pe.dphi_dtheta) < tan_thresh ||
            std::fabs(pl.dphi_dtheta) < tan_thresh) {
            ++ctr.fb_seed_polish;
            return sd;
        }
        const double t_lo = std::tan(0.5 * te), t_hi = std::tan(0.5 * tl);
        if (!std::isfinite(t_lo) || !std::isfinite(t_hi) || t_hi <= t_lo) {
            ++ctr.fb_seed_polish;
            return sd;
        }
        const double half = 0.5 * (t_hi - t_lo);
        const double m = 0.5 * (t_lo + t_hi), v = half * half;
        if (std::fabs(m) + half > kHoloTMax || v <= kHoloVFloor) {
            ++ctr.fb_seed_tmax;
            return sd;
        }
        sd.mv[2 * na] = m;
        sd.mv[2 * na + 1] = v;
        ++na;
    }
    if (na == 0) return sd;
    sd.na = na;
    sd.ok = true;
    return sd;
}

// ---- RHS: d/dR of the coupled state ------------------------------------
// Layout (WantJac): [ (m,v)_0..na-1 | F0 | F_half | dF0i[0..4] | dFhi[0..4] ]
//          (value): [ (m,v)_0..na-1 | F0 | F_half ]
namespace ode_detail {

// d(delta_theta)/d{m,v} for delta_theta = 2 atan2(2 sqrt(v), 1 + m^2 - v).
struct DThetaDeriv {
    double val, d_m, d_v;
    bool ok;
};
inline DThetaDeriv dtheta_derivs(double m, double v) {
    const double s = std::sqrt(v);
    const double c = 1.0 + m * m - v;
    const double D = 4.0 * v + c * c;
    if (!(D > 0.0) || !(s > 0.0)) return {0.0, 0.0, 0.0, false};
    return {2.0 * std::atan2(2.0 * s, c),
            -8.0 * s * m / D,
            2.0 * (c / s + 2.0 * s) / D,
            true};
}

// Algebraic d(m,v)/dp_j for j in {X,Y,rho,m0,a} at fixed R, from the (m,v)
// implicit-function Jacobian and boundary_quartic_dp.  Same 2x2 solve as
// root_pair_dR with the R-derivative RHS replaced by the p-derivative RHS.
inline bool mv_param_jac(const RootPair& rp, const std::array<double, 5>& pc,
                         const QuarticParamJac& dpc, const EOJac& J, double det,
                         std::array<double, 5>& dm_dp,
                         std::array<double, 5>& dv_dp) {
    if (std::fabs(det) < kRootPairDetFloor || !std::isfinite(det)) return false;
    const double m = rp.m, v = rp.v;
    for (int j = 0; j < 5; ++j) {
        const std::array<double, 5>& g = dpc.dp[j];
        const double G0 =
            g[0] + m * (g[1] + m * (g[2] + m * (g[3] + m * g[4])));
        const double G1 =
            g[1] + m * (2 * g[2] + m * (3 * g[3] + m * 4 * g[4]));
        const double G2 = 2 * g[2] + m * (6 * g[3] + m * 12 * g[4]);
        const double G3 = 6 * g[3] + m * 24 * g[4];
        const double E_pj = G0 + 0.5 * v * G2 + v * v * g[4];
        const double O_pj = G1 + (v / 6.0) * G3;
        dm_dp[j] = -(J.O_v * E_pj - J.E_v * O_pj) / det;
        dv_dp[j] = -(-J.O_m * E_pj + J.E_m * O_pj) / det;
    }
    return true;
}

}  // namespace ode_detail

inline bool ode_rhs_jac(double R, const double* y, int na, const PrimaryFrame& pf,
                        const JetGerm* jets, double* dy, OdeCounters& ctr) {
    ++ctr.rhs_evals;
    const int A = 2 * na;
    for (int i = 0; i < A + 12; ++i) dy[i] = 0.0;

    const QuarticCoeffs pc = boundary_quartic(R, pf);
    const QuarticCoeffs pcR = boundary_quartic_dR(R, pf);
    const QuarticParamJac dpc = boundary_quartic_dp(R, pf);

    for (int a = 0; a < na; ++a) {
        const RootPair rp{y[2 * a], y[2 * a + 1]};
        if (!(rp.v > kHoloVFloor) || !std::isfinite(rp.m)) {
            ++ctr.fb_march_vfloor;
            return false;
        }

        const RootPairDR d = root_pair_dR(rp, pc.p, pcR.p);
        if (!d.ok) { ++ctr.fb_march_rpdr; return false; }
        dy[2 * a] = d.dm_dR;
        dy[2 * a + 1] = d.dv_dR;

        const EOJac J = eo_jacobian(rp, pc.p);
        const double det = J.E_m * J.O_v - J.E_v * J.O_m;
        std::array<double, 5> dm_dp{}, dv_dp{};
        if (!ode_detail::mv_param_jac(rp, pc.p, dpc, J, det, dm_dp, dv_dp)) {
            ++ctr.fb_march_mvdet;
            return false;
        }

        const ode_detail::DThetaDeriv dt =
            ode_detail::dtheta_derivs(rp.m, rp.v);
        if (!dt.ok) { ++ctr.fb_march_dtheta; return false; }
        dy[A + 0] += R * dt.val;
        for (int j = 0; j < 5; ++j)
            dy[A + 2 + j] += R * (dt.d_m * dm_dp[j] + dt.d_v * dv_dp[j]);

        if (jets && jets[a].active &&
            std::fabs(R - jets[a].R_c) <= jets[a].r_half) {
            ++ctr.jet_rhs_evals;
            dy[A + 1] += ode_detail::jet_eval(jets[a].c_val, jets[a].R_c, R);
            for (int j = 0; j < 5; ++j)
                dy[A + 7 + j] +=
                    ode_detail::jet_eval(jets[a].c_dp[j], jets[a].R_c, R);
        } else {
            ++ctr.krule_gc2_evals;
            const VKJacobian vkj = v_times_K_jac(rp.m, rp.v, dm_dp, dv_dp, R, pf,
                                                 pc.p, dpc.dp);
            if (!vkj.ok) { ++ctr.fb_march_krule; return false; }
            dy[A + 1] += vkj.vK;
            for (int j = 0; j < 5; ++j) dy[A + 7 + j] += vkj.dvK[j];
        }
    }
    return true;
}

inline bool ode_rhs_value(double R, const double* y, int na,
                          const PrimaryFrame& pf, bool want_fh,
                          const JetGerm* jets, double* dy, OdeCounters& ctr) {
    ++ctr.rhs_evals;
    const int A = 2 * na;
    for (int i = 0; i < A + 2; ++i) dy[i] = 0.0;

    const QuarticCoeffs pc = boundary_quartic(R, pf);
    const QuarticCoeffs pcR = boundary_quartic_dR(R, pf);

    for (int a = 0; a < na; ++a) {
        const RootPair rp{y[2 * a], y[2 * a + 1]};
        if (!(rp.v > kHoloVFloor) || !std::isfinite(rp.m)) {
            ++ctr.fb_march_vfloor;
            return false;
        }

        const RootPairDR d = root_pair_dR(rp, pc.p, pcR.p);
        if (!d.ok) { ++ctr.fb_march_rpdr; return false; }
        dy[2 * a] = d.dm_dR;
        dy[2 * a + 1] = d.dv_dR;

        const ode_detail::DThetaDeriv dt =
            ode_detail::dtheta_derivs(rp.m, rp.v);
        if (!dt.ok) { ++ctr.fb_march_dtheta; return false; }
        dy[A + 0] += R * dt.val;

        if (want_fh) {
            if (jets && jets[a].active &&
                std::fabs(R - jets[a].R_c) <= jets[a].r_half) {
                ++ctr.jet_rhs_evals;
                dy[A + 1] +=
                    ode_detail::jet_eval(jets[a].c_val, jets[a].R_c, R);
            } else {
                ++ctr.krule_gc2_evals;
                const VKValue vk = v_times_K(rp.m, rp.v, R, pf, pc.p);
                if (!vk.ok) { ++ctr.fb_march_krule; return false; }
                dy[A + 1] += vk.vK;
            }
        }
    }
    return true;
}

// ---- RK4 march (fixed substep, v-work stop control) ------------------
// Marches [R0, R1].  If an arc's v is projected to fall below kOdeVWork the
// step is shortened to land that arc exactly on kOdeVWork and the march
// STOPS there: *R_end is set to the stop radius and *fold_arc to the arc
// index (a fold edge).  Otherwise *R_end = R1, *fold_arc = -1.  The caller
// hands [*R_end, R1] to ode_fold_tail.
template <bool WantJac>
inline bool ode_march(double R0, double R1, double* y, int na,
                      const PrimaryFrame& pf, bool want_fh, double H,
                      const JetGerm* jets, OdeCounters& ctr, double* R_end,
                      int* fold_arc) {
    *R_end = R1;
    *fold_arc = -1;
    const double span = R1 - R0;
    if (span == 0.0) return true;
    const int ND = WantJac ? (2 * na + 12) : (2 * na + 2);

    const double hmax = std::min(holo_ode_substep_cap(), H / kOdeSubstepFrac);
    const double sgn = (span > 0.0) ? 1.0 : -1.0;
    const double hbase = sgn * std::max(hmax, 1e-300);
    const long budget =
        (long)std::ceil(std::fabs(span) / std::max(hmax, 1e-300)) + 4;
    if (budget > kOdeMaxSteps) { ++ctr.fb_march_steps; return false; }

    std::array<double, kOdeStateDim> k1{}, k2{}, k3{}, k4{}, tmp{};
    auto rhs = [&](double R, const double* yy, double* out) -> bool {
        return WantJac ? ode_rhs_jac(R, yy, na, pf, jets, out, ctr)
                       : ode_rhs_value(R, yy, na, pf, want_fh, jets, out, ctr);
    };

    const double vwork = holo_ode_vwork();
    double R = R0;
    for (long st = 0; st < kOdeMaxSteps; ++st) {
        const double remain = R1 - R;
        if (std::fabs(remain) <= 1e-15 * std::max(1.0, std::fabs(R1))) {
            *R_end = R1;
            return true;
        }
        // already at / past the working floor -> stop cleanly here
        for (int a = 0; a < na; ++a)
            if (y[2 * a + 1] < vwork) {
                *R_end = R;
                *fold_arc = a;
                return true;
            }

        double h = (std::fabs(remain) < std::fabs(hbase)) ? remain : hbase;

        if (!rhs(R, y, k1.data())) return false;
        // adaptive: limit |v change| per step to kOdeVStepFrac * v
        for (int a = 0; a < na; ++a) {
            const double v_now = y[2 * a + 1];
            const double dv = k1[2 * a + 1];
            const double adv = std::fabs(dv);
            if (adv > 0.0) {
                const double h_v = kOdeVStepFrac * v_now / adv;
                if (h_v < std::fabs(h)) h = sgn * h_v;
            }
        }
        // v-work step control: shrink h to land the first arc to reach
        // kOdeVWork exactly on it, then stop after taking that step.
        int hit = -1;
        for (int a = 0; a < na; ++a) {
            const double v_now = y[2 * a + 1];
            const double dv = k1[2 * a + 1];
            if (dv >= 0.0) continue;
            if (v_now + h * dv < vwork) {
                const double h_hit = (vwork - v_now) / dv;
                if (h_hit > 0.0 && std::fabs(h_hit) <= std::fabs(h)) {
                    h = h_hit;
                    hit = a;
                }
            }
        }

        for (int i = 0; i < ND; ++i) tmp[i] = y[i] + 0.5 * h * k1[i];
        if (!rhs(R + 0.5 * h, tmp.data(), k2.data())) return false;
        for (int i = 0; i < ND; ++i) tmp[i] = y[i] + 0.5 * h * k2[i];
        if (!rhs(R + 0.5 * h, tmp.data(), k3.data())) return false;
        for (int i = 0; i < ND; ++i) tmp[i] = y[i] + h * k3[i];
        if (!rhs(R + h, tmp.data(), k4.data())) return false;
        for (int i = 0; i < ND; ++i)
            y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        R += h;
        ++ctr.rk4_steps;

        if (hit >= 0) {
            *R_end = R;
            *fold_arc = hit;
            return true;
        }
    }
    ++ctr.fb_march_steps;
    return false;
}

// ---- (m,v)-only RK4 (no flux accumulators): used inside a fold tail -----
inline bool mv_only_march(double R0, double R1, double* mv, int na,
                          const PrimaryFrame& pf, OdeCounters& ctr) {
    const double span = R1 - R0;
    if (span == 0.0) return true;
    constexpr int nsub = 8;
    const double h = span / (double)nsub;
    std::array<double, 2 * kOdeMaxArcs> k1{}, k2{}, k3{}, k4{}, tmp{};
    auto rhs = [&](double R, const double* m, double* out) -> bool {
        const QuarticCoeffs pc = boundary_quartic(R, pf);
        const QuarticCoeffs pcR = boundary_quartic_dR(R, pf);
        for (int a = 0; a < na; ++a) {
            const RootPair rp{m[2 * a], m[2 * a + 1]};
            if (!std::isfinite(rp.m) || !(rp.v > kOdeTailVFloor)) return false;
            const RootPairDR d = root_pair_dR(rp, pc.p, pcR.p);
            if (!d.ok) return false;
            out[2 * a] = d.dm_dR;
            out[2 * a + 1] = d.dv_dR;
        }
        return true;
    };
    (void)ctr;
    for (int st = 0; st < nsub; ++st) {
        const double R = R0 + (double)st * h;
        if (!rhs(R, mv, k1.data())) return false;
        for (int i = 0; i < 2 * na; ++i) tmp[i] = mv[i] + 0.5 * h * k1[i];
        if (!rhs(R + 0.5 * h, tmp.data(), k2.data())) return false;
        for (int i = 0; i < 2 * na; ++i) tmp[i] = mv[i] + 0.5 * h * k2[i];
        if (!rhs(R + 0.5 * h, tmp.data(), k3.data())) return false;
        for (int i = 0; i < 2 * na; ++i) tmp[i] = mv[i] + h * k3[i];
        if (!rhs(R + h, tmp.data(), k4.data())) return false;
        for (int i = 0; i < 2 * na; ++i)
            mv[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    return true;
}

// ---- regime-2 jet: seed the per-arc Taylor germ once per cell ----------
// Samples v*K and d(v*K)/dp[5] at kOdeJetSeedN Chebyshev-Gauss nodes about
// the anchor (each reached by an (m,v)-only RK4 from the seed -- NO quartic
// re-solve), fits a degree-(kOdeJetSeedN-1) Chebyshev series per arc per
// column, gates on the coefficient-decay rho of the value column, and
// converts to a degree-kOdeJetDeg Taylor packet about the anchor.  All arcs
// must clear the gate; otherwise every germ is left inactive and the cell
// runs the universal K-rule.  Never writes an accumulator -- pure setup.
inline void ode_jet_build(const PrimaryFrame& pf, double anchor, double H,
                          const OdeSeed& sd, OdeCounters& ctr, JetGerm* germs) {
    constexpr int NS = kOdeJetSeedN;
    const int na = sd.na;

    double nodeR[NS], nodeX[NS];
    for (int k = 0; k < NS; ++k) {
        const double ck = std::cos(kHoloPi * ((NS - 1 - k) + 0.5) / NS);
        nodeR[k] = anchor + kOdeJetSeedSpan * H * ck;  // ascending in k
    }
    const double A = nodeR[0], B = nodeR[NS - 1];
    if (!(B > A)) { ++ctr.jet_demote_seed; return; }
    for (int k = 0; k < NS; ++k)
        nodeX[k] = (2.0 * nodeR[k] - (A + B)) / (B - A);

    // samp[arc][node][col]:  col 0 = v*K,  col 1..5 = d(v*K)/dp_j
    // sampmv[arc][node][col]: col 0 = m, 1 = v, 2..6 = dm/dp_j, 7..11 = dv/dp_j
    double samp[kOdeMaxArcs][NS][kOdeJetCols];
    double sampmv[kOdeMaxArcs][NS][kOdeJetCols];
    for (int k = 0; k < NS; ++k) {
        std::array<double, 2 * kOdeMaxArcs> mv{};
        for (int i = 0; i < 2 * na; ++i) mv[i] = sd.mv[i];
        if (!mv_only_march(anchor, nodeR[k], mv.data(), na, pf, ctr)) {
            ++ctr.jet_demote_seed;
            return;
        }
        const QuarticCoeffs pc = boundary_quartic(nodeR[k], pf);
        const QuarticParamJac dpc = boundary_quartic_dp(nodeR[k], pf);
        for (int a = 0; a < na; ++a) {
            const RootPair rp{mv[2 * a], mv[2 * a + 1]};
            if (!(rp.v > kHoloVFloor) || !std::isfinite(rp.m)) {
                ++ctr.jet_demote_seed;
                return;
            }
            const EOJac J = eo_jacobian(rp, pc.p);
            const double det = J.E_m * J.O_v - J.E_v * J.O_m;
            std::array<double, 5> dm_dp{}, dv_dp{};
            if (!ode_detail::mv_param_jac(rp, pc.p, dpc, J, det, dm_dp, dv_dp)) {
                ++ctr.jet_demote_seed;
                return;
            }
            sampmv[a][k][0] = rp.m;
            sampmv[a][k][1] = rp.v;
            for (int j = 0; j < 5; ++j) {
                sampmv[a][k][2 + j] = dm_dp[j];
                sampmv[a][k][7 + j] = dv_dp[j];
            }
            ++ctr.jet_seed_krule_evals;
            ++ctr.krule_gc2_evals;
            const VKJacobian vkj = v_times_K_jac(rp.m, rp.v, dm_dp, dv_dp,
                                                 nodeR[k], pf, pc.p, dpc.dp);
            if (!vkj.ok) { ++ctr.jet_demote_seed; return; }
            samp[a][k][0] = vkj.vK;
            for (int j = 0; j < 5; ++j) samp[a][k][1 + j] = vkj.dvK[j];
        }
    }

    for (int a = 0; a < na; ++a) {
        double coef[NS][kOdeJetCols];
        if (!ode_detail::cheb_fit(nodeX, samp[a], 6, coef)) {
            ++ctr.jet_demote_seed;
            return;
        }
        // Gate on the WORST-decaying column: v*K and every d(v*K)/dp_j must
        // be analytic across the cell for the degree-kOdeJetDeg packet to
        // track them.  The 'a' (separation) sensitivity in particular can
        // stay rough where v*K itself looks smooth.
        double drho = 1.0e18;
        for (int col = 0; col < 6; ++col) {
            double cc[NS];
            for (int k = 0; k < NS; ++k) cc[k] = coef[k][col];
            const double r = ode_detail::cheb_decay_rho(cc, A, B, H);
            if (r < drho) drho = r;
        }
        if (std::getenv("HOLO_ODE_JET_DEBUG")) {
            double cv[NS];
            for (int k = 0; k < NS; ++k) cv[k] = coef[k][0];
            const double c0 = std::fabs(cv[0]) + 1e-300;
            std::fprintf(stderr,
                         "JET a=%d drho=%.3g  |c|/c0: %.2e %.2e %.2e %.2e %.2e "
                         "%.2e %.2e %.2e\n",
                         a, drho, std::fabs(cv[1]) / c0, std::fabs(cv[2]) / c0,
                         std::fabs(cv[3]) / c0, std::fabs(cv[4]) / c0,
                         std::fabs(cv[5]) / c0, std::fabs(cv[6]) / c0,
                         std::fabs(cv[7]) / c0, std::fabs(cv[8]) / c0);
        }
        if (!std::isfinite(drho) || drho < holo_ode_jet_decay_min()) {
            ++ctr.jet_demote_decay;
            return;
        }
        JetGerm& g = germs[a];
        g.R_c = anchor;
        g.r_half = H;  // jet spans the half-cell; K-rule beyond (fold tails)
        g.decay_rho = drho;
        {
            double cf_val[NS];
            for (int k = 0; k < NS; ++k) cf_val[k] = coef[k][0];
            ode_detail::cheb_to_taylor(cf_val, A, B, anchor, g.c_val);
        }
        for (int j = 0; j < 5; ++j) {
            double col[NS];
            for (int k = 0; k < NS; ++k) col[k] = coef[k][1 + j];
            ode_detail::cheb_to_taylor(col, A, B, anchor, g.c_dp[j]);
        }
        bool fin = true;
        for (int k = 0; k <= kOdeJetDeg; ++k) {
            if (!std::isfinite(g.c_val[k])) fin = false;
            for (int j = 0; j < 5; ++j)
                if (!std::isfinite(g.c_dp[j][k])) fin = false;
        }
        if (!fin) { ++ctr.jet_demote_seed; return; }
        g.active = true;

        // ---- (m,v) geometry packet (independent of the flux gate) --------
        // A degree-kOdeJetDeg packet must reproduce all 9 seed samples of m,
        // v and every dm/dp, dv/dp.  A root branch mis-tracked between seed
        // nodes -- the one failure a geometry packet must never accept --
        // blows up either the coefficient decay or the truncation residual
        // (|c7| + |c8|, Chebyshev coeffs are <= 1 on the seed span), so both
        // are gated explicitly.  Failure leaves mv_active false; the cell
        // then keeps the ODE march, never a silent bad packet.
        double coefmv[NS][kOdeJetCols];
        if (ode_detail::cheb_fit(nodeX, sampmv[a], 12, coefmv)) {
            const double mv_scale = holo_ode_r2_packet_tail_scale();
            double mv_rho = 1.0e18;
            bool mv_ok = true;
            for (int col = 0; col < 12; ++col) {
                double cc[NS], sc = 1e-300;
                for (int k = 0; k < NS; ++k) {
                    cc[k] = coefmv[k][col];
                    sc = std::max(sc, std::fabs(sampmv[a][k][col]));
                }
                const double resid =
                    std::fabs(coefmv[7][col]) + std::fabs(coefmv[8][col]);
                const double tol = (col < 2 ? 3.0e-5 : 3.0e-3) * mv_scale;
                if (!(resid <= tol * sc)) mv_ok = false;
                const double r = ode_detail::cheb_decay_rho(cc, A, B, H);
                if (r < mv_rho) mv_rho = r;
            }
            if (mv_ok && std::isfinite(mv_rho) &&
                mv_rho >= holo_ode_jet_decay_min()) {
                double cm[NS], cv[NS];
                for (int k = 0; k < NS; ++k) {
                    cm[k] = coefmv[k][0];
                    cv[k] = coefmv[k][1];
                }
                ode_detail::cheb_to_taylor(cm, A, B, anchor, g.c_m);
                ode_detail::cheb_to_taylor(cv, A, B, anchor, g.c_v);
                for (int j = 0; j < 5; ++j) {
                    double dm[NS], dv[NS];
                    for (int k = 0; k < NS; ++k) {
                        dm[k] = coefmv[k][2 + j];
                        dv[k] = coefmv[k][7 + j];
                    }
                    ode_detail::cheb_to_taylor(dm, A, B, anchor, g.c_dm[j]);
                    ode_detail::cheb_to_taylor(dv, A, B, anchor, g.c_dv[j]);
                }
                bool mvfin = true;
                for (int k = 0; k <= kOdeJetDeg; ++k) {
                    if (!std::isfinite(g.c_m[k]) || !std::isfinite(g.c_v[k]))
                        mvfin = false;
                    for (int j = 0; j < 5; ++j)
                        if (!std::isfinite(g.c_dm[j][k]) ||
                            !std::isfinite(g.c_dv[j][k]))
                            mvfin = false;
                }
                if (mvfin) {
                    g.mv_decay_rho = mv_rho;
                    g.mv_active = true;
                }
            }
        }
    }
    for (int a = 0; a < na; ++a)
        if (!germs[a].active) { ++ctr.jet_demote_seed; return; }
    ++ctr.jet_cells;
}

// ---- fold tail: analytic integration of [R_s, R_f] -------------------
// R_f is the D14 event radius (the exact fold birth/death radius).  The
// substitution u = sqrt(|R_f - R|), dR = 2u du, turns the 1/sqrt(v) ~ 1/u
// singularity in the (m,v)->F0 Jacobian integrand into a bounded function
// of u; 3-point Gauss-Legendre in u (interior nodes only, v > 0 always)
// then integrates it.  (m,v) at each node is propagated from R_s by an
// (m,v)-only RK4 -- NO quartic root solve, NO angular sqrt(phi) sweep.
// Returns the UNSIGNED integral over [min(R_s,R_f), max(R_s,R_f)]; the
// caller adds it (both the up and the down tail carry a + sign).
template <bool WantJac>
inline bool ode_fold_tail(double R_s, double R_f, const double* mv_s, int na,
                          const PrimaryFrame& pf, bool want_fh,
                          OdeCounters& ctr, double& t_F0, double& t_Fh,
                          double* t_dF0, double* t_dFh) {
    t_F0 = 0.0;
    t_Fh = 0.0;
    if (WantJac)
        for (int j = 0; j < 5; ++j) { t_dF0[j] = 0.0; t_dFh[j] = 0.0; }
    const double dir = (R_f > R_s) ? 1.0 : -1.0;
    const double u_s = std::sqrt(std::fabs(R_f - R_s));
    if (!(u_s > 0.0)) return true;  // zero-width tail
    ++ctr.fold_tails;

    for (int g = 0; g < kOdeGLn; ++g) {
        const double u_k = 0.5 * u_s * (1.0 + kOdeGLX[g]);
        const double W_k = 0.5 * u_s * kOdeGLW[g];
        const double R_k = R_f - dir * u_k * u_k;

        std::array<double, 2 * kOdeMaxArcs> mv{};
        for (int i = 0; i < 2 * na; ++i) mv[i] = mv_s[i];
        if (!mv_only_march(R_s, R_k, mv.data(), na, pf, ctr)) {
            ++ctr.fb_march_tail;
            return false;
        }

        const QuarticCoeffs pc = boundary_quartic(R_k, pf);
        QuarticParamJac dpc;
        if (WantJac) dpc = boundary_quartic_dp(R_k, pf);

        double g_F0 = 0.0, g_Fh = 0.0;
        double g_dF0[5] = {0, 0, 0, 0, 0}, g_dFh[5] = {0, 0, 0, 0, 0};
        for (int a = 0; a < na; ++a) {
            const RootPair rp{mv[2 * a], mv[2 * a + 1]};
            if (!(rp.v > kOdeTailVFloor) || !std::isfinite(rp.m)) {
                ++ctr.fb_march_tail;
                return false;
            }
            // v*K -> 0 as v -> 0: below the shared near-tangency floor the
            // K-rule is skipped and F_half picks up 0 from this node.
            const bool do_krule = (rp.v > kHoloVFloor);
            const ode_detail::DThetaDeriv dt =
                ode_detail::dtheta_derivs(rp.m, rp.v);
            if (!dt.ok) { ++ctr.fb_march_tail; return false; }
            g_F0 += R_k * dt.val;

            if (WantJac) {
                const EOJac J = eo_jacobian(rp, pc.p);
                const double det = J.E_m * J.O_v - J.E_v * J.O_m;
                std::array<double, 5> dm_dp{}, dv_dp{};
                if (!ode_detail::mv_param_jac(rp, pc.p, dpc, J, det, dm_dp,
                                              dv_dp)) {
                    ++ctr.fb_march_tail;
                    return false;
                }
                for (int j = 0; j < 5; ++j)
                    g_dF0[j] += R_k * (dt.d_m * dm_dp[j] + dt.d_v * dv_dp[j]);
                if (do_krule) {
                    ++ctr.krule_gc2_evals;
                    const VKJacobian vkj = v_times_K_jac(
                        rp.m, rp.v, dm_dp, dv_dp, R_k, pf, pc.p, dpc.dp);
                    if (!vkj.ok) { ++ctr.fb_march_tail; return false; }
                    g_Fh += vkj.vK;
                    for (int j = 0; j < 5; ++j) g_dFh[j] += vkj.dvK[j];
                }
            } else if (want_fh && do_krule) {
                ++ctr.krule_gc2_evals;
                const VKValue vk = v_times_K(rp.m, rp.v, R_k, pf, pc.p);
                if (!vk.ok) { ++ctr.fb_march_tail; return false; }
                g_Fh += vk.vK;
            }
        }

        const double jac = W_k * 2.0 * u_k;  // dR = 2u du
        t_F0 += jac * g_F0;
        t_Fh += jac * g_Fh;
        if (WantJac)
            for (int j = 0; j < 5; ++j) {
                t_dF0[j] += jac * g_dF0[j];
                t_dFh[j] += jac * g_dFh[j];
            }
    }
    return true;
}

// ---- per-cell drivers -------------------------------------------------
struct OdeCellJac {
    double F0 = 0.0, F_half = 0.0;
    std::array<double, 5> dF0i{}, dFhi{};
    bool reliable = true;
};
struct OdeCellValue {
    double F0 = 0.0, F_half = 0.0;
    bool reliable = true;
};

namespace ode_detail {
// dist_fold router (spike (c) regime split) -- diagnostic only for now.
inline void route_regimes(const PrimaryFrame& pf, const OdeSeed& sd,
                          double anchor, double H, OdeCounters& ctr, bool* ok,
                          bool* jet_eligible) {
    *ok = true;
    *jet_eligible = true;
    const QuarticCoeffs pc = boundary_quartic(anchor, pf);
    const QuarticCoeffs pcR = boundary_quartic_dR(anchor, pf);
    for (int a = 0; a < sd.na; ++a) {
        const RootPair rp{sd.mv[2 * a], sd.mv[2 * a + 1]};
        const RootPairDR d = root_pair_dR(rp, pc.p, pcR.p);
        if (!d.ok) {
            ++ctr.fb_route_det;
            *ok = false;
            return;
        }
        const double dist =
            (d.dv_dR != 0.0) ? std::fabs(rp.v / d.dv_dR) : 1e300;
        if (dist < 2.0 * H) {
            ++ctr.regime1;
            *jet_eligible = false;
        } else {
            ++ctr.regime2;
        }
    }
}
}  // namespace ode_detail

// ======================================================================
// Aggressive regime-2 cell rule.  When route_regimes reports every arc in
// the cell regime-2 (fold out of reach, dist_fold >= 2H), the integrands
//   col 0      = R * delta_theta(m,v)                      -> F0
//   col 1..5   = R * (dtheta_m dm/dp_j + dtheta_v dv/dp_j) -> dF0/dp_j
//   col 6      = v*K                                       -> F_half
//   col 7..11  = d(v*K)/dp_j                               -> dF_half/dp_j
// are each smooth across the WHOLE cell.  We sample a fixed kOdeR2GLn-point
// Gauss-Legendre rule over [lo,hi] -- (m,v) at each node reached by an
// (m,v)-only RK4 from the cell-centre seed, NO quartic re-solve -- and take
// the cell integral straight from the GL weights.  No RK4 flux march, no
// fold tail: the whole cell closes on one fixed rule.
//
// Safety: the GL-node samples of every column are also fitted to a
// degree-(kOdeR2GLn-1) Chebyshev interpolant; if the two top coefficients
// are not down by kOdeR2TailTol (value columns) / *100 (derivative columns)
// relative to c0 the fixed rule is not resolving that integrand and the
// cell is demoted to the flux-priority jet + march.  Parity-immune (a top-
// coefficient magnitude test, not a decay slope).  Never a silent approx.
namespace ode_detail {

struct R2Result {
    int na = 0;
    double F0 = 0.0, F_half = 0.0;
    double dF0[5] = {}, dFh[5] = {};
    bool ok = false;
};

// Gauss 7 / Kronrod 15 on [-1, 1].  The 15 Kronrod abscissae kOdeR2X[] give
// the cell integral; the 7 Gauss abscissae are the subset at indices 1,3,..,13
// and kOdeR2GW[] are their weights (kOdeR2GW[i] pairs with kOdeR2X[2i+1]).
// |I_K15 - I_G7| is the standard embedded error estimate -- a real per-cell
// convergence check for the fixed rule, not a decay heuristic.
constexpr int kOdeR2N = 15;
constexpr double kOdeR2X[15] = {
    -0.9914553711208126, -0.9491079123427585, -0.8648644233597691,
    -0.7415311855993945, -0.5860872354676911, -0.4058451513773972,
    -0.2077849550078985,  0.0,                 0.2077849550078985,
     0.4058451513773972,  0.5860872354676911,  0.7415311855993945,
     0.8648644233597691,  0.9491079123427585,  0.9914553711208126};
constexpr double kOdeR2KW[15] = {
    0.0229353220105292, 0.0630920926299786, 0.1047900103222502,
    0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
    0.2044329400752989, 0.2094821410847278, 0.2044329400752989,
    0.1903505780647854, 0.1690047266392679, 0.1406532597155259,
    0.1047900103222502, 0.0630920926299786, 0.0229353220105292};
constexpr double kOdeR2GW[7] = {
    0.1294849661688697, 0.2797053914892766, 0.3818300505051189,
    0.4179591836734694, 0.3818300505051189, 0.2797053914892766,
    0.1294849661688697};

// Analytic integral of a degree-kOdeJetDeg Taylor packet (coeffs c[k] about
// R_c, c[k] = f^(k)(R_c)/k!) over [lo, hi]:  int f dR = sum_k c[k]/(k+1) *
// ((hi-R_c)^{k+1} - (lo-R_c)^{k+1}).  The jet germ is valid over the whole
// cell (g.R_c = anchor, g.r_half = H), so F_half and d(F_half)/dp on a
// regime-2 cell close in closed form -- no K-rule sampling at all.
inline double jet_integrate(const double* c, double R_c, double lo, double hi) {
    const double a = lo - R_c, b = hi - R_c;
    double ap = a, bp = b, acc = 0.0;  // ap = a^{k+1}, bp = b^{k+1}
    for (int k = 0; k <= kOdeJetDeg; ++k) {
        acc += c[k] / (double)(k + 1) * (bp - ap);
        ap *= a;
        bp *= b;
    }
    return acc;
}

// Composite Kronrod-15 of sum_a R * delta_theta(m_a(R), v_a(R)) over `np`
// equal panels of [lo,hi], with (m,v) from the geometry packet (Horner, no
// march).  delta_theta = 2 atan2(2 sqrt(v), .) carries a ~1/sqrt(v) bend
// that even an exact (m,v) does not remove, so the fixed rule is checked by
// panel refinement (np=1 vs np=2) in the caller.  false if any packet node
// leaves the valid (v > 0) sheet.
inline bool r2_jet_f0(double lo, double hi, int na, const JetGerm* germs,
                      int np, double* F0) {
    const double pw = (hi - lo) / (double)np;
    double acc = 0.0;
    for (int p = 0; p < np; ++p) {
        const double pmid = lo + (p + 0.5) * pw, phlf = 0.5 * pw;
        for (int k = 0; k < kOdeR2N; ++k) {
            const double R = pmid + phlf * kOdeR2X[k];
            double f = 0.0;
            for (int a = 0; a < na; ++a) {
                const double m = jet_eval(germs[a].c_m, germs[a].R_c, R);
                const double v = jet_eval(germs[a].c_v, germs[a].R_c, R);
                if (!(v > kHoloVFloor) || !std::isfinite(m)) return false;
                const DThetaDeriv dt = dtheta_derivs(m, v);
                if (!dt.ok) return false;
                f += R * dt.val;
            }
            acc += phlf * kOdeR2KW[k] * f;
        }
    }
    *F0 = acc;
    return true;
}

// Value-lane regime-2 packet: (m,v) from the geometry packet (Horner, no
// march), F0 by panel-refined composite K15 gated on the np=1 vs np=2 delta,
// F_half by the analytic v*K jet integral.  No RK4 (m,v) march, no flux
// march, no fold tail, no K-rule sampling -- the cell closes on packet
// coefficients.
inline R2Result r2_value_cell(double lo, double hi, const OdeSeed& sd,
                              const JetGerm* germs, bool want_fh,
                              OdeCounters& ctr) {
    R2Result res;
    const int na = sd.na;
    for (int a = 0; a < na; ++a)
        if (!germs[a].mv_active) { ++ctr.r2_packet_demote; return res; }

    double F0_1 = 0.0, F0_2 = 0.0;
    if (!r2_jet_f0(lo, hi, na, germs, 1, &F0_1) ||
        !r2_jet_f0(lo, hi, na, germs, 2, &F0_2)) {
        ++ctr.r2_packet_seed_fail;
        return res;
    }

    const double tol_f0 = 2.0e-4 * holo_ode_r2_packet_tail_scale();
    const bool dbg = std::getenv("HOLO_ODE_R2_DEBUG") != nullptr;
    const double err = std::fabs(F0_2 - F0_1) / (std::fabs(F0_2) + 1e-300);
    if (dbg)
        std::fprintf(stderr, "R2val F0 err=%.2e tol=%.0e%s\n", err, tol_f0,
                     (err < tol_f0) ? "" : "  X");
    if ((!(err < tol_f0) || !std::isfinite(err)) && !dbg) {
        ++ctr.r2_packet_demote;
        return res;
    }
    res.F0 = F0_2;

    if (want_fh) {
        double Fh = 0.0;
        for (int a = 0; a < na; ++a)
            Fh += jet_integrate(germs[a].c_val, germs[a].R_c, lo, hi);
        if (!std::isfinite(Fh)) return res;
        res.F_half = Fh;
    }
    if (!std::isfinite(res.F0)) return res;
    res.na = na;
    res.ok = true;
    return res;
}

// jac-lane regime-2: F0 and its 5 parameter-derivatives by composite K15
// over `np` panels, everything from the geometry packet by Horner.  The
// dF0/dp_j integrand R*(dtheta_m*dm/dp + dtheta_v*dv/dp) carries dtheta_v ~
// 1/sqrt(v); in a regime-2 cell v can still dip to ~0.01 at the fold-facing
// edge, so v(R)^{-1/2} has a complex branch point ~0.14 in R from the cell
// centre and composite Gauss-Kronrod converges only ~np^{-1/2} on dF0/dp
// (checkpoint 37 / evidence sec.9 -- this is why r2_jac_cell demotes ~90 %
// of regime-2 jac cells and the V3 line is frozen as reference).  The np=1
// vs np=2 refinement gate in r2_jac_cell catches exactly that.
// out[0] = F0, out[1..5] = dF0/dp_j.  false on a bad packet node.
inline bool r2_jet_f0_jac(double lo, double hi, int na, const JetGerm* germs,
                          int np, double* out) {
    for (int c = 0; c < 6; ++c) out[c] = 0.0;
    const double pw = (hi - lo) / (double)np;
    for (int p = 0; p < np; ++p) {
        const double pmid = lo + (p + 0.5) * pw, phlf = 0.5 * pw;
        for (int k = 0; k < kOdeR2N; ++k) {
            const double R = pmid + phlf * kOdeR2X[k];
            const double w = phlf * kOdeR2KW[k];
            double f0 = 0.0, fd[5] = {};
            for (int a = 0; a < na; ++a) {
                const double m = jet_eval(germs[a].c_m, germs[a].R_c, R);
                const double v = jet_eval(germs[a].c_v, germs[a].R_c, R);
                if (!(v > kHoloVFloor) || !std::isfinite(m)) return false;
                const DThetaDeriv dt = dtheta_derivs(m, v);
                if (!dt.ok) return false;
                f0 += R * dt.val;
                for (int j = 0; j < 5; ++j) {
                    const double dm =
                        jet_eval(germs[a].c_dm[j], germs[a].R_c, R);
                    const double dv =
                        jet_eval(germs[a].c_dv[j], germs[a].R_c, R);
                    fd[j] += R * (dt.d_m * dm + dt.d_v * dv);
                }
            }
            out[0] += w * f0;
            for (int j = 0; j < 5; ++j) out[1 + j] += w * fd[j];
        }
    }
    return true;
}

// jac-lane regime-2 packet: F_half + d(F_half)/dp by the analytic v*K jet
// integral, F0 + dF0/dp by panel-refined K15 of the geometry packet.  No
// RK4 march of any kind -- the (m,v) transport that dominated the jac cell
// cost (root_pair_dR + eo_jacobian + IFT per substep) is Horner now.
inline R2Result r2_jac_cell(double lo, double hi, const OdeSeed& sd,
                            const JetGerm* germs, OdeCounters& ctr) {
    R2Result res;
    const int na = sd.na;
    for (int a = 0; a < na; ++a)
        if (!germs[a].mv_active || !germs[a].active) {
            ++ctr.r2_jac_demote;
            return res;
        }

    double c1[6], c2[6];
    if (!r2_jet_f0_jac(lo, hi, na, germs, 1, c1) ||
        !r2_jet_f0_jac(lo, hi, na, germs, 2, c2)) {
        ++ctr.r2_jac_demote;
        return res;
    }

    const double sc = holo_ode_r2_packet_tail_scale();
    const double tol_f0 = 2.0e-4 * sc;
    const double tol_d = 5.0e-3 * sc;
    const bool dbg = std::getenv("HOLO_ODE_R2_DEBUG") != nullptr;
    const double eF0 = std::fabs(c2[0] - c1[0]) / (std::fabs(c2[0]) + 1e-300);
    double eDmax = 0.0;
    for (int j = 0; j < 5; ++j) {
        const double den =
            std::fabs(c2[1 + j]) + 1.0e-3 * std::fabs(c2[0]) + 1e-300;
        const double e = std::fabs(c2[1 + j] - c1[1 + j]) / den;
        if (e > eDmax) eDmax = e;
    }
    if (dbg)
        std::fprintf(stderr, "R2jac F0 e=%.2e (tol %.0e)  dF0 e=%.2e (tol %.0e)%s\n",
                     eF0, tol_f0, eDmax, tol_d,
                     (eF0 < tol_f0 && eDmax < tol_d) ? "" : "  X");
    if (((!(eF0 < tol_f0) || !std::isfinite(eF0)) ||
         (!(eDmax < tol_d) || !std::isfinite(eDmax))) &&
        !dbg) {
        ++ctr.r2_jac_demote;
        return res;
    }

    res.F0 = c2[0];
    for (int j = 0; j < 5; ++j) res.dF0[j] = c2[1 + j];

    double Fh = 0.0, dFh[5] = {};
    for (int a = 0; a < na; ++a) {
        Fh += jet_integrate(germs[a].c_val, germs[a].R_c, lo, hi);
        for (int j = 0; j < 5; ++j)
            dFh[j] += jet_integrate(germs[a].c_dp[j], germs[a].R_c, lo, hi);
    }
    res.F_half = Fh;
    for (int j = 0; j < 5; ++j) res.dFh[j] = dFh[j];

    if (!std::isfinite(res.F0) || !std::isfinite(res.F_half)) {
        ++ctr.r2_jac_demote;
        return res;
    }
    for (int j = 0; j < 5; ++j)
        if (!std::isfinite(res.dF0[j]) || !std::isfinite(res.dFh[j])) {
            ++ctr.r2_jac_demote;
            return res;
        }

    res.na = na;
    res.ok = true;
    return res;
}

}  // namespace ode_detail

inline bool ode_transport_cell_jac(const PrimaryFrame& pf, double lo, double hi,
                                   double H, OdeCellJac& out) {
    OdeCounters& ctr = ode_counters();
    ++ctr.cells_seen;
    const double anchor = 0.5 * (lo + hi);

    const OdeSeed sd = ode_seed_anchor(pf, anchor, ctr);
    if (!sd.ok) {
        ++ctr.cells_fallback;
        return false;
    }
    bool route_ok = false, jet_elig = false;
    ode_detail::route_regimes(pf, sd, anchor, H, ctr, &route_ok, &jet_elig);
    if (!route_ok) {
        ++ctr.cells_fallback;
        return false;
    }

    // regime-2 flux-priority jet: build the per-arc Taylor germ once here;
    // if every arc's germ clears the decay gate the marches below evaluate
    // v*K + d(v*K)/dp by Horner instead of a per-stage 16-node K-rule.  The
    // same build also fits the (m,v) geometry packet (c_m/c_v/c_dm/c_dv);
    // when it clears its own fidelity gate on every arc, r2_jac_cell closes
    // the whole cell -- F0/dF0 by panel-refined K15 of the packet, F_half/
    // dFh by the analytic v*K jet integral -- with no RK4 march at all.
    JetGerm germs[kOdeMaxArcs];
    bool jet_on = false, mv_on = false;
    if (jet_elig) {
        ++ctr.jet_eligible_cells;
        ode_jet_build(pf, anchor, H, sd, ctr, germs);
        jet_on = true;
        mv_on = true;
        for (int a = 0; a < sd.na; ++a) {
            jet_on = jet_on && germs[a].active;
            mv_on = mv_on && germs[a].mv_active;
        }
    }
    const JetGerm* jets = jet_on ? germs : nullptr;

    if (jet_elig && holo_ode_r2_packet_enabled() && mv_on && jet_on) {
        const ode_detail::R2Result r =
            ode_detail::r2_jac_cell(lo, hi, sd, germs, ctr);
        if (r.ok) {
            out.F0 = r.F0;
            out.F_half = r.F_half;
            for (int j = 0; j < 5; ++j) {
                out.dF0i[j] = r.dF0[j];
                out.dFhi[j] = r.dFh[j];
            }
            out.reliable = true;
            ++ctr.r2_jac_cells;
            ++ctr.cells_ode;
            return true;
        }
        // fall through: flux-priority jet + march
    }

    const int na = sd.na;
    const int ND = 2 * na + 12;
    std::array<double, kOdeStateDim> y{};

    auto seed_state = [&] {
        for (int a = 0; a < na; ++a) {
            y[2 * a] = sd.mv[2 * a];
            y[2 * a + 1] = sd.mv[2 * a + 1];
        }
        for (int i = 2 * na; i < ND; ++i) y[i] = 0.0;
    };

    // up: anchor -> hi  (accumulators hold int_anchor^{R_end}; a fold edge
    // stops the march early -> [R_end, hi] is done by ode_fold_tail).
    seed_state();
    double R_end_up = hi;
    int fa_up = -1;
    if (!ode_march<true>(anchor, hi, y.data(), na, pf, true, H, jets, ctr,
                         &R_end_up, &fa_up)) {
        ++ctr.cells_fallback;
        return false;
    }
    // a regime-2 jet cell can still reach a v-work fold edge mid-march: the
    // router's linear dist_fold estimate is optimistic. The jet germ covers
    // [anchor, R_end] (bounded by r_half); ode_fold_tail runs the K-rule on
    // [R_end, edge]. No full fallback needed.
    const double F0_up = y[2 * na + 0], Fh_up = y[2 * na + 1];
    std::array<double, 5> dF0_up{}, dFh_up{};
    for (int j = 0; j < 5; ++j) {
        dF0_up[j] = y[2 * na + 2 + j];
        dFh_up[j] = y[2 * na + 7 + j];
    }
    double tU_F0 = 0.0, tU_Fh = 0.0;
    std::array<double, 5> tU_dF0{}, tU_dFh{};
    if (fa_up >= 0) {
        std::array<double, 2 * kOdeMaxArcs> mv_s{};
        for (int i = 0; i < 2 * na; ++i) mv_s[i] = y[i];
        if (!ode_fold_tail<true>(R_end_up, hi, mv_s.data(), na, pf, true, ctr,
                                 tU_F0, tU_Fh, tU_dF0.data(), tU_dFh.data())) {
            ++ctr.cells_fallback;
            return false;
        }
    }

    // down: anchor -> lo (accumulators hold int_anchor^{R_end} = -int; a
    // fold edge at lo stops early -> [lo, R_end] is done by ode_fold_tail).
    seed_state();
    double R_end_dn = lo;
    int fa_dn = -1;
    if (!ode_march<true>(anchor, lo, y.data(), na, pf, true, H, jets, ctr,
                         &R_end_dn, &fa_dn)) {
        ++ctr.cells_fallback;
        return false;
    }
    double tD_F0 = 0.0, tD_Fh = 0.0;
    std::array<double, 5> tD_dF0{}, tD_dFh{};
    if (fa_dn >= 0) {
        std::array<double, 2 * kOdeMaxArcs> mv_s{};
        for (int i = 0; i < 2 * na; ++i) mv_s[i] = y[i];
        if (!ode_fold_tail<true>(R_end_dn, lo, mv_s.data(), na, pf, true, ctr,
                                 tD_F0, tD_Fh, tD_dF0.data(), tD_dFh.data())) {
            ++ctr.cells_fallback;
            return false;
        }
    }

    // int_lo^hi = (int_anchor^{R_end_up} - int_anchor^{R_end_dn})
    //           + tail[R_end_up, hi] + tail[lo, R_end_dn]   (both tails +).
    out.F0 = (F0_up - y[2 * na + 0]) + tU_F0 + tD_F0;
    out.F_half = (Fh_up - y[2 * na + 1]) + tU_Fh + tD_Fh;
    for (int j = 0; j < 5; ++j) {
        out.dF0i[j] = (dF0_up[j] - y[2 * na + 2 + j]) + tU_dF0[j] + tD_dF0[j];
        out.dFhi[j] = (dFh_up[j] - y[2 * na + 7 + j]) + tU_dFh[j] + tD_dFh[j];
    }
    if (!std::isfinite(out.F0) || !std::isfinite(out.F_half)) {
        ++ctr.fb_nonfinite;
        ++ctr.cells_fallback;
        return false;
    }
    for (int j = 0; j < 5; ++j)
        if (!std::isfinite(out.dF0i[j]) || !std::isfinite(out.dFhi[j])) {
            ++ctr.fb_nonfinite;
            ++ctr.cells_fallback;
            return false;
        }
    out.reliable = true;  // any unreliability already forced a fallback
    ++ctr.cells_ode;
    return true;
}

inline bool ode_transport_cell_value(const PrimaryFrame& pf, double lo, double hi,
                                     double H, bool want_fh, OdeCellValue& out) {
    OdeCounters& ctr = ode_counters();
    ++ctr.cells_seen;
    const double anchor = 0.5 * (lo + hi);

    const OdeSeed sd = ode_seed_anchor(pf, anchor, ctr);
    if (!sd.ok) {
        ++ctr.cells_fallback;
        return false;
    }
    bool route_ok = false, jet_elig = false;
    ode_detail::route_regimes(pf, sd, anchor, H, ctr, &route_ok, &jet_elig);
    if (!route_ok) {
        ++ctr.cells_fallback;
        return false;
    }

    JetGerm germs[kOdeMaxArcs];
    bool jet_on = false, mv_on = false;
    if (jet_elig) {
        ++ctr.jet_eligible_cells;
        ode_jet_build(pf, anchor, H, sd, ctr, germs);
        jet_on = true;
        mv_on = true;
        for (int a = 0; a < sd.na; ++a) {
            jet_on = jet_on && germs[a].active;
            mv_on = mv_on && germs[a].mv_active;
        }
    }
    const JetGerm* jets = jet_on ? germs : nullptr;

    // aggressive regime-2 packet (value lane): (m,v) reconstructed by Horner
    // from the geometry packet -- no march at all -- F0 by the G7/K15 rule,
    // F_half by the analytic v*K jet integral.  want_fh cells also need the
    // flux packet (jet_on); every cell needs the geometry packet (mv_on).
    if (jet_elig && holo_ode_r2_packet_enabled() && mv_on &&
        (!want_fh || jet_on)) {
        const ode_detail::R2Result r =
            ode_detail::r2_value_cell(lo, hi, sd, germs, want_fh, ctr);
        if (r.ok) {
            out.F0 = r.F0;
            out.F_half = want_fh ? r.F_half : 0.0;
            out.reliable = true;
            ++ctr.r2_packet_cells;
            ++ctr.cells_ode;
            return true;
        }
        // fall through: flux-priority jet + march
    }

    const int na = sd.na;
    const int ND = 2 * na + 2;
    std::array<double, kOdeStateDim> y{};

    auto seed_state = [&] {
        for (int a = 0; a < na; ++a) {
            y[2 * a] = sd.mv[2 * a];
            y[2 * a + 1] = sd.mv[2 * a + 1];
        }
        for (int i = 2 * na; i < ND; ++i) y[i] = 0.0;
    };

    seed_state();
    double R_end_up = hi;
    int fa_up = -1;
    if (!ode_march<false>(anchor, hi, y.data(), na, pf, want_fh, H, jets, ctr,
                          &R_end_up, &fa_up)) {
        ++ctr.cells_fallback;
        return false;
    }
    const double F0_up = y[2 * na + 0], Fh_up = y[2 * na + 1];
    double tU_F0 = 0.0, tU_Fh = 0.0, tU_junk[5];
    if (fa_up >= 0) {
        std::array<double, 2 * kOdeMaxArcs> mv_s{};
        for (int i = 0; i < 2 * na; ++i) mv_s[i] = y[i];
        if (!ode_fold_tail<false>(R_end_up, hi, mv_s.data(), na, pf, want_fh,
                                  ctr, tU_F0, tU_Fh, tU_junk, tU_junk)) {
            ++ctr.cells_fallback;
            return false;
        }
    }

    seed_state();
    double R_end_dn = lo;
    int fa_dn = -1;
    if (!ode_march<false>(anchor, lo, y.data(), na, pf, want_fh, H, jets, ctr,
                          &R_end_dn, &fa_dn)) {
        ++ctr.cells_fallback;
        return false;
    }
    double tD_F0 = 0.0, tD_Fh = 0.0, tD_junk[5];
    if (fa_dn >= 0) {
        std::array<double, 2 * kOdeMaxArcs> mv_s{};
        for (int i = 0; i < 2 * na; ++i) mv_s[i] = y[i];
        if (!ode_fold_tail<false>(R_end_dn, lo, mv_s.data(), na, pf, want_fh,
                                  ctr, tD_F0, tD_Fh, tD_junk, tD_junk)) {
            ++ctr.cells_fallback;
            return false;
        }
    }

    out.F0 = (F0_up - y[2 * na + 0]) + tU_F0 + tD_F0;
    out.F_half = (Fh_up - y[2 * na + 1]) + tU_Fh + tD_Fh;
    if (!std::isfinite(out.F0) || !std::isfinite(out.F_half)) {
        ++ctr.fb_nonfinite;
        ++ctr.cells_fallback;
        return false;
    }
    out.reliable = true;
    ++ctr.cells_ode;
    return true;
}

}  // namespace lcbinint::holonomic
