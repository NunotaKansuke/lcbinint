#pragma once

// ATPT holonomic solver -- fast-planner local confidence screen.
//
// Phase A step 3 (unified-engine implementation order, checkpoint sec.0.7 /
// sec.5.2).  The seed-anchored fast planner (fast_bands.hpp) is a HEURISTIC
// discovery path.  This screen raises confidence that the planner's
// SIGNIFICANT band set (arcs wide enough to move mu at the M7 reference
// tolerance) is complete and well-conditioned.  It is NOT a completeness
// proof (checkpoint sec.5.4/5.5): it governs only WHETHER the D14 oracle
// (classify_cells / radial_events, the completeness authority) is invoked,
// never whether completeness holds.  Any epoch whose screen does not pass
// goes to the oracle; any epoch the oracle cannot resolve fails closed.
//
// Four checks, all of which must pass.  They deviate from the checkpoint
// sec.5.2 sketch where the available primitives forced it (noted per check):
//
//   1. Partition validity.  No image-free sub-region hides inside a
//      significant band: quartic_topology at seven interior fractions of
//      each significant band is non-empty everywhere (kFull, or kArcs with a
//      positive even crossing count) -- the fast planner must not have
//      bridged a real kEmpty gap.  Deviation from sec.5.2 ("uniform in-band
//      topology x3"): the crossing COUNT is allowed to vary within the band
//      (an arc splitting 2->4->2 is a source straddling a caustic fold, not
//      a gap; the moment integrator re-derives the arc set at every
//      quadrature radius, so it needs only valid band bounds).  The test is
//      "no interior kEmpty", not "uniform topology".
//
//   2. Tangency regularity.  A band endpoint is a fold: phi = phi_theta = 0
//      there, so the endpoint's own |phi_theta| is ~0 by construction and is
//      not a usable margin.  Deviation from sec.5.2 ("det dG/d(R,theta) =
//      phi_R phi_thetatheta"): phi.hpp exposes no second derivatives, so the
//      metric is the scale-free equivalent -- whether the fold's arc
//      half-width grows like sqrt(dR) into the band.  For the fold normal
//      form  phi ~ phi_R (R-R*) + 1/2 phi_thetatheta (theta-theta*)^2  the
//      half-width is  hw(dR) = sqrt(phi_R / |phi_thetatheta|) * sqrt(dR), so
//      hw(4d)/hw(d) = 2 for a regular fold.  Ratio near 4^(1/4)=1.41 =>
//      higher-order contact (phi_thetatheta -> 0, cusp-like); ratio well
//      above 2 => edge stationary in R (phi_R -> 0, two bands merging).
//      Needs only arc_intervals.
//
//   3. Seed coverage (probe-free missing-band guard).  The source-centre
//      point lies inside the source disk, so each binary point-source image
//      z_i lies inside the finite-source image region; the circle |z| =
//      |z_i| then meets that region, so |z_i| falls inside SOME band.  Every
//      reliable point-source seed radius whose own local arc is significant
//      must therefore land inside one of the planner's bands.  A seed
//      covered by no band means the planner dropped that seed's band -> fail
//      to the oracle.  This is the primary missing-band guard; it is
//      O(seeds * bands) and spends one arc_intervals call per seed.
//      Deviation from sec.5.2 ("up/down march band-count agreement"): a
//      blind independent radial recount cannot cheaply match the
//      seed-anchored planner's resolution (traces on rand000 / on-axis-off);
//      the necessary condition "every image radius is covered" is both
//      cheaper and stronger against the failure that matters (a dropped
//      band).
//
//   4. Complement scan (runs unconditionally).  A band with no point-source
//      seed can only form where the source disk straddles a caustic fold.
//      Every inter-band gap wider than 2 * the edge inset, plus the
//      stretches below the innermost and above the outermost significant
//      band, are swept and must be image-free.  Gaps the planner already
//      resolved (narrower than the inset) are skipped: a newborn band cannot
//      hide in a sub-0.3-rho gap.  This check ran only when a straddle was
//      indicated in an earlier revision; fault injection (dbg_seedless)
//      showed the gate let adversarial seed-less-band drops through, so it
//      is now always on.  Even always-on it is a coarse linearly-spaced
//      sweep and CANNOT certify a thin fully-seed-less band cold
//      (checkpoint sec.5.4 residual): that is handled by D14-as-authority in
//      the Phase A step 4 wiring (oracle on the first epoch of a trajectory
//      / low-rate audit), not here.
//
// Razor bands (checkpoint sec.15.1: sub-3072-grid arcs classify_cells drops)
// are NOT screened.  Fast-accepting a wrong razor band moves mu by < 1e-7
// relative, below the oracle's own reference tolerance.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "lcbinint/magnification/holonomic/fast_bands.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/point_images.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"

namespace lcbinint::holonomic {

struct FastBandsScreen {
    bool pass = true;
    const char* reason = "";
    int probe_calls = 0;  // arc_intervals / quartic_topology evaluations spent

    // diagnostics (populated regardless of pass/fail)
    int n_significant = 0;      // significant bands in the planner's set
    int down_significant = -1;  // n_significant if the complement scan was
                                // clean; -1 if it failed
    unsigned failed_mask = 0;   // bit0 check1 (partition), bit1 check2 (fold),
                                // bit2 check3 (seed coverage),
                                // bit3 check4 (complement scan)
    double worst_fold = 2.0;    // fold ratio furthest from 2.0 over all endpoints
};

namespace fast_bands_screen_detail {

using fast_bands_detail::has_image;
using fast_bands_detail::Img;

constexpr double kPi2 = 6.283185307179586476925286766559;

// --- screen tunables ------------------------------------------------------
constexpr double kRazorAng = 3.0e-3;  // significance floor (matches bench)
constexpr double kFoldLo = 1.55;      // fold-ratio pass window: below ~1.41 is
constexpr double kFoldHi = 2.35;      // cusp-like, above ~2 is band-merging;
                                     // measured good-epoch range 1.66..2.10
constexpr double kStepFrac = 0.30;    // complement-sweep probe spacing = kStepFrac * rho

// widest phi>0 arc (rad) at radius R; 2pi for the full circle
inline double max_arc_at(double R, const PrimaryFrame& pf, int* calls) {
    ++*calls;
    ArcSet as = arc_intervals(R, pf);
    if (as.kind == ArcKind::kFull) return kPi2;
    if (as.kind != ArcKind::kArcs) return 0.0;
    double w = 0.0;
    for (const auto& a : as.arcs) {
        double d = a[1] - a[0];
        if (d < 0.0) d += kPi2;
        w = std::max(w, d);
    }
    return w;
}

// significance test consistent with bench_fast_bands::is_significant
inline bool band_significant(const FastBand& b, const PrimaryFrame& pf,
                             int* calls) {
    for (double f : {0.25, 0.5, 0.75}) {
        const double R = b.r_lo + f * (b.r_hi - b.r_lo);
        if (max_arc_at(R, pf, calls) >= kRazorAng) return true;
    }
    return false;
}

// narrowest arc + its centre at radius R.  returns <0 on degenerate / empty.
inline double min_arc_at(double R, const PrimaryFrame& pf, int* calls,
                         double* centre) {
    ++*calls;
    ArcSet as = arc_intervals(R, pf);
    if (as.kind == ArcKind::kFull) {
        *centre = 0.0;
        return kPi2;
    }
    if (as.kind != ArcKind::kArcs || as.arcs.empty()) return -1.0;
    double w = std::numeric_limits<double>::infinity(), c = 0.0;
    for (const auto& a : as.arcs) {
        double lo = a[0], hi = a[1];
        if (hi < lo) hi += kPi2;
        if (hi - lo < w) {
            w = hi - lo;
            c = 0.5 * (lo + hi);
        }
    }
    *centre = c;
    return w;
}

// width of the arc whose centre is nearest theta_c at radius R
inline double arc_near(double R, const PrimaryFrame& pf, double theta_c,
                       int* calls) {
    ++*calls;
    ArcSet as = arc_intervals(R, pf);
    if (as.kind == ArcKind::kFull) return kPi2;
    if (as.kind != ArcKind::kArcs || as.arcs.empty()) return -1.0;
    double best = std::numeric_limits<double>::infinity(), bw = -1.0;
    for (const auto& a : as.arcs) {
        double lo = a[0], hi = a[1];
        if (hi < lo) hi += kPi2;
        const double c = 0.5 * (lo + hi);
        const double dc = std::fabs(std::remainder(c - theta_c, kPi2));
        if (dc < best) {
            best = dc;
            bw = hi - lo;
        }
    }
    return bw;
}

// fold half-width ratio hw(4d)/hw(d) for the endpoint at R_edge; the band
// interior is on the +sgn side.  ~2 for a regular fold.  Returns 2.0 when the
// band is too narrow to probe the fold interior (already significance-gated),
// -1 when a probe landed on a degenerate radius.
inline double fold_ratio(double R_edge, double sgn, double W,
                         const PrimaryFrame& pf, int* calls) {
    const double d1 = std::max(1e-9 * (1.0 + std::fabs(R_edge)), 0.02 * W);
    if (5.0 * d1 >= W) return 2.0;
    const double d2 = 4.0 * d1;
    double c1 = 0.0;
    const double w1 = min_arc_at(R_edge + sgn * d1, pf, calls, &c1);
    if (!(w1 > 0.0)) return -1.0;
    const double w2 = arc_near(R_edge + sgn * d2, pf, c1, calls);
    if (!(w2 > 0.0)) return -1.0;
    return w2 / w1;
}

// Sweep one radial interval [lo, hi] for any image-bearing radius.
// kYes   -> 1 (an unlisted band is present)
// kNo    -> 0 (swept clean)
// kUncertain / hi <= lo -> -1
inline int sweep_clear(double lo, double hi, int n, const PrimaryFrame& pf,
                       int* calls) {
    if (!(hi > lo) || n < 1) return 0;
    for (int k = 1; k <= n; ++k) {
        const double t = (double)k / (double)(n + 1);
        const double R = lo + t * (hi - lo);
        const Img im = has_image(R, pf, calls);
        if (im == Img::kUncertain) return -1;
        if (im == Img::kYes) return 1;
    }
    return 0;
}

// Complement scan: is there an image anywhere OUTSIDE the significant bands,
// within (r_floor, r_top)?  Sweeps every inter-band gap wider than 2*inset,
// the stretch below the innermost band, and the stretch above the outermost.
//  1 -> an unlisted band exists (fail to the oracle)
//  0 -> complement is image-free (planner band count is complete)
// -1 -> an uncertain probe (fail to the oracle)
inline int complement_has_extra_band(const PrimaryFrame& pf,
                                     const std::vector<FastBand>& sig,
                                     double r_top, double r_floor, int* calls) {
    const double rho = pf.rho;
    // below the innermost band
    {
        const double w0 = sig.front().r_hi - sig.front().r_lo;
        const double inset = std::max(2e-2 * w0, 0.15 * rho);
        int r = sweep_clear(r_floor, sig.front().r_lo - inset, 20, pf, calls);
        if (r != 0) return r;
    }
    // inter-band gaps
    for (size_t i = 1; i < sig.size(); ++i) {
        const double lo_e = sig[i - 1].r_hi, hi_e = sig[i].r_lo;
        const double wl = sig[i - 1].r_hi - sig[i - 1].r_lo;
        const double wr = sig[i].r_hi - sig[i].r_lo;
        const double inset =
            std::max(2e-2 * std::min(wl, wr), 0.15 * rho);
        const double a = lo_e + inset, b = hi_e - inset;
        if (!(b > a)) continue;  // gap the planner already resolved; too
                                 // narrow for a newborn band to hide in
        const int n = std::min(24, 6 + (int)((b - a) / (kStepFrac * rho)));
        int r = sweep_clear(a, b, n, pf, calls);
        if (r != 0) return r;
    }
    // above the outermost band
    {
        const double wN = sig.back().r_hi - sig.back().r_lo;
        const double inset = std::max(2e-2 * wN, 0.15 * rho);
        int r = sweep_clear(sig.back().r_hi + inset, r_top, 8, pf, calls);
        if (r != 0) return r;
    }
    return 0;
}

}  // namespace fast_bands_screen_detail

inline FastBandsScreen fast_bands_screen(const PrimaryFrame& pf,
                                         const FastBands& fb,
                                         const PointImages& seeds) {
    using namespace fast_bands_screen_detail;
    FastBandsScreen scr;
    int* calls = &scr.probe_calls;

    if (!fb.reliable) {
        scr.pass = false;
        scr.reason = "planner not reliable";
        return scr;
    }
    if (fb.bands.empty()) {
        scr.pass = false;
        scr.reason = "planner returned no bands";
        return scr;
    }

    // significant sub-set of the planner's bands
    std::vector<FastBand> sig;
    for (const auto& b : fb.bands)
        if (band_significant(b, pf, calls)) sig.push_back(b);
    scr.n_significant = (int)sig.size();
    if (sig.empty()) {
        // Every planner band is razor-thin.  Individually each contributes mu
        // below the oracle's reference tolerance, but "no arc anywhere is
        // wider than kRazorAng" is itself a near-tangency geometry: the whole
        // image region is collapsing onto a fold and a thin arc the planner
        // dropped could still be the dominant one (rand008: classify_cells
        // marks a [0.5513,0.5520] near-tangency arc TOPOLOGY_UNCERTAIN).  The
        // screen cannot certify this cold -- route to the D14 oracle.
        scr.pass = false;
        scr.reason = "no significant band -- near-tangency, D14 authority";
        return scr;
    }

    // ---- check 1: partition validity (no hidden interior kEmpty) -------
    // The moment integrator re-derives the arc set at every quadrature
    // radius, so a crossing-count change within a band (an arc splitting
    // 2->4->2 as the source straddles a caustic fold) is harmless -- but a
    // kEmpty sub-region means the fast planner bridged a real gap and the
    // band set is not a valid integration partition.  A crossing-count
    // change also arms the independent down-march (check 3): a straddle is
    // exactly where a seed-less newborn band can hide.
    for (const auto& b : sig) {
        const double w = b.r_hi - b.r_lo;
        for (double fr : {0.08, 0.18, 0.32, 0.50, 0.68, 0.82, 0.92}) {
            ++*calls;
            GridArcs g = quartic_topology(b.r_lo + fr * w, pf);
            const bool nonempty =
                g.kind == ArcKind::kFull ||
                (g.kind == ArcKind::kArcs && g.n_crossings >= 2 &&
                 g.n_crossings % 2 == 0);
            if (!nonempty) {
                scr.pass = false;
                scr.failed_mask |= 1u;
                scr.reason = "significant band contains an image-free sub-region";
                return scr;
            }
        }
    }

    // ---- check 2: tangency (fold) regularity ---------------------------
    for (size_t i = 0; i < sig.size(); ++i) {
        const FastBand& b = sig[i];
        const double W = b.r_hi - b.r_lo;
        // lower endpoint -- skip only the origin-covering inner edge of the
        // innermost band
        if (i > 0 || b.r_lo > 1e-6 * (1.0 + b.r_lo)) {
            const double r = fold_ratio(b.r_lo, +1.0, W, pf, calls);
            if (r < 0.0) {
                scr.pass = false;
                scr.failed_mask |= 2u;
                scr.reason = "fold probe hit a degenerate radius";
                return scr;
            }
            if (std::fabs(r - 2.0) > std::fabs(scr.worst_fold - 2.0))
                scr.worst_fold = r;
            if (r < kFoldLo || r > kFoldHi) {
                scr.pass = false;
                scr.failed_mask |= 2u;
                scr.reason = "band lower endpoint not a regular fold";
                return scr;
            }
        }
        // upper endpoint -- always a tangency
        {
            const double r = fold_ratio(b.r_hi, -1.0, W, pf, calls);
            if (r < 0.0) {
                scr.pass = false;
                scr.failed_mask |= 2u;
                scr.reason = "fold probe hit a degenerate radius";
                return scr;
            }
            if (std::fabs(r - 2.0) > std::fabs(scr.worst_fold - 2.0))
                scr.worst_fold = r;
            if (r < kFoldLo || r > kFoldHi) {
                scr.pass = false;
                scr.failed_mask |= 2u;
                scr.reason = "band upper endpoint not a regular fold";
                return scr;
            }
        }
    }

    // ---- check 3: seed coverage (probe-free missing-band guard) --------
    // Every reliable point-source seed radius must fall inside one of the
    // planner's bands (significant OR razor).  A seed in no band => a
    // dropped band.
    for (const auto& im : seeds.images) {
        const double Ri = im.radius;
        if (!(Ri > 0.0)) continue;
        // A seed whose own local arc is sub-significant does not need a band:
        // the planner legitimately omits razor arcs, and dropping one moves
        // mu below the oracle's reference tolerance.  Costs one arc_intervals
        // call per seed.
        if (max_arc_at(Ri, pf, calls) < kRazorAng) continue;
        bool covered = false;
        for (const auto& b : fb.bands) {
            const double w = b.r_hi - b.r_lo;
            const double tol = std::max(0.05 * w, 0.15 * pf.rho);
            if (Ri >= b.r_lo - tol && Ri <= b.r_hi + tol) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            scr.pass = false;
            scr.failed_mask |= 4u;
            scr.reason = "a point-source seed radius lies in no planner band";
            return scr;
        }
    }

    // ---- check 4: complement scan for a seed-less newborn band --------
    // Runs unconditionally: an earlier revision gated this on a straddle
    // indicator, but fault injection (dbg_seedless) showed the gate let
    // adversarial seed-less-band drops through.  Still coarse -- see header.
    double s_max = 0.0, s_min = std::numeric_limits<double>::infinity();
    for (const auto& im : seeds.images) {
        s_max = std::max(s_max, im.radius);
        if (im.radius > 0.0) s_min = std::min(s_min, im.radius);
    }
    const double outer = std::max(sig.back().r_hi, s_max);
    const double r_top = outer + std::max(0.5 * pf.rho, 1e-3 * (1.0 + outer));
    const double r_floor =
        std::max(1e-6, 1e-4 * (std::isfinite(s_min) ? s_min : sig.front().r_lo));

    const int extra =
        complement_has_extra_band(pf, sig, r_top, r_floor, calls);
    scr.down_significant = (extra == 0) ? scr.n_significant : -1;
    if (extra < 0) {
        scr.pass = false;
        scr.failed_mask |= 8u;
        scr.reason = "complement scan hit an uncertain probe";
        return scr;
    }
    if (extra > 0) {
        scr.pass = false;
        scr.failed_mask |= 8u;
        scr.reason = "image found outside the planner's significant bands";
        return scr;
    }

    scr.pass = true;
    scr.reason = "all checks pass";
    return scr;
}

}  // namespace lcbinint::holonomic
