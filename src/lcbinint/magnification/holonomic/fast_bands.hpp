#pragma once

// ATPT holonomic solver -- fast seed-anchored radial-band planner.
//
// Phase A (unified-engine implementation order, checkpoint sec.0.7): the
// normal-path replacement for radial_events / classify_cells' D14 solve, which
// the E1 cost-split (checkpoint sec.14) measured at 74% of a value-only epoch.
//
// Seed-anchored port of the algebraic backend's find_radial_bands: from each
// certified point-source image radius, march outward and inward in R until the
// fixed-radius boundary circle stops producing an image, then boolean-bisect
// the bracket.  The march is BOUNDED by the neighbouring seed radii (an image
// band cannot span two point-source images that belong to different bands
// without an intervening image-free R interval), and every candidate band is
// re-checked for interior image-free notches ("solidity pass") -- the raw
// step-doubling march tunnels a notch narrower than one step, which merges two
// genuine bands (observed on multi-band geometries with a sub-rho pinch, e.g.
// plan15, extreme-q-planet).
//
// Primitives (holonomic frame):
//   "radius has image"  ==  arc_intervals(R, pf) yields a non-empty phi>0 arc
//                           set, or the full circle (kFull).
//   kEmpty / empty kArcs                       ->  no image.
//   kDegenerate (unresolved chart/root ambiguity) -> UNCERTAIN -> fail closed.
//
// This is a HEURISTIC discovery path, not a completeness proof: it assumes
// every finite-source image band contains at least one point-source image
// radius (true unless the source straddles a caustic at this rho).  It can
// also legitimately report MORE bands than an old sampling implementation:
// the boundary quartic resolves razor-thin image arcs that a fixed angular
// grid can miss.  The caller
// must run the checkpoint sec.5.2 confidence screen and fall back to
// classify_cells / D14 when `reliable` is false or the screen does not pass.
// No solver is replaced: classify_cells stays the oracle.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/point_images.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"

namespace lcbinint::holonomic {

struct FastBand {
    double r_lo, r_hi;
};

struct FastBands {
    std::vector<FastBand> bands;
    bool reliable = true;     // false -> caller must use classify_cells / D14
    const char* reason = "";  // why it bailed (when !reliable)
    int has_image_calls = 0;  // arc_intervals evaluations spent
};

namespace fast_bands_detail {

enum class Img { kNo, kYes, kUncertain };

inline Img has_image(double R, const PrimaryFrame& pf, int* calls) {
    ++*calls;
    if (!(R > 0.0)) return Img::kNo;
    ArcSet as = arc_intervals(R, pf);  // cold quartic solve, no warm chain
    switch (as.kind) {
        case ArcKind::kFull:
            return Img::kYes;
        case ArcKind::kArcs:
            return as.arcs.empty() ? Img::kNo : Img::kYes;
        case ArcKind::kEmpty:
            return Img::kNo;
        case ArcKind::kDegenerate:
        default:
            // A chart/root ambiguity is not an image classification.  The
            // reciprocal chart is attempted inside arc_intervals(); if it
            // still cannot certify the result, fail closed to the D14 oracle.
            return Img::kUncertain;
    }
}

// Bisect the [a,b] bracket (a in-image, b out) for the in->out transition.
// Returns the last-in-image radius.  UNCERTAIN anywhere -> *uncertain, NaN.
inline double bisect_edge(const PrimaryFrame& pf, double a_in, double b_out,
                          int iters, int* calls, bool* uncertain) {
    double lo = a_in, hi = b_out;
    // Absolute convergence floor: band edges feed GK node placement, which is
    // insensitive well below ~1e-9 relative.  Razor bands (width ~1e-6) are
    // excluded from the significance parity check, so a coarse edge there is
    // acceptable -- the floor still resolves them to ~1e-9 absolute.
    const double efloor = 1e-9 * (1.0 + std::fabs(a_in));
    for (int it = 0; it < iters; ++it) {
        if (std::fabs(hi - lo) <= efloor) break;
        const double mid = 0.5 * (lo + hi);
        if (mid == lo || mid == hi) break;
        const Img m = has_image(mid, pf, calls);
        if (m == Img::kUncertain) {
            *uncertain = true;
            return std::nan("");
        }
        if (m == Img::kYes)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

// One side of the seed: bounded step-doubling march to a sign change, then a
// bisection.  `limit` bounds the reach (a neighbouring seed, or 0 / +inf).
// On success sets *edge to the band boundary in that direction and *closed to
// whether an image-free R was actually reached (false => the march ran into
// `limit` still in-image: this seed shares a band with the neighbour).
inline bool march_side(const PrimaryFrame& pf, double seed_r, int dir,
                       double limit, int bis_iters, int* calls, double* edge,
                       bool* closed, int* steps_in) {
    const double sgn = dir < 0 ? -1.0 : 1.0;
    const double reach = std::isfinite(limit) ? std::fabs(limit - seed_r)
                                              : std::fabs(seed_r) + 1.0;
    double step = std::max(1e-12 * (1.0 + seed_r),
                           std::min(pf.rho, 0.5 * reach));
    if (!(step > 0.0)) step = 1e-12 * (1.0 + seed_r);
    *steps_in = 0;

    double active = seed_r;  // last in-image radius
    for (int i = 0; i < 128; ++i) {
        double cand = active + sgn * step;
        bool at_limit = false;
        if (std::isfinite(limit) &&
            ((dir < 0 && cand <= limit) || (dir > 0 && cand >= limit))) {
            cand = limit;
            at_limit = true;
        }
        if (dir < 0 && cand <= 0.0) {
            *edge = std::max(active > 0.0 ? 0.0 : active, 0.0);
            *steps_in = 99;  // spanned to R->0: treat as a wide march
            // march reached R -> 0 still in image: bisect (0, active].
            bool unc = false;
            double e = bisect_edge(pf, active, 0.0, bis_iters, calls, &unc);
            if (unc) return false;
            *edge = e;
            *closed = true;
            return true;
        }
        if (cand == active) {  // underflow against magnitude
            *edge = active;
            *closed = true;
            return true;
        }
        const Img in = has_image(cand, pf, calls);
        if (in == Img::kUncertain) return false;
        if (in == Img::kNo) {
            bool unc = false;
            double e = bisect_edge(pf, active, cand, bis_iters, calls, &unc);
            if (unc) return false;
            *edge = e;
            *closed = true;
            return true;
        }
        // still in image
        active = cand;
        ++*steps_in;
        if (at_limit) {
            *edge = active;
            *closed = false;  // ran into the neighbour still in-image
            return true;
        }
        step = std::min(step * 2.0, 0.5 * std::fabs(
                                        (std::isfinite(limit) ? limit : active +
                                             sgn * (std::fabs(active) + 1.0)) -
                                        active));
        if (!(step > 1e-15 * (1.0 + std::fabs(active)))) {
            *edge = active;
            *closed = true;
            return true;
        }
    }
    return false;  // never resolved
}

// Split one candidate band [lo,hi] (endpoints in-image) into maximal solid
// sub-bands, catching interior image-free notches the step-doubling march
// tunnelled.  A merged band's notch, if any, always lies between two
// point-source seeds that belong to different bands, so the interior probes
// are placed per seed gap: {lo, interior seeds, hi} are the anchors, and each
// anchor gap is sampled at a resolution fine enough to resolve a notch
// (~rho/50, capped).  UNCERTAIN anywhere -> *uncertain.  The image-free
// notches at this scale never nest, so one probe level suffices.
inline void solidify(const PrimaryFrame& pf, double lo, double hi,
                     const std::vector<double>& seeds_in_band, int bis_iters,
                     int* calls, bool* uncertain, std::vector<FastBand>* out) {
    const double width = hi - lo;
    if (!(width > 0.0)) return;
    const double tol = 1e-11 * (1.0 + std::fabs(hi));
    if (width <= tol) {
        out->push_back({lo, hi});
        return;
    }

    // anchors: band edges plus every seed strictly interior to (lo, hi)
    std::vector<double> anchors;
    anchors.push_back(lo);
    for (double s : seeds_in_band)
        if (s > lo + tol && s < hi - tol) anchors.push_back(s);
    anchors.push_back(hi);
    std::sort(anchors.begin(), anchors.end());

    // interior probe radii, per anchor gap.  Resolution ~rho/50 resolves the
    // notches seen in practice (plan15, extreme-q-planet, on-axis-off); the
    // per-gap probe count is capped and the whole-band budget is bounded so a
    // wide solid band (the common gated case: >1 seed, no notch) stays cheap.
    const int n_gaps = (int)anchors.size() - 1;
    const int budget = 44;
    const int per_gap_cap = std::max(6, budget / std::max(1, n_gaps));
    const double res = std::max(1e-9 * (1.0 + hi),
                                std::min(0.02 * pf.rho, width));
    std::vector<double> xs;
    for (size_t g = 0; g + 1 < anchors.size(); ++g) {
        const double a = anchors[g], b = anchors[g + 1], d = b - a;
        if (!(d > 2.0 * tol)) continue;
        int n = (int)std::ceil(d / res);
        if (n < 3) n = 3;
        if (n > per_gap_cap) n = per_gap_cap;
        for (int k = 1; k < n; ++k) xs.push_back(a + d * k / n);
    }
    std::sort(xs.begin(), xs.end());
    const int NS = (int)xs.size();
    if (NS == 0) {
        out->push_back({lo, hi});
        return;
    }

    std::vector<Img> st(NS);
    for (int k = 0; k < NS; ++k) {
        st[k] = has_image(xs[k], pf, calls);
        if (st[k] == Img::kUncertain) {
            *uncertain = true;
            return;
        }
    }
    // Walk lo -> hi through knots {lo(in), xs[0..NS-1], hi(in)} and emit a
    // solid sub-band for every maximal in-image run, bisecting each in<->out
    // transition against its bracketing knots.
    double run_lo = lo;             // start of the current in-image run
    double prev_x = lo;             // previous knot radius (in-image)
    bool prev_in = true;
    for (int k = 0; k <= NS; ++k) {
        const double cur_x = (k < NS) ? xs[k] : hi;
        const bool cur_in = (k < NS) ? (st[k] == Img::kYes) : true;
        if (prev_in && !cur_in) {
            // in -> out : close the run at the bisected edge in (prev_x, cur_x)
            bool unc = false;
            double e = bisect_edge(pf, prev_x, cur_x, bis_iters, calls, &unc);
            if (unc) { *uncertain = true; return; }
            if (e > run_lo) out->push_back({run_lo, e});
        } else if (!prev_in && cur_in) {
            // out -> in : open a new run at the bisected edge (from cur side)
            bool unc = false;
            double e = bisect_edge(pf, cur_x, prev_x, bis_iters, calls, &unc);
            if (unc) { *uncertain = true; return; }
            run_lo = e;
        }
        prev_x = cur_x;
        prev_in = cur_in;
    }
    if (prev_in && hi > run_lo) out->push_back({run_lo, hi});
}

}  // namespace fast_bands_detail

inline FastBands fast_bands(const PrimaryFrame& pf, const PointImages& seeds,
                            int bisection_iters = 26) {
    using fast_bands_detail::has_image;
    using fast_bands_detail::Img;
    using fast_bands_detail::march_side;
    using fast_bands_detail::solidify;

    FastBands out;
    if (!seeds.reliable || seeds.images.empty()) {
        out.reliable = false;
        out.reason = "point-source seeds unreliable";
        return out;
    }

    // sorted unique seed radii
    std::vector<double> S;
    for (const auto& im : seeds.images) {
        const double r = im.radius;
        if (!(r > 0.0)) continue;
        if (!S.empty() && std::fabs(r - S.back()) <= 1e-9 * (1.0 + r)) continue;
        S.push_back(r);
    }
    if (S.empty()) {
        out.reliable = false;
        out.reason = "no positive seed radii";
        return out;
    }

    int* calls = &out.has_image_calls;
    struct RawBand { double lo, hi; bool wide; };
    std::vector<RawBand> raw;

    for (size_t i = 0; i < S.size(); ++i) {
        double seed_r = S[i];

        // seed must sit on an image radius; if not, try a local recovery
        // (a point-source image right at the planet can land marginally
        // outside its own razor finite-source band).
        Img si = has_image(seed_r, pf, calls);
        if (si == Img::kUncertain) {
            out.reliable = false;
            out.reason = "seed radius topology uncertain";
            return out;
        }
        if (si == Img::kNo) {
            const double lo_n = (i == 0) ? 0.0 : S[i - 1];
            const double hi_n =
                (i + 1 < S.size()) ? S[i + 1] : seed_r + pf.rho;
            bool rec = false;
            for (double f : {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2}) {
                for (int s = -1; s <= 1; s += 2) {
                    double rr = seed_r + s * f * (1.0 + seed_r);
                    if (rr <= lo_n || rr >= hi_n || rr <= 0.0) continue;
                    if (has_image(rr, pf, calls) == Img::kYes) {
                        seed_r = rr;
                        rec = true;
                        break;
                    }
                }
                if (rec) break;
            }
            if (!rec) continue;  // no band here -- completeness screen's job
        }

        bool covered = false;
        for (const auto& b : raw)
            if (seed_r >= b.lo && seed_r <= b.hi) covered = true;
        if (covered) continue;

        const double lo_lim = (i == 0) ? 0.0 : S[i - 1];
        const double hi_lim = (i + 1 < S.size())
                                  ? S[i + 1]
                                  : std::numeric_limits<double>::infinity();

        double lo_edge = seed_r, hi_edge = seed_r;
        bool lo_closed = true, hi_closed = true;
        int lo_steps = 0, hi_steps = 0;
        if (!march_side(pf, seed_r, -1, lo_lim, bisection_iters, calls,
                        &lo_edge, &lo_closed, &lo_steps) ||
            !march_side(pf, seed_r, +1, hi_lim, bisection_iters, calls,
                        &hi_edge, &hi_closed, &hi_steps)) {
            out.reliable = false;
            out.reason = "radial march failed (uncertain probe)";
            return out;
        }
        // A march that accepted >=2 in-image steps on a side could have
        // step-doubled across an image-free notch -> that band needs the
        // solidity pass.  A single-step-or-less march on both sides brackets
        // the seed within one (un-doubled) step and cannot hide a notch.
        const bool wide = (lo_steps >= 2 || hi_steps >= 2 || !lo_closed ||
                           !hi_closed);
        if (hi_edge > lo_edge) raw.push_back({lo_edge, hi_edge, wide});
    }

    if (raw.empty()) {
        out.reliable = false;
        out.reason = "no radial bands found";
        return out;
    }

    // merge overlapping / touching raw brackets; a merged band needs the
    // solidity pass if it absorbed >1 bracket or any bracket had a wide march.
    std::sort(raw.begin(), raw.end(),
              [](const RawBand& a, const RawBand& b) { return a.lo < b.lo; });
    struct MBand { double lo, hi; bool solid_check; };
    std::vector<MBand> merged;
    for (const auto& b : raw) {
        const double tol = 1e-11 * (1.0 + std::fabs(b.hi));
        if (merged.empty() || b.lo > merged.back().hi + tol)
            merged.push_back({b.lo, b.hi, b.wide});
        else {
            merged.back().hi = std::max(merged.back().hi, b.hi);
            merged.back().solid_check = true;
        }
    }

    // solidity pass: split any band that tunnelled an interior image-free notch
    std::vector<FastBand> solid;
    for (const auto& b : merged) {
        if (!b.solid_check) {
            solid.push_back({b.lo, b.hi});
            continue;
        }
        bool uncertain = false;
        solidify(pf, b.lo, b.hi, S, bisection_iters, calls, &uncertain, &solid);
        if (uncertain) {
            out.reliable = false;
            out.reason = "interior probe topology uncertain";
            return out;
        }
    }
    std::sort(solid.begin(), solid.end(),
              [](const FastBand& a, const FastBand& b) {
                  return a.r_lo < b.r_lo;
              });
    // final merge (bisection edges of adjacent sub-bands can meet)
    std::vector<FastBand> final_bands;
    for (const auto& b : solid) {
        const double tol = 1e-10 * (1.0 + std::fabs(b.r_hi));
        if (final_bands.empty() || b.r_lo > final_bands.back().r_hi + tol)
            final_bands.push_back(b);
        else
            final_bands.back().r_hi = std::max(final_bands.back().r_hi, b.r_hi);
    }

    out.bands = std::move(final_bands);
    return out;
}

}  // namespace lcbinint::holonomic
