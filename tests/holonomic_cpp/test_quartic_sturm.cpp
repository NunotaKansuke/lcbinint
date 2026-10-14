// Degree-4 Sturm topology certificate tests.
//
// The two physical cases below deliberately use arcs narrower than the old
// fixed 3072-point angular sampler.  The sampler is retained as a diagnostic
// witness; the certified quartic path must still report the two crossings.

#include <array>
#include <cmath>
#include <cstdio>

#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/quartic_sturm.hpp"
#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"
#include "lcbinint/magnification/holonomic/v2_profile.hpp"

using namespace lcbinint::holonomic;

namespace {

int failures = 0;

void require(bool condition, const char* what) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

void check_polynomial_counts() {
    // t^4 + 1 has no real roots.
    auto none = certify_quartic({{1.0, 0.0, 0.0, 0.0, 1.0}});
    require(none.certified && none.root_count == 0,
            "Sturm count for zero-real-root quartic");

    // (t-1)(t-2)(t+3)(t+4).
    auto four = certify_quartic({{24.0, -22.0, -7.0, 4.0, 1.0}});
    require(four.certified && four.root_count == 4,
            "Sturm count for four-real-root quartic");
    const auto isolated =
        sturm_isolate_real_roots(four.chart_coeffs, four.root_count);
    require(isolated.size() == 4, "Sturm isolation returns four roots");

    // Two roots at a large t value exercise the reciprocal projective chart.
    auto large = certify_quartic({{-1.0, 0.0, 0.0, 0.0, 1.0e-12}});
    require(large.certified && large.root_count == 2,
            "Sturm count for large-t roots");
    require(large.reciprocal, "large-t certificate selects reciprocal chart");

    // Exact p4=0 with a simple projective root at theta=pi.  The reciprocal
    // chart counts u=0 as an ordinary simple root; this is distinct from a
    // multiple chart-event root, which must remain fail-closed.
    auto infinity = certify_quartic({{-1.0, 0.0, 0.0, 1.0, 0.0}});
    require(infinity.certified && infinity.chart_singular &&
                infinity.reciprocal && infinity.root_count == 2,
            "Sturm reciprocal certificate handles simple root at infinity");
}

void check_thin_physical_arcs() {
    // Correct VBM-corpus mapping: positive source x and inverse raw mass ratio.
    const LensParams corpus{
        1.1409565850100225, -4.0472265932659397,
        0.00017266345397718498, 1.0 / 0.5633224083213993,
        0.2310772011499552, true};
    const PrimaryFrame pf = PrimaryFrame::from(corpus);
    const double R = 4.4729290649084206;
    V2Profile profile;
    {
        V2ProfileScope scope(profile);
        const GridArcs certified = quartic_topology(R, pf);
        require(certified.certified && certified.n_crossings == 2,
                "thin outer arc has certified two crossings");
        require(profile.grid512_calls == 0 && profile.grid3072_calls == 0,
                "quartic topology certificate does not call angular grid");
    }
    const GridArcs sampled = arcs_at(R, pf, 3072);
    require(sampled.kind == ArcKind::kEmpty,
            "3072 sampler misses the deliberately thin outer arc");

    const LensParams wide{1.4, 0.10, 0.03, 1.0e-3, 2.5, false};
    const PrimaryFrame wide_pf = PrimaryFrame::from(wide);
    const GridArcs wide_cert =
        quartic_topology(2.5013971414257967, wide_pf);
    const GridArcs wide_sample =
        arcs_at(2.5013971414257967, wide_pf, 3072);
    require(wide_cert.certified && wide_cert.n_crossings == 2,
            "wide-planet thin arc has certified two crossings");
    require(wide_sample.kind == ArcKind::kEmpty,
            "3072 sampler misses the wide-planet thin arc");
}

void check_reciprocal_arc_path() {
    const LensParams p{0.5, 0.0, 0.05, 0.4, 1.1, false};
    const PrimaryFrame pf = PrimaryFrame::from(p);
    const auto crossings = re_detail::chart_p4_factor_roots(pf);
    require(!crossings.empty(), "chart_p4 case has a positive crossing");
    for (double R0 : crossings) {
        for (double delta : {-1e-10, -1e-12, 0.0, 1e-12, 1e-10}) {
            const double R = R0 + delta;
            const QuarticCoeffs q = boundary_quartic(R, pf);
            double scale = 0.0;
            for (double c : q.p) scale = std::max(scale, std::fabs(c));
            require(std::fabs(q.p[4]) <= 2e-6 * (1.0 + scale),
                    "chart_p4 root makes the t^4 coefficient small");
            V2Profile profile;
            ArcSet as;
            {
                V2ProfileScope scope(profile);
                as = arc_intervals(R, pf);
            }
            require(profile.grid512_calls == 0 &&
                        profile.grid3072_calls == 0 &&
                        profile.grid4096_calls == 0,
                    "chart_p4 arc path does not call an angular grid");
            if (delta == 0.0)
                require(profile.arc_reciprocal_attempts > 0,
                        "chart_p4 path attempts the reciprocal chart");
            if (delta != 0.0)
                require(as.kind != ArcKind::kDegenerate,
                        "reciprocal chart resolves near-chart arc geometry");
            else {
                // The second chart_p4 root is an exact double root at theta=pi
                // (Y=0 symmetry).  A zero-width tangency is intentionally
                // unresolved at the event itself; neighboring open radii are
                // handled by the reciprocal path above and the radial event
                // partition keeps this point out of production cells.
                if (as.kind == ArcKind::kDegenerate)
                    require(std::fabs(q.p[3]) <= 1e-14 *
                                std::max(1.0, std::fabs(q.p[2])),
                            "only a multiple chart-event root remains degenerate");
            }
        }
    }
}

}  // namespace

int main() {
    check_polynomial_counts();
    check_thin_physical_arcs();
    check_reciprocal_arc_path();
    std::printf("quartic_sturm %s failures=%d\n",
                failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
