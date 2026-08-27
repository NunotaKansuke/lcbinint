// Focused checks for the forward-mode analytic derivatives of the
// algebraic-boundary kernel (docs/design/algebraic-boundary-autodiff-20260828.md).
//
//   1. The templated real trig-quartic form, instantiated on double, matches
//      the primal complex construction bit for bit.
//   2. dF/dR from a ForwardJet lane matches a centered finite difference.
//   3. d(theta_k)/d(source_x) from the quartic IFT matches a finite difference
//      of the primal boundary angle.

#include "lcbinint/magnification/algebraic_boundary.hpp"
#include "lcbinint/magnification/algebraic_boundary_autodiff.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace ad = lcbinint::magnification::autodiff;
using lcbinint::magnification::AlgebraicBoundaryLens;
using lcbinint::magnification::AlgebraicBoundarySettings;
using lcbinint::SourcePosition;

namespace {

int failures = 0;

void check(bool ok, const char* what, double got, double want, double tol)
{
    if (!ok) {
        ++failures;
        std::printf("FAIL %-40s got=%.17g want=%.17g |d|=%.3e tol=%.1e\n",
                    what, got, want, std::abs(got - want), tol);
    } else {
        std::printf("ok   %-40s (|d|=%.3e)\n", what, std::abs(got - want));
    }
}

ad::FrozenLensBranch branch_for(double separation_raw, double q_raw)
{
    ad::FrozenLensBranch b;
    b.separation_sign = separation_raw < 0.0 ? -1.0 : 1.0;
    b.q_below_one = std::abs(q_raw) < 1.0;
    return b;
}

// params: [source_x, source_y, rho, q_raw, separation_raw]
ad::BoundaryTrigForm<double> primal_trig_form(
    const std::array<double, 5>& p, double image_radius)
{
    const double s = std::abs(p[4]);
    const double q_input = std::abs(p[3]);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const double m1 = 1.0 / (1.0 + q);
    AlgebraicBoundaryLens lens{q_input < 1.0 ? -s : s, m1, q * m1};
    const auto coeffs = lcbinint::magnification::algebraic_boundary_quartic(
        lens, {p[0], p[1]}, p[2], image_radius);
    return {coeffs[4].real(), coeffs[4].imag(), coeffs[3].real(),
            coeffs[3].imag(), coeffs[2].real()};
}

double primal_boundary_angle(
    const std::array<double, 5>& p, double image_radius, int which)
{
    const double s = std::abs(p[4]);
    const double q_input = std::abs(p[3]);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const double m1 = 1.0 / (1.0 + q);
    AlgebraicBoundaryLens lens{q_input < 1.0 ? -s : s, m1, q * m1};
    AlgebraicBoundarySettings settings;
    const auto intervals = lcbinint::magnification::algebraic_boundary_intervals_at_radius(
        lens, {p[0], p[1]}, p[2], image_radius, settings, nullptr);
    if (!intervals.success ||
        which >= static_cast<int>(intervals.boundary_angles.size())) {
        return std::nan("");
    }
    return intervals.boundary_angles[static_cast<std::size_t>(which)];
}

}  // namespace

int main()
{
    // A resonant-ish binary with the source near (but not on) a caustic.
    const std::array<double, 5> params{0.11, 0.04, 0.02, 0.6, 1.05};
    const auto branch = branch_for(params[4], params[3]);

    // Find an image radius whose fixed-radius circle carries boundary
    // crossings, and which stays a crossing under small source_x perturbation
    // (so the finite-difference reference in check 3 is well posed).
    double R = 0.0;
    for (double candidate = 0.20; candidate < 2.60; candidate += 0.02) {
        const double a0 = primal_boundary_angle(params, candidate, 0);
        std::array<double, 5> pp = params;
        std::array<double, 5> pm = params;
        pp[0] += 1e-7;
        pm[0] -= 1e-7;
        if (!std::isnan(a0) &&
            !std::isnan(primal_boundary_angle(pp, candidate, 0)) &&
            !std::isnan(primal_boundary_angle(pm, candidate, 0))) {
            R = candidate;
            break;
        }
    }
    if (R == 0.0) {
        std::printf("FAIL could not locate a testable image radius\n");
        return 1;
    }
    std::printf("using image radius R = %.4f\n", R);

    // ---- 1. double parity -------------------------------------------------
    {
        std::array<double, 5> pd = params;
        ad::BinaryGeometry<double> g = ad::make_geometry<double>(pd, branch);
        ad::BoundaryTrigForm<double> mine = ad::make_trig_form<double>(g, R);
        ad::BoundaryTrigForm<double> ref = primal_trig_form(params, R);
        check(std::abs(mine.A - ref.A) <= 1e-12 * (1 + std::abs(ref.A)),
              "trig-form A vs primal", mine.A, ref.A, 1e-12);
        check(std::abs(mine.B - ref.B) <= 1e-12 * (1 + std::abs(ref.B)),
              "trig-form B vs primal", mine.B, ref.B, 1e-12);
        check(std::abs(mine.C - ref.C) <= 1e-12 * (1 + std::abs(ref.C)),
              "trig-form C vs primal", mine.C, ref.C, 1e-12);
        check(std::abs(mine.D - ref.D) <= 1e-12 * (1 + std::abs(ref.D)),
              "trig-form D vs primal", mine.D, ref.D, 1e-12);
        check(std::abs(mine.Epsilon - ref.Epsilon) <= 1e-12 * (1 + std::abs(ref.Epsilon)),
              "trig-form Epsilon vs primal", mine.Epsilon, ref.Epsilon, 1e-12);
    }

    // ---- 2. dF/dR via a ForwardJet lane vs centered FD -------------------
    {
        const double theta = 1.3;
        // 6 lanes: 0..4 = params, 5 = R.
        std::array<ad::ForwardJet<6>, 5> pj;
        for (std::size_t i = 0; i < 5; ++i)
            pj[i] = ad::ForwardJet<6>::variable(params[i], i);
        ad::ForwardJet<6> Rj = ad::ForwardJet<6>::variable(R, 5);
        auto gj = ad::make_geometry<ad::ForwardJet<6>>(pj, branch);
        auto formj = ad::make_trig_form<ad::ForwardJet<6>>(gj, Rj);
        ad::ForwardJet<6> Fj = ad::trig_value(formj, theta);
        const double analytic_dF_dR = Fj.derivative[5];

        const double h = 1e-6;
        auto F_at_R = [&](double r) {
            std::array<double, 5> pd = params;
            auto g = ad::make_geometry<double>(pd, branch);
            auto form = ad::make_trig_form<double>(g, r);
            return ad::trig_value(form, theta);
        };
        const double fd = (F_at_R(R + h) - F_at_R(R - h)) / (2.0 * h);
        check(std::abs(analytic_dF_dR - fd) <= 1e-5 * (1 + std::abs(fd)),
              "dF/dR jet vs centered FD", analytic_dF_dR, fd, 1e-5);
    }

    // ---- 3. d(theta_k)/d(source_x) via quartic IFT vs FD ----------------
    {
        const int which = 0;
        const double theta_k = primal_boundary_angle(params, R, which);
        if (std::isnan(theta_k)) {
            std::printf("FAIL primal produced no boundary angle to test\n");
            return 1;
        }
        std::array<ad::ForwardJet<5>, 5> pj;
        for (std::size_t i = 0; i < 5; ++i)
            pj[i] = ad::ForwardJet<5>::variable(params[i], i);
        auto gj = ad::make_geometry<ad::ForwardJet<5>>(pj, branch);
        auto formj = ad::make_trig_form<ad::ForwardJet<5>>(gj, ad::ForwardJet<5>(R));
        ad::ForwardJet<5> Fj = ad::trig_value(formj, theta_k);
        ad::ForwardJet<5> F1j = ad::trig_first(formj, theta_k);
        // dtheta_k/dp = -(dF/dp) / F'(theta_k)
        const double analytic = -Fj.derivative[0] / F1j.value;

        const double h = 1e-7;
        std::array<double, 5> pp = params;
        std::array<double, 5> pm = params;
        pp[0] += h;
        pm[0] -= h;
        const double ap = primal_boundary_angle(pp, R, which);
        const double am = primal_boundary_angle(pm, R, which);
        const double fd = (ap - am) / (2.0 * h);
        check(std::abs(analytic - fd) <= 5e-5 * (1 + std::abs(fd)),
              "dtheta_k/dsource_x IFT vs FD", analytic, fd, 5e-5);
    }

    if (failures == 0) {
        std::printf("\nall autodiff-core checks passed\n");
        return 0;
    }
    std::printf("\n%d check(s) failed\n", failures);
    return 1;
}
