// Forward-mode analytic Jacobian for the experimental polar algebraic-boundary
// finite-source binary kernel. See algebraic_boundary_autodiff.hpp and
// docs/design/algebraic-boundary-autodiff-20260828.md.
//
// The primal integral runs unchanged and decides every discrete quantity
// (radial bands, per-radius boundary classification, real-quartic-vs-Skowron).
// This driver re-reads that frozen structure and carries parameter derivatives
// through one templated ForwardJet<5> pass:
//
//   * band endpoints R*(p):   2x2 tangency IFT on (F, F') = 0, with dF/dR and
//                              dF'/dR from a ForwardJet<6> whose sixth lane is R.
//   * crossing angles th_k(p): quartic IFT  dth_k/dp = -(dF/dp)/F'(th_k),
//                              evaluated at the frozen th_k with F built as a
//                              ForwardJet whose R lane already carries the
//                              moving radial node R(t;p) (Leibniz limits fold
//                              in automatically).
//   * the radial quadrature is refined by panel doubling until the derivative
//     of the band flux stops moving; the primal Gauss/Kronrod split drives the
//     two-form reliability estimate.

#include "lcbinint/magnification/algebraic_boundary_autodiff.hpp"

#include "lcbinint/magnification/algebraic_boundary.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace lcbinint::magnification::autodiff {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;

using Jet = ForwardJet<5>;
using JetR = ForwardJet<6>;  // params + R lane

// ---- frozen Gauss-Kronrod rules (copied from algebraic_boundary.cpp) --------

struct GK15 {
    std::array<double, 15> nodes{{
        -0.9914553711208126, -0.9491079123427585, -0.8648644233597691,
        -0.7415311855993945, -0.5860872354676911, -0.4058451513773972,
        -0.2077849550078985, 0.0, 0.2077849550078985, 0.4058451513773972,
        0.5860872354676911, 0.7415311855993945, 0.8648644233597691,
        0.9491079123427585, 0.9914553711208126}};
    std::array<double, 15> kw{{
        0.02293532201052922, 0.06309209262997855, 0.1047900103222502,
        0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
        0.2044329400752989, 0.2094821410847278, 0.2044329400752989,
        0.1903505780647854, 0.1690047266392679, 0.1406532597155259,
        0.1047900103222502, 0.06309209262997855, 0.02293532201052922}};
    std::array<double, 15> gw{{
        0.0, 0.1294849661688697, 0.0, 0.2797053914892766, 0.0,
        0.3818300505051189, 0.0, 0.4179591836734694, 0.0, 0.3818300505051189,
        0.0, 0.2797053914892766, 0.0, 0.1294849661688697, 0.0}};
};

struct GK21 {
    std::array<double, 21> nodes{{
        -0.9956571630258081, -0.9739065285171717, -0.9301574913557082,
        -0.8650633666889845, -0.7808177265864169, -0.6794095682990244,
        -0.5627571346686047, -0.4333953941292472, -0.2943928627014602,
        -0.1488743389816312, 0.0, 0.1488743389816312, 0.2943928627014602,
        0.4333953941292472, 0.5627571346686047, 0.6794095682990244,
        0.7808177265864169, 0.8650633666889845, 0.9301574913557082,
        0.9739065285171717, 0.9956571630258081}};
    std::array<double, 21> kw{{
        0.01169463886737187, 0.03255816230796473, 0.05475589657435199,
        0.07503967481091995, 0.09312545458369761, 0.1093871588022976,
        0.1234919762620659, 0.1347092173114733, 0.1427759385770601,
        0.1477391049013385, 0.1494455540029169, 0.1477391049013385,
        0.1427759385770601, 0.1347092173114733, 0.1234919762620659,
        0.1093871588022976, 0.09312545458369761, 0.07503967481091995,
        0.05475589657435199, 0.03255816230796473, 0.01169463886737187}};
};

template <std::size_t N>
struct SineMap {
    std::array<double, N> coord{};
    std::array<double, N> jac{};
};

template <std::size_t N>
SineMap<N> make_sine_map(const std::array<double, N>& nodes)
{
    SineMap<N> m;
    for (std::size_t i = 0; i < N; ++i) {
        const double phase = 0.5 * kPi * nodes[i];
        m.coord[i] = std::sin(phase);
        m.jac[i] = 0.5 * kPi * std::cos(phase);
    }
    return m;
}

const GK15& gk15() { static const GK15 r; return r; }
const GK21& gk21() { static const GK21 r; return r; }
const SineMap<15>& smap15() { static const auto m = make_sine_map(gk15().nodes); return m; }
const SineMap<21>& smap21() { static const auto m = make_sine_map(gk21().nodes); return m; }

// ---- helpers --------------------------------------------------------------

template <typename S>
double scale_of(const BoundaryTrigForm<S>& f)
{
    const double p0 = std::hypot(f.A.value, f.B.value);
    const double p1 = std::hypot(f.C.value, f.D.value);
    const double p2 = std::fabs(f.Epsilon.value);
    return std::max(1.0, std::max(p0, std::max(p1, p2)));
}

template <std::size_t N>
struct MomentsJet {
    ForwardJet<N> m0{0.0};
    ForwardJet<N> mhalf{0.0};
    ForwardJet<N> mquarter{0.0};
};

template <std::size_t N>
ForwardJet<N> combined_flux_jet(const MomentsJet<N>& m, double c, double d)
{
    return (1.0 - c - d) * m.m0 + c * m.mhalf + d * m.mquarter;
}

// Angular contribution of one fixed radial node R(t;p), as a ForwardJet<5>
// Moments triple. `reliable` is cleared on any grazing crossing angle.
MomentsJet<5> radial_integrand_jet(
    const BinaryGeometry<Jet>& geom_j,
    const AlgebraicBoundaryLens& lens_d,
    SourcePosition source_d,
    double rho_d,
    const Jet& radius_j,
    double c,
    double d,
    const AlgebraicBoundarySettings& settings,
    bool* reliable)
{
    MomentsJet<5> out;
    const double R = radius_j.value;
    const auto iv = algebraic_boundary_intervals_at_radius(
        lens_d, source_d, rho_d, R, settings, nullptr);
    if (!iv.success) {
        *reliable = false;
        return out;
    }
    if (iv.inside_intervals.empty()) {
        return out;
    }

    const auto form_j = make_trig_form<Jet>(geom_j, radius_j);
    const double scale = scale_of(form_j);
    const bool need_half = (c != 0.0);
    const bool need_quarter = (d != 0.0);

    // Full-circle support: constant angular limits, no crossing angles.
    if (iv.boundary_angles.empty()) {
        const Jet width(kTwoPi);
        out.m0 = radius_j * width;
        if (need_half || need_quarter) {
            const Jet mid_j(kPi);
            const Jet hw_j(kPi);
            for (std::size_t a = 0; a < 21; ++a) {
                const Jet ang_j = mid_j + hw_j * smap21().coord[a];
                const Jet phi_j = scalar_max0(phi_value_jet<Jet>(geom_j, radius_j, ang_j));
                const Jet half_j = scalar_sqrt(phi_j);
                const Jet common_j = radius_j * hw_j * smap21().jac[a];
                if (need_half) out.mhalf = out.mhalf + (common_j * gk21().kw[a]) * half_j;
                if (need_quarter)
                    out.mquarter = out.mquarter + (common_j * gk21().kw[a]) * scalar_sqrt(half_j);
            }
        }
        return out;
    }

    const auto& angles = iv.boundary_angles;
    const std::size_t n = angles.size();
    std::vector<Jet> theta_j(n);
    for (std::size_t k = 0; k < n; ++k) {
        const Jet F = trig_value<Jet>(form_j, angles[k]);
        const double Fp = trig_first<Jet>(form_j, angles[k]).value;
        if (std::fabs(Fp) <= 1.0e-9 * scale) {
            *reliable = false;
        }
        Jet tj(angles[k]);
        const double inv = (Fp != 0.0) ? 1.0 / Fp : 0.0;
        for (std::size_t p = 0; p < 5; ++p) tj.derivative[p] = -F.derivative[p] * inv;
        theta_j[k] = tj;
    }

    const auto is_inside = [&](double lo) {
        for (const auto& seg : iv.inside_intervals) {
            if (std::fabs(seg.first - lo) < 1.0e-9) return true;
        }
        return false;
    };

    for (std::size_t i = 0; i < n; ++i) {
        if (!is_inside(angles[i])) continue;
        const Jet lo_j = theta_j[i];
        const Jet hi_j = (i + 1 < n) ? theta_j[i + 1] : (theta_j[0] + kTwoPi);
        out.m0 = out.m0 + radius_j * (hi_j - lo_j);
        if (!need_half && !need_quarter) continue;
        const Jet mid_j = 0.5 * (lo_j + hi_j);
        const Jet hw_j = 0.5 * (hi_j - lo_j);
        for (std::size_t a = 0; a < 21; ++a) {
            const Jet ang_j = mid_j + hw_j * smap21().coord[a];
            const Jet phi_j = scalar_max0(phi_value_jet<Jet>(geom_j, radius_j, ang_j));
            const Jet half_j = scalar_sqrt(phi_j);
            const Jet common_j = radius_j * hw_j * smap21().jac[a];
            if (need_half) out.mhalf = out.mhalf + (common_j * gk21().kw[a]) * half_j;
            if (need_quarter)
                out.mquarter = out.mquarter + (common_j * gk21().kw[a]) * scalar_sqrt(half_j);
        }
    }
    return out;
}

// 2x2 tangency IFT for a band endpoint R*. Returns R* as a ForwardJet<5>.
Jet endpoint_derivative(
    const BinaryGeometry<JetR>& geom_r,
    const AlgebraicBoundaryLens& lens_d,
    SourcePosition source_d,
    double rho_d,
    double r_star,
    double band_lo,
    double band_hi,
    const AlgebraicBoundarySettings& settings,
    bool* reliable)
{
    Jet result(r_star);
    if (r_star < 1.0e-9) {
        return result;  // inner band touches the origin; endpoint is fixed.
    }
    const double span = std::max(band_hi - band_lo, 1.0e-12);
    double probe = r_star;
    AlgebraicBoundaryIntervals iv =
        algebraic_boundary_intervals_at_radius(lens_d, source_d, rho_d, probe, settings, nullptr);
    for (int attempt = 0; attempt < 4 &&
         (!iv.success || iv.inside_intervals.empty()); ++attempt) {
        const double inward = (std::fabs(r_star - band_lo) < std::fabs(r_star - band_hi))
            ? 1.0 : -1.0;
        probe = r_star + inward * (attempt + 1) * 1.0e-6 * span;
        iv = algebraic_boundary_intervals_at_radius(
            lens_d, source_d, rho_d, probe, settings, nullptr);
    }
    if (!iv.success || iv.inside_intervals.empty()) {
        *reliable = false;
        return result;
    }
    double best_w = std::numeric_limits<double>::infinity();
    double theta_star = 0.0;
    for (const auto& seg : iv.inside_intervals) {
        const double w = seg.second - seg.first;
        if (w < best_w) { best_w = w; theta_star = 0.5 * (seg.first + seg.second); }
    }

    const auto form_r = make_trig_form<JetR>(geom_r, JetR::variable(r_star, 5));
    const JetR F = trig_value<JetR>(form_r, theta_star);
    const JetR Fp = trig_first<JetR>(form_r, theta_star);
    const JetR Fpp = trig_second<JetR>(form_r, theta_star);
    const double dF_dR = F.derivative[5];
    const double dFp_dR = Fp.derivative[5];
    const double Fp_v = Fp.value;
    const double Fpp_v = Fpp.value;
    const double scale = scale_of(form_r);
    const double det = dF_dR * Fpp_v - Fp_v * dFp_dR;
    const double det_terms = std::fabs(dF_dR * Fpp_v) + std::fabs(Fp_v * dFp_dR);
    if (std::fabs(Fpp_v) <= 1.0e-9 * scale ||
        std::fabs(det) <= 1.0e-8 * std::max(det_terms, scale)) {
        *reliable = false;
        return result;
    }
    for (std::size_t p = 0; p < 5; ++p) {
        const double rhs0 = -F.derivative[p];
        const double rhs1 = -Fp.derivative[p];
        result.derivative[p] = (rhs0 * Fpp_v - Fp_v * rhs1) / det;
    }
    return result;
}

}  // namespace

AlgebraicBoundaryJacobian experimental_algebraic_boundary_jacobian(
    const std::array<double, 5>& params,
    const FrozenLensBranch& branch,
    double c,
    double d,
    const std::vector<SourcePosition>& seeds,
    const AlgebraicBoundarySettings& settings,
    double grad_tol)
{
    AlgebraicBoundaryJacobian out;
    for (auto& j : out.jacobian) j = std::numeric_limits<double>::quiet_NaN();

    const double rho_d = params[2];
    const auto geom_d = make_geometry<double>(params, branch);
    const AlgebraicBoundaryLens lens_d{geom_d.a, geom_d.m1, geom_d.m2};
    const SourcePosition source_d{params[0], params[1]};

    if (grad_tol < 0.0) {
        grad_tol = std::clamp(10.0 * settings.relative_tolerance, 1.0e-4, 1.0e-1);
    }

    // Authoritative primal value.
    const auto primal = experimental_algebraic_boundary_integral(
        lens_d, source_d, rho_d, c, d, seeds, settings);
    out.magnification = primal.magnification;
    if (!primal.success) {
        out.unsafe_reason = primal.unsafe_reason;
        return out;
    }

    const auto bands = experimental_algebraic_boundary_radial_bands(
        lens_d, source_d, rho_d, seeds, settings);
    if (!bands.success || bands.bands.empty()) {
        out.unsafe_reason = bands.success ? "no radial bands" : bands.unsafe_reason;
        return out;
    }

    std::array<Jet, 5> pj;
    for (std::size_t i = 0; i < 5; ++i) pj[i] = Jet::variable(params[i], i);
    const auto geom_j = make_geometry<Jet>(pj, branch);

    std::array<JetR, 5> pj6;
    for (std::size_t i = 0; i < 5; ++i) pj6[i] = JetR::variable(params[i], i);
    const auto geom_r = make_geometry<JetR>(pj6, branch);

    const double norm_v = kPi * rho_d * rho_d * (1.0 - c / 3.0 - d / 5.0);
    if (!(norm_v > 0.0)) {
        out.unsafe_reason = "invalid normalization";
        return out;
    }

    bool reliable = true;

    Jet flux_k(0.0);
    Jet flux_g(0.0);

    for (const auto& band : bands.bands) {
        if (!reliable) break;  // an unreliable row is discarded; do not refine it.
        const double R_lo = band.first;
        const double R_hi = band.second;
        const Jet Rlo_j = endpoint_derivative(
            geom_r, lens_d, source_d, rho_d, R_lo, R_lo, R_hi, settings, &reliable);
        const Jet Rhi_j = endpoint_derivative(
            geom_r, lens_d, source_d, rho_d, R_hi, R_lo, R_hi, settings, &reliable);
        const Jet dR_j = Rhi_j - Rlo_j;

        MomentsJet<5> prev_k;
        bool have_prev = false;
        MomentsJet<5> acc_k;
        MomentsJet<5> acc_g;
        for (int nsub = 1; nsub <= 64; nsub *= 2) {
            if (!reliable) break;
            MomentsJet<5> tk;
            MomentsJet<5> tg;
            for (int s = 0; s < nsub; ++s) {
                const double t_a = static_cast<double>(s) / nsub;
                const double t_b = static_cast<double>(s + 1) / nsub;
                const double mid_t = 0.5 * (t_a + t_b);
                const double hw_t = 0.5 * (t_b - t_a);
                for (std::size_t i = 0; i < 15; ++i) {
                    const double t_i = mid_t + hw_t * smap15().coord[i];
                    const Jet Ri_j = Rlo_j + dR_j * t_i;
                    const Jet wR_j = (dR_j * hw_t) * smap15().jac[i];
                    const auto integ = radial_integrand_jet(
                        geom_j, lens_d, source_d, rho_d, Ri_j, c, d, settings, &reliable);
                    const Jet wk = wR_j * gk15().kw[i];
                    const Jet wg = wR_j * gk15().gw[i];
                    tk.m0 = tk.m0 + wk * integ.m0;
                    tk.mhalf = tk.mhalf + wk * integ.mhalf;
                    tk.mquarter = tk.mquarter + wk * integ.mquarter;
                    tg.m0 = tg.m0 + wg * integ.m0;
                    tg.mhalf = tg.mhalf + wg * integ.mhalf;
                    tg.mquarter = tg.mquarter + wg * integ.mquarter;
                }
            }
            acc_k = tk;
            acc_g = tg;
            if (have_prev) {
                const Jet a = combined_flux_jet<5>(tk, c, d);
                const Jet b = combined_flux_jet<5>(prev_k, c, d);
                double worst = std::fabs(a.value - b.value);
                for (std::size_t p = 0; p < 5; ++p)
                    worst = std::max(worst, std::fabs(a.derivative[p] - b.derivative[p]));
                if (worst <= 1.0e-10 * std::max(1.0, std::fabs(a.value))) break;
            }
            prev_k = tk;
            have_prev = true;
        }
        flux_k = flux_k + combined_flux_jet<5>(acc_k, c, d);
        flux_g = flux_g + combined_flux_jet<5>(acc_g, c, d);
    }

    Jet norm_j = kPi * (pj[2] * pj[2]) * (1.0 - c / 3.0 - d / 5.0);
    const Jet mag_j = flux_k / norm_j;

    double reduced = 0.0;
    for (std::size_t p = 0; p < 5; ++p) {
        const double est = std::fabs(flux_k.derivative[p] - flux_g.derivative[p]) / norm_v;
        reduced = std::max(reduced, est / std::max(1.0, std::fabs(mag_j.derivative[p])));
    }
    out.grad_error = reduced;

    out.success = true;
    out.grad_reliable = reliable && (reduced <= grad_tol);
    if (out.grad_reliable) {
        for (std::size_t p = 0; p < 5; ++p) out.jacobian[p] = mag_j.derivative[p];
    }
    return out;
}

}  // namespace lcbinint::magnification::autodiff
