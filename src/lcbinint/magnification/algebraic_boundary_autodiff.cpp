// Analytic point-source and hexadecapole tangents shared by the selector.
#include "lcbinint/magnification/algebraic_boundary_autodiff.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>
namespace lcbinint::magnification::autodiff {
namespace {
using Jet = ForwardJet<5>;

// Binary point-source magnification carried through a ForwardJet<5> pass.
// The physical image set is frozen from the primal double roots; each root's
// parameter derivative comes from one implicit-function step on the lens
// equation, then 1/|det A| is re-accumulated in Jet arithmetic.  Frame matches
// map_lens_jet / BinaryLensMapper (lenses at {a, 0}, the -a*m1 source shift
// folded into the map, source compared raw) -- the same frame binary_images()
// returns positions in.
Jet binary_point_mag_jet(
    const BinaryGeometry<Jet>& g, const std::vector<BinaryImage>& frozen_images)
{
    Jet total(0.0);
    for (const auto& image : frozen_images) {
        const double x0 = image.position.x;
        const double y0 = image.position.y;

        // Residual r(z0; p) = map_lens(z0; p) - source(p) with z0 held fixed.
        Jet mx;
        Jet my;
        map_lens_jet(g, Jet(x0), Jet(y0), &mx, &my);
        const Jet rx = mx - g.src_x;
        const Jet ry = my - g.src_y;

        // Frozen 2x2 Jacobian of the lens map wrt (x, y) at z0 (holomorphic:
        // fyy = 2 - fxx, fxy = fyx).
        const double a = g.a.value;
        const double m1 = g.m1.value;
        const double m2 = g.m2.value;
        const double dxa = x0 - a;
        const double d1 = dxa * dxa + y0 * y0;
        const double d2 = x0 * x0 + y0 * y0;
        const double inv_d1sq = 1.0 / (d1 * d1);
        const double inv_d2sq = 1.0 / (d2 * d2);
        const double fxx = 1.0
            - m1 * (y0 * y0 - dxa * dxa) * inv_d1sq
            - m2 * (y0 * y0 - x0 * x0) * inv_d2sq;
        const double fxy =
            2.0 * y0 * (m1 * dxa * inv_d1sq + m2 * x0 * inv_d2sq);
        const double fyy = 2.0 - fxx;
        const double detJ = fxx * fyy - fxy * fxy;
        if (!(std::fabs(detJ) > 1.0e-14)) {
            continue;
        }
        const double inv_detJ = 1.0 / detJ;

        // Solve J . dz/dp = -dr/dp for every parameter lane.
        Jet zx(x0);
        Jet zy(y0);
        for (std::size_t p = 0; p < 5; ++p) {
            const double bx = -rx.derivative[p];
            const double by = -ry.derivative[p];
            zx.derivative[p] = (fyy * bx - fxy * by) * inv_detJ;
            zy.derivative[p] = (-fxy * bx + fxx * by) * inv_detJ;
        }

        // Re-accumulate 1/|det A| with z(p) and g(p) both live.
        const Jet za = zx - g.a;
        const Jet D1 = za * za + zy * zy;
        const Jet D2 = zx * zx + zy * zy;
        const Jet Fxx = 1.0
            - g.m1 * (zy * zy - za * za) / (D1 * D1)
            - g.m2 * (zy * zy - zx * zx) / (D2 * D2);
        const Jet Fxy =
            2.0 * zy * (g.m1 * za / (D1 * D1) + g.m2 * zx / (D2 * D2));
        const Jet Fyy = 2.0 - Fxx;
        const Jet det = Fxx * Fyy - Fxy * Fxy;
        const double sign = det.value < 0.0 ? -1.0 : 1.0;
        total = total + sign / det;
    }
    return total;
}

}  // namespace

AlgebraicBoundaryJacobian algebraic_boundary_multipole_jacobian(
    const std::array<double, 5>& params,
    const FrozenLensBranch& branch,
    double c,
    double d,
    bool use_hexadecapole,
    double authoritative_magnification)
{
    AlgebraicBoundaryJacobian out;
    for (auto& j : out.jacobian) j = std::numeric_limits<double>::quiet_NaN();
    out.magnification = authoritative_magnification;
    out.primal_magnification = authoritative_magnification;
    out.primal_success = std::isfinite(authoritative_magnification) &&
        authoritative_magnification > 0.0;
    // The point / multipole stencil has no quadrature error estimate in this
    // ABI. The caller reports this as an explicit unavailable diagnostic.
    out.value_error = std::numeric_limits<double>::quiet_NaN();

    std::array<Jet, 5> pj;
    for (std::size_t i = 0; i < 5; ++i) pj[i] = Jet::variable(params[i], i);

    const PointSourceMagnifier point_magnifier;
    const auto geom_j = make_geometry<Jet>(pj, branch);
    // binary_images() takes abs(separation) / abs(mass_ratio) internally, so
    // the frozen |params| reproduce the same lens frame as the caller's raw
    // (s, q); the same frame binary_point_mag_jet / map_lens_jet assume.
    const double abs_sep = params[4];
    const double abs_q = params[3];
    const SourcePosition source_d{params[0], params[1]};

    const auto mag_at = [&](double ux, double uy) -> Jet {
        std::array<Jet, 5> ps = pj;
        ps[0] = pj[0] + ux * pj[2];
        ps[1] = pj[1] + uy * pj[2];
        const auto g = make_geometry<Jet>(ps, branch);
        const SourcePosition s{
            params[0] + ux * params[2], params[1] + uy * params[2]};
        const auto images =
            point_magnifier.binary_images(abs_sep, abs_q, s);
        return binary_point_mag_jet(g, images);
    };

    if (!use_hexadecapole) {
        const auto images =
            point_magnifier.binary_images(abs_sep, abs_q, source_d);
        // One quintic image solve followed by one lens-map/Jacobian evaluation
        // per physical image.  These are actual point-driver work counts, not
        // availability sentinels; the caller adds its own primal solve work.
        out.diagnostics.root_solve_count = 1;
        out.diagnostics.lens_evaluation_count = static_cast<int>(images.size());
        const Jet mag_j = binary_point_mag_jet(geom_j, images);
        // Report the reconstruction's own value so it is exactly the function
        // whose derivative this row carries (see the deep driver above).
        if (std::isfinite(mag_j.value) && mag_j.value > 0.0) {
            out.magnification = mag_j.value;
        }
        for (std::size_t p = 0; p < 5; ++p) out.jacobian[p] = mag_j.derivative[p];
        out.success = true;
        out.grad_reliable = true;
        return out;
    }

    // 13-point hexadecapole stencil (mirrors hexadecapole_binary): centre, the
    // rho ring at 0/90/180/270 deg, the rho/2 ring, and the rho ring at the
    // 45-deg diagonals.
    constexpr double kSqrtHalf = 0.70710678118654752440;
    const Jet a0 = mag_at(0.0, 0.0);
    const Jet a1_plus =
        0.25 * (mag_at(1.0, 0.0) + mag_at(0.0, 1.0)
                + mag_at(-1.0, 0.0) + mag_at(0.0, -1.0))
        - a0;
    const Jet a2_plus =
        0.25 * (mag_at(0.5, 0.0) + mag_at(0.0, 0.5)
                + mag_at(-0.5, 0.0) + mag_at(0.0, -0.5))
        - a0;
    const Jet a1_cross =
        0.25 * (mag_at(kSqrtHalf, kSqrtHalf) + mag_at(-kSqrtHalf, kSqrtHalf)
                + mag_at(-kSqrtHalf, -kSqrtHalf) + mag_at(kSqrtHalf, -kSqrtHalf))
        - a0;
    const Jet a2rho2 = (16.0 * a2_plus - a1_plus) / 3.0;
    const Jet a4rho4 = 0.5 * (a1_plus + a1_cross) - a2rho2;

    const double denominator = 15.0 - 5.0 * c - 3.0 * d;
    const double gamma = denominator != 0.0 ? 10.0 * c / denominator : 0.0;
    const double lambda = denominator != 0.0 ? 12.0 * d / denominator : 0.0;
    const Jet mag_j = a0
        + 0.5 * a2rho2 * (1.0 - 0.2 * gamma - lambda / 9.0)
        + a4rho4 / 3.0 * (1.0 - 11.0 * gamma / 35.0 - 7.0 * lambda / 39.0);

    if (std::isfinite(mag_j.value) && mag_j.value > 0.0) {
        out.magnification = mag_j.value;
    }
    for (std::size_t p = 0; p < 5; ++p) out.jacobian[p] = mag_j.derivative[p];
    out.success = true;
    out.grad_reliable = true;
    return out;
}

AlgebraicBoundaryJacobian algebraic_boundary_point_jacobian(
    const std::array<double, 5>& params,
    const FrozenLensBranch& branch,
    const std::vector<SourcePosition>& frozen_images,
    double authoritative_magnification)
{
    AlgebraicBoundaryJacobian out;
    out.magnification = authoritative_magnification;
    out.primal_magnification = authoritative_magnification;
    out.primal_success = std::isfinite(authoritative_magnification) &&
        authoritative_magnification > 0.0;
    if (!out.primal_success || frozen_images.empty()) {
        out.unsafe_reason = frozen_images.empty()
            ? "point-source image set is empty"
            : "point-source magnification is not finite and positive";
        return out;
    }

    std::array<Jet, 5> pj;
    for (std::size_t i = 0; i < 5; ++i)
        pj[i] = Jet::variable(params[i], i);
    std::vector<BinaryImage> images;
    images.reserve(frozen_images.size());
    for (const auto& position : frozen_images) {
        images.push_back({position, 0.0});
    }
    const auto geom_j = make_geometry<Jet>(pj, branch);
    const Jet mag_j = binary_point_mag_jet(geom_j, images);
    if (!std::isfinite(mag_j.value) || mag_j.value <= 0.0) {
        out.unsafe_reason = "point-source ForwardJet reconstruction is invalid";
        return out;
    }
    for (std::size_t p = 0; p < 5; ++p) {
        if (!std::isfinite(mag_j.derivative[p])) {
            out.unsafe_reason = "point-source ForwardJet derivative is invalid";
            return out;
        }
        out.jacobian[p] = mag_j.derivative[p];
    }
    out.success = true;
    out.grad_reliable = true;
    out.diagnostics.lens_evaluation_count =
        static_cast<int>(images.size());
    return out;
}

}  // namespace lcbinint::magnification::autodiff
