#include "lcbinint/magnification/algebraic_boundary.hpp"
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/math/polynomial_roots.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

lcbinint::magnification::AlgebraicBoundaryLens native_lens(double separation, double mass_ratio)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const double m1 = 1.0 / (1.0 + q);
    return {q_input < 1.0 ? -s : s, m1, q * m1};
}

lcbinint::SourcePosition map_lens(
    const lcbinint::magnification::AlgebraicBoundaryLens& lens,
    double x,
    double y)
{
    const double xa = x - lens.lens_position;
    const double d1 = xa * xa + y * y;
    const double d2 = x * x + y * y;
    return {
        x - lens.m1 * xa / d1 - lens.m2 * x / d2 - lens.lens_position * lens.m1,
        y - lens.m1 * y / d1 - lens.m2 * y / d2,
    };
}

double phi(
    const lcbinint::magnification::AlgebraicBoundaryLens& lens,
    lcbinint::SourcePosition source,
    double rho,
    double radius,
    double angle)
{
    const auto mapped = map_lens(lens, radius * std::cos(angle), radius * std::sin(angle));
    const double dx = mapped.x - source.x;
    const double dy = mapped.y - source.y;
    return 1.0 - (dx * dx + dy * dy) / (rho * rho);
}

bool test_quartic_identity()
{
    const auto lens = native_lens(1.17, 0.031);
    const lcbinint::SourcePosition source {0.074, -0.026};
    const double rho = 0.013;
    for (const double radius : {0.21, 0.73, 1.42}) {
        const auto coefficients = lcbinint::magnification::algebraic_boundary_quartic(
            lens, source, rho, radius);
        for (int i = 0; i < 31; ++i) {
            const double angle = 2.0 * kPi * (i + 0.37) / 31.0;
            const lcbinint::Complex u = std::polar(1.0, angle);
            const auto polynomial =
                lcbinint::math::PolynomialRootSolver::evaluate(coefficients, u) / (u * u);
            const double a = lens.lens_position;
            const lcbinint::Complex shifted(source.x + a * lens.m1, source.y);
            const lcbinint::Complex numerator =
                -shifted * radius * radius +
                radius * (radius * radius - 1.0 + a * shifted) * u +
                a * (lens.m2 - radius * radius) * u * u;
            const lcbinint::Complex denominator =
                radius * radius - a * radius * u;
            const double expected =
                std::norm(numerator) - rho * rho * std::norm(denominator);
            if (std::abs(polynomial.imag()) > 2.0e-11 * (1.0 + std::abs(expected)) ||
                std::abs(polynomial.real() - expected) >
                    2.0e-11 * (1.0 + std::abs(expected))) {
                std::cerr << "quartic identity failed R=" << radius
                          << " angle=" << angle << '\n';
                return false;
            }
        }
    }
    return true;
}

bool test_cartesian_sextic_identity()
{
    const auto lens = native_lens(1.17, 0.031);
    const lcbinint::SourcePosition source {0.074, -0.026};
    const double rho = 0.013;
    for (const double image_x : {-0.8, 0.17, 1.4}) {
        const auto coefficients =
            lcbinint::magnification::algebraic_cartesian_boundary_sextic(
                lens, source, rho, image_x);
        for (int i = 0; i < 41; ++i) {
            const double y = -2.0 + 4.0 * (i + 0.31) / 41.0;
            const auto polynomial = lcbinint::math::PolynomialRootSolver::evaluate(
                coefficients, lcbinint::Complex(y, 0.0));
            const lcbinint::Complex z(image_x, y);
            const lcbinint::Complex zbar(image_x, -y);
            const lcbinint::Complex shifted(source.x + lens.lens_position * lens.m1, source.y);
            const lcbinint::Complex denominator =
                (zbar - lens.lens_position) * zbar;
            const lcbinint::Complex numerator =
                (z - shifted) * denominator - lens.m1 * zbar -
                lens.m2 * (zbar - lens.lens_position);
            const double expected = std::norm(numerator) - rho * rho * std::norm(denominator);
            if (std::abs(polynomial.imag()) > 2.0e-11 * (1.0 + std::abs(expected)) ||
                std::abs(polynomial.real() - expected) >
                    2.0e-11 * (1.0 + std::abs(expected))) {
                std::cerr << "Cartesian sextic identity failed x=" << image_x
                          << " y=" << y << '\n';
                return false;
            }
        }
    }
    return true;
}

bool test_intervals_cover_dense_crossings()
{
    const auto lens = native_lens(1.1, 1.0e-3);
    const lcbinint::SourcePosition source {0.002, 0.0007};
    const double rho = 0.001;
    lcbinint::magnification::AlgebraicBoundarySettings settings;
    settings.unit_circle_tolerance = 2.0e-6;
    settings.boundary_residual_tolerance = 2.0e-8;

    lcbinint::magnification::PointSourceMagnifier point;
    const auto images = point.binary_images(1.1, 1.0e-3, source);
    if (images.empty()) return false;
    for (const auto& image : images) {
        const double radius = std::hypot(image.position.x, image.position.y);
        const auto intervals =
            lcbinint::magnification::algebraic_boundary_intervals_at_radius(
                lens, source, rho, radius, settings);
        if (!intervals.success || intervals.inside_intervals.empty()) {
            std::cerr << "active image radius not classified: "
                      << intervals.unsafe_reason << '\n';
            return false;
        }
        constexpr int samples = 32768;
        int dense_crossings = 0;
        double previous = phi(lens, source, rho, radius, 0.0);
        for (int i = 1; i <= samples; ++i) {
            const double angle = 2.0 * kPi * i / static_cast<double>(samples);
            const double current = phi(lens, source, rho, radius, angle);
            if ((previous < 0.0 && current > 0.0) ||
                (previous > 0.0 && current < 0.0)) {
                ++dense_crossings;
            }
            previous = current;
        }
        if (dense_crossings != static_cast<int>(intervals.boundary_angles.size())) {
            std::cerr << "dense crossing mismatch: dense=" << dense_crossings
                      << " roots=" << intervals.boundary_angles.size() << '\n';
            return false;
        }
        for (const double angle : intervals.boundary_angles) {
            if (std::abs(phi(lens, source, rho, radius, angle)) > 2.0e-7) {
                std::cerr << "boundary residual too large\n";
                return false;
            }
        }
    }
    return true;
}

bool test_cartesian_intervals_cover_dense_crossings()
{
    const auto lens = native_lens(1.1, 1.0e-3);
    const lcbinint::SourcePosition source {0.002, 0.0007};
    const double rho = 0.001;
    lcbinint::magnification::AlgebraicBoundarySettings settings;
    settings.unit_circle_tolerance = 2.0e-6;
    settings.boundary_residual_tolerance = 2.0e-8;
    lcbinint::magnification::PointSourceMagnifier point;
    const auto images = point.binary_images(1.1, 1.0e-3, source);
    for (const auto& image : images) {
        const double x = image.position.x;
        const auto intervals = lcbinint::magnification::algebraic_cartesian_intervals_at_x(
            lens, source, rho, x, settings);
        if (!intervals.success || intervals.inside_intervals.empty()) {
            std::cerr << "active image x not classified: " << intervals.unsafe_reason << '\n';
            return false;
        }
        int dense_crossings = 0;
        constexpr int samples = 65536;
        constexpr double limit = 5.0;
        auto signed_distance = [&](double y) {
            const auto mapped = map_lens(lens, x, y);
            const double dx = mapped.x - source.x;
            const double dy = mapped.y - source.y;
            return rho * rho - dx * dx - dy * dy;
        };
        double previous = signed_distance(-limit);
        for (int i = 1; i <= samples; ++i) {
            const double y = -limit + 2.0 * limit * i / static_cast<double>(samples);
            const double current = signed_distance(y);
            if ((previous < 0.0 && current > 0.0) ||
                (previous > 0.0 && current < 0.0)) {
                ++dense_crossings;
            }
            previous = current;
        }
        if (dense_crossings != static_cast<int>(intervals.boundary_coordinates.size())) {
            std::cerr << "Cartesian dense crossing mismatch: dense=" << dense_crossings
                      << " roots=" << intervals.boundary_coordinates.size() << '\n';
            return false;
        }
    }
    return true;
}

bool test_interval_edge_cases()
{
    const auto lens = native_lens(1.1, 1.0e-3);
    lcbinint::magnification::AlgebraicBoundarySettings settings;
    settings.unit_circle_tolerance = 2.0e-6;
    settings.boundary_residual_tolerance = 2.0e-8;

    const auto empty = lcbinint::magnification::algebraic_boundary_intervals_at_radius(
        lens, {0.1, 0.1}, 0.01, 20.0, settings);
    if (!empty.success || !empty.inside_intervals.empty()) {
        std::cerr << "far-radius empty classification failed\n";
        return false;
    }
    const auto full = lcbinint::magnification::algebraic_boundary_intervals_at_radius(
        lens, {0.0, 0.0}, 20.0, 0.7, settings);
    if (!full.success || !full.full_circle || full.inside_intervals.size() != 1) {
        std::cerr << "full-circle classification failed\n";
        return false;
    }
    const auto pole = lcbinint::magnification::algebraic_boundary_intervals_at_radius(
        lens, {0.0, 0.0}, 0.01, 0.0, settings);
    if (pole.success || pole.unsafe_reason.empty()) {
        std::cerr << "collapsed polar circle was not rejected\n";
        return false;
    }
    return true;
}

bool test_magnification_profiles()
{
    constexpr double separation = 1.1;
    constexpr double mass_ratio = 1.0e-3;
    const lcbinint::SourcePosition source {0.08, 0.03};
    constexpr double rho = 0.01;
    for (const auto profile : std::vector<std::pair<double, double>> {
             {0.0, 0.0}, {0.5, 0.0}, {0.25, 0.35}}) {
        lcbinint::magnification::FiniteSourceSettings native_settings;
        native_settings.finite_mode = 2;
        native_settings.source_bins = 320;
        native_settings.polar_source_bins = 320;
        native_settings.polar_grid_ratio = 4.0;
        native_settings.automatic_source_bins = false;
        native_settings.limb_darkening_c = profile.first;
        native_settings.limb_darkening_d = profile.second;
        lcbinint::magnification::FiniteSourceMagnifier magnifier(native_settings);

        lcbinint::magnification::AlgebraicBoundarySettings algebraic_settings;
        algebraic_settings.absolute_tolerance = 1.0e-4;
        algebraic_settings.relative_tolerance = 1.0e-4;
        algebraic_settings.maximum_radial_depth = 22;
        algebraic_settings.unit_circle_tolerance = 2.0e-6;
        algebraic_settings.boundary_residual_tolerance = 2.0e-8;
        const auto algebraic = magnifier.experimental_algebraic_boundary_binary_mag(
            separation, mass_ratio, source, rho, algebraic_settings);
        if (!algebraic.success) {
            std::cerr << "algebraic integration failed: " << algebraic.unsafe_reason << '\n';
            return false;
        }
        const auto cartesian = magnifier.experimental_algebraic_cartesian_binary_mag(
            separation, mass_ratio, source, rho, algebraic_settings);
        if (!cartesian.success) {
            std::cerr << "Cartesian algebraic integration failed: "
                      << cartesian.unsafe_reason << '\n';
            return false;
        }
        const auto continuation_is_accounted = [](const auto& diagnostics) {
            return diagnostics.cold_root_solve_count > 0 &&
                (diagnostics.warm_root_solve_count > 0 ||
                    diagnostics.real_quartic_solve_count > 0) &&
                diagnostics.real_quartic_solve_count +
                diagnostics.warm_root_solve_count +
                        diagnostics.cold_root_solve_count ==
                    diagnostics.root_solve_count + diagnostics.cold_retry_count;
        };
        if (!continuation_is_accounted(algebraic.diagnostics) ||
            !continuation_is_accounted(cartesian.diagnostics)) {
            std::cerr << "root-continuation diagnostics were inconsistent\n";
            return false;
        }
        if (algebraic.diagnostics.tangency_solve_count == 0) {
            std::cerr << "radial tangency fast path was not exercised\n";
            return false;
        }

        lcbinint::magnification::PointSourceMagnifier point;
        const auto point_result = point.binary_mag0(separation, mass_ratio, source);
        const auto native = magnifier.binary_mag(
            separation, mass_ratio, source, rho, point_result.magnification,
            nullptr, true, &point);
        const double relative = std::abs(algebraic.magnification - native.magnification) /
            std::max(std::abs(native.magnification), 1.0);
        if (!std::isfinite(native.magnification) || relative > 4.0e-4) {
            std::cerr << "magnification mismatch c=" << profile.first
                      << " d=" << profile.second
                      << " algebraic=" << algebraic.magnification
                      << " native=" << native.magnification
                      << " rel=" << relative << '\n';
            return false;
        }
        const double cartesian_relative =
            std::abs(cartesian.magnification - native.magnification) /
            std::max(std::abs(native.magnification), 1.0);
        if (cartesian_relative > 4.0e-4) {
            std::cerr << "Cartesian magnification mismatch c=" << profile.first
                      << " d=" << profile.second
                      << " Cartesian=" << cartesian.magnification
                      << " native=" << native.magnification
                      << " rel=" << cartesian_relative << '\n';
            return false;
        }
    }
    return true;
}

bool test_redundant_seeds_preserve_integral()
{
    constexpr double separation = 1.1;
    constexpr double mass_ratio = 1.0e-3;
    const lcbinint::SourcePosition source {0.08, 0.03};
    constexpr double rho = 0.01;
    const auto lens = native_lens(separation, mass_ratio);
    lcbinint::magnification::PointSourceMagnifier point;
    const auto images = point.binary_images(separation, mass_ratio, source);
    std::vector<lcbinint::SourcePosition> seeds;
    for (const auto& image : images) seeds.push_back(image.position);
    if (seeds.empty()) return false;
    auto redundant = seeds;
    redundant.insert(redundant.end(), seeds.begin(), seeds.end());
    redundant.insert(redundant.end(), seeds.begin(), seeds.end());

    lcbinint::magnification::AlgebraicBoundarySettings settings;
    settings.absolute_tolerance = 1.0e-4;
    settings.relative_tolerance = 1.0e-4;
    settings.unit_circle_tolerance = 2.0e-6;
    settings.boundary_residual_tolerance = 2.0e-8;
    const auto polar = lcbinint::magnification::experimental_algebraic_boundary_integral(
        lens, source, rho, 0.5, 0.0, seeds, settings);
    const auto redundant_polar =
        lcbinint::magnification::experimental_algebraic_boundary_integral(
            lens, source, rho, 0.5, 0.0, redundant, settings);
    const auto cartesian =
        lcbinint::magnification::experimental_algebraic_cartesian_integral(
            lens, source, rho, 0.5, 0.0, seeds, settings);
    const auto redundant_cartesian =
        lcbinint::magnification::experimental_algebraic_cartesian_integral(
            lens, source, rho, 0.5, 0.0, redundant, settings);
    if (!polar.success || !redundant_polar.success ||
        !cartesian.success || !redundant_cartesian.success) {
        std::cerr << "redundant-seed integration failed\n";
        return false;
    }
    const auto close = [](double lhs, double rhs) {
        return std::abs(lhs - rhs) <= 2.0e-11 * std::max({std::abs(lhs), std::abs(rhs), 1.0});
    };
    if (!close(polar.magnification, redundant_polar.magnification) ||
        !close(cartesian.magnification, redundant_cartesian.magnification)) {
        std::cerr << "redundant seeds changed algebraic integral\n";
        return false;
    }
    return true;
}

bool test_ill_conditioned_real_quartic_falls_back()
{
    constexpr double separation = 0.8;
    constexpr double mass_ratio = 1.0e-6;
    constexpr double centre_x = -0.45054318004640337;
    constexpr double centre_y = -0.0014718037129124418;
    constexpr double rho = 3.0e-5;
    lcbinint::magnification::FiniteSourceSettings finite_settings;
    finite_settings.limb_darkening_c = 0.5;
    lcbinint::magnification::FiniteSourceMagnifier magnifier(finite_settings);
    lcbinint::magnification::AlgebraicBoundarySettings settings;
    settings.absolute_tolerance = 1.0e-3;
    settings.relative_tolerance = 1.0e-3;
    settings.unit_circle_tolerance = 2.0e-6;
    settings.boundary_residual_tolerance = 2.0e-8;
    int fallbacks = 0;
    for (int index = 0; index < 24; ++index) {
        const double fraction = static_cast<double>(index) / 23.0;
        const lcbinint::SourcePosition source {
            centre_x - (fraction - 0.5) * 0.4 * rho, centre_y};
        const auto result = magnifier.experimental_algebraic_boundary_binary_mag(
            separation, mass_ratio, source, rho, settings);
        if (!result.success || !std::isfinite(result.magnification)) {
            std::cerr << "ill-conditioned quartic fallback failed: "
                      << result.unsafe_reason << '\n';
            return false;
        }
        fallbacks += result.diagnostics.real_quartic_fallback_count;
    }
    if (fallbacks == 0) {
        std::cerr << "ill-conditioned quartic did not exercise SG fallback\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!test_quartic_identity()) return 1;
    if (!test_cartesian_sextic_identity()) return 2;
    if (!test_intervals_cover_dense_crossings()) return 3;
    if (!test_cartesian_intervals_cover_dense_crossings()) return 4;
    if (!test_interval_edge_cases()) return 5;
    if (!test_magnification_profiles()) return 6;
    if (!test_redundant_seeds_preserve_integral()) return 7;
    if (!test_ill_conditioned_real_quartic_falls_back()) return 8;
    std::cout << "algebraic boundary tests passed\n";
    return 0;
}
