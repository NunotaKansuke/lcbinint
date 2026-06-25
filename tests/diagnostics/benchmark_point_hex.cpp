#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Case {
    std::string name;
    double separation;
    double mass_ratio;
    lcbinint::SourcePosition source;
    double source_radius;
    double limb_darkening_c;
};

template <typename Func>
double best_ns_per_call(Func func, int iterations, int repeats, double& sink)
{
    double best = std::numeric_limits<double>::infinity();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        double local = 0.0;
        for (int i = 0; i < iterations; ++i) {
            local += func(i);
        }
        const auto stop = std::chrono::steady_clock::now();
        sink += local;
        const double ns = std::chrono::duration<double, std::nano>(stop - start).count();
        best = std::min(best, ns / static_cast<double>(iterations));
    }
    return best;
}

} // namespace

int main()
{
    using lcbinint::magnification::FiniteSourceSettings;
    using lcbinint::magnification::PointSourceMagnifier;
    using lcbinint::magnification::diagnostic_hexadecapole_binary;

    const std::vector<Case> cases = {
        {"planet_small", 1.0, 1.0e-3, {0.01, 0.0}, 1.0e-4, 0.0},
        {"planet_ld", 1.0, 1.0e-3, {0.01, 0.0}, 1.0e-4, 0.5},
        {"resonant", 1.0, 1.0e-1, {0.05, 0.02}, 3.0e-3, 0.0},
        {"close", 0.7, 3.0e-1, {0.05, 0.01}, 3.0e-3, 0.0},
        {"wide", 1.5, 1.0e-3, {0.02, -0.01}, 1.0e-4, 0.5},
    };

    const PointSourceMagnifier point_magnifier;
    const int point_iterations = 20000;
    const int hex_iterations = 2000;
    const int repeats = 5;
    double sink = 0.0;

    std::cout << "case point_warm_ns point_curve_ns point_cold_ns candidates_ns images_ns hex_ns hex_over_point point_mag hex_mag hex_relerr deriv_relerr\n";
    for (const auto& c : cases) {
        FiniteSourceSettings settings;
        settings.limb_darkening_c = c.limb_darkening_c;

        const double point_warm_ns = best_ns_per_call(
            [&](int i) {
                const double jitter = 1.0e-12 * static_cast<double>((i & 7) - 3);
                return point_magnifier
                    .binary_mag0(c.separation, c.mass_ratio, {c.source.x + jitter, c.source.y - jitter})
                    .magnification;
            },
            point_iterations, repeats, sink);

        const double point_curve_ns = best_ns_per_call(
            [&](int i) {
                const double phase =
                    -1.0 + 2.0 * static_cast<double>(i) /
                    static_cast<double>(std::max(point_iterations - 1, 1));
                return point_magnifier
                    .binary_mag0(c.separation, c.mass_ratio,
                        {c.source.x + 0.5 * phase, c.source.y + 0.2 * phase})
                    .magnification;
            },
            point_iterations, repeats, sink);

        const double point_cold_ns = best_ns_per_call(
            [&](int i) {
                const double jitter = 1.0e-12 * static_cast<double>((i & 7) - 3);
                const PointSourceMagnifier cold_point_magnifier;
                return cold_point_magnifier
                    .binary_mag0(c.separation, c.mass_ratio, {c.source.x + jitter, c.source.y - jitter})
                    .magnification;
            },
            point_iterations, repeats, sink);

        const double candidates_ns = best_ns_per_call(
            [&](int i) {
                const double jitter = 1.0e-12 * static_cast<double>((i & 7) - 3);
                const auto candidates = point_magnifier.binary_image_candidates(
                    c.separation, c.mass_ratio, {c.source.x + jitter, c.source.y - jitter});
                return static_cast<double>(candidates.size());
            },
            point_iterations, repeats, sink);

        const double images_ns = best_ns_per_call(
            [&](int i) {
                const double jitter = 1.0e-12 * static_cast<double>((i & 7) - 3);
                const auto images = point_magnifier.binary_images(
                    c.separation, c.mass_ratio, {c.source.x + jitter, c.source.y - jitter});
                return static_cast<double>(images.size());
            },
            point_iterations, repeats, sink);

        const double hex_ns = best_ns_per_call(
            [&](int i) {
                const double jitter = 1.0e-12 * static_cast<double>((i & 7) - 3);
                return diagnostic_hexadecapole_binary(
                    c.separation, c.mass_ratio, {c.source.x + jitter, c.source.y - jitter},
                    c.source_radius, settings)
                    .magnification;
            },
            hex_iterations, repeats, sink);

        const auto point = point_magnifier.binary_mag0(c.separation, c.mass_ratio, c.source);
        const auto hex = diagnostic_hexadecapole_binary(
            c.separation, c.mass_ratio, c.source, c.source_radius, settings);

        std::cout << std::setprecision(10)
                  << c.name << ' '
                  << point_warm_ns << ' '
                  << point_curve_ns << ' '
                  << point_cold_ns << ' '
                  << candidates_ns << ' '
                  << images_ns << ' '
                  << hex_ns << ' '
                  << hex_ns / point_warm_ns << ' '
                  << point.magnification << ' '
                  << hex.magnification << ' '
                  << hex.relative_error << ' '
                  << hex.derivative_relative_error << '\n';
    }

    if (!std::isfinite(sink)) {
        return 2;
    }
    return 0;
}
