#include "lcbinint/magnification/finite_source_magnifier.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using lcbinint::SourcePosition;
using lcbinint::magnification::AlgebraicBoundaryResult;
using lcbinint::magnification::AlgebraicBoundarySettings;
using lcbinint::magnification::FiniteSourceMagnifier;
using lcbinint::magnification::FiniteSourceSettings;

struct TimedResult {
    AlgebraicBoundaryResult result;
    double seconds = 0.0;
    double warmup_seconds = 0.0;
};

template <typename Evaluate>
TimedResult timed(
    Evaluate&& evaluate, int repeats)
{
    // The first call warms the finite-source object's caustic cache and the
    // process-local allocator state. Only subsequent calls are reported.
    const auto warmup_started = Clock::now();
    AlgebraicBoundaryResult result = evaluate();
    const double warmup_seconds = std::chrono::duration<double>(
        Clock::now() - warmup_started).count();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto started = Clock::now();
        result = evaluate();
        samples.push_back(std::chrono::duration<double>(
            Clock::now() - started).count());
    }
    std::sort(samples.begin(), samples.end());
    return {result, samples[samples.size() / 2], warmup_seconds};
}

std::string safe_reason(const std::string& reason)
{
    std::string result = reason;
    for (char& character : result) {
        if (character == '\t' || character == '\n' || character == '\r' ||
            character == ' ') {
            character = '_';
        }
    }
    return result;
}

void print_result(
    const char* method,
    int job,
    int epoch,
    const AlgebraicBoundaryResult& result,
    double seconds,
    double warmup_seconds)
{
    std::cout << job << '\t' << epoch << '\t' << method << '\t'
              << (result.success ? 1 : 0) << '\t'
              << std::setprecision(17) << result.magnification << '\t'
              << seconds << '\t'
              << result.diagnostics.root_solve_count << '\t'
              << result.diagnostics.real_quartic_solve_count << '\t'
              << result.diagnostics.real_quartic_fallback_count << '\t'
              << result.diagnostics.tangency_solve_count << '\t'
              << result.diagnostics.tangency_fallback_count << '\t'
              << safe_reason(result.unsafe_reason) << '\t'
              << warmup_seconds << '\t'
              << result.diagnostics.real_sextic_solve_count << '\t'
              << result.diagnostics.real_sextic_fallback_count << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cerr << "usage: " << argv[0]
                  << " REPEATS TARGET LIMB_C LIMB_D < input.tsv\n"
                  << "input columns: job epoch separation mass_ratio source_x "
                     "source_y source_radius\n";
        return 2;
    }
    const int repeats = std::max(1, std::stoi(argv[1]));
    const double target = std::stod(argv[2]);
    const double limb_c = std::stod(argv[3]);
    const double limb_d = std::stod(argv[4]);
    if (!(target > 0.0) || !std::isfinite(target) ||
        !std::isfinite(limb_c) || !std::isfinite(limb_d)) {
        std::cerr << "invalid target or limb-darkening parameters\n";
        return 2;
    }

    FiniteSourceSettings finite_settings;
    finite_settings.limb_darkening_c = limb_c;
    finite_settings.limb_darkening_d = limb_d;
    // Keep method caches independent so the reported warm-up cost for
    // Cartesian is not accidentally hidden by a preceding polar call.
    FiniteSourceMagnifier polar_magnifier(finite_settings);
    FiniteSourceMagnifier cartesian_magnifier(finite_settings);
    AlgebraicBoundarySettings settings;
    settings.absolute_tolerance = target;
    settings.relative_tolerance = target;
    settings.maximum_radial_depth = 26;
    settings.unit_circle_tolerance = 2.0e-6;
    settings.boundary_residual_tolerance = 2.0e-8;

    int job = 0;
    int epoch = 0;
    double separation = 0.0;
    double mass_ratio = 0.0;
    double source_x = 0.0;
    double source_y = 0.0;
    double source_radius = 0.0;
    while (std::cin >> job >> epoch >> separation >> mass_ratio >>
           source_x >> source_y >> source_radius) {
        const auto polar = timed([&] {
            return polar_magnifier.experimental_algebraic_boundary_binary_mag(
                separation, mass_ratio,
                SourcePosition {source_x, source_y}, source_radius, settings);
        }, repeats);
        const auto cartesian = timed([&] {
            return cartesian_magnifier.experimental_algebraic_cartesian_binary_mag(
                separation, mass_ratio,
                SourcePosition {source_x, source_y}, source_radius, settings);
        }, repeats);
        print_result("polar", job, epoch, polar.result, polar.seconds,
                     polar.warmup_seconds);
        print_result("cartesian", job, epoch, cartesian.result,
                     cartesian.seconds, cartesian.warmup_seconds);
    }
    if (!std::cin.eof()) {
        std::cerr << "malformed input row\n";
        return 3;
    }
    return 0;
}
