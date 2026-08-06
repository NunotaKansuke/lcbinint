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

struct Case {
    const char* name;
    const char* family;
    double separation;
    double mass_ratio;
    lcbinint::SourcePosition source;
    double rho;
};

struct Profile {
    const char* name;
    double c;
    double d;
};

template <typename Callable>
std::pair<double, double> timed(Callable&& callable, int repeats)
{
    double value = callable();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto started = Clock::now();
        value = callable();
        samples.push_back(std::chrono::duration<double>(Clock::now() - started).count());
    }
    std::sort(samples.begin(), samples.end());
    return {value, samples[samples.size() / 2]};
}

double relative_error(double value, double reference)
{
    return std::abs(value - reference) / std::max(std::abs(reference), 1.0);
}

double percentile(std::vector<double> values, double probability)
{
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(std::ceil(
        probability * static_cast<double>(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
}

} // namespace

int main(int argc, char** argv)
{
    int repeats = 5;
    if (argc >= 2) repeats = std::max(1, std::stoi(argv[1]));
    const std::string suite = argc >= 3 ? argv[2] : "core";
    if (suite == "block") {
        if (argc != 10) {
            std::cerr << "usage: " << argv[0]
                      << " REPEATS block S Q X Y RHO C D\n";
            return 2;
        }
        const double separation = std::stod(argv[3]);
        const double mass_ratio = std::stod(argv[4]);
        const double centre_x = std::stod(argv[5]);
        const double centre_y = std::stod(argv[6]);
        const double rho = std::stod(argv[7]);
        const double profile_c = std::stod(argv[8]);
        const double profile_d = std::stod(argv[9]);
        constexpr int epochs = 24;
        std::vector<lcbinint::SourcePosition> sources;
        sources.reserve(epochs);
        for (int index = 0; index < epochs; ++index) {
            const double fraction = static_cast<double>(index) / (epochs - 1);
            // The recal2026 corpus records the public VBM-compatible
            // trajectory frame.  This block mode receives its mirrored
            // internal centre, so increasing public time runs toward
            // decreasing internal x.
            sources.push_back({centre_x - (fraction - 0.5) * 0.4 * rho, centre_y});
        }

        lcbinint::magnification::FiniteSourceSettings finite_settings;
        finite_settings.limb_darkening_c = profile_c;
        finite_settings.limb_darkening_d = profile_d;
        lcbinint::magnification::FiniteSourceMagnifier engine(finite_settings);
        const std::vector<double> tolerances {
            1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5,
            1.0e-5, 3.0e-6};
        std::cout << "tolerance,polar_seconds_per_epoch,cart_seconds_per_epoch,"
                     "polar_success_epochs,cart_success_epochs,polar_unsafe,cart_unsafe,"
                     "polar_values,cart_values\n";
        for (const double tolerance : tolerances) {
            lcbinint::magnification::AlgebraicBoundarySettings settings;
            settings.absolute_tolerance = tolerance;
            settings.relative_tolerance = tolerance;
            settings.maximum_radial_depth = 26;
            settings.unit_circle_tolerance = 2.0e-6;
            settings.boundary_residual_tolerance = 2.0e-8;
            std::vector<lcbinint::magnification::AlgebraicBoundaryResult> polar_results;
            std::vector<lcbinint::magnification::AlgebraicBoundaryResult> cart_results;
            const auto polar_timing = timed([&] {
                polar_results.clear();
                double checksum = 0.0;
                for (const auto& source : sources) {
                    polar_results.push_back(
                        engine.experimental_algebraic_boundary_binary_mag(
                            separation, mass_ratio, source, rho, settings));
                    checksum += polar_results.back().magnification;
                }
                return checksum;
            }, repeats);
            const auto cart_timing = timed([&] {
                cart_results.clear();
                double checksum = 0.0;
                for (const auto& source : sources) {
                    cart_results.push_back(
                        engine.experimental_algebraic_cartesian_binary_mag(
                            separation, mass_ratio, source, rho, settings));
                    checksum += cart_results.back().magnification;
                }
                return checksum;
            }, repeats);
            const auto successes = [](const auto& results) {
                return std::count_if(results.begin(), results.end(),
                    [](const auto& result) { return result.success; });
            };
            const auto values = [](const auto& results) {
                std::string output;
                for (std::size_t index = 0; index < results.size(); ++index) {
                    if (index) output += ';';
                    output += std::to_string(results[index].magnification);
                }
                return output;
            };
            const auto first_unsafe = [](const auto& results) {
                const auto found = std::find_if(results.begin(), results.end(),
                    [](const auto& result) { return !result.success; });
                return found == results.end() ? std::string {} : found->unsafe_reason;
            };
            std::cout << std::setprecision(12) << tolerance << ','
                      << polar_timing.second / epochs << ','
                      << cart_timing.second / epochs << ','
                      << successes(polar_results) << ',' << successes(cart_results) << ','
                      << '"' << first_unsafe(polar_results) << "\",\""
                      << first_unsafe(cart_results) << "\","
                      << '"' << values(polar_results) << "\",\""
                      << values(cart_results) << "\"\n";
        }
        return 0;
    }
    std::vector<Case> cases;
    std::vector<Profile> profiles;
    if (suite == "core") {
        cases = {
            {"smooth_planetary", "smooth", 1.1, 1.0e-3, {0.08, 0.03}, 1.0e-2},
            {"central_high_mag", "central", 1.1, 1.0e-3, {0.002, 0.0007}, 1.0e-3},
            {"tiny_source", "central", 1.1, 1.0e-3, {0.0015, -0.0004}, 3.0e-4},
            {"wide_binary", "wide", 2.0, 0.1, {0.10, 0.05}, 1.0e-2},
            {"equal_resonant", "resonant", 1.0, 1.0, {0.01, 0.02}, 2.0e-2},
        };
        profiles = {
            {"uniform", 0.0, 0.0},
            {"linear", 0.5, 0.0},
            {"square_root", 0.25, 0.35},
        };
    } else if (suite == "limb-sweep") {
        // A deliberately heterogeneous set.  Besides smooth sources this
        // includes regression geometries with high magnification, close and
        // wide caustics, tiny sources, and nearly tangent image components.
        cases = {
            {"smooth_planetary", "smooth", 1.1, 1.0e-3, {0.08, 0.03}, 1.0e-2},
            {"smooth_low_q", "smooth", 1.0, 0.1, {0.4, 0.3}, 2.0e-2},
            {"smooth_close", "smooth", 0.7, 0.3, {-0.6, 0.4}, 3.0e-2},
            {"smooth_wide_hard", "smooth", 1.4, 0.4, {0.2, -0.15}, 2.5e-2},
            {"far_planetary", "smooth", 1.0, 1.0e-3, {0.4, 0.3}, 1.0e-2},
            {"central_high_mag", "central", 1.1, 1.0e-3, {0.002, 0.0007}, 1.0e-3},
            {"tiny_source", "central", 1.1, 1.0e-3, {0.0015, -0.0004}, 3.0e-4},
            {"central_close", "central", 0.6, 0.5, {0.05, 0.02}, 1.0e-2},
            {"central_planetary", "central", 1.05, 1.0e-3, {0.05, 0.02}, 5.0e-3},
            {"central_equal", "central", 1.0, 1.0, {0.01, 0.02}, 2.0e-2},
            {"caustic_crossing", "crossing", 1.0, 0.2, {0.10, 0.05}, 1.0e-2},
            {"compact_close_caustic", "crossing", 0.3, 1.0e-3,
                {0.008845738870878478, 0.001678002323850983}, 3.0e-3},
            {"wide_binary", "wide", 2.0, 0.1, {0.10, 0.05}, 1.0e-2},
            {"wide_equal", "wide", 1.8, 1.0, {0.30, 0.10}, 2.0e-2},
            {"near_tangent_cap", "tangent", 1.2, 0.1, {0.653, 0.020}, 2.0e-2},
            {"tiny_q_tangent", "tangent", 1.3, 1.0e-4,
                {0.0014470002051158422, 0.0006812879524093549},
                0.0015515122255040925},
            {"order_sensitive_tangent", "tangent", 1.05, 1.0e-3,
                {0.010495776906606516, 0.004729643382853728},
                0.012087561335986492},
        };
        profiles = {{"linear", 0.5, 0.0}};
    } else {
        std::cerr << "unknown suite: " << suite << " (expected core or limb-sweep)\n";
        return 2;
    }
    struct Level { double tolerance; int polar_bins; };
    const std::vector<Level> levels {{1.0e-2, 24}, {1.0e-3, 100}, {1.0e-4, 320}};
    struct Aggregate {
        int total = 0;
        int polar_algebraic_successes = 0;
        int cartesian_algebraic_successes = 0;
        int cartesian_faster_than_polar_algebraic = 0;
        std::vector<double> polar_algebraic_speedups;
        std::vector<double> cartesian_algebraic_speedups;
        std::vector<double> cartesian_vs_polar_algebraic;
        std::vector<double> polar_algebraic_errors;
        std::vector<double> cartesian_algebraic_errors;
        std::vector<double> polar_seconds;
    };
    std::vector<Aggregate> aggregates(levels.size());

    std::cout << "case,family,profile,tolerance,polar_bins,reference,polar_algebraic,"
                 "cartesian_algebraic,polar_raster,polar_algebraic_error,"
                 "cartesian_algebraic_error,polar_raster_error,polar_algebraic_seconds,"
                 "cartesian_algebraic_seconds,polar_raster_seconds,polar_alg_speedup,"
                 "cart_alg_speedup,polar_alg_success,cart_alg_success,polar_alg_unsafe,"
                 "cart_alg_unsafe,polar_root_solves,polar_warm_sg,polar_cold_sg,"
                 "polar_cold_retries,polar_real_quartic,polar_real_fallbacks,"
                 "polar_tangency_solves,polar_tangency_fallbacks,"
                 "cart_root_solves,cart_warm_sg,cart_cold_sg,"
                 "cart_cold_retries\n";
    for (const auto& test : cases) {
        for (const auto& profile : profiles) {
            lcbinint::magnification::FiniteSourceSettings reference_settings;
            reference_settings.finite_mode = 2;
            reference_settings.source_bins = 640;
            reference_settings.polar_source_bins = 640;
            reference_settings.polar_grid_ratio = 4.0;
            reference_settings.automatic_source_bins = false;
            reference_settings.limb_darkening_c = profile.c;
            reference_settings.limb_darkening_d = profile.d;
            lcbinint::magnification::FiniteSourceMagnifier reference_engine(reference_settings);
            const double reference = reference_engine.experimental_raster_polar_binary_mag(
                test.separation, test.mass_ratio, test.source, test.rho);

            for (std::size_t level_index = 0; level_index < levels.size(); ++level_index) {
                const auto level = levels[level_index];
                lcbinint::magnification::FiniteSourceSettings polar_settings = reference_settings;
                polar_settings.source_bins = level.polar_bins;
                polar_settings.polar_source_bins = level.polar_bins;
                lcbinint::magnification::FiniteSourceMagnifier polar_engine(polar_settings);

                lcbinint::magnification::AlgebraicBoundarySettings algebraic_settings;
                algebraic_settings.absolute_tolerance = level.tolerance;
                algebraic_settings.relative_tolerance = level.tolerance;
                algebraic_settings.maximum_radial_depth = 26;
                algebraic_settings.unit_circle_tolerance = 2.0e-6;
                algebraic_settings.boundary_residual_tolerance = 2.0e-8;
                lcbinint::magnification::FiniteSourceMagnifier algebraic_engine(polar_settings);

                const auto polar_timing = timed([&] {
                    return polar_engine.experimental_raster_polar_binary_mag(
                        test.separation, test.mass_ratio, test.source, test.rho);
                }, repeats);
                lcbinint::magnification::AlgebraicBoundaryResult algebraic_result;
                const auto algebraic_timing = timed([&] {
                    algebraic_result = algebraic_engine.experimental_algebraic_boundary_binary_mag(
                        test.separation, test.mass_ratio, test.source, test.rho,
                        algebraic_settings);
                    return algebraic_result.magnification;
                }, repeats);
                lcbinint::magnification::AlgebraicBoundaryResult cartesian_result;
                const auto cartesian_timing = timed([&] {
                    cartesian_result =
                        algebraic_engine.experimental_algebraic_cartesian_binary_mag(
                            test.separation, test.mass_ratio, test.source, test.rho,
                            algebraic_settings);
                    return cartesian_result.magnification;
                }, repeats);
                const double speedup = algebraic_timing.second > 0.0
                    ? polar_timing.second / algebraic_timing.second : 0.0;
                const double cartesian_speedup = cartesian_timing.second > 0.0
                    ? polar_timing.second / cartesian_timing.second : 0.0;
                auto& aggregate = aggregates[level_index];
                ++aggregate.total;
                aggregate.polar_seconds.push_back(polar_timing.second);
                if (algebraic_result.success) {
                    ++aggregate.polar_algebraic_successes;
                    aggregate.polar_algebraic_speedups.push_back(speedup);
                    aggregate.polar_algebraic_errors.push_back(
                        relative_error(algebraic_timing.first, reference));
                }
                if (cartesian_result.success) {
                    ++aggregate.cartesian_algebraic_successes;
                    aggregate.cartesian_algebraic_speedups.push_back(cartesian_speedup);
                    aggregate.cartesian_algebraic_errors.push_back(
                        relative_error(cartesian_timing.first, reference));
                    if (algebraic_result.success && cartesian_timing.second > 0.0) {
                        const double relative_speed =
                            algebraic_timing.second / cartesian_timing.second;
                        aggregate.cartesian_vs_polar_algebraic.push_back(relative_speed);
                        if (relative_speed > 1.0) {
                            ++aggregate.cartesian_faster_than_polar_algebraic;
                        }
                    }
                }
                std::cout << std::setprecision(12)
                          << test.name << ',' << test.family << ',' << profile.name << ','
                          << level.tolerance << ','
                          << level.polar_bins << ',' << reference << ','
                          << algebraic_timing.first << ',' << cartesian_timing.first << ','
                          << polar_timing.first << ','
                          << (algebraic_result.success
                                  ? relative_error(algebraic_timing.first, reference) : -1.0) << ','
                          << (cartesian_result.success
                                  ? relative_error(cartesian_timing.first, reference) : -1.0) << ','
                          << relative_error(polar_timing.first, reference) << ','
                          << algebraic_timing.second << ',' << cartesian_timing.second << ','
                          << polar_timing.second << ',' << speedup << ',' << cartesian_speedup << ','
                          << (algebraic_result.success ? 1 : 0) << ','
                          << (cartesian_result.success ? 1 : 0) << ','
                          << '"' << algebraic_result.unsafe_reason << '"' << ','
                          << '"' << cartesian_result.unsafe_reason << '"' << ','
                          << algebraic_result.diagnostics.root_solve_count << ','
                          << algebraic_result.diagnostics.warm_root_solve_count << ','
                          << algebraic_result.diagnostics.cold_root_solve_count << ','
                          << algebraic_result.diagnostics.cold_retry_count << ','
                          << algebraic_result.diagnostics.real_quartic_solve_count << ','
                          << algebraic_result.diagnostics.real_quartic_fallback_count << ','
                          << algebraic_result.diagnostics.tangency_solve_count << ','
                          << algebraic_result.diagnostics.tangency_fallback_count << ','
                          << cartesian_result.diagnostics.root_solve_count << ','
                          << cartesian_result.diagnostics.warm_root_solve_count << ','
                          << cartesian_result.diagnostics.cold_root_solve_count << ','
                          << cartesian_result.diagnostics.cold_retry_count << '\n';
            }
        }
    }
    std::cout << "# summary,tolerance,total,polar_alg_success,cart_alg_success,"
                 "polar_alg_median_speedup,cart_alg_median_speedup,polar_alg_p90_error,"
                 "cart_alg_p90_error,cart_faster_count,cart_vs_polar_alg_median,"
                 "p90_polar_raster_seconds\n";
    for (std::size_t i = 0; i < levels.size(); ++i) {
        const auto& aggregate = aggregates[i];
        std::cout << "# summary," << levels[i].tolerance << ','
                  << aggregate.total << ',' << aggregate.polar_algebraic_successes << ','
                  << aggregate.cartesian_algebraic_successes << ','
                  << percentile(aggregate.polar_algebraic_speedups, 0.5) << ','
                  << percentile(aggregate.cartesian_algebraic_speedups, 0.5) << ','
                  << percentile(aggregate.polar_algebraic_errors, 0.9) << ','
                  << percentile(aggregate.cartesian_algebraic_errors, 0.9) << ','
                  << aggregate.cartesian_faster_than_polar_algebraic << ','
                  << percentile(aggregate.cartesian_vs_polar_algebraic, 0.5) << ','
                  << percentile(aggregate.polar_seconds, 0.9) << '\n';
    }
    return 0;
}
