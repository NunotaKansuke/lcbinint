#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/probe_diagnostics.hpp"
#include "lcbinint/magnification/adaptive_route_selector.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace lcbinint;
using namespace lcbinint::magnification;

namespace {

struct FluxScope {
    AdaptiveFluxRequest* previous = adaptive_flux_request();
    bool previous_lightweight = lightweight_flux_error_mode();
    ProbeCounters previous_counters = probe_counters();

    explicit FluxScope(AdaptiveFluxRequest& request)
    {
        adaptive_flux_request() = &request;
        lightweight_flux_error_mode() = true;
    }

    ~FluxScope()
    {
        adaptive_flux_request() = previous;
        lightweight_flux_error_mode() = previous_lightweight;
        probe_counters() = previous_counters;
    }
};

struct PolarScope {
    AdaptivePolarRequest* previous = adaptive_polar_request();

    explicit PolarScope(AdaptivePolarRequest& request)
    {
        adaptive_polar_request() = &request;
    }

    ~PolarScope()
    {
        adaptive_polar_request() = previous;
    }
};

struct Evaluation {
    double magnification = NAN;
    double normalized_indicator = NAN;
    bool met = false;
    std::size_t maps = 0;
    std::size_t subdivisions = 0;
    std::string stop_reason;
};

Evaluation evaluate_cartesian(
    const FiniteSourceMagnifier& magnifier,
    double separation,
    double native_mass_ratio,
    SourcePosition source,
    double rho,
    double point_magnification,
    const std::vector<SourcePosition>& center_seeds,
    const PointSourceMagnifier& point_magnifier,
    double target,
    double source_flux,
    int initial_bins)
{
    AdaptiveFluxRequest request;
    request.target = target * source_flux;
    request.relative_target = target;
    request.max_bins = 512;
    request.budget = 1000000;
    request.local_vertical_caps = true;
    request.defect_correction = true;
    FluxScope scope(request);
    const auto result = magnifier.binary_mag_preplanned(
        separation,
        native_mass_ratio,
        source,
        rho,
        point_magnification,
        FiniteSourceMethod::inverse_ray_cartesian,
        initial_bins,
        &center_seeds,
        &point_magnifier);
    if (!request.executed || !std::isfinite(result.magnification)) {
        throw std::runtime_error("adaptive Cartesian request did not return a finite value");
    }
    // The preplanned Cartesian wrapper's result flag describes its fixed-grid
    // plan; AdaptiveFluxRequest::met is the authority for this diagnostic run.
    return {
        result.magnification,
        request.acceptance_indicator / source_flux,
        request.met,
        request.new_maps,
        request.refinements,
        request.stop_reason,
    };
}

Evaluation evaluate_polar(
    const FiniteSourceMagnifier& magnifier,
    double separation,
    double native_mass_ratio,
    SourcePosition source,
    double rho,
    double point_magnification,
    const std::vector<SourcePosition>& center_seeds,
    const PointSourceMagnifier& point_magnifier,
    double target,
    int initial_bins)
{
    AdaptivePolarRequest request;
    request.relative_tolerance = target;
    request.absolute_tolerance = target;
    request.initial_angular_ratio = 256.0;
    request.budget = 1000000;
    PolarScope scope(request);
    const auto result = magnifier.binary_mag_preplanned(
        separation,
        native_mass_ratio,
        source,
        rho,
        point_magnification,
        FiniteSourceMethod::inverse_ray_polar,
        initial_bins,
        &center_seeds,
        &point_magnifier);
    if (!request.executed || !std::isfinite(result.magnification)) {
        throw std::runtime_error("adaptive Polar request did not return a finite value");
    }
    if (result.converged != request.met) {
        throw std::runtime_error("Polar result disagrees with request convergence state");
    }
    return {
        result.magnification,
        // Polar already reports absolute magnification error. Cartesian's
        // admission indicator above is an unnormalized image flux instead.
        request.error_indicator,
        request.met,
        request.maps,
        request.angular_refinements + request.radial_refinements,
        request.stop_reason,
    };
}

double median3(std::array<double, 3> values)
{
    std::sort(values.begin(), values.end());
    return values[1];
}

bool same_evaluation(const Evaluation& a, const Evaluation& b)
{
    return a.magnification == b.magnification &&
        a.normalized_indicator == b.normalized_indicator &&
        a.met == b.met && a.maps == b.maps &&
        a.subdivisions == b.subdivisions && a.stop_reason == b.stop_reason;
}

void write_failure(std::ofstream& output, int id, const char* method,
                   double target, const std::string& message)
{
    for (int repeat = 0; repeat < 3; ++repeat) {
        output << id << '\t' << method << '\t' << target << '\t' << repeat
               << "\tnan\tnan\tnan\t0\t0\tnan\tnan\tnan\tnan\tinf\tinf"
               << "\t0\texception\t0\t0\t0\t0\t0"
               << '\t' << message << '\n';
    }
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 5) {
        std::cerr << "usage: seed_inclusive_cart_polar_runner INPUT.tsv OUTPUT.tsv"
                     " [CARTESIAN_INITIAL_BINS POLAR_INITIAL_BINS]\n";
        return 2;
    }
    const int cartesian_bins = argc == 5 ? std::atoi(argv[3]) : 16;
    const int polar_bins = argc == 5 ? std::atoi(argv[4]) : 2;
    if ((cartesian_bins != 8 && cartesian_bins != 16) ||
        (polar_bins != 2 && polar_bins != 4)) {
        std::cerr << "diagnostic grid comparison requires Cartesian8/16 and Polar2/4\n";
        return 2;
    }

    for (const char* key : {
             "LCBININT_DIAGNOSTIC_DIRECTIONAL_ADAPTIVE",
             "LCBININT_DIAGNOSTIC_LOCAL_ROW_INTERPOLATION",
             "LCBININT_DIAGNOSTIC_LOCAL_BOUNDARY_STRIPS",
             "LCBININT_DIAGNOSTIC_LOCAL_VERTICAL_CAPS",
             "LCBININT_DIAGNOSTIC_DEFECT_CORRECTION"}) {
        setenv(key, "1", 1);
    }
    unsetenv("LCBININT_REPRESENTATIVE_PROBES");

    std::ifstream input(argv[1]);
    std::ofstream output(argv[2]);
    if (!input || !output) {
        std::cerr << "could not open input or output\n";
        return 2;
    }

    output << "sample_id\tmethod\ttarget\trepeat\tcenter_root_us\tcaustic_build_ms"
              "\tseed_only_us\tseed_count\tsupport_proven\tcold_ms\twarm_ms"
              "\tcold_magnification\twarm_magnification\tcold_indicator\twarm_indicator"
              "\tcold_met\twarm_met\tcold_maps\twarm_maps\tcold_subdivisions"
              "\twarm_subdivisions\tcold_warm_identical\tstop_reason\n"
           << std::setprecision(17);

    std::string line;
    while (std::getline(input, line)) {
        int sample_id = -1;
        double separation = 0.0, q = 0.0, rho = 0.0, x = 0.0, y = 0.0, limb = 0.0;
        std::istringstream fields(line);
        if (!(fields >> sample_id >> separation >> q >> rho >> x >> y >> limb)) {
            continue;  // Header or comment row.
        }

        try {
            const SourcePosition source{x, y};
            PointSourceMagnifier point_magnifier;
            std::vector<SourcePosition> center_seeds;
            center_seeds.reserve(5);
            double point_magnification = 0.0;
            const auto root_start = std::chrono::steady_clock::now();
            const auto center_images = point_magnifier.binary_images(separation, 1.0 / q, source);
            for (const auto& image : center_images) {
                point_magnification += 1.0 / std::abs(image.jacobian_determinant);
                center_seeds.push_back(image.position);
            }
            const double center_root_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - root_start).count();

            FiniteSourceSettings settings;
            settings.source_bins = cartesian_bins;
            settings.polar_source_bins = polar_bins;
            settings.max_source_bins = cartesian_bins;
            settings.caustic_bins = 1400;
            settings.grid_ratio = 4.0;
            settings.polar_grid_ratio = 4.0;
            settings.finite_mode = 1;
            settings.kinji_threshold = 0.0;
            settings.hex_threshold = 0.0;
            settings.adaptive_hex_threshold = 0.0;
            settings.limb_darkening_c = limb;
            settings.limb_darkening_d = 0.0;
            settings.automatic_source_bins = false;
            FiniteSourceMagnifier magnifier(settings);

            const auto caustic_start = std::chrono::steady_clock::now();
            (void)magnifier.binary_caustic_branches(separation, 1.0 / q);
            const double caustic_build_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - caustic_start).count();
            magnifier.ensure_limb_darkening_table();

            const double source_flux =
                std::acos(-1.0) * rho * rho * (1.0 - limb / 3.0);
            if (!(std::isfinite(source_flux) && source_flux > 0.0)) {
                throw std::runtime_error("invalid source flux normalization");
            }

            std::array<double, 3> seed_times{};
            std::size_t seed_count = 0;
            bool support_proven = false;
            // Prime the indexed path and allocator once, then make every
            // measured provider call a genuine source-seed-cache miss.
            (void)magnifier.cached_binary_image_seeds(
                point_magnifier, separation, 1.0 / q, source, rho,
                std::numeric_limits<double>::infinity(), &center_seeds, &support_proven);
            for (double& elapsed : seed_times) {
                magnifier.clear_binary_image_seed_cache_for_diagnostics();
                const auto start = std::chrono::steady_clock::now();
                bool proven = false;
                const auto seeds = magnifier.cached_binary_image_seeds(
                    point_magnifier, separation, 1.0 / q, source, rho,
                    std::numeric_limits<double>::infinity(), &center_seeds, &proven);
                elapsed = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count();
                seed_count = seeds.size();
                support_proven = proven;
            }

            for (const auto& method : std::array<FiniteSourceMethod, 1>{
                     FiniteSourceMethod::inverse_ray_cartesian}) {
                const char* name = method == FiniteSourceMethod::inverse_ray_cartesian
                    ? "automatic" : "automatic";
                for (double target : {1.0e-3, 1.0e-4}) {
                    try {
                    auto evaluate = [&](const FiniteSourceMagnifier& current) {
                        const auto decision = adaptive_detail::choose_adaptive_route(separation, 1.0/q, rho, limb, target, center_images);
                        return !decision.polar
                            ? evaluate_cartesian(current, separation, 1.0 / q, source, rho,
                                point_magnification, center_seeds, point_magnifier,
                                target, source_flux, cartesian_bins)
                            : evaluate_polar(current, separation, 1.0 / q, source, rho,
                                point_magnification, center_seeds, point_magnifier,
                                target, polar_bins);
                    };

                    // Warm the target path and allocator before timings. The
                    // first call also builds the seed set; later calls hit it.
                    for (int warmup = 0; warmup < 2; ++warmup) {
                        (void)evaluate(magnifier);
                    }
                    magnifier.clear_binary_image_seed_cache_for_diagnostics();

                    std::array<double, 3> cold_times{}, warm_times{};
                    Evaluation cold_result, warm_result;
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        const auto timed_call = [&](Evaluation& result) {
                            const auto start = std::chrono::steady_clock::now();
                            result = evaluate(magnifier);
                            return std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start).count();
                        };
                        if (repeat % 2 == 0) {
                            magnifier.clear_binary_image_seed_cache_for_diagnostics();
                            cold_times[repeat] = timed_call(cold_result);
                            warm_times[repeat] = timed_call(warm_result);
                        } else {
                            magnifier.clear_binary_image_seed_cache_for_diagnostics();
                            bool proven = false;
                            (void)magnifier.cached_binary_image_seeds(
                                point_magnifier, separation, 1.0 / q, source, rho,
                                std::numeric_limits<double>::infinity(), &center_seeds, &proven);
                            warm_times[repeat] = timed_call(warm_result);
                            magnifier.clear_binary_image_seed_cache_for_diagnostics();
                            cold_times[repeat] = timed_call(cold_result);
                        }
                    }

                    const bool identical = same_evaluation(cold_result, warm_result);
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        output << sample_id << '\t' << name << '\t' << target << '\t' << repeat
                               << '\t' << center_root_us << '\t' << caustic_build_ms
                               << '\t' << median3(seed_times) << '\t' << seed_count
                               << '\t' << (support_proven ? 1 : 0)
                               << '\t' << cold_times[repeat] << '\t' << warm_times[repeat]
                               << '\t' << cold_result.magnification << '\t'
                               << warm_result.magnification << '\t'
                               << cold_result.normalized_indicator << '\t'
                               << warm_result.normalized_indicator << '\t'
                               << (cold_result.met ? 1 : 0) << '\t'
                               << (warm_result.met ? 1 : 0) << '\t'
                               << cold_result.maps << '\t' << warm_result.maps << '\t'
                               << cold_result.subdivisions << '\t'
                               << warm_result.subdivisions << '\t' << (identical ? 1 : 0)
                               << '\t' << cold_result.stop_reason << '\n';
                    }
                    output.flush();
                    } catch (const std::exception& error) {
                        std::cerr << sample_id << ' ' << name << ' ' << target
                                  << ' ' << error.what() << '\n';
                        write_failure(output, sample_id, name, target, "exception");
                        output.flush();
                    }
                }
            }
        } catch (const std::exception& error) {
            std::cerr << sample_id << ' ' << error.what() << '\n';
            for (const char* method : {"cartesian", "polar"})
                for (double target : {1.0e-3, 1.0e-4})
                    write_failure(output, sample_id, method, target, "exception");
            output.flush();
        }
    }
    return 0;
}
