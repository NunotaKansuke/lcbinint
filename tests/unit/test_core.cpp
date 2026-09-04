#include "lcbinint/lcbinint.h"
#include "lcbinint/magnification/cartesian_run_fill.hpp"
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/probe_diagnostics.hpp"
#include "lcbinint/math/polynomial_roots.hpp"
#include "lcbinint/model/triple_lens_geometry.hpp"
#include "lcbinint/model/orbital_motion.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

// The C ABI starts with parallax_mode.  Keep Python-only backend selection
// out of this public structure so existing C callers retain their layout.
static_assert(offsetof(lcbi_options, parallax_mode) == 0);

bool close_to_zero(lcbinint::Complex value)
{
    return std::abs(value) < 1e-9;
}

bool all_roots_satisfy(
    const std::vector<lcbinint::Complex>& coefficients,
    const std::vector<lcbinint::Complex>& roots)
{
    for (const auto& root : roots) {
        if (!close_to_zero(lcbinint::math::PolynomialRootSolver::evaluate(coefficients, root))) {
            return false;
        }
    }
    return true;
}

struct TripleReferenceCase {
    lcbinint::SourcePosition source;
    double separation = 0.0;
    double mass_ratio = 0.0;
    double secondary_mass_ratio = 0.0;
    double secondary_separation = 0.0;
    double secondary_angle = 0.0;
    double reference_magnification = 0.0;
    double relative_tolerance = 0.0;
};

bool close_relative(double actual, double expected, double tolerance)
{
    return std::abs(actual - expected) <= tolerance * std::abs(expected);
}

using LatticeCell = std::pair<std::int64_t, std::int64_t>;
using LatticeCells = std::set<LatticeCell>;
using lcbinint::magnification::detail::CartesianBoundaryContribution;
using lcbinint::magnification::detail::CartesianLatticeSeed;
using lcbinint::magnification::detail::CartesianRunFillLimits;
using lcbinint::magnification::detail::fill_cartesian_runs;

struct MaskCellState {
    bool inside = false;
};

LatticeCells mask_cells(const std::vector<std::string>& rows)
{
    LatticeCells cells;
    for (std::size_t y = 0; y < rows.size(); ++y) {
        for (std::size_t x = 0; x < rows[y].size(); ++x) {
            if (rows[y][x] == '#') {
                cells.emplace(
                    static_cast<std::int64_t>(x),
                    static_cast<std::int64_t>(y));
            }
        }
    }
    return cells;
}

bool run_mask_case(
    const std::vector<std::string>& rows,
    std::vector<CartesianLatticeSeed> seeds,
    std::size_t minimum_multi_run_rows = 0)
{
    const LatticeCells expected = mask_cells(rows);
    LatticeCells visited;
    const auto result = fill_cartesian_runs<MaskCellState>(
        std::move(seeds),
        [&](std::int64_t ix, std::int64_t iy) {
            return MaskCellState {expected.count({ix, iy}) != 0};
        },
        [](const MaskCellState& state) { return state.inside; },
        [&](std::int64_t ix, std::int64_t iy, const MaskCellState&) {
            visited.emplace(ix, iy);
            return 1.0;
        },
        [](const auto&, const auto&, const auto&, const auto&, const auto&) {
            return CartesianBoundaryContribution {};
        },
        CartesianRunFillLimits {100000, 10000});
    if (!result.ok() || visited != expected) {
        return false;
    }
    if (result.counters.rows_with_multiple_runs <
        static_cast<std::int64_t>(minimum_multi_run_rows)) {
        return false;
    }
    long double area = 0.0L;
    for (std::size_t index = 0;
         index < result.component_roots.size(); ++index) {
        if (result.component_roots[index] == index) {
            area += result.component_areas[index];
        }
    }
    return area == static_cast<long double>(expected.size());
}

bool cartesian_run_topology_tests()
{
    const std::vector<std::string> banana {
        "..#####..",
        ".##...##.",
        "##.....##",
        "##.....##",
    };
    if (!run_mask_case(banana, {{4, 0}}, 2)) {
        return false;
    }

    const std::vector<std::string> pinched {
        ".#####.",
        ".##.##.",
        ".#####.",
    };
    if (!run_mask_case(pinched, {{3, 0}}, 1)) {
        return false;
    }

    const std::vector<std::string> horseshoe {
        "#######",
        "##...##",
        "##...##",
        "##.....",
    };
    if (!run_mask_case(horseshoe, {{0, 0}}, 2)) {
        return false;
    }

    const std::vector<std::string> diagonal {
        "#....",
        ".#...",
        "..#..",
        "...#.",
        "....#",
    };
    if (!run_mask_case(diagonal, {{0, 0}})) {
        return false;
    }

    const std::vector<std::string> disconnected {
        "##....##",
        "##....##",
    };
    if (!run_mask_case(disconnected, {{0, 0}, {6, 0}})) {
        return false;
    }

    const std::vector<CartesianLatticeSeed> duplicate_permuted {
        {7, 1}, {0, 1}, {7, 1}, {1, 0}, {6, 0}, {0, 1},
    };
    if (!run_mask_case(disconnected, duplicate_permuted)) {
        return false;
    }
    std::vector<CartesianLatticeSeed> reverse = duplicate_permuted;
    std::reverse(reverse.begin(), reverse.end());
    return run_mask_case(disconnected, std::move(reverse));
}

struct DiskCellState {
    double radius2 = 0.0;
    bool inside = false;
};

double identity_disk_area(int bins, std::vector<CartesianLatticeSeed> seeds)
{
    const double spacing = 1.0 / static_cast<double>(bins);
    const auto result = fill_cartesian_runs<DiskCellState>(
        std::move(seeds),
        [spacing](std::int64_t ix, std::int64_t iy) {
            const double x = static_cast<double>(ix) * spacing;
            const double y = static_cast<double>(iy) * spacing;
            const double radius2 = x * x + y * y;
            return DiskCellState {radius2, radius2 <= 1.0};
        },
        [](const DiskCellState& state) { return state.inside; },
        [](std::int64_t, std::int64_t, const DiskCellState&) { return 1.0; },
        [](const auto&,
           const DiskCellState& left_inside,
           const DiskCellState& left_outside,
           const DiskCellState& right_inside,
           const DiskCellState& right_outside) {
            const auto correction = [](double inside2, double outside2) {
                const double inside = std::sqrt(inside2);
                const double outside = std::sqrt(outside2);
                return (1.0 - inside) / (outside - inside) - 0.5;
            };
            return CartesianBoundaryContribution {
                correction(left_inside.radius2, left_outside.radius2) +
                    correction(right_inside.radius2, right_outside.radius2),
                2,
                true,
            };
        },
        CartesianRunFillLimits {1000000, 100000});
    if (!result.ok()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    long double cells = 0.0L;
    for (std::size_t index = 0;
         index < result.component_roots.size(); ++index) {
        if (result.component_roots[index] == index) {
            cells += result.component_areas[index];
        }
    }
    return static_cast<double>(cells) * spacing * spacing;
}

bool cartesian_run_boundary_tests()
{
    const double area16 = identity_disk_area(16, {{0, 0}});
    const double area32 = identity_disk_area(32, {{0, 0}});
    const double area64 = identity_disk_area(64, {{0, 0}});
    const double pi = std::acos(-1.0);
    if (!std::isfinite(area16) || !std::isfinite(area32) ||
        !std::isfinite(area64) ||
        !(std::abs(area64 - pi) < std::abs(area32 - pi) &&
          std::abs(area32 - pi) < std::abs(area16 - pi))) {
        return false;
    }
    const double duplicate = identity_disk_area(
        64, {{0, 0}, {20, 0}, {-20, 0}, {0, 20}, {0, 0}});
    return std::abs(area64 - duplicate) <= 1.0e-14 &&
        std::abs(area64 - pi) / pi < 1.0e-3;
}

} // namespace

int main()
{
    if (!cartesian_run_topology_tests()) {
        return 62;
    }
    if (!cartesian_run_boundary_tests()) {
        return 63;
    }

    const lcbinint::magnification::ProbePolicy default_probe_policy;
    if (!default_probe_policy.normals ||
        !default_probe_policy.tangents ||
        default_probe_policy.offsets != 8) {
        return 61;
    }

    lcbi_params params = lcbi_default_params();
    lcbi_options options = lcbi_default_options();
    lcbi_result result = {};

    if (params.tE != 1.0) {
        return 1;
    }
    if (params.orbital_motion_mode != LCBI_ORBIT_STATIC || std::abs(params.lom_ar - 1.0) > 1e-12) {
        return 37;
    }
    if (std::isfinite(params.obs_lat) || std::isfinite(params.obs_lon)) {
        return 47;
    }
    if (options.caustic_bins != 1400 || options.mode != 4 ||
        std::abs(options.point_source_threshold - 20.0) > 1e-12 ||
        std::abs(options.hexadecapole_threshold - 3.0) > 1e-12 ||
        options.source_bins != 50 ||
        options.automatic_source_bins != 1 || options.max_source_bins != 400 ||
        std::abs(options.finite_source_tol) > 1e-12 ||
        std::abs(options.finite_source_reltol) > 1e-12) {
        return 2;
    }
    const auto high_resolution =
        lcbinint::magnification::calibrated_binary_resolution(
            1.0e-3, 1.0e-3, 2.0e-4, 1000.0, 0.0, 4, 0.0, 1.0e-3, 400);
    if (!high_resolution.prefer_polar || high_resolution.source_bins != 106) {
        return 44;
    }
    const auto loose_resolution =
        lcbinint::magnification::calibrated_binary_resolution(
            1.0e-3, 1.0e-3, 2.0e-4, 10.0, 0.0, 4, 0.0, 1.0e-2, 400);
    if (loose_resolution.prefer_polar || loose_resolution.source_bins != 17) {
        return 52;
    }
    const auto default_resolution =
        lcbinint::magnification::calibrated_binary_resolution(
            1.0e-3, 1.0e-3, 2.0e-4, 10.0, 0.0, 4, 0.0, 0.0, 400);
    if (default_resolution.prefer_polar || default_resolution.source_bins != 50) {
        return 53;
    }
    const auto tight_high_resolution =
        lcbinint::magnification::calibrated_binary_resolution(
            1.0e-3, 1.0e-3, 2.0e-4, 1000.0, 0.0, 4, 0.0, 1.0e-5, 400);
    if (!tight_high_resolution.prefer_polar || tight_high_resolution.source_bins != 400) {
        return 50;
    }
    const auto tangent_resolution =
        lcbinint::magnification::calibrated_binary_resolution(
            0.1, 1.0e-3, 1.0e-3, 10.0, 0.0, 4, 0.0, 1.0e-3, 400);
    if (tangent_resolution.prefer_polar || tangent_resolution.source_bins != 50) {
        return 45;
    }
    const auto tight_tangent_resolution =
        lcbinint::magnification::calibrated_binary_resolution(
            0.1, 1.0e-3, 1.0e-3, 10.0, 0.0, 4, 0.0, 1.0e-5, 400);
    if (tight_tangent_resolution.prefer_polar ||
        tight_tangent_resolution.source_bins != 400) {
        return 51;
    }
    const auto absolute_resolution =
        lcbinint::magnification::calibrated_binary_resolution(
            0.1, 1.0e-3, 1.0e-3, 10.0, 0.0, 4, 1.0e-3, 0.0, 400);
    if (absolute_resolution.prefer_polar || absolute_resolution.source_bins != 303) {
        return 54;
    }
    const auto mixed_resolution =
        lcbinint::magnification::calibrated_binary_resolution(
            0.1, 1.0e-3, 1.0e-3, 10.0, 0.0, 4, 1.0e-2, 1.0e-4, 400);
    if (mixed_resolution.prefer_polar || mixed_resolution.source_bins != 114) {
        return 55;
    }
    if (!lcbinint::magnification::binary_auto_tolerance_supported(0.0, 0.0) ||
        !lcbinint::magnification::binary_auto_tolerance_supported(2.0e-4, 0.0) ||
        !lcbinint::magnification::binary_auto_tolerance_supported(0.0, 1.0e-4) ||
        !lcbinint::magnification::binary_auto_tolerance_supported(1.0e-5, 1.0e-3) ||
        lcbinint::magnification::binary_auto_tolerance_supported(1.0e-4, 0.0) ||
        lcbinint::magnification::binary_auto_tolerance_supported(1.0e-5, 0.0) ||
        lcbinint::magnification::binary_auto_tolerance_supported(0.0, 1.0e-5)) {
        return 59;
    }
    const auto forced_polar_resolution =
        lcbinint::magnification::calibrated_binary_resolution(
            0.1, 1.0e-3, 1.0e-3, 10.0, 0.0, 2, 0.0, 1.0e-3, 400);
    if (forced_polar_resolution.prefer_polar ||
        forced_polar_resolution.source_bins != 106) {
        return 56;
    }
    const auto forced_cartesian_resolution =
        lcbinint::magnification::calibrated_binary_resolution(
            0.1, 1.0e-3, 1.0e-3, 1000.0, 0.0, 1, 0.0, 1.0e-3, 400);
    if (!forced_cartesian_resolution.prefer_polar ||
        forced_cartesian_resolution.source_bins != 50) {
        return 57;
    }
    const auto triple_calibration_geometry =
        lcbinint::model::make_triple_lens_geometry(1.0, 1.0e-3, 1.0e-4, 0.5, 1.2);
    const auto triple_near_resolution =
        lcbinint::magnification::calibrated_triple_resolution(
            triple_calibration_geometry, 1.0e-3, 2.0e-4, 10.0, 0.0, 1.0e-3, 400);
    if (triple_near_resolution.prefer_polar ||
        triple_near_resolution.source_bins <= 0 || triple_near_resolution.source_bins > 400) {
        return 48;
    }
    const auto triple_high_resolution =
        lcbinint::magnification::calibrated_triple_resolution(
            triple_calibration_geometry, 1.0e-3, 2.0e-4, 1000.0, 0.0, 1.0e-3, 80);
    if (triple_high_resolution.prefer_polar ||
        triple_high_resolution.source_bins <= 0 || triple_high_resolution.source_bins > 80) {
        return 49;
    }
    const auto triple_tight_resolution =
        lcbinint::magnification::calibrated_triple_resolution(
            triple_calibration_geometry, 1.0e-3, 2.0e-4, 10.0, 0.0, 1.0e-5, 400);
    if (triple_tight_resolution.prefer_polar ||
        triple_tight_resolution.source_bins != 400) {
        return 52;
    }
    if (lcbi_magnification(0.0, &params, &options, &result) != LCBI_OK) {
        return 3;
    }
    if (std::abs(result.source_x) > 1e-12 || std::abs(result.source_y) > 1e-12) {
        return 4;
    }
    params.umin = 0.1;
    params.theta = 0.0;
    params.q = 0.1;
    params.sep = 1.0;
    if (lcbi_magnification(0.2, &params, &options, &result) != LCBI_OK) {
        return 5;
    }
    if (std::abs(result.source_x - 0.2) > 1e-12 || std::abs(result.source_y - 0.1) > 1e-12) {
        return 6;
    }
    if (std::abs(result.magnification - 5.871444912771214) > 1e-10) {
        return 7;
    }
    params.sep = 1.5;
    params.q = 1.0;
    if (lcbi_magnification(0.2, &params, &options, &result) != LCBI_OK) {
        return 17;
    }
    if (std::abs(result.magnification - 3.5659775904852786) > 1e-10) {
        return 18;
    }
    params.sep = 1.0;
    params.q = 0.1;
    params.umin = 1.1;
    params.rho = 0.001;
    options.center_of_mass = 1;
    if (lcbi_magnification(1.2, &params, &options, &result) != LCBI_OK) {
        return 19;
    }
    if (!std::isfinite(result.finite_source_magnification)) {
        return 20;
    }
    if (std::strcmp(lcbi_status_string(LCBI_UNSUPPORTED), "unsupported") != 0) {
        return 23;
    }
    if (std::strcmp(
            lcbi_status_string(LCBI_UNSUPPORTED_TOLERANCE),
            "unsupported_tolerance") != 0) {
        return 58;
    }
    lcbi_params unsupported_tolerance_params = lcbi_default_params();
    unsupported_tolerance_params.q = 0.1;
    unsupported_tolerance_params.sep = 1.0;
    unsupported_tolerance_params.rho = 1.0e-2;
    lcbi_options unsupported_tolerance_options = lcbi_default_options();
    unsupported_tolerance_options.finite_source_tol = 1.0e-4;
    unsupported_tolerance_options.finite_source_reltol = 0.0;
    lcbi_result unsupported_tolerance_result = {};
    if (lcbi_magnification(
            0.0,
            &unsupported_tolerance_params,
            &unsupported_tolerance_options,
            &unsupported_tolerance_result) != LCBI_UNSUPPORTED_TOLERANCE) {
        return 60;
    }
    if (!lcbinint::model::kepler_orbit_is_valid(
            0.004, 0.011, 0.006, 0.2, 1.4)) {
        return 38;
    }
    if (lcbinint::model::kepler_orbit_is_valid(
            0.004, 0.011, 0.006, 0.2, 0.5)) {
        return 39;
    }
    if (lcbinint::model::kepler_orbit_is_valid(0.0, 0.0, 0.0, 0.2, 1.4)) {
        return 40;
    }

    params.orbital_motion_mode = LCBI_ORBIT_KEPLER;
    params.g1 = 0.004;
    params.g2 = 0.011;
    params.g3 = 0.006;
    params.lom_szs = 0.2;
    params.lom_ar = 0.5;
    if (lcbi_magnification(0.2, &params, &options, &result) != LCBI_INVALID_ARGUMENT) {
        return 41;
    }
    params.orbital_motion_mode = LCBI_ORBIT_STATIC;
    params.lom_ar = 1.0;

    lcbinint::math::PolynomialRootSolver solver;
    auto linear = solver.solve({-2.0, 1.0});
    if (linear.status != lcbinint::math::RootSolverStatus::ok || linear.roots.size() != 1) {
        return 8;
    }
    if (std::abs(linear.roots[0] - lcbinint::Complex(2.0, 0.0)) > 1e-12) {
        return 9;
    }

    auto quadratic = solver.solve({-1.0, 0.0, 1.0});
    if (quadratic.status != lcbinint::math::RootSolverStatus::ok || quadratic.roots.size() != 2) {
        return 10;
    }
    if (!close_to_zero(lcbinint::math::PolynomialRootSolver::evaluate({-1.0, 0.0, 1.0}, quadratic.roots[0]))) {
        return 11;
    }
    if (!close_to_zero(lcbinint::math::PolynomialRootSolver::evaluate({-1.0, 0.0, 1.0}, quadratic.roots[1]))) {
        return 12;
    }

    const std::vector<lcbinint::Complex> cubic_coefficients = {-1.0, 0.0, 0.0, 1.0};
    auto cubic = solver.solve(cubic_coefficients);
    if (cubic.status != lcbinint::math::RootSolverStatus::ok || cubic.roots.size() != 3) {
        return 13;
    }
    if (!all_roots_satisfy(cubic_coefficients, cubic.roots)) {
        return 14;
    }

    const std::vector<lcbinint::Complex> fifth_coefficients = {
        lcbinint::Complex(-1.0, 0.25),
        lcbinint::Complex(0.5, -0.75),
        lcbinint::Complex(-1.0, 0.0),
        lcbinint::Complex(0.25, 0.5),
        lcbinint::Complex(-0.25, 0.0),
        lcbinint::Complex(1.0, 0.0),
    };
    auto fifth = solver.solve(fifth_coefficients);
    if (fifth.status != lcbinint::math::RootSolverStatus::ok || fifth.roots.size() != 5) {
        return 15;
    }
    if (!all_roots_satisfy(fifth_coefficients, fifth.roots)) {
        return 16;
    }

    lcbinint::magnification::FiniteSourceSettings finite_settings;
    finite_settings.source_bins = 20;
    finite_settings.caustic_bins = 128;
    lcbinint::magnification::FiniteSourceMagnifier finite_magnifier(finite_settings);
    auto finite_result = finite_magnifier.binary_mag(1.0, 0.1, {1.2, 1.1}, 0.001, 1.0);
    if (finite_result.decision.method != lcbinint::magnification::FiniteSourceMethod::point_source) {
        return 21;
    }
    if (!finite_result.converged) {
        return 22;
    }
    finite_settings.adaptive_hex_threshold = 1.0;
    finite_settings.hex_threshold = 0.0;
    lcbinint::magnification::FiniteSourceMagnifier hex_finite_magnifier(finite_settings);
    auto uniform_hex_result = hex_finite_magnifier.binary_mag(1.0, 0.1, {0.2, 0.2}, 0.02, 1.0);
    if (uniform_hex_result.decision.method != lcbinint::magnification::FiniteSourceMethod::hexadecapole) {
        return 32;
    }
    if (!std::isfinite(uniform_hex_result.magnification) || !uniform_hex_result.converged) {
        return 33;
    }
    auto limb_darkened_settings = finite_settings;
    limb_darkened_settings.limb_darkening_c = 0.5;
    limb_darkened_settings.limb_darkening_d = 0.2;
    lcbinint::magnification::FiniteSourceMagnifier limb_darkened_finite_magnifier(limb_darkened_settings);
    auto limb_darkened_hex_result =
        limb_darkened_finite_magnifier.binary_mag(1.0, 0.1, {0.2, 0.2}, 0.02, 1.0);
    if (limb_darkened_hex_result.decision.method != lcbinint::magnification::FiniteSourceMethod::hexadecapole) {
        return 34;
    }
    if (!std::isfinite(limb_darkened_hex_result.magnification)) {
        return 35;
    }
    if (std::abs(limb_darkened_hex_result.magnification - uniform_hex_result.magnification) < 1.0e-12) {
        return 36;
    }
    const auto triple_geometry =
        lcbinint::model::make_triple_lens_geometry(1.0, 0.001, 0.0001, 0.5, 1.2);
    if (std::abs(triple_geometry.masses[0] + triple_geometry.masses[1] +
                 triple_geometry.masses[2] - 1.0) > 1.0e-12) {
        return 38;
    }
    const auto triple_mapped =
        lcbinint::model::triple_lens_equation(triple_geometry, {0.8, 0.3});
    lcbinint::magnification::PointSourceMagnifier point_magnifier;
    const auto triple_images = point_magnifier.triple_images(triple_geometry, triple_mapped);
    if (triple_images.empty() || triple_images.size() > 10) {
        return 39;
    }
    const auto triple_point = point_magnifier.triple_mag0(triple_geometry, triple_mapped);
    if (!std::isfinite(triple_point.magnification) || triple_point.image_count <= 0) {
        return 40;
    }
    params = lcbi_default_params();
    options = lcbi_default_options();
    options.vbm_compatible = 1;
    params.umin = 0.01;
    params.theta = 0.5;
    params.sep = 1.0;
    params.q = 1.0e-3;
    params.q2 = 1.0e-4;
    params.sep2 = 0.5;
    params.ang = 1.2;
    params.rho = 0.0;
    if (lcbi_magnification(0.0, &params, &options, &result) != LCBI_OK ||
        !std::isfinite(result.magnification) || result.image_count <= 0) {
        return 41;
    }
    params.rho = 1.0e-3;
    options.source_bins = 8;
    if (lcbi_magnification(0.0, &params, &options, &result) != LCBI_OK ||
        !std::isfinite(result.finite_source_magnification)) {
        return 42;
    }
    const TripleReferenceCase triple_reference_cases[] = {
        // Generated from /moao38_7/nunota/binfit/integral/lcbinint.c amp_point3.
        {{-0.09263782795758546, -0.03908195790173323},
            1.0, 1.0e-3, 1.0e-4, 0.5, 1.2, 10.529790084883288, 5.0e-4},
        {{-0.00479425538604203, 0.008775825618903728},
            1.0, 1.0e-3, 1.0e-4, 0.5, 1.2, 118.58394756835955, 5.0e-4},
        {{0.17067435180044185, 0.10449139266017765},
            1.0, 1.0e-3, 1.0e-4, 0.5, 1.2, 5.0081788428186362, 5.0e-4},
        {{0.35, -0.22},
            0.8, 0.03, 0.02, 0.35, -0.7, 2.3663298774361103, 1.0e-9},
        {{-0.45, 0.18},
            1.4, 0.2, 0.05, 0.7, 2.1, 2.5951753373288202, 1.0e-9},
    };
    for (const auto& reference_case : triple_reference_cases) {
        const auto geometry = lcbinint::model::make_triple_lens_geometry(
            reference_case.separation,
            reference_case.mass_ratio,
            reference_case.secondary_mass_ratio,
            reference_case.secondary_separation,
            reference_case.secondary_angle);
        const auto point = point_magnifier.triple_mag0(geometry, reference_case.source);
        if (!close_relative(
                point.magnification,
                reference_case.reference_magnification,
                reference_case.relative_tolerance)) {
            return 43;
        }
    }

    // lcbi_finite_source_geometry[_array]: a cheap, root-solve-free primitive
    // that must agree with the separation/mass_ratio/caustic_distance fields
    // lcbi_magnification[_array] already populates.
    lcbi_params geometry_params = lcbi_default_params();
    geometry_params.sep = 1.25;
    geometry_params.q = 2.0e-3;
    geometry_params.rho = 1.0e-3;
    geometry_params.umin = 0.05;
    lcbi_options geometry_options = lcbi_default_options();

    lcbi_result geometry_result = {};
    if (lcbi_magnification(0.0, &geometry_params, &geometry_options, &geometry_result) != LCBI_OK) {
        return 50;
    }
    if (std::abs(geometry_result.separation - geometry_params.sep) > 1e-12 ||
        std::abs(geometry_result.mass_ratio - geometry_params.q) > 1e-12 ||
        !std::isfinite(geometry_result.caustic_distance)) {
        return 51;
    }

    lcbi_geometry geometry = {};
    if (lcbi_finite_source_geometry(0.0, &geometry_params, &geometry_options, &geometry) != LCBI_OK ||
        !geometry.valid) {
        return 52;
    }
    // geometry.source_{x,y} reflect the wide-binary-offset-shifted position
    // used internally for finite-source integration, which is intentionally
    // not the same frame as lcbi_result::source_{x,y} (captured before that
    // shift) -- only finiteness is checked here, not numeric equality.
    if (std::abs(geometry.separation - geometry_params.sep) > 1e-12 ||
        std::abs(geometry.mass_ratio - geometry_params.q) > 1e-12 ||
        !std::isfinite(geometry.source_x) || !std::isfinite(geometry.source_y) ||
        std::abs(geometry.source_radius - geometry_params.rho) > 1e-12) {
        return 53;
    }

    const double geometry_times[3] = {-0.5, 0.0, 0.5};
    lcbi_geometry geometry_array[3] = {};
    if (lcbi_finite_source_geometry_array(
            geometry_times, 3, &geometry_params, &geometry_options, geometry_array) != LCBI_OK) {
        return 54;
    }
    for (int i = 0; i < 3; ++i) {
        if (!geometry_array[i].valid ||
            std::abs(geometry_array[i].separation - geometry_params.sep) > 1e-12) {
            return 55;
        }
    }

    // Orbital motion: separation/mass_ratio must be time-evolved consistently
    // between lcbi_magnification and lcbi_finite_source_geometry.
    lcbi_params orbit_params = geometry_params;
    orbit_params.orbital_motion_mode = LCBI_ORBIT_CIRCULAR;
    orbit_params.g1 = 0.3;
    orbit_params.g2 = 0.1;
    orbit_params.g3 = 0.05;

    lcbi_result orbit_result_early = {};
    lcbi_result orbit_result_late = {};
    if (lcbi_magnification(-2.0, &orbit_params, &geometry_options, &orbit_result_early) != LCBI_OK ||
        lcbi_magnification(2.0, &orbit_params, &geometry_options, &orbit_result_late) != LCBI_OK) {
        return 56;
    }
    if (std::abs(orbit_result_early.separation - orbit_result_late.separation) < 1e-6) {
        return 57;
    }

    lcbi_geometry orbit_geometry_early = {};
    lcbi_geometry orbit_geometry_late = {};
    if (lcbi_finite_source_geometry(-2.0, &orbit_params, &geometry_options, &orbit_geometry_early) != LCBI_OK ||
        lcbi_finite_source_geometry(2.0, &orbit_params, &geometry_options, &orbit_geometry_late) != LCBI_OK) {
        return 58;
    }
    if (std::abs(orbit_geometry_early.separation - orbit_result_early.separation) > 1e-9 ||
        std::abs(orbit_geometry_late.separation - orbit_result_late.separation) > 1e-9) {
        return 59;
    }

    return 0;
}
