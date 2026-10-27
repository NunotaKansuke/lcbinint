#include "lcbinint/lcbinint.h"
#include "lcbinint/magnification/cartesian_run_fill.hpp"
#include "lcbinint/magnification/binary_caustic_enclosure.hpp"
#include "lcbinint/magnification/binary_caustic_tree.hpp"
#include "lcbinint/magnification/binary_local_seeds.hpp"
#include "lcbinint/magnification/algebraic_boundary/radial_support_plan.hpp"
#include "lcbinint/magnification/algebraic_boundary/tangent/geometry.hpp"
#include "lcbinint/magnification/algebraic_boundary/adaptive_radial.hpp"
#include "lcbinint/magnification/algebraic_boundary/fused_gradient.hpp"
#include "lcbinint/magnification/algebraic_boundary/tangent/adaptive_radial.hpp"
#include "lcbinint/magnification/algebraic_boundary/tangent/fused_gaussian.hpp"
#include "lcbinint/magnification/algebraic_boundary/tangent/fused_k.hpp"
#include "lcbinint/magnification/algebraic_boundary/tangent/adaptive_epoch.hpp"
#include "lcbinint/magnification/algebraic_boundary/tangent/seeded_radial_frontier.hpp"
#include "lcbinint/magnification/algebraic_boundary/seeded_radial_frontier_value.hpp"
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
using lcbinint::magnification::detail::CartesianRunFillStatus;
using lcbinint::magnification::detail::fill_cartesian_runs;
using lcbinint::magnification::detail::lift_cartesian_component_run_seeds;

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

bool cartesian_run_refinement_split_test()
{
    // One coarse component: the full upper run joins the two lower runs.
    const LatticeCells coarse_cells = mask_cells({"###", "#.#"});
    const auto coarse = fill_cartesian_runs<MaskCellState>(
        {{0, 0}},
        [&](std::int64_t ix, std::int64_t iy) {
            return MaskCellState {coarse_cells.count({ix, iy}) != 0};
        },
        [](const MaskCellState& state) { return state.inside; },
        [](std::int64_t, std::int64_t, const MaskCellState&) { return 1.0; },
        [](const auto&, const auto&, const auto&, const auto&, const auto&) {
            return CartesianBoundaryContribution {};
        },
        CartesianRunFillLimits {1000, 100});
    if (!coarse.ok() || coarse.runs.empty()) {
        return false;
    }

    constexpr std::int64_t factor = 3;
    const std::size_t component = coarse.runs.front().component;
    const auto seeds = lift_cartesian_component_run_seeds(
        coarse, component, factor);
    if (!seeds.has_value() || seeds->size() != 3) {
        return false;
    }

    // At the fine centers the coarse bridge aliases away and the same coarse
    // component is represented by two disconnected pieces.  Its first seed
    // reaches only the left piece; one exact seed per coarse run reaches both.
    LatticeCells fine_cells;
    for (std::int64_t x = 0; x <= 3; ++x) {
        fine_cells.emplace(x, 0);
    }
    for (std::int64_t y = 1; y <= 3; ++y) {
        fine_cells.emplace(0, y);
        fine_cells.emplace(6, y);
    }
    fine_cells.emplace(6, 0);
    const auto fill_fine = [&](std::vector<CartesianLatticeSeed> fine_seeds) {
        LatticeCells visited;
        const auto result = fill_cartesian_runs<MaskCellState>(
            std::move(fine_seeds),
            [&](std::int64_t ix, std::int64_t iy) {
                return MaskCellState {fine_cells.count({ix, iy}) != 0};
            },
            [](const MaskCellState& state) { return state.inside; },
            [&](std::int64_t ix, std::int64_t iy, const MaskCellState&) {
                visited.emplace(ix, iy);
                return 1.0;
            },
            [](const auto&, const auto&, const auto&, const auto&, const auto&) {
                return CartesianBoundaryContribution {};
            },
            CartesianRunFillLimits {1000, 100});
        return std::make_pair(result.ok(), std::move(visited));
    };
    const auto one_seed = fill_fine({seeds->front()});
    const auto all_seeds = fill_fine(*seeds);
    return one_seed.first && one_seed.second != fine_cells &&
        all_seeds.first && all_seeds.second == fine_cells;
}

bool cartesian_run_budget_test()
{
    const LatticeCells cells = mask_cells({"#..", ".#.", "..#"});
    const auto result = fill_cartesian_runs<MaskCellState>(
        {{0, 0}},
        [&](std::int64_t ix, std::int64_t iy) {
            return MaskCellState {cells.count({ix, iy}) != 0};
        },
        [](const MaskCellState& state) { return state.inside; },
        [](std::int64_t, std::int64_t, const MaskCellState&) { return 1.0; },
        [](const auto&, const auto&, const auto&, const auto&, const auto&) {
            return CartesianBoundaryContribution {};
        },
        CartesianRunFillLimits {100, 2});
    return result.status == CartesianRunFillStatus::run_budget_exhausted &&
        result.runs.size() == 2;
}

} // namespace

int main()
{
    const auto check_complete_candidate=[](auto config, auto workspace, auto sample, auto integrate) {
        using Panel=typename decltype(workspace.panels)::value_type;
        for(int index=0;index<2;++index) {
            Panel panel;
            panel.map={double(2*index),double(2*index+2),false,false};
            panel.cell=panel.cell_end=index;
            workspace.panels.push_back(panel);
        }
        config.initial_level=8;
        const auto evaluate=[&](double, double jacobian, int cell,
                                const auto*, bool, double, int level) {
            auto result=sample;
            result.value.fill(jacobian*(cell+1));
            result.reliable=cell!=0 || level==3;
            return result;
        };
        const auto result=integrate(workspace,config,evaluate);
        return !result.value_converged &&
            std::abs(result.mu-6.0)<1e-10 &&
            std::isinf(result.estimated_abs_error_mu) &&
            (!config.with_jacobian || std::abs(result.grad_mu[0]-6.0)<1e-10) &&
            workspace.panels[0].level==3 && workspace.panels[1].level==8;
    };
    namespace boundary=lcbinint::algebraic_boundary;
    if(!check_complete_candidate(boundary::AdaptiveConfig{},boundary::AdaptiveWorkspace{},
        boundary::AdaptiveSample{},[](auto& workspace,const auto& config,auto evaluate) {
            return boundary::adaptive_detail::integrate(workspace,config,evaluate);
        }))return 110;
    namespace tangent=boundary::tangent;
    tangent::AdaptiveConfig tangent_config;
    tangent_config.with_jacobian=true;
    if(!check_complete_candidate(tangent_config,tangent::AdaptiveWorkspace{},
        tangent::AdaptiveSample{},[](auto& workspace,const auto& config,auto evaluate) {
            return tangent::adaptive_detail::integrate(workspace,config,evaluate);
        }))return 111;
    if (!cartesian_run_topology_tests()) {
        return 62;
    }
    if (!cartesian_run_boundary_tests()) {
        return 63;
    }
    if (!cartesian_run_refinement_split_test()) {
        return 64;
    }
    if (!cartesian_run_budget_test()) {
        return 65;
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
    // Retained roots/physical masks must survive source jumps, mass swaps
    // and evolving geometry. Compare to independent cold starts; subsequent
    // scalar requests at the same centre also exercise cache reuse.
    for (const double q : {0.01, 100.0, 0.3, 1.0/0.3}) {
        for (int i = 0; i < 80; ++i) {
            const double separation = 0.7+0.008*i;
            const lcbinint::SourcePosition source {
                (i%2 == 0 ? 1.0 : -1.0)*(0.5-0.01*i), 0.04+0.001*i};
            lcbinint::magnification::PointSourceMagnifier cold;
            const auto expected = cold.binary_safety_diagnostic_cached(separation, q, source);
            const auto retained = point_magnifier.binary_safety_diagnostic_cached(separation, q, source);
            if (expected.image_count != retained.image_count ||
                expected.ghost_count != retained.ghost_count ||
                !close_relative(retained.magnification, expected.magnification, 1e-10)) return 150;
            const auto scalar = point_magnifier.binary_mag0_cached(separation, q, source);
            const auto after_scalar = point_magnifier.binary_safety_diagnostic_cached(separation, q, source);
            if (scalar.image_count != expected.image_count ||
                !close_relative(after_scalar.magnification, expected.magnification, 1e-10)) return 151;
        }
    }
    // Geometry derivatives must agree with independent cold solves even when
    // only the lenses move, and on both sides of the q=1 frame change.
    for(double q_base : {0.01, 100.0, 0.3, 1.0/0.3, 1.0}) {
        for(bool moving_source : {false, true}) {
            lcbinint::magnification::PointSourceMagnifier retained;
            for(int i=0;i<160;++i) {
                const double s=0.9+0.002*std::sin(0.13*i);
                const double q=q_base*(1.0+0.0005*std::sin(0.17*i));
                const lcbinint::SourcePosition source{
                    0.02+(moving_source?0.01*std::sin(0.11*i):0.0), 0.03};
                lcbinint::magnification::PointSourceMagnifier cold;
                const auto expected=cold.binary_safety_diagnostic_cached(s,q,source);
                const auto actual=retained.binary_safety_diagnostic_cached(s,q,source);
                if(actual.image_count!=expected.image_count ||
                   actual.ghost_count!=expected.ghost_count ||
                   !close_relative(actual.magnification,expected.magnification,1e-9))return 155;
            }
        }
    }
    // Optional batch image handoff must preserve the scalar flux and satisfy
    // the physical lens equation on both sides of the q=1 frame convention.
    for (const double q : {0.01, 0.5, 2.0}) {
        const std::array<lcbinint::SourcePosition, 3> sources{{
            {0.2, 0.03}, {0.4, -0.02}, {-0.1, 0.08}}};
        std::array<double, 3> plain{}, captured{};
        std::array<std::vector<lcbinint::magnification::BinaryImage>, 3> images;
        point_magnifier.binary_mag0_batch(1.3, q, sources.data(), plain.data(), sources.size());
        point_magnifier.binary_mag0_batch(1.3, q, sources.data(), captured.data(),
            sources.size(), images.data());
        for (std::size_t i = 0; i < sources.size(); ++i) {
            if (plain[i] != captured[i] || (images[i].size() != 3 && images[i].size() != 5)) return 152;
            double flux = 0.0;
            for (const auto& image : images[i]) {
                const auto mapped = point_magnifier.binary_lens_equation(1.3, q, image.position);
                if (std::hypot(mapped.x - sources[i].x, mapped.y - sources[i].y) > 1e-8) return 153;
                flux += 1.0 / std::abs(image.jacobian_determinant);
            }
            if (!close_relative(flux, captured[i], 1e-13)) return 154;
        }
    }
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

    // Check the radial caustic bound against independently solved phase roots,
    // including close, resonant, wide and reciprocal-mass configurations.
    for (double s : {0.1, 0.7, 1.0, 1.2, 3.0}) {
        for (double q_input : {1e-4, 0.1, 1.0, 10.0, 1e4}) {
            const double q = std::min(q_input, 1.0/q_input);
            const double a = q_input < 1.0 ? -s : s;
            const double m1 = 1.0/(1.0+q), m2 = q*m1;
            for (int k = 0; k < 137; ++k) {
                const auto phase = std::polar(1.0, 6.283185307179586*(k+0.371)/137);
                const auto roots = solver.solve({m2*a*a, -2.0*m2*a,
                    1.0-phase*a*a, 2.0*phase*a, -phase});
                if (roots.status != lcbinint::math::RootSolverStatus::ok) return 92;
                for (const auto z : roots.roots) {
                    const auto w = z-m1/(std::conj(z)-a)-m2/std::conj(z)-a*m1;
                    for (const lcbinint::SourcePosition source : {
                            lcbinint::SourcePosition{20.0, 0.0}, {-30.0, 10.0}, {0.0, 0.0},
                            {-0.8, 0.02}, {0.8, -0.02}, {0.2, 0.1}, {-1.0, 0.05}}) {
                        const double lower = lcbinint::magnification::detail::
                            binary_caustic_radial_clearance(s, q_input, source);
                        if (lower > std::hypot(source.x-w.real(), source.y-w.imag())+1e-8)
                            return 93;
                    }
                }
            }
        }
    }

    int proved_intervals = 0;
    for (double s : {0.7, 0.89, 1.2, 2.4}) {
        for (double q_input : {0.011, 0.1, 10.0, 1.0/0.011}) {
            const double width = 1e-4*std::min(1.0, s);
            const auto cover = lcbinint::magnification::detail::
                build_binary_caustic_enclosure(s, q_input, 4, 32, width);
            if (cover.empty()) continue;
            ++proved_intervals;
            const double q = std::min(q_input, 1.0/q_input);
            const double m1 = 1.0/(1.0+q), m2 = q*m1;
            for (double offset : {-1.0, -0.5, 0.0, 0.5, 1.0}) {
                const double current = s+offset*width;
                const double a = q_input < 1.0 ? -current : current;
                for (int k = 0; k < 137; ++k) {
                    const auto phase = std::polar(1.0, 6.283185307179586*(k+0.371)/137);
                    const auto roots = solver.solve({m2*a*a, -2.0*m2*a,
                        1.0-phase*a*a, 2.0*phase*a, -phase});
                    if (roots.status != lcbinint::math::RootSolverStatus::ok) return 94;
                    for (const auto z : roots.roots) {
                        const auto w = z-m1/(std::conj(z)-a)-m2/std::conj(z)-a*m1;
                        bool included = false;
                        for (const auto& disk : cover)
                            included = included || std::hypot(w.real()-disk.first.x,
                                w.imag()-disk.first.y) <= disk.second+1e-8;
                        if (!included) return 95;
                    }
                }
            }
        }
    }
    if (proved_intervals == 0) return 96;
    // A one-root disk cover cannot certify a parameter interval spanning a
    // topology transition. Failure leaves the exact-geometry fallback active.
    if (!lcbinint::magnification::detail::build_binary_caustic_enclosure(
            2.0, 1.0, 4, 32, 1e-4).empty()) return 97;

    // Warm critical roots must reproduce all independent phase roots, including
    // odd phase counts, reciprocal masses and topology changes.
    for (int bins : {63, 64}) {
        lcbinint::magnification::FiniteSourceSettings settings;
        settings.caustic_bins = bins;
        lcbinint::magnification::FiniteSourceMagnifier finite(settings);
        for (double q_base : {1e-4, 0.1, 1.0, 10.0, 1e4}) {
            for (double s : {0.7, 0.700001, 1.999999, 2.0, 2.000001, 3.0, 0.7}) {
              for(double factor : {1.0, 1.0001, 0.9998, 1.002}) {
                const double q_input=q_base*factor;
                const auto curves = finite.binary_critical_curve_branches(s, q_input);
                std::size_t count = 0;
                for (const auto& curve : curves) count += curve.size();
                if (count != static_cast<std::size_t>(4*bins)) return 98;
                const double q = std::min(q_input, 1.0/q_input);
                const double a = q_input < 1.0 ? -s : s;
                const double m1 = 1.0/(1.0+q), m2 = q*m1;
                for (int k = 0; k < bins; ++k) {
                    const auto phase = std::polar(1.0, 6.283185307179586*k/bins);
                    const auto roots = solver.solve({m2*a*a,-2.0*m2*a,
                        1.0-phase*a*a,2.0*phase*a,-phase});
                    for (const auto z : roots.roots) {
                        double nearest = std::numeric_limits<double>::infinity();
                        for (const auto& curve : curves) for (const auto pt : curve)
                            nearest = std::min(nearest,std::abs(z-lcbinint::Complex(pt.x,pt.y)));
                        if (!(nearest < 1e-7*(1.0+std::abs(z)))) return 99;
                    }
                }
              }
            }
        }
    }

    int tree_accepts = 0;
    lcbinint::magnification::detail::BinaryCausticTree tree;
    for (double s : {0.3, 0.7, 1.0, 1.2, 2.0, 3.0}) {
        for (double q_input : {1e-4, 0.01, 1.0, 10.0, 1e4}) {
            const double width = 1e-4*std::min(s, 1.0);
            tree.clearance(s, q_input, {3.0, 2.0}, 1e-4, width);
            for (double offset : {-2.0, -0.9, 0.0, 0.9, 2.0}) {
                const double current = s+offset*width;
                const double q = std::min(q_input, 1.0/q_input);
                const double a = q_input < 1.0 ? -current : current;
                const double m1 = 1.0/(1.0+q), m2 = 1.0-m1;
                for (int k = 0; k < 17; ++k) {
                    const auto phase = std::polar(1.0, 6.283185307179586*(k+0.371)/17);
                    const auto roots = solver.solve({m2*a*a, -2.0*m2*a,
                        1.0-phase*a*a, 2.0*phase*a, -phase});
                    for (const auto z : roots.roots) {
                        const auto w = z-m1/(std::conj(z)-a)-m2/std::conj(z)-a*m1;
                        for (double radius : {1e-4, 0.01, 0.1}) {
                            const lcbinint::SourcePosition source {
                                w.real()+0.7*radius, w.imag()+0.2*radius};
                            if (tree.clearance(current, q_input, source, radius, width)>radius)
                                return 100;
                        }
                    }
                }
                tree_accepts += tree.clearance(current, q_input, {3.0, 2.0}, 1e-4, width)>1e-4;
            }
        }
    }
    if (tree_accepts == 0) return 101;
    lcbinint::magnification::detail::clear_binary_caustic_tree_cache();
    if (lcbinint::magnification::detail::binary_caustic_disk_clearance(
            1.2, 0.01, {3.0, 2.0}, 0.001, 0.001)<=0.001) return 102;
    if (lcbinint::magnification::detail::binary_caustic_disk_clearance(
            1.2, 0.01, {NAN, 2.0}, 0.001, 0.001)!=0.0) return 103;

    // A source disk containing a critical point must not be classified as
    // having no critical arcs. Verify returned seeds through the lens map,
    // independently of the phase enclosure's source-plane disk centres.
    int local_completed = 0;
    for(double separation : {0.5, 1.0, 1.7, 3.0})
      for(double mass_ratio : {1e-3, 0.1, 1.0, 10.0, 1e3}) {
        const double q=std::min(mass_ratio,1.0/mass_ratio);
        const double a=mass_ratio<1 ? -separation : separation;
        const double m1=1/(1+q),m2=q*m1;
        const auto phase=std::polar(1.0,0.731);
        const auto critical=solver.solve({m2*a*a,-2*m2*a,
            1.0-phase*a*a,2.0*phase*a,-phase});
        for(auto z:critical.roots)for(double rho : {1e-4, 0.01, 0.3}) {
            const auto w=z-m1/(std::conj(z)-a)-m2/std::conj(z)-a*m1;
            const lcbinint::SourcePosition source{w.real()+0.5*rho,w.imag()+0.2*rho};
            auto local=lcbinint::magnification::detail::binary_local_image_seeds(
                separation,mass_ratio,source,rho,1024);
            if(local.completed) {
                ++local_completed;
                if(local.seeds.empty())return 104;
                for(auto seed:local.seeds) {
                    const lcbinint::Complex image(seed.x,seed.y);
                    const auto mapped=image-m1/(std::conj(image)-a)-m2/std::conj(image)-a*m1;
                    if(!(std::abs(mapped-lcbinint::Complex(source.x,source.y))<rho))return 105;
                }
            }
        }
      }
    if(local_completed==0)return 106;
    if(lcbinint::magnification::detail::binary_local_image_seeds(
        1.2,0.01,{0.0,0.0},0.001,0).completed)return 107;
    if(lcbinint::magnification::detail::binary_local_image_seeds(
        1.2,0.01,{NAN,0.0},0.001).completed)return 108;

    namespace exact = lcbinint::algebraic_boundary::exact_support_detail;
    auto exact_roots_match = [](const std::array<exact::Dyadic, 5>& p,
                               const std::vector<double>& expected) {
        const auto roots = exact::angular_roots(p);
        if (!roots.completed || roots.boxes.size() != expected.size()) return false;
        for (std::size_t i = 0; i < expected.size(); ++i)
            if (!(roots.boxes[i].lo <= expected[i] &&
                  roots.boxes[i].hi >= expected[i])) return false;
        return true;
    };
    using D = exact::Dyadic;
    if (!exact_roots_match({D(.03515625), D(0), D(-.625), D(0), D(1)},
                          {-.75, -.25, .25, .75})) return 109;
    if (!exact_roots_match({D(-.0625), D(0), D(0), D(0), D(1)},
                          {-.5, .5})) return 110;
    if (!exact_roots_match({D(0), D(.5), D(-1), D(-.5), D(1)},
                          {-1, 0, .5, 1})) return 111;
    if (!exact_roots_match({D(.0625), D(-.5), D(1), D(0), D(0)},
                          {.25})) return 112;
    if (exact::angular_roots({}).completed) return 113;

    const D exact_zero = D(0)*D(.1);
    if (exact_zero.n != 0 || exact_zero.e != 0) return 114;
    if (!exact_roots_match({D(1), D(0), D(0), D(0), exact_zero}, {})) return 115;

    if(exact::exact_angular_count({D(.140625),D(0),D(-.8125),D(0),D(1)},1)!=4)return 116;
    if(exact::exact_angular_count({D(-.0625),D(0),D(0),D(0),D(1)},-1)!=2)return 117;
    if(exact::exact_angular_count({D(1),D(0),D(0),D(0),D(1)},1)!=0)return 118;
    if(exact::exact_angular_count({D(1),D(0),D(0),D(0),D(1)},0)!=-1)return 119;
    if(exact::radial_plan({1.,.5,0.,0.,0.}).completed)return 120;
    namespace bounded = exact::bounded_detail;
    for(int n=-101;n<=101;++n)for(int bits=0;bits<=14;++bits)
        if(bounded::floor_div(bounded::Coefficient(n),bits)!=
           static_cast<long long>(std::floor(std::ldexp(double(n),-bits))))return 121;
    for(unsigned error:{0u,3u,1000u}) {
        bounded::CoefficientBox box;box.hi=1;box.error=error;
        exact::Box lower,upper;lower.hi=upper.hi=1;
        for(int i=0;i<15;++i) {
            box.b[i]=(bounded::Coefficient(1)<<223)+(17*i-143);
            if(i%2)box.b[i]=-box.b[i];
            lower.b[i]=exact::Integer(box.b[i])-error;
            upper.b[i]=exact::Integer(box.b[i])+error;
        }
        exact::Integer scale=1;
        for(int depth=0;depth<12;++depth) {
            auto ball_children=bounded::split(box);
            auto low_children=exact::split(lower),high_children=exact::split(upper);
            box=depth%2?std::move(ball_children.first):std::move(ball_children.second);
            lower=depth%2?std::move(low_children.first):std::move(low_children.second);
            upper=depth%2?std::move(high_children.first):std::move(high_children.second);
            scale*=16384;
            for(int i=0;i<15;++i) {
                const exact::Integer center(box.b[i]);
                if((center-box.error)*scale>lower.b[i] ||
                   (center+box.error)*scale<upper.b[i])return 122;
            }
        }
    }
    exact::Box uncertain;uncertain.hi=1;
    for(auto& coefficient:uncertain.b)coefficient=exact::Integer(1)<<200;
    uncertain.b[0]=1;
    if(bounded::isolate_box(uncertain,32).completed)return 123;
    if(bounded::isolate_box(uncertain,225).completed)return 124;
    {
        const auto plan=exact::radial_plan(lcbinint::algebraic_boundary::PrimaryFrame::from(
            {6.020866474779279e-6,-8.529681383148756e-6,2.6295524980988746e-6,
             33561.65273177007,0.5860467981775465,true}));
        if(!plan.completed)return 144;
        for(const auto& bracket:plan.brackets)for(std::size_t slot=0;slot<4;++slot) {
            const auto witness=std::find(plan.radii.begin(),plan.radii.end(),bracket[slot]);
            if(witness==plan.radii.end())return 145;
            const int count=plan.crossings[witness-plan.radii.begin()];
            if((slot<2)!=(count!=0))return 146;
        }
    }
    {
        using lcbinint::SourcePosition;
        using lcbinint::magnification::PointSourceMagnifier;
        PointSourceMagnifier cached;
        const auto first = cached.binary_mag0_cached(1.2, 0.1, {0.2, 0.3});
        const auto nearby = cached.binary_mag0_cached(1.2, 0.1, {0.20001, 0.30001});
        const auto distant = cached.binary_mag0_cached(1.2, 0.1, {10.0, -20.0});
        const auto changed_lens = cached.binary_mag0_cached(1.8, 0.1, {10.0, -20.0});
        if (first.root_used_warm_start || !nearby.root_used_warm_start ||
            distant.root_used_warm_start || changed_lens.root_used_warm_start)
            return 125;
        const SourcePosition sources[]{{0.2, 0.3}, {10.0, -20.0},
                                      {-8.0, 15.0}, {0.20001, 0.30001}};
        double values[4]{};
        cached.binary_mag0_batch(1.2, 0.1, sources, values, 4);
        for (int j = 0; j < 4; ++j) {
            PointSourceMagnifier cold;
            const auto reference = cold.binary_mag0(1.2, 0.1, sources[j]);
            if (!std::isfinite(values[j]) || std::abs(values[j] - reference.magnification) >
                    1e-9 * std::max(1.0, std::abs(reference.magnification)))
                return 126;
        }
    }
    {
        using namespace lcbinint::magnification;
        FiniteSourceSettings settings;
        settings.source_bins = 80;
        settings.max_source_bins = 400;
        settings.limb_darkening_c = 0.6584;
        settings.finite_source_tol = 1e-4;
        settings.finite_source_reltol = 1e-3;
        const FiniteSourceMagnifier magnifier(settings);
        const lcbinint::SourcePosition sources[] = {
            {1.9483916678579025e-05, -2.6048659958853258e-05},
            {9.510709675452556e-06, -1.3070887672932013e-05},
            {2.8765908804010926e-06, -4.4381496767249629e-06},
            {-1.3291057832425399e-05, 1.6600224772742906e-05}
        };
        AlgebraicBoundarySettings tolerances;
        tolerances.absolute_tolerance = 1e-4;
        tolerances.relative_tolerance = 1e-3;
        for (const auto& source : sources) {
            magnifier.clear_binary_epoch_caches();
            FiniteSourceResult value;
            magnifier.algebraic_boundary_binary_mag(
                0.5860467981775465, 1 / 2.9795910469372738e-5, source,
                2.6295524980988746e-6, tolerances, &value, nullptr, true);
            if (!(std::isfinite(value.magnification) && value.magnification > 0.0) ||
                value.decision.estimated_evaluations == 0)
                return 132;
            if (value.decision.method == FiniteSourceMethod::algebraic_boundary &&
                value.diagnostic_stop.empty())
                return 133;
            FiniteSourceResult derivative;
            const auto jacobian = magnifier.algebraic_boundary_binary_jacobian(
                0.5860467981775465, 1 / 2.9795910469372738e-5, source,
                2.6295524980988746e-6, tolerances, &derivative, nullptr, true);
            if (std::abs(value.magnification - derivative.magnification) >
                tolerances.absolute_tolerance + tolerances.relative_tolerance * std::abs(value.magnification))
                return 134;
            if (source.y == -4.4381496767249629e-06) {
                bool nonzero = false;
                for (const auto component : jacobian.jacobian) {
                    if (!std::isfinite(component)) return 135;
                    nonzero = nonzero || component != 0.0;
                }
                if (!nonzero || jacobian.grad_reliable) return 136;
                if (derivative.diagnostic_stop.find("unresolved_outer_edge_pair") == std::string::npos)
                    return 137;
            }
        }
    }
    {
        namespace tangent = lcbinint::algebraic_boundary::tangent;
        tangent::AdaptiveConfig config;
        struct Endpoint {
            double radius=0,radius_lo=0,pair_m=0,t_seed=0,u_seed=0;
            bool pair_motion_valid=false,pair_reciprocal=false;
            bool t_seed_valid=false,u_seed_valid=false;
        };
        tangent::PrimaryFrame frame;
        frame.a=1.2;frame.m0=1;frame.X=.2;frame.Y=0;frame.rho=1e-7;
        const long double edge=(long double)frame.X+frame.rho;
        const long double radius=(edge+sqrtl(edge*edge+4))/2;
        Endpoint left,right;
        left.radius=.9;
        right.radius=double(radius);right.radius_lo=double(radius-right.radius);
        right.t_seed_valid=true;
        const auto material=tangent::fused_gradient::make_material_map(left,right,frame);
        const double expected_motion=1/(1+1/double(radius*radius));
        if(!material.valid||material.derivatives_complete||
            std::fabs(material.db[0]-expected_motion)>1e-10)return 147;

        tangent::AdaptiveWorkspace workspace;
        tangent::AdaptivePanel panel;
        panel.map={0.0,1.0,false,false};
        panel.level=3;panel.active=true;
        for(int axis=0;axis<5;++axis)panel.q[axis+1]=axis+1;
        workspace.panels.push_back(panel);
        const auto evaluator=[](double radius,double jacobian,int,
            const tangent::AdaptiveSample*,bool,double,int,tangent::AdaptiveHermiteJet*) {
            tangent::AdaptiveSample sample;
            for(int axis=0;axis<5;++axis)
                sample.value[axis+1]=(axis+1)*jacobian*std::exp(40*(radius-1));
            return sample;
        };
        {
            const tangent::LensParams thin{2.026496540589604e-5,-2.7065010318787755e-5,
                2.6295524980988746e-6,33561.65273177007,.5860467981775465,true};
            const auto thin_frame=tangent::PrimaryFrame::from(thin);
            const auto coefficients=tangent::boundary_quartic_extended(1.3225423068538684,thin_frame);
            const auto crossings=tangent::certify_quartic(coefficients.p);
            if(!crossings.certified || crossings.root_count!=2)return 150;
            const tangent::PrimaryFrame input{.7,.3,.15,-.12,.3};
            const double radius=1.2,angle=.7,step=1e-6;
            const tangent::fused_gradient::PreciseTangentScope precision;
            const tangent::fused_gradient::ComTangentScope com_coordinates(true);
            const auto center=tangent::fused_precise_phi(radius,angle,input);
            const auto shifted=[&](int axis,double offset) {
                auto changed=input;
                if(axis==3)changed.m0+=offset;
                else changed.a+=offset;
                const double image_x=radius*std::cos(angle)+(1-changed.m0)*changed.a-
                    (1-input.m0)*input.a;
                const double image_y=radius*std::sin(angle);
                return tangent::fused_precise_phi(std::hypot(image_x,image_y),
                    std::atan2(image_y,image_x),changed).phi;
            };
            for(int axis:{3,4}) {
                const double reference=(shifted(axis,step)-shifted(axis,-step))/(2*step);
                if(std::fabs(center.dP[axis]-reference)>1e-6*std::max(1.,std::fabs(reference)))return 149;
            }
        }
        {
            const tangent::LensParams parameters{.5,.3,.01,.2,.9,true};
            tangent::AdaptiveConfig limited;
            limited.with_jacobian=true;limited.fused_value_mesh=true;
            limited.gradient_policy=tangent::GradientPolicy::ValueFirst;
            limited.fold_maps=false;limited.max_node_evals=3;
            tangent::AdaptiveWorkspace limited_workspace;
            const auto candidate=tangent::epoch_adaptive(parameters,.5,limited,limited_workspace);
            if(candidate.value_converged || !(candidate.mu>0) ||
                candidate.stats.node_evaluations<=limited.max_node_evals)return 152;
            if(!std::all_of(candidate.grad_mu.begin(),candidate.grad_mu.end(),
                [](double value){return std::isfinite(value);}) ||
                !std::any_of(candidate.grad_mu.begin(),candidate.grad_mu.end(),
                [](double value){return value!=0;}))return 153;
        }
        const auto replay=tangent::fused_gradient::gaussian_replay(workspace,config,evaluator,0);
        auto changed_tolerance=config;
        changed_tolerance.tol.mu_rtol=.5;

        const auto unchanged=tangent::fused_gradient::gaussian_replay(
            workspace,changed_tolerance,evaluator,0);
        auto exhausted=config;
        exhausted.max_node_evals=1;
        const auto completed=tangent::fused_gradient::gaussian_replay(workspace,exhausted,evaluator,1);
        for(int axis=0;axis<5;++axis) {
            const double expected=(axis+1)*(1-std::exp(-40.0))/40;
            if(std::fabs(replay.grad_mu[axis]-expected)>1e-8*expected)return 141;
            if(replay.grad_mu[axis]!=unchanged.grad_mu[axis]||
                replay.grad_quality[axis]!=tangent::GradientQuality::FiniteValid)return 148;
            if(completed.grad_mu[axis]!=replay.grad_mu[axis] ||
                completed.grad_quality[axis]!=tangent::GradientQuality::FiniteValid)return 151;
        }
        const auto failed=[](double,double,int,const tangent::AdaptiveSample*,
            bool,double,int,tangent::AdaptiveHermiteJet*) {
            tangent::AdaptiveSample sample;sample.reliable=false;return sample;
        };
        const auto retained=tangent::fused_gradient::gaussian_replay(workspace,config,failed,0);
        for(int axis=0;axis<5;++axis) {
            if(retained.grad_mu[axis]!=axis+1)return 142;
            if(retained.grad_quality[axis]!=tangent::GradientQuality::FiniteValid)return 143;
        }
    }
    {
        namespace boundary = lcbinint::algebraic_boundary;
        boundary::RadialSupportPlan unresolved;
        unresolved.completed = true;
        unresolved.brackets.push_back({10.0, 11.0, 9.0, 12.0});
        const auto recovered = boundary::epoch_fused_gradient(
            {0.2, 0.3, 0.01, 0.1, 1.2, true}, 0.6, {{1.0, 0.0}},
            {1e-4, 1e-3}, &unresolved);
        if (!(recovered.mu > 0.0) || !std::isfinite(recovered.mu) ||
            recovered.node_evaluations == 0) return 138;
        bool nonzero = false;
        for (const double component : recovered.grad_mu) {
            if (!std::isfinite(component)) return 139;
            nonzero = nonzero || component != 0.0;
        }
        if (!nonzero || recovered.support_certified) return 140;
    }
    {
        namespace tangent = lcbinint::algebraic_boundary::tangent;
        const tangent::LensParams parameters{5.2624457017694967e-7,2.434553071365692e-8,
            4.5103932631467525e-6,26862.57757654888,.562732272146788,true};
        const auto frame=tangent::PrimaryFrame::from(parameters);
        const double radius=1.1474092888475518;
        tangent::fused_gradient::PreciseTangentScope precision;
        tangent::fused_gradient::ComTangentScope transport(true);
        const auto coefficients=tangent::boundary_quartic_extended(radius,frame);
        tangent::QuarticWarm warm;
        tangent::RootPairWarm roots;
        tangent::QuarticCoeffs output;
        const auto arcs=tangent::adaptive_detail::adaptive_arc_intervals(
            radius,frame,&warm,&roots,&output,2,&coefficients);
        if(arcs.kind!=tangent::ArcKind::kArcs||arcs.arcs.size()!=1)return 155;
        const auto& interval=arcs.arcs.front();
        if(!(interval[1]-interval[0]>0&&interval[1]-interval[0]<2e-9))return 156;
        for(int index=1;index<64;++index)
            if(!(tangent::phi_val(radius,interval[0]+(interval[1]-interval[0])*index/64.,frame)>0))return 157;
    }
    {
        namespace tangent = lcbinint::algebraic_boundary::tangent;
        const tangent::LensParams parameters{-3.8732481187648279e-7,2.4231648232616164e-8,
            2.132335711843209e-6,40725.031958015978,.56542633134691833,true};
        const auto frame=tangent::PrimaryFrame::from(parameters);
        const double radius=.9133331347803566;
        tangent::fused_gradient::PreciseTangentScope precision;
        tangent::fused_gradient::ComTangentScope transport(true);
        const auto coefficients=tangent::boundary_quartic_extended(radius,frame);
        tangent::QuarticWarm warm;
        tangent::RootPairWarm roots;
        tangent::QuarticCoeffs output;
        const auto arcs=tangent::adaptive_detail::adaptive_arc_intervals(
            radius,frame,&warm,&roots,&output,2,&coefficients);
        if(arcs.kind!=tangent::ArcKind::kArcs||arcs.arcs.size()!=1)return 158;
        const auto& interval=arcs.arcs.front();
        if(!(interval[1]-interval[0]>4e-9&&interval[1]-interval[0]<5e-9))return 159;
        for(double angle:interval) {
            const auto polished=tangent::polish_endpoint(radius,angle,frame);
            if(!polished.reliable||std::fabs(polished.theta-angle)>2e-15)return 160;
        }
        tangent::CellPlan cell;
        cell.kind=tangent::ArcKind::kArcs;cell.n_crossings=-1;
        tangent::AdaptiveConfig config;
        const auto sample=tangent::adaptive_detail::mapped_radius(radius,1,parameters,.6584,
            frame,cell,true,false,nullptr,&config);
        if(!sample.reliable)return 161;
        for(double component:sample.value)if(!std::isfinite(component))return 162;
        for(int inherited_count:{0,4}) {
            cell.n_crossings=inherited_count;
            cell.kind=inherited_count==0?tangent::ArcKind::kEmpty:tangent::ArcKind::kArcs;
            const auto inherited=tangent::adaptive_detail::mapped_radius(radius,1,parameters,.6584,
                frame,cell,true,false,nullptr,&config);
            if(!inherited.reliable||inherited.local_crossings!=2)return 163;
            if(inherited.value!=sample.value)return 164;
        }
    }
    {
        namespace boundary = lcbinint::algebraic_boundary;
        namespace tangent = boundary::tangent;
        const tangent::LensParams parameters{-4.4264421197973816e-7,-9.248813492409009e-9,
            1.9744542129953203e-6,44218.353678937485,.57918659571672504,true};
        lcbinint::magnification::FiniteSourceMagnifier finite({});
        lcbinint::magnification::PointSourceMagnifier point;
        const auto seeds=boundary::make_caustic_aware_binary_seeds(finite,point,
            parameters.a,parameters.q,{parameters.xs,parameters.ys},parameters.rho);
        std::vector<tangent::Cplx<double>> positions;
        for(const auto& seed:seeds.radial_seeds)positions.push_back({seed.re,seed.im});
        tangent::AdaptiveConfig config;
        config.with_jacobian=true;config.fused_value_mesh=false;config.fold_maps=false;
        config.material_coordinate_motion=false;
        config.gradient_policy=tangent::GradientPolicy::ValueFirst;

        config.tol.mu_atol=1e-4;config.tol.mu_rtol=1e-7;
        tangent::AdaptiveWorkspace workspace;
        const auto result=tangent::epoch_seeded_radial_frontier(parameters,.6584,positions,
            config,workspace,false,nullptr,true);
        const double expected_fold=1.0654764948272739;
        bool retained=false;
        for(const auto& event:result.topology.events)
            if(event.physically_real && std::fabs(event.radius-expected_fold)<1e-14)retained=true;
        if(!retained)return 165;
        for(const auto& cell:result.topology.cells)
            if(cell.r_lo<expected_fold && expected_fold<cell.r_hi)return 166;
        if(!std::isfinite(result.integral.grad_mu[2]) ||
           std::fabs(result.integral.grad_mu[2]/1.6253000169243486e9-1)>2e-3)return 167;
        if(result.frontier.coverage_certified)return 168;
    }
#if defined(__SSE__)
    {
        namespace tangent = lcbinint::algebraic_boundary::tangent;
        namespace geometry = tangent::geometry_experiment;
        const auto saved = _mm_getcsr();
        _mm_setcsr(saved | 0x8040u);
        const auto entry = _mm_getcsr();
        tangent::TopologyResult topology;
        tangent::CellPlan cell;
        cell.kind = tangent::ArcKind::kEmpty;
        cell.r_lo = 0.5;
        cell.r_hi = 1.5;
        topology.cells.push_back(cell);
        const auto frame = tangent::PrimaryFrame::from({0.2, 0.3, 0.001, 0.1, 1.2, true});
        const auto clear_before = geometry::screen_stats.epochs_clear;
        bool complete = false;
        const auto pairs = geometry::detect(frame, topology, {}, &complete);
        const bool restored = _mm_getcsr() == entry;
        const bool screened = geometry::screen_stats.epochs_clear == clear_before + 1;
        geometry::known_internal_pairs(frame, topology, 7);
        geometry::known_upper_edge_pairs(frame, topology, 7);
        const bool helpers_restored = _mm_getcsr() == entry;
        _mm_setcsr(saved);
        if (!restored || !helpers_restored) return 127;
        if (!screened || !complete || !pairs.empty()) return 128;
    }
#endif
    {
        namespace recovery=lcbinint::algebraic_boundary::tangent;
        if(recovery::seeded_frontier_detail::physical_crossing_count(1.,{0.,1.,0.,0.,0.})!=-1)
            return 202;
        for(double radius_factor:{.5,1.,2.}) {
            const recovery::LensParams parameters{9.3455853692711944e-8,-1.3751070916121869e-7,
                6.6117332425490384e-9*radius_factor,15499133.519658864,.53460147483028242,true};
            const auto frame=recovery::PrimaryFrame::from(parameters);
            const double radius=.76956415773798681,angle=1.719581327546285;
            double full_angular,full_radial,value_angular,value_radial;
            const auto full=recovery::fused_precise_phi(
                radius,angle,frame,&full_angular,&full_radial);
            const auto value=recovery::fused_precise_phi<false>(
                radius,angle,frame,&value_angular,&value_radial);
            if(full.phi!=value.phi || full_angular!=value_angular ||
               full_radial!=value_radial)return 203;
            if(!(recovery::fused_precise_phi(radius,angle,frame).phi>0) ||
               !(recovery::fused_precise_phi(radius,angle+.1,frame).phi<0))return 200;
            if(recovery::seeded_frontier_detail::physical_crossing_count(radius,frame)!=2)return 201;
        }
    }
    {
        namespace boundary=lcbinint::algebraic_boundary::tangent;
        const boundary::LensParams parameters{.0001763733600786604,.00025841859275296266,
            .00030760727910230425,685.48439331981217,.63860976638688738,true};
        const auto frame=boundary::PrimaryFrame::from(parameters);
        const double radius=.36312131494503952;
        boundary::fused_gradient::RadiusOffsetScope offset(2.2741140567883456e-17);
        boundary::PhiValDtheta endpoint;
        const auto polished=boundary::polish_endpoint(radius,3.1276483408048565,
            frame,6,&endpoint,0,2.1e-6,true);
        if(!polished.reliable || std::fabs(polished.theta-3.1276483380138075)>2e-14 ||
           std::fabs(endpoint.phi/endpoint.dphi_dtheta)>2e-15)return 204;
        boundary::CellPlan cell{0,.35,.38,.365,boundary::ArcKind::kArcs,-1,boundary::Status::OK};
        boundary::AdaptiveConfig config;
        config.tol.mu_atol=1e-8;config.tol.mu_rtol=1e-7;
        double value=0;
        for(bool with_jacobian:{false,true}) {
            const auto sample=boundary::adaptive_detail::mapped_radius(radius,1,parameters,
                .6584,frame,cell,with_jacobian,false,nullptr,&config);
            if(!sample.reliable || !(sample.value[0]>0))return 205;
            if(with_jacobian) {
                if(std::fabs(sample.value[0]/value-1)>1e-7)return 206;
                for(double component:sample.value)if(!std::isfinite(component))return 207;
            } else value=sample.value[0];
        }
    }
    {
        namespace boundary=lcbinint::algebraic_boundary::tangent;
        const boundary::LensParams parameters{-6.3359522138426355e-05,1.9809194818738246e-09,
            .00031754130897469404,704.16939274402819,.63244064975758063,true};
        const auto frame=boundary::PrimaryFrame::from(parameters);
        const double radius=1.1813151226673571;
        const auto rounded=boundary::certify_quartic(boundary::boundary_quartic(radius,frame).p);
        if(!rounded.certified || rounded.root_count!=4)return 208;
        if(boundary::seeded_frontier_detail::physical_crossing_count(radius,frame)!=2)return 209;
        const auto arcs=boundary::adaptive_detail::physical_sturm_arcs(radius,frame,2);
        if(arcs.arcs.size()!=1 || std::fabs(arcs.arcs[0][0]-1.0090756938412184)>2e-15 ||
           std::fabs(arcs.arcs[0][1]-1.0090758363986408)>2e-15)return 210;
        if(!(boundary::fused_precise_phi<false>(radius,5.274109545196602,frame).phi<0))return 211;
        boundary::CellPlan cell{0,1.17,1.19,1.18,boundary::ArcKind::kArcs,-1,boundary::Status::OK};
        boundary::AdaptiveConfig config;
        config.tol.mu_atol=1e-4;config.tol.mu_rtol=1e-3;
        for(double limb:{0.,.6584}) {
            const auto value=boundary::adaptive_detail::mapped_radius<true>(radius,1,parameters,
                limb,frame,cell,false,false,nullptr,&config);
            const auto gradient=boundary::adaptive_detail::mapped_radius(radius,1,parameters,
                limb,frame,cell,true,false,nullptr,&config);
            if(!value.reliable || !gradient.reliable || value.local_crossings!=2 ||
               gradient.local_crossings!=2 || !(value.value[0]>0))return 212;
            if(std::fabs(value.value[0]/gradient.value[0]-1)>1e-7)return 213;
            for(double component:gradient.value)if(!std::isfinite(component))return 214;
            for(int component=1;component<6;++component)if(value.value[component]!=0)return 215;
            config.angular_adaptive=true;
            const auto angular=boundary::adaptive_detail::mapped_radius(radius,1,parameters,
                limb,frame,cell,true,false,nullptr,&config);
            config.angular_adaptive=false;
            if(!angular.reliable || angular.local_crossings!=2 ||
               std::fabs(angular.value[0]/gradient.value[0]-1)>1e-7)return 216;
        }
    }
    {
        namespace boundary=lcbinint::algebraic_boundary::tangent;
        boundary::AdaptivePanel panel;
        panel.map={1.,2.,true,true};panel.xl=-1;panel.xr=1;
        boundary::geometry_experiment::Pair lower,upper;
        lower.center=.995;lower.scale=.005;lower.left=.99;lower.right=1.;
        lower.band_lo=.99;lower.band_hi=2.01;lower.real=lower.existing=lower.lower_edge=true;
        upper.center=2.005;upper.scale=.005;upper.left=2.;upper.right=2.01;
        upper.band_lo=.99;upper.band_hi=2.01;upper.real=upper.existing=upper.upper_edge=true;
        const std::vector<boundary::geometry_experiment::Pair> pairs{lower,upper};
        boundary::geometry_experiment::Scope scope(pairs);
        const auto pieces=boundary::fused_gradient::gaussian_panel_pieces(panel);
        if(pieces.size()!=2 || pieces[0].xl!=-1 || pieces[0].xr!=0 ||
           pieces[1].xl!=0 || pieces[1].xr!=1)return 217;
        if(boundary::fused_gradient::gaussian_panel_pair(pieces[0])!=&pairs[0] ||
           boundary::fused_gradient::gaussian_panel_pair(pieces[1])!=&pairs[1])return 218;
        const auto left_pieces=boundary::fused_gradient::gaussian_panel_pieces(pieces[0]);
        if(left_pieces.size()!=1)return 219;
        boundary::fused_gradient::ComTangentScope com_coordinates(true);
        if(boundary::fused_gradient::gaussian_panel_pieces(panel).size()!=1)return 220;
    }
    {
        // Smooth disks that previously used source-plane gradient repair.
        // References are independent five-point scalar differences, with a
        // two-step uncertainty sweep (source_disk_removal_20261028).
        struct Fixture { std::array<double,5> p, gradient; };
        const Fixture fixtures[] = {
            {{0.08402380797794332, 0.08003423785298788, 0.06308767614799801, 1.0814956521878432, 0.9309517049775516}, {20.712871505305465, 2.6597757091906256, 6.713013545241004, 2.312088813243109, -15.59773289983764}},
            {{-0.048110792014146295, -0.0010013921844797685, 0.022276575368024697, 10.529848697410054, 1.0762292921300691}, {68.63638510277899, 0.4111348363385521, 4.3937933968230345, 0.4338608720507941, -6.59650075374903}}
        };
        for (const auto& f : fixtures) for (double tolerance : {1e-3, 1e-4}) {
            lcbinint::magnification::FiniteSourceSettings settings;
            settings.finite_source_tol=0;settings.finite_source_reltol=tolerance;
            lcbinint::magnification::FiniteSourceMagnifier finite(settings);
            lcbinint::magnification::FiniteSourceResult value;
            const auto g=finite.algebraic_boundary_binary_jacobian(
                f.p[4],f.p[3],{f.p[0],f.p[1]},f.p[2],{0,tolerance},&value,nullptr,true);
            if(!g.success || !g.grad_reliable || !(value.value_converged || value.converged) ||
               (value.decision.method==lcbinint::magnification::FiniteSourceMethod::algebraic_boundary &&
                value.diagnostic_stop.find("tangent_chart=radial_frontier")==std::string::npos))return 221;
            for(int j=0;j<5;++j)if(std::abs(g.jacobian[j]-f.gradient[j])>
                1e-4+2e-4*std::abs(f.gradient[j]))return 222;
        }
    }
    return 0;
}
