#include <iostream>
#include "lcbinint/magnification/bennett_integration.hpp"
#include "lcbinint/magnification/nested_disk_area.hpp"
#include "lcbinint/magnification/limb_layer_integral.hpp"
#include "lcbinint/magnification/validated_hermite_boundary.hpp"
#include "lcbinint/magnification/adaptive_image_area.hpp"
#include "lcbinint/lcbinint.h"
#include "lcbinint/magnification/cartesian_run_fill.hpp"
#include "lcbinint/magnification/global_row_residual.hpp"
#include "lcbinint/magnification/adaptive_row_panels.hpp"
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/polar_run_walker.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/probe_diagnostics.hpp"
#include "lcbinint/math/polynomial_roots.hpp"
#include "lcbinint/model/triple_lens_geometry.hpp"
#include "lcbinint/model/orbital_motion.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <cstdlib>
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
using lcbinint::magnification::detail::CartesianRunFillTrace;
using lcbinint::magnification::detail::CartesianRunFillTraceEventKind;
using lcbinint::magnification::detail::bennett_boundary_residual;
using lcbinint::magnification::detail::bennett_boundary_weights;
using lcbinint::magnification::detail::bennett_outer_boundary_weights;
using lcbinint::magnification::detail::fill_cartesian_runs;
using lcbinint::magnification::detail::lift_cartesian_component_run_seeds;
using lcbinint::magnification::detail::PolarRunInterval;
using lcbinint::magnification::detail::walk_polar_outer_segments;
using lcbinint::magnification::detail::walk_cartesian_outer_segments;

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

bool cartesian_run_separate_components_share_row_test()
{
    const LatticeCells expected = mask_cells({"##....##", "##....##"});
    LatticeCells visited;
    const auto result = fill_cartesian_runs<MaskCellState>(
        {{0, 0}, {6, 0}},
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
    // Two separate image components can occupy one row without making that
    // row a multi-run topology event for either component.
    return result.ok() && visited == expected &&
        result.counters.maximum_runs_in_row == 2 &&
        result.counters.rows_with_multiple_runs == 0;
}

bool cartesian_outer_segment_walk_tests()
{
    const auto fill = [](const std::vector<std::string>& rows) {
        const LatticeCells expected = mask_cells(rows);
        return fill_cartesian_runs<MaskCellState>(
            {{static_cast<std::int64_t>(rows.front().size() / 2), 0}},
            [&](std::int64_t ix, std::int64_t iy) {
                return MaskCellState {expected.count({ix, iy}) != 0};
            },
            [](const MaskCellState& state) { return state.inside; },
            [](std::int64_t, std::int64_t, const MaskCellState&) {
                return 1.0;
            },
            [](const auto&, const auto&, const auto&, const auto&, const auto&) {
                return CartesianBoundaryContribution {};
            },
            CartesianRunFillLimits {100000, 10000});
    };
    const auto check_partition = [](const auto& result, bool require_junction) {
        if (!result.ok() || result.runs.empty()) {
            return false;
        }
        const std::size_t component = result.runs.front().component;
        const auto walk = walk_cartesian_outer_segments(result.runs, component);
        if (!walk.complete || walk.segments.empty() ||
            (require_junction && walk.junction_nodes == 0)) {
            return false;
        }
        std::set<std::size_t> seen;
        std::size_t component_runs = 0;
        for (const auto& record : result.runs) {
            component_runs += record.component == component ? 1U : 0U;
        }
        for (const auto& segment : walk.segments) {
            if (segment.run_indices.empty()) {
                return false;
            }
            for (std::size_t position = 0;
                 position < segment.run_indices.size(); ++position) {
                const std::size_t run_index = segment.run_indices[position];
                if (!seen.insert(run_index).second ||
                    result.runs[run_index].component != component) {
                    return false;
                }
                if (position > 0) {
                    const auto& previous =
                        result.runs[segment.run_indices[position - 1]].run;
                    const auto& current = result.runs[run_index].run;
                    if (current.iy != previous.iy + 1) {
                        return false;
                    }
                }
            }
        }
        return seen.size() == component_runs;
    };

    const auto clean = fill({
        "###",
        "###",
        "###",
        "###",
        "###",
    });
    if (!check_partition(clean, false) ||
        walk_cartesian_outer_segments(clean.runs, clean.runs.front().component)
                .segments.size() != 1U) {
        return false;
    }

    // The upper run fans out into two lower runs.  This is the raster analogue
    // of the strip that Dave's check_bktrack rescans after a boundary turns
    // around; the walker must retain all three chains without duplicating the
    // junction run.
    const auto banana = fill({
        "..#####..",
        ".##...##.",
        "##.....##",
        "##.....##",
    });
    if (!check_partition(banana, true) ||
        walk_cartesian_outer_segments(
            banana.runs, banana.runs.front().component).segments.size() < 3U) {
        return false;
    }

    const auto pinched = fill({
        ".#####.",
        ".##.##.",
        ".#####.",
    });
    return check_partition(pinched, true);
}

bool polar_outer_segment_walk_tests()
{
    const auto check_partition = [](
        const std::vector<PolarRunInterval>& runs,
        int phi_bins,
        bool require_junction,
        bool require_closed) {
        const auto walk = walk_polar_outer_segments(runs, phi_bins);
        if (!walk.complete || walk.segments.empty() ||
            (require_junction && walk.junction_nodes == 0)) {
            return false;
        }
        std::set<std::size_t> seen;
        const auto find_run = [&](std::size_t run_index)
            -> const PolarRunInterval* {
            const auto found = std::find_if(
                runs.begin(), runs.end(),
                [run_index](const PolarRunInterval& run) {
                    return run.run_index == run_index;
                });
            return found == runs.end() ? nullptr : &*found;
        };
        for (const auto& segment : walk.segments) {
            if (segment.run_indices.empty()) {
                return false;
            }
            if (segment.closed &&
                (segment.lower_boundary || segment.upper_boundary ||
                 segment.lower_junction || segment.upper_junction)) {
                return false;
            }
            for (std::size_t position = 0;
                 position < segment.run_indices.size(); ++position) {
                const std::size_t run_index = segment.run_indices[position];
                if (!seen.insert(run_index).second) {
                    return false;
                }
                const auto* current = find_run(run_index);
                if (current == nullptr) {
                    return false;
                }
                if (position > 0) {
                    const auto* previous =
                        find_run(segment.run_indices[position - 1]);
                    if (previous == nullptr ||
                        current->iphi !=
                            (previous->iphi + 1) % phi_bins ||
                        static_cast<std::int64_t>(previous->left) >
                            static_cast<std::int64_t>(current->right) + 1 ||
                        static_cast<std::int64_t>(current->left) >
                            static_cast<std::int64_t>(previous->right) + 1) {
                        return false;
                    }
                }
            }
        }
        if (seen.size() != runs.size()) {
            return false;
        }
        if (require_closed) {
            return walk.segments.size() == 1U &&
                walk.segments.front().closed;
        }
        return true;
    };

    const std::vector<PolarRunInterval> clean {
        {0, 2, 4, 0}, {1, 2, 4, 1}, {2, 2, 4, 2},
        {3, 2, 4, 3}, {4, 2, 4, 4}, {5, 2, 4, 5},
    };
    const auto clean_walk = walk_polar_outer_segments(clean, 12);
    if (!check_partition(clean, 12, false, false) ||
        clean_walk.segments.size() != 1U ||
        !clean_walk.segments.front().lower_boundary ||
        !clean_walk.segments.front().upper_boundary) {
        return false;
    }

    const std::vector<PolarRunInterval> wrapped {
        {10, 2, 4, 0}, {11, 2, 4, 1}, {0, 2, 4, 2}, {1, 2, 4, 3},
    };
    const auto wrapped_walk = walk_polar_outer_segments(wrapped, 12);
    if (!check_partition(wrapped, 12, false, false) ||
        wrapped_walk.segments.size() != 1U ||
        !wrapped_walk.segments.front().lower_boundary ||
        !wrapped_walk.segments.front().upper_boundary) {
        return false;
    }

    // A radial interval splitting into two runs is the polar graph form of a
    // backtrack.  The junction run must not be duplicated across the chains.
    const std::vector<PolarRunInterval> split {
        {0, 2, 5, 0}, {1, 2, 3, 1}, {1, 5, 6, 2},
        {2, 2, 3, 3}, {2, 5, 6, 4},
    };
    if (!check_partition(split, 12, true, false)) {
        return false;
    }

    const std::vector<PolarRunInterval> ring {
        {0, 2, 4, 0}, {1, 2, 4, 1}, {2, 2, 4, 2},
        {3, 2, 4, 3}, {4, 2, 4, 4}, {5, 2, 4, 5},
    };
    return check_partition(ring, 6, false, true);
}

bool cartesian_run_trace_order_test()
{
    const LatticeCells expected = mask_cells({
        "##....##",
        "##....##",
    });
    LatticeCells visited;
    CartesianRunFillTrace trace;
    const auto result = fill_cartesian_runs<MaskCellState>(
        {{6, 0}, {0, 0}},
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
        CartesianRunFillLimits {100000, 10000},
        trace);
    if (!result.ok() || visited != expected) {
        return false;
    }

    std::size_t discovered = 0;
    std::size_t popped = 0;
    std::size_t first_popped_event = trace.events.size();
    for (std::size_t event_index = 0;
         event_index < trace.events.size(); ++event_index) {
        const auto& event = trace.events[event_index];
        if (event.kind == CartesianRunFillTraceEventKind::run_discovered) {
            if (event.run_index != discovered || event.fill_level != 0) {
                return false;
            }
            ++discovered;
            continue;
        }
        if (event.kind == CartesianRunFillTraceEventKind::frontier_popped) {
            if (first_popped_event == trace.events.size()) {
                first_popped_event = event_index;
            }
            if (event.run_index != popped || event.fill_level != 0) {
                return false;
            }
            ++popped;
            continue;
        }
        if (event.kind == CartesianRunFillTraceEventKind::components_merged) {
            if (event.fill_level != 0) {
                return false;
            }
            continue;
        }
        return false;
    }
    // Default scheduling offers both seeds before popping; the archived
    // explicit sequential mode drains the first seed immediately.
    const std::size_t expected_first_pop = std::getenv("LCBININT_SEQUENTIAL_SEED_FILL") ? 1 : 2;
    return first_popped_event == expected_first_pop && discovered == result.runs.size() &&
        popped == static_cast<std::size_t>(result.counters.frontier_intervals_popped);
}

bool cartesian_run_trace_cap_test()
{
    const LatticeCells expected = mask_cells({
        "##....##",
        "##....##",
    });
    CartesianRunFillTrace trace;
    trace.maximum_events = 1;
    const auto result = fill_cartesian_runs<MaskCellState>(
        {{0, 0}, {6, 0}},
        [&](std::int64_t ix, std::int64_t iy) {
            return MaskCellState {expected.count({ix, iy}) != 0};
        },
        [](const MaskCellState& state) { return state.inside; },
        [](std::int64_t, std::int64_t, const MaskCellState&) { return 1.0; },
        [](const auto&, const auto&, const auto&, const auto&, const auto&) {
            return CartesianBoundaryContribution {};
        },
        CartesianRunFillLimits {100000, 10000},
        trace);
    return result.ok() && trace.truncated && trace.events.size() == 1;
}

bool binary_cartesian_trace_route_test()
{
    lcbinint::magnification::FiniteSourceSettings settings;
    settings.source_bins = 12;
    settings.caustic_bins = 64;
    settings.finite_mode = 1;
    settings.automatic_source_bins = false;
    const lcbinint::magnification::FiniteSourceMagnifier magnifier(settings);
    const auto trace = magnifier.binary_cartesian_trace(
        1.0, 0.1, {-0.075, 0.09}, 0.04, 12);
    if (!trace.valid() || trace.source_bins != 12 ||
        !(trace.lattice_spacing > 0.0) || !trace.complete ||
        trace.events.empty()) {
        return false;
    }
    bool found_discovery = false;
    bool found_frontier = false;
    for (const auto& event : trace.events) {
        found_discovery = found_discovery ||
            event.kind == CartesianRunFillTraceEventKind::run_discovered;
        found_frontier = found_frontier ||
            event.kind == CartesianRunFillTraceEventKind::frontier_popped;
    }
    if (!found_discovery || !found_frontier) {
        return false;
    }

    // The trace-enabled route must be observational: retaining the event
    // order cannot change the numerical area returned by the ordinary API.
    const auto point_source = lcbinint::magnification::PointSourceMagnifier{};
    const double point_magnification = std::abs(
        point_source.binary_mag0(1.0, 0.1, {-0.075, 0.09}).magnification);
    const auto ordinary = magnifier.binary_mag(
        1.0, 0.1, {-0.075, 0.09}, 0.04, point_magnification);
    const double scale = std::max(1.0, std::abs(ordinary.magnification));
    return std::isfinite(ordinary.magnification) &&
        std::abs(trace.magnification - ordinary.magnification) <= 1e-12 * scale;
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

bool bennett_integration_rule_tests()
{
    // Equation (14): the guarded branch keeps the zeroth-order moment
    // exact, while replacing the unstable square-root coefficient near a
    // grid node.
    const auto guarded = bennett_boundary_weights(0.05, 0.15);
    if (std::abs(guarded.limb - 0.05 / 3.0) > 1.0e-15 ||
        std::abs(guarded.node - (0.5 + 2.0 * 0.05 / 3.0)) > 1.0e-15 ||
        std::abs(guarded.limb + guarded.node - (0.5 + 0.05)) > 1.0e-15) {
        return false;
    }

    const double delta = 0.4;
    const auto unguarded = bennett_boundary_weights(delta, 0.15);
    const double b = (2.0 / 3.0) * std::sqrt((delta + 0.5) / delta);
    if (std::abs(unguarded.limb - (delta + 0.5) * (1.0 - b)) > 1.0e-15 ||
        std::abs(unguarded.node - (delta + 0.5) * b) > 1.0e-15 ||
        std::abs(unguarded.limb + unguarded.node - (delta + 0.5)) > 1.0e-15) {
        return false;
    }

    // The residual form used by the grid walkers must reduce to the usual
    // (delta - 1/2) strip correction for a uniform source.
    for (const double d : {0.0, 0.03, 0.15, 0.7, 1.0}) {
        if (std::abs(bennett_boundary_residual(d, 1.0, 1.0, 0.15) -
                     (d - 0.5)) > 1.0e-15) {
            return false;
        }
    }

    // Equation (15), retained as a separately testable building block for
    // the smooth p=0 endpoint limit.
    const auto outer = bennett_outer_boundary_weights(0.25);
    if (std::abs(outer.first - (3.0 / 8.0 + 0.25 + 0.5 * 0.25 * 0.25)) > 1.0e-15 ||
        std::abs(outer.second - (9.0 / 8.0 - 0.5 * 0.25 * 0.25)) > 1.0e-15) {
        return false;
    }

    return true;
}

} // namespace

bool limb_layer_tests() {
 using namespace lcbinint::magnification::detail;
 const auto disk=[](double r,double){return std::array<double,2>{2*3.141592653589793*r*r,0};};
 LimbLayerIntegral uniform(.1,0,0,disk);
 if(!uniform.refine(1e-10,1)||uniform.calls!=1||std::abs(uniform.flux()/uniform.source_flux()-2)>1e-12)return false;
 LimbLayerIntegral ld(.1,.3,.2,disk);
 if(ld.refine(1e-12,4)||!ld.refine(1e-12,1000)||std::abs(ld.flux()/ld.source_flux()-2)>1e-10)return false;
 const auto calls=ld.calls;if(!ld.refine(1e-12,1000)||calls!=ld.calls)return false;
 return true;
}

bool validated_boundary_tests() {
 using namespace lcbinint::magnification::detail;
 for(bool outer:{false,true}) {
  const double rho=.2,r=(std::sqrt(4+rho*rho)+(outer?rho:-rho))/2;
  const double ya=.1,yb=.11,a=std::sqrt(r*r-yb*yb),b=std::sqrt(r*r-ya*ya),pad=b-a;
  const auto v=validated_hermite_boundary(0,1,{0,0},rho,a-pad,b+pad,ya,yb);
  const auto integral=[&](double y){return .5*(y*std::sqrt(r*r-y*y)+r*r*std::asin(y/r));};
  const double exact=integral(yb)-integral(ya);
  const double fourth=3*r*r*(r*r+4*yb*yb)/std::pow(r*r-yb*yb,3.5);
  if(!v.resolved||exact<v.area_lower||exact>v.area_upper||v.fourth_derivative_upper<fourth)return false;
 }
 if(validated_hermite_boundary(0,1,{0,0},.2,-.1,.1,-.01,.01).resolved)return false;
 if(validated_hermite_boundary(0,-1,{0,0},.2,1,2,0,.1).resolved)return false;
 return true;
}

bool adaptive_complete_area_tests() {
 using namespace lcbinint::magnification::detail;
 const auto disk_rows=[](double x,double y,double outer,double inner,double h) {
  std::vector<CartesianRunBoundaryTraceEvent> rows;
  for(int iy=static_cast<int>(std::ceil((y-outer)/h));iy<=static_cast<int>(std::floor((y+outer)/h));++iy) {
   const double dy=iy*h-y;if(std::abs(dy)>=outer)continue;
   const double a=std::sqrt(outer*outer-dy*dy),b=std::abs(dy)<inner?std::sqrt(inner*inner-dy*dy):0;
   const auto add=[&](double lo,double hi) {
    CartesianRunBoundaryTraceEvent e;e.run={iy,static_cast<std::int64_t>(std::ceil(lo/h)),static_cast<std::int64_t>(std::floor(hi/h))};
    if(e.run.lo>e.run.hi)return;
    e.cell_spacing=h;e.diagnostic.left_fraction=e.run.lo-lo/h;e.diagnostic.right_fraction=hi/h-e.run.hi;
    rows.push_back(e);
   };
   if(b>0){add(x-a,x-b);add(x+b,x+a);}else add(x-a,x+a);
  }
  return rows;
 };
 const BoundaryTurnGeometry circle{2,0,0,0,.01,.37,.19};
 LensAdaptiveArea disk(circle,disk_rows(.37,.19,.1,0,.035));
 if(!disk.certifies_interior_chord(.37,.19,.39,.20)||disk.certifies_interior_chord(.37,.19,.5,.2))return false;
 NestedDiskArea nested(circle,disk_rows(.37,.19,.1,0,.035),{{.37,.19}});
 for(double r:{.1,.03,.001}){const auto v=nested.area(r,1e-9);if(!std::isfinite(v[0])||std::abs(v[0]-3.141592653589793*r*r)>1e-9)return false;const auto maps=nested.lattice_map_evaluations;const auto again=nested.area(r,1e-9);if(again[0]!=v[0]||maps!=nested.lattice_map_evaluations)return false;}
 if(disk.unresolved_topology||!disk.panels->refine(1e-10,20000)||std::abs(disk.area()-3.141592653589793*.01)>1e-9)return false;
 const double rho=.2,outer=(std::sqrt(rho*rho+4)+rho)/2,inner=1/outer;
 const BoundaryTurnGeometry single{1,1,0,1,rho*rho,0,0};
 LensAdaptiveArea annulus(single,disk_rows(1,0,outer,inner,.06));
 if(annulus.unresolved_topology||!annulus.panels->refine(1e-9,20000)||std::abs(annulus.area()-3.141592653589793*rho*std::sqrt(rho*rho+4))>1e-8)return false;
 auto multilevel=disk_rows(.37,.19,.1,0,.035);multilevel[0].fill_level=1;
 LensAdaptiveArea rejected(circle,multilevel);if(!rejected.unresolved_topology)return false;
 const auto before=annulus.map_evaluations;
 if(!annulus.panels->refine(1e-9,20000)||annulus.map_evaluations!=before)return false;
 return true;
}

bool incremental_adaptive_panel_tests() {
 using namespace lcbinint::magnification::detail;
 std::size_t calls=0;
 const auto evaluate=[&](double x){++calls;return AdaptiveRowNode{x,std::exp(x),std::exp(x),true};};
 const AdaptiveRowNode a{0,1,1,true},b{1,std::exp(1.),std::exp(1.),true};
 AdaptiveRowPanels state({{a,b,evaluate}});
 if(calls!=1)return false;
 if(!state.refine(1e-6,1000)||std::abs(state.area()-std::expm1(1.))>2e-6)return false;
 const auto before=calls;const auto stored=state.retained_nodes();
 if(!state.refine(1e-10,10000)||std::abs(state.area()-std::expm1(1.))>2e-10)return false;
 if(calls<=before||state.retained_nodes()<=stored||calls!=state.new_evaluations)return false;
 const auto ready=calls;
 if(!state.refine(1e-10,10000)||calls!=ready)return false;
 AdaptiveRowPanels budget({{a,b,evaluate}});
 if(budget.refine(1e-15,1)||budget.new_evaluations!=1)return false;
 if(!budget.refine(1e-10,10000))return false;
 const auto m=AdaptiveRowNode{.5,std::exp(.5),std::exp(.5),true};
 AdaptiveRowPanels retained({{a,b,evaluate,m}});
 if(retained.new_evaluations!=0)return false;
 const auto quartic=[](double t){return AdaptiveRowNode{t,std::pow(t,4),4*std::pow(t,3),true};};
 AdaptiveRowPanels fourth({{quartic(-.5),quartic(.5),quartic}});
 if(std::abs(fourth.error()-std::abs(fourth.area()-.0125))>1e-14)return false;
 if(!retained.refine(1e-10,10000))return false;
 return true;
}

bool global_row_residual_regression_tests() {
    using namespace lcbinint::magnification::detail;
    if(std::abs(endpoint_hurwitz(-.5,1.)+0.207886224977354566)>2e-13)return false;
    if(std::abs(endpoint_hurwitz(-1.5,1.)+0.025485201889833035)>2e-12)return false;
    std::vector<CartesianRunBoundaryTraceEvent> gaps(2);
    gaps[0].run={0,-2,1};gaps[1].run={1,2,10};
    gaps[0].diagnostic.right_fraction=-.69;gaps[1].diagnostic.left_fraction=-.86;
    if(RowResidualGraph(gaps).after[0].empty())return false;
    if(!RowResidualGraph(gaps,true).after[0].empty())return false;
    const double q=1e-5,m1=1/(1+q),rho=.006045479312123226;
    const BoundaryTurnGeometry m{1.1,m1,q*m1,1.1*m1,rho*rho,-.21052980010453926,.0004534109484092418};
    std::vector<CartesianRunBoundaryTraceEvent> runs(1);
    runs[0].run={0,-38,-22};runs[0].cell_spacing=rho/16;
    GlobalRowResidual repaired;
    add_missing_centre_image(m,.0004872569763104967,-1.0814662561661846e-5,runs,repaired);
    if(repaired.local_images!=1||repaired.unresolved_images!=0 ||
       std::abs(repaired.missing_image-7.278115491892072e-8)>1e-16)return false;
    runs[0].run={0,1,1};GlobalRowResidual represented;
    add_missing_centre_image(m,.0004872569763104967,-1.0814662561661846e-5,runs,represented);
    if(represented.local_images!=0||represented.map_evaluations!=0||represented.total()!=0)return false;
    runs[0].run={0,-38,-22};runs[0].cell_spacing=rho/256;GlobalRowResidual rejected;
    add_missing_centre_image(m,.0004872569763104967,-1.0814662561661846e-5,runs,rejected);
    if(rejected.local_images!=0||rejected.unresolved_images!=1)return false;
    return true;
}

bool shared_endpoint_moment_tests() {
    using namespace lcbinint::magnification::detail;
    for(double delta:{0.,.0003,.03,.14999,.15,.5,.9,1.})for(int degree:{0,1,2}) {
        const auto weights=bennett_outer_boundary_weights(delta);
        constexpr int count=256;const long double end=static_cast<long double>(delta)+count-.5L,alpha=.5L+degree;
        const auto power=[degree](long double t){long double v=std::sqrt(t);for(int j=0;j<degree;++j)v*=t;return v;};
        long double exact=end*power(end)/(alpha+1);
        for(int k=0;k<count;++k)exact-=power(static_cast<long double>(delta)+k);
        exact-=(weights.first-1)*power(delta)+(weights.second-1)*power(delta+1);
        exact-=alpha*power(end)/end/24;
        exact+=7*alpha*(alpha-1)*(alpha-2)*power(end)/(end*end*end)/5760;
        const auto got=power_endpoint_quadrature_defect(delta,weights.first,weights.second,degree);
        if(!std::isfinite(got)||std::abs(got-exact)>1e-8){ std::cerr << "shared error regression at line " << __LINE__ << "\n"; return false; }
    }
    for(double delta:{0.,.03,.2,.5,.9,1.})for(int power:{1,2}) {
        const auto weights=bennett_outer_boundary_weights(delta);
        const long double d=delta,end=d+64-.5L;
        long double exact=std::pow(end,power+1)/(power+1);
        for(int k=0;k<64;++k)exact-=std::pow(d+k,power);
        exact-=power*std::pow(end,power-1)/24;
        exact-=(weights.first-1)*std::pow(d,power)+(weights.second-1)*std::pow(d+1,power);
        if(std::abs(integer_endpoint_quadrature_defect(delta,weights.first,weights.second,power)-exact)>1e-10){ std::cerr << "shared error regression at line " << __LINE__ << "\n"; return false; }
    }
    for(int bins:{64,256,20000}) {
        long double profile=.5L;
        for(int k=1;k<bins;++k)profile+=std::sqrt(static_cast<long double>(k)/bins);
        const long double exact=2.L/3-profile/bins;
        if(std::abs(linear_ld_table_bias_moment(bins)-exact)>1e-10){ std::cerr << "shared error regression at line " << __LINE__ << "\n"; return false; }
    }
    for(double delta:{.0003,.03,.2,.5,.9,1.}) {
        const long double d=delta,end=d+256-.5L;
        long double exact=2*std::sqrt(end);
        for(int k=0;k<256;++k)exact-=1/std::sqrt(d+k);
        exact+=1/(48*end*std::sqrt(end));
        exact-=7*1.875L/(5760*end*end*end*std::sqrt(end));
        if(std::abs(inverse_sqrt_endpoint_quadrature_defect(delta)-exact)>1e-9){ std::cerr << "shared error regression at line " << __LINE__ << "\n"; return false; }
        const auto weights=bennett_boundary_weights(delta,.15);
        const auto depth=[](double t){const double mu=.2*std::sqrt(t)+.01*t*std::sqrt(t);return mu*mu;};
        for(double c:{0.,.2,.5,.8,1.}) {
            const auto fast=bennett_linear_ld_endpoint_flux_residual(delta,depth(delta),depth(delta+1),c,weights);
            const auto regular=bennett_endpoint_flux_residual(delta,1-c+c*std::sqrt(depth(delta)),1-c+c*std::sqrt(depth(delta+1)),1-c,.15);
            if(!fast.resolved||!regular.resolved||std::abs(fast.signed_residual-regular.signed_residual)>1e-12||std::abs(fast.model_remainder-regular.model_remainder)>1e-12){ std::cerr << "shared error regression at line " << __LINE__ << "\n"; return false; }
        }
    }
    // An enclosed parabolic hole. The occupied runs have no new component
    // endpoints: only the gap's two endpoints contribute to this model.
    CartesianRunFillResult hole;hole.component_roots={0};
    for(int y=0;y<40;++y) {
        const double t=y-8+.3,width=(t>0&&t<20.6)?std::sqrt(t*(20.6-t)):0;
        const auto add=[&](int lo,int hi,double left,double right) {
            hole.runs.push_back({{y,lo,hi},0});hole.run_areas.push_back(hi-lo+left+right);
            hole.run_error_fractions.push_back({left,right});
        };
        if(width==0)add(-50,50,0,0);
        else {
            const int edge=int(std::ceil(width/2));const double fraction=edge-width/2;
            add(-50,-edge,0,fraction);add(edge,50,fraction,0);
        }
    }
    const auto uniform_gap=same_grid_internal_error(hole,0,1,nullptr);
    const auto ld_gap=same_grid_internal_error(hole,0,.8,nullptr);
    long double sampled=0;
    for(int y=8;y<=28;++y){const long double t=y-8+.3L;sampled+=std::sqrt(t*(20.6L-t));}
    const long double exact=sampled-3.141592653589793238L*20.6L*20.6L/8;
    if(!uniform_gap.resolved||std::abs(uniform_gap.signed_residual-exact)>1e-3||
       std::abs(ld_gap.signed_residual-.8*uniform_gap.signed_residual)>1e-12){ std::cerr << "shared error regression at line " << __LINE__ << "\n"; return false; }
    double modeled_junction=0;
    for(const auto& term:uniform_gap.modeled_second)if(term.first==8)modeled_junction+=term.second;
    const auto width=[](int y){const double t=y-8+.3;return t>0&&t<20.6?std::sqrt(t*(20.6-t)):0.;};
    const double actual_junction=-width(7)+2*width(8)-width(9);
    if(std::abs(actual_junction-modeled_junction)>.02*std::abs(actual_junction)){ std::cerr << "shared error regression at line " << __LINE__ << "\n"; return false; }
    for(double delta:{.03,.2,.5,.9}) {
        const auto model=bennett_endpoint_flux_residual(delta,1+2*std::sqrt(delta),1+2*std::sqrt(delta+1),1,.15);
        const auto weights=bennett_boundary_weights(delta,.15);
        if(!model.resolved||std::abs(model.signed_residual-2*sqrt_endpoint_quadrature_defect(delta,weights.node))>1e-12){ std::cerr << "shared error regression at line " << __LINE__ << "\n"; return false; }
        const auto uniform=bennett_endpoint_flux_residual(delta,1,1,1,.15);
        if(!uniform.resolved||uniform.signed_residual!=0){ std::cerr << "shared error regression at line " << __LINE__ << "\n"; return false; }
        for(double curvature:{-.05,.1}) {
            const auto f=[curvature](double t){return 1+2*std::sqrt(t)+curvature*t*std::sqrt(t);};
            const auto higher=bennett_endpoint_flux_residual(delta,f(delta),f(delta+1),1,.15);
            const double expected=2*power_endpoint_quadrature_defect(delta,weights.node,1,0)+
                curvature*power_endpoint_quadrature_defect(delta,weights.node,1,1);
            if(!higher.resolved||std::abs(higher.three_half_coefficient-curvature)>1e-12||
                std::abs(higher.signed_residual-expected)>1e-12){ std::cerr << "shared error regression at line " << __LINE__ << "\n"; return false; }
        }
    }
    return true;
}

int main()
{
    if(!shared_endpoint_moment_tests())return 78;
    if(!global_row_residual_regression_tests())return 73;
    if(!incremental_adaptive_panel_tests())return 74;
    if(!adaptive_complete_area_tests())return 75;
    if(!validated_boundary_tests())return 76;
    if(!limb_layer_tests())return 77;
    if (!cartesian_run_topology_tests()) {
        return 62;
    }
    if (!cartesian_run_separate_components_share_row_test()) {
        return 66;
    }
    if (!cartesian_outer_segment_walk_tests()) {
        return 71;
    }
    if (!polar_outer_segment_walk_tests()) {
        return 72;
    }
    if (!cartesian_run_trace_order_test()) {
        return 67;
    }
    if (!cartesian_run_trace_cap_test()) {
        return 69;
    }
    if (!binary_cartesian_trace_route_test()) {
        return 68;
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
    if (!bennett_integration_rule_tests()) {
        return 70;
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
        std::abs(options.finite_source_reltol) > 1e-12 ||
        std::abs(options.bennett_delta_c - 0.15) > 1e-12) {
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
