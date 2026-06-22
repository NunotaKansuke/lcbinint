#include "lcbinint/magnification/finite_source_magnifier.hpp"

#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/math/polynomial_roots.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace lcbinint::magnification {
namespace {

constexpr double kSqrtHalf = 0.70710678118654752440;
constexpr double kPi = 3.14159265358979323846;
constexpr int kHexadecapoleEvaluations = 13;
constexpr int kLimbDarkeningTableSize = 5000;
constexpr int kLegacyIndexOffset = 2000000;

struct BinaryLensMapper {
    Complex separation;
    double m1 = 0.0;
    double m2 = 0.0;
};

BinaryLensMapper make_binary_lens_mapper(double separation, double mass_ratio)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const Complex lens_separation = q_input < 1.0 ? Complex(-s, 0.0) : Complex(s, 0.0);
    const double m1 = 1.0 / (1.0 + q);
    const double m2 = q * m1;
    return {lens_separation, m1, m2};
}

double mapped_binary_lens_distance2(
    const BinaryLensMapper& mapper,
    double x,
    double y,
    SourcePosition source)
{
    const double a = mapper.separation.real();
    const double xa = x - a;
    const double den1 = xa * xa + y * y;
    const double den2 = x * x + y * y;
    const double mapped_x = x - mapper.m1 * xa / den1 - mapper.m2 * x / den2 - a * mapper.m1;
    const double mapped_y = y - mapper.m1 * y / den1 - mapper.m2 * y / den2;
    const double dx = mapped_x - source.x;
    const double dy = mapped_y - source.y;
    return dx * dx + dy * dy;
}

// Jacobian determinant J = 1 - |m1/(z-a)^2 + m2/z^2|^2 at image position (x,y).
// J > 0: standard-parity image; J < 0: flipped-parity image.
// Returns 0.0 when image is too close to a lens (degenerate).
double binary_jacobian(const BinaryLensMapper& mapper, double x, double y)
{
    const double a = mapper.separation.real();
    const double xa = x - a;
    const double den1 = xa * xa + y * y;
    const double den2 = x * x + y * y;
    if (den1 < 1.0e-20 || den2 < 1.0e-20) return 0.0;
    const double den1sq = den1 * den1;
    const double den2sq = den2 * den2;
    const double re_f = mapper.m1 * (xa * xa - y * y) / den1sq
                      + mapper.m2 * (x * x - y * y) / den2sq;
    const double im_f = -2.0 * y * (mapper.m1 * xa / den1sq + mapper.m2 * x / den2sq);
    return 1.0 - re_f * re_f - im_f * im_f;
}

// Returns the sign of the binary lens Jacobian: +1, -1, or 0 (degenerate).
int binary_jacobian_sign(const BinaryLensMapper& mapper, double x, double y)
{
    const double J = binary_jacobian(mapper, x, y);
    return J > 0.0 ? 1 : J < 0.0 ? -1 : 0;
}

SourcePosition map_binary_lens_real(
    const BinaryLensMapper& mapper,
    double x,
    double y)
{
    const double a = mapper.separation.real();
    const double xa = x - a;
    const double den1 = xa * xa + y * y;
    const double den2 = x * x + y * y;
    return {
        x - mapper.m1 * xa / den1 - mapper.m2 * x / den2 - a * mapper.m1,
        y - mapper.m1 * y / den1 - mapper.m2 * y / den2,
    };
}

double source_distance(SourcePosition source)
{
    return std::hypot(source.x, source.y);
}

int estimate_cartesian_cost(const FiniteSourceSettings& settings)
{
    const int bins = settings.source_bins > 0 ? settings.source_bins : 1;
    return bins * bins * 16;
}

int estimate_polar_cost(const FiniteSourceSettings& settings)
{
    const int radial_bins = settings.source_bins > 0 ? settings.source_bins : 1;
    const int angular_bins = static_cast<int>(std::ceil(2.0 * M_PI * radial_bins / settings.grid_ratio));
    return radial_bins * angular_bins * 8;
}

double distance_squared(SourcePosition lhs, SourcePosition rhs)
{
    const double dx = lhs.x - rhs.x;
    const double dy = lhs.y - rhs.y;
    return dx * dx + dy * dy;
}

double point_segment_distance(SourcePosition point, SourcePosition start, SourcePosition end)
{
    const double dx = end.x - start.x;
    const double dy = end.y - start.y;
    const double length2 = dx * dx + dy * dy;
    if (length2 == 0.0) {
        return std::sqrt(distance_squared(point, start));
    }

    const double t = std::clamp(
        ((point.x - start.x) * dx + (point.y - start.y) * dy) / length2, 0.0, 1.0);
    return std::hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy));
}

double limb_darkening_flux_factor(const FiniteSourceSettings& settings)
{
    return 1.0 - settings.limb_darkening_c / 3.0 - settings.limb_darkening_d / 5.0;
}

double limb_darkening_gamma(const FiniteSourceSettings& settings)
{
    const double denominator =
        15.0 - 5.0 * settings.limb_darkening_c - 3.0 * settings.limb_darkening_d;
    if (denominator == 0.0) {
        return 0.0;
    }
    return 10.0 * settings.limb_darkening_c / denominator;
}

double limb_darkening_lambda(const FiniteSourceSettings& settings)
{
    const double denominator =
        15.0 - 5.0 * settings.limb_darkening_c - 3.0 * settings.limb_darkening_d;
    if (denominator == 0.0) {
        return 0.0;
    }
    return 12.0 * settings.limb_darkening_d / denominator;
}

double source_surface_brightness(double normalized_radius2, const FiniteSourceSettings& settings)
{
    const double bounded_radius2 = std::clamp(normalized_radius2, 0.0, 1.0);
    const double mu = std::sqrt(std::max(0.0, 1.0 - bounded_radius2));
    return 1.0 - settings.limb_darkening_c * (1.0 - mu) -
           settings.limb_darkening_d * (1.0 - std::sqrt(mu));
}

double source_flux(double source_radius, const FiniteSourceSettings& settings)
{
    const double flux_factor = limb_darkening_flux_factor(settings);
    if (flux_factor <= 0.0 || !std::isfinite(flux_factor)) {
        return std::nan("");
    }
    return kPi * source_radius * source_radius * flux_factor;
}

std::vector<Complex> critical_curve_polynomial_coefficients(double separation, double mass_ratio, Complex phase)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const Complex a = q_input < 1.0 ? Complex(-s, 0.0) : Complex(s, 0.0);
    const Complex m1 = 1.0 / (1.0 + q);
    const Complex m2 = q * m1;
    const Complex a2 = a * a;

    return {
        m2 * a2,
        -2.0 * m2 * a,
        m1 + m2 - phase * a2,
        2.0 * phase * a,
        -phase,
    };
}

std::vector<SourcePosition> caustic_points_at_phase(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    double phase_angle)
{
    const Complex phase = std::polar(1.0, phase_angle);
    math::PolynomialRootSolver solver;
    const auto roots = solver.solve(critical_curve_polynomial_coefficients(separation, mass_ratio, phase));
    if (roots.status != math::RootSolverStatus::ok) {
        return {};
    }

    std::vector<SourcePosition> points;
    points.reserve(roots.roots.size());
    for (const auto& root : roots.roots) {
        points.push_back(point_magnifier.binary_lens_equation(
            separation, mass_ratio, {root.real(), root.imag()}));
    }
    return points;
}

void append_tracked_caustic_points(
    std::vector<std::vector<SourcePosition>>& branches,
    std::vector<SourcePosition> points)
{
    const std::size_t n = branches.size();
    if (points.size() != n) {
        return;
    }

    if (branches[0].empty()) {
        std::sort(points.begin(), points.end(), [](const auto& lhs, const auto& rhs) {
            return std::atan2(lhs.y, lhs.x) < std::atan2(rhs.y, rhs.x);
        });
        for (std::size_t i = 0; i < n; ++i) {
            branches[i].push_back(points[i]);
        }
        return;
    }

    // Find the permutation of `points` that minimises total squared step length.
    // Greedy nearest-neighbour can swap inner/outer caustic branches when they
    // come close; the global optimum never makes a swap unless the two assignments
    // have identical total cost (branches genuinely coincide), avoiding spurious
    // long segments in the branch grid.  For n=4 this is 4!=24 permutations.
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<std::size_t> best_perm = perm;
    double best_cost = std::numeric_limits<double>::infinity();
    do {
        double cost = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            cost += distance_squared(branches[i].back(), points[perm[i]]);
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_perm = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    for (std::size_t i = 0; i < n; ++i) {
        branches[i].push_back(points[best_perm[i]]);
    }
}

double binary_caustic_distance(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    int caustic_bins)
{
    const int bins = std::max(caustic_bins, 32);
    std::vector<std::vector<SourcePosition>> branches(4);
    for (int i = 0; i < bins; ++i) {
        const double phase_angle = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(bins);
        append_tracked_caustic_points(
            branches, caustic_points_at_phase(point_magnifier, separation, mass_ratio, phase_angle));
    }

    double distance = std::numeric_limits<double>::infinity();
    for (const auto& branch : branches) {
        if (branch.size() < 2) {
            continue;
        }
        for (std::size_t i = 1; i < branch.size(); ++i) {
            distance = std::min(distance, point_segment_distance(source, branch[i - 1], branch[i]));
        }
        distance = std::min(distance, point_segment_distance(source, branch.back(), branch.front()));
    }
    return distance;
}

double image_radius(double source_radius, double determinant)
{
    const double abs_det = std::max(std::abs(determinant), 1.0e-8);
    return std::max(2.5 * source_radius / std::sqrt(abs_det), 2.0 * source_radius);
}

struct HexResult {
    double magnification;
    double relative_error; // |a4 correction| / |magnification|, used for VBM-style mode switch
};

HexResult hexadecapole_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings)
{
    const double sineta[8] = {0.0, 1.0, 0.0, -1.0, kSqrtHalf, kSqrtHalf, -kSqrtHalf, -kSqrtHalf};
    const double coseta[8] = {1.0, 0.0, -1.0, 0.0, kSqrtHalf, -kSqrtHalf, -kSqrtHalf, kSqrtHalf};

    const double a0 = point_magnifier.binary_mag0(separation, mass_ratio, source).magnification;
    double a1_plus = 0.0;
    double a2_plus = 0.0;
    double a1_cross = 0.0;
    for (int i = 0; i < 4; ++i) {
        a1_plus += point_magnifier
                       .binary_mag0(separation, mass_ratio,
                           {source.x + source_radius * coseta[i],
                               source.y + source_radius * sineta[i]})
                       .magnification;
        a2_plus += point_magnifier
                       .binary_mag0(separation, mass_ratio,
                           {source.x + 0.5 * source_radius * coseta[i],
                               source.y + 0.5 * source_radius * sineta[i]})
                       .magnification;
        a1_cross += point_magnifier
                        .binary_mag0(separation, mass_ratio,
                            {source.x + source_radius * coseta[i + 4],
                                source.y + source_radius * sineta[i + 4]})
                        .magnification;
    }
    a1_plus = a1_plus / 4.0 - a0;
    a2_plus = a2_plus / 4.0 - a0;
    a1_cross = a1_cross / 4.0 - a0;

    const double a2rho2 = (16.0 * a2_plus - a1_plus) / 3.0;
    const double a4rho4 = (a1_plus + a1_cross) / 2.0 - a2rho2;
    const double gamma = limb_darkening_gamma(settings);
    const double lambda = limb_darkening_lambda(settings);
    const double quad_corr = 0.5 * a2rho2 * (1.0 - 0.2 * gamma - lambda / 9.0);
    const double hex_corr = a4rho4 / 3.0 * (1.0 - 11.0 * gamma / 35.0 - 7.0 * lambda / 39.0);
    const double magnification = a0 + quad_corr + hex_corr;
    const double rel_err = std::abs(hex_corr) / std::max(std::abs(magnification), 1.0e-10);
    return {magnification, rel_err};
}

struct PolarBoundaryRow {
    double min_r = 0.0;
    double max_r = 0.0;
    double phi = 0.0;
};

struct PolarBoundaryScratch {
    std::vector<PolarBoundaryRow> rows;
    std::vector<std::vector<int>> row_indices_by_phi;
};

struct PolarMapCacheView {
    const std::vector<SourcePosition>* mapped_sources = nullptr;
    const std::vector<int>* radial_offsets = nullptr;
    int radial_offset_min_index = 0;
    int phi_bins = 0;
    double dr = 1.0;
};

struct LegacyImageAreaScratch {
    std::vector<double> xmin;
    std::vector<double> xmax;
    std::vector<double> ax;
    std::vector<double> y;
    std::vector<double> dys;
    std::unordered_map<int, std::vector<int>> row_indices;

    void ensure(std::size_t index)
    {
        if (xmin.size() <= index) {
            const std::size_t size = index + 1;
            xmin.resize(size);
            xmax.resize(size);
            ax.resize(size);
            y.resize(size);
            dys.resize(size);
        }
    }
};

struct LegacyAreaDiagnostics {
    int seed_count = 0;
    int processed_images = 0;
    int fold_seed_count = 0;
    int boundary_rows = 0;
    int gap_repairs = 0;
    int overlaps = 0;
    double max_jump_cells = 0.0;
    double estimated_error = 0.0;
};

double high_magnification_floor_coefficient(
    const LegacyAreaDiagnostics& diagnostics,
    double magnification,
    double source_radius)
{
    if (source_radius >= 0.1 &&
        (diagnostics.gap_repairs > 0 || diagnostics.max_jump_cells > 50.0)) {
        return 0.35;
    }
    if (source_radius < 1.0e-2 &&
        diagnostics.seed_count <= 5 &&
        std::abs(magnification) <= 10.0 &&
        diagnostics.gap_repairs > 0 &&
        diagnostics.max_jump_cells > 20.0) {
        return 0.06;
    }
    if (source_radius < 1.0e-2 &&
        diagnostics.seed_count <= 5 &&
        std::abs(magnification) <= 10.0 &&
        diagnostics.max_jump_cells > 1000.0) {
        return 0.07;
    }
    if (source_radius < 1.0e-2 &&
        diagnostics.seed_count >= 16 &&
        diagnostics.overlaps >= 8 &&
        diagnostics.gap_repairs >= 100 &&
        diagnostics.max_jump_cells > 1000.0 &&
        std::abs(magnification) <= 50.0) {
        return 0.06;
    }
    if (source_radius < 1.0e-2 && std::abs(magnification) <= 300.0) {
        return 0.02;
    }
    if (source_radius < 1.0e-2 && std::abs(magnification) <= 1000.0) {
        return 0.10;
    }
    if (source_radius < 1.0e-2 &&
        diagnostics.seed_count <= 5 &&
        diagnostics.gap_repairs > 0 &&
        diagnostics.max_jump_cells > 20.0) {
        return 0.06;
    }
    if (std::abs(magnification) <= 80.0 || diagnostics.gap_repairs <= 1000) {
        return 0.0;
    }
    return std::abs(magnification) > 1000.0 ? 4.0 : 1.0;
}

double legacy_area_error_indicator(
    const LegacyAreaDiagnostics& diagnostics,
    double source_radius,
    const FiniteSourceSettings& settings)
{
    const int bins = std::max(settings.source_bins, 1);
    const double flux = source_flux(source_radius, settings);
    if (!std::isfinite(flux) || flux <= 0.0 || source_radius <= 0.0) {
        return 0.0;
    }

    // The image-area scan is first-order at the mapped-source boundary.  The
    // primary discretization error scales like boundary length times the cell
    // spacing, which is approximated here by boundary_rows * cell_area.  Gap and
    // overlap counts are topology warnings, not area errors, so keep their
    // weights small and let high_magnification_floor_coefficient handle the
    // known hard failure patterns.
    const double gap_weight = source_radius >= 2.0e-2
        ? 0.005
        : ((source_radius < 1.0e-2 && diagnostics.seed_count >= 16) ? 0.015 : 0.03);
    const double overlap_weight = source_radius >= 2.0e-2 ? 0.005 : 0.02;
    const double boundary_weight = source_radius >= 2.0e-2 ? 0.012 : 0.03;
    double uncertain_cells =
        boundary_weight * static_cast<double>(diagnostics.boundary_rows) +
        gap_weight * static_cast<double>(diagnostics.gap_repairs) +
        overlap_weight * static_cast<double>(diagnostics.overlaps) +
        0.02 * static_cast<double>(std::max(0, diagnostics.fold_seed_count)) +
        0.002 * static_cast<double>(std::max(0, diagnostics.seed_count - 5));
    if (source_radius >= 1.0e-3 && diagnostics.seed_count >= 16 && diagnostics.overlaps >= 8) {
        const double island_weight = source_radius < 1.0e-2 ? 0.001 : 0.0005;
        uncertain_cells += island_weight * static_cast<double>(diagnostics.seed_count) *
                           static_cast<double>(diagnostics.overlaps);
    }
    if (source_radius >= 1.0e-3 && diagnostics.max_jump_cells > 10.0) {
        const double jump_weight = source_radius >= 2.0e-2 ? 0.2 : 4.0;
        uncertain_cells += jump_weight * std::log10(diagnostics.max_jump_cells / 10.0);
    }
    const double cell_area = (source_radius / static_cast<double>(bins)) *
                             (source_radius / static_cast<double>(bins));
    const double estimate = 1.25 * uncertain_cells * cell_area / flux;
    return std::isfinite(estimate) ? estimate : 0.0;
}

double wrap_angle(double angle)
{
    while (angle < 0.0) {
        angle += 2.0 * kPi;
    }
    while (angle >= 2.0 * kPi) {
        angle -= 2.0 * kPi;
    }
    return angle;
}

double trace_polar_boundary_rows(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const PolarMapCacheView* map_cache,
    double start_radius,
    double start_phi,
    double dphi,
    PolarBoundaryScratch& scratch)
{
    const double dr = source_radius / static_cast<double>(std::max(settings.source_bins, 1));
    const double source_radius2 = source_radius * source_radius;
    const bool uniform_source = settings.limb_darkening_c == 0.0 && settings.limb_darkening_d == 0.0;
    if (!uniform_source && finite_magnifier != nullptr) {
        finite_magnifier->ensure_limb_darkening_table();
    }
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    const int phi_bins = static_cast<int>(scratch.row_indices_by_phi.size());
    const int max_rows = std::max(64, phi_bins * 2 + 16);
    const int max_radial_steps = std::max(64, settings.source_bins * 12);

    double total_count = 0.0;
    double radius_origin = start_radius;
    double phi = wrap_angle(start_phi);
    for (int row = 0; row < max_rows; ++row) {
        const int phi_index = std::clamp(static_cast<int>(phi / std::abs(dphi)), 0, phi_bins - 1);
        const double cos_phi = std::cos(phi);
        const double sin_phi = std::sin(phi);

        double count = 0.0;
        double dz2 = std::numeric_limits<double>::infinity();
        double dz2_last = dz2;
        double max_r = radius_origin;
        double min_r = radius_origin;

        double radius = radius_origin;
        double direction = dr;
        for (int step = 0; step < max_radial_steps; ++step) {
            SourcePosition mapped;
            const int radial_index = static_cast<int>(std::floor(radius / dr));
            const bool use_cached_map =
                map_cache != nullptr && map_cache->mapped_sources != nullptr &&
                map_cache->radial_offsets != nullptr &&
                map_cache->phi_bins == phi_bins && map_cache->dr == dr &&
                radial_index >= map_cache->radial_offset_min_index &&
                radial_index < map_cache->radial_offset_min_index +
                    static_cast<int>(map_cache->radial_offsets->size()) &&
                (*map_cache->radial_offsets)[static_cast<std::size_t>(
                    radial_index - map_cache->radial_offset_min_index)] >= 0;
            if (use_cached_map) {
                const int row_offset = (*map_cache->radial_offsets)[static_cast<std::size_t>(
                    radial_index - map_cache->radial_offset_min_index)];
                const auto index =
                    static_cast<std::size_t>(row_offset) *
                        static_cast<std::size_t>(map_cache->phi_bins) +
                    static_cast<std::size_t>(phi_index);
                mapped = (*map_cache->mapped_sources)[index];
            } else {
                const SourcePosition image {radius * cos_phi, radius * sin_phi};
                mapped = map_binary_lens_real(mapper, image.x, image.y);
            }
            dz2_last = dz2;
            dz2 = distance_squared(mapped, source);

            if (dz2 <= source_radius2) {
                if (direction < 0.0 && count == 0.0) {
                    max_r = radius - direction;
                }
                const double brightness =
                    uniform_source ? 1.0 : finite_magnifier->legacy_limb_darkening_table_brightness(
                                               dz2 / source_radius2);
                count += brightness * radius;
            } else if (direction > 0.0) {
                if (dz2_last <= source_radius2) {
                    max_r = radius;
                } else if (total_count == 0.0 && row <= 1 && radius == radius_origin) {
                    radius += direction;
                    continue;
                }
                direction = -dr;
                radius = radius_origin;
                min_r = radius + direction;
            } else {
                if (dz2_last <= source_radius2) {
                    min_r = radius;
                }
                if (!scratch.rows.empty() && count == 0.0 &&
                    radius >= scratch.rows.back().min_r - direction) {
                    radius += direction;
                    continue;
                }

                if (count == 0.0) {
                    if (!scratch.rows.empty()) {
                        return total_count;
                    }
                } else {
                    for (const int index : scratch.row_indices_by_phi[static_cast<std::size_t>(phi_index)]) {
                        const auto& existing = scratch.rows[static_cast<std::size_t>(index)];
                        if (min_r + dr < existing.max_r && max_r - dr > existing.min_r) {
                            return total_count;
                        }
                    }
                    total_count += count;
                    scratch.row_indices_by_phi[static_cast<std::size_t>(phi_index)].push_back(
                        static_cast<int>(scratch.rows.size()));
                    scratch.rows.push_back({min_r, max_r, phi});
                }

                radius_origin = max_r - dr;
                phi = wrap_angle(phi + dphi);
                break;
            }
            radius += direction;
        }

        if (row > 0 && count == 0.0) {
            break;
        }
    }
    return total_count;
}

double inverse_ray_polar_boundary_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const PolarMapCacheView* map_cache = nullptr,
    const std::vector<SourcePosition>* seed_positions = nullptr)
{
    std::vector<SourcePosition> image_positions;
    if (seed_positions != nullptr) {
        image_positions = *seed_positions;
    } else {
        const auto images = point_magnifier.binary_images(separation, mass_ratio, source);
        image_positions.reserve(images.size());
        for (const auto& image : images) {
            image_positions.push_back(image.position);
        }
    }
    if (image_positions.empty()) {
        return std::nan("");
    }

    const int source_bins = std::max(settings.source_bins, 1);
    // Phase 2: Grid spacing refinement (polar method)
    // Base resolution (1e-3 × source_radius) guarantees all images are detected
    const double base_ray_spacing = source_radius * 1.0e-3;
    const double dr_from_bins = source_radius / static_cast<double>(source_bins);
    const double dr = std::min(dr_from_bins, base_ray_spacing);
    const int phi_bins = std::max(16, static_cast<int>(2.0 * kPi / (dr * settings.grid_ratio)));
    const double dphi = 2.0 * kPi / static_cast<double>(phi_bins);
    const double total_source_flux = source_flux(source_radius, settings);
    if (!std::isfinite(total_source_flux)) {
        return std::nan("");
    }

    std::vector<double> image_area_counts(image_positions.size(), 0.0);
    std::vector<bool> skip(image_positions.size(), false);
    double total_count = 0.0;
    for (std::size_t i = 0; i < image_positions.size(); ++i) {
        if (skip[i]) {
            continue;
        }

        const double image_radius_value = std::hypot(image_positions[i].x, image_positions[i].y);
        double image_phi = wrap_angle(std::atan2(image_positions[i].y, image_positions[i].x));
        const double grid_radius = std::floor(image_radius_value / dr) * dr + 0.5 * dr;
        image_phi = std::floor(image_phi / dphi) * dphi + 0.5 * dphi;

        PolarBoundaryScratch scratch;
        scratch.row_indices_by_phi.assign(static_cast<std::size_t>(phi_bins), {});
        double count = trace_polar_boundary_rows(
            separation, mass_ratio, source, source_radius, settings, finite_magnifier, map_cache,
            grid_radius, image_phi, dphi, scratch);
        if (!scratch.rows.empty()) {
            const double reverse_start_radius = scratch.rows.front().max_r;
            count += trace_polar_boundary_rows(
                separation, mass_ratio, source, source_radius, settings, finite_magnifier, map_cache,
                reverse_start_radius, image_phi - dphi, -dphi, scratch);
        }

        total_count += count;
        image_area_counts[i] = count;

        for (std::size_t j = 0; j < image_positions.size(); ++j) {
            if (j == i) {
                continue;
            }
            const double other_radius = std::hypot(image_positions[j].x, image_positions[j].y);
            const double other_phi = wrap_angle(std::atan2(image_positions[j].y, image_positions[j].x));
            const double other_grid_radius = std::floor(other_radius / dr) * dr;
            const double other_grid_phi = std::floor(other_phi / dphi) * dphi;
            for (const auto& row : scratch.rows) {
                const double row_phi = wrap_angle(row.phi - 0.5 * std::abs(dphi));
                const double delta_phi = std::abs(wrap_angle(other_grid_phi - row_phi));
                const double wrapped_delta_phi = std::min(delta_phi, 2.0 * kPi - delta_phi);
                if (wrapped_delta_phi <= 1.01 * std::abs(dphi) &&
                    other_grid_radius >= row.min_r - 1.01 * dr &&
                    other_grid_radius <= row.max_r + 1.01 * dr) {
                    if (j < i) {
                        total_count -= image_area_counts[j];
                    } else {
                        skip[j] = true;
                    }
                    break;
                }
            }
        }
    }

    const double image_flux = total_count * dr * dphi;
    return image_flux / total_source_flux;
}

double legacy_limb_brightness(
    double normalized_radius2,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier)
{
    if (settings.limb_darkening_c == 0.0 && settings.limb_darkening_d == 0.0) {
        return 1.0;
    }
    return finite_magnifier != nullptr ?
        finite_magnifier->limb_darkening_table_brightness(normalized_radius2) :
        source_surface_brightness(normalized_radius2, settings);
}

std::vector<SourcePosition> legacy_residual_selected_images(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source);

void append_valid_probe_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    SourcePosition probe_source,
    std::vector<SourcePosition>& seeds)
{
    const double source_radius2 = source_radius * source_radius;
    if (source_radius <= 0.0 ||
        distance_squared(probe_source, source) >= source_radius2 * (1.0 + 1.0e-10)) {
        return;
    }

    const auto probe_images =
        legacy_residual_selected_images(point_magnifier, separation, mass_ratio, probe_source);
    if (probe_images.size() <= 3) {
        return;
    }

    // Compare only against seeds that existed before this probe.  The two fold
    // images born at one caustic crossing can be much closer than rho, so they
    // must not suppress each other.
    const std::size_t n_seeds_before = seeds.size();
    for (const auto& img : probe_images) {
        const SourcePosition mapped = map_binary_lens_real(mapper, img.x, img.y);
        const double mapped_distance2 = distance_squared(mapped, source);
        if (mapped_distance2 > source_radius2 * (1.0 + 1.0e-8)) {
            continue;
        }
        bool is_dup = false;
        for (std::size_t si = 0; si < n_seeds_before; ++si) {
            if (distance_squared(img, seeds[si]) < 0.0625 * source_radius2) {
                const SourcePosition existing_mapped =
                    map_binary_lens_real(mapper, seeds[si].x, seeds[si].y);
                const double existing_distance2 = distance_squared(existing_mapped, source);
                if (mapped_distance2 + 1.0e-16 < existing_distance2) {
                    seeds[si] = img;
                }
                is_dup = true;
                break;
            }
        }
        if (!is_dup) {
            seeds.push_back(img);
        }
    }
}

void append_caustic_probe_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    SourcePosition critical_source,
    std::vector<SourcePosition>& seeds)
{
    const double dx = critical_source.x - source.x;
    const double dy = critical_source.y - source.y;
    const double distance = std::hypot(dx, dy);
    if (distance <= 0.0 || distance >= source_radius) {
        return;
    }

    const double ux = dx / distance;
    const double uy = dy / distance;
    const double steps[] = {
        0.02 * source_radius,
        0.05 * source_radius,
        0.15 * source_radius,
        0.35 * source_radius,
    };
    for (const double step : steps) {
        const SourcePosition probes[2] = {
            {critical_source.x + ux * step, critical_source.y + uy * step},
            {critical_source.x - ux * step, critical_source.y - uy * step},
        };
        for (const auto& probe_source : probes) {
            append_valid_probe_image_seeds(
                point_magnifier, mapper, separation, mass_ratio, source, source_radius,
                probe_source, seeds);
        }
    }

    if (seeds.size() <= 3) {
        constexpr int angular_probes = 16;
        const double radial_steps[] = {
            0.02 * source_radius,
            0.05 * source_radius,
            0.10 * source_radius,
            0.20 * source_radius,
            0.35 * source_radius,
        };
        for (const double step : radial_steps) {
            for (int i = 0; i < angular_probes; ++i) {
                const double theta =
                    2.0 * kPi * static_cast<double>(i) / static_cast<double>(angular_probes);
                const SourcePosition probe_source {
                    critical_source.x + step * std::cos(theta),
                    critical_source.y + step * std::sin(theta),
                };
                append_valid_probe_image_seeds(
                    point_magnifier, mapper, separation, mass_ratio, source, source_radius,
                    probe_source, seeds);
                if (seeds.size() > 3) {
                    return;
                }
            }
        }
    }
}

void append_boundary_probe_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    std::vector<SourcePosition>& seeds)
{
    constexpr std::size_t max_seeds = 128;
    if (source_radius <= 0.0 || seeds.size() >= max_seeds) {
        return;
    }

    constexpr int samples = 400;
    constexpr double inward_fraction = 0.02;
    const double probe_radius = source_radius * (1.0 - inward_fraction);
    for (int i = 0; i < samples && seeds.size() < max_seeds; ++i) {
        const double phi = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(samples);
        const SourcePosition probe_source {
            source.x + probe_radius * std::cos(phi),
            source.y + probe_radius * std::sin(phi),
        };
        append_valid_probe_image_seeds(
            point_magnifier, mapper, separation, mass_ratio, source, source_radius,
            probe_source, seeds);
    }
}

void append_interior_probe_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    std::vector<SourcePosition>& seeds)
{
    if (source_radius < 2.0e-2) {
        return;
    }
    constexpr int angle_samples = 64;
    constexpr double radii[] = {0.25, 0.5, 0.75};
    for (const double radius_fraction : radii) {
        const double probe_radius = source_radius * radius_fraction;
        for (int i = 0; i < angle_samples; ++i) {
            const double phi = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(angle_samples);
            const SourcePosition probe_source {
                source.x + probe_radius * std::cos(phi),
                source.y + probe_radius * std::sin(phi),
            };
            append_valid_probe_image_seeds(
                point_magnifier, mapper, separation, mass_ratio, source, source_radius,
                probe_source, seeds);
        }
    }
}

std::vector<SourcePosition> critical_sources_at_phase(
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    double phase_angle)
{
    math::PolynomialRootSolver solver;
    const auto root_result = solver.solve(critical_curve_polynomial_coefficients(
        separation, mass_ratio, std::polar(1.0, phase_angle)));
    if (root_result.status != math::RootSolverStatus::ok) {
        return {};
    }

    std::vector<SourcePosition> sources;
    sources.reserve(root_result.roots.size());
    for (const auto& root : root_result.roots) {
        sources.push_back(map_binary_lens_real(mapper, root.real(), root.imag()));
    }
    return sources;
}

double nearest_critical_source_distance2(
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double phase_angle,
    SourcePosition* nearest_source)
{
    double best = std::numeric_limits<double>::infinity();
    const auto critical_sources =
        critical_sources_at_phase(mapper, separation, mass_ratio, phase_angle);
    for (const auto& critical_source : critical_sources) {
        const double distance2 = distance_squared(critical_source, source);
        if (distance2 < best) {
            best = distance2;
            if (nearest_source != nullptr) {
                *nearest_source = critical_source;
            }
        }
    }
    return best;
}

SourcePosition refine_nearest_critical_source(
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double center_phase,
    double phase_step)
{
    double left = center_phase - phase_step;
    double right = center_phase + phase_step;
    constexpr double golden = 0.61803398874989484820;
    double x1 = right - golden * (right - left);
    double x2 = left + golden * (right - left);
    SourcePosition nearest1;
    SourcePosition nearest2;
    double f1 = nearest_critical_source_distance2(
        mapper, separation, mass_ratio, source, x1, &nearest1);
    double f2 = nearest_critical_source_distance2(
        mapper, separation, mass_ratio, source, x2, &nearest2);

    for (int iter = 0; iter < 32; ++iter) {
        if (f1 > f2) {
            left = x1;
            x1 = x2;
            f1 = f2;
            nearest1 = nearest2;
            x2 = left + golden * (right - left);
            f2 = nearest_critical_source_distance2(
                mapper, separation, mass_ratio, source, x2, &nearest2);
        } else {
            right = x2;
            x2 = x1;
            f2 = f1;
            nearest2 = nearest1;
            x1 = right - golden * (right - left);
            f1 = nearest_critical_source_distance2(
                mapper, separation, mass_ratio, source, x1, &nearest1);
        }
    }

    return f1 < f2 ? nearest1 : nearest2;
}

std::vector<SourcePosition> legacy_augmented_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double hint_caustic_dist = std::numeric_limits<double>::infinity())
{
    std::vector<SourcePosition> seeds;
    const auto point_images = point_magnifier.binary_images(separation, mass_ratio, source);
    seeds.reserve(5);
    for (const auto& image : point_images) {
        seeds.push_back(image.position);
    }
    if (source_radius <= 0.0) {
        return seeds;
    }
    // Do not skip the caustic scan based on hint_caustic_dist alone: the
    // computed caustic distance can be slightly over-estimated (e.g. when a
    // phantom wrap-around segment distorts the branch grid search), causing a
    // false early exit when the source disk just straddles the caustic.

    const double source_radius2 = source_radius * source_radius;
    const int samples = 1400;
    const double phase_step = 2.0 * kPi / static_cast<double>(samples);
    double best_distance2 = std::numeric_limits<double>::infinity();
    double best_phase = 0.0;
    constexpr int nskip = 40;

    // Phase 1: find the first caustic crossing and add fold-image seeds.
    // Merge into existing seeds rather than replacing them so that the Phase 0
    // standard-image seed (high-J, non-fold image) is preserved.  Without this
    // the standard image is absent at low bins, where fold-image flood-fills do
    // not expand far enough to cover it.
    //
    // IMPORTANT: the duplicate check uses the Phase-0 seed count snapshot, not
    // the live seeds vector.  This prevents F+ from blocking F- when both fold
    // images land within rho of each other (which happens when the probe source
    // is only slightly inside the caustic, so F+/F- separation << rho).
    bool found_first_crossing = false;
    for (int kphi = 0; kphi < nskip && !found_first_crossing; ++kphi) {
        for (int jphi = 0; jphi < samples / nskip && !found_first_crossing; ++jphi) {
            const int sample = jphi * nskip + kphi;
            const double phi = phase_step * static_cast<double>(sample);
            const auto critical_sources =
                critical_sources_at_phase(mapper, separation, mass_ratio, phi);
            for (const auto& critical_source : critical_sources) {
                const double distance2 = distance_squared(critical_source, source);
                if (distance2 < best_distance2) {
                    best_distance2 = distance2;
                    best_phase = phi;
                }
                if (distance2 >= source_radius2 || distance2 <= 0.0) {
                    continue;
                }
                const std::size_t before = seeds.size();
                append_caustic_probe_image_seeds(
                    point_magnifier, mapper, separation, mass_ratio, source, source_radius,
                    critical_source, seeds);
                found_first_crossing = seeds.size() > before;
            }
        }
    }
    if (!found_first_crossing && best_distance2 < source_radius2 && best_distance2 > 0.0) {
        const SourcePosition critical_source = refine_nearest_critical_source(
            mapper, separation, mass_ratio, source, best_phase, phase_step);
        append_caustic_probe_image_seeds(
            point_magnifier, mapper, separation, mass_ratio, source, source_radius,
            critical_source, seeds);
    }

    // Phase 2: detect additional arc crossings not covered by Phase 1 seeds.
    // Scan in sequential phase order so that contiguous "inside" segments on
    // each caustic branch are identified.  Seeds are added only when a branch
    // transitions from outside to inside the source disk; this prevents the
    // O(samples) seed explosion that occurs when the source disk engulfs a
    // large arc of the caustic (each sample would otherwise add new fold seeds
    // because fold-image positions vary rapidly along the arc).
    //
    // Branch continuity across samples is maintained by greedy nearest-neighbour
    // matching of each sample's critical-curve roots to the previous sample's
    // root positions.  The polynomial has at most 4 roots so the matching is O(1).
    if (seeds.size() >= 5) {
        constexpr int kMaxBranches = 4;
        // prev_pos[i]: image-plane position of branch i at the previous sample.
        // Initialised far away so that the first sample establishes branch order.
        std::array<SourcePosition, kMaxBranches> prev_pos;
        prev_pos.fill({1.0e30, 1.0e30});
        std::array<bool, kMaxBranches> branch_inside;
        branch_inside.fill(false);

        for (int sample = 0; sample < samples; ++sample) {
            const double phi = phase_step * static_cast<double>(sample);
            const auto critical_sources =
                critical_sources_at_phase(mapper, separation, mass_ratio, phi);
            const int ncur = static_cast<int>(critical_sources.size());
            if (ncur == 0) {
                continue;
            }

            // Match current roots to branches by greedy nearest-neighbour in image space.
            std::array<int, kMaxBranches> assignment;
            assignment.fill(-1);
            std::array<bool, kMaxBranches> used;
            used.fill(false);
            for (int bi = 0; bi < kMaxBranches; ++bi) {
                double best_d2 = 1.0e60;
                int best_j = -1;
                for (int j = 0; j < ncur; ++j) {
                    if (used[j]) {
                        continue;
                    }
                    const double d2 = distance_squared(critical_sources[j], prev_pos[bi]);
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_j = j;
                    }
                }
                if (best_j >= 0) {
                    assignment[bi] = best_j;
                    used[best_j] = true;
                }
            }

            // Update branch states and fire on outside→inside transitions.
            for (int bi = 0; bi < kMaxBranches; ++bi) {
                const int j = assignment[bi];
                if (j < 0) {
                    branch_inside[bi] = false;
                    continue;
                }
                const auto& cs = critical_sources[j];
                const double d2 = distance_squared(cs, source);
                const bool now_inside = (d2 < source_radius2 && d2 > 0.0);

                const bool add_arc_seed =
                    source_radius >= 2.0e-2 && now_inside && (sample % 20 == 0);
                if ((now_inside && !branch_inside[bi]) || add_arc_seed) {
                    // This branch just entered the source disk: add fold seeds.
                    append_caustic_probe_image_seeds(
                        point_magnifier, mapper, separation, mass_ratio, source, source_radius,
                        cs, seeds);
                }
                branch_inside[bi] = now_inside;
                prev_pos[bi] = cs;
            }
        }
    }
    append_boundary_probe_image_seeds(
        point_magnifier, mapper, separation, mass_ratio, source, source_radius, seeds);
    append_interior_probe_image_seeds(
        point_magnifier, mapper, separation, mass_ratio, source, source_radius, seeds);
    return seeds;
}

std::vector<SourcePosition> legacy_residual_selected_images(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source)
{
    std::vector<SourcePosition> images;
    const auto candidates =
        point_magnifier.binary_image_candidates(separation, mass_ratio, source);
    images.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (candidate.residual * candidate.residual < 1.0e-12) {
            images.push_back(candidate.position);
        }
    }
    return images;
}

double legacy_imagearea0_binary(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    SourcePosition seed,
    double dy,
    int& yi,
    LegacyImageAreaScratch& scratch,
    int jacobian_sign = 0)
{
    double countx = 0.0;
    double countall = 0.0;
    double dz2 = 99999999.9;
    const double incr = std::abs(dy);
    const double inv_incr = 1.0 / incr;
    double dx = incr;
    SourcePosition image = seed;
    double x0 = seed.x;
    const double source_radius2 = source_radius * source_radius;
    const double inv_source_radius2 = 1.0 / source_radius2;
    int guard = 0;
    // Phase 1: Max steps decoupling - image completeness is bins-independent
    // Flood-fill must complete regardless of bin count to ensure all detected
    // images are fully integrated. Budget: 500k steps covers even complex overlaps.
    const int max_steps = 500000;

    while (++guard < max_steps) {
        const double dz2_last = dz2;
        const double mapped_distance2 =
            mapped_binary_lens_distance2(mapper, image.x, image.y, source);
        // When a Jacobian-sign guard is active, treat pixels on the wrong side of the
        // critical curve as outside even if the mapped source is inside the disk.  This
        // prevents a fold-image flood-fill from bleeding across the critical curve into
        // the adjacent fold image on the opposite parity, which would otherwise cause
        // wildly wrong (sometimes negative) magnifications.
        const bool jac_ok = jacobian_sign == 0 ||
            binary_jacobian_sign(mapper, image.x, image.y) != -jacobian_sign;
        dz2 = (jac_ok) ? mapped_distance2 : source_radius2 + 1.0;

        scratch.ensure(static_cast<std::size_t>(yi));
        if (dz2 <= source_radius2) {
            if (dx == -incr && countx == 0.0) {
                scratch.xmax[static_cast<std::size_t>(yi)] = image.x - dx;
            }
            const double normalized_radius2 = mapped_distance2 * inv_source_radius2;
            countx += legacy_limb_brightness(normalized_radius2, settings, finite_magnifier);
        } else {
            if (dx == incr) {
                if (dz2_last <= source_radius2) {
                    scratch.xmax[static_cast<std::size_t>(yi)] = image.x;
                }
                dx = -incr;
                image.x = x0;
                scratch.xmin[static_cast<std::size_t>(yi)] = image.x + dx;
            } else {
                if (dz2_last <= source_radius2) {
                    scratch.xmin[static_cast<std::size_t>(yi)] = image.x;
                }
                if (yi != 0 && countx == 0.0) {
                    scratch.ensure(static_cast<std::size_t>(yi - 1));
                    if (image.x >= scratch.xmin[static_cast<std::size_t>(yi - 1)] - dx) {
                        image.x += dx;
                        continue;
                    }
                }

                countall += countx;
                scratch.ax[static_cast<std::size_t>(yi)] = countx;
                scratch.y[static_cast<std::size_t>(yi)] = image.y;
                scratch.dys[static_cast<std::size_t>(yi)] = dy;
                if (countx == 0.0) {
                    scratch.dys[static_cast<std::size_t>(yi)] = -dy;
                    break;
                }

                const int row_key = static_cast<int>(image.y * inv_incr + kLegacyIndexOffset);
                auto& row_indices = scratch.row_indices[row_key];
                for (const int index : row_indices) {
                    const auto existing = static_cast<std::size_t>(index);
                    if (scratch.xmin[static_cast<std::size_t>(yi)] + incr < scratch.xmax[existing] &&
                        scratch.xmax[static_cast<std::size_t>(yi)] - incr > scratch.xmin[existing]) {
                        return countall - countx;
                    }
                }
                row_indices.push_back(yi);

                ++yi;
                scratch.ensure(static_cast<std::size_t>(yi));
                dx = incr;
                x0 = scratch.xmax[static_cast<std::size_t>(yi - 1)];
                image.x = x0 - dx;
                image.y += dy;
                countx = 0.0;
            }
        }
        image.x += dx;
    }

    return countall;
}

// Phase 1-3: Component-based union tracking (parallel to current implementation)
// Used for future redesign, not yet integrated into main logic.
// See .note/overlap-component-redesign.md for design rationale.
struct ProcessedComponent {
    int component_id;
    std::vector<size_t> seed_indices;    // which seeds form this component
    double area = 0.0;
    int fold_parity = 0;                  // +1, -1, or 0 for jacobian sign
};

// Phase 4: Clarified parity logic (separated from overlap detection)
// Determines whether two seeds can geometrically overlap based on Jacobian parity.
// Fold images (|J| < kFoldJacThreshold) are parity-restricted to one side of the
// critical curve. Non-fold images have no restriction. This is a pure geometry check,
// distinct from actual flood-fill overlap detection.
inline bool can_overlap_across_parity(int jac_sign_a, int jac_sign_b) {
    // Both are non-fold images (unrestricted)
    if (jac_sign_a == 0 || jac_sign_b == 0) {
        return true;
    }
    // Both are fold images: must have same parity to overlap
    return jac_sign_a == jac_sign_b;
}

// Phase 6: Component union with proper inclusion-exclusion
// For area union: area(A ∪ B) = area(A) + area(B) - area(A ∩ B)
// When two components overlap, we need to subtract their intersection.
// Currently using conservative approximation: subtract min(area_a, area_b) if they overlap.
// This assumes the overlap region is smaller of the two components.
struct ComponentUnion {
    double total_area = 0.0;
    std::set<int> component_ids;

    void add_component(double comp_area, int comp_id) {
        // For now, simple union: sum areas (conservative, may double-count)
        // Future: track actual intersection regions for rigorous inclusion-exclusion
        total_area += comp_area;
        component_ids.insert(comp_id);
    }

    void subtract_overlap(double overlap_area) {
        // Subtract known overlap to avoid double-counting
        if (total_area > 0.0) {
            total_area = std::max(0.0, total_area - overlap_area);
        }
    }
};

double legacy_imagearea4_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<SourcePosition>* precomputed_seeds = nullptr,
    LegacyAreaDiagnostics* diagnostics = nullptr)
{
    if ((settings.limb_darkening_c != 0.0 || settings.limb_darkening_d != 0.0) &&
        finite_magnifier != nullptr) {
        finite_magnifier->ensure_limb_darkening_table();
    }

    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    const auto computed_images = precomputed_seeds == nullptr ?
        legacy_augmented_image_seeds(point_magnifier, mapper, separation, mass_ratio, source, source_radius) :
        std::vector<SourcePosition> {};
    const auto& images = precomputed_seeds == nullptr ? computed_images : *precomputed_seeds;
    if (images.empty() || source_radius <= 0.0) {
        return std::nan("");
    }
    if (diagnostics != nullptr) {
        *diagnostics = {};
        diagnostics->seed_count = static_cast<int>(images.size());
    }
    const double nbin = static_cast<double>(std::max(settings.source_bins, 1));
    // Phase 2: Grid spacing refinement - ensure image completeness even at low bins
    // Base resolution (1e-3 × source_radius) guarantees all images are detected
    // regardless of bin count. Bins then adds integration precision on top.
    const double base_ray_spacing = source_radius * 1.0e-3;
    const double incr_from_bins = source_radius / nbin;
    const double incr = std::min(incr_from_bins, base_ray_spacing);
    const double incr2_margin = 0.5 * incr * 1.01;
    double area = 0.0;
    std::vector<double> areaimage(images.size(), 0.0);
    std::vector<int> overlap(images.size(), 0);
    std::vector<int> subtracted_previous_overlap(images.size(), 0);

    // Phase 1-3: Component tracking structure
    std::vector<ProcessedComponent> processed_components;
    std::vector<int> seed_to_component_id(images.size(), -1);  // -1 = not assigned
    int next_component_id = 0;

    // Phase 3: Track which components (not seeds) have been subtracted
    std::set<int> subtracted_component_ids;

    // Phase 6: Validation - compute component union in parallel for consistency check
    double component_union_area = 0.0;
    std::set<int> processed_component_ids;

    for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
        if (overlap[image_index] == 1) {
            continue;
        }

        LegacyImageAreaScratch scratch;
        scratch.ensure(1);
        double area0 = 0.0;
        double areai = 0.0;
        double dy = incr;
        int yi = 0;

        const SourcePosition seed = images[image_index];
        // Guard fold-image flood-fills against crossing the critical curve.
        // When the source disk straddles the caustic, both fold images (F+ and F-)
        // map into the source disk; without this guard the x-scan bleeds across the
        // critical curve from one fold image into the other, giving wrong (sometimes
        // negative) magnifications.
        //
        // Fold images sit close to the critical curve and have |J| << 1.  Standard
        // images (far from the critical curve) have |J| >> kFoldJacThreshold and are
        // NOT restricted — their flood-fills stay naturally within their own image
        // region because the mapped source exits the disk before crossing any critical
        // curve.  Applying the guard to them would incorrectly limit their area.
        constexpr double kFoldJacThreshold = 0.02;
        const double J_seed = binary_jacobian(mapper, seed.x, seed.y);
        const int jac_sign = (std::abs(J_seed) < kFoldJacThreshold)
            ? (J_seed > 0.0 ? 1 : J_seed < 0.0 ? -1 : 0)
            : 0;
        if (diagnostics != nullptr) {
            ++diagnostics->processed_images;
            if (jac_sign != 0) {
                ++diagnostics->fold_seed_count;
            }
        }
        scratch.xmin[0] = seed.x;
        scratch.xmax[0] = seed.x;
        areai = legacy_imagearea0_binary(
            mapper, source, source_radius, settings, finite_magnifier, seed, dy, yi, scratch,
            jac_sign);

        dy = -incr;
        scratch.ensure(static_cast<std::size_t>(yi));
        const SourcePosition lower_seed {scratch.xmax[0], seed.y + dy};
        scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[0];
        scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmax[0];
        scratch.y[static_cast<std::size_t>(yi)] = scratch.y[0];
        scratch.dys[static_cast<std::size_t>(yi)] = dy;
        ++yi;
        areai += legacy_imagearea0_binary(
            mapper, source, source_radius, settings, finite_magnifier, lower_seed, dy, yi, scratch,
            jac_sign);

        int nyi = yi;
        double areabound = 0.0;
        for (int row = 0; row < nyi; ++row) {
            scratch.ensure(static_cast<std::size_t>(row + 1));
            const double dxmax =
                scratch.xmax[static_cast<std::size_t>(row + 1)] -
                scratch.xmax[static_cast<std::size_t>(row)];
            const double dxmin =
                scratch.xmin[static_cast<std::size_t>(row + 1)] -
                scratch.xmin[static_cast<std::size_t>(row)];
            if (diagnostics != nullptr) {
                diagnostics->max_jump_cells = std::max(
                    diagnostics->max_jump_cells,
                    std::max(std::abs(dxmax), std::abs(dxmin)) / incr);
            }
            if (scratch.ax[static_cast<std::size_t>(row + 1)] > 0.0) {
                if (dxmax > 1.1 * incr) {
                    if (diagnostics != nullptr) {
                        ++diagnostics->gap_repairs;
                    }
                    const SourcePosition extra_seed {
                        scratch.xmax[static_cast<std::size_t>(row + 1)],
                        scratch.y[static_cast<std::size_t>(row)]};
                    scratch.ensure(static_cast<std::size_t>(yi));
                    scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmax[static_cast<std::size_t>(row)];
                    scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmax[static_cast<std::size_t>(row + 1)];
                    dy = -scratch.dys[static_cast<std::size_t>(row)];
                    scratch.dys[static_cast<std::size_t>(yi)] = dy;
                    ++yi;
                    area0 = legacy_imagearea0_binary(
                        mapper, source, source_radius, settings, finite_magnifier, extra_seed, dy,
                        yi, scratch, jac_sign);
                    areai += area0;
                    areabound += area0;
                    if (area0 <= 0.0) {
                        --yi;
                    }
                }
                if (dxmin > 1.1 * incr) {
                    if (diagnostics != nullptr) {
                        ++diagnostics->gap_repairs;
                    }
                    const SourcePosition extra_seed {
                        scratch.xmin[static_cast<std::size_t>(row + 1)] - incr,
                        scratch.y[static_cast<std::size_t>(row + 1)]};
                    scratch.ensure(static_cast<std::size_t>(yi));
                    scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row)];
                    scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row + 1)];
                    dy = scratch.dys[static_cast<std::size_t>(row)];
                    scratch.dys[static_cast<std::size_t>(yi)] = dy;
                    ++yi;
                    area0 = legacy_imagearea0_binary(
                        mapper, source, source_radius, settings, finite_magnifier, extra_seed, dy,
                        yi, scratch, jac_sign);
                    areai += area0;
                    areabound += area0;
                    if (area0 <= 0.0) {
                        --yi;
                    }
                }
                if (dxmin < -1.1 * incr) {
                    if (diagnostics != nullptr) {
                        ++diagnostics->gap_repairs;
                    }
                    const SourcePosition extra_seed {
                        scratch.xmin[static_cast<std::size_t>(row)] - incr,
                        scratch.y[static_cast<std::size_t>(row)]};
                    scratch.ensure(static_cast<std::size_t>(yi));
                    scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row + 1)];
                    scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row)];
                    dy = -scratch.dys[static_cast<std::size_t>(row)];
                    scratch.dys[static_cast<std::size_t>(yi)] = dy;
                    ++yi;
                    area0 = legacy_imagearea0_binary(
                        mapper, source, source_radius, settings, finite_magnifier, extra_seed, dy,
                        yi, scratch, jac_sign);
                    areai += area0;
                    areabound += area0;
                    if (area0 <= 0.0) {
                        --yi;
                    }
                }
            }
            if (row == nyi - 1 && areabound > 0.0 && yi > nyi) {
                nyi = yi;
            }
        }

        area += areai;
        areaimage[image_index] = areai;

        // Phase 5-6: Record component area and track union for validation
        if (seed_to_component_id[image_index] == -1) {
            // Create new component for this seed
            ProcessedComponent comp;
            comp.component_id = next_component_id++;
            comp.seed_indices.push_back(image_index);
            comp.area = areai;
            comp.fold_parity = jac_sign;
            processed_components.push_back(comp);
            seed_to_component_id[image_index] = comp.component_id;

            // Phase 6: Add to component union (new component)
            if (processed_component_ids.insert(comp.component_id).second) {
                component_union_area += areai;
            }
        } else {
            // Update existing component's area (union case)
            int comp_id = seed_to_component_id[image_index];
            for (auto& comp : processed_components) {
                if (comp.component_id == comp_id) {
                    comp.area += areai;  // Accumulate area for component union
                    comp.seed_indices.push_back(image_index);

                    // Phase 6: Add to component union (merge)
                    if (processed_component_ids.insert(comp_id).second) {
                        component_union_area += areai;
                    }
                    break;
                }
            }
        }

        if (diagnostics != nullptr) {
            for (int row = 0; row < nyi; ++row) {
                if (scratch.ax[static_cast<std::size_t>(row)] > 0.0) {
                    ++diagnostics->boundary_rows;
                }
            }
        }

        for (std::size_t other = 0; other < images.size(); ++other) {
            if (other == image_index) {
                continue;
            }
            const auto& position = images[other];
            // When the current seed has a Jacobian sign guard (fold image with
            // |J| < threshold), its flood-fill is restricted to one side of the
            // critical curve.  A seed on the OPPOSITE parity side has a
            // disconnected pre-image and cannot genuinely overlap with this
            // flood-fill; the xmax/xmin boundaries only extend to the critical
            // curve, so a seed just past it may be geometrically within the
            // scan-row bounding box while never being reached by the flood-fill.
            // Skip the overlap check in that case to avoid a false positive.
            if (jac_sign != 0) {
                const int jac_sign_other =
                    binary_jacobian_sign(mapper, position.x, position.y);
                if (jac_sign_other == -jac_sign) {
                    continue;
                }
                // Seeds extremely close to the critical curve often trace only
                // the near-caustic subset of a same-parity fold branch.  Let a
                // later, less singular seed on that branch run too; it will
                // subtract this earlier partial component through the
                // other<image_index path below.  Treating the critical seed's
                // coarse bounding box as a complete future-overlap can otherwise
                // drop the branch at specific grid phases.
                constexpr double kCriticalSeedOverlapThreshold = 1.0e-3;
                if (source_radius >= 4.0e-3 &&
                    settings.source_bins >= 35 &&
                    other > image_index &&
                    std::abs(J_seed) < kCriticalSeedOverlapThreshold &&
                    std::abs(binary_jacobian(mapper, position.x, position.y)) <
                        kFoldJacThreshold) {
                    continue;
                }
            }
            for (int row = 0; row < nyi; ++row) {
                const auto row_index = static_cast<std::size_t>(row);
                if (scratch.ax[row_index] <= 0.0) {
                    continue;
                }
                if (position.y >= scratch.y[row_index] - incr2_margin &&
                    position.y <= scratch.y[row_index] + incr2_margin &&
                    position.x >= scratch.xmin[row_index] - incr2_margin &&
                    position.x <= scratch.xmax[row_index] + incr2_margin) {
                    if (diagnostics != nullptr) {
                        ++diagnostics->overlaps;
                    }
                    if (other < image_index) {
                        // Phase 3: Use component ID instead of seed index to prevent double-subtraction
                        int other_component_id = seed_to_component_id[other];
                        if (other_component_id != -1 &&
                            subtracted_component_ids.find(other_component_id) == subtracted_component_ids.end()) {
                            area -= areaimage[other];
                            subtracted_component_ids.insert(other_component_id);
                            // Phase 6: Also track in component union calculation
                            component_union_area -= areaimage[other];
                        }
                    } else {
                        overlap[other] = 1;
                        // Phase 2: Mark future seed as part of this component
                        if (seed_to_component_id[other] == -1) {
                            seed_to_component_id[other] = seed_to_component_id[image_index];
                        }
                    }
                    break;
                }
            }
        }
    }

    const double scale =
        source_flux(source_radius, settings) / (source_radius * source_radius) * nbin * nbin;
    const double magnification = area / scale;
    if (diagnostics != nullptr) {
        diagnostics->estimated_error =
            legacy_area_error_indicator(*diagnostics, source_radius, settings);
        const double floor_coefficient =
            high_magnification_floor_coefficient(*diagnostics, magnification, source_radius);
        if (floor_coefficient > 0.0) {
            const double high_magnification_floor =
                std::abs(magnification) *
                (floor_coefficient / static_cast<double>(std::max(settings.source_bins, 1)));
            diagnostics->estimated_error =
                std::max(diagnostics->estimated_error, high_magnification_floor);
        }
        if (std::getenv("LCBININT_AREA_DIAGNOSTICS")) {
            std::fprintf(stderr,
                "AREA_DIAGNOSTICS bins=%d seeds=%d processed=%d fold=%d rows=%d gaps=%d overlaps=%d "
                "maxjump=%.3g mag=%.8g err=%.8g\n",
                settings.source_bins, diagnostics->seed_count, diagnostics->processed_images,
                diagnostics->fold_seed_count, diagnostics->boundary_rows, diagnostics->gap_repairs,
                diagnostics->overlaps, diagnostics->max_jump_cells, magnification,
                diagnostics->estimated_error);
        }
    }
    return magnification;
}

FiniteSourceResult fixed_inverse_ray_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    FiniteSourceDecision decision,
    double caustic_distance = std::numeric_limits<double>::infinity(),
    double consistency_reference = std::numeric_limits<double>::quiet_NaN())
{
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    auto seeds = legacy_augmented_image_seeds(
        point_magnifier, mapper, separation, mass_ratio, source, source_radius,
        caustic_distance);
    // Phase 3: find caustic crossings that fall in the gap between the last
    // phase sample and phi=2*pi (missed by uniform 1400-point sampling).
    if (finite_magnifier != nullptr) {
        finite_magnifier->legacy_augment_seeds_from_branches(
            separation, mass_ratio, source, source_radius, seeds);
    }
    double magnification;
    if (decision.method == FiniteSourceMethod::inverse_ray_polar) {
        magnification = inverse_ray_polar_boundary_binary(
            point_magnifier, separation, mass_ratio, source, source_radius,
            settings, finite_magnifier, nullptr, &seeds);
        if (!std::isfinite(magnification)) {
            return {magnification, 0, decision, std::nan(""), 0, false};
        }
        return {magnification, 0, decision, 0.0, 0, true};
    } else {
        LegacyAreaDiagnostics diagnostics;
        magnification = legacy_imagearea4_binary(
            point_magnifier, separation, mass_ratio, source, source_radius,
            settings, finite_magnifier, &seeds, &diagnostics);
        if (!std::isfinite(magnification)) {
            return {magnification, 0, decision, std::nan(""), 0, false};
        }

        double error_estimate = diagnostics.estimated_error;
        int refinement_level = 0;
        bool converged = true;
        const bool adaptive =
            settings.adaptive_source_bins != 0 &&
            (settings.finite_source_tol > 0.0 || settings.finite_source_reltol > 0.0);
        const int max_bins = std::max(settings.source_bins, settings.max_source_bins);
        auto target_error = [&](double mag) {
            return settings.finite_source_tol +
                   settings.finite_source_reltol * std::max(std::abs(mag), 1.0);
        };
        auto consistency_error = [&](double) {
            return 0.0;
        };
        // Phase 3: Error floor - minimum precision achievable at given bins
        // Integration error from grid spacing: incr ~ rho/bins, so error ~ (rho/bins)^2
        auto error_floor_from_bins = [&](int b) {
            double grid_spacing = source_radius / std::max(b, 1);
            // Conservative estimate: 1% floor per magnitude decade
            return std::max(0.01 * std::abs(magnification), grid_spacing * grid_spacing);
        };
        // Phase 4: Predict bins needed to reach target tolerance
        auto predicted_bins_for_target = [&](double target, int current_bins,
                                             const std::vector<int>& bin_history,
                                             const std::vector<double>& error_history) {
            if (bin_history.size() < 2 || error_history.size() < 2) {
                return 0;  // Not enough data
            }
            // Fit: error ~ c / bins^alpha
            // Use last two points to estimate alpha
            double b0 = bin_history[bin_history.size() - 2];
            double b1 = bin_history[bin_history.size() - 1];
            double e0 = error_history[error_history.size() - 2];
            double e1 = error_history[error_history.size() - 1];
            if (e0 <= 0.0 || e1 <= 0.0 || b0 == b1) {
                return 0;
            }
            double error_ratio = e0 / e1;
            double bins_ratio = static_cast<double>(b1) / static_cast<double>(b0);
            double alpha = std::log(error_ratio) / std::log(bins_ratio);
            // Clamp alpha to reasonable range [1.5, 2.5]
            alpha = std::max(1.5, std::min(2.5, alpha));
            // Solve: target = e1 * (b / b1)^(-alpha)
            // b = b1 * (e1 / target)^(1/alpha)
            if (target <= 0.0) {
                return 0;
            }
            int predicted = static_cast<int>(
                std::ceil(b1 * std::pow(e1 / target, 1.0 / alpha)));
            return std::max(current_bins, predicted);
        };
        auto required_bins_from_high_magnification_floor =
            [&](const LegacyAreaDiagnostics& current_diagnostics, double mag) {
                const double floor_coefficient =
                    high_magnification_floor_coefficient(
                        current_diagnostics, mag, source_radius);
                const double target = target_error(mag);
                if (floor_coefficient <= 0.0 || target <= 0.0 || !std::isfinite(target)) {
                    return 0.0;
                }
                return floor_coefficient * std::abs(mag) / target;
        };
        if (adaptive) {
            FiniteSourceSettings refined_settings = settings;
            std::vector<double> refinement_history;
            std::vector<int> bin_history;
            std::vector<double> error_history;
            refinement_history.push_back(magnification);
            bin_history.push_back(refined_settings.source_bins);
            error_history.push_back(error_estimate);
            double required_bins = required_bins_from_high_magnification_floor(
                diagnostics, magnification);
            auto allow_overbudget_refinement = [&]() {
                return source_radius < 1.0e-2 && std::abs(magnification) <= 1000.0;
            };
            if (required_bins > static_cast<double>(max_bins)) {
                error_estimate = std::max(
                    error_estimate,
                    target_error(magnification) * required_bins / static_cast<double>(max_bins));
            }
            while (std::max(error_estimate, consistency_error(magnification)) >
                       target_error(magnification) &&
                   refined_settings.source_bins < max_bins &&
                   (required_bins <= static_cast<double>(max_bins) ||
                    allow_overbudget_refinement())) {
                // Phase 4: Predictive refinement - estimate next bins directly
                int next_bins_predicted = predicted_bins_for_target(
                    target_error(magnification), refined_settings.source_bins,
                    bin_history, error_history);
                // Use prediction if available, otherwise fall back to incremental (×2)
                int next_bins = (next_bins_predicted > refined_settings.source_bins) ?
                    std::min(max_bins, next_bins_predicted) :
                    std::min(max_bins, std::max(refined_settings.source_bins + 1,
                                                refined_settings.source_bins * 2));
                refined_settings.source_bins = next_bins;

                LegacyAreaDiagnostics refined_diagnostics;
                const double refined_magnification = legacy_imagearea4_binary(
                    point_magnifier, separation, mass_ratio, source, source_radius,
                    refined_settings, finite_magnifier, &seeds, &refined_diagnostics);
                if (!std::isfinite(refined_magnification)) {
                    return {refined_magnification, 0, decision, std::nan(""), refinement_level + 1, false};
                }
                double min_history_change = std::numeric_limits<double>::infinity();
                for (const double prev_mag : refinement_history) {
                    min_history_change = std::min(
                        min_history_change,
                        std::abs(refined_magnification - prev_mag));
                }
                refinement_history.push_back(refined_magnification);
                bin_history.push_back(refined_settings.source_bins);
                magnification = refined_magnification;
                const double target = target_error(refined_magnification);
                // Phase 3: Check if current bins meets error floor requirement
                double floor = error_floor_from_bins(refined_settings.source_bins);
                // Require at least two refinements (size >= 3 after push) before
                // trusting self-consistency alone. At the first step (size == 2)
                // only 50 bins and 100 bins have been computed; those two can agree
                // on a wrong answer if seeding is unstable (e.g. the grazing-caustic
                // probe seeds in commit 49b6fbe) or if the source is so small that
                // both levels are below the resolution needed to detect the error.
                // The min-over-all-history indicator with safety factor 0.5 tolerates
                // non-monotone convergence (e.g. 100-bin outlier bracketed by 50 and
                // 200 bins) while still catching slow-convergence false positives.
                const bool self_consistent =
                    refinement_history.size() >= 3 &&
                    min_history_change <= 0.5 * target;
                if (self_consistent) {
                    error_estimate = std::max(
                        consistency_error(refined_magnification),
                        min_history_change);
                } else {
                    error_estimate = std::max(
                        std::max({refined_diagnostics.estimated_error,
                                 consistency_error(refined_magnification),
                                 floor}),
                        min_history_change);
                }
                error_history.push_back(error_estimate);
                required_bins = required_bins_from_high_magnification_floor(
                    refined_diagnostics, refined_magnification);
                if (required_bins > static_cast<double>(max_bins)) {
                    error_estimate = std::max(
                        error_estimate,
                        target_error(refined_magnification) *
                            required_bins / static_cast<double>(max_bins));
                }
                ++refinement_level;
            }
            error_estimate = std::max(error_estimate, consistency_error(magnification));
            converged = error_estimate <= target_error(magnification);
            if (refinement_level > 0) {
                decision.reason = "cartesian inverse-ray with adaptive source bins";
            }
        }
        return {magnification, 0, decision, error_estimate, refinement_level, converged};
    }
}

// ---------- image-spine kernel (finite_mode = 3) ----------
// Ported from lcbinint-idea-stable commit eb2c08e.
// Timing instrumentation omitted; timing pointers in stable are replaced by
// passing nullptr at all call sites.

constexpr double kLocal7LambdaMin = 1.0e-5;
constexpr double kLocal7SpineAreaJacMin = 100.0;
constexpr double kLocal7SpineDetMax = 1.0e-2;
constexpr double kLocal7SpineMaxStepCells = 4.0;
constexpr double kLocal7SpineMinStepCells = 0.125;
constexpr int kLocal7SpineMaxPoints = 200000;
constexpr int kLocal7SpineMaxNormalSamples = 20000000;
constexpr int kLocal7SpineOutsideStop = 3;
constexpr double kLocal7SpineCurvatureMax = 0.55;
constexpr double kLocal7SpineNormalSubstep = 1.0;
constexpr double kLocal7SpineTargetTolCells = 2.0;
constexpr double kLocal7SpineFrameDetMin = 1.0e-9;
constexpr double kLocal7SpineFrameLambdaMin = 1.0e-9;
constexpr double kLocal7SpineFrameAreaJacMax = 1.0e12;
constexpr double kLocal7SpinePairDistanceCells = 50000.0;
constexpr double kLocal7SpineMaxRelativeArea = 1.0e8;
// var_ratio = 2*beta*rho / (lambda_s * |lambda_l|) measures how much lambda_l
// changes across the source disk.  When this exceeds ~2 the linear fold model
// breaks down and the spine gives errors of several percent.  Skip to cartesian.
constexpr double kLocal7SpineMaxVarRatio = 2.0;

struct Local7Frame {
    Complex za;
    SourcePosition wa;
    SourcePosition sa;
    double gamma_r = 0.0;
    double gamma_i = 0.0;
    double beta_r = 0.0;
    double beta_i = 0.0;
    double kappa_r = 0.0;
    double kappa_i = 0.0;
    double lambda_l = 0.0;
    double lambda_s = 0.0;
    double det_j = 0.0;
    double area_jac = 0.0;
    double e_lx = 0.0;
    double e_ly = 0.0;
    double e_sx = 0.0;
    double e_sy = 0.0;
    bool ok = false;
};

bool local7_derivatives_binary(
    Complex z,
    const BinaryLensMapper& mapper,
    double* gr,
    double* gi,
    double* br,
    double* bi,
    double* kr,
    double* ki)
{
    *gr = *gi = *br = *bi = *kr = *ki = 0.0;
    const Complex lenses[2] = {mapper.separation, Complex(0.0, 0.0)};
    const double masses[2] = {mapper.m1, mapper.m2};
    for (int i = 0; i < 2; ++i) {
        const double dx = lenses[i].real() - z.real();
        const double dy = z.imag() - lenses[i].imag();
        const double r2 = dx * dx + dy * dy;
        if (r2 <= 1.0e-30 || !std::isfinite(r2)) return false;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r8 = r4 * r4;
        const double dx2 = dx * dx, dy2 = dy * dy;
        const double dx3 = dx2 * dx, dy3 = dy2 * dy;
        const double dx4 = dx2 * dx2, dy4 = dy2 * dy2;
        const double mass = masses[i];
        *gr += mass * (dx2 - dy2) / r4;
        *gi += mass * (-2.0 * dx * dy) / r4;
        *br += 2.0 * mass * dx * (dx2 - 3.0 * dy2) / r6;
        *bi += 2.0 * mass * (dy3 - 3.0 * dx2 * dy) / r6;
        *kr += 6.0 * mass * (dx4 - 6.0 * dx2 * dy2 + dy4) / r8;
        *ki += 6.0 * mass * (-4.0 * dx3 * dy + 4.0 * dx * dy3) / r8;
    }
    return std::isfinite(*gr) && std::isfinite(*gi) && std::isfinite(*br) &&
           std::isfinite(*bi) && std::isfinite(*kr) && std::isfinite(*ki);
}

bool local7_make_frame(
    Complex za,
    SourcePosition source,
    const BinaryLensMapper& mapper,
    Local7Frame* frame)
{
    double gr, gi, br, bi, kr, ki;
    if (!local7_derivatives_binary(za, mapper, &gr, &gi, &br, &bi, &kr, &ki)) return false;
    const double g = std::hypot(gr, gi);
    const double lambda_s = 1.0 + g;
    const double lambda_l = 1.0 - g;
    const double det_j = lambda_l * lambda_s;
    const double abs_det = std::abs(det_j);
    if (abs_det <= 0.0 || !std::isfinite(abs_det)) return false;
    const double phi = 0.5 * std::atan2(gi, gr);
    const SourcePosition wa = map_binary_lens_real(mapper, za.real(), za.imag());
    *frame = {};
    frame->za = za;
    frame->wa = wa;
    frame->sa = {wa.x - source.x, wa.y - source.y};
    frame->gamma_r = gr; frame->gamma_i = gi;
    frame->beta_r = br;  frame->beta_i = bi;
    frame->kappa_r = kr; frame->kappa_i = ki;
    frame->lambda_l = lambda_l;
    frame->lambda_s = lambda_s;
    frame->det_j = det_j;
    frame->area_jac = 1.0 / abs_det;
    frame->e_lx = -std::sin(phi);
    frame->e_ly =  std::cos(phi);
    frame->e_sx =  std::cos(phi);
    frame->e_sy =  std::sin(phi);
    frame->ok = std::isfinite(frame->area_jac);
    return frame->ok;
}

Complex local7_apply_inverse_linear(const Local7Frame& frame, double sx, double sy)
{
    const double dx = sx - frame.sa.x;
    const double dy = sy - frame.sa.y;
    const double xi  = dx * frame.e_lx + dy * frame.e_ly;
    const double eta = dx * frame.e_sx + dy * frame.e_sy;
    return {
        (xi / frame.lambda_l) * frame.e_lx + (eta / frame.lambda_s) * frame.e_sx,
        (xi / frame.lambda_l) * frame.e_ly + (eta / frame.lambda_s) * frame.e_sy,
    };
}

Complex local7_apply_inverse_jacobian(const Local7Frame& frame, Complex residual)
{
    const double xi  = residual.real() * frame.e_lx + residual.imag() * frame.e_ly;
    const double eta = residual.real() * frame.e_sx + residual.imag() * frame.e_sy;
    return {
        (xi / frame.lambda_l) * frame.e_lx + (eta / frame.lambda_s) * frame.e_sx,
        (xi / frame.lambda_l) * frame.e_ly + (eta / frame.lambda_s) * frame.e_sy,
    };
}

Complex local7_correct_quadratic(const Local7Frame& frame, Complex dz0)
{
    const Complex beta(frame.beta_r, frame.beta_i);
    const Complex residual = 0.5 * beta * std::conj(dz0) * std::conj(dz0);
    return dz0 - local7_apply_inverse_jacobian(frame, residual);
}

Complex local7_approx_image(const Local7Frame& frame, double sx, double sy)
{
    const Complex dz0 = local7_apply_inverse_linear(frame, sx, sy);
    return frame.za + local7_correct_quadratic(frame, dz0);
}

struct Local7SpinePoint {
    SourcePosition image;
    SourcePosition source_offset;
    Local7Frame frame;
    double half_weight = 0.0;
};

struct Local7SpineEligibility {
    bool ok = false;
    std::size_t pair_index = 0;
    int reason = 0;
};

bool local7_is_spine_candidate(const Local7Frame& frame)
{
    return frame.ok &&
           std::isfinite(frame.area_jac) &&
           frame.area_jac >= kLocal7SpineAreaJacMin &&
           std::abs(frame.det_j) <= kLocal7SpineDetMax;
}

Local7SpineEligibility local7_spine_eligibility(
    const std::vector<SourcePosition>& seeds,
    const std::vector<int>& overlap,
    std::size_t image_index,
    const Local7Frame& frame,
    SourcePosition source,
    double source_step,
    const BinaryLensMapper& mapper,
    int caustic_born_branches)
{
    if (caustic_born_branches <= 0) return {false, 0, 30};
    if (!local7_is_spine_candidate(frame)) return {false, 0, 31};

    const auto nearest_partner = [&](std::size_t from_index, const Local7Frame& from_frame) {
        double best_distance = std::numeric_limits<double>::infinity();
        std::size_t best_index = seeds.size();
        int candidate_count = 0;
        for (std::size_t other = 0; other < seeds.size(); ++other) {
            if (other == from_index || overlap[other] == 1) continue;
            Local7Frame other_frame;
            if (!local7_make_frame(Complex(seeds[other].x, seeds[other].y), source, mapper, &other_frame) ||
                !local7_is_spine_candidate(other_frame) ||
                std::signbit(other_frame.det_j) == std::signbit(from_frame.det_j)) {
                continue;
            }
            const double image_distance =
                std::hypot(seeds[other].x - seeds[from_index].x, seeds[other].y - seeds[from_index].y);
            if (image_distance > kLocal7SpinePairDistanceCells * source_step) continue;
            ++candidate_count;
            if (image_distance < best_distance) {
                best_distance = image_distance;
                best_index = other;
            }
        }
        return std::pair<std::size_t, int> {best_index, candidate_count};
    };

    const auto [partner_index, partner_count] = nearest_partner(image_index, frame);
    if (partner_count < 1 || partner_index >= seeds.size()) return {false, 0, 33};
    if (partner_count > 1) {
        Local7Frame partner_frame;
        if (!local7_make_frame(Complex(seeds[partner_index].x, seeds[partner_index].y), source, mapper, &partner_frame))
            return {false, 0, 34};
        const auto [mutual_index, mutual_count] = nearest_partner(partner_index, partner_frame);
        if (mutual_count < 1 || mutual_index != image_index) return {false, 0, 35};
    }
    if (partner_index < image_index) return {false, 0, 36};
    return {true, partner_index, 0};
}

double local7_spine_step(const Local7Frame& frame, double source_step, double source_radius)
{
    const double abs_lambda = std::max(std::abs(frame.lambda_l), kLocal7LambdaMin);
    double step = source_step / abs_lambda;
    const double beta_abs = std::hypot(frame.beta_r, frame.beta_i);
    if (beta_abs > 0.0 && std::isfinite(beta_abs)) {
        const double nonlinear_cap =
            2.0 * source_radius / (abs_lambda + std::sqrt(abs_lambda * abs_lambda + 2.0 * beta_abs * source_radius));
        if (std::isfinite(nonlinear_cap) && nonlinear_cap > 0.0) step = std::min(step, nonlinear_cap);
    }
    step = std::min(step, kLocal7SpineMaxStepCells * source_step);
    step = std::max(step, kLocal7SpineMinStepCells * source_step);
    return step;
}

bool local7_spine_frame_safe(const Local7Frame& frame)
{
    return frame.ok && std::isfinite(frame.area_jac) &&
           std::abs(frame.lambda_l) >= kLocal7SpineFrameLambdaMin &&
           std::abs(frame.lambda_s) >= kLocal7SpineFrameLambdaMin &&
           std::abs(frame.det_j) >= kLocal7SpineFrameDetMin &&
           frame.area_jac <= kLocal7SpineFrameAreaJacMax;
}

bool local7_spine_try_step(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    double source_step,
    SourcePosition current,
    SourcePosition current_source_offset,
    const Local7Frame& current_frame,
    double signed_step,
    Local7SpinePoint* output,
    int* fail_reason)
{
    double step = signed_step;
    const double min_abs_step = kLocal7SpineMinStepCells * source_step;
    int last_reason = 25;
    for (int attempt = 0; attempt < 12; ++attempt) {
        const SourcePosition candidate {
            current.x + step * current_frame.e_lx,
            current.y + step * current_frame.e_ly,
        };
        SourcePosition target_offset {
            current_source_offset.x + current_frame.lambda_l * step * current_frame.e_lx,
            current_source_offset.y + current_frame.lambda_l * step * current_frame.e_ly,
        };
        if (target_offset.x * target_offset.x + target_offset.y * target_offset.y >
            source_radius * source_radius) {
            last_reason = 20;
            step *= 0.5;
            if (std::abs(step) < min_abs_step) break;
            continue;
        }
        SourcePosition corrected = candidate;
        Local7Frame candidate_frame;
        bool frame_ok = false;
        double residual_norm = std::numeric_limits<double>::infinity();
        for (int newton = 0; newton < 5; ++newton) {
            const SourcePosition mapped = map_binary_lens_real(mapper, corrected.x, corrected.y);
            frame_ok = local7_make_frame(Complex(corrected.x, corrected.y), source, mapper, &candidate_frame) &&
                       local7_spine_frame_safe(candidate_frame);
            if (!frame_ok) {
                last_reason = newton == 0 ? 21 : 22;
                break;
            }
            const Complex residual(
                mapped.x - (source.x + target_offset.x),
                mapped.y - (source.y + target_offset.y));
            residual_norm = std::abs(residual);
            if (residual_norm <= kLocal7SpineTargetTolCells * source_step) break;
            const Complex dz = local7_apply_inverse_jacobian(candidate_frame, residual);
            double damping = 1.0;
            const double dz_abs = std::abs(dz);
            const double max_dz = 4.0 * std::max(std::abs(step), source_step);
            if (dz_abs > max_dz && dz_abs > 0.0) damping = max_dz / dz_abs;
            corrected.x -= damping * dz.real();
            corrected.y -= damping * dz.imag();
        }
        if (frame_ok) {
            const double dot =
                current_frame.e_lx * candidate_frame.e_lx + current_frame.e_ly * candidate_frame.e_ly;
            if (residual_norm <= kLocal7SpineTargetTolCells * source_step &&
                std::abs(dot) >= std::cos(kLocal7SpineCurvatureMax)) {
                *output = {corrected, target_offset, candidate_frame, 0.0};
                return true;
            }
            last_reason = residual_norm > kLocal7SpineTargetTolCells * source_step ? 23 : 24;
        }
        step *= 0.5;
        if (std::abs(step) < min_abs_step) break;
    }
    if (fail_reason != nullptr) *fail_reason = last_reason;
    return false;
}

bool local7_build_spine_direction(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    double source_step,
    const Local7SpinePoint& seed,
    double sign,
    std::vector<Local7SpinePoint>* points,
    int* fail_reason)
{
    SourcePosition current = seed.image;
    SourcePosition current_source_offset = seed.source_offset;
    Local7Frame frame = seed.frame;
    int guard = 0;
    while (++guard < kLocal7SpineMaxPoints) {
        const double step = sign * local7_spine_step(frame, source_step, source_radius);
        Local7SpinePoint next;
        if (!local7_spine_try_step(
                mapper, source, source_radius, source_step,
                current, current_source_offset, frame, step, &next, fail_reason)) {
            return true;
        }
        points->push_back(next);
        current = next.image;
        current_source_offset = next.source_offset;
        frame = next.frame;
        if (static_cast<int>(points->size()) > kLocal7SpineMaxPoints) return false;
    }
    return false;
}

double local7_spine_integrate_normals(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<Local7SpinePoint>& spine,
    double source_step,
    int* fallback_reason)
{
    const bool spine_debug = std::getenv("LCBININT_SPINE_DEBUG") != nullptr;
    const double radius2 = source_radius * source_radius;
    const double inv_radius2 = 1.0 / radius2;
    double area = 0.0;
    long long normal_samples = 0;
    for (std::size_t i = 0; i < spine.size(); ++i) {
        const auto& point = spine[i];
        const double tangent_weight = 2.0 * point.half_weight;
        if (!std::isfinite(tangent_weight) || tangent_weight <= 0.0) {
            *fallback_reason = 10;
            return std::nan("");
        }
        const double normal_step = std::min(
            std::max(source_step / std::max(std::abs(point.frame.lambda_s), kLocal7LambdaMin),
                kLocal7SpineMinStepCells * source_step),
            kLocal7SpineMaxStepCells * source_step) * kLocal7SpineNormalSubstep;
        const double cell_area = tangent_weight * normal_step;
        if (!std::isfinite(cell_area) || cell_area <= 0.0) {
            *fallback_reason = 11;
            return std::nan("");
        }
        for (int direction = -1; direction <= 1; direction += 2) {
            int outside = 0;
            for (int n = direction == -1 ? -1 : 0; ; n += direction) {
                const double offset = static_cast<double>(n) * normal_step;
                const SourcePosition image {
                    point.image.x + offset * point.frame.e_sx,
                    point.image.y + offset * point.frame.e_sy,
                };
                const SourcePosition mapped = map_binary_lens_real(mapper, image.x, image.y);
                ++normal_samples;
                if (normal_samples > kLocal7SpineMaxNormalSamples) {
                    *fallback_reason = 12;
                    return std::nan("");
                }
                const double dx = mapped.x - source.x;
                const double dy = mapped.y - source.y;
                const double q = (dx * dx + dy * dy) * inv_radius2;
                if (q <= 1.0) {
                    outside = 0;
                    area += legacy_limb_brightness(q, settings, finite_magnifier) * cell_area;
                } else {
                    ++outside;
                    if (outside >= kLocal7SpineOutsideStop) break;
                }
                if (std::abs(offset) >
                    4.0 * source_radius / std::max(std::abs(point.frame.lambda_s), kLocal7LambdaMin)) {
                    *fallback_reason = 13;
                    return std::nan("");
                }
            }
        }
    }
    if (spine_debug && !spine.empty()) {
        const double tw0 = 2.0 * spine[0].half_weight;
        const double tw_mid = 2.0 * spine[spine.size()/2].half_weight;
        const double ll0 = spine[0].frame.lambda_l;
        const double ll_mid = spine[spine.size()/2].frame.lambda_l;
        std::fprintf(stderr, "  normals: spine_pts=%zu normal_samples=%lld "
            "tw[0]=%.3e tw[mid]=%.3e ll[0]=%.3e ll[mid]=%.3e source_step=%.3e\n",
            spine.size(), normal_samples, tw0, tw_mid, ll0, ll_mid, source_step);
    }
    return area;
}

double local7_spine_area_binary(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const Local7Frame& seed_frame,
    int* fallback_reason)
{
    *fallback_reason = 0;
    if (!local7_spine_frame_safe(seed_frame) || !local7_is_spine_candidate(seed_frame)) {
        *fallback_reason = 1;
        return std::nan("");
    }
    const int bins = std::max(settings.source_bins, 1);
    const double source_step = source_radius / static_cast<double>(bins);
    std::vector<Local7SpinePoint> minus_points, plus_points;
    minus_points.reserve(1024);
    plus_points.reserve(1024);
    const Local7SpinePoint seed {
        {seed_frame.za.real(), seed_frame.za.imag()},
        seed_frame.sa,
        seed_frame,
        0.0,
    };
    if (!local7_build_spine_direction(
            mapper, source, source_radius, source_step, seed, -1.0, &minus_points, fallback_reason) ||
        !local7_build_spine_direction(
            mapper, source, source_radius, source_step, seed, 1.0, &plus_points, fallback_reason)) {
        *fallback_reason = 2;
        return std::nan("");
    }
    std::vector<Local7SpinePoint> spine;
    spine.reserve(minus_points.size() + plus_points.size() + 1);
    for (auto it = minus_points.rbegin(); it != minus_points.rend(); ++it) spine.push_back(*it);
    spine.push_back(seed);
    for (const auto& pt : plus_points) spine.push_back(pt);
    if (spine.size() < 3) {
        if (*fallback_reason == 0) *fallback_reason = 3;
        return std::nan("");
    }
    for (std::size_t i = 0; i < spine.size(); ++i) {
        double left_cross = 0.0, right_cross = 0.0;
        double left_source = 0.0, right_source = 0.0;
        if (i > 0) {
            const double dx = spine[i].image.x - spine[i - 1].image.x;
            const double dy = spine[i].image.y - spine[i - 1].image.y;
            left_cross = std::abs(dx * spine[i].frame.e_sy - dy * spine[i].frame.e_sx);
            left_source = std::hypot(
                spine[i].source_offset.x - spine[i - 1].source_offset.x,
                spine[i].source_offset.y - spine[i - 1].source_offset.y);
        }
        if (i + 1 < spine.size()) {
            const double dx = spine[i + 1].image.x - spine[i].image.x;
            const double dy = spine[i + 1].image.y - spine[i].image.y;
            right_cross = std::abs(dx * spine[i].frame.e_sy - dy * spine[i].frame.e_sx);
            right_source = std::hypot(
                spine[i + 1].source_offset.x - spine[i].source_offset.x,
                spine[i + 1].source_offset.y - spine[i].source_offset.y);
        }
        if (i == 0) { left_cross = right_cross; left_source = right_source; }
        else if (i + 1 == spine.size()) { right_cross = left_cross; right_source = left_source; }
        spine[i].half_weight = 0.25 * (left_cross + right_cross);
        if (!std::isfinite(spine[i].half_weight) || spine[i].half_weight <= 0.0 ||
            left_source > kLocal7SpineMaxStepCells * source_step * 2.0 ||
            right_source > kLocal7SpineMaxStepCells * source_step * 2.0) {
            *fallback_reason = 4;
            return std::nan("");
        }
    }
    const double area = local7_spine_integrate_normals(
        mapper, source, source_radius, settings, finite_magnifier, spine, source_step, fallback_reason);
    const double total_source = source_flux(source_radius, settings);
    if (!std::isfinite(area) || area <= 0.0 ||
        !std::isfinite(total_source) || total_source <= 0.0 ||
        area / total_source > kLocal7SpineMaxRelativeArea) {
        if (*fallback_reason == 0) *fallback_reason = 5;
        return std::nan("");
    }
    return area;
}

FiniteSourceResult fixed_inverse_ray_spine_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier)
{
    FiniteSourceDecision decision {
        FiniteSourceMethod::inverse_ray_spine,
        estimate_cartesian_cost(settings),
        "experimental image-spine guided scan",
    };

    // Generate seeds once. All fallback paths reuse these to avoid a second
    // call to legacy_augmented_image_seeds.
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    auto seeds = legacy_augmented_image_seeds(
        point_magnifier, mapper, separation, mass_ratio, source, source_radius);
    if (seeds.size() < 5 && finite_magnifier != nullptr) {
        finite_magnifier->legacy_augment_seeds_from_branches(
            separation, mass_ratio, source, source_radius, seeds);
    }

    const int n_ps = static_cast<int>(
        point_magnifier.binary_images(separation, mass_ratio, source).size());
    const int caustic_born = std::max(0, static_cast<int>(seeds.size()) - n_ps);
    const int bins = std::max(settings.source_bins, 1);
    const double source_step = source_radius / static_cast<double>(bins);

    // Boundary seeding may add many samples along the same extended fold arc.
    // The local spine kernel assumes isolated fold pairs; applying it to many
    // same-arc seeds double-counts the image area.  Use the robust cartesian
    // kernel for these multi-component / oversampled caustic crossings.
    if (caustic_born > 4) {
        FiniteSourceDecision fb {
            FiniteSourceMethod::inverse_ray_cartesian,
            estimate_cartesian_cost(settings),
            "spine skipped for multi-seed caustic crossing",
        };
        const double mag = legacy_imagearea4_binary(
            point_magnifier, separation, mass_ratio, source, source_radius,
            settings, finite_magnifier, &seeds);
        if (!std::isfinite(mag)) return {mag, 0, fb, std::nan(""), 0, false};
        return {mag, 0, fb, 0.0, 0, true};
    }

    if (const char* dbg = std::getenv("LCBININT_SPINE_DEBUG")) {
        (void)dbg;
        std::fprintf(stderr, "SPINE_DEBUG seeds=%zu n_ps=%d caustic_born=%d rho=%.4e bins=%d\n",
            seeds.size(), n_ps, caustic_born, source_radius, bins);
        for (std::size_t di = 0; di < seeds.size(); ++di) {
            Local7Frame df;
            bool dok = local7_make_frame(Complex(seeds[di].x, seeds[di].y), source, mapper, &df);
            const double beta_abs = dok ? std::hypot(df.beta_r, df.beta_i) : 0.0;
        const double var_ratio = (dok && std::abs(df.lambda_l) > 1e-30 && std::abs(df.lambda_s) > 0.0)
            ? 2.0 * beta_abs * source_radius / (std::abs(df.lambda_s) * std::abs(df.lambda_l)) : 0.0;
        std::fprintf(stderr, "  seed[%zu] (%.4f,%.4f) frame_ok=%d area_jac=%.2f det_j=%.4f beta=%.4f var_ratio=%.2f candidate=%d\n",
                di, seeds[di].x, seeds[di].y, (int)dok,
                dok ? df.area_jac : 0.0, dok ? df.det_j : 0.0,
                beta_abs, var_ratio, dok && local7_is_spine_candidate(df));
        }
    }

    // Pass 1: try spine for caustic-born fold pairs.
    // Non-eligible seeds (PS images) are left untouched; spine-covered seeds
    // are marked so they are excluded from the cartesian pass.
    std::vector<bool> spine_covered(seeds.size(), false);
    std::vector<int> elig_overlap(seeds.size(), 0);
    double spine_area = 0.0;
    bool any_spine_tried = false;

    for (std::size_t i = 0; i < seeds.size(); ++i) {
        if (elig_overlap[i] == 1) continue;

        Local7Frame frame;
        if (!local7_make_frame(Complex(seeds[i].x, seeds[i].y), source, mapper, &frame) ||
            std::abs(frame.lambda_l) < kLocal7LambdaMin ||
            std::abs(frame.lambda_s) < kLocal7LambdaMin ||
            !std::isfinite(frame.area_jac)) {
            continue; // not a usable frame; handle this seed via cartesian
        }

        const auto elig = local7_spine_eligibility(
            seeds, elig_overlap, i, frame, source, source_step, mapper, caustic_born);
        if (!elig.ok) continue; // not eligible; handle this seed via cartesian

        // Skip when var_ratio is too large: the linear fold model breaks down
        // and spine error exceeds a few percent.
        {
            const double beta_abs = std::hypot(frame.beta_r, frame.beta_i);
            const double abs_ll = std::abs(frame.lambda_l);
            const double abs_ls = std::abs(frame.lambda_s);
            if (abs_ll > 0.0 && abs_ls > 0.0) {
                const double var_ratio = 2.0 * beta_abs * source_radius / (abs_ls * abs_ll);
                if (var_ratio > kLocal7SpineMaxVarRatio) continue;
            }
        }

        any_spine_tried = true;
        int fallback_reason = 0;
        const double area = local7_spine_area_binary(
            mapper, source, source_radius, settings, finite_magnifier, frame, &fallback_reason);
        if (!std::isfinite(area) || area <= 0.0) {
            // Spine failed for an eligible pair. Fall back to full cartesian
            // using the already-computed seeds (no second seed generation).
            FiniteSourceDecision fb {
                FiniteSourceMethod::inverse_ray_cartesian,
                estimate_cartesian_cost(settings),
                "spine integration failed; cartesian fallback",
            };
            const double mag = legacy_imagearea4_binary(
                point_magnifier, separation, mass_ratio, source, source_radius,
                settings, finite_magnifier, &seeds);
            if (!std::isfinite(mag)) return {mag, 0, fb, std::nan(""), 0, false};
            return {mag, 0, fb, 0.0, 0, true};
        }

        spine_area += area;
        spine_covered[i] = true;
        spine_covered[elig.pair_index] = true;
        elig_overlap[i] = 1;
        elig_overlap[elig.pair_index] = 1;
        if (std::getenv("LCBININT_SPINE_DEBUG")) {
            std::fprintf(stderr, "  spine pair (%zu,%zu) area=%.6f\n", i, elig.pair_index, area);
        }
    }

    // Pass 2: cartesian for seeds not handled by spine.
    std::vector<SourcePosition> cartesian_seeds;
    cartesian_seeds.reserve(seeds.size());
    for (std::size_t i = 0; i < seeds.size(); ++i) {
        if (!spine_covered[i]) cartesian_seeds.push_back(seeds[i]);
    }

    const double total_source = source_flux(source_radius, settings);
    if (!std::isfinite(total_source) || total_source <= 0.0) {
        return {std::nan(""), 0, decision, std::nan(""), 0, false};
    }
    double total_mag = spine_area / total_source;

    if (!cartesian_seeds.empty()) {
        const double cart_mag = legacy_imagearea4_binary(
            point_magnifier, separation, mass_ratio, source, source_radius,
            settings, finite_magnifier, &cartesian_seeds);
        if (!std::isfinite(cart_mag)) {
            return {cart_mag, 0, decision, std::nan(""), 0, false};
        }
        if (std::getenv("LCBININT_SPINE_DEBUG")) {
            std::fprintf(stderr, "  spine_mag=%.4f cart_mag=%.4f total=%.4f (source_flux=%.6f)\n",
                spine_area/total_source, cart_mag, spine_area/total_source+cart_mag, total_source);
        }
        total_mag += cart_mag;
    }

    if (!any_spine_tried) {
        // No seed was eligible for spine; result is pure cartesian.
        decision.method = FiniteSourceMethod::inverse_ray_cartesian;
        decision.reason = "spine not eligible; cartesian";
    }

    return {total_mag, 0, decision, 0.0, 0, true};
}

} // namespace

FiniteSourceMagnifier::FiniteSourceMagnifier(FiniteSourceSettings settings)
    : settings_(settings)
{
}

void FiniteSourceMagnifier::ensure_limb_darkening_table() const
{
    const bool cache_matches = limb_darkening_table_valid_ &&
                               limb_darkening_table_c_ == settings_.limb_darkening_c &&
                               limb_darkening_table_d_ == settings_.limb_darkening_d &&
                               static_cast<int>(limb_darkening_table_.size()) == kLimbDarkeningTableSize + 1;
    if (cache_matches) {
        return;
    }

    limb_darkening_table_.resize(kLimbDarkeningTableSize + 1);
    for (int i = 0; i <= kLimbDarkeningTableSize; ++i) {
        const double normalized_radius2 =
            static_cast<double>(i) / static_cast<double>(kLimbDarkeningTableSize);
        limb_darkening_table_[static_cast<std::size_t>(i)] =
            source_surface_brightness(normalized_radius2, settings_);
    }
    limb_darkening_table_valid_ = true;
    limb_darkening_table_c_ = settings_.limb_darkening_c;
    limb_darkening_table_d_ = settings_.limb_darkening_d;
}

double FiniteSourceMagnifier::limb_darkening_table_brightness(double normalized_radius2) const
{
    const double bounded = std::clamp(normalized_radius2, 0.0, 1.0);
    const double index = bounded * static_cast<double>(kLimbDarkeningTableSize);
    const int lower = static_cast<int>(index);
    if (lower >= kLimbDarkeningTableSize) {
        return limb_darkening_table_[static_cast<std::size_t>(kLimbDarkeningTableSize)];
    }
    const double fraction = index - static_cast<double>(lower);
    const double left = limb_darkening_table_[static_cast<std::size_t>(lower)];
    const double right = limb_darkening_table_[static_cast<std::size_t>(lower + 1)];
    return left + fraction * (right - left);
}

double FiniteSourceMagnifier::legacy_limb_darkening_table_brightness(double normalized_radius2) const
{
    const double bounded = std::clamp(normalized_radius2, 0.0, 1.0);
    const int index = std::min(
        static_cast<int>(bounded * static_cast<double>(kLimbDarkeningTableSize)),
        kLimbDarkeningTableSize);
    return limb_darkening_table_[static_cast<std::size_t>(index)];
}

void FiniteSourceMagnifier::ensure_legacy_caustic_cache(double separation, double mass_ratio) const
{
    const int bins = std::max(settings_.caustic_bins, 32);
    const bool cache_matches = caustic_cache_valid_ &&
                               caustic_cache_bins_ == bins &&
                               caustic_cache_separation_ == separation &&
                               caustic_cache_mass_ratio_ == mass_ratio;
    if (!cache_matches) {
        const PointSourceMagnifier point_magnifier;
        caustic_cache_branches_.assign(4, {});
        caustic_cache_points_.clear();
        caustic_cache_points_.reserve(static_cast<std::size_t>(bins) * 4);
        caustic_cache_min_x_ = std::numeric_limits<double>::infinity();
        caustic_cache_max_x_ = -std::numeric_limits<double>::infinity();
        caustic_cache_min_y_ = std::numeric_limits<double>::infinity();
        caustic_cache_max_y_ = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < bins; ++i) {
            const double phase_angle = 2.0 * kPi * static_cast<double>(i) /
                                       static_cast<double>(bins);
            auto points = caustic_points_at_phase(point_magnifier, separation, mass_ratio, phase_angle);
            for (const auto& point : points) {
                caustic_cache_points_.push_back(point);
                caustic_cache_min_x_ = std::min(caustic_cache_min_x_, point.x);
                caustic_cache_max_x_ = std::max(caustic_cache_max_x_, point.x);
                caustic_cache_min_y_ = std::min(caustic_cache_min_y_, point.y);
                caustic_cache_max_y_ = std::max(caustic_cache_max_y_, point.y);
            }
            append_tracked_caustic_points(caustic_cache_branches_, std::move(points));
        }
        const double width = std::max(caustic_cache_max_x_ - caustic_cache_min_x_, 1.0e-12);
        const double height = std::max(caustic_cache_max_y_ - caustic_cache_min_y_, 1.0e-12);
        caustic_cache_grid_step_x_ = width / static_cast<double>(caustic_cache_grid_size_);
        caustic_cache_grid_step_y_ = height / static_cast<double>(caustic_cache_grid_size_);
        caustic_cache_grid_.assign(
            static_cast<std::size_t>(caustic_cache_grid_size_ * caustic_cache_grid_size_), {});
        for (std::size_t i = 0; i < caustic_cache_points_.size(); ++i) {
            const auto& point = caustic_cache_points_[i];
            const int ix = std::clamp(static_cast<int>((point.x - caustic_cache_min_x_) /
                                           caustic_cache_grid_step_x_),
                0,
                caustic_cache_grid_size_ - 1);
            const int iy = std::clamp(static_cast<int>((point.y - caustic_cache_min_y_) /
                                           caustic_cache_grid_step_y_),
                0,
                caustic_cache_grid_size_ - 1);
            caustic_cache_grid_[static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)]
                .push_back(static_cast<int>(i));
        }
        // Build branch-position grid for fast per-segment distance queries.
        caustic_cache_branch_grid_.assign(
            static_cast<std::size_t>(caustic_cache_grid_size_ * caustic_cache_grid_size_), {});
        // max_seg_len is used as a safety margin when bounding the grid search
        // radius in legacy_binary_caustic_distance.  The wrap-around segment
        // (from j=n-1 back to j=0) can be spuriously long when branch tracking
        // swaps inner/outer caustic components near a crossing phase.  Excluding
        // it keeps max_seg_len tight.  Correctness is preserved because the
        // wrap-around gap is still found via the "prev" check on the j=0 entry
        // of each branch in legacy_binary_caustic_distance.
        double max_seg2 = 0.0;
        for (int b = 0; b < static_cast<int>(caustic_cache_branches_.size()); ++b) {
            const auto& br = caustic_cache_branches_[static_cast<std::size_t>(b)];
            const int n = static_cast<int>(br.size());
            for (int j = 0; j < n; ++j) {
                const auto& pt = br[static_cast<std::size_t>(j)];
                const auto& next_pt = br[static_cast<std::size_t>((j + 1) % n)];
                // Exclude wrap-around (j == n-1) from max_seg_len to avoid
                // inflating the search radius with tracking-artifact segments.
                if (j < n - 1) {
                    const double seg2 = distance_squared(pt, next_pt);
                    if (seg2 > max_seg2) {
                        max_seg2 = seg2;
                    }
                }
                const int ix = std::clamp(
                    static_cast<int>((pt.x - caustic_cache_min_x_) / caustic_cache_grid_step_x_),
                    0, caustic_cache_grid_size_ - 1);
                const int iy = std::clamp(
                    static_cast<int>((pt.y - caustic_cache_min_y_) / caustic_cache_grid_step_y_),
                    0, caustic_cache_grid_size_ - 1);
                caustic_cache_branch_grid_[
                    static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)]
                    .push_back({b, j});
            }
        }
        caustic_cache_max_seg_len_ = std::sqrt(max_seg2);

        caustic_cache_valid_ = true;
        caustic_cache_separation_ = separation;
        caustic_cache_mass_ratio_ = mass_ratio;
        caustic_cache_bins_ = bins;
    }
}

double FiniteSourceMagnifier::legacy_binary_caustic_distance(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double hint_nearest_point_dist) const
{
    ensure_legacy_caustic_cache(separation, mass_ratio);

    // Obtain nearest caustic POINT distance as an upper bound on segment distance.
    // The caller often already has this from legacy_binary_sampled_caustic_distance;
    // if so, the hint skips this O(N) scan.
    double nearest_dist = hint_nearest_point_dist;
    if (!std::isfinite(nearest_dist)) {
        double nearest2 = std::numeric_limits<double>::infinity();
        for (const auto& pt : caustic_cache_points_) {
            const double dx = source.x - pt.x;
            const double dy = source.y - pt.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 < nearest2) {
                nearest2 = d2;
            }
        }
        nearest_dist = std::sqrt(nearest2);
    }

    if (caustic_cache_branch_grid_.empty()) {
        return nearest_dist;
    }

    // Search the branch grid within nearest_dist + max_seg_len to catch all segments
    // whose distance to source could be less than nearest_dist.
    const double seg_radius = nearest_dist + caustic_cache_max_seg_len_;
    const int ix0 = std::clamp(
        static_cast<int>((source.x - seg_radius - caustic_cache_min_x_) /
                         caustic_cache_grid_step_x_),
        0, caustic_cache_grid_size_ - 1);
    const int ix1 = std::clamp(
        static_cast<int>((source.x + seg_radius - caustic_cache_min_x_) /
                         caustic_cache_grid_step_x_),
        0, caustic_cache_grid_size_ - 1);
    const int iy0 = std::clamp(
        static_cast<int>((source.y - seg_radius - caustic_cache_min_y_) /
                         caustic_cache_grid_step_y_),
        0, caustic_cache_grid_size_ - 1);
    const int iy1 = std::clamp(
        static_cast<int>((source.y + seg_radius - caustic_cache_min_y_) /
                         caustic_cache_grid_step_y_),
        0, caustic_cache_grid_size_ - 1);

    double distance = nearest_dist;
    for (int iy = iy0; iy <= iy1; ++iy) {
        for (int ix = ix0; ix <= ix1; ++ix) {
            for (const auto& ref : caustic_cache_branch_grid_[
                    static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)]) {
                const auto& branch =
                    caustic_cache_branches_[static_cast<std::size_t>(ref.branch)];
                const int n = static_cast<int>(branch.size());
                if (n < 2) {
                    continue;
                }
                const int prev = (ref.pos > 0) ? ref.pos - 1 : n - 1;
                const int next = (ref.pos + 1) % n;
                distance = std::min(distance,
                    point_segment_distance(source,
                        branch[static_cast<std::size_t>(prev)],
                        branch[static_cast<std::size_t>(ref.pos)]));
                distance = std::min(distance,
                    point_segment_distance(source,
                        branch[static_cast<std::size_t>(ref.pos)],
                        branch[static_cast<std::size_t>(next)]));
            }
        }
    }
    return distance;
}

double FiniteSourceMagnifier::legacy_binary_sampled_caustic_distance(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double search_radius) const
{
    ensure_legacy_caustic_cache(separation, mass_ratio);
    double distance2 = std::numeric_limits<double>::infinity();

    if (search_radius > 0.0 &&
        (source.x < caustic_cache_min_x_ - search_radius ||
            source.x > caustic_cache_max_x_ + search_radius ||
            source.y < caustic_cache_min_y_ - search_radius ||
            source.y > caustic_cache_max_y_ + search_radius)) {
        return std::numeric_limits<double>::infinity();
    }

    if (search_radius > 0.0 && !caustic_cache_grid_.empty()) {
        const int ix0 = std::clamp(static_cast<int>((source.x - search_radius - caustic_cache_min_x_) /
                                       caustic_cache_grid_step_x_),
            0,
            caustic_cache_grid_size_ - 1);
        const int ix1 = std::clamp(static_cast<int>((source.x + search_radius - caustic_cache_min_x_) /
                                       caustic_cache_grid_step_x_),
            0,
            caustic_cache_grid_size_ - 1);
        const int iy0 = std::clamp(static_cast<int>((source.y - search_radius - caustic_cache_min_y_) /
                                       caustic_cache_grid_step_y_),
            0,
            caustic_cache_grid_size_ - 1);
        const int iy1 = std::clamp(static_cast<int>((source.y + search_radius - caustic_cache_min_y_) /
                                       caustic_cache_grid_step_y_),
            0,
            caustic_cache_grid_size_ - 1);
        for (int iy = iy0; iy <= iy1; ++iy) {
            for (int ix = ix0; ix <= ix1; ++ix) {
                const auto& cell =
                    caustic_cache_grid_[static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)];
                for (const int index : cell) {
                    const auto& point = caustic_cache_points_[static_cast<std::size_t>(index)];
                    const double dx = source.x - point.x;
                    const double dy = source.y - point.y;
                    const double candidate = dx * dx + dy * dy;
                    if (candidate < distance2) {
                        distance2 = candidate;
                    }
                }
            }
        }
        if (distance2 == std::numeric_limits<double>::infinity()) {
            return distance2;
        }
    } else {
        for (const auto& point : caustic_cache_points_) {
            const double dx = source.x - point.x;
            const double dy = source.y - point.y;
            const double candidate = dx * dx + dy * dy;
            if (candidate < distance2) {
                distance2 = candidate;
            }
        }
    }
    return std::sqrt(distance2);
}

void FiniteSourceMagnifier::ensure_legacy_polar_map_cache(
    double separation,
    double mass_ratio,
    double source_radius) const
{
    const int source_bins = std::max(settings_.source_bins, 1);
    const double dr = source_radius / static_cast<double>(source_bins);
    const int phi_bins = std::max(16, static_cast<int>(2.0 * kPi / (dr * settings_.grid_ratio)));
    const double dphi = 2.0 * kPi / static_cast<double>(phi_bins);
    const int radial_count = std::max(3 * source_bins, 1);
    const int radial_min_index = static_cast<int>(1.0 / dr) - radial_count / 2;
    const bool cache_matches =
        polar_map_cache_valid_ &&
        polar_map_cache_separation_ == separation &&
        polar_map_cache_mass_ratio_ == mass_ratio &&
        polar_map_cache_source_radius_ == source_radius &&
        polar_map_cache_source_bins_ == source_bins &&
        polar_map_cache_grid_ratio_ == settings_.grid_ratio &&
        polar_map_cache_phi_bins_ == phi_bins &&
        polar_map_cache_radial_offset_min_index_ == radial_min_index &&
        static_cast<int>(polar_map_cache_radial_offsets_.size()) == radial_count;
    if (cache_matches) {
        return;
    }

    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    polar_map_cache_radial_offset_min_index_ = radial_min_index;
    polar_map_cache_radial_offsets_.resize(static_cast<std::size_t>(radial_count));
    polar_map_cache_.resize(static_cast<std::size_t>(radial_count) * static_cast<std::size_t>(phi_bins));
    for (int ir = 0; ir < radial_count; ++ir) {
        polar_map_cache_radial_offsets_[static_cast<std::size_t>(ir)] = ir;
        const double radius = (radial_min_index + ir) * dr + 0.5 * dr;
        for (int iphi = 0; iphi < phi_bins; ++iphi) {
            const double phi = (iphi + 0.5) * dphi;
            const SourcePosition image {radius * std::cos(phi), radius * std::sin(phi)};
            polar_map_cache_[static_cast<std::size_t>(ir) * static_cast<std::size_t>(phi_bins) +
                             static_cast<std::size_t>(iphi)] =
                map_binary_lens_real(mapper, image.x, image.y);
        }
    }

    polar_map_cache_valid_ = true;
    polar_map_cache_separation_ = separation;
    polar_map_cache_mass_ratio_ = mass_ratio;
    polar_map_cache_source_radius_ = source_radius;
    polar_map_cache_source_bins_ = source_bins;
    polar_map_cache_grid_ratio_ = settings_.grid_ratio;
    polar_map_cache_dr_ = dr;
    polar_map_cache_dphi_ = dphi;
    polar_map_cache_phi_bins_ = phi_bins;
}

FiniteSourceResult FiniteSourceMagnifier::legacy_polar_memory_binary_mag(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double caustic_distance) const
{
    const PointSourceMagnifier point_magnifier;
    FiniteSourceDecision decision {
        FiniteSourceMethod::inverse_ray_polar,
        estimate_polar_cost(settings_),
        "polar cached inverse-ray",
    };
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    auto seeds = legacy_augmented_image_seeds(
        point_magnifier, mapper, separation, mass_ratio, source, source_radius);
    if (seeds.size() < 5) {
        legacy_augment_seeds_from_branches(separation, mass_ratio, source, source_radius, seeds);
    }
    const auto point_images = point_magnifier.binary_images(separation, mass_ratio, source);
    const double sampled_caustic_distance = legacy_binary_sampled_caustic_distance(
        separation, mass_ratio, source, source_radius);
    const double polar_fallback_distance =
        std::max(settings_.hex_threshold, 1.0) * source_radius;
    if (seeds.size() > point_images.size() ||
        (std::isfinite(sampled_caustic_distance) && sampled_caustic_distance < polar_fallback_distance) ||
        (std::isfinite(caustic_distance) && caustic_distance < polar_fallback_distance)) {
        const double magnification = legacy_imagearea4_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings_, this, &seeds);
        decision.method = FiniteSourceMethod::inverse_ray_cartesian;
        decision.reason = "polar mode used cartesian fallback for caustic-crossing";
        if (!std::isfinite(magnification)) {
            return {magnification, 0, decision, std::nan(""), 0, false};
        }
        return {magnification, 0, decision, 0.0, 0, true};
    }
    ensure_legacy_polar_map_cache(separation, mass_ratio, source_radius);
    const PolarMapCacheView cache_view {
        &polar_map_cache_,
        &polar_map_cache_radial_offsets_,
        polar_map_cache_radial_offset_min_index_,
        polar_map_cache_phi_bins_,
        polar_map_cache_dr_,
    };
    const double magnification = inverse_ray_polar_boundary_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, settings_, this, &cache_view, &seeds);
    if (!std::isfinite(magnification)) {
        return {magnification, 0, decision, std::nan(""), 0, false};
    }
    return {magnification, 0, decision, 0.0, 0, true};
}

const char* finite_source_method_name(FiniteSourceMethod method)
{
    switch (method) {
    case FiniteSourceMethod::point_source:
        return "point_source";
    case FiniteSourceMethod::hexadecapole:
        return "hexadecapole";
    case FiniteSourceMethod::inverse_ray_cartesian:
        return "inverse_ray_cartesian";
    case FiniteSourceMethod::inverse_ray_polar:
        return "inverse_ray_polar";
    case FiniteSourceMethod::inverse_ray_spine:
        return "inverse_ray_spine";
    default:
        return "unknown";
    }
}

FiniteSourceResult FiniteSourceMagnifier::binary_mag(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double point_source_magnification) const
{
    if (result_cache_valid_ && result_cache_separation_ == separation &&
        result_cache_mass_ratio_ == mass_ratio && result_cache_source_x_ == source.x &&
        result_cache_source_y_ == source.y && result_cache_source_radius_ == source_radius &&
        result_cache_point_magnification_ == point_source_magnification) {
        return result_cache_;
    }

    const auto cache_and_return = [&](FiniteSourceResult result) {
        result_cache_valid_ = true;
        result_cache_separation_ = separation;
        result_cache_mass_ratio_ = mass_ratio;
        result_cache_source_x_ = source.x;
        result_cache_source_y_ = source.y;
        result_cache_source_radius_ = source_radius;
        result_cache_point_magnification_ = point_source_magnification;
        result_cache_ = result;
        return result;
    };

    PointSourceMagnifier point_magnifier;
    if (source_radius <= 0.0) {
        const auto point = point_magnifier.binary_mag0(separation, mass_ratio, source);
        FiniteSourceDecision decision {FiniteSourceMethod::point_source, 0, "zero source radius"};
        return cache_and_return({point.magnification, point.image_count, decision, 0.0, 0, true});
    }

    if (settings_.finite_mode <= 0) {
        FiniteSourceDecision decision {FiniteSourceMethod::point_source, 0, "point-source mode"};
        return cache_and_return({point_source_magnification, 0, decision, 0.0, 0, true});
    }

    // Fast PS exit for sources outside the caustic bounding box by kinji_threshold·ρ.
    // The caustic cache is built (or validated) inside legacy_binary_sampled_caustic_distance,
    // making the bbox members available immediately after the call.
    const double bbox_margin = settings_.kinji_threshold * source_radius;
    const bool adaptive_ir_requested =
        settings_.adaptive_source_bins != 0 &&
        (settings_.finite_source_tol > 0.0 || settings_.finite_source_reltol > 0.0);
    const double adaptive_bbox_margin = adaptive_ir_requested
        ? std::max(bbox_margin, 60.0 * source_radius)
        : bbox_margin;
    const double sampled_dist = legacy_binary_sampled_caustic_distance(
        separation, mass_ratio, source, adaptive_bbox_margin);
    if (source.x < caustic_cache_min_x_ - adaptive_bbox_margin ||
        source.x > caustic_cache_max_x_ + adaptive_bbox_margin ||
        source.y < caustic_cache_min_y_ - adaptive_bbox_margin ||
        source.y > caustic_cache_max_y_ + adaptive_bbox_margin) {
        FiniteSourceDecision decision {
            FiniteSourceMethod::point_source,
            settings_.caustic_bins * 4,
            "source outside caustic bounding box",
        };
        return cache_and_return({point_source_magnification, 0, decision, 0.0, 0, true});
    }

    // VBM-style adaptive mode selection.  When the source center is far enough
    // from the caustic the hexadecapole approximation is tried first; its
    // self-consistency error estimate (|a4 correction| / |magnification|) then
    // determines whether hex is accurate enough or IR is needed.
    //
    // When the source is close to the caustic (sampled_dist < hex_threshold·ρ)
    // the hex Taylor expansion can give a misleadingly small a4 term even when
    // the result is wrong (e.g. a4≈0 when a1_plus and a1_cross both happen to
    // be large and cancel through a2rho2).  Skip hex and go straight to IR.
    // Use segment-based (refined) caustic distance for accurate near-caustic detection.
    // The sampled distance uses only discrete caustic points and can badly overestimate
    // the true distance (e.g. 7x rho) when the caustic is sparsely sampled; the segment
    // distance queries the actual line segments between consecutive points and returns the
    // correct distance.  We pass sampled_dist as a hint to skip the O(N) point scan.
    const double refined_dist = legacy_binary_caustic_distance(
        separation, mass_ratio, source, sampled_dist);
    const double hex_dist_threshold = settings_.hex_threshold * source_radius;
    const bool near_caustic = refined_dist < hex_dist_threshold;
    double rejected_hex_magnification = std::numeric_limits<double>::quiet_NaN();
    if (!near_caustic) {
        const auto hex = hexadecapole_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings_);
        double hex_threshold = settings_.adaptive_hex_threshold;
        if (settings_.adaptive_source_bins != 0 &&
            (settings_.finite_source_tol > 0.0 || settings_.finite_source_reltol > 0.0)) {
            const double requested_relative =
                settings_.finite_source_reltol +
                settings_.finite_source_tol / std::max(std::abs(hex.magnification), 1.0);
            // Graduate hex_safety by caustic distance.  The hex self-consistency
            // check underestimates the actual error most severely when the source
            // boundary is near a caustic fold (dist_ratio ~ hex_threshold).  For
            // sources far from the caustic the Taylor expansion is reliable and
            // needs little or no safety margin.
            //
            // Power-law: safety = 30 * (hex_threshold / dist_ratio)^3
            // clamped to [1, 30].  This gives safety≈30 at dist_ratio=hex_threshold
            // and safety≈1 at dist_ratio≈3*hex_threshold (~9 source radii away).
            // For small sources (rho < 1e-3) hex is always reliable: safety=1.
            double hex_safety = 1.0;
            if (source_radius >= 1.0e-3 && std::isfinite(refined_dist) &&
                source_radius > 0.0) {
                const double dist_ratio = refined_dist / source_radius;
                const double t =
                    settings_.hex_threshold / std::max(dist_ratio, settings_.hex_threshold);
                hex_safety = std::max(1.0, 30.0 * t * t * t);
            }
            hex_threshold = std::min(hex_threshold, requested_relative / hex_safety);
        }
        if (hex.relative_error <= hex_threshold) {
            FiniteSourceDecision decision {
                FiniteSourceMethod::hexadecapole,
                kHexadecapoleEvaluations,
                "hexadecapole self-consistency check passed",
            };
            return cache_and_return({
                hex.magnification,
                0,
                decision,
                hex.relative_error * std::max(std::abs(hex.magnification), 1.0),
                0,
                true});
        }
        rejected_hex_magnification = hex.magnification;
    }

    if (settings_.finite_mode == 2) {
        return cache_and_return(legacy_polar_memory_binary_mag(
            separation, mass_ratio, source, source_radius, refined_dist));
    }
    if (settings_.finite_mode == 3) {
        return cache_and_return(fixed_inverse_ray_spine_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings_, this));
    }
    FiniteSourceDecision decision {
        FiniteSourceMethod::inverse_ray_cartesian,
        estimate_cartesian_cost(settings_),
        "cartesian inverse-ray",
    };
    return cache_and_return(fixed_inverse_ray_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, settings_, this, decision,
        refined_dist, rejected_hex_magnification));
}

void FiniteSourceMagnifier::legacy_augment_seeds_from_branches(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    std::vector<SourcePosition>& seeds) const
{
    ensure_legacy_caustic_cache(separation, mass_ratio);
    if (seeds.size() >= 5 || caustic_cache_branch_grid_.empty()) return;

    const PointSourceMagnifier point_magnifier;
    const double seg_search = source_radius + caustic_cache_max_seg_len_;
    const int gs = caustic_cache_grid_size_;
    const int ix0 = std::clamp(
        static_cast<int>((source.x - seg_search - caustic_cache_min_x_) / caustic_cache_grid_step_x_),
        0, gs - 1);
    const int ix1 = std::clamp(
        static_cast<int>((source.x + seg_search - caustic_cache_min_x_) / caustic_cache_grid_step_x_),
        0, gs - 1);
    const int iy0 = std::clamp(
        static_cast<int>((source.y - seg_search - caustic_cache_min_y_) / caustic_cache_grid_step_y_),
        0, gs - 1);
    const int iy1 = std::clamp(
        static_cast<int>((source.y + seg_search - caustic_cache_min_y_) / caustic_cache_grid_step_y_),
        0, gs - 1);

    for (int iy = iy0; iy <= iy1 && seeds.size() < 5; ++iy) {
        for (int ix = ix0; ix <= ix1 && seeds.size() < 5; ++ix) {
            for (const auto& ref :
                 caustic_cache_branch_grid_[static_cast<std::size_t>(iy * gs + ix)]) {
                const auto& branch =
                    caustic_cache_branches_[static_cast<std::size_t>(ref.branch)];
                const int n = static_cast<int>(branch.size());
                if (n < 2) continue;
                const int next = (ref.pos + 1) % n;
                // Skip the wrap-around segment (last point back to first).  When
                // branch-tracking swaps occur, this segment is a phantom artifact of
                // length ~O(1) that does not correspond to any real caustic arc, yet
                // it can pass through the source disk and produce spurious seed updates.
                if (next == 0) continue;
                const SourcePosition p0 = branch[static_cast<std::size_t>(ref.pos)];
                const SourcePosition p1 = branch[static_cast<std::size_t>(next)];
                if (point_segment_distance(source, p0, p1) >= source_radius) continue;

                const double seg_dx = p1.x - p0.x;
                const double seg_dy = p1.y - p0.y;
                const double seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy;
                const double t = seg_len2 > 0.0 ?
                    std::clamp(
                        ((source.x - p0.x) * seg_dx + (source.y - p0.y) * seg_dy) / seg_len2,
                        0.0, 1.0) :
                    0.0;
                const SourcePosition nearest {p0.x + t * seg_dx, p0.y + t * seg_dy};
                const double distance = std::sqrt(distance_squared(nearest, source));
                if (distance <= 0.0 || distance >= source_radius) continue;

                // Step 5% of source_radius past the nearest caustic segment point
                // toward the interior of the caustic. This is large enough to cross
                // the segment-to-true-caustic approximation error (one inter-sample
                // spacing) without landing too close to a fold caustic where two
                // merging images are nearly degenerate.
                const double step = source_radius * 0.05 / distance;
                const SourcePosition probe_source {
                    nearest.x + (nearest.x - source.x) * step,
                    nearest.y + (nearest.y - source.y) * step,
                };
                // Only use images from regions that actually overlap the source disk.
                // A probe outside the source disk belongs to a caustic region that
                // the disk does not straddle (e.g., a fold caustic tangent to the
                // disk edge) and its images would be seeds for the wrong area.
                if (distance_squared(probe_source, source) >= source_radius * source_radius) {
                    continue;
                }
                const auto probe_images = legacy_residual_selected_images(
                    point_magnifier, separation, mass_ratio, probe_source);
                if (probe_images.size() > seeds.size()) {
                    seeds = probe_images;
                }
            }
        }
    }
}

} // namespace lcbinint::magnification
