#include "lcbinint/magnification/algebraic_boundary.hpp"

#include "lcbinint/math/polynomial_roots.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <queue>
#include <unordered_map>

namespace lcbinint::magnification {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;

struct Moments {
    double m0 = 0.0;
    double mhalf = 0.0;
    double mquarter = 0.0;
};

Moments operator+(Moments lhs, const Moments& rhs)
{
    lhs.m0 += rhs.m0;
    lhs.mhalf += rhs.mhalf;
    lhs.mquarter += rhs.mquarter;
    return lhs;
}

Moments operator-(Moments lhs, const Moments& rhs)
{
    lhs.m0 -= rhs.m0;
    lhs.mhalf -= rhs.mhalf;
    lhs.mquarter -= rhs.mquarter;
    return lhs;
}

Moments operator*(double scale, Moments value)
{
    value.m0 *= scale;
    value.mhalf *= scale;
    value.mquarter *= scale;
    return value;
}

double combined_flux(Moments value, double c, double d)
{
    return (1.0 - c - d) * value.m0 + c * value.mhalf + d * value.mquarter;
}

bool finite_complex(Complex value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

SourcePosition map_lens(
    const AlgebraicBoundaryLens& lens,
    double x,
    double y)
{
    const double a = lens.lens_position;
    const double xa = x - a;
    const double den1 = xa * xa + y * y;
    const double den2 = x * x + y * y;
    if (den1 <= 1.0e-28 || den2 <= 1.0e-28) {
        const double inf = std::numeric_limits<double>::infinity();
        return {inf, inf};
    }
    return {
        x - lens.m1 * xa / den1 - lens.m2 * x / den2 - a * lens.m1,
        y - lens.m1 * y / den1 - lens.m2 * y / den2,
    };
}

double phi_value(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double radius,
    double angle,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    if (diagnostics != nullptr) {
        ++diagnostics->lens_evaluation_count;
    }
    const auto mapped = map_lens(
        lens, radius * std::cos(angle), radius * std::sin(angle));
    if (!std::isfinite(mapped.x) || !std::isfinite(mapped.y)) {
        return -std::numeric_limits<double>::infinity();
    }
    const double dx = mapped.x - source.x;
    const double dy = mapped.y - source.y;
    return 1.0 - (dx * dx + dy * dy) / (source_radius * source_radius);
}

template <std::size_t M, std::size_t N>
std::array<Complex, M + N - 1> convolve_fixed(
    const std::array<Complex, M>& lhs,
    const std::array<Complex, N>& rhs)
{
    std::array<Complex, M + N - 1> out {};
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) out[i + j] += lhs[i] * rhs[j];
    }
    return out;
}

double circular_distance(double lhs, double rhs)
{
    double delta = std::abs(lhs - rhs);
    return std::min(delta, kTwoPi - delta);
}

double wrap_angle(double angle)
{
    angle = std::fmod(angle, kTwoPi);
    return angle < 0.0 ? angle + kTwoPi : angle;
}

struct GaussKronrod15Rule {
    std::array<double, 15> nodes;
    std::array<double, 15> kronrod_weights;
    // Zero at Kronrod-only nodes, allowing both estimates to be accumulated
    // in one pass without a lookup or a second integrand evaluation.
    std::array<double, 15> gauss_weights;
};

const GaussKronrod15Rule& gauss_kronrod15()
{
    static const GaussKronrod15Rule rule {{
        -0.9914553711208126, -0.9491079123427585,
        -0.8648644233597691, -0.7415311855993945,
        -0.5860872354676911, -0.4058451513773972,
        -0.2077849550078985,  0.0,
         0.2077849550078985,  0.4058451513773972,
         0.5860872354676911,  0.7415311855993945,
         0.8648644233597691,  0.9491079123427585,
         0.9914553711208126,
    }, {
        0.02293532201052922, 0.06309209262997855,
        0.1047900103222502,  0.1406532597155259,
        0.1690047266392679,  0.1903505780647854,
        0.2044329400752989,  0.2094821410847278,
        0.2044329400752989,  0.1903505780647854,
        0.1690047266392679,  0.1406532597155259,
        0.1047900103222502,  0.06309209262997855,
        0.02293532201052922,
    }, {
        0.0, 0.1294849661688697,
        0.0, 0.2797053914892766,
        0.0, 0.3818300505051189,
        0.0, 0.4179591836734694,
        0.0, 0.3818300505051189,
        0.0, 0.2797053914892766,
        0.0, 0.1294849661688697,
        0.0,
    }};
    return rule;
}

const std::pair<std::vector<double>, std::vector<double>>& gauss_legendre_rule(int n)
{
    thread_local std::unordered_map<
        int, std::pair<std::vector<double>, std::vector<double>>> cache;
    const auto cached = cache.find(n);
    if (cached != cache.end()) return cached->second;
    std::vector<double> nodes(static_cast<std::size_t>(n));
    std::vector<double> weights(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        double x = std::cos(kPi * (static_cast<double>(i) + 0.75) /
                            (static_cast<double>(n) + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 100; ++iteration) {
            double p0 = 1.0;
            double p1 = x;
            for (int degree = 2; degree <= n; ++degree) {
                const double p2 =
                    ((2.0 * degree - 1.0) * x * p1 -
                     (degree - 1.0) * p0) / static_cast<double>(degree);
                p0 = p1;
                p1 = p2;
            }
            derivative = static_cast<double>(n) * (x * p1 - p0) /
                (x * x - 1.0);
            const double correction = p1 / derivative;
            x -= correction;
            if (std::abs(correction) < 1.0e-15) break;
        }
        nodes[static_cast<std::size_t>(i)] = x;
        weights[static_cast<std::size_t>(i)] =
            2.0 / ((1.0 - x * x) * derivative * derivative);
    }
    return cache.emplace(n, std::make_pair(std::move(nodes), std::move(weights)))
        .first->second;
}

struct NestedMoments {
    Moments gauss;
    Moments kronrod;
};

NestedMoments angular_moments_rule(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double radius,
    const std::vector<std::pair<double, double>>& intervals,
    const GaussKronrod15Rule& rule,
    bool need_half,
    bool need_quarter,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    NestedMoments result;
    for (const auto& interval : intervals) {
        const double midpoint = 0.5 * (interval.first + interval.second);
        const double half_width = 0.5 * (interval.second - interval.first);
        result.gauss.m0 += radius * 2.0 * half_width;
        result.kronrod.m0 += radius * 2.0 * half_width;
        for (std::size_t i = 0; i < rule.nodes.size(); ++i) {
            // Sine-map the Legendre coordinate.  Near x=+-1 the distance to
            // the boundary is quadratic and the Jacobian vanishes linearly,
            // smoothing both Phi^(1/2) and the harder Phi^(1/4) endpoint.
            const double phase = 0.5 * kPi * rule.nodes[i];
            const double angle = midpoint + half_width * std::sin(phase);
            const double phi = std::max(
                0.0, phi_value(lens, source, source_radius, radius, angle, diagnostics));
            const double common = radius * half_width * 0.5 * kPi * std::cos(phase);
            const double half = std::sqrt(phi);
            const double kronrod_weight = common * rule.kronrod_weights[i];
            const double gauss_weight = common * rule.gauss_weights[i];
            if (need_half) {
                result.kronrod.mhalf += kronrod_weight * half;
                result.gauss.mhalf += gauss_weight * half;
            }
            if (need_quarter) {
                const double quarter = std::sqrt(half);
                result.kronrod.mquarter += kronrod_weight * quarter;
                result.gauss.mquarter += gauss_weight * quarter;
            }
        }
    }
    return result;
}

Moments angular_moments_high_rule(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double radius,
    const std::vector<std::pair<double, double>>& intervals,
    int order,
    bool need_half,
    bool need_quarter,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    Moments result;
    const auto& [nodes, weights] = gauss_legendre_rule(order);
    for (const auto& interval : intervals) {
        const double midpoint = 0.5 * (interval.first + interval.second);
        const double half_width = 0.5 * (interval.second - interval.first);
        result.m0 += radius * 2.0 * half_width;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const double phase = 0.5 * kPi * nodes[i];
            const double angle = midpoint + half_width * std::sin(phase);
            const double phi = std::max(
                0.0, phi_value(lens, source, source_radius, radius, angle, diagnostics));
            const double weighted_area = radius * half_width * 0.5 * kPi *
                std::cos(phase) * weights[i];
            const double half = std::sqrt(phi);
            if (need_half) result.mhalf += weighted_area * half;
            if (need_quarter) result.mquarter += weighted_area * std::sqrt(half);
        }
    }
    return result;
}

struct IntegrandEvaluation {
    bool success = false;
    Moments value;
    double angular_error = 0.0;
    std::string unsafe_reason;
};

thread_local std::unordered_map<double, IntegrandEvaluation>* active_integrand_cache = nullptr;

class IntegrandCacheScope {
public:
    explicit IntegrandCacheScope(
        std::unordered_map<double, IntegrandEvaluation>& cache)
        : previous_(active_integrand_cache)
    {
        active_integrand_cache = &cache;
    }
    ~IntegrandCacheScope() { active_integrand_cache = previous_; }

private:
    std::unordered_map<double, IntegrandEvaluation>* previous_;
};

IntegrandEvaluation evaluate_radial_integrand(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double radius,
    double c,
    double d,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    if (active_integrand_cache != nullptr) {
        const auto found = active_integrand_cache->find(radius);
        if (found != active_integrand_cache->end()) return found->second;
    }
    auto intervals = algebraic_boundary_intervals_at_radius(
        lens, source, source_radius, radius, settings, diagnostics);
    IntegrandEvaluation result;
    if (!intervals.success) {
        result = {false, {}, 0.0, intervals.unsafe_reason};
    } else if (c == 0.0 && d == 0.0) {
        Moments exact;
        for (const auto& interval : intervals.inside_intervals) {
            exact.m0 += radius * (interval.second - interval.first);
        }
        result = {true, exact, 0.0, {}};
    } else {
        const auto nested = angular_moments_rule(
            lens, source, source_radius, radius, intervals.inside_intervals,
            gauss_kronrod15(), c != 0.0, d != 0.0, diagnostics);
        const auto high = angular_moments_high_rule(
            lens, source, source_radius, radius, intervals.inside_intervals,
            16, c != 0.0, d != 0.0, diagnostics);
        result = {true, high,
            std::abs(combined_flux(high - nested.kronrod, c, d)), {}};
    }
    if (active_integrand_cache != nullptr) {
        active_integrand_cache->emplace(radius, result);
    }
    return result;
}

bool radius_has_image(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double radius,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics,
    std::string* reason)
{
    const auto result = algebraic_boundary_intervals_at_radius(
        lens, source, source_radius, radius, settings, diagnostics);
    if (!result.success) {
        if (reason != nullptr) {
            *reason = result.unsafe_reason;
        }
        return false;
    }
    return !result.inside_intervals.empty();
}

struct BandSearchResult {
    bool success = false;
    std::vector<std::pair<double, double>> bands;
    std::string unsafe_reason;
};

BandSearchResult find_radial_bands(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    const std::vector<SourcePosition>& seeds,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    std::vector<std::pair<double, double>> bands;
    // The image width opens as sqrt(distance) at a band endpoint, so the
    // omitted area from an endpoint bracket is O(delta^(3/2)).  Sixteen
    // halvings put that contribution below the tightest 3e-5
    // experimental tolerance; thirty spent almost half the root solves on
    // sub-picometre endpoint locations that quadrature cannot use.
    constexpr int bisection_iterations = 16;
    for (const auto& seed : seeds) {
        const double seed_radius = std::hypot(seed.x, seed.y);
        if (std::any_of(bands.begin(), bands.end(), [&](const auto& band) {
                return seed_radius >= band.first && seed_radius <= band.second;
            })) {
            // Keep one solve at every certified component's own coordinate.
            // Besides revalidating the seed, this populates the continuation
            // workspace with that component's root branch before quadrature.
            std::string reason;
            if (!radius_has_image(
                    lens, source, source_radius, seed_radius, settings,
                    diagnostics, &reason)) {
                return {false, {}, reason.empty()
                    ? "covered certified seed was not active" : reason};
            }
            continue;
        }
        std::string reason;
        if (!radius_has_image(
                lens, source, source_radius, seed_radius, settings, diagnostics, &reason)) {
            if (!reason.empty()) {
                return {false, {}, reason};
            }
            // A certified seed may be extremely close to a lens pole.  Such a
            // component cannot be safely banded by this experimental path.
            return {false, {}, "certified seed did not lie on an active radius"};
        }

        auto search_side = [&](int direction, double* endpoint) {
            double active = seed_radius;
            double step = std::max(source_radius, 1.0e-10 * (1.0 + seed_radius));
            double outside = active;
            bool found = false;
            for (int i = 0; i < settings.maximum_band_search_steps; ++i) {
                const double candidate = direction < 0
                    ? std::max(0.0, active - step)
                    : active + step;
                if (direction < 0 && candidate == 0.0) {
                    outside = 0.0;
                    found = true;
                    break;
                }
                if (candidate == active) {
                    outside = candidate;
                    found = true;
                    break;
                }
                reason.clear();
                const bool inside = radius_has_image(
                    lens, source, source_radius, candidate, settings, diagnostics, &reason);
                if (!reason.empty()) {
                    return false;
                }
                if (!inside) {
                    outside = candidate;
                    found = true;
                    break;
                }
                active = candidate;
                step *= 2.0;
            }
            if (!found) {
                reason = "radial band search limit reached";
                return false;
            }
            if (outside == active) {
                *endpoint = outside;
                return true;
            }
            double lo = std::min(active, outside);
            double hi = std::max(active, outside);
            // Do not converge all the way onto the double root at a radial
            // tangency.  A sub-nanometric image-radius bracket is already far
            // below the quadrature error and keeps the endpoint classifier on
            // the simple-root side of the event.
            for (int iteration = 0; iteration < bisection_iterations; ++iteration) {
                const double mid = 0.5 * (lo + hi);
                reason.clear();
                const bool inside = radius_has_image(
                    lens, source, source_radius, mid, settings, diagnostics, &reason);
                if (!reason.empty()) {
                    return false;
                }
                if (direction < 0) {
                    if (inside) hi = mid; else lo = mid;
                } else {
                    if (inside) lo = mid; else hi = mid;
                }
            }
            *endpoint = direction < 0 ? hi : lo;
            return true;
        };

        double lower = seed_radius;
        double upper = seed_radius;
        if (!search_side(-1, &lower) || !search_side(1, &upper)) {
            return {false, {}, reason.empty() ? "radial band search failed" : reason};
        }
        if (upper > lower) {
            bands.emplace_back(lower, upper);
        }
    }

    std::sort(bands.begin(), bands.end());
    std::vector<std::pair<double, double>> merged;
    for (const auto& band : bands) {
        const double tolerance = 1.0e-11 * (1.0 + std::abs(band.second));
        if (merged.empty() || band.first > merged.back().second + tolerance) {
            merged.push_back(band);
        } else {
            merged.back().second = std::max(merged.back().second, band.second);
        }
    }
    if (merged.empty()) {
        return {false, {}, "no radial bands found"};
    }
    diagnostics->radial_band_count = static_cast<int>(merged.size());
    return {true, std::move(merged), {}};
}

struct RadialIntegralResult {
    bool success = false;
    Moments value;
    double error = 0.0;
    std::string unsafe_reason;
};

template <typename Evaluate>
RadialIntegralResult integrate_outer_adaptive(
    Evaluate&& evaluate,
    double c,
    double d,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics,
    double lower,
    double upper,
    double tolerance,
    int initial_depth,
    const char* depth_reason)
{
    struct Panel {
        double lower = 0.0;
        double upper = 0.0;
        int depth = 0;
        Moments value;
        double radial_error = 0.0;
        double reported_error = 0.0;
    };
    const auto evaluate_panel = [&](double panel_lower, double panel_upper, int depth,
                                    Panel* panel, std::string* reason) {
        const auto& rule = gauss_kronrod15();
        Moments gauss;
        Moments kronrod;
        double inner_error = 0.0;
        const double midpoint = 0.5 * (panel_lower + panel_upper);
        const double half_width = 0.5 * (panel_upper - panel_lower);
        for (std::size_t i = 0; i < rule.nodes.size(); ++i) {
            const double phase = 0.5 * kPi * rule.nodes[i];
            const double coordinate = midpoint + half_width * std::sin(phase);
            const double transformed_weight =
                half_width * 0.5 * kPi * std::cos(phase);
            const auto evaluated = evaluate(coordinate);
            if (!evaluated.success) {
                *reason = evaluated.unsafe_reason;
                return false;
            }
            kronrod = kronrod +
                (transformed_weight * rule.kronrod_weights[i]) * evaluated.value;
            gauss = gauss +
                (transformed_weight * rule.gauss_weights[i]) * evaluated.value;
            inner_error += transformed_weight * rule.kronrod_weights[i] *
                evaluated.angular_error;
        }
        panel->lower = panel_lower;
        panel->upper = panel_upper;
        panel->depth = depth;
        panel->value = kronrod;
        panel->radial_error = std::abs(combined_flux(kronrod - gauss, c, d));
        panel->reported_error = panel->radial_error + inner_error;
        return true;
    };

    const auto lower_priority = [](const Panel& left, const Panel& right) {
        return left.radial_error < right.radial_error;
    };
    std::priority_queue<Panel, std::vector<Panel>, decltype(lower_priority)>
        refinable(lower_priority);
    Panel first;
    std::string reason;
    if (!evaluate_panel(lower, upper, initial_depth, &first, &reason)) {
        return {false, {}, 0.0, reason};
    }
    if (first.depth < settings.maximum_radial_depth) refinable.push(first);
    Moments total = first.value;
    double total_radial_error = first.radial_error;
    double total_reported_error = first.reported_error;
    while (total_radial_error > tolerance) {
        if (refinable.empty()) {
            return {false, {}, total_radial_error, depth_reason};
        }
        const Panel parent = refinable.top();
        refinable.pop();
        const double midpoint = 0.5 * (parent.lower + parent.upper);
        Panel left;
        Panel right;
        if (!evaluate_panel(parent.lower, midpoint, parent.depth + 1, &left, &reason) ||
            !evaluate_panel(midpoint, parent.upper, parent.depth + 1, &right, &reason)) {
            return {false, {}, total_radial_error, reason};
        }
        total = total - parent.value + left.value + right.value;
        total_radial_error = std::max(
            0.0, total_radial_error - parent.radial_error +
                left.radial_error + right.radial_error);
        total_reported_error = std::max(
            0.0, total_reported_error - parent.reported_error +
                left.reported_error + right.reported_error);
        if (left.depth < settings.maximum_radial_depth) refinable.push(left);
        if (right.depth < settings.maximum_radial_depth) refinable.push(right);
        ++diagnostics->radial_subdivision_count;
    }
    return {true, total, total_reported_error, {}};
}

RadialIntegralResult integrate_radial_interval(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double c,
    double d,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics,
    double lower,
    double upper,
    double tolerance,
    int depth)
{
    return integrate_outer_adaptive(
        [&](double radius) {
            return evaluate_radial_integrand(
                lens, source, source_radius, radius, c, d, settings, diagnostics);
        },
        c, d, settings, diagnostics, lower, upper, tolerance, depth,
        "radial quadrature depth limit reached");
}

struct RootContinuationWorkspace {
    struct Entry {
        double coordinate = 0.0;
        std::array<Complex, 6> roots {};
        std::size_t degree = 0;
    };
    // One accepted panel evaluates 8- and 16-point rules.  Keeping both
    // generations lets child panels warm-start from the nearest parent node.
    static constexpr std::size_t capacity = 32;
    std::array<Entry, capacity> entries {};
    std::size_t entry_count = 0;
    std::size_t next_entry = 0;

    const Entry* nearest(double coordinate, std::size_t degree) const
    {
        const Entry* best = nullptr;
        double best_distance = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < entry_count; ++i) {
            const auto& entry = entries[i];
            if (entry.degree != degree) continue;
            const double distance = std::abs(entry.coordinate - coordinate);
            if (distance < best_distance) {
                best = &entry;
                best_distance = distance;
            }
        }
        return best;
    }

    void remember(double coordinate, const std::vector<Complex>& roots)
    {
        const bool replacing = entry_count == capacity;
        const std::size_t index = replacing ? next_entry : entry_count++;
        auto& entry = entries[index];
        entry.coordinate = coordinate;
        entry.degree = roots.size();
        std::copy(roots.begin(), roots.end(), entry.roots.begin());
        if (replacing) next_entry = (next_entry + 1) % capacity;
    }
};

thread_local RootContinuationWorkspace* active_root_workspace = nullptr;

class RootWorkspaceScope {
public:
    explicit RootWorkspaceScope(RootContinuationWorkspace& workspace)
        : previous_(active_root_workspace)
    {
        active_root_workspace = &workspace;
    }
    ~RootWorkspaceScope() { active_root_workspace = previous_; }

private:
    RootContinuationWorkspace* previous_;
};

Complex polynomial_derivative(
    const std::vector<Complex>& coefficients,
    Complex root)
{
    Complex value = 0.0;
    for (std::size_t degree = coefficients.size() - 1; degree > 0; --degree) {
        value = value * root + static_cast<double>(degree) * coefficients[degree];
    }
    return value;
}

bool roots_satisfy_polynomial(
    const std::vector<Complex>& coefficients,
    const std::vector<Complex>& roots)
{
    for (const Complex root : roots) {
        if (!finite_complex(root)) return false;
        const double radius = std::max(std::abs(root), 1.0);
        double scale = 0.0;
        double power = 1.0;
        for (const Complex coefficient : coefficients) {
            scale += std::abs(coefficient) * power;
            power *= radius;
        }
        const double residual = std::abs(
            math::PolynomialRootSolver::evaluate(coefficients, root));
        if (!std::isfinite(residual) || residual > 2.0e-13 * std::max(scale, 1.0e-300)) {
            return false;
        }
    }
    return true;
}

bool roots_satisfy_vieta(
    const std::vector<Complex>& coefficients,
    const std::vector<Complex>& roots)
{
    if (coefficients.size() != roots.size() + 1 || coefficients.back() == Complex {}) {
        return false;
    }
    Complex sum = 0.0;
    Complex product = 1.0;
    for (const Complex root : roots) {
        sum += root;
        product *= root;
    }
    const Complex leading = coefficients.back();
    const Complex expected_sum = -coefficients[coefficients.size() - 2] / leading;
    Complex expected_product = coefficients.front() / leading;
    if ((roots.size() % 2) != 0) expected_product = -expected_product;
    const auto close = [](Complex value, Complex expected) {
        return std::abs(value - expected) <= 2.0e-10 * (1.0 + std::abs(expected));
    };
    return close(sum, expected_sum) && close(product, expected_product);
}

math::PolynomialRootResult solve_boundary_polynomial(
    const std::vector<Complex>& coefficients,
    double coordinate,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    math::PolynomialRootSolver solver;
    const std::size_t degree = coefficients.empty() ? 0 : coefficients.size() - 1;
    const auto* nearby = active_root_workspace != nullptr
        ? active_root_workspace->nearest(coordinate, degree) : nullptr;
    if (nearby != nullptr) {
        std::vector<Complex> predicted(
            nearby->roots.begin(), nearby->roots.begin() + nearby->degree);
        for (auto& root : predicted) {
            const Complex derivative = polynomial_derivative(coefficients, root);
            if (std::abs(derivative) > 1.0e-18 && finite_complex(derivative)) {
                root -= math::PolynomialRootSolver::evaluate(coefficients, root) / derivative;
            }
        }
        math::PolynomialRootOptions options;
        options.polish_roots = true;
        options.use_roots_as_starting_points = true;
        // For the sextic, polishing six continued roots is substantially
        // cheaper than rediscovery plus four deflations.  The quartic needs
        // only two iterative roots before its direct quadratic tail, so its
        // ordinary warm SG path remains faster.
        options.polish_only_starting_roots = degree >= 6;
        if (diagnostics != nullptr) diagnostics->newton_prediction_count +=
            static_cast<int>(predicted.size());
        if (diagnostics != nullptr) ++diagnostics->warm_root_solve_count;
        auto warm = solver.solve_from_roots(coefficients, std::move(predicted), options);
        if (warm.status == math::RootSolverStatus::ok &&
            roots_satisfy_polynomial(coefficients, warm.roots) &&
            (!options.polish_only_starting_roots ||
                roots_satisfy_vieta(coefficients, warm.roots))) {
            active_root_workspace->remember(coordinate, warm.roots);
            return warm;
        }
        if (diagnostics != nullptr) ++diagnostics->cold_retry_count;
    }

    if (diagnostics != nullptr) ++diagnostics->cold_root_solve_count;
    auto cold = solver.solve(coefficients);
    if (cold.status == math::RootSolverStatus::ok && active_root_workspace != nullptr &&
        roots_satisfy_polynomial(coefficients, cold.roots)) {
        active_root_workspace->remember(coordinate, cold.roots);
    }
    return cold;
}

} // namespace

std::vector<Complex> algebraic_boundary_quartic(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_radius)
{
    const double a = lens.lens_position;
    const double r = image_radius;
    const Complex shifted_source(source.x + a * lens.m1, source.y);

    // (zeta-source) = A(u)/D(u), z=R*u, |u|=1.
    const std::array<Complex, 3> numerator {
        -shifted_source * (r * r),
        r * (r * r - 1.0 + a * shifted_source),
        a * (lens.m2 - r * r),
    };
    const std::array<Complex, 2> denominator {
        r * r,
        -a * r,
    };
    const std::array<Complex, 3> numerator_reverse {
        std::conj(numerator[2]), std::conj(numerator[1]), std::conj(numerator[0]),
    };
    const std::array<Complex, 2> denominator_reverse {
        std::conj(denominator[1]), std::conj(denominator[0]),
    };
    auto coefficients = convolve_fixed(numerator, numerator_reverse);
    const auto denominator_product = convolve_fixed(denominator, denominator_reverse);
    const double rho2 = source_radius * source_radius;
    for (std::size_t i = 0; i < denominator_product.size(); ++i) {
        coefficients[i + 1] -= rho2 * denominator_product[i];
    }
    return {coefficients.begin(), coefficients.end()};
}

AlgebraicBoundaryIntervals algebraic_boundary_intervals_at_radius(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_radius,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    if (!(source_radius > 0.0) || !(image_radius >= 0.0) ||
        !std::isfinite(source_radius) || !std::isfinite(image_radius)) {
        return {false, false, "invalid source or image radius", {}, {}};
    }
    if (image_radius <= 1.0e-14) {
        // At R=0 the angular parameterisation collapses to one point.
        return {false, false, "fixed-radius circle crosses a lens pole", {}, {}};
    }
    auto coefficients = algebraic_boundary_quartic(
        lens, source, source_radius, image_radius);
    for (const auto coefficient : coefficients) {
        if (!finite_complex(coefficient)) {
            return {false, false, "non-finite quartic coefficient", {}, {}};
        }
    }
    if (diagnostics != nullptr) {
        ++diagnostics->root_solve_count;
    }
    const auto solved = solve_boundary_polynomial(
        coefficients, image_radius, diagnostics);
    if (solved.status != math::RootSolverStatus::ok) {
        return {false, false, "quartic root solve failed", {}, {}};
    }

    std::vector<double> angles;
    angles.reserve(4);
    for (const auto root : solved.roots) {
        if (!finite_complex(root)) {
            if (diagnostics != nullptr) ++diagnostics->rejected_root_count;
            continue;
        }
        const double modulus = std::abs(root);
        if (std::abs(modulus - 1.0) > settings.unit_circle_tolerance) {
            continue;
        }
        if (!(modulus > 0.0)) {
            if (diagnostics != nullptr) ++diagnostics->rejected_root_count;
            continue;
        }
        const Complex projected = root / modulus;
        const double angle = wrap_angle(std::arg(projected));
        const double phi = phi_value(
            lens, source, source_radius, image_radius, angle, diagnostics);
        const double residual = std::abs(phi) * source_radius;
        if (!std::isfinite(residual) ||
            residual > settings.boundary_residual_tolerance * (1.0 + source_radius)) {
            if (diagnostics != nullptr) ++diagnostics->rejected_root_count;
            continue;
        }
        angles.push_back(angle);
    }
    std::sort(angles.begin(), angles.end());
    std::vector<double> unique;
    unique.reserve(angles.size());
    constexpr double duplicate_tolerance = 1.0e-10;
    for (const double angle : angles) {
        if (!unique.empty() && circular_distance(angle, unique.back()) < duplicate_tolerance) {
            if (diagnostics != nullptr) ++diagnostics->tangent_root_count;
            // A double tangency touches the boundary without changing sign.
            // Cancel the pair.  Near-tangent simple roots farther apart than
            // this tolerance remain two endpoints and retain their thin arc.
            unique.pop_back();
            continue;
        }
        unique.push_back(angle);
    }
    if (unique.size() > 1 && circular_distance(unique.front(), unique.back()) < duplicate_tolerance) {
        if (diagnostics != nullptr) ++diagnostics->tangent_root_count;
        unique.erase(unique.begin());
        unique.pop_back();
    }
    if ((unique.size() % 2) != 0) {
        return {false, false, "odd physical boundary-root count", {}, {}};
    }

    AlgebraicBoundaryIntervals result;
    result.success = true;
    result.boundary_angles = unique;
    result.inside_intervals.reserve(unique.size() / 2);
    if (unique.empty()) {
        const double phi = phi_value(
            lens, source, source_radius, image_radius, 0.371, diagnostics);
        if (phi > 0.0) {
            result.full_circle = true;
            result.inside_intervals.emplace_back(0.0, kTwoPi);
        }
        return result;
    }
    for (std::size_t i = 0; i < unique.size(); ++i) {
        const double lower = unique[i];
        const double upper = i + 1 < unique.size() ? unique[i + 1] : unique[0] + kTwoPi;
        const double middle = 0.5 * (lower + upper);
        if (phi_value(
                lens, source, source_radius, image_radius, middle, diagnostics) > 0.0) {
            result.inside_intervals.emplace_back(lower, upper);
        }
    }
    return result;
}

AlgebraicBoundaryResult experimental_algebraic_boundary_integral(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double limb_darkening_c,
    double limb_darkening_d,
    const std::vector<SourcePosition>& certified_image_seeds,
    const AlgebraicBoundarySettings& settings)
{
    AlgebraicBoundaryResult result;
    RootContinuationWorkspace root_workspace;
    RootWorkspaceScope root_scope(root_workspace);
    std::unordered_map<double, IntegrandEvaluation> integrand_cache;
    integrand_cache.reserve(64);
    IntegrandCacheScope integrand_cache_scope(integrand_cache);
    if (certified_image_seeds.empty()) {
        result.unsafe_reason = "no certified image seeds";
        return result;
    }
    const double normalization =
        kPi * source_radius * source_radius *
        (1.0 - limb_darkening_c / 3.0 - limb_darkening_d / 5.0);
    if (!(normalization > 0.0) || !std::isfinite(normalization)) {
        result.unsafe_reason = "invalid source flux normalization";
        return result;
    }
    const auto bands = find_radial_bands(
        lens, source, source_radius, certified_image_seeds, settings,
        &result.diagnostics);
    if (!bands.success) {
        result.unsafe_reason = bands.unsafe_reason;
        return result;
    }

    // Obtain a one-panel estimate first so relative tolerance scales with the
    // finite-source magnification instead of implicitly assuming A=1.
    Moments preliminary_total;
    double preliminary_error = 0.0;
    for (const auto& band : bands.bands) {
        const auto estimate = integrate_radial_interval(
            lens, source, source_radius, limb_darkening_c, limb_darkening_d,
            settings, &result.diagnostics, band.first, band.second,
            std::numeric_limits<double>::infinity(), 0);
        if (!estimate.success) {
            result.unsafe_reason = estimate.unsafe_reason;
            return result;
        }
        preliminary_total = preliminary_total + estimate.value;
        preliminary_error += estimate.error;
    }
    const double preliminary_magnification =
        combined_flux(preliminary_total, limb_darkening_c, limb_darkening_d) /
        normalization;
    const double requested_magnification_error =
        std::max(settings.absolute_tolerance, 0.0) +
        std::max(settings.relative_tolerance, 0.0) *
            std::max(std::abs(preliminary_magnification), 1.0);
    const double requested_flux_error = requested_magnification_error * normalization;
    if (preliminary_error <= requested_flux_error) {
        result.magnification = preliminary_magnification;
        result.error_estimate = preliminary_error / normalization;
        result.success = std::isfinite(result.magnification);
        if (!result.success) {
            result.unsafe_reason = "non-finite integrated magnification";
        }
        return result;
    }

    Moments total;
    double total_error = 0.0;
    const double tolerance_per_band =
        requested_flux_error / static_cast<double>(bands.bands.size());
    for (const auto& band : bands.bands) {
        const auto integrated = integrate_radial_interval(
            lens, source, source_radius, limb_darkening_c, limb_darkening_d,
            settings, &result.diagnostics, band.first, band.second,
            tolerance_per_band, 0);
        if (!integrated.success) {
            result.error_estimate = integrated.error / normalization;
            result.unsafe_reason = integrated.unsafe_reason;
            return result;
        }
        total = total + integrated.value;
        total_error += integrated.error;
    }
    result.magnification =
        combined_flux(total, limb_darkening_c, limb_darkening_d) / normalization;
    result.error_estimate = total_error / normalization;
    result.success = std::isfinite(result.magnification);
    if (!result.success) {
        result.unsafe_reason = "non-finite integrated magnification";
    }
    return result;
}

std::vector<Complex> algebraic_cartesian_boundary_sextic(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x)
{
    const Complex imaginary(0.0, 1.0);
    const Complex shifted_source(source.x + lens.lens_position * lens.m1, source.y);
    const std::array<Complex, 2> zbar {image_x, -imaginary};
    const std::array<Complex, 2> zbar_minus_a {
        image_x - lens.lens_position, -imaginary};
    const auto denominator = convolve_fixed(zbar_minus_a, zbar);
    const std::array<Complex, 2> shifted_z {image_x - shifted_source, imaginary};
    auto numerator = convolve_fixed(shifted_z, denominator);
    for (std::size_t i = 0; i < zbar.size(); ++i) {
        numerator[i] -= lens.m1 * zbar[i];
        numerator[i] -= lens.m2 * zbar_minus_a[i];
    }
    std::array<Complex, 4> numerator_conjugate;
    std::array<Complex, 3> denominator_conjugate;
    std::transform(
        numerator.begin(), numerator.end(), numerator_conjugate.begin(),
        [](Complex value) { return std::conj(value); });
    std::transform(
        denominator.begin(), denominator.end(), denominator_conjugate.begin(),
        [](Complex value) { return std::conj(value); });
    auto coefficients = convolve_fixed(numerator, numerator_conjugate);
    const auto denominator_norm = convolve_fixed(denominator, denominator_conjugate);
    const double rho2 = source_radius * source_radius;
    for (std::size_t i = 0; i < denominator_norm.size(); ++i) {
        coefficients[i] -= rho2 * denominator_norm[i];
    }
    return {coefficients.begin(), coefficients.end()};
}

AlgebraicCartesianIntervals algebraic_cartesian_intervals_at_x(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    if (!(source_radius > 0.0) || !std::isfinite(source_radius) ||
        !std::isfinite(image_x)) {
        return {false, "invalid source radius or scan coordinate", {}, {}};
    }
    auto coefficients = algebraic_cartesian_boundary_sextic(
        lens, source, source_radius, image_x);
    for (auto& coefficient : coefficients) {
        if (!finite_complex(coefficient)) {
            return {false, "non-finite sextic coefficient", {}, {}};
        }
        const double scale = 1.0 + std::abs(coefficient.real());
        if (std::abs(coefficient.imag()) > 2.0e-12 * scale) {
            return {false, "fixed-x boundary polynomial was not real", {}, {}};
        }
        coefficient = Complex(coefficient.real(), 0.0);
    }
    if (diagnostics != nullptr) ++diagnostics->root_solve_count;
    const auto solved = solve_boundary_polynomial(coefficients, image_x, diagnostics);
    if (solved.status != math::RootSolverStatus::ok) {
        return {false, "sextic root solve failed", {}, {}};
    }

    std::vector<double> coordinates;
    coordinates.reserve(6);
    for (const auto root : solved.roots) {
        if (!finite_complex(root)) {
            if (diagnostics != nullptr) ++diagnostics->rejected_root_count;
            continue;
        }
        const double real_tolerance = settings.unit_circle_tolerance *
            (1.0 + std::abs(root.real()));
        if (std::abs(root.imag()) > real_tolerance) continue;
        const double y = root.real();
        if (diagnostics != nullptr) ++diagnostics->lens_evaluation_count;
        const auto mapped = map_lens(lens, image_x, y);
        const double dx = mapped.x - source.x;
        const double dy = mapped.y - source.y;
        const double residual = std::abs(
            1.0 - (dx * dx + dy * dy) / (source_radius * source_radius)) *
            source_radius;
        if (!std::isfinite(residual) ||
            residual > settings.boundary_residual_tolerance * (1.0 + source_radius)) {
            if (diagnostics != nullptr) ++diagnostics->rejected_root_count;
            continue;
        }
        coordinates.push_back(y);
    }
    std::sort(coordinates.begin(), coordinates.end());
    std::vector<double> unique;
    unique.reserve(coordinates.size());
    for (const double coordinate : coordinates) {
        const double duplicate_tolerance = 1.0e-10 * (1.0 + std::abs(coordinate));
        if (!unique.empty() && coordinate - unique.back() < duplicate_tolerance) {
            if (diagnostics != nullptr) ++diagnostics->tangent_root_count;
            unique.pop_back();
        } else {
            unique.push_back(coordinate);
        }
    }
    if ((unique.size() % 2) != 0) {
        return {false, "odd physical fixed-x boundary-root count", {}, {}};
    }

    AlgebraicCartesianIntervals result;
    result.success = true;
    result.boundary_coordinates = unique;
    result.inside_intervals.reserve(unique.size() / 2);
    for (std::size_t i = 0; i + 1 < unique.size(); ++i) {
        const double lower = unique[i];
        const double upper = unique[i + 1];
        const double middle = 0.5 * (lower + upper);
        if (diagnostics != nullptr) ++diagnostics->lens_evaluation_count;
        const auto mapped = map_lens(lens, image_x, middle);
        const double dx = mapped.x - source.x;
        const double dy = mapped.y - source.y;
        if (dx * dx + dy * dy < source_radius * source_radius) {
            result.inside_intervals.emplace_back(lower, upper);
        }
    }
    return result;
}

namespace {

NestedMoments cartesian_inner_moments_rule(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x,
    const std::vector<std::pair<double, double>>& intervals,
    const GaussKronrod15Rule& rule,
    bool need_half,
    bool need_quarter,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    NestedMoments result;
    for (const auto& interval : intervals) {
        const double midpoint = 0.5 * (interval.first + interval.second);
        const double half_width = 0.5 * (interval.second - interval.first);
        result.gauss.m0 += 2.0 * half_width;
        result.kronrod.m0 += 2.0 * half_width;
        for (std::size_t i = 0; i < rule.nodes.size(); ++i) {
            const double phase = 0.5 * kPi * rule.nodes[i];
            const double image_y = midpoint + half_width * std::sin(phase);
            if (diagnostics != nullptr) ++diagnostics->lens_evaluation_count;
            const auto mapped = map_lens(lens, image_x, image_y);
            const double dx = mapped.x - source.x;
            const double dy = mapped.y - source.y;
            const double phi = std::max(
                0.0, 1.0 - (dx * dx + dy * dy) /
                    (source_radius * source_radius));
            const double common = half_width * 0.5 * kPi * std::cos(phase);
            const double half = std::sqrt(phi);
            const double kronrod_weight = common * rule.kronrod_weights[i];
            const double gauss_weight = common * rule.gauss_weights[i];
            if (need_half) {
                result.kronrod.mhalf += kronrod_weight * half;
                result.gauss.mhalf += gauss_weight * half;
            }
            if (need_quarter) {
                const double quarter = std::sqrt(half);
                result.kronrod.mquarter += kronrod_weight * quarter;
                result.gauss.mquarter += gauss_weight * quarter;
            }
        }
    }
    return result;
}

Moments cartesian_inner_moments_high_rule(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x,
    const std::vector<std::pair<double, double>>& intervals,
    int order,
    bool need_half,
    bool need_quarter,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    Moments result;
    const auto& [nodes, weights] = gauss_legendre_rule(order);
    for (const auto& interval : intervals) {
        const double midpoint = 0.5 * (interval.first + interval.second);
        const double half_width = 0.5 * (interval.second - interval.first);
        result.m0 += 2.0 * half_width;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const double phase = 0.5 * kPi * nodes[i];
            const double image_y = midpoint + half_width * std::sin(phase);
            if (diagnostics != nullptr) ++diagnostics->lens_evaluation_count;
            const auto mapped = map_lens(lens, image_x, image_y);
            const double dx = mapped.x - source.x;
            const double dy = mapped.y - source.y;
            const double phi = std::max(
                0.0, 1.0 - (dx * dx + dy * dy) /
                    (source_radius * source_radius));
            const double weighted_area = half_width * 0.5 * kPi *
                std::cos(phase) * weights[i];
            const double half = std::sqrt(phi);
            if (need_half) result.mhalf += weighted_area * half;
            if (need_quarter) result.mquarter += weighted_area * std::sqrt(half);
        }
    }
    return result;
}

IntegrandEvaluation evaluate_cartesian_integrand(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x,
    double c,
    double d,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    if (active_integrand_cache != nullptr) {
        const auto found = active_integrand_cache->find(image_x);
        if (found != active_integrand_cache->end()) return found->second;
    }
    const auto intervals = algebraic_cartesian_intervals_at_x(
        lens, source, source_radius, image_x, settings, diagnostics);
    IntegrandEvaluation result;
    if (!intervals.success) {
        result = {false, {}, 0.0, intervals.unsafe_reason};
    } else if (c == 0.0 && d == 0.0) {
        Moments exact;
        for (const auto& interval : intervals.inside_intervals) {
            exact.m0 += interval.second - interval.first;
        }
        result = {true, exact, 0.0, {}};
    } else {
        const auto nested = cartesian_inner_moments_rule(
            lens, source, source_radius, image_x, intervals.inside_intervals,
            gauss_kronrod15(), c != 0.0, d != 0.0, diagnostics);
        const auto high = cartesian_inner_moments_high_rule(
            lens, source, source_radius, image_x, intervals.inside_intervals,
            16, c != 0.0, d != 0.0, diagnostics);
        result = {true, high,
            std::abs(combined_flux(high - nested.kronrod, c, d)), {}};
    }
    if (active_integrand_cache != nullptr) {
        active_integrand_cache->emplace(image_x, result);
    }
    return result;
}

bool x_has_image(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics,
    std::string* reason)
{
    const auto intervals = algebraic_cartesian_intervals_at_x(
        lens, source, source_radius, image_x, settings, diagnostics);
    if (!intervals.success) {
        if (reason != nullptr) *reason = intervals.unsafe_reason;
        return false;
    }
    return !intervals.inside_intervals.empty();
}

BandSearchResult find_cartesian_bands(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    const std::vector<SourcePosition>& seeds,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    std::vector<std::pair<double, double>> bands;
    constexpr int bisection_iterations = 16;
    for (const auto& seed : seeds) {
        if (std::any_of(bands.begin(), bands.end(), [&](const auto& band) {
                return seed.x >= band.first && seed.x <= band.second;
            })) {
            std::string reason;
            if (!x_has_image(
                    lens, source, source_radius, seed.x, settings,
                    diagnostics, &reason)) {
                return {false, {}, reason.empty()
                    ? "covered certified seed was not active" : reason};
            }
            continue;
        }
        std::string reason;
        if (!x_has_image(
                lens, source, source_radius, seed.x, settings, diagnostics, &reason)) {
            return {false, {}, reason.empty()
                ? "certified seed did not lie on an active x scanline" : reason};
        }
        auto search_side = [&](int direction, double* endpoint) {
            double active = seed.x;
            double step = std::max(source_radius, 1.0e-10 * (1.0 + std::abs(seed.x)));
            double outside = active;
            bool found = false;
            for (int i = 0; i < settings.maximum_band_search_steps; ++i) {
                const double candidate = active + static_cast<double>(direction) * step;
                reason.clear();
                const bool inside = x_has_image(
                    lens, source, source_radius, candidate, settings, diagnostics, &reason);
                if (!reason.empty()) return false;
                if (!inside) {
                    outside = candidate;
                    found = true;
                    break;
                }
                active = candidate;
                step *= 2.0;
            }
            if (!found) {
                reason = "Cartesian outer-band search limit reached";
                return false;
            }
            double lo = std::min(active, outside);
            double hi = std::max(active, outside);
            for (int iteration = 0; iteration < bisection_iterations; ++iteration) {
                const double mid = 0.5 * (lo + hi);
                reason.clear();
                const bool inside = x_has_image(
                    lens, source, source_radius, mid, settings, diagnostics, &reason);
                if (!reason.empty()) return false;
                if (direction < 0) {
                    if (inside) hi = mid; else lo = mid;
                } else {
                    if (inside) lo = mid; else hi = mid;
                }
            }
            *endpoint = direction < 0 ? hi : lo;
            return true;
        };
        double lower = seed.x;
        double upper = seed.x;
        if (!search_side(-1, &lower) || !search_side(1, &upper)) {
            return {false, {}, reason.empty() ? "Cartesian band search failed" : reason};
        }
        if (upper > lower) bands.emplace_back(lower, upper);
    }
    std::sort(bands.begin(), bands.end());
    std::vector<std::pair<double, double>> merged;
    for (const auto& band : bands) {
        const double tolerance = 1.0e-11 * (1.0 + std::abs(band.second));
        if (merged.empty() || band.first > merged.back().second + tolerance) {
            merged.push_back(band);
        } else {
            merged.back().second = std::max(merged.back().second, band.second);
        }
    }
    if (merged.empty()) return {false, {}, "no Cartesian outer bands found"};
    diagnostics->radial_band_count = static_cast<int>(merged.size());
    return {true, std::move(merged), {}};
}

RadialIntegralResult integrate_cartesian_interval(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double c,
    double d,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics,
    double lower,
    double upper,
    double tolerance,
    int depth)
{
    return integrate_outer_adaptive(
        [&](double image_x) {
            return evaluate_cartesian_integrand(
                lens, source, source_radius, image_x, c, d, settings, diagnostics);
        },
        c, d, settings, diagnostics, lower, upper, tolerance, depth,
        "Cartesian quadrature depth limit reached");
}

} // namespace

AlgebraicBoundaryResult experimental_algebraic_cartesian_integral(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double limb_darkening_c,
    double limb_darkening_d,
    const std::vector<SourcePosition>& certified_image_seeds,
    const AlgebraicBoundarySettings& settings)
{
    AlgebraicBoundaryResult result;
    RootContinuationWorkspace root_workspace;
    RootWorkspaceScope root_scope(root_workspace);
    std::unordered_map<double, IntegrandEvaluation> integrand_cache;
    integrand_cache.reserve(64);
    IntegrandCacheScope integrand_cache_scope(integrand_cache);
    if (certified_image_seeds.empty()) {
        result.unsafe_reason = "no certified image seeds";
        return result;
    }
    const double normalization = kPi * source_radius * source_radius *
        (1.0 - limb_darkening_c / 3.0 - limb_darkening_d / 5.0);
    if (!(normalization > 0.0) || !std::isfinite(normalization)) {
        result.unsafe_reason = "invalid source flux normalization";
        return result;
    }
    const auto bands = find_cartesian_bands(
        lens, source, source_radius, certified_image_seeds, settings,
        &result.diagnostics);
    if (!bands.success) {
        result.unsafe_reason = bands.unsafe_reason;
        return result;
    }

    Moments preliminary_total;
    double preliminary_error = 0.0;
    for (const auto& band : bands.bands) {
        const auto estimate = integrate_cartesian_interval(
            lens, source, source_radius, limb_darkening_c, limb_darkening_d,
            settings, &result.diagnostics, band.first, band.second,
            std::numeric_limits<double>::infinity(), 0);
        if (!estimate.success) {
            result.unsafe_reason = estimate.unsafe_reason;
            return result;
        }
        preliminary_total = preliminary_total + estimate.value;
        preliminary_error += estimate.error;
    }
    const double preliminary_magnification =
        combined_flux(preliminary_total, limb_darkening_c, limb_darkening_d) /
        normalization;
    const double requested_error =
        std::max(settings.absolute_tolerance, 0.0) +
        std::max(settings.relative_tolerance, 0.0) *
            std::max(std::abs(preliminary_magnification), 1.0);
    const double requested_flux_error = requested_error * normalization;
    if (preliminary_error <= requested_flux_error) {
        result.magnification = preliminary_magnification;
        result.error_estimate = preliminary_error / normalization;
        result.success = std::isfinite(result.magnification);
        return result;
    }

    Moments total;
    double total_error = 0.0;
    const double tolerance_per_band = requested_flux_error /
        static_cast<double>(bands.bands.size());
    for (const auto& band : bands.bands) {
        const auto integrated = integrate_cartesian_interval(
            lens, source, source_radius, limb_darkening_c, limb_darkening_d,
            settings, &result.diagnostics, band.first, band.second,
            tolerance_per_band, 0);
        if (!integrated.success) {
            result.error_estimate = integrated.error / normalization;
            result.unsafe_reason = integrated.unsafe_reason;
            return result;
        }
        total = total + integrated.value;
        total_error += integrated.error;
    }
    result.magnification = combined_flux(
        total, limb_darkening_c, limb_darkening_d) / normalization;
    result.error_estimate = total_error / normalization;
    result.success = std::isfinite(result.magnification);
    if (!result.success) result.unsafe_reason = "non-finite Cartesian magnification";
    return result;
}

} // namespace lcbinint::magnification
