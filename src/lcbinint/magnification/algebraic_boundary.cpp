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

double phi_value_at_point(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x,
    double image_y,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    if (diagnostics != nullptr) {
        ++diagnostics->lens_evaluation_count;
    }
    const auto mapped = map_lens(lens, image_x, image_y);
    if (!std::isfinite(mapped.x) || !std::isfinite(mapped.y)) {
        return -std::numeric_limits<double>::infinity();
    }
    const double dx = mapped.x - source.x;
    const double dy = mapped.y - source.y;
    return 1.0 - (dx * dx + dy * dy) / (source_radius * source_radius);
}

double phi_value(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double radius,
    double angle,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    return phi_value_at_point(
        lens, source, source_radius,
        radius * std::cos(angle), radius * std::sin(angle), diagnostics);
}

std::array<Complex, 5> algebraic_boundary_quartic_fixed(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_radius)
{
    const double a = lens.lens_position;
    const double r = image_radius;
    const double r2 = r * r;
    const Complex shifted_source(source.x + a * lens.m1, source.y);
    const Complex n0 = -shifted_source * r2;
    const Complex n1 = r * (r2 - 1.0 + a * shifted_source);
    const double n2 = a * (lens.m2 - r2);
    const double rho2 = source_radius * source_radius;
    const double radial_denominator_edge = rho2 * a * r2 * r;
    const Complex p0 = n0 * n2;
    const Complex p1 = n0 * std::conj(n1) + n1 * n2 + radial_denominator_edge;
    const double p2 = std::norm(n0) + std::norm(n1) + n2 * n2 -
        rho2 * (r2 * r2 + a * a * r2);
    // The polynomial is Hermitian-palindromic by construction.  Building
    // only its independent half avoids the generic convolution's duplicate
    // complex products on every radial quadrature node.
    return {p0, p1, Complex(p2, 0.0), std::conj(p1), std::conj(p0)};
}

std::array<double, 7> algebraic_cartesian_boundary_sextic_real_fixed(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x)
{
    const Complex imaginary(0.0, 1.0);
    const Complex shifted_source(
        source.x + lens.lens_position * lens.m1, source.y);
    const double denominator0 = image_x * (image_x - lens.lens_position);
    const double denominator1 = 2.0 * image_x - lens.lens_position;
    const Complex shifted_z0 = image_x - shifted_source;
    std::array<Complex, 4> numerator {
        shifted_z0 * denominator0 -
            lens.m1 * image_x -
            lens.m2 * (image_x - lens.lens_position),
        shifted_z0 * (-imaginary * denominator1) +
            imaginary * denominator0 +
            imaginary * (lens.m1 + lens.m2),
        -shifted_z0 + denominator1,
        -imaginary,
    };
    const auto twice_real_product = [](Complex lhs, Complex rhs) {
        return 2.0 * std::real(lhs * std::conj(rhs));
    };
    std::array<double, 7> coefficients {
        std::norm(numerator[0]),
        twice_real_product(numerator[0], numerator[1]),
        std::norm(numerator[1]) +
            twice_real_product(numerator[0], numerator[2]),
        twice_real_product(numerator[0], numerator[3]) +
            twice_real_product(numerator[1], numerator[2]),
        std::norm(numerator[2]) +
            twice_real_product(numerator[1], numerator[3]),
        twice_real_product(numerator[2], numerator[3]),
        std::norm(numerator[3]),
    };
    const double rho2 = source_radius * source_radius;
    coefficients[0] -= rho2 * denominator0 * denominator0;
    coefficients[2] -= rho2 *
        (denominator1 * denominator1 - 2.0 * denominator0);
    coefficients[4] -= rho2;
    return coefficients;
}

std::array<Complex, 7> algebraic_cartesian_boundary_sextic_fixed(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x)
{
    const auto real_coefficients = algebraic_cartesian_boundary_sextic_real_fixed(
        lens, source, source_radius, image_x);
    std::array<Complex, 7> coefficients;
    std::transform(
        real_coefficients.begin(), real_coefficients.end(), coefficients.begin(),
        [](double value) { return Complex(value, 0.0); });
    return coefficients;
}

bool real_cartesian_boundary_coefficients(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x,
    std::array<double, 7>* real_coefficients,
    std::array<Complex, 7>* complex_coefficients = nullptr)
{
    *real_coefficients = algebraic_cartesian_boundary_sextic_real_fixed(
        lens, source, source_radius, image_x);
    for (const double coefficient : *real_coefficients) {
        if (!std::isfinite(coefficient)) return false;
    }
    if (complex_coefficients != nullptr) {
        std::transform(
            real_coefficients->begin(), real_coefficients->end(),
            complex_coefficients->begin(),
            [](double value) { return Complex(value, 0.0); });
    }
    return true;
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

double evaluate_real_polynomial(
    const std::array<double, 5>& coefficients,
    int degree,
    double x)
{
    double value = coefficients[static_cast<std::size_t>(degree)];
    for (int index = degree - 1; index >= 0; --index) {
        value = value * x + coefficients[static_cast<std::size_t>(index)];
    }
    return value;
}

double evaluate_real_derivative(
    const std::array<double, 5>& coefficients,
    int degree,
    double x)
{
    double value = degree * coefficients[static_cast<std::size_t>(degree)];
    for (int index = degree - 1; index > 0; --index) {
        value = value * x + index * coefficients[static_cast<std::size_t>(index)];
    }
    return value;
}

int real_quadratic_roots(double a, double b, double c, std::array<double, 3>* roots)
{
    const double scale = std::max({std::abs(a), std::abs(b), std::abs(c), 1.0});
    if (std::abs(a) <= 8.0 * std::numeric_limits<double>::epsilon() * scale) {
        if (std::abs(b) <= 8.0 * std::numeric_limits<double>::epsilon() * scale) return 0;
        (*roots)[0] = -c / b;
        return 1;
    }
    const double discriminant = b * b - 4.0 * a * c;
    const double discriminant_scale = b * b + std::abs(4.0 * a * c) + 1.0;
    if (discriminant < -32.0 * std::numeric_limits<double>::epsilon() *
            discriminant_scale) {
        return 0;
    }
    const double square_root = std::sqrt(std::max(0.0, discriminant));
    const double q = -0.5 * (b + std::copysign(square_root, b));
    if (q == 0.0) {
        (*roots)[0] = -0.5 * b / a;
        return 1;
    }
    (*roots)[0] = q / a;
    (*roots)[1] = c / q;
    if ((*roots)[1] < (*roots)[0]) std::swap((*roots)[0], (*roots)[1]);
    const double duplicate_tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
        (1.0 + std::max(std::abs((*roots)[0]), std::abs((*roots)[1])));
    return std::abs((*roots)[1] - (*roots)[0]) <= duplicate_tolerance ? 1 : 2;
}

int real_cubic_roots(double a, double b, double c, double d,
                     std::array<double, 3>* roots)
{
    const double scale = std::max(
        {std::abs(a), std::abs(b), std::abs(c), std::abs(d), 1.0});
    if (std::abs(a) <= 8.0 * std::numeric_limits<double>::epsilon() * scale) {
        return real_quadratic_roots(b, c, d, roots);
    }
    const double aa = b / a;
    const double p = c / a - aa * aa / 3.0;
    const double q = 2.0 * aa * aa * aa / 27.0 - aa * c / (3.0 * a) + d / a;
    const double half_q = 0.5 * q;
    const double third_p = p / 3.0;
    const double discriminant = half_q * half_q + third_p * third_p * third_p;
    const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() *
        (half_q * half_q + std::abs(third_p * third_p * third_p) + 1.0);
    if (discriminant > tolerance) {
        const double square_root = std::sqrt(discriminant);
        (*roots)[0] = std::cbrt(-half_q + square_root) +
            std::cbrt(-half_q - square_root) - aa / 3.0;
        return 1;
    }
    if (std::abs(p) <= tolerance && std::abs(q) <= tolerance) {
        (*roots)[0] = -aa / 3.0;
        return 1;
    }
    if (p >= 0.0) {
        (*roots)[0] = 2.0 * std::cbrt(-half_q) - aa / 3.0;
        return 1;
    }
    const double radius = 2.0 * std::sqrt(-third_p);
    const double cosine = std::clamp(
        -half_q / std::sqrt(-(third_p * third_p * third_p)), -1.0, 1.0);
    const double phase = std::acos(cosine);
    for (int index = 0; index < 3; ++index) {
        (*roots)[static_cast<std::size_t>(index)] =
            radius * std::cos((phase + kTwoPi * index) / 3.0) - aa / 3.0;
    }
    std::sort(roots->begin(), roots->end());
    return 3;
}

struct RealPolynomialRoots {
    std::array<double, 6> values {};
    int count = 0;
};

double evaluate_real_polynomial(
    const std::array<double, 7>& coefficients,
    int degree,
    double x)
{
    double value = coefficients[static_cast<std::size_t>(degree)];
    for (int index = degree - 1; index >= 0; --index) {
        value = value * x + coefficients[static_cast<std::size_t>(index)];
    }
    return value;
}

double evaluate_real_derivative(
    const std::array<double, 7>& coefficients,
    int degree,
    double x)
{
    double value = degree * coefficients[static_cast<std::size_t>(degree)];
    for (int index = degree - 1; index > 0; --index) {
        value = value * x + index * coefficients[static_cast<std::size_t>(index)];
    }
    return value;
}

double real_polynomial_scale(
    const std::array<double, 7>& coefficients,
    int degree,
    double x)
{
    const double magnitude = std::abs(x);
    double scale = std::abs(coefficients[static_cast<std::size_t>(degree)]);
    for (int index = degree - 1; index >= 0; --index) {
        scale = scale * magnitude +
            std::abs(coefficients[static_cast<std::size_t>(index)]);
    }
    return std::max(scale, 1.0e-300);
}

bool isolate_real_polynomial_roots(
    const std::array<double, 7>& coefficients,
    int degree,
    RealPolynomialRoots* roots)
{
    roots->count = 0;
    double coefficient_scale = 0.0;
    for (int index = 0; index <= degree; ++index) {
        coefficient_scale = std::max(
            coefficient_scale,
            std::abs(coefficients[static_cast<std::size_t>(index)]));
    }
    if (!(coefficient_scale > 0.0) || !std::isfinite(coefficient_scale) ||
        std::abs(coefficients[static_cast<std::size_t>(degree)]) <=
            64.0 * std::numeric_limits<double>::epsilon() * coefficient_scale) {
        return false;
    }
    if (degree == 1) {
        const double root = -coefficients[0] / coefficients[1];
        if (!std::isfinite(root)) return false;
        roots->values[0] = root;
        roots->count = 1;
        return true;
    }

    std::array<double, 7> derivative {};
    for (int index = 1; index <= degree; ++index) {
        derivative[static_cast<std::size_t>(index - 1)] =
            static_cast<double>(index) *
            coefficients[static_cast<std::size_t>(index)];
    }
    RealPolynomialRoots critical_points;
    if (!isolate_real_polynomial_roots(
            derivative, degree - 1, &critical_points)) {
        return false;
    }

    const double leading =
        std::abs(coefficients[static_cast<std::size_t>(degree)]);
    double bound = 1.0;
    for (int index = 0; index < degree; ++index) {
        bound = std::max(
            bound,
            1.0 + std::abs(coefficients[static_cast<std::size_t>(index)]) /
                leading);
    }
    if (!std::isfinite(bound) || bound > 1.0e14) return false;

    std::array<double, 8> partition {};
    int partition_count = 0;
    partition[static_cast<std::size_t>(partition_count++)] = -bound;
    for (int index = 0; index < critical_points.count; ++index) {
        const double critical =
            critical_points.values[static_cast<std::size_t>(index)];
        if (!(critical > -bound && critical < bound)) continue;
        const double value = evaluate_real_polynomial(coefficients, degree, critical);
        const double scale = real_polynomial_scale(coefficients, degree, critical);
        // An even-multiplicity root does not delimit an inside interval, but
        // a numerically unresolved tangency can hide a very close pair of
        // crossings.  Let the certified complex solver handle that rare case.
        if (!std::isfinite(value) ||
            std::abs(value) <= 256.0 * std::numeric_limits<double>::epsilon() * scale) {
            return false;
        }
        partition[static_cast<std::size_t>(partition_count++)] = critical;
    }
    partition[static_cast<std::size_t>(partition_count++)] = bound;

    const auto append_root = [&](double root) {
        if (!std::isfinite(root) || roots->count >= degree) return false;
        if (roots->count > 0) {
            const double previous =
                roots->values[static_cast<std::size_t>(roots->count - 1)];
            const double tolerance = 256.0 * std::numeric_limits<double>::epsilon() *
                (1.0 + std::max(std::abs(previous), std::abs(root)));
            if (root - previous <= tolerance) return false;
        }
        roots->values[static_cast<std::size_t>(roots->count++)] = root;
        return true;
    };

    for (int interval = 0; interval + 1 < partition_count; ++interval) {
        double lower = partition[static_cast<std::size_t>(interval)];
        double upper = partition[static_cast<std::size_t>(interval + 1)];
        double lower_value = evaluate_real_polynomial(coefficients, degree, lower);
        double upper_value = evaluate_real_polynomial(coefficients, degree, upper);
        if (!std::isfinite(lower_value) || !std::isfinite(upper_value)) return false;
        if (std::signbit(lower_value) == std::signbit(upper_value)) continue;

        double x = lower - lower_value * (upper - lower) /
            (upper_value - lower_value);
        if (!(x > lower && x < upper) || !std::isfinite(x)) {
            x = 0.5 * (lower + upper);
        }
        double root = x;
        for (int iteration = 0; iteration < 48; ++iteration) {
            const double value = evaluate_real_polynomial(coefficients, degree, x);
            const double scale = real_polynomial_scale(coefficients, degree, x);
            if (!std::isfinite(value) || !std::isfinite(scale)) return false;
            if (std::abs(value) <=
                    16.0 * std::numeric_limits<double>::epsilon() * scale) {
                root = x;
                break;
            }
            if (std::signbit(value) == std::signbit(lower_value)) {
                lower = x;
                lower_value = value;
            } else {
                upper = x;
                upper_value = value;
            }
            if (upper - lower <= 8.0 * std::numeric_limits<double>::epsilon() *
                    (1.0 + std::abs(x))) {
                root = 0.5 * (lower + upper);
                break;
            }
            const double slope =
                evaluate_real_derivative(coefficients, degree, x);
            double candidate = slope != 0.0 ? x - value / slope
                                            : 0.5 * (lower + upper);
            if (!(candidate > lower && candidate < upper) ||
                !std::isfinite(candidate)) {
                candidate = 0.5 * (lower + upper);
            }
            x = candidate;
            root = x;
        }
        const double residual =
            std::abs(evaluate_real_polynomial(coefficients, degree, root));
        const double residual_scale =
            real_polynomial_scale(coefficients, degree, root);
        if (!std::isfinite(residual) ||
            residual > 2.0e-12 * residual_scale || !append_root(root)) {
            return false;
        }
    }
    return true;
}

struct SelfInversiveTrigQuartic {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    double e = 0.0;
    double scale = 1.0;
};

template <typename Coefficients>
bool self_inversive_trig_quartic(
    const Coefficients& coefficients,
    SelfInversiveTrigQuartic* result)
{
    if (coefficients.size() != 5) return false;
    double coefficient_scale = 1.0;
    for (const Complex coefficient : coefficients) {
        coefficient_scale = std::max(coefficient_scale, std::abs(coefficient));
    }
    const double tolerance = 2.0e-12 * coefficient_scale;
    if (std::abs(coefficients[0] - std::conj(coefficients[4])) > tolerance ||
        std::abs(coefficients[1] - std::conj(coefficients[3])) > tolerance ||
        std::abs(coefficients[2].imag()) > tolerance) {
        return false;
    }
    result->a = coefficients[4].real();
    result->b = coefficients[4].imag();
    result->c = coefficients[3].real();
    result->d = coefficients[3].imag();
    result->e = coefficients[2].real();
    result->scale = coefficient_scale;
    return true;
}

std::array<double, 3> evaluate_trig_quartic(
    const SelfInversiveTrigQuartic& q,
    double theta)
{
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    const double cosine2 = cosine * cosine - sine * sine;
    const double sine2 = 2.0 * sine * cosine;
    return {{
        q.e + 2.0 * (q.c * cosine - q.d * sine) +
            2.0 * (q.a * cosine2 - q.b * sine2),
        2.0 * (-q.c * sine - q.d * cosine) +
            4.0 * (-q.a * sine2 - q.b * cosine2),
        2.0 * (-q.c * cosine + q.d * sine) +
            8.0 * (-q.a * cosine2 + q.b * sine2),
    }};
}

template <typename Coefficients>
bool self_inversive_unit_roots(
    const Coefficients& complex_coefficients,
    std::vector<Complex>* roots)
{
    SelfInversiveTrigQuartic trig;
    if (!self_inversive_trig_quartic(complex_coefficients, &trig)) return false;
    const double a = trig.a;
    const double b = trig.b;
    const double c = trig.c;
    const double d = trig.d;
    const double e = trig.e;
    std::array<double, 5> coefficients {{
        e + 2.0 * c + 2.0 * a,
        -4.0 * d - 8.0 * b,
        2.0 * e - 12.0 * a,
        -4.0 * d + 8.0 * b,
        e - 2.0 * c + 2.0 * a,
    }};
    double scale = 1.0;
    for (double coefficient : coefficients) scale = std::max(scale, std::abs(coefficient));
    int degree = 4;
    const double degree_tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    const bool root_at_infinity = std::abs(coefficients[4]) <= degree_tolerance;
    while (degree > 0 && std::abs(coefficients[static_cast<std::size_t>(degree)]) <=
            degree_tolerance) {
        --degree;
    }
    if (degree == 0) return false;

    std::array<double, 3> critical_points {};
    int critical_count = 0;
    if (degree == 4) {
        critical_count = real_cubic_roots(
            4.0 * coefficients[4], 3.0 * coefficients[3],
            2.0 * coefficients[2], coefficients[1], &critical_points);
    } else if (degree == 3) {
        critical_count = real_quadratic_roots(
            3.0 * coefficients[3], 2.0 * coefficients[2],
            coefficients[1], &critical_points);
    } else if (degree == 2) {
        if (coefficients[2] != 0.0) {
            critical_points[0] = -coefficients[1] / (2.0 * coefficients[2]);
            critical_count = 1;
        }
    }
    std::sort(critical_points.begin(), critical_points.begin() + critical_count);
    const double leading = std::abs(coefficients[static_cast<std::size_t>(degree)]);
    double bound = 1.0;
    for (int index = 0; index < degree; ++index) {
        bound = std::max(bound, 1.0 +
            std::abs(coefficients[static_cast<std::size_t>(index)]) / leading);
    }
    if (!std::isfinite(bound) || bound > 1.0e14) return false;

    std::array<double, 5> partition {};
    int partition_count = 0;
    partition[partition_count++] = -bound;
    for (int index = 0; index < critical_count; ++index) {
        if (critical_points[static_cast<std::size_t>(index)] > -bound &&
            critical_points[static_cast<std::size_t>(index)] < bound) {
            partition[partition_count++] = critical_points[static_cast<std::size_t>(index)];
        }
    }
    partition[partition_count++] = bound;

    std::array<double, 4> real_roots {};
    int root_count = 0;
    const auto append_root = [&](double value) {
        const double tolerance = 2.0e-11 * (1.0 + std::abs(value));
        if (root_count > 0 && std::abs(real_roots[static_cast<std::size_t>(root_count - 1)] -
                value) <= tolerance) {
            return;
        }
        if (root_count < 4) real_roots[static_cast<std::size_t>(root_count++)] = value;
    };
    for (int index = 1; index + 1 < partition_count; ++index) {
        const double x = partition[static_cast<std::size_t>(index)];
        const double value = evaluate_real_polynomial(coefficients, degree, x);
        double value_scale = 1.0;
        double power = 1.0;
        for (int coefficient_index = 0; coefficient_index <= degree; ++coefficient_index) {
            value_scale += std::abs(coefficients[static_cast<std::size_t>(coefficient_index)]) *
                power;
            power *= std::abs(x);
        }
        if (std::abs(value) <= 2.0e-12 * value_scale) {
            append_root(x);
            append_root(x + 8.0e-12 * (1.0 + std::abs(x)));
        }
    }
    for (int interval = 0; interval + 1 < partition_count; ++interval) {
        double lower = partition[static_cast<std::size_t>(interval)];
        double upper = partition[static_cast<std::size_t>(interval + 1)];
        double lower_value = evaluate_real_polynomial(coefficients, degree, lower);
        double upper_value = evaluate_real_polynomial(coefficients, degree, upper);
        if (!std::isfinite(lower_value) || !std::isfinite(upper_value)) return false;
        if (std::signbit(lower_value) == std::signbit(upper_value) ||
            lower_value == 0.0 || upper_value == 0.0) {
            continue;
        }
        double x = lower - lower_value * (upper - lower) / (upper_value - lower_value);
        if (!(x > lower && x < upper) || !std::isfinite(x)) x = 0.5 * (lower + upper);
        double root = x;
        for (int iteration = 0; iteration < 40; ++iteration) {
            const double value = evaluate_real_polynomial(coefficients, degree, x);
            double value_scale = 1.0;
            double power = 1.0;
            for (int coefficient_index = 0; coefficient_index <= degree;
                 ++coefficient_index) {
                value_scale += std::abs(
                    coefficients[static_cast<std::size_t>(coefficient_index)]) * power;
                power *= std::abs(x);
            }
            if (std::abs(value) <= 8.0 * std::numeric_limits<double>::epsilon() *
                    value_scale) {
                root = x;
                break;
            }
            if (std::signbit(value) == std::signbit(lower_value)) {
                lower = x;
                lower_value = value;
            } else {
                upper = x;
                upper_value = value;
            }
            if (upper - lower <= 4.0e-14 * (1.0 + std::abs(x))) {
                root = 0.5 * (lower + upper);
                break;
            }
            const double derivative = evaluate_real_derivative(coefficients, degree, x);
            double candidate = derivative != 0.0 ? x - value / derivative
                                                 : 0.5 * (lower + upper);
            if (!(candidate > lower && candidate < upper) || !std::isfinite(candidate)) {
                candidate = 0.5 * (lower + upper);
            }
            x = candidate;
            root = x;
        }
        append_root(root);
    }
    if (root_at_infinity && root_count < 4) {
        roots->emplace_back(-1.0, 0.0);
    }
    for (int index = 0; index < root_count; ++index) {
        double t = real_roots[static_cast<std::size_t>(index)];
        for (int polish = 0; polish < 4; ++polish) {
            const double value = evaluate_real_polynomial(coefficients, degree, t);
            const double derivative = evaluate_real_derivative(coefficients, degree, t);
            if (derivative == 0.0 || !std::isfinite(derivative)) break;
            const double correction = value / derivative;
            t -= correction;
            if (!std::isfinite(t) || std::abs(correction) <=
                    8.0 * std::numeric_limits<double>::epsilon() *
                        (1.0 + std::abs(t))) {
                break;
            }
        }
        double cosine;
        double sine;
        if (std::abs(t) <= 1.0) {
            const double denominator = 1.0 + t * t;
            cosine = (1.0 - t * t) / denominator;
            sine = 2.0 * t / denominator;
        } else {
            const double inverse = 1.0 / t;
            const double denominator = 1.0 + inverse * inverse;
            cosine = (inverse * inverse - 1.0) / denominator;
            sine = 2.0 * inverse / denominator;
        }
        roots->emplace_back(cosine, sine);
    }
    if (roots->size() > 4 || (roots->size() % 2) != 0) return false;
    double residual_scale = 1.0;
    for (const Complex coefficient : complex_coefficients) {
        residual_scale += std::abs(coefficient);
    }
    // Roots produced by the tan(theta/2) map lie on the unit circle by
    // construction, so their polynomial scale is shared.  Recomputing five
    // coefficient moduli and four powers for every root was pure overhead.
    residual_scale *= 1.0 + 2.0e-13;
    for (const Complex root : *roots) {
        Complex residual = complex_coefficients.back();
        for (int index = static_cast<int>(complex_coefficients.size()) - 2;
             index >= 0; --index) {
            residual = residual * root +
                complex_coefficients[static_cast<std::size_t>(index)];
        }
        if (std::abs(residual) > 2.0e-13 * residual_scale) {
            roots->clear();
            return false;
        }
    }
    return true;
}

struct GaussKronrod15Rule {
    std::array<double, 15> nodes;
    std::array<double, 15> kronrod_weights;
    // Zero at Kronrod-only nodes, allowing both estimates to be accumulated
    // in one pass without a lookup or a second integrand evaluation.
    std::array<double, 15> gauss_weights;
};

struct GaussKronrod21Rule {
    std::array<double, 21> nodes;
    std::array<double, 21> kronrod_weights;
    std::array<double, 21> gauss_weights;
};

template <std::size_t N>
struct SineMappedNodes {
    std::array<double, N> coordinates {};
    std::array<double, N> jacobians {};
};

template <std::size_t N>
SineMappedNodes<N> make_sine_mapped_nodes(const std::array<double, N>& nodes)
{
    SineMappedNodes<N> mapped;
    for (std::size_t i = 0; i < N; ++i) {
        const double phase = 0.5 * kPi * nodes[i];
        mapped.coordinates[i] = std::sin(phase);
        mapped.jacobians[i] = 0.5 * kPi * std::cos(phase);
    }
    return mapped;
}

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

const GaussKronrod21Rule& gauss_kronrod21()
{
    static const GaussKronrod21Rule rule {{
        -0.9956571630258081, -0.9739065285171717,
        -0.9301574913557082, -0.8650633666889845,
        -0.7808177265864169, -0.6794095682990244,
        -0.5627571346686047, -0.4333953941292472,
        -0.2943928627014602, -0.1488743389816312,
         0.0,
         0.1488743389816312,  0.2943928627014602,
         0.4333953941292472,  0.5627571346686047,
         0.6794095682990244,  0.7808177265864169,
         0.8650633666889845,  0.9301574913557082,
         0.9739065285171717,  0.9956571630258081,
    }, {
        0.01169463886737187, 0.03255816230796473,
        0.05475589657435199, 0.07503967481091995,
        0.09312545458369761, 0.1093871588022976,
        0.1234919762620659,  0.1347092173114733,
        0.1427759385770601,  0.1477391049013385,
        0.1494455540029169,
        0.1477391049013385,  0.1427759385770601,
        0.1347092173114733,  0.1234919762620659,
        0.1093871588022976,  0.09312545458369761,
        0.07503967481091995, 0.05475589657435199,
        0.03255816230796473, 0.01169463886737187,
    }, {
        0.0, 0.06667134430868814,
        0.0, 0.1494513491505806,
        0.0, 0.2190863625159820,
        0.0, 0.2692667193099964,
        0.0, 0.2955242247147529,
        0.0,
        0.2955242247147529, 0.0,
        0.2692667193099964, 0.0,
        0.2190863625159820, 0.0,
        0.1494513491505806, 0.0,
        0.06667134430868814, 0.0,
    }};
    return rule;
}

const SineMappedNodes<15>& sine_mapped_gauss_kronrod15()
{
    static const auto mapped = make_sine_mapped_nodes(gauss_kronrod15().nodes);
    return mapped;
}

const SineMappedNodes<21>& sine_mapped_gauss_kronrod21()
{
    static const auto mapped = make_sine_mapped_nodes(gauss_kronrod21().nodes);
    return mapped;
}

struct GaussLegendre16Rule {
    std::array<double, 16> nodes {{
        -0.9894009349916499, -0.9445750230732326,
        -0.8656312023878318, -0.7554044083550030,
        -0.6178762444026438, -0.4580167776572274,
        -0.2816035507792589, -0.09501250983763744,
         0.09501250983763744, 0.2816035507792589,
         0.4580167776572274, 0.6178762444026438,
         0.7554044083550030, 0.8656312023878318,
         0.9445750230732326, 0.9894009349916499,
    }};
    std::array<double, 16> weights {{
        0.02715245941175409, 0.06225352393864789,
        0.09515851168249278, 0.1246289712555339,
        0.1495959888165767, 0.1691565193950025,
        0.1826034150449236, 0.1894506104550685,
        0.1894506104550685, 0.1826034150449236,
        0.1691565193950025, 0.1495959888165767,
        0.1246289712555339, 0.09515851168249278,
        0.06225352393864789, 0.02715245941175409,
    }};
};

const GaussLegendre16Rule& gauss_legendre16()
{
    static const GaussLegendre16Rule rule;
    return rule;
}

const SineMappedNodes<16>& sine_mapped_gauss_legendre16()
{
    static const auto mapped = make_sine_mapped_nodes(gauss_legendre16().nodes);
    return mapped;
}

// ---------------------------------------------------------------------------
// Proposal 1: algebraic limb-darkening moment series.
//
// On an angular interval bounded by a simple crossing pair t_- < t_+ (with
// t = tan(theta/2)) the finite-source indicator factors exactly as
//     phi(R, theta(t)) = h^2 (1 - x^2) G_R(m + h x),        t = m + h x,
// with m = (t_- + t_+)/2, h = (t_+ - t_-)/2 and
//     G_R(t) = S_2(t) / ( rho^2 R^2 (1 + t^2) B_2(t) ),
//     B_2(t) = (R - a)^2 + (R + a)^2 t^2,
// where S_2(t) is the quadratic left after deflating the real quartic P_4(t)
// by (t - t_-)(t_+ - t).  The moment M_beta(R) = R \int phi^beta dtheta is then
//     M_beta = 2 R h^{2 beta + 1} \int_{-1}^{1} (1 - x^2)^beta g_beta(x) dx,
//     g_beta(x) = G_R(m + h x)^beta / (1 + (m + h x)^2),
// and with g_beta(x) = sum_n a_n x^n this collapses to the closed form
//     M_beta = 2 R h^{2 beta + 1} sum_j a_{2j} Beta(j + 1/2, beta + 1).
// The a_n obey a recurrence from L(x) g_beta'(x) = W_beta(x) g_beta(x) with
//     L      = S A B                                        (deg <= 6),
//     W_beta = beta S' A B - (beta + 1) S A' B - beta S A B' (deg <= 5),
// where S(x) = S_2(m + h x), A(x) = 1 + (m + h x)^2, B(x) = B_2(m + h x).
//
// The series replaces the sine-mapped Gauss-Kronrod 21 angular quadrature only
// when it is provably safe on this interval: the deflation remainder is
// negligible, S_2(m) > 0, the nearest complex singularity of g_beta in the
// x-plane sits well outside the unit disc, the closed-form phi matches the
// trusted phi_value sampler at interior probes, and the moment series
// converges inside the coefficient budget.  Any failure falls back unchanged.
struct AngularMomentSeries {
    bool ok = false;
    double half_full = 0.0;
    double half_reduced = 0.0;
    double quarter_full = 0.0;
    double quarter_reduced = 0.0;
};

// Substitute t = m + h*x into p0 + p1 t + p2 t^2, returning coefficients in x.
inline std::array<double, 3> shift_quadratic_to_x(
    const std::array<double, 3>& p, double m, double h)
{
    return {{
        p[0] + p[1] * m + p[2] * m * m,
        h * (p[1] + 2.0 * p[2] * m),
        p[2] * h * h,
    }};
}

template <std::size_t NA, std::size_t NB, std::size_t NOut>
inline void poly_mul_add(
    const std::array<double, NA>& lhs, std::size_t lhs_degree,
    const std::array<double, NB>& rhs, std::size_t rhs_degree,
    double scale, std::array<double, NOut>* out)
{
    for (std::size_t i = 0; i <= lhs_degree; ++i) {
        for (std::size_t j = 0; j <= rhs_degree; ++j) {
            (*out)[i + j] += scale * lhs[i] * rhs[j];
        }
    }
}

AngularMomentSeries algebraic_moment_series(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double radius,
    double interval_lower,
    double interval_upper,
    const std::array<double, 5>& real_quartic,
    bool need_half,
    bool need_quarter,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    AngularMomentSeries out;

    // The tan(theta/2) chart is singular at theta = pi (mod 2*pi).  Reject any
    // interval that reaches or straddles it; a wrap through theta = 0 is fine.
    const double kEdge = 1.0e-6;
    for (double pole = kPi; pole < interval_upper + kPi; pole += kTwoPi) {
        if (pole > interval_lower + kEdge && pole < interval_upper - kEdge) {
            return out;
        }
        if (std::abs(pole - interval_lower) <= kEdge ||
            std::abs(pole - interval_upper) <= kEdge) {
            return out;
        }
    }

    double theta_lo = interval_lower;
    double theta_hi = interval_upper;
    if (theta_hi >= kTwoPi) {  // cyclic interval unwrapped through theta = 0
        theta_lo -= kTwoPi;
        theta_hi -= kTwoPi;
    }
    const double t_lo = std::tan(0.5 * theta_lo);
    const double t_hi = std::tan(0.5 * theta_hi);
    if (!std::isfinite(t_lo) || !std::isfinite(t_hi) || !(t_hi > t_lo)) {
        return out;
    }
    const double m = 0.5 * (t_lo + t_hi);
    const double h = 0.5 * (t_hi - t_lo);
    if (!std::isfinite(m) || !(h > 0.0) || std::abs(m) > 60.0 || h > 60.0) {
        return out;
    }

    double quartic_scale_pre = 1.0;
    for (double coefficient : real_quartic) {
        quartic_scale_pre = std::max(quartic_scale_pre, std::abs(coefficient));
    }
    // The reconstructed quartic carries an arbitrary overall sign; fix it so
    // P_4(m) > 0, matching the strictly positive true numerator of phi at the
    // interior midpoint.
    double p_mid = real_quartic[4];
    for (int index = 3; index >= 0; --index) {
        p_mid = p_mid * m + real_quartic[static_cast<std::size_t>(index)];
    }
    if (std::abs(p_mid) <= 1.0e-12 * quartic_scale_pre) {
        return out;
    }
    const double quartic_sign = p_mid > 0.0 ? 1.0 : -1.0;

    // Deflate the real quartic by the monic quadratic t^2 - 2 m t + (m^2 - h^2).
    const double c0 = m * m - h * h;
    const double p4 = quartic_sign * real_quartic[4];
    const double p3 = quartic_sign * real_quartic[3];
    const double p2 = quartic_sign * real_quartic[2];
    const double p1 = quartic_sign * real_quartic[1];
    const double p0 = quartic_sign * real_quartic[0];
    const double q2 = p4;
    const double q1 = p3 + 2.0 * m * p4;
    const double q0 = p2 - p4 * c0 + 2.0 * m * q1;
    const double r1 = p1 - q1 * c0 + 2.0 * m * q0;
    const double r0 = p0 - q0 * c0;
    if (std::abs(r0) + std::abs(r1) > 1.0e-7 * quartic_scale_pre) {
        return out;  // (m, h^2) is not an accurate crossing pair of P_4
    }

    // S_2(t) = -(q0 + q1 t + q2 t^2): the residual quadratic, signed so that
    // S_2 > 0 wherever phi > 0 between the crossings.
    const std::array<double, 3> s_t {{-q0, -q1, -q2}};
    const std::array<double, 3> a_t {{1.0, 0.0, 1.0}};
    const double edge_minus = radius - lens.lens_position;
    const double edge_plus = radius + lens.lens_position;
    const std::array<double, 3> b_t {{edge_minus * edge_minus, 0.0,
                                      edge_plus * edge_plus}};

    const std::array<double, 3> s_x = shift_quadratic_to_x(s_t, m, h);
    const std::array<double, 3> a_x = shift_quadratic_to_x(a_t, m, h);
    const std::array<double, 3> b_x = shift_quadratic_to_x(b_t, m, h);
    if (!(s_x[0] > 0.0) || !(b_x[0] > 0.0) || !(a_x[0] > 0.0)) {
        return out;
    }

    // Nearest complex singularity of g_beta in the x-plane: poles of 1/A,
    // zeros of B_2, and zeros of S_2, all mapped through t = m + h x.
    double convergence_radius = std::numeric_limits<double>::infinity();
    convergence_radius = std::min(
        convergence_radius, std::sqrt(m * m + 1.0) / h);
    {
        const double k = std::abs(edge_minus) / edge_plus;
        convergence_radius = std::min(
            convergence_radius, std::sqrt(m * m + k * k) / h);
    }
    {
        const double disc = q1 * q1 - 4.0 * q0 * q2;
        if (std::abs(q2) > 0.0) {
            if (disc >= 0.0) {
                const double root = std::sqrt(disc);
                for (const double sign : {1.0, -1.0}) {
                    const double t_root = (-q1 + sign * root) / (2.0 * q2);
                    convergence_radius = std::min(
                        convergence_radius, std::abs((t_root - m) / h));
                }
            } else {
                const double re = -q1 / (2.0 * q2) - m;
                const double im = std::sqrt(-disc) / (2.0 * q2);
                convergence_radius = std::min(
                    convergence_radius, std::sqrt(re * re + im * im) / h);
            }
        }
    }
    if (!std::isfinite(convergence_radius) || convergence_radius < 1.8) {
        return out;
    }

    int half_terms = static_cast<int>(
        std::ceil(13.0 / std::log(convergence_radius)));
    half_terms = std::min(std::max(half_terms, 6), 20);
    const int coefficient_count = 2 * half_terms + 3;
    const int reduced_terms = half_terms / 2;

    const auto run_beta = [&](double beta, double* full, double* reduced) -> bool {
        const double a_x0 = a_x[0];
        const double c_radial =
            std::pow(source_radius * source_radius * radius * radius, -beta);
        const double a0 = c_radial * std::pow(s_x[0], beta) /
            (std::pow(a_x0, beta + 1.0) * std::pow(b_x[0], beta));
        if (!std::isfinite(a0)) return false;

        // L = S A B (deg 6), W = beta S' A B - (beta+1) S A' B - beta S A B'.
        std::array<double, 5> sa {};
        poly_mul_add(s_x, 2, a_x, 2, 1.0, &sa);
        std::array<double, 7> el {};
        poly_mul_add(sa, 4, b_x, 2, 1.0, &el);

        const std::array<double, 3> s_prime {{s_x[1], 2.0 * s_x[2], 0.0}};
        const std::array<double, 3> a_prime {{a_x[1], 2.0 * a_x[2], 0.0}};
        const std::array<double, 3> b_prime {{b_x[1], 2.0 * b_x[2], 0.0}};
        std::array<double, 4> sa_prime {};   // S' A  (deg 3)
        poly_mul_add(s_prime, 1, a_x, 2, 1.0, &sa_prime);
        std::array<double, 4> s_ap {};       // S A'  (deg 3)
        poly_mul_add(s_x, 2, a_prime, 1, 1.0, &s_ap);
        std::array<double, 6> wb {};
        poly_mul_add(sa_prime, 3, b_x, 2, beta, &wb);
        poly_mul_add(s_ap, 3, b_x, 2, -(beta + 1.0), &wb);
        poly_mul_add(sa, 4, b_prime, 1, -beta, &wb);

        const double l0 = el[0];
        if (!(std::abs(l0) > 0.0)) return false;

        std::array<double, 45> a {};
        a[0] = a0;
        for (int n = 0; n + 1 < coefficient_count; ++n) {
            double acc = 0.0;
            for (int k = 0; k <= 5 && k <= n; ++k) {
                acc += wb[static_cast<std::size_t>(k)] *
                    a[static_cast<std::size_t>(n - k)];
            }
            for (int k = 1; k <= 6 && k <= n; ++k) {
                acc -= el[static_cast<std::size_t>(k)] *
                    static_cast<double>(n - k + 1) *
                    a[static_cast<std::size_t>(n - k + 1)];
            }
            const double next = acc / (l0 * static_cast<double>(n + 1));
            if (!std::isfinite(next)) return false;
            a[static_cast<std::size_t>(n + 1)] = next;
        }

        // M_beta / (2 R h^{2 beta + 1}) = sum_j a_{2j} Beta(j + 1/2, beta + 1).
        double beta_coefficient = std::tgamma(0.5) * std::tgamma(beta + 1.0) /
            std::tgamma(beta + 1.5);
        double partial = 0.0;
        double partial_reduced = 0.0;
        double last_term = 0.0;
        for (int j = 0; j < half_terms; ++j) {
            const double term =
                a[static_cast<std::size_t>(2 * j)] * beta_coefficient;
            partial += term;
            if (j <= reduced_terms) partial_reduced = partial;
            last_term = term;
            beta_coefficient *=
                (static_cast<double>(j) + 0.5) /
                (static_cast<double>(j) + 1.5 + beta);
        }
        if (!std::isfinite(partial) ||
            std::abs(last_term) > 1.0e-9 * (std::abs(partial) + 1.0e-300)) {
            return false;  // series has not converged within the budget
        }

        const double prefactor =
            2.0 * radius * std::pow(h, 2.0 * beta + 1.0);
        *full = prefactor * partial;
        *reduced = prefactor * partial_reduced;
        return std::isfinite(*full) && std::isfinite(*reduced);
    };

    // Validate the closed-form phi against the trusted sampler before trusting
    // the series: any disagreement means the factorisation does not hold here.
    for (const double x_probe : {-0.7, -0.25, 0.25, 0.7}) {
        const double t = m + h * x_probe;
        const double s2 = -(q0 + q1 * t + q2 * t * t);
        const double b2 = b_t[0] + b_t[2] * t * t;
        const double denominator =
            source_radius * source_radius * radius * radius *
            (1.0 + t * t) * b2;
        if (!(denominator > 0.0)) return out;
        const double phi_closed = h * h * (1.0 - x_probe * x_probe) * s2 /
            denominator;
        const double phi_reference = phi_value(
            lens, source, source_radius, radius, 2.0 * std::atan(t),
            diagnostics);
        if (!std::isfinite(phi_closed) ||
            std::abs(phi_closed - phi_reference) >
                1.0e-7 * (1.0 + std::abs(phi_reference))) {
            return out;
        }
    }

    if (need_half && !run_beta(0.5, &out.half_full, &out.half_reduced)) {
        return out;
    }
    if (need_quarter &&
        !run_beta(0.25, &out.quarter_full, &out.quarter_reduced)) {
        return out;
    }
    out.ok = true;
    if (diagnostics != nullptr) ++diagnostics->ld_moment_series_count;
    return out;
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
    const GaussKronrod21Rule& rule,
    bool need_half,
    bool need_quarter,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    NestedMoments result;
    const auto& mapped_rule = sine_mapped_gauss_kronrod21();

    // Proposal 1 fast path: reconstruct the real quartic P_4(t) once so each
    // interval can attempt the closed-form limb-darkening moment series.
    std::array<double, 5> real_quartic {};
    bool have_real_quartic = false;
    if (need_half || need_quarter) {
        const auto complex_quartic = algebraic_boundary_quartic_fixed(
            lens, source, source_radius, radius);
        SelfInversiveTrigQuartic trig;
        if (self_inversive_trig_quartic(complex_quartic, &trig)) {
            real_quartic = {{
                trig.e + 2.0 * trig.c + 2.0 * trig.a,
                -4.0 * trig.d - 8.0 * trig.b,
                2.0 * trig.e - 12.0 * trig.a,
                -4.0 * trig.d + 8.0 * trig.b,
                trig.e - 2.0 * trig.c + 2.0 * trig.a,
            }};
            have_real_quartic = std::isfinite(real_quartic[0]) &&
                std::isfinite(real_quartic[4]) &&
                std::abs(real_quartic[4]) > 0.0;
        }
    }

    for (const auto& interval : intervals) {
        const double midpoint = 0.5 * (interval.first + interval.second);
        const double half_width = 0.5 * (interval.second - interval.first);
        result.gauss.m0 += radius * 2.0 * half_width;
        result.kronrod.m0 += radius * 2.0 * half_width;

        if ((need_half || need_quarter) && have_real_quartic) {
            const auto series = algebraic_moment_series(
                lens, source, source_radius, radius, interval.first,
                interval.second, real_quartic, need_half, need_quarter,
                diagnostics);
            if (series.ok) {
                if (need_half) {
                    result.kronrod.mhalf += series.half_full;
                    result.gauss.mhalf += series.half_reduced;
                }
                if (need_quarter) {
                    result.kronrod.mquarter += series.quarter_full;
                    result.gauss.mquarter += series.quarter_reduced;
                }
                continue;
            }
            if (diagnostics != nullptr) {
                ++diagnostics->ld_moment_series_fallback_count;
            }
        }

        for (std::size_t i = 0; i < rule.nodes.size(); ++i) {
            // Sine-map the Legendre coordinate.  Near x=+-1 the distance to
            // the boundary is quadratic and the Jacobian vanishes linearly,
            // smoothing both Phi^(1/2) and the harder Phi^(1/4) endpoint.
            const double angle = midpoint + half_width * mapped_rule.coordinates[i];
            const double phi = std::max(
                0.0, phi_value(lens, source, source_radius, radius, angle, diagnostics));
            const double common =
                radius * half_width * mapped_rule.jacobians[i];
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

// ---------------------------------------------------------------------------
// Proposal 2: (m, v) root-pair transport across radial quadrature nodes.
//
// The fixed-radius boundary quartic P_4(t) (real, tan(theta/2) form) varies
// smoothly with the image radius R inside a topology band.  Instead of a cold
// bracket-and-Newton root search at every radial node we track each real
// quadratic factor  t^2 - 2 m t + (m^2 - v)  through the sqrt-free system
//
//     E(m, v) = P(m) + (v/2) P''(m) + (v^2/24) P''''(m) = 0
//     O(m, v) = P'(m) + (v/6) P'''(m)                    = 0
//
// which is exact for a quartic (E = O = 0  <=>  m +- sqrt(v) are both roots of
// P).  The Jacobian determinant is  -1/2 (P'')^2  at an ordinary double root,
// so Newton stays well conditioned away from a genuine tangency (which the
// cold path still owns).  We warm-start (m, v) from the nearest cached radius
// and accept only when the recovered quadratic factors multiply back to P_4 to
// tight residual -- a complete, topology-checked factorisation.
// ---------------------------------------------------------------------------

struct QuarticDerivsAtM {
    double p0 = 0.0;  // P(m)
    double p1 = 0.0;  // P'(m)
    double p2 = 0.0;  // P''(m)
    double p3 = 0.0;  // P'''(m)
    double p4 = 0.0;  // P''''(m)  (constant, 24 * leading)
};

inline QuarticDerivsAtM quartic_derivs_at(
    const std::array<double, 5>& p, double m)
{
    QuarticDerivsAtM r;
    r.p0 = (((p[4] * m + p[3]) * m + p[2]) * m + p[1]) * m + p[0];
    r.p1 = ((4.0 * p[4] * m + 3.0 * p[3]) * m + 2.0 * p[2]) * m + p[1];
    r.p2 = (12.0 * p[4] * m + 6.0 * p[3]) * m + 2.0 * p[2];
    r.p3 = 24.0 * p[4] * m + 6.0 * p[3];
    r.p4 = 24.0 * p[4];
    return r;
}

// Newton-refine one (m, v) pair on the E/O system; true on a converged
// solution with v > 0.
inline bool refine_mv_pair(
    const std::array<double, 5>& p, double scale, double* m_io, double* v_io)
{
    double m = *m_io;
    double v = *v_io;
    for (int iteration = 0; iteration < 12; ++iteration) {
        const QuarticDerivsAtM d = quartic_derivs_at(p, m);
        const double e = d.p0 + 0.5 * v * d.p2 + (v * v / 24.0) * d.p4;
        const double o = d.p1 + (v / 6.0) * d.p3;
        const double de_dm = d.p1 + 0.5 * v * d.p3;
        const double de_dv = 0.5 * d.p2 + (v / 12.0) * d.p4;
        const double do_dm = d.p2 + (v / 6.0) * d.p4;
        const double do_dv = d.p3 / 6.0;
        const double det = de_dm * do_dv - de_dv * do_dm;
        if (!std::isfinite(det) || std::abs(det) <= 1.0e-14 * scale * scale) {
            return false;
        }
        const double dm = -(e * do_dv - o * de_dv) / det;
        const double dv = -(de_dm * o - do_dm * e) / det;
        m += dm;
        v += dv;
        if (!std::isfinite(m) || !std::isfinite(v)) return false;
        if (std::abs(dm) + std::abs(dv) <=
                1.0e-13 * (1.0 + std::abs(m) + std::abs(v))) {
            break;
        }
    }
    const QuarticDerivsAtM d = quartic_derivs_at(p, m);
    const double e = d.p0 + 0.5 * v * d.p2 + (v * v / 24.0) * d.p4;
    const double o = d.p1 + (v / 6.0) * d.p3;
    if (!std::isfinite(e) || !std::isfinite(o)) return false;
    if (std::abs(e) + std::abs(o) > 1.0e-9 * scale) return false;
    if (!(v > 0.0)) return false;
    *m_io = m;
    *v_io = v;
    return true;
}

struct MvPairs {
    std::array<std::pair<double, double>, 2> pairs {};  // (m, v)
    int count = 0;
};

// True when the tracked (m, v) pairs reproduce the full real quartic p
// (constant-first) to tight residual: a complete, topology-checked factoring.
inline bool validate_mv_factorization(
    const std::array<double, 5>& p, double scale, const MvPairs& mv)
{
    const double tol = 1.0e-8 * scale;
    if (mv.count == 2) {
        const double a = -2.0 * mv.pairs[0].first;
        const double b = mv.pairs[0].first * mv.pairs[0].first - mv.pairs[0].second;
        const double c = -2.0 * mv.pairs[1].first;
        const double g = mv.pairs[1].first * mv.pairs[1].first - mv.pairs[1].second;
        const double e3 = p[4] * (a + c);
        const double e2 = p[4] * (b + g + a * c);
        const double e1 = p[4] * (a * g + b * c);
        const double e0 = p[4] * (b * g);
        return std::abs(e3 - p[3]) <= tol && std::abs(e2 - p[2]) <= tol &&
               std::abs(e1 - p[1]) <= tol && std::abs(e0 - p[0]) <= tol;
    }
    if (mv.count == 1) {
        const double a = -2.0 * mv.pairs[0].first;
        const double b = mv.pairs[0].first * mv.pairs[0].first - mv.pairs[0].second;
        // Synthetic division of [p4,p3,p2,p1,p0] by monic (t^2 + a t + b).
        const double s2 = p[4];
        const double s1 = p[3] - a * s2;
        const double s0 = p[2] - a * s1 - b * s2;
        const double rem1 = p[1] - a * s0 - b * s1;
        const double rem0 = p[0] - b * s0;
        if (std::abs(rem1) > tol || std::abs(rem0) > tol) return false;
        // The complementary factor must carry no further real roots.
        const double disc = s1 * s1 - 4.0 * s2 * s0;
        return disc < -1.0e-9 * scale * scale;
    }
    return false;
}

inline void append_unit_circle_root(std::vector<Complex>* roots, double t)
{
    double cosine;
    double sine;
    if (std::abs(t) <= 1.0) {
        const double denominator = 1.0 + t * t;
        cosine = (1.0 - t * t) / denominator;
        sine = 2.0 * t / denominator;
    } else {
        const double inverse = 1.0 / t;
        const double denominator = 1.0 + inverse * inverse;
        cosine = (inverse * inverse - 1.0) / denominator;
        sine = 2.0 * inverse / denominator;
    }
    roots->emplace_back(cosine, sine);
}

// Recover adjacently-paired (m, v) factors from unit-circle boundary roots.
inline MvPairs mv_pairs_from_unit_roots(const std::vector<Complex>& roots)
{
    MvPairs out;
    std::array<double, 4> t {};
    int count = 0;
    for (const Complex z : roots) {
        const double denominator = 1.0 + z.real();
        if (!(std::abs(denominator) > 1.0e-9) || count >= 4) return out;
        t[static_cast<std::size_t>(count++)] = z.imag() / denominator;
    }
    std::sort(t.begin(), t.begin() + count);
    const auto make_pair = [](double lo, double hi) {
        const double half = 0.5 * (hi - lo);
        return std::pair<double, double> {0.5 * (lo + hi), half * half};
    };
    if (count == 4) {
        out.count = 2;
        out.pairs[0] = make_pair(t[0], t[1]);
        out.pairs[1] = make_pair(t[2], t[3]);
    } else if (count == 2) {
        out.count = 1;
        out.pairs[0] = make_pair(t[0], t[1]);
    }
    return out;
}

struct MvCacheEntry {
    double radius = 0.0;
    MvPairs pairs;
};

thread_local std::vector<MvCacheEntry>* active_mv_cache = nullptr;

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
        : previous_(active_integrand_cache),
          previous_mv_(active_mv_cache)
    {
        active_integrand_cache = &cache;
        mv_cache_.reserve(16);
        active_mv_cache = &mv_cache_;
    }
    ~IntegrandCacheScope()
    {
        active_integrand_cache = previous_;
        active_mv_cache = previous_mv_;
    }

private:
    std::unordered_map<double, IntegrandEvaluation>* previous_;
    std::vector<MvCacheEntry>* previous_mv_;
    std::vector<MvCacheEntry> mv_cache_;
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
            gauss_kronrod21(), c != 0.0, d != 0.0, diagnostics);
        // The embedded 10/21 pair already evaluates both estimates at the same
        // nodes.  A separate 16-point Legendre pass duplicated every angular
        // map evaluation and provided no nested error information.  Use the
        // Kronrod member as the value and its embedded Gauss difference as the
        // angular error, which is precisely what this rule is designed for.
        result = {true, nested.kronrod,
            std::abs(combined_flux(nested.kronrod - nested.gauss, c, d)), {}};
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

bool solve_radial_tangency(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double bracket_lower,
    double bracket_upper,
    double initial_radius,
    double initial_angle,
    double* endpoint)
{
    double radius = std::clamp(initial_radius, bracket_lower, bracket_upper);
    double angle = initial_angle;
    for (int iteration = 0; iteration < 12; ++iteration) {
        SelfInversiveTrigQuartic q;
        if (!self_inversive_trig_quartic(
                algebraic_boundary_quartic_fixed(
                    lens, source, source_radius, radius), &q)) {
            return false;
        }
        const auto value = evaluate_trig_quartic(q, angle);
        const double step = std::max(
            2.0e-7 * (1.0 + std::abs(radius)),
            1.0e-5 * (bracket_upper - bracket_lower));
        const double radius_minus = std::max(0.0, radius - step);
        const double radius_plus = radius + step;
        SelfInversiveTrigQuartic q_minus;
        SelfInversiveTrigQuartic q_plus;
        if (!self_inversive_trig_quartic(
                algebraic_boundary_quartic_fixed(
                    lens, source, source_radius, radius_minus), &q_minus) ||
            !self_inversive_trig_quartic(
                algebraic_boundary_quartic_fixed(
                    lens, source, source_radius, radius_plus), &q_plus)) {
            return false;
        }
        const auto minus = evaluate_trig_quartic(q_minus, angle);
        const auto plus = evaluate_trig_quartic(q_plus, angle);
        const double inverse_span = 1.0 / (radius_plus - radius_minus);
        const double radial_derivative = (plus[0] - minus[0]) * inverse_span;
        const double mixed_derivative = (plus[1] - minus[1]) * inverse_span;
        const double determinant =
            radial_derivative * value[2] - value[1] * mixed_derivative;
        const double determinant_scale =
            std::abs(radial_derivative * value[2]) +
            std::abs(value[1] * mixed_derivative) + q.scale;
        if (!std::isfinite(determinant) ||
            std::abs(determinant) <= 1.0e-14 * determinant_scale) {
            return false;
        }
        double radius_correction =
            (value[1] * value[1] - value[0] * value[2]) / determinant;
        double angle_correction =
            (mixed_derivative * value[0] - radial_derivative * value[1]) /
            determinant;
        const double maximum_radius_step = 0.45 * (bracket_upper - bracket_lower);
        radius_correction = std::clamp(
            radius_correction, -maximum_radius_step, maximum_radius_step);
        angle_correction = std::clamp(angle_correction, -0.75, 0.75);
        radius += radius_correction;
        angle += angle_correction;
        if (!(radius > bracket_lower && radius < bracket_upper) ||
            !std::isfinite(radius) || !std::isfinite(angle)) {
            return false;
        }
        if (std::abs(radius_correction) <=
                2.0e-12 * (1.0 + std::abs(radius)) &&
            std::abs(angle_correction) <= 2.0e-11) {
            SelfInversiveTrigQuartic final_q;
            if (!self_inversive_trig_quartic(
                    algebraic_boundary_quartic_fixed(
                        lens, source, source_radius, radius), &final_q)) {
                return false;
            }
            const auto final_value = evaluate_trig_quartic(final_q, angle);
            if (std::abs(final_value[0]) <= 2.0e-9 * final_q.scale &&
                std::abs(final_value[1]) <= 2.0e-8 * final_q.scale) {
                *endpoint = radius;
                return true;
            }
        }
    }
    return false;
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
    constexpr int tangency_prefilter_iterations = 3;
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
            int completed_bisections = 0;
            for (; completed_bisections < tangency_prefilter_iterations;
                 ++completed_bisections) {
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

            // At a radial support endpoint two angular roots coalesce, so the
            // endpoint is the simultaneous solution F(R, theta)=F_theta=0.
            // A few boolean halvings give Newton the correct component; the
            // algebraic tangency then replaces the remaining thirteen quartic
            // solves.  Classification on both sides makes this only an
            // acceleration: any ambiguous solve falls back to the original
            // certified bisection path.
            const double active_radius = direction < 0 ? hi : lo;
            const double outside_radius = direction < 0 ? lo : hi;
            const auto active_intervals = algebraic_boundary_intervals_at_radius(
                lens, source, source_radius, active_radius, settings, diagnostics);
            if (!active_intervals.success) {
                reason = active_intervals.unsafe_reason;
                return false;
            }
            bool have_candidate = false;
            double selected_candidate = direction < 0 ? hi : lo;
            for (const auto& interval : active_intervals.inside_intervals) {
                double candidate = 0.0;
                if (!solve_radial_tangency(
                        lens, source, source_radius, lo, hi, active_radius,
                        0.5 * (interval.first + interval.second), &candidate)) {
                    continue;
                }
                if (!have_candidate ||
                    (direction < 0 ? candidate < selected_candidate
                                   : candidate > selected_candidate)) {
                    selected_candidate = candidate;
                    have_candidate = true;
                }
            }
            if (have_candidate) {
                reason.clear();
                const bool active_side = radius_has_image(
                    lens, source, source_radius,
                    0.5 * (selected_candidate + active_radius), settings,
                    diagnostics, &reason);
                if (reason.empty()) {
                    const bool outside_side = radius_has_image(
                        lens, source, source_radius,
                        0.5 * (selected_candidate + outside_radius), settings,
                        diagnostics, &reason);
                    if (reason.empty() && active_side && !outside_side) {
                        if (diagnostics != nullptr) {
                            ++diagnostics->tangency_solve_count;
                        }
                        *endpoint = selected_candidate;
                        return true;
                    }
                }
            }
            if (diagnostics != nullptr) {
                ++diagnostics->tangency_fallback_count;
            }
            reason.clear();
            for (; completed_bisections < bisection_iterations;
                 ++completed_bisections) {
                const double mid = 0.5 * (lo + hi);
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
        const auto& mapped_rule = sine_mapped_gauss_kronrod15();
        Moments gauss;
        Moments kronrod;
        double inner_error = 0.0;
        const double midpoint = 0.5 * (panel_lower + panel_upper);
        const double half_width = 0.5 * (panel_upper - panel_lower);
        for (std::size_t i = 0; i < rule.nodes.size(); ++i) {
            const double coordinate =
                midpoint + half_width * mapped_rule.coordinates[i];
            const double transformed_weight =
                half_width * mapped_rule.jacobians[i];
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
    const auto coefficients = algebraic_boundary_quartic_fixed(
        lens, source, source_radius, image_radius);
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
    const auto coefficients = algebraic_boundary_quartic_fixed(
        lens, source, source_radius, image_radius);
    for (const auto coefficient : coefficients) {
        if (!finite_complex(coefficient)) {
            return {false, false, "non-finite quartic coefficient", {}, {}};
        }
    }
    if (diagnostics != nullptr) {
        ++diagnostics->root_solve_count;
    }
    // Reused scratch: this function runs once per radial quadrature node
    // (~100+ times per magnification) and is not re-entrant, so a thread-local
    // buffer removes one heap allocation per call without changing any value.
    static thread_local std::vector<Complex> physical_roots;
    physical_roots.clear();
    physical_roots.reserve(4);

    // Proposal 2: try to transport the previous radial node's (m, v) root
    // pairs instead of a cold quartic solve.
    bool used_real_quartic = false;
    bool used_mv_transport = false;
    std::array<double, 5> real_quartic {};
    double real_quartic_scale = 1.0;
    bool have_real_quartic = false;
    {
        SelfInversiveTrigQuartic trig;
        if (self_inversive_trig_quartic(coefficients, &trig)) {
            real_quartic = {{
                trig.e + 2.0 * trig.c + 2.0 * trig.a,
                -4.0 * trig.d - 8.0 * trig.b,
                2.0 * trig.e - 12.0 * trig.a,
                -4.0 * trig.d + 8.0 * trig.b,
                trig.e - 2.0 * trig.c + 2.0 * trig.a,
            }};
            for (const double coefficient : real_quartic) {
                real_quartic_scale =
                    std::max(real_quartic_scale, std::abs(coefficient));
            }
            have_real_quartic = std::isfinite(real_quartic[0]) &&
                std::isfinite(real_quartic[4]) &&
                std::abs(real_quartic[4]) > 1.0e-300;
        }
    }
    if (have_real_quartic && active_mv_cache != nullptr &&
        !active_mv_cache->empty()) {
        const MvCacheEntry* nearest = nullptr;
        double best_relative = 0.35;
        for (const auto& entry : *active_mv_cache) {
            const double relative = std::abs(entry.radius - image_radius) /
                std::max(image_radius, 1.0e-12);
            if (relative < best_relative) {
                best_relative = relative;
                nearest = &entry;
            }
        }
        if (nearest != nullptr && nearest->pairs.count > 0) {
            MvPairs candidate = nearest->pairs;
            bool refined = true;
            for (int i = 0; i < candidate.count && refined; ++i) {
                refined = refine_mv_pair(
                    real_quartic, real_quartic_scale,
                    &candidate.pairs[static_cast<std::size_t>(i)].first,
                    &candidate.pairs[static_cast<std::size_t>(i)].second);
            }
            if (refined && candidate.count == 2) {
                const double dm =
                    candidate.pairs[0].first - candidate.pairs[1].first;
                const double dv =
                    candidate.pairs[0].second - candidate.pairs[1].second;
                if (std::abs(dm) + std::abs(dv) <=
                        1.0e-9 * (1.0 + real_quartic_scale)) {
                    refined = false;  // pairs collapsed onto each other
                }
            }
            if (refined && validate_mv_factorization(
                    real_quartic, real_quartic_scale, candidate)) {
                for (int i = 0; i < candidate.count; ++i) {
                    const double m =
                        candidate.pairs[static_cast<std::size_t>(i)].first;
                    const double spread = std::sqrt(
                        candidate.pairs[static_cast<std::size_t>(i)].second);
                    append_unit_circle_root(&physical_roots, m - spread);
                    append_unit_circle_root(&physical_roots, m + spread);
                }
                used_real_quartic = true;
                used_mv_transport = true;
            }
        }
    }

    if (used_mv_transport) {
        if (diagnostics != nullptr) {
            ++diagnostics->real_quartic_solve_count;
            ++diagnostics->mv_transport_count;
        }
    } else if ((used_real_quartic =
                    self_inversive_unit_roots(coefficients, &physical_roots))) {
        if (diagnostics != nullptr) {
            ++diagnostics->real_quartic_solve_count;
            if (active_mv_cache != nullptr) {
                ++diagnostics->mv_transport_fallback_count;
            }
        }
    } else {
        if (diagnostics != nullptr) ++diagnostics->real_quartic_fallback_count;
        const std::vector<Complex> dynamic_coefficients(
            coefficients.begin(), coefficients.end());
        const auto solved = solve_boundary_polynomial(
            dynamic_coefficients, image_radius, diagnostics);
        if (solved.status != math::RootSolverStatus::ok) {
            return {false, false, "quartic root solve failed", {}, {}};
        }
        physical_roots = solved.roots;
    }

    const auto extract_angles = [&](const std::vector<Complex>& roots) {
        static thread_local std::vector<double> angles;
        angles.clear();
        angles.reserve(4);
        for (const auto root : roots) {
            if (!finite_complex(root)) {
                if (diagnostics != nullptr) ++diagnostics->rejected_root_count;
                continue;
            }
            if (!used_real_quartic) {
                const double modulus = std::abs(root);
                if (std::abs(modulus - 1.0) > settings.unit_circle_tolerance) continue;
                if (!(modulus > 0.0)) {
                    if (diagnostics != nullptr) ++diagnostics->rejected_root_count;
                    continue;
                }
            }
            // Projection onto the unit circle cannot change the argument, so
            // the residual gate can take the root's own components directly and
            // skip the atan2 -> sin/cos round trip.  wrap_angle(atan2()) still
            // supplies the angle the interval bookkeeping below needs.
            const double root_modulus =
                std::sqrt(root.real() * root.real() + root.imag() * root.imag());
            const double inverse_modulus =
                root_modulus > 0.0 ? 1.0 / root_modulus : 0.0;
            const double angle = wrap_angle(std::atan2(root.imag(), root.real()));
            const double phi = phi_value_at_point(
                lens, source, source_radius,
                image_radius * root.real() * inverse_modulus,
                image_radius * root.imag() * inverse_modulus, diagnostics);
            const double residual = std::abs(phi) * source_radius;
            if (!std::isfinite(residual) || residual >
                    settings.boundary_residual_tolerance * (1.0 + source_radius)) {
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
            if (!unique.empty() &&
                circular_distance(angle, unique.back()) < duplicate_tolerance) {
                if (diagnostics != nullptr) ++diagnostics->tangent_root_count;
                unique.pop_back();
                continue;
            }
            unique.push_back(angle);
        }
        if (unique.size() > 1 &&
            circular_distance(unique.front(), unique.back()) < duplicate_tolerance) {
            if (diagnostics != nullptr) ++diagnostics->tangent_root_count;
            unique.erase(unique.begin());
            unique.pop_back();
        }
        return unique;
    };
    static thread_local std::vector<Complex> retry_roots;
    const std::vector<Complex>* final_roots = &physical_roots;
    std::vector<double> unique = extract_angles(physical_roots);
    if ((unique.size() % 2) != 0 && used_real_quartic) {
        // Polynomial residual alone can accept a numerically ill-conditioned
        // tan(theta/2) root whose mapped lens residual is not physical.  An
        // odd post-filter count is therefore a certification failure of the
        // fast path, not a failed magnification: retry with Skowron--Gould.
        if (diagnostics != nullptr) {
            --diagnostics->real_quartic_solve_count;
            ++diagnostics->real_quartic_fallback_count;
        }
        const std::vector<Complex> dynamic_coefficients(
            coefficients.begin(), coefficients.end());
        const auto solved = solve_boundary_polynomial(
            dynamic_coefficients, image_radius, diagnostics);
        if (solved.status != math::RootSolverStatus::ok) {
            return {false, false, "quartic root solve failed", {}, {}};
        }
        retry_roots = solved.roots;
        final_roots = &retry_roots;
        unique = extract_angles(solved.roots);
    }
    if ((unique.size() % 2) != 0) {
        return {false, false, "odd physical boundary-root count", {}, {}};
    }

    // Proposal 2: refresh the (m, v) transport cache from this node's
    // validated roots so the next radial node can warm-start from it.
    if (have_real_quartic && active_mv_cache != nullptr) {
        const MvPairs refreshed = mv_pairs_from_unit_roots(*final_roots);
        if (refreshed.count > 0) {
            bool replaced = false;
            for (auto& entry : *active_mv_cache) {
                if (entry.radius == image_radius) {
                    entry.pairs = refreshed;
                    replaced = true;
                    break;
                }
            }
            if (!replaced) {
                if (active_mv_cache->size() >= 64) {
                    active_mv_cache->erase(active_mv_cache->begin());
                }
                active_mv_cache->push_back({image_radius, refreshed});
            }
        }
    }

    AlgebraicBoundaryIntervals result;
    result.success = true;
    result.boundary_angles = std::move(unique);
    const auto& boundary_angles = result.boundary_angles;
    result.inside_intervals.reserve(boundary_angles.size() / 2);
    if (boundary_angles.empty()) {
        const double phi = phi_value(
            lens, source, source_radius, image_radius, 0.371, diagnostics);
        if (phi > 0.0) {
            result.full_circle = true;
            result.inside_intervals.emplace_back(0.0, kTwoPi);
        }
        return result;
    }
    for (std::size_t i = 0; i < boundary_angles.size(); ++i) {
        const double lower = boundary_angles[i];
        const double upper = i + 1 < boundary_angles.size()
            ? boundary_angles[i + 1] : boundary_angles[0] + kTwoPi;
        const double middle = 0.5 * (lower + upper);
        if (phi_value(
                lens, source, source_radius, image_radius, middle, diagnostics) > 0.0) {
            result.inside_intervals.emplace_back(lower, upper);
        }
    }
    return result;
}

AlgebraicBoundaryRadialBands experimental_algebraic_boundary_radial_bands(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    const std::vector<SourcePosition>& certified_image_seeds,
    const AlgebraicBoundarySettings& settings)
{
    AlgebraicBoundaryDiagnostics diagnostics;
    if (certified_image_seeds.empty()) {
        return {false, "no certified image seeds", {}};
    }
    const auto bands = find_radial_bands(
        lens, source, source_radius, certified_image_seeds, settings,
        &diagnostics);
    return {bands.success, bands.unsafe_reason, bands.bands};
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
    const auto coefficients = algebraic_cartesian_boundary_sextic_fixed(
        lens, source, source_radius, image_x);
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
    std::array<Complex, 7> fixed_coefficients;
    std::array<double, 7> real_coefficients {};
    if (!real_cartesian_boundary_coefficients(
            lens, source, source_radius, image_x, &real_coefficients,
            &fixed_coefficients)) {
        return {false, "fixed-x boundary polynomial was not finite and real", {}, {}};
    }
    if (diagnostics != nullptr) ++diagnostics->root_solve_count;
    RealPolynomialRoots real_roots;
    bool used_real_sextic =
        isolate_real_polynomial_roots(real_coefficients, 6, &real_roots);
    std::vector<Complex> physical_roots;
    physical_roots.reserve(6);
    if (used_real_sextic) {
        if (diagnostics != nullptr) ++diagnostics->real_sextic_solve_count;
        for (int i = 0; i < real_roots.count; ++i) {
            physical_roots.emplace_back(
                real_roots.values[static_cast<std::size_t>(i)], 0.0);
        }
    } else {
        if (diagnostics != nullptr) ++diagnostics->real_sextic_fallback_count;
        const std::vector<Complex> coefficients(
            fixed_coefficients.begin(), fixed_coefficients.end());
        const auto solved = solve_boundary_polynomial(
            coefficients, image_x, diagnostics);
        if (solved.status != math::RootSolverStatus::ok) {
            return {false, "sextic root solve failed", {}, {}};
        }
        physical_roots = solved.roots;
    }

    const auto extract_coordinates = [&](const std::vector<Complex>& roots) {
        std::vector<double> coordinates;
        coordinates.reserve(6);
        for (const auto root : roots) {
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
            if (!std::isfinite(residual) || residual >
                    settings.boundary_residual_tolerance * (1.0 + source_radius)) {
                if (diagnostics != nullptr) ++diagnostics->rejected_root_count;
                continue;
            }
            coordinates.push_back(y);
        }
        std::sort(coordinates.begin(), coordinates.end());
        std::vector<double> unique;
        unique.reserve(coordinates.size());
        for (const double coordinate : coordinates) {
            const double duplicate_tolerance =
                1.0e-10 * (1.0 + std::abs(coordinate));
            if (!unique.empty() &&
                coordinate - unique.back() < duplicate_tolerance) {
                if (diagnostics != nullptr) ++diagnostics->tangent_root_count;
                unique.pop_back();
            } else {
                unique.push_back(coordinate);
            }
        }
        return unique;
    };
    std::vector<double> unique = extract_coordinates(physical_roots);
    if ((unique.size() % 2) != 0 && used_real_sextic) {
        // The lens-map residual is the final physical certification.  If it
        // rejects one root from the real isolator, retry the established
        // complex solver instead of returning an incomplete interval set.
        if (diagnostics != nullptr) {
            --diagnostics->real_sextic_solve_count;
            ++diagnostics->real_sextic_fallback_count;
        }
        const std::vector<Complex> coefficients(
            fixed_coefficients.begin(), fixed_coefficients.end());
        const auto solved = solve_boundary_polynomial(
            coefficients, image_x, diagnostics);
        if (solved.status != math::RootSolverStatus::ok) {
            return {false, "sextic root solve failed", {}, {}};
        }
        unique = extract_coordinates(solved.roots);
        used_real_sextic = false;
    }
    if ((unique.size() % 2) != 0) {
        return {false, "odd physical fixed-x boundary-root count", {}, {}};
    }

    AlgebraicCartesianIntervals result;
    result.success = true;
    result.boundary_coordinates = std::move(unique);
    const auto& boundary_coordinates = result.boundary_coordinates;
    result.inside_intervals.reserve(boundary_coordinates.size() / 2);
    for (std::size_t i = 0; i + 1 < boundary_coordinates.size(); ++i) {
        const double lower = boundary_coordinates[i];
        const double upper = boundary_coordinates[i + 1];
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
    const GaussKronrod21Rule& rule,
    bool need_half,
    bool need_quarter,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    NestedMoments result;
    const auto& mapped_rule = sine_mapped_gauss_kronrod21();
    for (const auto& interval : intervals) {
        const double midpoint = 0.5 * (interval.first + interval.second);
        const double half_width = 0.5 * (interval.second - interval.first);
        result.gauss.m0 += 2.0 * half_width;
        result.kronrod.m0 += 2.0 * half_width;
        for (std::size_t i = 0; i < rule.nodes.size(); ++i) {
            const double image_y =
                midpoint + half_width * mapped_rule.coordinates[i];
            if (diagnostics != nullptr) ++diagnostics->lens_evaluation_count;
            const auto mapped = map_lens(lens, image_x, image_y);
            const double dx = mapped.x - source.x;
            const double dy = mapped.y - source.y;
            const double phi = std::max(
                0.0, 1.0 - (dx * dx + dy * dy) /
                    (source_radius * source_radius));
            const double common = half_width * mapped_rule.jacobians[i];
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
    bool need_half,
    bool need_quarter,
    AlgebraicBoundaryDiagnostics* diagnostics)
{
    Moments result;
    const auto& rule = gauss_legendre16();
    const auto& mapped_rule = sine_mapped_gauss_legendre16();
    for (const auto& interval : intervals) {
        const double midpoint = 0.5 * (interval.first + interval.second);
        const double half_width = 0.5 * (interval.second - interval.first);
        result.m0 += 2.0 * half_width;
        for (std::size_t i = 0; i < rule.nodes.size(); ++i) {
            const double image_y =
                midpoint + half_width * mapped_rule.coordinates[i];
            if (diagnostics != nullptr) ++diagnostics->lens_evaluation_count;
            const auto mapped = map_lens(lens, image_x, image_y);
            const double dx = mapped.x - source.x;
            const double dy = mapped.y - source.y;
            const double phi = std::max(
                0.0, 1.0 - (dx * dx + dy * dy) /
                    (source_radius * source_radius));
            const double weighted_area = half_width * mapped_rule.jacobians[i] *
                rule.weights[i];
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
            gauss_kronrod21(), c != 0.0, d != 0.0, diagnostics);
        const auto high = cartesian_inner_moments_high_rule(
            lens, source, source_radius, image_x, intervals.inside_intervals,
            c != 0.0, d != 0.0, diagnostics);
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
