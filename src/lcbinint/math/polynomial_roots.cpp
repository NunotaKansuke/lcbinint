#include "lcbinint/math/polynomial_roots.hpp"

#include "SkowronGould.h"

#include <array>
#include <cmath>

namespace lcbinint::math {
namespace {

constexpr double kZeroTolerance = 0.0;
constexpr int kSkowronGouldMaxDegree = MAXM - 1;

int effective_degree(const std::vector<Complex>& coefficients)
{
    for (int degree = static_cast<int>(coefficients.size()) - 1; degree >= 0; --degree) {
        if (std::abs(coefficients[static_cast<std::size_t>(degree)]) > kZeroTolerance) {
            return degree;
        }
    }
    return -1;
}

complex to_sg_complex(Complex value)
{
    return {value.real(), value.imag()};
}

Complex from_sg_complex(complex value)
{
    return {value.re, value.im};
}

} // namespace

PolynomialRootResult PolynomialRootSolver::solve(
    const std::vector<Complex>& coefficients,
    const PolynomialRootOptions& options) const
{
    const int degree = effective_degree(coefficients);
    if (degree < 1) {
        return {RootSolverStatus::invalid_polynomial, {}};
    }
    if (degree == 1) {
        return solve_linear(coefficients);
    }
    if (degree == 2) {
        return solve_quadratic(coefficients);
    }
    if (degree > kSkowronGouldMaxDegree) {
        return {RootSolverStatus::unsupported_degree, {}};
    }

    // Degrees are bounded by MAXM; fixed-size buffers keep the hot root-solve
    // paths free of heap allocations.
    std::array<complex, MAXM + 1> sg_coefficients;
    std::array<complex, MAXM> sg_roots;
    for (int i = 0; i <= degree; ++i) {
        sg_coefficients[static_cast<std::size_t>(i)] =
            to_sg_complex(coefficients[static_cast<std::size_t>(i)]);
    }

    cmplx_roots_gen(
        sg_roots.data(),
        sg_coefficients.data(),
        degree,
        options.polish_roots,
        options.use_roots_as_starting_points);

    std::vector<Complex> roots;
    roots.reserve(static_cast<std::size_t>(degree));
    for (int i = 0; i < degree; ++i) {
        roots.push_back(from_sg_complex(sg_roots[static_cast<std::size_t>(i)]));
    }

    return {RootSolverStatus::ok, roots};
}

Complex PolynomialRootSolver::evaluate(const std::vector<Complex>& coefficients, Complex z)
{
    Complex value = 0.0;
    for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
        value = value * z + *it;
    }
    return value;
}

PolynomialRootResult PolynomialRootSolver::solve_linear(
    const std::vector<Complex>& coefficients) const
{
    return {RootSolverStatus::ok, {-coefficients[0] / coefficients[1]}};
}

PolynomialRootResult PolynomialRootSolver::solve_quadratic(
    const std::vector<Complex>& coefficients) const
{
    const Complex c = coefficients[0];
    const Complex b = coefficients[1];
    const Complex a = coefficients[2];
    const Complex delta = std::sqrt(b * b - 4.0 * a * c);

    Complex q;
    if (std::real(std::conj(b) * delta) >= 0.0) {
        q = -0.5 * (b + delta);
    } else {
        q = -0.5 * (b - delta);
    }

    if (std::abs(q) == 0.0) {
        return {RootSolverStatus::ok, {(-b + delta) / (2.0 * a), (-b - delta) / (2.0 * a)}};
    }

    return {RootSolverStatus::ok, {q / a, c / q}};
}

} // namespace lcbinint::math
