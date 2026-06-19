#pragma once

#include "lcbinint/types.hpp"

#include <vector>

namespace lcbinint::math {

enum class RootSolverStatus {
    ok,
    invalid_polynomial,
    unsupported_degree,
};

struct PolynomialRootOptions {
    bool polish_roots = true;
    bool use_roots_as_starting_points = false;
};

struct PolynomialRootResult {
    RootSolverStatus status = RootSolverStatus::invalid_polynomial;
    std::vector<Complex> roots;
};

class PolynomialRootSolver {
public:
    // Coefficients are constant-first:
    // c[0] + c[1] z + ... + c[n] z^n.
    PolynomialRootResult solve(
        const std::vector<Complex>& coefficients,
        const PolynomialRootOptions& options = {}) const;

    static Complex evaluate(const std::vector<Complex>& coefficients, Complex z);

private:
    PolynomialRootResult solve_linear(const std::vector<Complex>& coefficients) const;
    PolynomialRootResult solve_quadratic(const std::vector<Complex>& coefficients) const;
};

} // namespace lcbinint::math
