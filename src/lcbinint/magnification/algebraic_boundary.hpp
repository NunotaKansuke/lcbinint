#pragma once
#include <string>
namespace lcbinint::magnification {
struct AlgebraicBoundarySettings {
    double absolute_tolerance = 1.0e-8;
    double relative_tolerance = 1.0e-6;
};
struct AlgebraicBoundaryDiagnostics {
    int root_solve_count = 0;
    int lens_evaluation_count = 0;
};
struct AlgebraicBoundaryResult {
    double magnification = 0.0;
    double error_estimate = 0.0;
    bool success = false;
    std::string unsafe_reason;
    AlgebraicBoundaryDiagnostics diagnostics;
    bool support_certified = false;
};
} // namespace lcbinint::magnification
