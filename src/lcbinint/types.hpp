#pragma once

#include <complex>
#include <limits>

namespace lcbinint {

using Complex = std::complex<double>;

enum class EvaluationStatus {
    ok,
    unsupported,
    unsupported_tolerance,
    numerical_error,
};

constexpr const char* evaluation_status_name(EvaluationStatus status)
{
    switch (status) {
    case EvaluationStatus::ok:
        return "ok";
    case EvaluationStatus::unsupported:
        return "unsupported";
    case EvaluationStatus::unsupported_tolerance:
        return "unsupported_tolerance";
    case EvaluationStatus::numerical_error:
        return "numerical_error";
    }
    return "unknown";
}

struct SourcePosition {
    double x = 0.0;
    double y = 0.0;
};

struct MagnificationResult {
    double magnification = 0.0;
    double point_source_magnification = 0.0;
    double finite_source_magnification = 0.0;
    double finite_source_error_estimate = 0.0;
    SourcePosition source;
    int image_count = 0;
    int finite_source_method = 0;
    int finite_source_refinement_level = 0;
    bool finite_source_converged = true;
    int root_candidate_count = 0;
    int root_duplicate_count = 0;
    int root_polish_failure_count = 0;
    int root_used_warm_start = 0;
    int root_used_cold_retry = 0;
    int root_used_high_precision = 0;
    int root_needs_high_precision = 0;
    double root_max_residual = 0.0;
    double point_source_quadrupole_indicator = 0.0;
    double point_source_cusp_indicator = 0.0;
    double point_source_ghost_indicator = 0.0;
    double point_source_planetary_distance2 = 0.0;
    double point_source_safety_tolerance = 0.0;
    int point_source_ghost_count = 0;
    int point_source_safety_flags = 0;
    double separation = 0.0;
    double mass_ratio = 0.0;
    double caustic_distance = std::numeric_limits<double>::infinity();
    EvaluationStatus status = EvaluationStatus::unsupported;
};

} // namespace lcbinint
