#pragma once

#include <complex>

namespace lcbinint {

using Complex = std::complex<double>;

enum class EvaluationStatus {
    ok,
    unsupported,
    numerical_error,
};

struct SourcePosition {
    double x = 0.0;
    double y = 0.0;
};

struct MagnificationResult {
    double magnification = 0.0;
    double point_source_magnification = 0.0;
    double finite_source_magnification = 0.0;
    SourcePosition source;
    int image_count = 0;
    EvaluationStatus status = EvaluationStatus::unsupported;
};

} // namespace lcbinint
