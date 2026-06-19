#pragma once

#include "lcbinint/types.hpp"

#include <vector>

namespace lcbinint::magnification {

struct PointSourceResult {
    double magnification = 0.0;
    int image_count = 0;
};

struct BinaryImage {
    SourcePosition position;
    double jacobian_determinant = 0.0;
};

struct BinaryImageCandidate {
    SourcePosition position;
    double jacobian_determinant = 0.0;
    double residual = 0.0;
    bool physical = false;
};

class PointSourceMagnifier {
public:
    PointSourceResult binary_mag0(double separation, double mass_ratio, SourcePosition source) const;
    std::vector<BinaryImage> binary_images(double separation, double mass_ratio, SourcePosition source) const;
    std::vector<BinaryImageCandidate> binary_image_candidates(
        double separation,
        double mass_ratio,
        SourcePosition source) const;
    SourcePosition binary_lens_equation(double separation, double mass_ratio, SourcePosition image) const;
};

} // namespace lcbinint::magnification
