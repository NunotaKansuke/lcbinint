#pragma once

#include "lcbinint/types.hpp"

#include <array>
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
    PointSourceResult binary_mag0_cached(double separation, double mass_ratio, SourcePosition source) const;
    std::vector<BinaryImage> binary_images(double separation, double mass_ratio, SourcePosition source) const;
    std::vector<BinaryImageCandidate> binary_image_candidates(
        double separation,
        double mass_ratio,
        SourcePosition source) const;
    SourcePosition binary_lens_equation(double separation, double mass_ratio, SourcePosition image) const;

private:
    PointSourceResult binary_mag0_impl(
        double separation,
        double mass_ratio,
        SourcePosition source,
        bool use_root_cache) const;

    mutable bool root_cache_valid_ = false;
    mutable double root_cache_separation_ = 0.0;
    mutable double root_cache_mass_ratio_ = 0.0;
    mutable SourcePosition root_cache_source_ {};
    mutable std::array<SourcePosition, 5> root_cache_roots_ {};
};

} // namespace lcbinint::magnification
