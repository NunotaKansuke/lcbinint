#pragma once

#include "lcbinint/types.hpp"

namespace lcbinint::magnification {

struct PointSourceResult {
    double magnification = 0.0;
    int image_count = 0;
};

class PointSourceMagnifier {
public:
    PointSourceResult binary_mag0(double separation, double mass_ratio, SourcePosition source) const;
};

} // namespace lcbinint::magnification
