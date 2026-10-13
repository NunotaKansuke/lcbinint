#include "lcbinint/magnification/adaptive_outer_endpoint_model.hpp"
#include <array>
#include <cmath>
#include <iostream>

int main() {
    using lcbinint::magnification::adaptive_detail::outer_endpoint_cubic_model;
    for (double phase : {0.2, 0.8, 1.2}) {
        for (int sample = 0; sample < 20; ++sample) {
            const double a = 3.0 + 0.1 * sample;
            const double b = 0.02 * std::cos(sample);
            const double c = 0.001;
            std::array<double, 4> original, shifted;
            for (int k = 0; k < 4; ++k) {
                const double t = phase + k;
                const auto width = [&](double u) { return std::sqrt(a*u + b*u*u + c*u*u*u); };
                original[k] = width(t);
                shifted[k] = width(t + 1);
            }
            const auto first = outer_endpoint_cubic_model(original);
            const auto second = outer_endpoint_cubic_model(shifted, 2.5);
            if (!first.resolved || !second.resolved ||
                std::abs(first.eta - phase) > 1e-10 ||
                std::abs(second.eta - 1 - phase) > 1e-10 ||
                std::abs(second.leading_coefficient - std::sqrt(a)) > 1e-10 ||
                std::abs(second.next_coefficient - b/(2*std::sqrt(a))) > 1e-10 ||
                std::abs(second.squared_cubic_coefficient - c) > 1e-10) {
                std::cerr << "cap transport failed: " << phase << ' ' << sample << '\n';
                return 1;
            }
            if (phase > 0.5 && outer_endpoint_cubic_model(shifted).resolved) {
                std::cerr << "cap phase limit was not enforced\n";
                return 1;
            }
        }
    }
    std::cout << "60 analytic cap transports passed\n";
}
