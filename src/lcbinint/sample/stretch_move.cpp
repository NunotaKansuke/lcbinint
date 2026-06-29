#include "stretch_move.hpp"
#include <cmath>

namespace lcbinint::sample {

std::pair<std::vector<double>, double>
StretchMove::propose(const std::vector<double>&              walker_k,
                     const std::vector<std::vector<double>>& complement,
                     std::mt19937_64&                        rng,
                     int                                     ndim)
{
    const int half = static_cast<int>(complement.size());
    auto rand01 = [&]{ return std::uniform_real_distribution<double>(0.0, 1.0)(rng); };

    // Sample stretch factor z ~ p(z) ∝ 1/sqrt(z) on [1/a, a]
    const double u = rand01();
    const double w = (a_ - 1.0) * u + 1.0;
    const double z = w * w / a_;

    // Pick j uniformly from complement
    const int j = std::uniform_int_distribution<int>(0, half - 1)(rng);
    const auto& walker_j = complement[j];

    // Proposal: Y = X_j + z * (X_k - X_j)
    std::vector<double> prop(ndim);
    for (int d = 0; d < ndim; ++d)
        prop[d] = walker_j[d] + z * (walker_k[d] - walker_j[d]);

    const double log_factor = static_cast<double>(ndim - 1) * std::log(z);
    return {std::move(prop), log_factor};
}

} // namespace lcbinint::sample
