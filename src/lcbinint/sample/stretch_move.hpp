#pragma once
#include "move.hpp"

namespace lcbinint::sample {

// Affine-invariant stretch move (Goodman & Weare 2010, emcee algorithm).
// Samples z ~ p(z) ∝ 1/sqrt(z) on [1/a, a] via the mapping
//   z = ((a-1)*u + 1)^2 / a,  u ~ U(0,1)
// then proposes Y = X_j + z * (X_k - X_j) with
// log_acceptance_factor = (ndim-1) * log(z).
class StretchMove : public Move {
public:
    explicit StretchMove(double a = 2.0) : a_(a) {}

    std::pair<std::vector<double>, double>
    propose(const std::vector<double>&              walker_k,
            const std::vector<std::vector<double>>& complement,
            std::mt19937_64&                        rng,
            int                                     ndim) override;

    double a() const noexcept { return a_; }

private:
    double a_;
};

} // namespace lcbinint::sample
