#pragma once
#include <random>
#include <utility>
#include <vector>

namespace lcbinint::sample {

// Base class for ensemble move proposals.
// A Move generates a proposal for walker k given the complementary half-ensemble
// and returns log(acceptance_factor) — the part of log(alpha) that depends only
// on the move geometry (not on the posterior ratio). For stretch move this is
// (ndim-1)*log(z); for moves where the proposal is symmetric it is 0.
class Move {
public:
    virtual ~Move() = default;

    // Returns {proposal_position, log_acceptance_factor}.
    // walker_k: position of the walker being updated (size ndim).
    // complement: the complementary half-ensemble (size half × ndim).
    // rng: shared RNG (not thread-safe — single-threaded inner loop).
    virtual std::pair<std::vector<double>, double>
    propose(const std::vector<double>&              walker_k,
            const std::vector<std::vector<double>>& complement,
            std::mt19937_64&                        rng,
            int                                     ndim) = 0;
};

} // namespace lcbinint::sample
