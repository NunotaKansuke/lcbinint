#pragma once
#include "result.hpp"
#include "../bayes/model.hpp"

namespace lcbinint::optimize {

class DifferentialEvolution {
public:
    explicit DifferentialEvolution(
        int          population_size = 64,
        int          max_iter        = 2000,
        unsigned int seed            = 0
    );

    // target: "chi2" | "neg_log_prob"
    Result minimize(bayes::Model& model, std::string target = "chi2");

private:
    int          population_size_;
    int          max_iter_;
    unsigned int seed_;
};

} // namespace lcbinint::optimize
