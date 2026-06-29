#pragma once
#include "result.hpp"
#include "../bayes/model.hpp"
#include <string>

namespace lcbinint::optimize {

// Differential Evolution optimizer (DE/rand/1/bin or DE/best/1/bin).
// Operates entirely in C++ with no Python callbacks in the inner loop.
// All parameters are optimized in transformed space (log for LogUniform priors).
class DifferentialEvolution {
public:
    explicit DifferentialEvolution(
        int          population_size = 64,
        int          max_iter        = 2000,
        double       mutation_factor = 0.8,   // F: scale factor for difference vector
        double       crossover_prob  = 0.9,   // CR: crossover probability
        unsigned int seed            = 0,
        std::string  strategy        = "rand1bin"  // "rand1bin" | "best1bin"
    );

    // target: "chi2" (default) | "neg_log_prob"
    Result minimize(bayes::Model& model, std::string target = "chi2");

private:
    int          pop_size_;
    int          max_iter_;
    double       F_;
    double       CR_;
    unsigned int seed_;
    std::string  strategy_;
};

} // namespace lcbinint::optimize
