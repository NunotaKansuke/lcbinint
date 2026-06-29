#include "differential_evolution.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace lcbinint::optimize {

DifferentialEvolution::DifferentialEvolution(
    int pop_size, int max_iter, double F, double CR,
    unsigned int seed, std::string strategy)
    : pop_size_(pop_size), max_iter_(max_iter)
    , F_(F), CR_(CR), seed_(seed)
    , strategy_(std::move(strategy))
{
    if (strategy_ != "rand1bin" && strategy_ != "best1bin")
        throw std::invalid_argument("strategy must be 'rand1bin' or 'best1bin'");
}

Result DifferentialEvolution::minimize(bayes::Model& model, std::string target)
{
    if (target != "chi2" && target != "neg_log_prob")
        throw std::invalid_argument("target must be 'chi2' or 'neg_log_prob'");

    const int ndim = model.n_params();
    if (ndim == 0) throw std::invalid_argument("Model has no free parameters");

    const auto bounds = model.optimizer_bounds();
    const int  NP    = pop_size_;

    // --- Objective function: lower is better ---
    const bool use_chi2 = (target == "chi2");
    auto objective = [&](const std::vector<double>& th) -> double {
        return use_chi2 ? model.chi2(th) : -model.log_prob(th);
    };

    // --- RNG ---
    std::mt19937_64 rng(seed_);
    auto rand01  = [&]{ return std::uniform_real_distribution<double>(0.0, 1.0)(rng); };
    auto rand_i  = [&](int n){ return std::uniform_int_distribution<int>(0, n-1)(rng); };

    // --- Initialize population uniformly within bounds ---
    // pop[i][j]: i-th individual, j-th dimension (in transformed space)
    std::vector<std::vector<double>> pop(NP, std::vector<double>(ndim));
    std::vector<double> fitness(NP);

    for (int i = 0; i < NP; ++i) {
        for (int j = 0; j < ndim; ++j)
            pop[i][j] = bounds[j].lo + rand01() * (bounds[j].hi - bounds[j].lo);
        fitness[i] = objective(pop[i]);
    }

    // --- Track global best ---
    int best_idx = static_cast<int>(
        std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
    double best_fitness = fitness[best_idx];

    // Pre-allocate reusable scratch vectors
    std::vector<double> mutant(ndim), trial(ndim);

    int n_eval = NP;
    int gen;
    for (gen = 0; gen < max_iter_; ++gen) {
        for (int i = 0; i < NP; ++i) {

            // --- Select distinct random indices (r1, r2, r3 ≠ i) ---
            int r1, r2, r3;
            do { r1 = rand_i(NP); } while (r1 == i);
            do { r2 = rand_i(NP); } while (r2 == i || r2 == r1);
            do { r3 = rand_i(NP); } while (r3 == i || r3 == r1 || r3 == r2);

            // --- Mutation ---
            if (strategy_ == "rand1bin") {
                for (int j = 0; j < ndim; ++j)
                    mutant[j] = pop[r1][j] + F_ * (pop[r2][j] - pop[r3][j]);
            } else { // best1bin
                for (int j = 0; j < ndim; ++j)
                    mutant[j] = pop[best_idx][j] + F_ * (pop[r1][j] - pop[r2][j]);
            }

            // Clip mutant to bounds
            for (int j = 0; j < ndim; ++j)
                mutant[j] = std::clamp(mutant[j], bounds[j].lo, bounds[j].hi);

            // --- Crossover (binomial) ---
            const int jrand = rand_i(ndim);
            for (int j = 0; j < ndim; ++j)
                trial[j] = (rand01() < CR_ || j == jrand) ? mutant[j] : pop[i][j];

            // --- Selection ---
            const double f_trial = objective(trial);
            ++n_eval;
            if (f_trial <= fitness[i]) {
                pop[i]     = trial;
                fitness[i] = f_trial;
                if (f_trial < best_fitness) {
                    best_fitness = f_trial;
                    best_idx     = i;
                }
            }
        }
    }

    // --- Build result ---
    const std::vector<double>& best_theta = pop[best_idx];
    const auto& param_defs = model.param_defs();

    Result r;
    r.position  = best_theta;
    r.chi2           = model.chi2(best_theta);
    r.log_likelihood = model.log_likelihood(best_theta);
    r.log_prob       = model.log_prob(best_theta);
    r.success        = true;
    r.message        = "completed " + std::to_string(gen) + " generations";
    r.n_eval         = n_eval;
    r.n_iter         = gen;

    // Physical parameter values
    int idx = 0;
    for (const auto& def : param_defs) {
        if (def.fixed) {
            r.parameters[def.name] = def.fixed_value;
        } else {
            const double t = best_theta[idx++];
            r.parameters[def.name] =
                (def.transform == bayes::Transform::log) ? std::exp(t) : t;
        }
    }

    return r;
}

} // namespace lcbinint::optimize
