#include "differential_evolution.hpp"
#include <stdexcept>

namespace lcbinint::optimize {

DifferentialEvolution::DifferentialEvolution(
    int population_size,
    int max_iter,
    unsigned int seed)
    : population_size_(population_size)
    , max_iter_(max_iter)
    , seed_(seed)
{}

Result DifferentialEvolution::minimize(bayes::Model& model, std::string target)
{
    if (target != "chi2" && target != "neg_log_prob")
        throw std::invalid_argument("target must be 'chi2' or 'neg_log_prob'");
    // TODO: implement differential evolution
    (void)model;
    throw std::runtime_error("DifferentialEvolution::minimize: not yet implemented");
}

} // namespace lcbinint::optimize
