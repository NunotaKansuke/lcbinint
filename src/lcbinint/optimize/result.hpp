#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace lcbinint::optimize {

struct Result {
    std::vector<double>                     position;     // optimizer-space theta (transformed)
    std::unordered_map<std::string, double> parameters;   // physical parameter values
    double                                  chi2           = 0.0;
    double                                  log_likelihood = 0.0;
    double                                  log_prob       = 0.0;
    bool                                    success        = false;
    std::string                             message;
    int                                     n_eval         = 0;
    int                                     n_iter         = 0;
};

} // namespace lcbinint::optimize
