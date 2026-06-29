#pragma once
#include "prior.hpp"
#include "../lc/evaluator.hpp"
#include "../obs/event.hpp"
#include <memory>
#include <string>
#include <vector>

namespace lcbinint::bayes {

struct ParameterDef {
    std::string            name;
    std::shared_ptr<Prior> prior;
    bool                   fixed       = false;
    double                 fixed_value = 0.0;
};

// Central class for Bayesian inference.
// Owns parameter definitions, priors, flux model, and likelihood configuration.
// Samplers and optimizers call log_prob(theta) / chi2(theta) and know nothing else.
class Model {
public:
    Model(
        std::shared_ptr<lc::IEvaluator> evaluator,
        std::shared_ptr<obs::Event>     event
    );

    // Single-dataset convenience constructor
    Model(
        std::shared_ptr<lc::IEvaluator>      evaluator,
        std::shared_ptr<obs::LightCurveData> data
    );

    void param(std::string name, std::shared_ptr<Prior> prior);
    void flux(std::string mode = "linear_blend");
    void likelihood(std::string mode = "gaussian");

    int                            n_params()   const noexcept;
    const std::vector<ParameterDef>& param_defs() const noexcept { return params_; }

    // Core evaluation — all run in C++
    double log_prior(      const std::vector<double>& theta) const;
    double log_likelihood( const std::vector<double>& theta) const;
    double log_prob(       const std::vector<double>& theta) const;
    double chi2(           const std::vector<double>& theta) const;

private:
    lcbi_params theta_to_params(const std::vector<double>& theta) const;

    std::shared_ptr<lc::IEvaluator> evaluator_;
    std::shared_ptr<obs::Event>     event_;
    std::vector<ParameterDef>       params_;
    std::string                     flux_mode_{"linear_blend"};
    std::string                     likelihood_mode_{"gaussian"};
};

} // namespace lcbinint::bayes
