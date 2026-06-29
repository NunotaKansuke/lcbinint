#pragma once
#include "prior.hpp"
#include "lcbinint/lcbinint.h"
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
// Holds lcbi_options and calls lcbi_magnification_array directly in the hot path.
// Samplers and optimizers only call log_prob(theta) / chi2(theta).
class Model {
public:
    Model(lcbi_options options, std::shared_ptr<obs::Event> event);
    Model(lcbi_options options, std::shared_ptr<obs::LightCurveData> data);

    void param(std::string name, std::shared_ptr<Prior> prior);
    void flux(std::string mode = "linear_blend");
    void likelihood(std::string mode = "gaussian");

    int                              n_params()    const noexcept;
    const std::vector<ParameterDef>& param_defs()  const noexcept { return params_; }
    const lcbi_options&              options()      const noexcept { return options_; }

    double log_prior(      const std::vector<double>& theta) const;
    double log_likelihood( const std::vector<double>& theta) const;
    double log_prob(       const std::vector<double>& theta) const;
    double chi2(           const std::vector<double>& theta) const;

private:
    lcbi_params theta_to_params(const std::vector<double>& theta) const;

    lcbi_options              options_;
    std::shared_ptr<obs::Event> event_;
    std::vector<ParameterDef> params_;
    std::string               flux_mode_{"linear_blend"};
    std::string               likelihood_mode_{"gaussian"};
};

} // namespace lcbinint::bayes
