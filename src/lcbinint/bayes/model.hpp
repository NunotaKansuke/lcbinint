#pragma once
#include "prior.hpp"
#include "lcbinint/lcbinint.h"
#include "lcbinint/lc/light_curve.hpp"
#include "lcbinint/obs/event.hpp"
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
// Owns lc::LightCurve (the magnification evaluator) and obs::Event (observational data).
// Samplers and optimizers call only log_prob(theta) / chi2(theta).
class Model {
public:
    Model(std::shared_ptr<lc::LightCurve> lc, std::shared_ptr<obs::Event> event);
    // Convenience: single dataset
    Model(std::shared_ptr<lc::LightCurve> lc, std::shared_ptr<obs::LightCurveData> data);

    void param(std::string name, std::shared_ptr<Prior> prior);
    void flux(std::string mode = "linear_blend");
    void likelihood(std::string mode = "gaussian");

    int                              n_params()   const noexcept;
    const std::vector<ParameterDef>& param_defs() const noexcept { return params_; }
    const lc::LightCurve&            light_curve() const noexcept { return *lc_; }

    double log_prior(      const std::vector<double>& theta) const;
    double log_likelihood( const std::vector<double>& theta) const;
    double log_prob(       const std::vector<double>& theta) const;
    double chi2(           const std::vector<double>& theta) const;

private:
    lcbi_params theta_to_params(const std::vector<double>& theta) const;

    std::shared_ptr<lc::LightCurve>  lc_;
    std::shared_ptr<obs::Event>       event_;
    std::vector<ParameterDef>         params_;
    std::string                       flux_mode_{"linear_blend"};
    std::string                       likelihood_mode_{"gaussian"};
};

} // namespace lcbinint::bayes
