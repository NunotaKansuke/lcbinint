#include "model.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace lcbinint::bayes {

// Single-dataset convenience: wrap in an Event
Model::Model(
    std::shared_ptr<lc::IEvaluator>      evaluator,
    std::shared_ptr<obs::LightCurveData> data)
    : evaluator_(std::move(evaluator))
    , event_(std::make_shared<obs::Event>())
{
    if (!evaluator_)
        throw std::invalid_argument("evaluator must not be null");
    event_->add(std::move(data));
}

Model::Model(
    std::shared_ptr<lc::IEvaluator> evaluator,
    std::shared_ptr<obs::Event>     event)
    : evaluator_(std::move(evaluator))
    , event_(std::move(event))
{
    if (!evaluator_)
        throw std::invalid_argument("evaluator must not be null");
    if (!event_)
        throw std::invalid_argument("event must not be null");
}

void Model::param(std::string name, std::shared_ptr<Prior> prior)
{
    if (!prior)
        throw std::invalid_argument("prior must not be null");
    params_.push_back({std::move(name), std::move(prior)});
}

void Model::flux(std::string mode)
{
    if (mode != "linear_blend")
        throw std::invalid_argument("unsupported flux mode: " + mode);
    flux_mode_ = std::move(mode);
}

void Model::likelihood(std::string mode)
{
    if (mode != "gaussian")
        throw std::invalid_argument("unsupported likelihood mode: " + mode);
    likelihood_mode_ = std::move(mode);
}

int Model::n_params() const noexcept
{
    int n = 0;
    for (const auto& p : params_)
        if (!p.fixed) ++n;
    return n;
}

double Model::log_prior(const std::vector<double>& theta) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");

    double lp = 0.0;
    int idx = 0;
    for (const auto& p : params_) {
        if (p.fixed) continue;
        lp += p.prior->log_prob(theta[idx++]);
    }
    return lp;
}

lcbi_params Model::theta_to_params(const std::vector<double>& theta) const
{
    // TODO: map named parameters to lcbi_params fields
    // For now return default params — full mapping implemented when param name registry is added
    (void)theta;
    return lcbi_default_params();
}

double Model::log_likelihood(const std::vector<double>& theta) const
{
    // TODO: evaluate magnification, solve Fs/Fb, compute Gaussian chi2
    (void)theta;
    throw std::runtime_error("Model::log_likelihood: not yet implemented");
}

double Model::log_prob(const std::vector<double>& theta) const
{
    const double lp = log_prior(theta);
    if (!std::isfinite(lp)) return lp;
    return lp + log_likelihood(theta);
}

double Model::chi2(const std::vector<double>& theta) const
{
    // TODO: evaluate magnification, solve linear flux, return chi2
    (void)theta;
    throw std::runtime_error("Model::chi2: not yet implemented");
}

} // namespace lcbinint::bayes
