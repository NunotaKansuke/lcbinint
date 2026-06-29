#include "prior.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lcbinint::bayes {

// --- Uniform ---

Uniform::Uniform(double lo, double hi)
    : lo_(lo), hi_(hi)
{
    if (hi <= lo)
        throw std::invalid_argument("Uniform: hi must be > lo");
    log_norm_ = -std::log(hi - lo);
}

double Uniform::log_prob(double x) const
{
    if (x < lo_ || x > hi_)
        return -std::numeric_limits<double>::infinity();
    return log_norm_;
}

// --- Normal ---

Normal::Normal(double mu, double sigma)
    : mu_(mu), sigma_(sigma)
{
    if (sigma <= 0.0)
        throw std::invalid_argument("Normal: sigma must be > 0");
}

double Normal::log_prob(double x) const
{
    const double z = (x - mu_) / sigma_;
    return -0.5 * z * z - std::log(sigma_) - 0.5 * std::log(2.0 * M_PI);
}

PriorBounds Normal::bounds() const
{
    return {mu_ - 10.0 * sigma_, mu_ + 10.0 * sigma_};
}

// --- LogUniform ---

LogUniform::LogUniform(double lo, double hi)
    : lo_(lo), hi_(hi)
{
    if (lo <= 0.0)
        throw std::invalid_argument("LogUniform: lo must be > 0");
    if (hi <= lo)
        throw std::invalid_argument("LogUniform: hi must be > lo");
    log_norm_ = -std::log(std::log(hi / lo));
}

double LogUniform::log_prob(double x) const
{
    if (x < lo_ || x > hi_)
        return -std::numeric_limits<double>::infinity();
    return log_norm_ - std::log(x);
}

} // namespace lcbinint::bayes
