#pragma once
#include <limits>
#include <stdexcept>
#include <string>

namespace lcbinint::bayes {

struct PriorBounds {
    double lo;
    double hi;
};

class Prior {
public:
    virtual ~Prior() = default;
    virtual double     log_prob(double x) const = 0;
    virtual PriorBounds bounds()          const = 0;
    virtual std::string name()            const = 0;
};

// Uniform distribution on [lo, hi]
class Uniform : public Prior {
public:
    Uniform(double lo, double hi);
    double      log_prob(double x) const override;
    PriorBounds bounds()           const override { return {lo_, hi_}; }
    std::string name()             const override { return "Uniform"; }
private:
    double lo_, hi_, log_norm_;
};

// Gaussian (normal) distribution
class Normal : public Prior {
public:
    Normal(double mu, double sigma);
    double      log_prob(double x) const override;
    PriorBounds bounds()           const override;
    std::string name()             const override { return "Normal"; }
private:
    double mu_, sigma_;
};

// Log-uniform (Jeffreys) distribution on [lo, hi]; sampling is done in log space
class LogUniform : public Prior {
public:
    LogUniform(double lo, double hi);
    double      log_prob(double x) const override;
    PriorBounds bounds()           const override { return {lo_, hi_}; }
    std::string name()             const override { return "LogUniform"; }
private:
    double lo_, hi_, log_norm_;
};

// Flat (improper) prior: log_prob = 0 everywhere, no hard bounds.
// Used when model.param(name) is called without an explicit prior.
class Flat : public Prior {
public:
    double      log_prob(double)   const override { return 0.0; }
    PriorBounds bounds()           const override { return {-1e15, 1e15}; }
    std::string name()             const override { return "Flat"; }
};

} // namespace lcbinint::bayes
