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

// Uniform distribution on [lo, hi].
// No-arg constructor gives a flat (improper) prior: log_prob=0, bounds=±1e15.
class Uniform : public Prior {
public:
    Uniform() : lo_(-1e15), hi_(1e15), log_norm_(0.0) {}
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

} // namespace lcbinint::bayes
