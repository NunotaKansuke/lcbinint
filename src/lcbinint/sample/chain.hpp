#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace lcbinint::sample {

class Chain {
public:
    Chain() = default;

    std::size_t size()       const noexcept { return samples_.size(); }
    int         n_params()   const noexcept { return n_params_; }
    double      acceptance() const noexcept { return acceptance_; }

    const std::vector<std::vector<double>>& samples()  const noexcept { return samples_; }
    const std::vector<double>&              log_prob()  const noexcept { return log_prob_; }

    // sampler-internal
    void push(std::vector<double> theta, double lp);
    void set_n_params(int n)      noexcept { n_params_ = n; }
    void set_acceptance(double f) noexcept { acceptance_ = f; }

private:
    std::vector<std::vector<double>> samples_;
    std::vector<double>              log_prob_;
    int                              n_params_  = 0;
    double                           acceptance_ = 0.0;
};

} // namespace lcbinint::sample
