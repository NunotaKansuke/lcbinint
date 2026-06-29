#include "ensemble_sampler.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace lcbinint::sample {

EnsembleSampler::EnsembleSampler(int nwalkers, unsigned int seed, double a)
    : nwalkers_(nwalkers), seed_(seed), a_(a)
{
    if (nwalkers_ < 4 || nwalkers_ % 2 != 0)
        throw std::invalid_argument("nwalkers must be even and >= 4");
}

// ---------------------------------------------------------------------------
// Initialization helpers
// ---------------------------------------------------------------------------

Chain EnsembleSampler::run(bayes::Model& model, int nsteps, int burnin)
{
    const int ndim = model.n_params();
    const auto bounds = model.optimizer_bounds();
    std::mt19937_64 rng(seed_);
    auto rand01 = [&]{ return std::uniform_real_distribution<double>(0.0, 1.0)(rng); };

    std::vector<std::vector<double>> pos(nwalkers_, std::vector<double>(ndim));
    for (int w = 0; w < nwalkers_; ++w)
        for (int j = 0; j < ndim; ++j)
            pos[w][j] = bounds[j].lo + rand01() * (bounds[j].hi - bounds[j].lo);

    return run_impl(model, std::move(pos), nsteps, burnin);
}

Chain EnsembleSampler::run(bayes::Model& model, const optimize::Result& start,
                            int nsteps, int burnin)
{
    const int ndim = model.n_params();
    if (static_cast<int>(start.position.size()) != ndim)
        throw std::invalid_argument("start.position size does not match n_params");

    const auto bounds = model.optimizer_bounds();
    std::mt19937_64 rng(seed_);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // Initialize in a tight gaussian ball around start.position.
    // sigma is 1% of the prior range per dimension, but additionally
    // capped to half the distance to the nearest prior boundary so walkers
    // don't pile up on a wall when the starting point is near a boundary.
    std::vector<std::vector<double>> pos(nwalkers_, std::vector<double>(ndim));
    for (int w = 0; w < nwalkers_; ++w) {
        for (int j = 0; j < ndim; ++j) {
            const double range   = bounds[j].hi - bounds[j].lo;
            const double dist_lo = start.position[j] - bounds[j].lo;
            const double dist_hi = bounds[j].hi - start.position[j];
            const double sigma   = 1e-2 * range;
            const double sigma_j = std::min(sigma, 0.5 * std::min(dist_lo, dist_hi) + 1e-12);
            // re-sample if outside bounds (avoids pile-up at boundaries)
            double v;
            for (int attempt = 0; attempt < 64; ++attempt) {
                v = start.position[j] + gauss(rng) * sigma_j;
                if (v >= bounds[j].lo && v <= bounds[j].hi) break;
            }
            pos[w][j] = std::clamp(v, bounds[j].lo, bounds[j].hi);
        }
    }
    return run_impl(model, std::move(pos), nsteps, burnin);
}

Chain EnsembleSampler::run(bayes::Model& model,
                            const std::vector<std::vector<double>>& start_pos,
                            int nsteps, int burnin)
{
    if (static_cast<int>(start_pos.size()) != nwalkers_)
        throw std::invalid_argument("start_pos must have nwalkers rows");
    return run_impl(model, start_pos, nsteps, burnin);
}

// ---------------------------------------------------------------------------
// Core: affine-invariant stretch-move ensemble sampler (emcee algorithm)
// ---------------------------------------------------------------------------

Chain EnsembleSampler::run_impl(bayes::Model& model,
                                 std::vector<std::vector<double>> pos,
                                 int nsteps, int burnin)
{
    const int ndim = model.n_params();
    const int NW   = nwalkers_;
    const int half = NW / 2;

    // Evaluate log_prob for all initial positions
    std::vector<double> lp(NW);
    for (int w = 0; w < NW; ++w)
        lp[w] = model.log_prob(pos[w]);

    // Pre-allocate reusable buffers
    std::vector<double> prop(ndim);
    std::vector<double> step_pos(NW * ndim);
    std::vector<double> step_lp(NW);

    Chain chain;
    chain.init(nsteps, NW, ndim);
    {
        std::vector<std::string> names;
        for (const auto& def : model.param_defs())
            if (!def.fixed) names.push_back(def.name);
        chain.set_param_names(std::move(names));
    }

    std::mt19937_64 rng(seed_);
    auto rand01 = [&]{ return std::uniform_real_distribution<double>(0.0, 1.0)(rng); };

    // Stretch factor: z = ((a-1)*u + 1)^2 / a  maps u~U(0,1) → z ∈ [1/a, a]
    // giving p(z) ∝ 1/sqrt(z)  (Goodman & Weare 2010)
    const double inv_a = 1.0 / a_;
    auto sample_z = [&]() -> double {
        const double u = rand01();
        const double w = (a_ - 1.0) * u + 1.0;
        return w * w * inv_a;
    };

    long long accepted = 0;

    for (int step = 0; step < nsteps + burnin; ++step) {
        // Complement-based update: process each half using the other as complement
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            const int begin_this = half_idx == 0 ? 0    : half;
            const int begin_comp = half_idx == 0 ? half : 0;

            for (int k = begin_this; k < begin_this + half; ++k) {
                // Pick j uniformly from complementary half
                const int j = begin_comp +
                    std::uniform_int_distribution<int>(0, half - 1)(rng);

                const double z = sample_z();

                // Proposal: Y = X_j + z * (X_k - X_j)
                for (int d = 0; d < ndim; ++d)
                    prop[d] = pos[j][d] + z * (pos[k][d] - pos[j][d]);

                const double lp_prop = model.log_prob(prop);
                const double log_acc = (ndim - 1) * std::log(z) + lp_prop - lp[k];

                if (std::log(rand01()) <= log_acc) {
                    pos[k] = prop;
                    lp[k]  = lp_prop;
                    ++accepted;
                }
            }
        }

        if (step >= burnin) {
            for (int w = 0; w < NW; ++w) {
                const std::size_t base = static_cast<std::size_t>(w * ndim);
                std::copy(pos[w].begin(), pos[w].end(), step_pos.begin() + base);
                step_lp[w] = lp[w];
            }
            chain.push_step(step_pos, step_lp);
        }
    }

    const long long total = static_cast<long long>(nsteps + burnin) * NW;
    chain.set_acceptance(static_cast<double>(accepted) / static_cast<double>(total));
    return chain;
}

} // namespace lcbinint::sample
