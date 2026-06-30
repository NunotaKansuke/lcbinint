#include "ensemble_sampler.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace lcbinint::sample {

EnsembleSampler::EnsembleSampler(int nwalkers, unsigned int seed,
                                   std::shared_ptr<Move> move)
    : nwalkers_(nwalkers), seed_(seed), move_(std::move(move))
{
    if (nwalkers_ < 4 || nwalkers_ % 2 != 0)
        throw std::invalid_argument("nwalkers must be even and >= 4");
    if (!move_)
        throw std::invalid_argument("move must not be null");
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
    const int n_ds = static_cast<int>(model.event().size());
    const int n_fl = n_ds * 2;

    // Evaluate log_prob and fluxes for all initial positions in a single pass.
    std::vector<double> lp(NW);
    std::vector<std::vector<double>> walker_fluxes(NW, std::vector<double>(n_fl, 0.0));
    {
        std::vector<bayes::Model::FluxSolution> fl_buf;
        for (int w = 0; w < NW; ++w) {
            lp[w] = model.log_prob_and_fluxes(pos[w], fl_buf);
            for (int k = 0; k < n_ds && k < static_cast<int>(fl_buf.size()); ++k) {
                walker_fluxes[w][2*k]   = fl_buf[k].Fs;
                walker_fluxes[w][2*k+1] = fl_buf[k].Fb;
            }
        }
    }

    // Pre-allocate step buffers
    std::vector<double> step_pos(NW * ndim);
    std::vector<double> step_lp(NW);
    std::vector<double> step_fl(NW * n_fl);

    Chain chain;
    chain.init(nsteps, NW, ndim);
    {
        std::vector<std::string> names;
        std::vector<std::string> transforms;
        for (const auto& def : model.param_defs()) {
            if (def.fixed) continue;
            names.push_back(def.name);
            transforms.push_back(
                def.transform == bayes::Transform::log ? "log" : "identity");
        }
        chain.set_param_names(std::move(names));
        chain.set_transforms(std::move(transforms));
    }
    {
        std::vector<std::string> ds_names;
        ds_names.reserve(n_ds);
        for (int k = 0; k < n_ds; ++k)
            ds_names.push_back(model.event().at(k).name());
        chain.init_fluxes(n_ds, std::move(ds_names));
    }

    std::mt19937_64 rng(seed_);
    auto rand01 = [&]{ return std::uniform_real_distribution<double>(0.0, 1.0)(rng); };

    long long accepted = 0;

    for (int step = 0; step < nsteps + burnin; ++step) {
        // Complement-based update: process each half using the other as complement
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            const int begin_this = half_idx == 0 ? 0    : half;
            const int begin_comp = half_idx == 0 ? half : 0;

            // Build complementary sub-ensemble view (stable during this half update)
            std::vector<std::vector<double>> complement(pos.begin() + begin_comp,
                                                         pos.begin() + begin_comp + half);

            std::vector<bayes::Model::FluxSolution> prop_fl;
            for (int k = begin_this; k < begin_this + half; ++k) {
                auto [prop_vec, log_factor] =
                    move_->propose(pos[k], complement, rng, ndim);

                // Single magnification pass: log_prob + fluxes together.
                const double lp_prop = model.log_prob_and_fluxes(prop_vec, prop_fl);
                const double log_acc = log_factor + lp_prop - lp[k];

                if (std::log(rand01()) <= log_acc) {
                    for (int j = 0; j < n_ds && j < static_cast<int>(prop_fl.size()); ++j) {
                        walker_fluxes[k][2*j]   = prop_fl[j].Fs;
                        walker_fluxes[k][2*j+1] = prop_fl[j].Fb;
                    }
                    pos[k] = std::move(prop_vec);
                    lp[k]  = lp_prop;
                    ++accepted;
                }
            }
        }

        if (step >= burnin) {
            for (int w = 0; w < NW; ++w) {
                const std::size_t base_p = static_cast<std::size_t>(w * ndim);
                std::copy(pos[w].begin(), pos[w].end(), step_pos.begin() + base_p);
                step_lp[w] = lp[w];
                const std::size_t base_f = static_cast<std::size_t>(w * n_fl);
                std::copy(walker_fluxes[w].begin(), walker_fluxes[w].end(),
                          step_fl.begin() + base_f);
            }
            chain.push_step(step_pos, step_lp, step_fl);
        }
    }

    const long long total = static_cast<long long>(nsteps + burnin) * NW;
    chain.set_acceptance(static_cast<double>(accepted) / static_cast<double>(total));
    return chain;
}

} // namespace lcbinint::sample
