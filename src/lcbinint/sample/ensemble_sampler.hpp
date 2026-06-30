#pragma once
#include "chain.hpp"
#include "move.hpp"
#include "sampler_state.hpp"
#include "stretch_move.hpp"
#include "../bayes/model.hpp"
#include "../optimize/result.hpp"
#include <memory>
#include <vector>

namespace lcbinint::sample {

// Affine-invariant ensemble sampler (Goodman & Weare 2010).
// Two usage modes:
//
//   Batch mode (convenience):
//     chain = sampler.run(model, start=result, nsteps=1000, burnin=200)
//
//   Step-by-step mode (full control: logging, h5py, progress bars, etc.):
//     state = sampler.init_state(model, start=result, hessian_init=True)
//     for i in range(nsteps):
//         sampler.step(model, state)
//         if i % 100 == 0: print(state.pos)
//     chain = sampler.collect(state, discard=burnin)
//
// All walker positions are in transformed space (log for LogUniform priors).
class EnsembleSampler {
public:
    explicit EnsembleSampler(
        int                    nwalkers = 64,
        unsigned int           seed     = 0,
        std::shared_ptr<Move>  move     = std::make_shared<StretchMove>(2.0)
    );

    // ----- Step-by-step API -----

    // Initialise from random positions within prior bounds.
    SamplerState init_state(bayes::Model& model);

    // Initialise around a best-fit result (gaussian ball or Laplace approx).
    SamplerState init_state(bayes::Model& model, const optimize::Result& start,
                            bool hessian_init = false);

    // Initialise from explicit walker positions (nwalkers × ndim).
    SamplerState init_state(bayes::Model& model,
                            const std::vector<std::vector<double>>& pos);

    // Advance by one full ensemble step (updates state in-place).
    void step(bayes::Model& model, SamplerState& state);

    // Build a Chain from a SamplerState (copies all accumulated steps).
    // Typically called after the step loop; discard removes leading burnin rows.
    Chain collect(bayes::Model& model, const SamplerState& state, int discard = 0) const;

    // ----- Batch mode (convenience wrappers around init_state + step loop) -----

    Chain run(bayes::Model& model, int nsteps = 1000, int burnin = 0);

    Chain run(bayes::Model& model, const optimize::Result& start,
              int nsteps = 1000, int burnin = 0, bool hessian_init = false);

    Chain run(bayes::Model& model,
              const std::vector<std::vector<double>>& start_pos,
              int nsteps = 1000, int burnin = 0);

private:
    // Build initial positions and evaluate log_prob for all walkers.
    SamplerState make_state(bayes::Model& model,
                            std::vector<std::vector<double>> init_pos);

    // Initialise walker positions via Hessian-based Laplace approximation,
    // falling back to gaussian ball if Hessian is not negative definite.
    std::vector<std::vector<double>> init_pos_hessian(
        bayes::Model& model, const optimize::Result& start,
        const std::vector<bayes::OptimizerBounds>& bounds, std::mt19937_64& rng);

    std::vector<std::vector<double>> init_pos_ball(
        const optimize::Result& start,
        const std::vector<bayes::OptimizerBounds>& bounds, std::mt19937_64& rng);

    Chain run_from_state(bayes::Model& model, SamplerState state,
                         int nsteps, int burnin);

    int                   nwalkers_;
    unsigned int          seed_;
    std::shared_ptr<Move> move_;
};

} // namespace lcbinint::sample
