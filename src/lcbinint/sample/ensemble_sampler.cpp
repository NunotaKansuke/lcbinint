#include "ensemble_sampler.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace lcbinint::sample {

// ---------------------------------------------------------------------------
// Finite-difference Hessian helpers (same as before)
// ---------------------------------------------------------------------------

namespace {

std::vector<double> finite_diff_hessian(
    bayes::Model&                              model,
    const std::vector<double>&                 theta,
    double                                     lp0,
    const std::vector<bayes::OptimizerBounds>& bounds)
{
    const int n = static_cast<int>(theta.size());
    const double eps_frac = 1e-3;
    std::vector<double> eps(n);
    for (int j = 0; j < n; ++j)
        eps[j] = eps_frac * (bounds[j].hi - bounds[j].lo);

    std::vector<double> H(n * n, 0.0);

    for (int j = 0; j < n; ++j) {
        auto tp = theta; tp[j] += eps[j];
        auto tm = theta; tm[j] -= eps[j];
        const double fp = model.log_prob(tp), fm = model.log_prob(tm);
        if (!std::isfinite(fp) || !std::isfinite(fm)) return {};
        H[j*n+j] = (fp + fm - 2.0*lp0) / (eps[j]*eps[j]);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            auto tpp = theta; tpp[i] += eps[i]; tpp[j] += eps[j];
            auto tpm = theta; tpm[i] += eps[i]; tpm[j] -= eps[j];
            auto tmp = theta; tmp[i] -= eps[i]; tmp[j] += eps[j];
            auto tmm = theta; tmm[i] -= eps[i]; tmm[j] -= eps[j];
            const double fpp = model.log_prob(tpp), fpm = model.log_prob(tpm);
            const double fmp = model.log_prob(tmp), fmm = model.log_prob(tmm);
            if (!std::isfinite(fpp) || !std::isfinite(fpm) ||
                !std::isfinite(fmp) || !std::isfinite(fmm)) return {};
            const double H_ij = (fpp - fpm - fmp + fmm) / (4.0 * eps[i] * eps[j]);
            H[i*n+j] = H_ij;
            H[j*n+i] = H_ij;
        }
    }
    return H;
}

std::vector<double> cholesky(const std::vector<double>& A, int n)
{
    std::vector<double> L(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i*n+j];
            for (int k = 0; k < j; ++k)
                s -= L[i*n+k] * L[j*n+k];
            if (i == j) {
                if (s <= 0.0) return {};
                L[i*n+j] = std::sqrt(s);
            } else {
                L[i*n+j] = s / L[j*n+j];
            }
        }
    }
    return L;
}

void solve_Lt(const std::vector<double>& L, int n,
              const std::vector<double>& z, std::vector<double>& w)
{
    w.resize(n);
    for (int i = n-1; i >= 0; --i) {
        double s = z[i];
        for (int k = i+1; k < n; ++k)
            s -= L[k*n+i] * w[k];
        w[i] = s / L[i*n+i];
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

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
// Position initialisation helpers
// ---------------------------------------------------------------------------

std::vector<std::vector<double>> EnsembleSampler::init_pos_hessian(
    bayes::Model& model, const optimize::Result& start,
    const std::vector<bayes::OptimizerBounds>& bounds, std::mt19937_64& rng)
{
    const int ndim = static_cast<int>(start.position.size());
    const double lp0 = model.log_prob(start.position);
    if (std::isfinite(lp0)) {
        const auto H = finite_diff_hessian(model, start.position, lp0, bounds);
        if (!H.empty()) {
            std::vector<double> neg_H(ndim * ndim);
            for (int i = 0; i < ndim * ndim; ++i) neg_H[i] = -H[i];
            const auto L = cholesky(neg_H, ndim);
            if (!L.empty()) {
                std::normal_distribution<double> gauss(0.0, 1.0);
                std::vector<std::vector<double>> pos(nwalkers_,
                                                      std::vector<double>(ndim));
                std::vector<double> z(ndim), w(ndim);
                for (int wk = 0; wk < nwalkers_; ++wk) {
                    for (int j = 0; j < ndim; ++j) z[j] = gauss(rng);
                    solve_Lt(L, ndim, z, w);
                    for (int j = 0; j < ndim; ++j)
                        pos[wk][j] = std::clamp(start.position[j] + w[j],
                                                  bounds[j].lo, bounds[j].hi);
                }
                return pos;
            }
        }
    }
    // Fallback to ball
    return init_pos_ball(start, bounds, rng);
}

std::vector<std::vector<double>> EnsembleSampler::init_pos_ball(
    const optimize::Result& start,
    const std::vector<bayes::OptimizerBounds>& bounds, std::mt19937_64& rng)
{
    const int ndim = static_cast<int>(start.position.size());
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<std::vector<double>> pos(nwalkers_, std::vector<double>(ndim));
    for (int w = 0; w < nwalkers_; ++w) {
        for (int j = 0; j < ndim; ++j) {
            const double range   = bounds[j].hi - bounds[j].lo;
            const double dist_lo = start.position[j] - bounds[j].lo;
            const double dist_hi = bounds[j].hi - start.position[j];
            const double sigma_j = std::min(
                1e-2 * range, 0.5 * std::min(dist_lo, dist_hi) + 1e-12);
            double v;
            for (int attempt = 0; attempt < 64; ++attempt) {
                v = start.position[j] + gauss(rng) * sigma_j;
                if (v >= bounds[j].lo && v <= bounds[j].hi) break;
            }
            pos[w][j] = std::clamp(v, bounds[j].lo, bounds[j].hi);
        }
    }
    return pos;
}

// ---------------------------------------------------------------------------
// make_state: evaluate log_prob for all initial walker positions
// ---------------------------------------------------------------------------

SamplerState EnsembleSampler::make_state(bayes::Model& model,
                                          std::vector<std::vector<double>> init_pos)
{
    const int ndim   = model.n_params();
    const int n_ds   = static_cast<int>(model.event().size());
    const int n_fl   = n_ds * 2;

    SamplerState st;
    st.nwalkers  = nwalkers_;
    st.ndim      = ndim;
    st.n_fluxes  = n_fl;
    st.pos.resize(nwalkers_ * ndim);
    st.log_prob.resize(nwalkers_);
    st.fluxes.resize(nwalkers_ * n_fl, 0.0);
    // Caller is responsible for setting st.rng after make_state returns.

    std::vector<bayes::Model::FluxSolution> fl_buf;
    for (int w = 0; w < nwalkers_; ++w) {
        std::copy(init_pos[w].begin(), init_pos[w].end(), st.pos_row(w));
        st.log_prob[w] = model.log_prob_and_fluxes(init_pos[w], fl_buf);
        for (int k = 0; k < n_ds && k < static_cast<int>(fl_buf.size()); ++k) {
            st.flux_row(w)[2*k]   = fl_buf[k].Fs;
            st.flux_row(w)[2*k+1] = fl_buf[k].Fb;
        }
    }
    return st;
}

// ---------------------------------------------------------------------------
// init_state overloads
// ---------------------------------------------------------------------------

SamplerState EnsembleSampler::init_state(bayes::Model& model)
{
    const int ndim = model.n_params();
    const auto bounds = model.optimizer_bounds();
    std::mt19937_64 rng(seed_);
    auto rand01 = [&]{ return std::uniform_real_distribution<double>(0,1)(rng); };
    std::vector<std::vector<double>> pos(nwalkers_, std::vector<double>(ndim));
    for (int w = 0; w < nwalkers_; ++w)
        for (int j = 0; j < ndim; ++j)
            pos[w][j] = bounds[j].lo + rand01() * (bounds[j].hi - bounds[j].lo);
    auto st = make_state(model, std::move(pos));
    st.rng = std::move(rng);  // advance past position draws, not replayed in step()
    return st;
}

SamplerState EnsembleSampler::init_state(bayes::Model& model,
                                           const optimize::Result& start,
                                           bool hessian_init)
{
    const int ndim = model.n_params();
    if (static_cast<int>(start.position.size()) != ndim)
        throw std::invalid_argument("start.position size does not match n_params");
    const auto bounds = model.optimizer_bounds();
    std::mt19937_64 rng(seed_);
    auto pos = hessian_init
        ? init_pos_hessian(model, start, bounds, rng)
        : init_pos_ball(start, bounds, rng);
    auto st = make_state(model, std::move(pos));
    st.rng = std::move(rng);
    return st;
}

SamplerState EnsembleSampler::init_state(
    bayes::Model& model, const std::vector<std::vector<double>>& pos)
{
    if (static_cast<int>(pos.size()) != nwalkers_)
        throw std::invalid_argument("pos must have nwalkers rows");
    auto st = make_state(model, pos);
    st.rng.seed(seed_);
    return st;
}

// ---------------------------------------------------------------------------
// step: one full ensemble update (both halves), in-place
// ---------------------------------------------------------------------------

void EnsembleSampler::step(bayes::Model& model, SamplerState& state)
{
    const int NW   = state.nwalkers;
    const int ndim = state.ndim;
    const int half = NW / 2;
    const int n_ds = static_cast<int>(model.event().size());

    auto rand01 = [&]{ return std::uniform_real_distribution<double>(0,1)(state.rng); };

    std::vector<bayes::Model::FluxSolution> prop_fl;
    std::vector<std::vector<double>> complement(half, std::vector<double>(ndim));
    std::vector<double> cur(ndim);

    for (int half_idx = 0; half_idx < 2; ++half_idx) {
        const int begin_this = half_idx == 0 ? 0    : half;
        const int begin_comp = half_idx == 0 ? half : 0;

        for (int i = 0; i < half; ++i)
            std::copy(state.pos_row(begin_comp + i),
                      state.pos_row(begin_comp + i) + ndim,
                      complement[i].begin());

        for (int k = begin_this; k < begin_this + half; ++k) {
            std::copy(state.pos_row(k), state.pos_row(k) + ndim, cur.begin());
            auto [prop_vec, log_factor] =
                move_->propose(cur, complement, state.rng, ndim);

            const double lp_prop = model.log_prob_and_fluxes(prop_vec, prop_fl);
            const double log_acc = log_factor + lp_prop - state.log_prob[k];

            if (std::log(rand01()) <= log_acc) {
                std::copy(prop_vec.begin(), prop_vec.end(), state.pos_row(k));
                state.log_prob[k] = lp_prop;
                for (int j = 0; j < n_ds && j < static_cast<int>(prop_fl.size()); ++j) {
                    state.flux_row(k)[2*j]   = prop_fl[j].Fs;
                    state.flux_row(k)[2*j+1] = prop_fl[j].Fb;
                }
                ++state.n_accepted;
            }
            ++state.n_total;
        }
    }
    ++state.n_step;
}

// ---------------------------------------------------------------------------
// collect: build Chain from accumulated state (no stored history — caller
// must have collected snapshots if they want per-step data; this just wraps
// the current walker positions as a single "step" for convenience, or the
// caller can build their own Chain from snapshots).
// ---------------------------------------------------------------------------

Chain EnsembleSampler::collect(bayes::Model& model,
                                const SamplerState& state, int discard) const
{
    // collect() is a thin convenience: build a 1-step Chain from current positions.
    // Full per-step chains are built by run_from_state when using batch mode.
    // In step-by-step mode, users typically build their own Chain or use numpy.
    (void)discard;
    const int ndim = state.ndim;
    const int n_ds = static_cast<int>(model.event().size());
    Chain chain;
    chain.init(1, state.nwalkers, ndim);
    {
        std::vector<std::string> names, transforms;
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
        for (int k = 0; k < n_ds; ++k)
            ds_names.push_back(model.event().at(k).name());
        chain.init_fluxes(n_ds, std::move(ds_names));
    }
    chain.push_step(state.pos, state.log_prob, state.fluxes);
    chain.set_acceptance(state.acceptance_fraction());
    return chain;
}

// ---------------------------------------------------------------------------
// run_from_state: batch loop — advance nsteps, collect into Chain
// ---------------------------------------------------------------------------

Chain EnsembleSampler::run_from_state(bayes::Model& model, SamplerState state,
                                       int nsteps, int burnin)
{
    const int ndim = state.ndim;
    const int n_ds = static_cast<int>(model.event().size());

    // burnin
    for (int i = 0; i < burnin; ++i)
        step(model, state);

    Chain chain;
    chain.init(nsteps, state.nwalkers, ndim);
    {
        std::vector<std::string> names, transforms;
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
        for (int k = 0; k < n_ds; ++k)
            ds_names.push_back(model.event().at(k).name());
        chain.init_fluxes(n_ds, std::move(ds_names));
    }

    for (int i = 0; i < nsteps; ++i) {
        step(model, state);
        chain.push_step(state.pos, state.log_prob, state.fluxes);
    }
    chain.set_acceptance(state.acceptance_fraction());
    return chain;
}

// ---------------------------------------------------------------------------
// Batch run() convenience wrappers
// ---------------------------------------------------------------------------

Chain EnsembleSampler::run(bayes::Model& model, int nsteps, int burnin)
{
    return run_from_state(model, init_state(model), nsteps, burnin);
}

Chain EnsembleSampler::run(bayes::Model& model, const optimize::Result& start,
                            int nsteps, int burnin, bool hessian_init)
{
    return run_from_state(model, init_state(model, start, hessian_init),
                          nsteps, burnin);
}

Chain EnsembleSampler::run(bayes::Model& model,
                            const std::vector<std::vector<double>>& start_pos,
                            int nsteps, int burnin)
{
    return run_from_state(model, init_state(model, start_pos), nsteps, burnin);
}

} // namespace lcbinint::sample
