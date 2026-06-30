#include "chain.hpp"
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>

namespace lcbinint::sample {

void Chain::init(int nsteps, int nwalkers, int ndim)
{
    nsteps_     = nsteps;
    nwalkers_   = nwalkers;
    ndim_       = ndim;
    step_count_ = 0;
    flat_samples_ .assign(static_cast<std::size_t>(nsteps * nwalkers * ndim), 0.0);
    flat_log_prob_.assign(static_cast<std::size_t>(nsteps * nwalkers), 0.0);
}

void Chain::init_fluxes(int n_datasets, std::vector<std::string> names)
{
    n_fluxes_      = n_datasets * 2;
    dataset_names_ = std::move(names);
    flat_fluxes_.assign(
        static_cast<std::size_t>(nsteps_ * nwalkers_ * n_fluxes_), 0.0);
}

void Chain::push_step(const std::vector<double>& positions,
                      const std::vector<double>& log_probs)
{
    const std::size_t base_s = static_cast<std::size_t>(step_count_ * nwalkers_ * ndim_);
    const std::size_t base_l = static_cast<std::size_t>(step_count_ * nwalkers_);
    std::copy(positions.begin(), positions.end(), flat_samples_.begin() + base_s);
    std::copy(log_probs.begin(), log_probs.end(), flat_log_prob_.begin() + base_l);
    ++step_count_;
}

void Chain::push_step(const std::vector<double>& positions,
                      const std::vector<double>& log_probs,
                      const std::vector<double>& fluxes)
{
    push_step(positions, log_probs);  // increments step_count_
    if (!flat_fluxes_.empty() && n_fluxes_ > 0) {
        const std::size_t base_f =
            static_cast<std::size_t>((step_count_ - 1) * nwalkers_ * n_fluxes_);
        std::copy(fluxes.begin(), fluxes.end(), flat_fluxes_.begin() + base_f);
    }
}

void Chain::assign_flat(const double* pos_src,
                        const double* lp_src,
                        const double* fl_src)
{
    std::copy(pos_src, pos_src + flat_samples_.size(),  flat_samples_.begin());
    std::copy(lp_src,  lp_src  + flat_log_prob_.size(), flat_log_prob_.begin());
    step_count_ = nsteps_;
    if (fl_src && !flat_fluxes_.empty())
        std::copy(fl_src, fl_src + flat_fluxes_.size(), flat_fluxes_.begin());
}

// ---------------------------------------------------------------------------
// Autocorrelation time (Sokal auto-window, FFT-based)
// ---------------------------------------------------------------------------

// Compute integrated autocorrelation time for a single parameter.
// Data layout: x[t * stride_s + w * stride_w] for t in 0..nsteps-1, w in 0..nwalkers-1.
// Averages the normalized ACF over walkers, then applies the Sokal window.
static double tau_1d(const double* x, int nsteps, int nwalkers,
                      int stride_s, int stride_w, double c,
                      gsl_fft_real_wavetable*        wt,
                      gsl_fft_halfcomplex_wavetable* hwt,
                      gsl_fft_real_workspace*        ws,
                      int n_fft, std::vector<double>& buf)
{
    std::vector<double> acf(nsteps, 0.0);

    for (int w = 0; w < nwalkers; ++w) {
        // Center the time series
        double mean = 0.0;
        for (int t = 0; t < nsteps; ++t)
            mean += x[t * stride_s + w * stride_w];
        mean /= nsteps;

        for (int t = 0; t < nsteps; ++t)
            buf[t] = x[t * stride_s + w * stride_w] - mean;
        std::fill(buf.begin() + nsteps, buf.end(), 0.0);

        // Forward real FFT → half-complex format
        gsl_fft_real_transform(buf.data(), 1, static_cast<std::size_t>(n_fft), wt, ws);

        // Power spectrum in-place (half-complex → half-complex with Im=0)
        // Half-complex layout for even n_fft:
        //   data[0]       = Re(X[0])
        //   data[2k-1]    = Re(X[k])  for k = 1..n_fft/2-1
        //   data[2k]      = Im(X[k])  for k = 1..n_fft/2-1
        //   data[n_fft-1] = Re(X[n_fft/2])  (Nyquist, real only)
        buf[0] = buf[0] * buf[0];
        for (int k = 1; k < n_fft / 2; ++k) {
            const double re = buf[2 * k - 1];
            const double im = buf[2 * k];
            buf[2 * k - 1] = re * re + im * im;
            buf[2 * k]     = 0.0;
        }
        buf[n_fft - 1] = buf[n_fft - 1] * buf[n_fft - 1];  // Nyquist

        // Inverse half-complex FFT → real autocorrelation (normalized by 1/n_fft)
        gsl_fft_halfcomplex_inverse(buf.data(), 1, static_cast<std::size_t>(n_fft), hwt, ws);

        for (int t = 0; t < nsteps; ++t)
            acf[t] += buf[t];
    }

    // Average over walkers; normalize by variance (acf[0] after averaging)
    const double total_var = acf[0];  // sum over walkers of variance / n_fft
    if (total_var <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    for (double& v : acf) v /= total_var;

    // Sokal auto-window: find first M where M < c * tau_est(M)
    // tau_est(M) = 1 + 2 * sum_{lag=1}^{M} acf[lag]
    double tau_est = 1.0;
    for (int m = 1; m < nsteps; ++m) {
        tau_est += 2.0 * acf[m];
        if (static_cast<double>(m) >= c * tau_est)
            return tau_est;
    }
    return tau_est;  // chain too short for window to close
}

std::vector<double> Chain::tau(double c) const
{
    const int N = nsteps_;
    const int W = nwalkers_;
    const int D = ndim_;

    std::vector<double> result(D, std::numeric_limits<double>::quiet_NaN());
    if (N < 4 || W < 2 || D < 1) return result;

    // Zero-pad to next power of 2 >= 2*nsteps
    int n_fft = 1;
    while (n_fft < 2 * N) n_fft <<= 1;

    auto* wt  = gsl_fft_real_wavetable_alloc(static_cast<std::size_t>(n_fft));
    auto* hwt = gsl_fft_halfcomplex_wavetable_alloc(static_cast<std::size_t>(n_fft));
    auto* ws  = gsl_fft_real_workspace_alloc(static_cast<std::size_t>(n_fft));
    std::vector<double> buf(n_fft);

    // flat_samples_ layout: [step * W * D + walker * D + dim]
    // stride_s = W * D  (step stride), stride_w = D  (walker stride)
    const int stride_s = W * D;
    const int stride_w = D;
    const double* base = flat_samples_.data();

    for (int j = 0; j < D; ++j)
        result[j] = tau_1d(base + j, N, W, stride_s, stride_w, c,
                            wt, hwt, ws, n_fft, buf);

    gsl_fft_real_wavetable_free(wt);
    gsl_fft_halfcomplex_wavetable_free(hwt);
    gsl_fft_real_workspace_free(ws);

    return result;
}

std::vector<double> Chain::ess() const
{
    const auto taus = tau();
    const double ntot = static_cast<double>(nsteps_) * nwalkers_;
    std::vector<double> result(ndim_);
    for (int j = 0; j < ndim_; ++j)
        result[j] = ntot / taus[j];
    return result;
}

} // namespace lcbinint::sample
