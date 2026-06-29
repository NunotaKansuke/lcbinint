#include "levenberg_marquardt.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace lcbinint::optimize {

// ---------------------------------------------------------------------------
// Tiny Cholesky solver for the (ndim x ndim) LM normal equations.
// Solves (A) x = b where A is symmetric positive definite.
// A is stored column-major in a flat vector of length ndim*ndim.
// ---------------------------------------------------------------------------

static bool cholesky_solve(std::vector<double>& A, // ndim×ndim, modified in place
                            const std::vector<double>& b,
                            std::vector<double>& x,
                            int n)
{
    // Cholesky decomposition: A = L L^T (lower-triangular L in place of lower A)
    for (int j = 0; j < n; ++j) {
        double sum = A[j * n + j];
        for (int k = 0; k < j; ++k)
            sum -= A[j * n + k] * A[j * n + k];
        if (sum <= 0.0) return false; // not positive definite
        A[j * n + j] = std::sqrt(sum);
        const double inv_ljj = 1.0 / A[j * n + j];
        for (int i = j + 1; i < n; ++i) {
            double s2 = A[i * n + j];
            for (int k = 0; k < j; ++k)
                s2 -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = s2 * inv_ljj;
        }
    }
    // Forward substitution: L y = b
    x.resize(n);
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k)
            s -= A[i * n + k] * x[k];
        x[i] = s / A[i * n + i];
    }
    // Back substitution: L^T x = y
    for (int i = n - 1; i >= 0; --i) {
        double s = x[i];
        for (int k = i + 1; k < n; ++k)
            s -= A[k * n + i] * x[k];
        x[i] = s / A[i * n + i];
    }
    return true;
}

// ---------------------------------------------------------------------------

LevenbergMarquardt::LevenbergMarquardt(
    int max_iter, double ftol, double xtol, double gtol,
    double fd_step, double lambda_init, double lambda_up, double lambda_down)
    : max_iter_(max_iter), ftol_(ftol), xtol_(xtol), gtol_(gtol)
    , fd_step_(fd_step)
    , lambda_init_(lambda_init), lambda_up_(lambda_up), lambda_down_(lambda_down)
{}

Result LevenbergMarquardt::minimize(bayes::Model& model, const Result& start)
{
    return minimize(model, start.position);
}

Result LevenbergMarquardt::minimize(bayes::Model& model,
                                    const std::vector<double>& start_theta)
{
    const int ndim   = model.n_params();
    const int n_data = model.n_data();
    if (ndim == 0) throw std::invalid_argument("Model has no free parameters");
    if (static_cast<int>(start_theta.size()) != ndim)
        throw std::invalid_argument("start_theta size does not match n_params");

    const auto bounds = model.optimizer_bounds();

    // Finite-difference step per parameter: fraction of prior range
    std::vector<double> h(ndim);
    for (int j = 0; j < ndim; ++j)
        h[j] = fd_step_ * (bounds[j].hi - bounds[j].lo);

    // Working state
    std::vector<double> theta = start_theta;
    std::vector<double> r     = model.residuals(theta);
    double chi2_cur = 0.0;
    for (double ri : r) chi2_cur += ri * ri;

    // Pre-allocate J (n_data × ndim), JtJ (ndim×ndim), Jtr (ndim), delta (ndim)
    std::vector<double> J(static_cast<std::size_t>(n_data) * ndim);
    std::vector<double> JtJ(ndim * ndim);
    std::vector<double> Jtr(ndim);
    std::vector<double> A(ndim * ndim);  // working copy for Cholesky
    std::vector<double> delta(ndim);
    std::vector<double> r_plus(n_data), r_minus(n_data);
    std::vector<double> theta_new(ndim);

    double lambda = lambda_init_;
    int n_eval = 1; // already evaluated start
    int iter;
    std::string msg = "max_iter reached";

    for (iter = 0; iter < max_iter_; ++iter) {
        // --- Build Jacobian via central finite differences ---
        // J[:, j] = (r(theta + h_j*e_j) - r(theta - h_j*e_j)) / (2*h_j)
        // Clip each probe point to bounds so lcbi never sees out-of-range params.
        for (int j = 0; j < ndim; ++j) {
            const double hj = h[j];
            theta_new = theta;
            theta_new[j] = std::clamp(theta[j] + hj, bounds[j].lo, bounds[j].hi);
            const double actual_plus = theta_new[j] - theta[j];
            r_plus = model.residuals(theta_new);

            theta_new = theta;
            theta_new[j] = std::clamp(theta[j] - hj, bounds[j].lo, bounds[j].hi);
            const double actual_minus = theta[j] - theta_new[j];
            r_minus = model.residuals(theta_new);
            n_eval += 2;

            // Use actual step size (may differ at boundary)
            const double denom = actual_plus + actual_minus;
            if (denom < 1e-30) continue; // degenerate (both sides clamped to same point)
            const double inv_denom = 1.0 / denom;
            for (int i = 0; i < n_data; ++i)
                J[static_cast<std::size_t>(i) * ndim + j] =
                    (r_plus[i] - r_minus[i]) * inv_denom;
        }

        // --- Form J^T J and J^T r ---
        std::fill(JtJ.begin(), JtJ.end(), 0.0);
        std::fill(Jtr.begin(), Jtr.end(), 0.0);
        for (int i = 0; i < n_data; ++i) {
            const std::size_t row_off = static_cast<std::size_t>(i) * ndim;
            for (int j = 0; j < ndim; ++j) {
                const double Jij = J[row_off + j];
                Jtr[j] += Jij * r[i];
                for (int k = j; k < ndim; ++k)
                    JtJ[j * ndim + k] += Jij * J[row_off + k];
            }
        }
        // Symmetrize
        for (int j = 0; j < ndim; ++j)
            for (int k = j + 1; k < ndim; ++k)
                JtJ[k * ndim + j] = JtJ[j * ndim + k];

        // --- Check gradient convergence ---
        double gnorm = 0.0;
        for (int j = 0; j < ndim; ++j) gnorm += Jtr[j] * Jtr[j];
        if (std::sqrt(gnorm) < gtol_) { msg = "gradient converged"; break; }

        // Diagonal floor: prevents zero regularization for insensitive directions.
        // Uses the largest diagonal of J^T J scaled by eps so the floor is relative.
        double max_diag = 0.0;
        for (int j = 0; j < ndim; ++j)
            if (JtJ[j * ndim + j] > max_diag) max_diag = JtJ[j * ndim + j];
        const double diag_floor = (max_diag > 0.0) ? max_diag * 1e-6 : 1.0;

        // --- LM step with adaptive lambda ---
        bool step_accepted = false;
        for (int try_ = 0; try_ < 16; ++try_) {
            // A = J^T J + lambda * max(diag(J^T J), floor)
            // The floor ensures regularization is non-zero even for zero-sensitivity dims.
            A = JtJ;
            for (int j = 0; j < ndim; ++j)
                A[j * ndim + j] += lambda * std::max(JtJ[j * ndim + j], diag_floor);

            // Solve A delta = -Jtr
            std::vector<double> rhs(ndim);
            for (int j = 0; j < ndim; ++j) rhs[j] = -Jtr[j];
            if (!cholesky_solve(A, rhs, delta, ndim)) {
                lambda *= lambda_up_;
                continue;
            }

            // Guard against NaN/inf delta (can occur if J^TJ is near-singular despite
            // regularization); treat the same as Cholesky failure.
            bool delta_ok = true;
            for (double d : delta) if (!std::isfinite(d)) { delta_ok = false; break; }
            if (!delta_ok) { lambda *= lambda_up_; continue; }

            // Clip proposed step to stay within bounds
            theta_new = theta;
            for (int j = 0; j < ndim; ++j)
                theta_new[j] = std::clamp(theta[j] + delta[j],
                                           bounds[j].lo, bounds[j].hi);

            const std::vector<double> r_new = model.residuals(theta_new);
            ++n_eval;
            double chi2_new = 0.0;
            for (double ri : r_new) chi2_new += ri * ri;

            if (chi2_new < chi2_cur) {
                // Accept step
                const double chi2_rel = std::abs(chi2_cur - chi2_new) / (chi2_cur + 1e-30);
                theta   = theta_new;
                r       = r_new;
                chi2_cur = chi2_new;
                lambda /= lambda_down_;

                // Update h to track scale
                for (int j = 0; j < ndim; ++j)
                    h[j] = fd_step_ * (bounds[j].hi - bounds[j].lo);

                // Check delta convergence
                double dnorm = 0.0, tnorm = 0.0;
                for (int j = 0; j < ndim; ++j) {
                    const double dj = theta[j] - (theta_new[j] - delta[j]);
                    dnorm += dj * dj;
                    tnorm += theta[j] * theta[j];
                }
                if (std::sqrt(dnorm) < xtol_ * (std::sqrt(tnorm) + 1e-30)) {
                    msg = "step converged"; step_accepted = true; break;
                }
                // Check chi2 convergence
                if (chi2_rel < ftol_) {
                    msg = "chi2 converged"; step_accepted = true; break;
                }
                step_accepted = true;
                break;
            } else {
                lambda *= lambda_up_;
            }
        }
        if (msg != "max_iter reached") break;
    }

    // --- Build result ---
    const auto& param_defs = model.param_defs();
    Result res;
    res.position       = theta;
    res.chi2           = chi2_cur;
    res.log_likelihood = -0.5 * chi2_cur;
    res.log_prob       = model.log_prob(theta);
    res.success        = (msg != "max_iter reached");
    res.message        = msg;
    res.n_eval         = n_eval;
    res.n_iter         = iter;

    int idx = 0;
    for (const auto& def : param_defs) {
        if (def.fixed) {
            res.parameters[def.name] = def.fixed_value;
        } else {
            const double t = theta[idx++];
            res.parameters[def.name] =
                (def.transform == bayes::Transform::log) ? std::exp(t) : t;
        }
    }

    const auto flux_sols = model.fluxes(theta);
    for (std::size_t k = 0; k < flux_sols.size(); ++k) {
        const std::string name = (k < model.event().size())
            ? model.event().at(k).name() : ("ds" + std::to_string(k));
        res.fluxes.push_back({name, flux_sols[k].Fs, flux_sols[k].Fb});
    }
    return res;
}

} // namespace lcbinint::optimize
