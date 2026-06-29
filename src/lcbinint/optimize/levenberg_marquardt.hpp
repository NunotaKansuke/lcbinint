#pragma once
#include "result.hpp"
#include "../bayes/model.hpp"

namespace lcbinint::optimize {

// Levenberg-Marquardt optimizer.
// Uses central finite differences to build the Jacobian J (n_data × ndim),
// then solves (J^T J + lambda * diag(J^T J)) delta = -J^T r at each step.
// The linear system is small (ndim × ndim) regardless of n_data, so no external
// linear algebra library is needed — a simple Cholesky solve suffices.
// All work is done in transformed space (log for LogUniform priors).
class LevenbergMarquardt {
public:
    explicit LevenbergMarquardt(
        int    max_iter    = 200,
        double ftol        = 1e-8,   // convergence: relative chi2 change
        double xtol        = 1e-8,   // convergence: |delta| / |theta|
        double gtol        = 1e-8,   // convergence: |gradient| (J^T r)
        double fd_step     = 1e-5,   // finite-difference step as fraction of param range
        double lambda_init = 1e-3,
        double lambda_up   = 3.0,
        double lambda_down = 3.0
    );

    // Start from a given theta in transformed space (e.g., from DifferentialEvolution result).
    Result minimize(bayes::Model& model, const Result& start);
    Result minimize(bayes::Model& model, const std::vector<double>& start_theta);

private:
    int    max_iter_;
    double ftol_;
    double xtol_;
    double gtol_;
    double fd_step_;
    double lambda_init_;
    double lambda_up_;
    double lambda_down_;
};

} // namespace lcbinint::optimize
