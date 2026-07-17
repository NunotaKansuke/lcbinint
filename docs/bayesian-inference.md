# Bayesian inference contract

This note defines the posterior sampled by `lcbinint` and the supported
integration boundary with an external Galactic model such as `gapmoe.Model`.
It is the reference for the inference API.

## Canonical API

Each public entry point owns one part of the calculation.

| API | Responsibility |
| --- | --- |
| `model.param(name, prior)` | Register a sampled coordinate and its prior. |
| `model.reparam(targets)` | Replace light-curve coordinates by user-defined sampling coordinates. |
| `model.likelihood(..., flux=...)` | Define the data likelihood and treatment of linear fluxes. |
| `model.theta_star(...)` | Define `p(log(thetaS) | Fs)` or the source magnitudes used by an isochrone. |
| `model.galactic_prior(provider, ...)` | Add one independently constructed Galactic-density provider. |
| `@model.prior` | Add an ordinary log-prior or a bound. |
| `@model.guard` | Reject numerically unsafe points before magnification is evaluated. |

`thetaS` is not registered with `model.param`. A fixed value is represented by
returning `(log(thetaS), 0.0)` from `model.theta_star`.

The Galactic provider owns its physical density, parameterization, Jacobian,
and hidden-distance integration. `lcbinint` only assembles its named input
coordinates and adds the returned log density.

## Joint posterior

Let

- `u` be the coordinates traversed by the sampler,
- `eta = T(u)` be the light-curve parameters,
- `F = {(Fs_j, Fb_j)}` be dataset fluxes,
- `k_j` be dataset error scales,
- `t = thetaS` be the angular source radius,
- `z = (ML, DL, DS, mu_N, mu_E, ...)` be hidden physical variables, and
- `y` be the observed light curves.

The full model represented by the exact sampling modes is, up to normalization,

```text
p(u, F, k, t, z | y)
  proportional to
  L(y | T(u), F, k)
  p_u(u)
  p(t | F)
  p_G(T(u), t, z | source photometry(F))
  p_extra(T(u), z).
```

The sampler evaluates either this density or an explicitly marginalized
version of it. `@model.guard` is not a probabilistic factor: it defines the
domain on which the numerical model is valid. A scientific bound belongs in
`@model.prior` or in `provider.prior`.

## Sampling coordinates and Jacobians

`LogUniform(a, b)` parameters are traversed in `log(x)`. The implementation
evaluates

```text
log p_x(exp(u)) + u,
```

so the coordinate Jacobian is included.

Priors registered through `model.reparam(...).param(...)` are priors on the
replacement sampling variables. No additional Jacobian is needed for that
model. If the intended prior is instead defined on the generated target
variables, the corresponding change-of-variables term must be supplied with
`@rp.log_jacobian` or expressed explicitly with `@model.prior`.

## Flux modes

For one dataset, write the Gaussian model as

```text
y = Fs A(eta) + Fb + epsilon,
epsilon ~ Normal(0, k^2 Sigma).
```

### `flux="sample"`

`Fs` and `Fb` are ordinary MCMC parameters. Both require explicit priors. This
is the general exact mode when a non-flat flux prior or arbitrary downstream
flux dependence is required.

### `flux="marginalize"`

This mode is available for the Gaussian likelihood. It analytically integrates
`Fs`, `Fb`, and `k` under

```text
p(Fs, Fb) proportional to 1,
p(k) proportional to 1/k.
```

For `n` data points and `X = [A, 1]`, define

```text
H = X^T W X,
D = det(H),
chi2_min = min_beta (y - X beta)^T W (y - X beta).
```

After removing constants independent of `eta`, the implemented likelihood is

```text
log L_marg(eta)
  = -(n - 2)/2 log(chi2_min) - 1/2 log(D).
```

The conditional marginal for `Fs` is Student-t with

```text
df = n - 2,
mean = Fs_hat,
scale^2 = chi2_min/(n - 2) * H^{-1}[Fs, Fs].
```

This conditional is used when `theta_star` or source magnitudes depend on
`Fs`. Different datasets are drawn jointly with fixed scrambled Sobol points.
Downstream dependence on marginalized `Fb`, or on the full `Fs`-`Fb`
covariance, is not supported; use `flux="sample"` for that case.

### `flux="fit"`

This inserts the weighted least-squares `Fs_hat, Fb_hat` into the likelihood.
It is a profile likelihood, not a Bayesian marginal likelihood. It is useful as
a fast approximation when source-flux uncertainty is negligible, but it must
not be interpreted as sampling the full flux posterior.

The Student-t data likelihood currently supports `fit` and `sample`, not
`marginalize`.

## Angular source radius

The empirical form

```python
@model.theta_star(samples=256, seed=0)
def source_radius(fluxes):
    return log_center, log_sigma
```

defines a normal density in `log(thetaS)`. A positive `log_sigma` is integrated
with fixed Sobol points; zero is a deterministic delta distribution. If flux is
also marginalized, each Sobol row contains a conditional `Fs` draw and a
`thetaS` draw. Therefore the evaluated factor is

```text
E_{F | y, eta} E_{thetaS | F} [p_G(eta, thetaS, ...)].
```

The isochrone form returns apparent source magnitudes instead. The matching
Galactic provider jointly integrates `thetaS`, source distance, and any other
hidden physical variables. The Galactic density is conditioned on those
magnitudes:

```text
p_G(z | magnitudes) = p_G(z, magnitudes) / p_G(magnitudes).
```

Consequently the CMD density is not silently applied as an additional prior on
the light-curve source flux.

## Galactic hidden variables

For parallax parameterizations, physical variables are either deterministic at
fixed `DS` or integrated over the provider's source-distance measure. For a
parallax-free model, `thetaE` and `mu` are known from the light-curve state but
`ML`, `DL`, `DS`, and proper-motion direction are hidden.

The Flow backend uses fixed importance points in `(DS, ML, phi)`. With

```text
x = DL/DS = kappa ML / (kappa ML + thetaE^2 DS),
```

the change-of-variables factor with respect to
`dML dDS dphi` is

```text
J = 2 x DS (1 - x) mu / thetaE.
```

For proposal `q(DS) q(ML) q(phi)`, each QMC term is

```text
log p_G(ML, DL, DS, mu_N, mu_E)
+ log J
- log q(DS) - log q(ML) - log q(phi).
```

The likelihood sees the log-mean-exp of these fixed terms. Fixed points make
the approximate posterior deterministic, which is required by the ensemble
MCMC transition. Increasing `integration_samples` changes the numerical
approximation, not the statistical model.

The Histogram backend directly evaluates its precomputed `DL x DS` quadrature
for an ordinary parallax-free model. Dynamic isochrone/source-flux conditioning
or hidden physical priors would require a separate importance proposal and are
intentionally unsupported for that backend. Use Flow or sample distances
explicitly.

## Conditional physical reconstruction

The MCMC chain contains only non-marginalized coordinates. When the sampled
model registered a Galactic provider, the returned chain contains hidden
physical draws produced during posterior evaluation:

```python
physical = chain.get_physical(
    flat=True,
    discard=burnin,
    thin=thin,
)
```

The Galactic provider, parameter names, context, and source-magnitude mapping
come from the model's existing `galactic_prior(...)` registration. During each
production step, deterministic values or one conditional hidden-variable draw
are stored with every walker. HDF5 stores the same arrays under `physical/`, so
`load_chain(path).get_physical()` does not need the original model.

Providers exposing `log_density_and_physical(...)` return the marginal log
density and conditional draw together. lcbinint then selects the draw from the
same quadrature/QMC weights used by the accepted posterior evaluation; neither
the magnification nor the Galactic integral is repeated. Providers implementing
only `sample_physical(...)` remain supported through post-step reconstruction.

```text
p(F, thetaS, z | eta, y)
  proportional to
  p(F | eta, y) p(thetaS | F) p_G(eta, thetaS, z).
```

For Flow integration, the implementation draws one stored QMC candidate with
probability proportional to its finite importance weight. For Histogram
integration, it draws from the existing analytic distance-quadrature terms and
uses the inverse CDF of the interpolated proper-motion direction histogram.
Neither path performs an additional Galactic-density evaluation.

## Supported combinations

| Combination | Status |
| --- | --- |
| Gaussian + flux sample | Exact MCMC target, subject to user priors. |
| Gaussian + flux marginalize | Exact analytic marginal under flat flux and Jeffreys scale priors. |
| Gaussian + flux fit | Supported profile approximation. |
| Student-t + flux fit/sample | Supported; fit remains a profile approximation. |
| Empirical thetaS + fit/sample/marginalize | Supported. |
| Flow + isochrone + parallax with sampled or marginalized distance | Supported. |
| Flow + isochrone + no parallax | Supported with fixed QMC; test convergence with `integration_samples`. |
| Histogram + ordinary no-parallax model | Supported by deterministic distance quadrature. |
| Histogram + dynamic isochrone/Fs conditioning + no parallax | Unsupported. |
| Flux-marginalized downstream dependence on Fb | Unsupported; sample fluxes instead. |

## Validation checklist

Before interpreting a chain:

1. Confirm every sampled coordinate has an explicit prior.
2. Treat `fit` results as a profile approximation.
3. For QMC modes, repeat representative log-density evaluations with at least
   two seeds and a larger `integration_samples` value.
4. Check that posterior summaries are stable under burn-in and thinning
   choices and that the ensemble has mixed.
5. Reconstruct hidden physical values with the same provider and context used
   during sampling.
