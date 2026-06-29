# lcbinint Sampler/Optimizer Architecture Design

## 1. Purpose

This document defines the next-stage architecture for `lcbinint`.

The main goal is to extend `lcbinint` from a magnification/light-curve evaluation library into a package that can also perform model fitting, optimization, and posterior sampling, while keeping the performance-critical path inside the C++ backend.

The design must minimize Python/C++ overhead during repeated model evaluation. In particular, samplers and optimizers should not repeatedly convert data through NumPy or call Python callbacks inside the inner loop. Observational data, parameter definitions, priors, likelihood evaluation, flux fitting, work buffers, and model evaluation should be represented in C++ objects and reused across evaluations.

The Python API should remain compact and object-oriented, while the backend should be structured so that high-volume computations are performed in C++.

## 2. Top-Level Module Layout

The proposed public Python module layout is:

```text
lcbinint
  ├─ lc
  │    ├─ LightCurve
  │    ├─ Options
  │    ├─ Parameters
  │    └─ LimbDarkening
  │
  ├─ obs
  │    ├─ LightCurveData
  │    └─ Event
  │
  ├─ bayes
  │    ├─ Model
  │    ├─ Uniform
  │    ├─ Normal
  │    └─ LogUniform
  │
  ├─ optimize
  │    ├─ DifferentialEvolution
  │    ├─ LevenbergMarquardt
  │    └─ Result
  │
  └─ sample
       ├─ EnsembleSampler
       ├─ MetropolisHastings
       └─ Chain
```

Each module has a narrow responsibility:

```text
lc
  Compute magnification A(t). This is the physical/numerical light-curve layer.

obs
  Store observed light-curve data and event-level observational context.

bayes
  Define the statistical model: parameters, priors, flux model, likelihood, and log probability.

optimize
  Run best-fit optimization against bayes.Model.

sample
  Run posterior sampling against bayes.Model.
```

## 3. `lc` Module

The `lc` module contains the existing magnification/light-curve evaluation functionality.

### Responsibility

`lc.LightCurve` computes magnification arrays:

```python
A = light_curve(times, params)
```

It should not know about observed fluxes, priors, likelihoods, samplers, optimizers, or fit results.

### Public API

```python
from lcbinint.lc import LightCurve, Options, Parameters, LimbDarkening

lc = LightCurve(
    options=Options(...),
    limb_darkening=LimbDarkening.linear(0.5),
)

params = Parameters(...)
A = lc(times, params)
```

### Backend Notes

The existing C++ magnification machinery should remain the core of this layer. Internally, a fitting/sampling-oriented evaluator may be added so that repeated evaluations can reuse workspaces and buffers instead of allocating new objects on every call.

The public API should expose simple magnification evaluation, while the C++ backend should provide a fast batch evaluation path for `bayes.Model`.

## 4. `obs` Module

The `obs` module stores observational data and event-level context.

## 4.1 `LightCurveData`

`LightCurveData` represents one observed light curve.

It should be backed by C++ containers, not NumPy arrays as the primary storage. NumPy should be used only for input/output convenience.

### Responsibilities

`LightCurveData` stores:

```text
time
flux
flux_err
weight        # usually 1 / flux_err^2, precomputed
mask          # optional validity mask
name
band
observatory
error_scale   # optional initial value or fixed scaling
metadata      # optional lightweight key/value metadata
```

### Python API

```python
from lcbinint.obs import LightCurveData

data = LightCurveData(
    time,
    flux,
    flux_err,
    name="OGLE_I",
    band="I",
    observatory="OGLE",
)
```

### C++ Storage Requirements

Prefer a structure-of-arrays layout:

```cpp
class LightCurveData {
public:
    std::span<const double> time() const;
    std::span<const double> flux() const;
    std::span<const double> flux_err() const;
    std::span<const double> weight() const;
    std::span<const uint8_t> mask() const;

    std::size_t size() const;

private:
    std::vector<double> time_;
    std::vector<double> flux_;
    std::vector<double> flux_err_;
    std::vector<double> weight_;
    std::vector<uint8_t> mask_;
};
```

The object should provide `.to_numpy()` or column-specific NumPy export methods for inspection, plotting, and interoperability. These exports should avoid copies when safe, but the C++ object must own the data used in repeated backend evaluation.

## 4.2 `Event`

`Event` represents one microlensing event and groups multiple `LightCurveData` objects.

### Responsibilities

`Event` stores:

```text
name
ra
dec
t_ref
list of LightCurveData
```

It should remain a lightweight observational container. It should not own priors, likelihoods, samplers, optimizers, chains, or fit results.

### Python API

```python
from lcbinint.obs import Event, LightCurveData

ogle = LightCurveData(..., name="OGLE_I", band="I", observatory="OGLE")
moa = LightCurveData(..., name="MOA_R", band="R", observatory="MOA")

event = Event(
    name="OGLE-2026-BLG-0001",
    ra=...,
    dec=...,
    t_ref=2460000.0,
)

event.add(ogle)
event.add(moa)
```

`Event` is required for multi-dataset modeling and for event-level context such as sky coordinates and reference time.

## 5. `bayes` Module

The `bayes` module defines the statistical model used by both optimizers and samplers.

This is the main object that combines:

```text
lc.LightCurve
obs.LightCurveData or obs.Event
free parameter definitions
priors
parameter transforms
flux model
likelihood
C++ evaluation workspace
```

The central class is `bayes.Model`.

## 5.1 `bayes.Model`

`bayes.Model` is a probabilistic/statistical model. It should not be named `LightCurveModel`, because that would be confused with `lc.LightCurve`, which is the magnification evaluator.

### Python API

```python
from lcbinint.lc import LightCurve, Options
from lcbinint.obs import Event
from lcbinint.bayes import Model, Uniform, LogUniform

lc = LightCurve(options=Options(...))

model = Model(light_curve=lc, event=event)

model.param("t0", Uniform(2460000.0, 2460100.0))
model.param("u0", Uniform(-1.0, 1.0))
model.param("tE", LogUniform(1.0, 1000.0))
model.param("sep", LogUniform(0.1, 10.0))
model.param("q", LogUniform(1e-6, 1.0))
model.param("theta", Uniform(0.0, 2 * np.pi))
model.param("rho", LogUniform(1e-5, 1e-1))

model.flux("linear_blend")
model.likelihood("gaussian")
```

A single `LightCurveData` may also be accepted as a convenience:

```python
model = Model(light_curve=lc, data=data)
```

Internally this can be treated as an event with one data object.

## 5.2 Parameter Definitions

Parameters are declared on `bayes.Model`.

Each parameter definition should include:

```text
name
prior distribution
mapping to lc.Parameters field, if needed
transform, if needed
fixed/free status
bounds implied by prior
```

The API should support simple declarations first:

```python
model.param("tE", LogUniform(1.0, 1000.0))
model.param("q", LogUniform(1e-6, 1.0))
```

If needed, explicit mapping can be added:

```python
model.param("log_q", Uniform(-6.0, 0.0), maps_to="q", transform="exp10")
```

The backend should maintain an ordered parameter vector for optimizers and samplers.

## 5.3 Prior Ownership

Priors belong to `bayes.Model`, not to samplers.

Samplers should only call:

```python
model.log_prob(theta)
```

Optimizers should call:

```python
model.chi2(theta)
```

or minimize:

```python
-model.log_prob(theta)
```

This keeps the sampling algorithms independent of microlensing-specific parameter logic.

## 5.4 Flux Model

The default flux model should be:

```text
F_model(t) = Fs * A(t) + Fb
```

For fixed lens parameters, `Fs` and `Fb` should be solved in C++ by linear least squares for each `LightCurveData` object.

The flux solver should be internal at first. The public API can be:

```python
model.flux("linear_blend")
```

The fitted fluxes should be included in optimization and sampling results where appropriate.

## 5.5 Likelihood

The initial built-in likelihood should be Gaussian:

```python
model.likelihood("gaussian")
```

For each data point:

```text
chi2 = sum(((flux_obs - flux_model) / flux_err)^2)
log_likelihood = -0.5 * chi2 + optional normalization
```

The model should expose:

```python
model.chi2(theta)
model.log_likelihood(theta)
model.log_prior(theta)
model.log_prob(theta)
model.predict(theta)
model.magnification(theta)
```

`log_prob(theta)` should return:

```text
log_prior(theta) + log_likelihood(theta)
```

## 5.6 Backend Evaluation Flow

The high-performance C++ evaluation flow should be:

```text
theta
  ↓
bayes.Model parameter table / transforms / priors
  ↓
lc.Parameters
  ↓
lc.LightCurve C++ evaluator
  ↓
A(t)
  ↓
linear flux solver: F = Fs * A + Fb
  ↓
likelihood
  ↓
chi2 / log_likelihood / log_prob
```

The inner loop must avoid Python callbacks and repeated NumPy conversions.

## 6. `optimize` Module

The `optimize` module contains optimization algorithms that operate on `bayes.Model`.

### API

```python
from lcbinint.optimize import DifferentialEvolution, LevenbergMarquardt

best = DifferentialEvolution(
    population_size=64,
    max_iter=2000,
    seed=1,
).minimize(model)

refined = LevenbergMarquardt(max_iter=500).minimize(
    model,
    start=best,
)
```

### Target Selection

Default optimizer target:

```text
chi2
```

Optional targets:

```python
best_chi2 = DifferentialEvolution().minimize(model, target="chi2")
best_map = DifferentialEvolution().minimize(model, target="neg_log_prob")
```

### Optimize Result

`optimize.Result` should expose:

```text
position          # optimizer/sampler-space theta
parameters        # physical parameter dictionary
chi2
log_likelihood
log_prob
fluxes            # per LightCurveData: Fs, Fb, etc.
success
message
n_eval
```

`start` arguments should accept either a raw parameter vector or an `optimize.Result`.

## 7. `sample` Module

The `sample` module contains posterior samplers that operate on `bayes.Model`.

### API

```python
from lcbinint.sample import EnsembleSampler

chain = EnsembleSampler(
    nwalkers=64,
    seed=2,
).run(
    model,
    start=refined,
    nsteps=5000,
    burnin=1000,
)
```

### Sampler Responsibility

Samplers should not own priors, likelihoods, data, or microlensing-specific parameter logic.

Samplers should only evaluate:

```python
model.log_prob(theta)
```

### Chain Object

`sample.Chain` should expose:

```text
samples          # raw theta samples
log_prob         # log probability per sample
parameters       # physical-parameter view/table
acceptance       # acceptance fraction or per-walker diagnostics
summary()
to_numpy()
to_pandas()
```

## 8. Full Usage Example

```python
import numpy as np

from lcbinint.lc import LightCurve, Options
from lcbinint.obs import LightCurveData, Event
from lcbinint.bayes import Model, Uniform, LogUniform
from lcbinint.optimize import DifferentialEvolution, LevenbergMarquardt
from lcbinint.sample import EnsembleSampler

lc = LightCurve(options=Options(...))

ogle = LightCurveData(
    time_ogle,
    flux_ogle,
    flux_err_ogle,
    name="OGLE_I",
    band="I",
    observatory="OGLE",
)

moa = LightCurveData(
    time_moa,
    flux_moa,
    flux_err_moa,
    name="MOA_R",
    band="R",
    observatory="MOA",
)

event = Event(
    name="OGLE-2026-BLG-0001",
    ra=...,
    dec=...,
    t_ref=2460000.0,
)
event.add(ogle)
event.add(moa)

model = Model(light_curve=lc, event=event)

model.param("t0", Uniform(2460000.0, 2460100.0))
model.param("u0", Uniform(-1.0, 1.0))
model.param("tE", LogUniform(1.0, 1000.0))
model.param("sep", LogUniform(0.1, 10.0))
model.param("q", LogUniform(1e-6, 1.0))
model.param("theta", Uniform(0.0, 2 * np.pi))
model.param("rho", LogUniform(1e-5, 1e-1))

model.flux("linear_blend")
model.likelihood("gaussian")

best = DifferentialEvolution().minimize(model)

refined = LevenbergMarquardt().minimize(
    model,
    start=best,
)

chain = EnsembleSampler(nwalkers=64).run(
    model,
    start=refined,
    nsteps=5000,
    burnin=1000,
)
```

## 9. C++ Backend Requirements

The main performance requirement is that repeated evaluation during optimization and sampling happens in C++.

### Required Backend Properties

1. `LightCurveData` owns C++ arrays for time, flux, error, weight, and mask.
2. `Event` owns or references multiple `LightCurveData` objects.
3. `bayes.Model` owns parameter definitions, priors, likelihood configuration, flux-model configuration, and reusable workspaces.
4. `bayes.Model::log_prob`, `bayes.Model::chi2`, and `bayes.Model::log_likelihood` are evaluated in C++.
5. Optimizers and samplers call C++ methods directly, avoiding Python callbacks in the inner loop.
6. Temporary arrays for magnification, residuals, model fluxes, and diagnostics are reused through workspaces.
7. Conversion to NumPy or pandas is for user-facing inspection only, not for internal repeated computation.

### Suggested Internal Objects

```text
LightCurveData
Event
BayesModel
ParameterDef
Prior
PriorSet
FluxSolver
Likelihood
EvaluationWorkspace
OptimizeResult
Chain
```

The exact C++ class names may differ, but the ownership and responsibility boundaries should follow this structure.

## 10. Future Extension: Custom Likelihoods

Custom likelihood support should be treated as a later feature.

Do not attempt automatic conversion of arbitrary Python functions to C++.

A future extension can support a restricted likelihood DSL:

```python
from lcbinint.bayes import likelihood as L

r = L.residual()
sigma = L.error()

custom = L.sum(
    -0.5 * (r / sigma) ** 2 - L.log(sigma)
)

model.likelihood(custom)
```

The DSL should build a C++ expression tree and evaluate it in the backend.

Recommended stages:

```text
Phase 1:
  Built-in likelihoods only: gaussian, student_t, huber.

Phase 2:
  Optional Python callback likelihood for debugging only. This is allowed to be slow.

Phase 3:
  Restricted likelihood DSL compiled into a C++ expression tree.

Phase 4:
  Optional C++ plugin interface for advanced users.
```

## 11. Initial Implementation Scope

The first implementation should focus on the minimal useful architecture:

```text
lcbinint.lc.LightCurve
lcbinint.lc.Options
lcbinint.lc.Parameters
lcbinint.lc.LimbDarkening

lcbinint.obs.LightCurveData
lcbinint.obs.Event

lcbinint.bayes.Model
lcbinint.bayes.Uniform
lcbinint.bayes.Normal
lcbinint.bayes.LogUniform

lcbinint.optimize.DifferentialEvolution
lcbinint.optimize.Result

lcbinint.sample.EnsembleSampler
lcbinint.sample.Chain
```

The implementation should prioritize stable object boundaries, C++ memory ownership, and reusable workspaces before adding advanced likelihoods or additional samplers.


