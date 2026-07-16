# Design: Python-extended `bayes.Model`

## Goal

Extend `bayes.Model` to support custom Python prior and likelihood terms via
decorators, while keeping the existing API and C++ fast path fully intact.

## API (user-facing)

### Case 1 — existing code, unchanged

```python
model = bayes.Model(lc, event)
model.param("tE",   bayes.LogUniform(1, 1000))
model.param("u0",   bayes.Uniform(0, 2))
model.param("piEN", bayes.Uniform(-1, 1))
model.param("piEE", bayes.Uniform(-1, 1))
model.likelihood("gaussian")
run_sampler(model, ...)
```

### Case 2 — gaussian + extra likelihood (e.g. parallax constraint)

```python
model = bayes.Model(lc, event)
model.param(...)
model.likelihood("gaussian")

@model.likelihood
def gaia(piEN, piEE, **_):
    piE = (piEN**2 + piEE**2)**0.5
    return -0.5 * ((piE - 0.3) / 0.05)**2
```

### Case 3 — gaussian + extra prior (Jacobian correction etc.)

```python
model = bayes.Model(lc, event)
model.param(...)
model.likelihood("gaussian")

@model.prior
def jacobian(tE, **_):
    return -math.log(tE)
```

### Case 4 — fully custom likelihood (no gaussian)

```python
model = bayes.Model(lc, event)
model.param(...)

@model.likelihood
def my_lik(tE, u0, **_):
    return my_student_t(tE, u0)
```

### Case 5 — local sampling reparameterization

```python
model = bayes.Model(lc, event)
model.param("tE", bayes.LogUniform(...))
model.likelihood("gaussian")

rp = model.reparam(["piEN", "piEE"])
rp.param("piE", bayes.LogUniform(...))
rp.param("phi_piE", bayes.Uniform(...))

@rp.transform
def to_phys(piE, phi_piE):
    return {
        "piEN": piE * math.cos(phi_piE),
        "piEE": piE * math.sin(phi_piE),
    }

run_sampler(model, ...)
```

## Semantics

```
log_prob(theta) = log_prior_cpp(theta)           # C++: param priors (always)
               + Σ extra_prior_fn(**vals)         # Python: @model.prior terms
               + log_likelihood_cpp(theta)        # C++: gaussian chi2 (if mode set)
               + Σ extra_lik_fn(**vals)           # Python: @model.likelihood terms
```

- `model.likelihood("gaussian")` sets the C++ likelihood mode (string → C++ path).
- `model.likelihood(fn)` adds a Python callable to `_extra_liks` (callable → Python path).
- `@model.likelihood` is syntactic sugar for `model.likelihood(fn)`.
- `model.prior(fn)` / `@model.prior` adds a callable to `_extra_priors`.
- Neither `model.likelihood("gaussian")` nor any `@model.likelihood` set → `RuntimeError`
  at sampling time.
- **`vals`**: physical parameter values as `**kwargs`. LogUniform params are `exp()`-converted
  from theta before being passed. Extra param names not in the function signature are
  absorbed by `**_`.

## Explicit sampling priors

Every sampled parameter requires an explicit prior. Calling
`model.param("x")` without one raises an error; use, for example,
`model.param("x", bayes.Uniform(-1, 1))`. External Python prior terms add to
this sampling prior and do not replace its explicit support.

## Supported inference patterns

The stable combinations are intentionally limited:

1. `flux="fit"`: linear best-fit fluxes with ordinary or Galactic priors.
2. `flux="marginalize"`: analytic Gaussian flux marginalization with priors
   that do not depend on flux.
3. `flux="sample"`: explicit `Fs_*` and `Fb_*` sampling.
4. Full marginalization: `flux="marginalize"`, `model.theta_star(...)`, and
   an optional Galactic prior that marginalizes hidden physical distances
   internally.

Flux-dependent Galactic magnitudes in mode 4 are evaluated jointly over the
conditional flux, thetaS, and Galactic distance draws. `thetaS` is defined
exclusively by `model.theta_star(...)`; it is never a sampling parameter.
Return `(log(thetaS), 0.0)` to fix it. lcbinint injects the resulting value
into Galactic context as `thS` automatically.

```python
@model.theta_star
def fixed_theta_star(_fluxes):
    return math.log(0.005), 0.0

model.galactic_prior(
    galactic_prior,
    context={"vEarth": (v_north, v_east)},
)
```

Do not register `thetaS` with `model.param()` and do not put `thS` in the
Galactic context. A flux-dependent relation uses the same entry point and
returns a positive log-space scatter, which lcbinint marginalizes internally.

For no-parallax Flow analyses, `DL` and `DS` may be registered as ordinary
sampled parameters; they are auxiliary to the magnification calculation and
are consumed by the external Galactic prior. If they are omitted, gapmoe may
marginalize `DL`, `DS`, and proper-motion direction internally. Circular and
Kepler LOM use the same `g1`, `g2`, `g3`, `lom_szs`, and `lom_ar` names in both
packages, including full flux/thetaS marginalization.

## Architecture

### C++ side — rename binding

```cpp
// bind_bayes.cpp
py::class_<Model>(bayes, "_Model")   // was "Model"
    ...
```

No other C++ changes.

### Python side — subclass in `python/lcbinint/model.py`

```python
from ._lcbinint import bayes as _b

class Model(_b._Model):
    def __init__(self, lc, event_or_data):
        super().__init__(lc, event_or_data)
        self._extra_liks   = []
        self._extra_priors = []
        self._param_is_log = {}   # name → bool
        self._lik_mode     = None  # "gaussian" | None

    def param(self, name, prior=None):
        if prior is None:
            prior = _FlatPrior()          # internal sentinel, log_prob = 0
        super().param(name, prior)
        self._param_is_log[name] = isinstance(prior, _b.LogUniform)
        return self

    def likelihood(self, arg="gaussian"):
        if isinstance(arg, str):
            super().likelihood(arg)       # C++ sets the mode
            self._lik_mode = arg
        elif callable(arg):
            self._extra_liks.append(arg)
            return arg                    # enables @model.likelihood
        else:
            raise TypeError(f"likelihood() expects str or callable, got {type(arg)}")

    def prior(self, fn):
        self._extra_priors.append(fn)
        return fn                         # enables @model.prior

    def has_py_extras(self):
        return bool(self._extra_liks or self._extra_priors)

    def _validate(self):
        if self._lik_mode is None and not self._extra_liks:
            raise RuntimeError(
                "bayes.Model: likelihood is not configured.\n"
                "  Use model.likelihood('gaussian')  — for gaussian chi2\n"
                "  or @model.likelihood               — for a custom function"
            )

    def _theta_to_vals(self, theta):
        return {
            name: math.exp(theta[i]) if self._param_is_log.get(name) else theta[i]
            for i, name in enumerate(self.param_names)
        }

    def log_prob_python(self, theta):
        """Full log_prob including Python extras (called by run_sampler dispatch)."""
        # C++ handles param priors + gaussian chi2
        lp = super().log_prob(theta)
        if not math.isfinite(lp):
            return lp
        vals = self._theta_to_vals(theta)
        for fn in self._extra_priors:
            lp += fn(**vals)
            if not math.isfinite(lp):
                return lp
        for fn in self._extra_liks:
            lp += fn(**vals)
            if not math.isfinite(lp):
                return lp
        return lp
```

`_FlatPrior` is a minimal internal class (not exposed to users) with `log_prob()→0`
and `bounds()→(-inf, inf)`. It is NOT a subclass of `bayes.Prior` (C++ class); it
is duck-typed for the Python-only path.

Actually, because `super().param(name, prior)` calls C++ which expects a `bayes.Prior`
shared_ptr, the flat-prior case must either:
- Be intercepted before calling `super().param()`, registering nothing in C++, or
- Have a real C++ `Flat` prior added (simplest).

**Decision**: add `bayes.Flat` as a proper C++ prior (log_prob = 0, bounds = ±1e15).
This avoids duck-typing issues and keeps the C++ model consistent.

### `run_sampler` dispatch

```python
has_reparam  = hasattr(model, 'has_reparams') and model.has_reparams()
has_py_extra = hasattr(model, 'has_py_extras') and model.has_py_extras()

if has_reparam or has_py_extra:
    # Python log_prob callback path (GIL acquire per walker).
    # The adapter is not a C++ bayes.Model, so pybind dispatch cannot
    # accidentally take the C++ fast path.
    step_model = model._sampling_adapter()
    state      = _py_init_state_extended(step_model, ...)
    _collect   = lambda: _py_collect_extended(step_model, state)
else:
    # C++ fast path
    state    = sampler.init_state(model, ...)
    _collect = lambda: sampler.collect(model, state)
```

### `__init__.py` patch

```python
# python/lcbinint/__init__.py
from .model import Model as _PyModel
from ._lcbinint import bayes as _bayes_cpp
_bayes_cpp.Model = _PyModel
```

After this, `bayes.Model(lc, event)` returns the Python-extended class.

## Files to change

| File | Change |
|---|---|
| `python/bind_bayes.cpp` | Rename `"Model"` → `"_Model"` |
| `src/lcbinint/bayes/prior.hpp/.cpp` | Add `Flat` prior (log_prob=0, bounds=±1e15) |
| `python/bind_bayes.cpp` | Expose `bayes.Flat` |
| `python/lcbinint/model.py` | New file: `Model` subclass |
| `python/lcbinint/__init__.py` | Import `Model`, patch `bayes.Model` |
| `python/lcbinint/sampler.py` | Update `run_sampler` dispatch |
