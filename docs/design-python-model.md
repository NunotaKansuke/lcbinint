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

### Case 5 — reparameterization (unchanged, separate class)

```python
reparam = Reparameterization(lc, event)
reparam.param("tEq", bayes.LogUniform(...))
@reparam.transform
def to_phys(tEq, ...): ...
run_sampler(reparam, ...)
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

## Flat prior

`model.param(name)` with no prior argument registers a flat (improper) prior:
`log_prior = 0` everywhere, no bounds. The user is responsible for ensuring the
posterior is proper. `bayes.Uniform(lo, hi)` continues to require both bounds.

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
is_reparam   = isinstance(model, Reparameterization)
has_py_extra = hasattr(model, 'has_py_extras') and model.has_py_extras()

if is_reparam or has_py_extra:
    # Python log_prob callback path (GIL acquire per walker)
    model._validate()   # only for Model subclass
    state    = _py_init_state_extended(model, ...)
    _collect = lambda: _py_collect_extended(model, state)
    # sampler.step(model, state) → duck-typed overload calls model.log_prob
else:
    # C++ fast path
    state    = sampler.init_state(model, ...)
    _collect = lambda: sampler.collect(model, state)
```

For `has_py_extra`, `model.log_prob` passed to the C++ `step(std::function<>)` overload
is `model.log_prob_python` (Python method).

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
