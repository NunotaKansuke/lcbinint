# lcbinint Implementation Status

## Completed

### `lcbinint.lc`

- `lc.Options` — `lcbi_options` struct exposed as a Python class.
  - Constructor: `Options(param_type='vbm', source_bins=50, ...)` with all fields as kwargs.
  - `param_type` property (string): `'vbm'`, `'lcbinint'`, `'center_of_mass'`, `'vbm_center_of_mass'`.
    Internally maps to `vbm_compatible` + `center_of_mass` fields on `lcbi_options`.
  - All fields exposed as `def_readwrite`.
  - Default: `param_type='vbm'`.

- `lc.Parameters` — `lcbi_params` struct exposed as a Python class.
  - Constructor kwargs use **VBM-compatible names**: `t0`, `tE`, `u0`, `alpha`, `s`, `q`, `rho`, etc.
  - Properties also use VBM names (`u0` → `umin`, `alpha` → `theta`, `s` → `sep`).

- `lc.LimbDarkening` — `PyLimbDarkening{c, d}` struct. Class methods: `linear(c)`, `square_root(c, d)`.

- `lc.LightCurve` — standalone magnification evaluator.
  - Constructor 1: `LightCurve(options=Options(), limb_darkening=LimbDarkening())`.
  - Constructor 2: `LightCurve(**kwargs)` e.g. `LightCurve(source_bins=12, param_type='vbm')`.
  - `__call__` / `magnification`: 3 overloads: `(times, lcbi_params)`, `(times, dict)`, `(times, **kwargs)`.
  - GIL released during C++ computation; capsule-based numpy ownership transfer.
  - Default `param_type='vbm'` on all construction paths.

### `lcbinint.obs`

- `obs.LightCurveData` — C++ storage for one observed light curve.
  - Constructor: `LightCurveData(time, flux, flux_err, name='', band='', observatory='')`.
  - Properties `.time`, `.flux`, `.flux_err`, `.weight` — zero-copy numpy views (C++ owns memory).
  - `__len__`, `.size`, `.name`, `.band`, `.observatory`.

- `obs.Event` — groups multiple `LightCurveData` objects.
  - Constructor: `Event(name='', ra=0.0, dec=0.0, t_ref=0.0)`.
  - `.add(data)`, `__len__`, `__getitem__`, `__iter__`.

### `lcbinint.bayes` (partial)

- `bayes.Uniform(lo, hi)`, `bayes.Normal(mu, sigma)`, `bayes.LogUniform(lo, hi)`.
  - All have `.log_prob(x)`, `.bounds()`, `__repr__`.

- `bayes.Model(options=opts, event=event)` — probabilistic model.
  - Alternative constructor: `Model(options=opts, data=data)`.
  - `.param(name, prior)` — register a free parameter.
  - `.flux(mode='linear_blend')`, `.likelihood(mode='gaussian')`.
  - `.log_prior(theta)` — **implemented**: sums prior log-probs over free parameters.
  - `.log_likelihood(theta)` — **stub** (throws not implemented).
  - `.chi2(theta)` — **stub** (throws not implemented).
  - `.log_prob(theta)` — calls `log_prior`; short-circuits if `-inf`.

## Architecture Decisions (Settled)

- `bayes.Model` takes `lcbi_options` directly (not `lc.LightCurve`).
  The Python-layer `lc.LightCurve` is only for standalone magnification use.
  In the hot path, `Model` calls `lcbi_magnification_array` in C++ directly.

- `param_type` string API everywhere (not `vbm_compatible` bool).

- VBM-compatible parameter names for user-facing API: `t0`, `tE`, `u0`, `alpha`, `s`, `q`, `rho`.
  Internal `lcbi_params` fields (`umin`, `theta`, `sep`) are always translated transparently.

## Next: `bayes.Model` Core

### 1. `theta_to_params()` — parameter name → `lcbi_params` mapping

User-facing names follow VBM convention:

| `model.param()` name | `lcbi_params` field |
|----------------------|---------------------|
| `t0`                 | `t0`                |
| `tE`                 | `tE`                |
| `u0`                 | `umin`              |
| `alpha`              | `theta`             |
| `s`                  | `sep`               |
| `q`                  | `q`                 |
| `rho`                | `rho`               |
| `piEN`               | `piEN`              |
| `piEE`               | `piEE`              |
| `q2`                 | `q2`                |
| `sep2`               | `sep2`              |
| `ang`                | `ang`               |

### 2. Linear flux solver — `F = Fs * A(t) + Fb`

Solve analytically per `LightCurveData` dataset. Define:

```
w_i   = 1 / flux_err_i^2
S_w   = sum(w_i)          — precomputed at construction
S_wf  = sum(w_i * f_i)   — precomputed
S_wf2 = sum(w_i * f_i^2) — precomputed
```

Per evaluation (hot path):

```
S_wA  = sum(w_i * A_i)
S_wA2 = sum(w_i * A_i^2)
S_wAf = sum(w_i * A_i * f_i)

D  = S_wA2 * S_w - S_wA^2
Fs = (S_wAf * S_w  - S_wA * S_wf) / D
Fb = (S_wA2 * S_wf - S_wA * S_wAf) / D

chi2 = S_wf2 - Fs * S_wAf - Fb * S_wf
```

Performance requirements:
- Precomputed sums (`S_w`, `S_wf`, `S_wf2`) stored on `Model` at construction time.
- Hot-path loop computes only the 3 A-dependent sums in a single pass (no branching).
- No heap allocation per evaluation: workspace vectors pre-allocated.
- GIL released for the entire C++ evaluation block.

### 3. `chi2(theta)` and `log_likelihood(theta)`

```
chi2_total = sum over datasets of chi2_k
log_likelihood = -0.5 * chi2_total   (Gaussian, no normalization term initially)
```

### 4. Evaluation workspace

Pre-allocate per-dataset:

```cpp
struct DatasetCache {
    double S_w;
    double S_wf;
    double S_wf2;
    std::vector<double> mag_buf;  // reused per evaluation
};
```

## Remaining Modules

- `optimize.DifferentialEvolution` / `LevenbergMarquardt` / `Result`
- `sample.EnsembleSampler` / `Chain`
