# Session 2026-06-30: Flux output, LimbDarkening, autocorrelation time

## Summary

Completed the remaining Python API gaps identified in the previous session:
`lc.LimbDarkening`, `Result.fluxes`, `Chain.flat_fluxes`, `chain.samples`, `chain.summary()`,
`chain.tau()`, and `chain.ess()`.

## Changes

### `lc.LimbDarkening` — exposed as proper Python class

Previously `PyLimbDarkening` existed only as an internal C++ struct (not importable from Python).
Now exposed in the `lc` submodule with factory class methods:

```python
ld = lci.lc.LimbDarkening.linear(0.5)          # linear: I(μ) = 1 - u*(1-μ)
ld = lci.lc.LimbDarkening.square_root(0.3, 0.2) # square-root: I(μ) = 1 - c*(1-μ) - d*(1-√μ)
ld = lci.lc.LimbDarkening(c=0.4, d=0.1)         # explicit coefficients
lc_obj = lci.lc.LightCurve(limb_darkening=ld)
```

Removed a duplicate registration that existed from a previous session (was causing
`"cannot initialize type 'LimbDarkening': an object with that name is already defined"`).

### `Model::fluxes(theta)` — new C++ method

Returns per-dataset analytical flux solutions {Fs, Fb} via Cramér's rule (same
math as `compute_chi2` but exposing the intermediate results). Also exposed to
Python as `model.fluxes(theta) → {name: {'Fs': ..., 'Fb': ...}}`.

### `Result.fluxes` — flux parameters in optimizer output

`optimize::Result` now carries a `std::vector<DatasetFlux>` (name, Fs, Fb).
Both `DifferentialEvolution` and `LevenbergMarquardt` populate this after their
optimization completes. Accessible as `result.fluxes[i].Fs`, `.Fb`, `.name`.

`optimize::DatasetFlux` is also exposed as `optimize.DatasetFlux` in Python.

### `Chain.flat_fluxes` — sampler stores Fs/Fb at every step

`Chain` now stores `flat_fluxes_`: shape `(nsteps*nwalkers, n_datasets*2)`.
Layout per row: `[Fs_0, Fb_0, Fs_1, Fb_1, ...]` (n_datasets * 2 values).

`EnsembleSampler` maintains a per-walker flux cache (`walker_fluxes`) updated
only on acceptance — rejected walkers reuse the previous value, so there are no
extra magnification evaluations for rejected proposals.

New Chain API:
- `chain.flat_fluxes` — shape `(nsteps*nwalkers, n_datasets*2)` numpy array
- `chain.dataset_names` — list of dataset names
- `chain.transforms` — `['log', 'identity', ...]` per parameter

### `chain.samples` — physical-space view

`chain.flat_samples` stays in transformed (optimizer) space. The new property
`chain.samples` applies the inverse transform (`exp()` for log-transformed params):

```python
chain.samples  # shape (nsteps*nwalkers, ndim), physical space
```

### `chain.summary()` — summary statistics

```python
s = chain.summary()
# s['tE'] = {'median': 29.8, 'lo': 28.1, 'hi': 31.5, 'std': 1.7}
```
lo/hi = 16th/84th percentile; computed in physical space.

### `chain.tau(c=5.0)` / `chain.ess()` — autocorrelation diagnostics

FFT-based integrated autocorrelation time using the Sokal auto-window method,
averaged over walkers. Implemented in C++ using GSL real/halfcomplex FFT.

Algorithm:
1. For each walker: subtract mean, zero-pad to next power of 2 ≥ 2N
2. Forward real FFT → compute power spectrum in-place (halfcomplex format)
3. Inverse halfcomplex FFT → unnormalized linear autocorrelation
4. Average ACF over walkers, normalize by variance (lag-0 value)
5. Sokal window: find first M where M ≥ c × τ(M); return τ(M)

```python
tau = chain.tau()        # array of τ per parameter; NaN if chain too short
ess = chain.ess()        # nsteps * nwalkers / tau
```

Validated: matches NumPy FFT reference implementation to < 0.01% for all parameters.

## API additions (complete)

```
lc.LimbDarkening(c, d)
lc.LimbDarkening.linear(u)
lc.LimbDarkening.square_root(c, d)

model.fluxes(theta) → dict

optimize.DatasetFlux  (.name, .Fs, .Fb)
optimize.Result.fluxes  → list[DatasetFlux]

chain.flat_fluxes    → np.ndarray (nsteps*nwalkers, n_datasets*2)
chain.dataset_names  → list[str]
chain.transforms     → list[str]  ("log" | "identity")
chain.samples        → np.ndarray (physical space)
chain.summary()      → dict {name: {median, lo, hi, std}}
chain.tau(c=5.0)     → list[float]  (integrated autocorr time)
chain.ess()          → list[float]  (effective sample size)
```

## Model accessor added

`Model::event() const → const obs::Event&` — needed by DE/LM to look up dataset
names when building `Result.fluxes`.
