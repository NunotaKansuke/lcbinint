# moasarc extraction boundary

The public `lcbinint` API remains unchanged during inference extraction.
Existing imports such as `lcbinint.bayes`, `lcbinint.optimize`,
`lcbinint.sample`, and `lcbinint.run_sampler` are compatibility entry points.

## Component ownership

`lcbinint_magnification` owns numerical lens evaluation, trajectories, root
solving, finite-source integration, and the existing C ABI. It does not depend
on LightCurve, observations, Bayesian models, optimizers, or samplers.

`lcbinint_lightcurve` adapts the magnification C ABI to LightCurve and
observation containers.

`lcbinint_inference` temporarily contains the legacy inference implementation.
It is isolated so its implementation can be replaced by `moasarc` without
changing the compatibility API.

`moasarc` owns backend-neutral optimization and sampling. Its `Problem`
contract is deliberately compatible with the current Python model protocol:
`n_params()`, `optimizer_bounds`, `log_prob()`, and optional
`log_prob_batch()`.

## Dependency rule

The permanent dependency direction is:

```text
moasarc <- lcbinint inference adapter -> lcbinint_magnification
```

`moasarc` must not include or import `lcbinint`. Optimized native integration
belongs in the adapter. Python callbacks remain the generic fallback for any
other numerical backend.

## Compatibility sequence

1. Preserve the current Python API and serialized chain layout.
2. Match sampler RNG ordering and chain metadata before switching defaults.
3. Move optimizer and sampler engines behind the compatibility bindings.
4. Move the Bayesian model after replacing its direct LightCurve dependency
   with a magnification-backend interface.
5. Remove the legacy inference target only after cross-backend regression and
   performance tests pass.

