# 7. Putting it together

You now have one visual example for every major physical extension. This final
page is the handoff from exploration to a repeatable model setup.

## Build the model in layers

Keep the two kinds of choices separate:

```python
model = lcbinint.Model(
    lens="binary",                 # change to "triple" with q2 > 0
    source="single",               # change to "binary" for two sources
    parallax=True,                  # also needs sky, t_ref, and piEN/piEE
    terrestrial=False,              # needs a ground Site when enabled
    orbital_motion="static",       # or "circular" / "kepler"
    xallarap="none",               # select one parameterization when needed
    sky=lcbinint.obs.SkyCoord(270.0, -30.0),
    t_ref=2459000.0,
)
options = lcbinint.Options(
    coordinates="vbm",
    nbin="auto",
    inverse_ray_grid="auto",
)
curve = lcbinint.LightCurve(model=model, options=options)
```

`Model` declares the event physics. `Options` controls numerical evaluation.
Keeping them separate means the same physical model can safely be reused for
ground and space observatories while each curve receives its own `Site`.

## Choose the smallest model that explains the event

| If the event needs… | Add… | Tutorial |
| --- | --- | --- |
| One binary lens and one source | `LightCurve()` | [1](01-binary-lens.md) |
| Earth/satellite observer displacement | `parallax=True`, `sky`, `t_ref`, and a `Site` when applicable | [2](02-parallax.md) |
| Changing lens separation/orientation | `orbital_motion="circular"` or `"kepler"` | [3](03-orbital-motion.md) |
| Source orbital displacement | an `xallarap` mode and its matching parameters | [4](04-xallarap.md) |
| Two luminous sources | `source="binary"` plus `q_source` and secondary-source fields | [5](05-binary-source.md) |
| A second lens companion | `lens="triple"` plus positive `q2`, `sep2`, and `ang` | [6](06-triple-lens.md) |

Do not enable a physical effect merely because its parameters are non-zero:
annual parallax, terrestrial parallax, orbital motion, and xallarap each need
their corresponding `Model` selection.

## Validate before fitting at scale

```python
info = curve.info(times, params)
if not info.all_converged:
    raise RuntimeError(f"finite-source budget missed at {info.unconverged_indices}")

print(info.finite_source_method_names)
print(info.finite_source_error_estimates)
```

For finite sources, `nbin="auto"` is the recommended starting point. If you
set `tol` or `reltol`, `finite_source_converged` is the statement that each
epoch met the requested numerical budget. Use
[Numerical methods](../numerical-methods.md) for the exact tolerance semantics.

## Where to go next

- Need a parameter name or return type? Read the [Python API reference](../python-api.md).
- Need a low-level or non-Python integration? Read the [C/C++ API reference](../c-api.md).
- Need plots of image positions and critical curves? Run the
  [image-plane example](../../example/image-plane/).
- Need performance or accuracy comparisons with VBMicrolensing? Run the
  [comparison examples](../../example/compare-vbm/).
- Need to change the core? Follow the [developer guide](../development.md) and
  preserve the numerical validation evidence.

You can now return to the [documentation home](../README.md) whenever you need
to branch into a reference or validation topic.

---

**Course navigation:** [← Previous: Triple lens](06-triple-lens.md) · [Tutorial home](../README.md)
