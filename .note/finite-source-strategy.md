# Finite Source Strategy

## Scope

Finite-source support should become a publishable C++ implementation exposed
through the same Python `LensModel` API as the point-source path.

This is not a subprocess wrapper around the legacy executable, and it should
not preserve the old `FINITE=1..6` mode numbers as public user controls.

User-facing controls should stay small:

```python
Options(
    tolerance=1.0e-3,
    relative_tolerance=0.0,
)
```

Internal implementation may use more strategies, but users should not need to
choose between the point-source fast path, cartesian inverse-ray, or polar
inverse-ray.

An optional expert override exists for inverse-ray only:

```python
Options(inverse_ray_method=InverseRayMethod.POLAR)
Options(inverse_ray_method=InverseRayMethod.CARTESIAN)
```

The default remains `InverseRayMethod.AUTO`.

## Current State

The repo currently has a first `FiniteSourceMagnifier` boundary:

```text
LensModel
  -> PointSourceMagnifier::binary_mag0()
  -> FiniteSourceMagnifier::binary_mag()
```

Implemented pieces:

- Public finite-source modes include `AUTO`, `POINT_SOURCE`, and expert
  `LEGACY`.
- `Options` carries `tolerance`, `relative_tolerance`, `source_bins`,
  `caustic_bins`, `grid_ratio`, optional `inverse_ray_method`, and legacy
  expert knobs.
- The cheap finite-source fast path is a Bozza/microlux-style quadrupole safety
  test. If it passes, the source is treated as point-like for that evaluation.
- Binary cartesian inverse-ray and polar inverse-ray estimators exist as initial
  image-centered grids around point-source images.
- Inverse-ray samples are weighted by finite-source surface brightness when
  limb-darkening coefficients are present:
  `I(r)=1-c(1-mu)-d(1-sqrt(mu))`, normalized by
  `1-c/3-d/5`.
- Hexadecapole includes the legacy Gamma/Lambda correction derived from
  `limb_darkening_c` and `limb_darkening_d`.
- The old quadrupole-vs-hexadecapole difference diagnostic has been removed.
- The C++ quadrupole safety test now combines quadrupole correction, cusp
  correction, ghost-image test, and planetary-caustic test, following
  `../microlux/src/microlux/basic_function.py`.
- Inverse-ray now has an internal coarse/fine refinement diagnostic:
  evaluate at `source_bins`, then at doubled bins, and compare the difference
  against `tolerance` or `relative_tolerance * |magnification|`.
- The result has internal diagnostics: `error_estimate`, `refinement_level`,
  and `converged`.
- Non-converged finite-source evaluations now propagate upward as
  `LCBI_NUMERICAL_ERROR` through the C ABI. Python `LensModel.magnification()`
  raises `RuntimeError("numerical error")` through the existing status wrapper.
- Initial benign finite-source cases are compared against
  `VBBinaryLensing().BinaryMag2(...)`.

## Legacy Compatibility Mode

`FiniteSourceMode.LEGACY` is an expert compatibility layer for preserving the
old finite-source switch behavior while the modern API keeps `tol/reltol` as
the normal user-facing control.

Exposed knobs:

- `legacy_finite_mode`: old `FINITE/smode`-style selector.
- `legacy_kinji`: point-source cutoff in units of source radius.
- `legacy_hex`: hexadecapole cutoff in units of source radius.
- `caustic_bins`: resolution used for binary-caustic distance sampling.
- `source_bins` and `grid_ratio`: inverse-ray resolution controls.

The legacy branch now measures source-to-caustic distance by sampling the binary
critical curve, mapping it into the source plane, tracking the four caustic
branches, and computing the nearest polyline distance. That distance drives the
old-style KINJI/HEX decisions:

```text
distance > legacy_kinji * rho -> point source
distance > legacy_hex * rho   -> hexadecapole
otherwise                     -> inverse-ray fallback
```

Current limitations:

- Old `smode=1` finite-source contour/integral code is not ported.
- Old `smode=6` memory/cache behavior is not ported.
- Hexadecapole includes Gamma/Lambda limb-darkening terms for the binary path.
- The caustic distance is numerical polyline distance, not a byte-for-byte port
  of the old routine.

Important limitation:

- The current inverse-ray estimator is still a rough first implementation. It
  samples regions around point-source images at the source center. Near caustics
  this can miss finite-source image area, so it is not yet a publishable
  finite-source solver.

## microJAX Reference Point

microJAX is useful here as a reference for the decision and error-control
philosophy, not as something to copy wholesale into this C++ implementation.

Relevant ideas to borrow:

- Use multipole safety tests to decide whether full inverse-ray is needed.
- Treat `tolerance` and `relative_tolerance` as accuracy controls.
- Use local tests and diagnostics rather than exposing low-level model switches.
- Keep full inverse-ray as the fallback when the cheap approximation is not
  trustworthy.

Do not conflate this with porting microJAX's JAX/GPU implementation details.
The lcbinint core remains C++.

## Error Evaluation

Current tolerance semantics:

- `tolerance` is the absolute accuracy control.
- `relative_tolerance` is the relative accuracy control for full inverse-ray
  refinement.
- The quadrupole safety fast path uses `tolerance`, following microlux/VBM.
  `relative_tolerance` is not used in that fast-path test.
- For inverse-ray refinement, either absolute or relative convergence is enough:

```text
accept if:
  error <= tolerance
  or
  relative_tolerance > 0 and error <= relative_tolerance * abs(magnification)
```

This matches the VBBinaryLensing stopping logic in structure:

```text
continue while:
  error > absolute_tolerance
  and
  error > relative_tolerance * magnification
```

microJAX's inverse-ray path is different from both VBM and microlux. It does
not expose a VBM-like `tol`/`retol` stopping condition for inverse-ray
integration. Instead, it controls accuracy mostly through fixed resolution and
budget parameters such as `r_resolution`, `th_resolution`, `Nlimb`, `bins_r`,
`bins_th`, and `MAX_FULL_CALLS`. Its fast/full switch uses multipole/caustic
heuristics with fixed constants (`c_m`, `gamma`, `c_f`, `rho_min`) rather than
user-supplied `tol`/`retol`.

This means microJAX is a useful reference for the inverse-ray geometry and the
fast/full decision structure, but not for the public `tol`/`retol` semantics.
For `tol`/`retol`, VBBinaryLensing remains the closer reference.

A practical first error estimate for inverse-ray is Richardson-like but modest:

```text
coarse = inverse_ray(bins)
fine   = inverse_ray(2 * bins)
delta  = abs(fine - coarse)
```

When at least two consecutive deltas are available, the code estimates the
remaining tail from the observed convergence ratio:

```text
ratio = delta_n / delta_(n-1)
error = max(delta_n, delta_n * ratio / (1 - ratio))  if ratio is decreasing
error = delta_n + delta_(n-1)                        otherwise
```

This is still not a rigorous mathematical error bound. It is a more conservative
numerical convergence diagnostic, and should be treated as such in docs and
tests.

If this does not converge, do not silently downgrade to a cheaper method. The
next public step should be to propagate a numerical error/status or expose a
diagnostic object from the Python API. Internally, the best estimate can be kept
for debugging, but it should not be presented as a guaranteed-accurate result.

## Strategy Layer

The strategy layer should eventually be:

```text
1. Compute point-source magnification and images.
2. Run a Bozza/microlux quadrupole safety test.
3. If accepted, return the point-source magnification.
4. Otherwise choose inverse-ray variant internally:
   - polar can be useful for high-magnification/image-centered cases
   - cartesian can be simpler and more predictable in other cases
   - expert users can force either inverse-ray variant
5. Refine inverse-ray until tolerance/reltol is met or report non-convergence.
```

The current inverse-ray variant selection is still a placeholder. It uses simple
heuristics such as point-source magnification and source distance to choose
cartesian versus polar. The point-source fast-path acceptance no longer uses
that heuristic, but the full strategy is still not publishable.

## Legacy-Speed Notes

The legacy mode should remain a fixed-resolution expert path. It now reuses a
`LensModel` and `FiniteSourceMagnifier` across array evaluations, caches the
legacy caustic curve per `(sep, q, caustic_bins)`, and performs a fast sampled
caustic-distance rejection before the full segment-distance pass. It also avoids
the `tol`/`reltol` refinement loop; `legacy_finite_mode=3..6` should use the
requested `source_bins` once, matching the old `NBIN` semantics.

Varied-time Release benchmarks against the old executable show:

```text
NBIN=80, 400 points:
  low:   old smode=4 0.056 ms/pt, new legacy cart 0.148 ms/pt
  close: old smode=4 0.012 ms/pt, new legacy cart 0.073 ms/pt
  wide:  old smode=4 0.577 ms/pt, new legacy cart 0.528 ms/pt
  wide:  old smode=5 0.282 ms/pt, new legacy polar 0.564 ms/pt

NBIN=20, 400 points:
  wide:  old smode=4 0.089 ms/pt, new legacy cart 0.116 ms/pt
  wide:  old smode=5 0.075 ms/pt, new legacy polar 0.139 ms/pt
```

The cartesian path is now close for the hard wide case, but low/close still pay
more point-source and caustic-rejection overhead than the old C executable.
The legacy polar path now follows the old `imagearea5` boundary-scanning shape
instead of sampling the full polar grid. The inverse-ray loops also use a local
binary lens mapper so each sample does not rebuild the lens geometry. Polar is
still slower than old `smode=5` at high `NBIN`; matching it more closely would
likely require the old `smode=6` dense polar memory table, but the sparse
on-demand cache was tested and rejected because hash overhead made it slower.

## What Not To Do

- Do not make legacy finite-source mode numbers the default public API.
- Do not make `max_evaluations` a public API knob as a way to mask numerical
  failure.
- Do not silently switch to a cheaper/less accurate method when refinement does
  not converge.
- Do not port Numerical Recipes finite-source integration routines.
- Do not treat current VBM finite-source tests as enough validation; they only
  cover benign cases.

## Next Work

1. Keep the current finite-source branch compile-clean and tested.
2. Add direct unit coverage for the coarse/fine tolerance helper and result
   diagnostics.
3. Compare the C++ quadrupole safety classification against VBM/microlux over a
   grid.
4. Expose finite-source diagnostics in Python without complicating the default
   scalar `magnification()` path.
5. Only after that, improve the inverse-ray region construction itself and add
   harder VBM comparisons near caustics.
