# Triple-Lens Implementation Handoff

Date: 2026-06-26

This file is private working context for the next Codex session.  `.note/` is
ignored and must not be committed or published.

## Current State

The public API is now centered on a reusable callable:

```python
lc = lcbinint.LightCurve(
    lens="binary_lens",
    options=lcbinint.Options(coordinates="vbm", nbin=50),
    limb_darkening=lcbinint.LimbDarkening.linear(0.5),
)
mag = lc(times, {
    "t0": 0.0,
    "tE": 1.0,
    "u0": -0.01,
    "alpha": 0.5,
    "s": 0.95,
    "q": 1.0e-2,
    "rho": 5.0e-3,
})
```

`LightCurve(...)` is the main user-facing construction path.  Low-level helpers
such as `light_curve(...)`, `magnification(...)`, and `binary_mag0(...)` still
exist, but normal repeated modeling should go through the callable object.

Binary lens is the only robust implemented path.  Triple parameters already
exist in the C/Python parameter structs:

- `q2`
- `sep2`
- `ang`

and `LensParameters::is_triple()` returns `q2 > 0.0`, but the current C++ model
does not actually evaluate triple magnification.  `LensModel::magnification`
enters the binary-only block only when `!params_.is_triple()`.  If triple is set,
the result remains unsupported/numerical rather than meaningful.

## Important Existing Files

- Public C API: `include/lcbinint/lcbinint.h`
- C API implementation: `src/lcbinint/lcbinint.cpp`
- Python binding and fast `LightCurve` callable: `python/lcbinint_pybind.cpp`
- Parameter conversion: `src/lcbinint/model/lens_parameters.{hpp,cpp}`
- Current binary point solver: `src/lcbinint/magnification/point_source_magnifier.{hpp,cpp}`
- Current binary finite-source inverse-ray code:
  `src/lcbinint/magnification/finite_source_magnifier.{hpp,cpp}`
- Current unit smoke test: `tests/unit/test_core.cpp`
- Old reference implementation:
  `/moao38_7/nunota/binfit/integral/lcbinint.c`

Old triple functions and geometry in the reference file:

- option parsing: around lines 333-338 (`q2`, `sep2`, `ang`)
- triple geometry in light-curve evaluation: around lines 1110-1322
- `amp_finite3`: starts around line 1525
- `lenseq2`: starts around line 1985
- `amp_point3` / `amp_point3q`: prototypes around lines 1057-1058, definitions
  later in the file
- `caustic3`: referenced around line 2935

Do not copy Numerical Recipes routines.  Keep the public-ready direction:
Skowron-Gould/GSL/generic C++ replacements only.

## Design Goal

The first triple implementation should be conservative:

1. Implement triple **point-source** magnification first.
2. Wire it through `LightCurve(lens="triple_lens")` and parameter dictionaries.
3. Add tests against a trusted reference.
4. Only after point-source is solid, decide how much of triple finite source is
   worth porting.

Do not start by porting all of `amp_finite3`.  That will mix root solving,
caustic handling, finite-source integration, and API design at once.

## API Direction For Triple

Recommended call shape:

```python
lc = lcbinint.LightCurve(
    lens="triple_lens",
    options=lcbinint.Options(coordinates="vbm", nbin=50),
)

mag = lc(times, {
    "t0": 0.0,
    "tE": 1.0,
    "u0": 0.01,
    "alpha": 0.5,
    "s": 1.0,
    "q": 1.0e-3,
    "q2": 1.0e-4,
    "sep2": 0.5,
    "ang": 1.2,
    "rho": 0.0,
})
```

Keep `binary_lens` and `triple_lens` explicit.  Avoid names like
`model="binary"` because that can be confused with binary source models later.

For now, reject or clearly mark unsupported combinations:

- `triple_lens` with `rho > 0` can initially throw `unsupported`.
- triple + parallax/LOM can be added after static point-source works.
- If dynamic effects are easy because they only alter the source trajectory and
  lens separation/angle, add them later, not in the first patch.

## Coordinate/Geometry Notes

The binary public default is VBM-compatible coordinates:

- `Options(coordinates="vbm")`
- internally `vbm_compatible = 1`
- for binary, the Python direct path uses `effective_q = 1/q`

Triple does not yet have a VBM reference path in this repo.  Do not blindly
apply the binary `effective_q = 1/q` rule to triple until the mass convention is
derived and tested.

From the old `lcbinint.c` triple block:

```c
eps2 = q  / (1.0 + q + q2);
eps3 = q2 / (1.0 + q + q2);
eps1 = 1.0 - eps2 - eps3;
eps4 = eps2 + eps3;

xx1 = -eps4 * sep;
yy1 = 0;
xx4 =  eps1 * sep;
yy4 = 0;

xx2 = xx4 + eps3 / eps4 * sep2 * cos(ang);
yy2 =       eps3 / eps4 * sep2 * sin(ang);
xx3 = xx4 - eps2 / eps4 * sep2 * cos(ang);
yy3 =     - eps2 / eps4 * sep2 * sin(ang);
```

Interpretation to confirm:

- lens 1 is the primary
- lenses 2 and 3 form the secondary subsystem
- `sep` is between lens 1 and the center of mass of lenses 2+3
- `sep2` is the separation between lens 2 and lens 3
- `ang` is the orientation of the 2-3 axis relative to the 1-(2+3) axis

For triple, the old code forces center-of-mass coordinates (`if eps3 > 0, CM=1`).
The new API should either document that triple uses center-of-mass internally or
provide a clean transform.  Do not silently mix the old wide-binary caustic
offset with triple.

## Implementation Plan

### Phase 1: Internal triple geometry class

Add a small internal representation, probably under `src/lcbinint/model/` or
inside a new magnifier file:

```cpp
struct TripleLensGeometry {
    std::array<SourcePosition, 3> lens_positions;
    std::array<double, 3> masses;
};
```

Build it from `q`, `q2`, `sep`, `sep2`, `ang` using the mass fractions above.
Keep this independent from Python binding code.

Add a low-level lens equation helper:

```cpp
SourcePosition triple_lens_equation(
    const TripleLensGeometry& geometry,
    SourcePosition image);
```

The complex form is:

```text
y = z - sum_i eps_i / conj(z - z_i)
```

Use the same sign conventions as current binary helpers and test against the old
code before exposing it.

### Phase 2: Triple point-source roots

For an N-lens point-source equation, the polynomial degree is `N^2 + 1`, so
triple lens gives degree 10.  The existing `PolynomialRootSolver` supports
generic complex polynomials through the SG/GSL-backed path, so use that first.

Needed work:

- derive or port the triple polynomial coefficient construction from the old
  `amp_point3`/`amp_point3q` logic, but avoid Numerical Recipes code
- solve degree-10 roots
- filter physical roots by mapping residual through `triple_lens_equation`
- compute Jacobian determinant:

```text
det J = 1 - |sum_i eps_i / (z - z_i)^2|^2
```

- point magnification is `sum 1 / abs(det J)` over physical images

Expected image count for triple point lens is up to 10.  The physical root
filtering must be robust near caustics.

Add API methods to `PointSourceMagnifier`:

```cpp
PointSourceResult triple_mag0(const TripleLensGeometry&, SourcePosition source) const;
std::vector<TripleImage> triple_images(...) const;
```

Do not overload binary methods with many scalar arguments if it makes call sites
ambiguous.  A geometry object is clearer.

### Phase 3: Wire through C++ model and Python callable

In `LensModel::magnification`, add a triple point-source branch:

- static only at first
- `rho == 0.0`
- no parallax/LOM at first unless trivial and tested
- return `unsupported` for finite-source triple

In `python/lcbinint_pybind.cpp`:

- extend `PyBinaryParams` or rename internally later; it already reads only
  binary fields, so add `q2`, `sep2`, `ang` dictionary parsing
- make `LightCurve(lens="triple_lens")` set/require triple mode
- keep the fast direct binary path untouched
- triple can initially use a direct static triple path or `LensModel`, whichever
  is simpler, but avoid per-point object construction inside loops

Suggested first user-visible behavior:

```python
lc = lcbinint.LightCurve(lens="triple_lens")
lc(times, {"s": 1.0, "q": 1e-3, "q2": 1e-4, "sep2": 0.5, "ang": 1.2, "rho": 0.0})
```

If `q2 <= 0`, raise `ValueError` for `triple_lens` rather than silently becoming
binary.

### Phase 4: Tests

Minimum tests before claiming triple point-source support:

1. Unit test for `TripleLensGeometry` mass fractions and positions.
2. Unit test for `triple_lens_equation` on one or two hand-calculated points.
3. Regression test against old `lcbinint.c` or another trusted triple reference
   for static point source.
4. Degenerate consistency:
   - `q2 -> 0` should approach binary behavior when geometry is mapped correctly.
   - This may require careful coordinates; do not use it as the only reference.
5. Python API smoke test:
   - `LightCurve(lens="triple_lens")` returns finite values for `rho=0`.
   - `rho>0` raises `unsupported` until finite-source triple is implemented.

If using the old executable as reference, create a diagnostic script under
`tests/diagnostics/`, not a fragile public unit test that depends on a local
absolute path.

## Validation References

Primary local reference:

- `/moao38_7/nunota/binfit/integral/lcbinint.c`

Potential external reference:

- VBM may have triple-lens support in newer versions.  If used, compare against
  `VBMicrolensing` with explicit parameter convention checks.  Do not assume the
  binary parameter order/convention carries over directly.

## Do Not Do First

- Do not port finite-source triple immediately.
- Do not add fallback calls to the old executable.
- Do not expose `mode=...` style controls for triple.
- Do not make triple support depend on `.note` files or local absolute paths.
- Do not put `.note` back under git tracking.

## Open Questions For The Next Session

1. What exact triple coordinate convention should the public API promise?
   The safest first release may state that triple uses center-of-mass lens
   coordinates internally and accepts `(s, q, q2, sep2, ang)` in the old
   `lcbinint` triple convention.
2. Should `coordinates="vbm"` be accepted for triple immediately, or should it
   raise until a VBM-compatible triple convention is verified?
3. Is degree-10 root solving with the current generic solver fast and robust
   enough, or does it need a triple-specific optimized path after correctness?
4. Which reference should be authoritative for point-source triple:
   old `lcbinint.c`, VBM, or both?

Recommended answer for the first implementation: old `lcbinint.c` for initial
regression, then compare to VBM if available.

## Suggested First Commit Scope

Keep the first triple commit small:

- add `TripleLensGeometry`
- add `triple_lens_equation`
- add degree-10 triple point-source solver
- expose through low-level C++ tests and one Python `LightCurve(lens="triple_lens")`
  smoke path for `rho=0`

Finite-source triple should be a later commit after the point-source root path is
trusted.
