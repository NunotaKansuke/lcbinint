# Point / Hex Core Benchmark

Date: 2026-06-24

## Goal

The VBBL comparison should not mix unrelated overheads.  The target design is:

- point-source calculation: shared-quality path
- hexadecapole approximation: shared-quality path
- point/hex/IR switching criteria: shared-quality path
- inverse-ray / finite-source integration: the main implementation difference

Before comparing light-curve timings, we need to know whether lcbinint's point
and hex kernels are already comparable.

## lcbinint C++ Core Timing

Diagnostic executable:

```text
build/benchmark_point_hex
```

This measures:

- `PointSourceMagnifier::binary_mag0`
- `diagnostic_hexadecapole_binary`

It does not construct Python objects or `LensModel` objects.

Initial result before point-source optimization:

```text
case          point_ns   hex_ns    hex/point
planet_small 2169.8     28423.4   13.10
planet_ld    2176.6     28460.4   13.08
resonant     2283.0     29767.7   13.04
close        2258.8     29831.5   13.21
wide         2030.0     26448.4   13.03
```

Interpretation:

- lcbinint hex is almost exactly 13 point-source evaluations.
- The hex kernel itself is not showing an unexpected overhead.
- The earlier `~3.5 ms` hex timing was dominated by Python/API/dispatcher
  overhead, not by the hex formula.

## VBBL Python API Timing

Measured with the public VBBL Python methods:

- `BinaryMag0(s,q,y1,y2)` for point-source.
- `BinaryMag2(s,q,y1,y2,rho)` for finite-source uniform brightness.
- `BinaryMagDark(s,q,y1,y2,rho,a1)` for finite-source limb darkening.

```text
case          BinaryMag0_ns  BinaryMag2_ns  BinaryMagDark_ns
planet_small 1027.6         111829.8       -
planet_ld    1029.6         111765.0       28687.8
resonant     1031.1           1039.1       -
close        1032.7           1045.0       -
wide         1008.0          15792.3       14871.7
```

Interpretation:

- VBBL point-source is about `1.0 us`, while lcbinint point-source is about
  `2.0-2.3 us`.
- VBBL finite-source methods can be very fast when they take a fast path, so
  `BinaryMag2` is not a clean "hex only" benchmark.
- lcbinint pure hex at `26-30 us` is comparable to VBBL's LD finite-source
  times in some cases, but not to VBBL point/fast-path times.

## Current Conclusion

The big hex slowdown seen in the previous point benchmark is not a hex kernel
problem.  It is mostly from benchmarking this expensive path:

```text
LensModel construction
source-position calculation
caustic/topology and point/hex/IR dispatch
LightCurve object construction
Python wrapper and diagnostic vector filling
```

However, lcbinint's point-source kernel is still about 2x slower than VBBL's
`BinaryMag0`.  If point and hex should be fully comparable, the next target is
the point-source solver/root filtering path, not the hex formula.

## Point-Source Optimization

VBBL's installed source was inspected at:

```text
/home/nunota/.miniconda3/envs/myenv/lib/python3.10/site-packages/VBBinaryLensing/lib/VBBinaryLensingLibrary.cpp
```

Two useful differences were found:

1. VBBL uses Skowron-Gould roots with previous roots as starting points.
2. VBBL's `cmplx_roots_gen` solves the final deflated quadratic directly with
   `solve_quadratic_eq`, instead of running another Laguerre/Newton solve.

Changes made:

- `PointSourceMagnifier::binary_mag0` now has a stack-array fast path for
  magnification-only calls.
- The fast path computes coefficients, residuals, and Jacobians with explicit
  real/imaginary arithmetic instead of `std::complex`.
- `PointSourceMagnifier` keeps a small mutable root cache and passes previous
  roots into `cmplx_roots_gen(..., use_roots_as_starting_points=true)` for
  nearby source positions.
- The bundled Skowron-Gould translation now uses the VBBL-style direct
  quadratic solve for the final two roots.
- The fast point-source path no longer sorts all residuals.  It tracks the
  three worst residuals directly, matching VBBL's physical-image selection
  structure more closely.
- Residual ranking now uses squared residuals, avoiding a square root in the
  common point-source path.

Updated C++ core timing after the low-level point-source changes:

```text
case          point_warm_ns  point_cold_ns  hex_ns
planet_small 1067.5         1726.3         17072.3
planet_ld    1067.9         1727.0         17042.2
resonant     1064.0         1830.2         17277.3
close        1081.9         1845.2         17910.6
wide         1071.8         1602.1         16018.4
```

VBBL public Python `BinaryMag0` timing on the same points:

```text
case          VBBL BinaryMag0 ns
planet_small 1031.9
resonant     1029.7
close        1032.6
wide         1005.3
```

Interpretation:

- Cold lcbinint point-source calls are still about `1.8-2.1 us`, so VBBL is
  still faster for completely isolated point-source evaluations.
- Warm lcbinint point-source calls are now `1.06-1.08 us`, very close to VBBL's
  `~1.0 us`.
- Hex improved from `26-30 us` to `16-18 us` because the 13 nearby point
  evaluations can reuse roots.
- This should help hex and finite-source seed/probe paths more than isolated
  one-point API calls.

## Quartic Formula Note

Using a closed-form quartic solver when the deflated polynomial reaches degree
4 was considered.  VBBL does not do this; it still uses Skowron-Gould for the
degree-5 to degree-3 deflation and only solves the final degree-2 polynomial
directly.  A complex quartic formula would add branch-heavy sqrt/cbrt logic and
is unlikely to beat the tuned Skowron-Gould path robustly.  The low-risk
optimization is therefore the VBBL-style direct quadratic tail, which is now in
place.

## Follow-Up

1. Add a C++ light-curve benchmark that reuses `LensModel` and reports timing by
   selected method.
2. Add an internal timing breakdown for dispatch versus point/hex/IR kernels.
3. Compare lcbinint point-source solver against VBBL more directly; the 2x gap
   likely comes from the polynomial solver / root filtering path.
