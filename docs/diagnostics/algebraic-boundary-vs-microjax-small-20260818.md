# Small value-only comparison: algebraic C++ boundary vs microJAX CPU

This is a deliberately small temperature check, not a production benchmark.
It compares the new native algebraic boundary kernel on the local
`algebraic-boundary-cpp` branch with microJAX `develop` CPU.

## Protocol

- Five binary-lens geometries: smooth planetary, central high magnification,
  tiny source, wide binary, and equal-mass resonant.
- Two profiles: uniform and linear limb darkening (`u1 = c = 0.5`).
- One scalar source position per case; no light-curve/model construction,
  dispatcher, FFI, or automatic differentiation is timed.
- C++: `build/benchmark_algebraic_boundary`, warmed native calls, five timing
  repeats.  Its algebraic kernel includes its certified image seeds and radial
  support search.
- microJAX: `microjax.inverse_ray.cpu.mag_uniform_cpu` or
  `mag_limb_dark_cpu`, JIT compilation excluded, three steady-state CPU
  repeats, one CPU thread.
- Direct VBMicrolensing calls are used only as an accuracy oracle and are not
  included in either timing.
- microJAX revision: `326cb0dd14617e5716a9d652b510709902d27b5b`.

The reproducible small harness is
[`benchmark_algebraic_vs_microjax_small.py`](../../tests/diagnostics/recal2026/benchmark_algebraic_vs_microjax_small.py).
Raw JSON is kept outside the repository under `/tmp`.

## Results

At the matched (10^{-3}) relative-error setting:

| profile | cases | C++ median | microJAX median | median microJAX/C++ | max C++ error | max microJAX error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 5 | 0.541 ms | 20.15 ms | 39.1x | (2.1\times10^{-4}) | (6.8\times10^{-6}) |
| linear | 5 | 0.470 ms | 22.65 ms | 48.2x | (4.5\times10^{-4}) | (1.8\times10^{-5}) |
| all | 10 | 0.506 ms | 21.07 ms | 43.7x | — | — |

All ten cases returned a successful microJAX status, and all ten C++ values
were within (10^{-3}) of the VBM oracle at this setting.

At the looser (10^{-2}) setting, C++ had a median of 0.406 ms and microJAX
20.99 ms, or 53.9x in the aggregate.  C++ was faster because its radial
adaptation stopped earlier; two of the ten C++ cases were above (10^{-3})
error, as expected for the looser request.  microJAX remained below
(2\times10^{-5}) on this small corpus.

These numbers are a rough native-kernel comparison, not an API-level claim.
They should not be used to claim a universal speedup: the corpus is small,
microJAX and C++ use different adaptive policies, and no difficult tangent or
capacity-failure corpus was included.
