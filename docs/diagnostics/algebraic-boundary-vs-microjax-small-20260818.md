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
  support search.  The reported steady-state call reuses the native binary
  caustic cache for fixed `(s, q)`.
- microJAX: the production one-shot schedulers
  `mag_uniform_cpu_one_shot` or `mag_limb_dark_cpu_one_shot`, which are the
  full-solve path used by `mag_binary(..., backend="cpu")`.  JIT compilation
  is excluded; three steady-state CPU repeats are measured with one CPU
  thread.  No adaptive retry ladder or diagnostic tuple is timed.
- Direct VBMicrolensing calls are used only as an accuracy oracle and are not
  included in either timing.
- microJAX revision: `326cb0dd14617e5716a9d652b510709902d27b5b`.

The reproducible small harness is
[`benchmark_algebraic_vs_microjax_small.py`](../../tests/diagnostics/recal2026/benchmark_algebraic_vs_microjax_small.py).
Raw JSON is kept outside the repository under `/tmp`.

## Results

At the matched (10^{-3}) relative-error setting (steady-state, with the native
caustic cache warm):

| profile | cases | C++ median | microJAX median | median microJAX/C++ | max C++ error | max microJAX error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 5 | 0.541 ms | 3.984 ms | 10.7x | (2.1\times10^{-4}) | (7.3\times10^{-8}) |
| linear | 5 | 0.470 ms | 4.064 ms | 6.2x | (4.5\times10^{-4}) | (1.3\times10^{-7}) |
| all | 10 | 0.506 ms | 4.024 ms | 8.4x | — | — |

All ten cases returned a successful microJAX status, and all ten C++ values
were within (10^{-3}) of the VBM oracle at this setting.

## Cache and workload audit

The earlier 43.7x number came from timing microJAX's older adaptive research
functions (`mag_*_cpu`), which execute a 32/1 tier followed by a 64/2 tier at
this tolerance.  That is not the production `backend="cpu"` path and was an
unfair handicap for microJAX.  The production one-shot path reduces the
steady-state aggregate ratio to 8.4x while retaining smaller VBM errors on this
corpus.

The native cache policy still matters.  Running the same native executable in
batch mode exposes the first-call cost: the five uniform cases had a median
cold call of 2.21 ms and the five linear cases 2.16 ms.  Against the one-shot
microJAX timings this reduces the median ratio to roughly 1.8--1.9x.  Which
number is appropriate depends on whether `(s, q)` is held fixed across many
source epochs (steady state) or changes for every call (cold path).

The old adaptive diagnostic's `return_info=True` added only 1--4% in a
separate check, so the main correction is selecting the production scheduler,
not merely removing diagnostics.

## Light-curve-level sanity check

As a second check, 24 source positions with one fixed `(s, q)` were evaluated
as one light curve.  C++ used one cache-building call followed by the 23
steady calls; microJAX used its production one-shot light-curve scheduler.  JIT
compilation was excluded and both sides used one CPU thread.

| geometry/profile | C++ total | microJAX total | microJAX/C++ |
| --- | ---: | ---: | ---: |
| smooth / uniform | 7.88 ms | 55.30 ms | 7.0x |
| smooth / linear | 8.57 ms | 50.12 ms | 5.9x |
| central / uniform | 35.15 ms | 188.53 ms | 5.4x |
| central / linear | 41.32 ms | 77.68 ms | 1.9x |

All microJAX light-curve statuses were zero.  This vectorized comparison is
the more relevant one when a lens geometry is reused over many epochs, and it
does not show a 40--50x gap.

These numbers are still a rough native-kernel comparison, not an API-level
claim.  The corpus is small, microJAX and C++ use different fixed quadrature
policies, and no difficult tangent or capacity-failure corpus was included.  The
conservative conclusion is that the new C++ quartic path is faster on this
small matched corpus, but the honest range is roughly 1.8--1.9x cold, 1.9--7.0x
for the 24-epoch light-curve check, or 8.4x scalar steady-state—not an
unconditional 43x.
