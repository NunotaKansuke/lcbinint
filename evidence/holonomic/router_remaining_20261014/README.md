# Remaining VBM deficits at 88f7de2c

Run from this checkout: `python evidence/holonomic/router_remaining_20261014/measure.py`.
Explicit native Release build, one thread, 201 epochs, uniform source,
absolute and relative tolerance both 1e-3; 41 alternating full-curve repetitions.
VBMicrolensing 5.5 uses BinaryLightCurve. No gradients. Compilation, constructor
and code warmup are excluded. Cold calls clear preparation every curve.

| Curve | lcbinint diagnostic ms | VBM ms | Native epoch sum ms |
|---|---:|---:|---:|
| ordinary | 0.264 | 0.178 | 0.242 |
| wide planetary | 5.761 | 4.476 | 5.723 |

| Wide planetary route | Epochs | Cold native ms | Retained preparation ms |
|---|---:|---:|---:|
| point | 114 | 0.103 | 0.103 |
| hex | 66 | 2.577 | 1.111 |
| seeded radial | 21 | 3.041 | 1.656 |

Each cold group is summed within the original uninterrupted curve. It includes
selection and any preparation incurred at that epoch, not just the final kernel.
The first wide epoch is hex: 1.513 ms cold versus 0.0156 ms retained. Thus the
large cold hex total primarily includes the initial detailed caustic preparation.
The radial group also saves 1.385 ms when caustic/seed preparation is retained.
This isolates the combined preparation/cache effect, not individual internal
stage timers. Cold and retained values and method codes agree in every repetition.
Retained preparation is a diagnostic counterfactual, not an advertised fresh-lens
timing: it reuses geometry and source-specific seeds from the identical curve.

Ordinary native epoch sum is 0.242 ms cold versus 0.192 ms retained. Roughly
0.050 ms is preparation-sensitive; the remaining native time is close to the
0.178 ms VBM whole-call time. Full diagnostics additionally incur about 0.022 ms
outside the epoch loops. The prior scalar/public matched measurements remain
the appropriate scalar-only API comparison.

VBM evaluations restricted to epochs grouped by *lcbinint* route take 0.472,
2.653 and 1.296 ms for point, hex and radial groups respectively. These are
separate BinaryLightCurve calls, not VBM's internal method labels or an additive
full-curve breakdown. In particular VBM may integrate epochs where lcbinint uses
point/hex. They indicate a radial-epoch deficit remains even after preparation
reuse, but cannot establish which VBM internal stage is responsible.

The previous 16-condition matrix has deficits only for ordinary point-only
at 1e-3 (uniform and linear LD), and wide planetary uniform at 1e-3. Other 13
conditions are faster than VBM. Mixed curves retain existing uncertified support
epochs and are timed through diagnostics; VBM returns values only.

Source code was not modified. `results.json` contains measurements and revision;
`measure.py` reproduces the route decomposition with the existing native timers.
