# Fold Overlap Aliasing Fix (2026-06-23)

## Trigger Case

Notebook/example case:

```python
separation = 0.95
mass_ratio = 0.01
u0 = -0.001
alpha = 0.5
rho = 5.0e-3
source_bins = 50
max_source_bins = 400
reltol = 1.0e-3
```

At `t = 0.006015037593984918`, the old installed binary returned:

| code | magnification | relative error vs VBBL |
|---|---:|---:|
| VBBL `Tol=1e-3` | 238.8033306 | - |
| lcbinint fixed50 before fix | 219.5206305 | 8.07e-2 |
| lcbinint fixed50 after fix | 238.7805431 | 9.54e-5 |

The notebook initially continued to show the old `8.075e-02` max relative error
because it was importing the stale site-packages extension. Reinstalling with
`pip install -e . --no-build-isolation` fixed the import.

## Root Cause

`legacy_imagearea4_binary` used a scan-row bounding-box overlap test for both:

1. deduplicating seeds that lie in an already integrated image component, and
2. identifying genuinely separate fold-branch components.

Near the critical curve this was too aggressive. A seed almost exactly on the
critical curve (`|J| ~ 2e-4`) traced only the near-caustic subset of a same-parity
fold branch, but its coarse scan-row bounds overlapped a later, less singular
seed on the same branch. The later seed was marked as a future overlap and never
processed, so the branch contribution was partially missed.

This produced grid-phase dependent failures:

- `source_bins=50` and `100`: missed one branch component.
- `source_bins=35, 65, 70, 80, 110, ...`: processed the later branch seed and
  agreed with VBBL.

## Fix

For same-parity fold images only, if the current seed is extremely close to the
critical curve, do not use its coarse bounding box to suppress a future fold seed.
Let the later seed run; the existing `other < image_index` path subtracts the
earlier partial component when the later scan overlaps it.

The condition is intentionally narrow:

- `source_radius >= 4e-3`
- `source_bins >= 35`
- current seed has `|J| < 1e-3`
- future seed is also fold-like: `|J| < kFoldJacThreshold`
- same parity, not opposite parity

Coarser `source_bins=25` and tiny-source high-magnification cases are excluded
because this rule can over-process many fold seeds there.

## Performance After Fix

For the trigger light curve (`400` points, `source_bins=50`, `max_source_bins=400`,
`reltol=1e-3`, VBBL `Tol=1e-3`), after reinstalling the extension:

| mode | VBBL ms/pt | lcbinint ms/pt | lc/VBBL | max rel |
|---|---:|---:|---:|---:|
| no LD | 0.101 | 2.458 | 24.2x | 2.45e-4 |
| LD `c=0.5` | 3.121 | 3.316 | 1.06x | 1.12e-4 |

The original `max rel = 8.075e-02` is gone.

## LD Overhead Interpretation

Fixed `source_bins=50` does not show a large LD-only cell-weight overhead. The
extra LD cost mainly comes from adaptive refinement:

- no LD adaptive: `53` refined points, max level `1`
- LD adaptive: `61` refined points, max level `2`

The extra level-2 point is around `t = 0.010025062656641603`.

At that point:

| mode | bins | mag | est | target | outcome |
|---|---:|---:|---:|---:|---|
| no LD | 100 | 123.757 | 0.1212 | 0.1238 | accept |
| LD | 100 | 116.105 | 0.1454 | 0.1161 | refine |

The actual LD error is already below `reltol=1e-3` at `source_bins=50`, so this is
a conservative diagnostic-estimate effect, not a real LD integration failure.
Future tuning can reduce LD over-refinement by adjusting the LD diagnostic
normalization/safety factor.

## Validation

Commands used:

```bash
cmake --build build
python -m pip install -e . --no-build-isolation
pytest tests/regression/test_vbm_consistency.py
ctest --test-dir build --output-on-failure
python tests/diagnostics/adaptive_source_bins_sweep.py \
  --source-bins 50 --max-bins 400 --reltol 1e-3 \
  --random 12 --random-times 51 --seed 20260622
```

Results:

- regression tests: `62 passed`
- ctest: passed
- random adaptive sweep: `accepted_bad=0`
- known extreme high-magnification tiny-source case remains `unconverged`, not
  silently accepted.
