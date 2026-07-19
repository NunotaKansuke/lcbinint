# Triple finite-source auto calibration report

## Scope

This report freezes the evidence behind triple-lens finite-source `auto`.
It covers the inverse-ray grid decision (Cartesian versus polar) and the
runtime `nbin='auto'` decision.  The accuracy target used for fixed-grid
convergence labels is

```
abs(error) <= 1e-4 + 1e-3 * max(abs(reference), 1)
```

No production auto result is used as a numerical reference.

## Fixed-grid discovery and holdout

`triple_grid_calibration.py` generated a six-parameter triple-lens ensemble:

| set | seed | geometries | source positions / geometry | limb coefficients |
| --- | ---: | ---: | ---: | --- |
| discovery | 20260724 | 256 | 12 | 0.0, 0.5, 0.8 |
| independent validation | 20260725 | 128 | 12 | 0.0, 0.5, 0.8 |

Every source/grid/resolution evaluation runs in a separate process with its
own timeout and is checkpointed atomically.  Fixed runs pass both
`source_bins=<bins>` and `nbin=<bins>`; this is essential because merely
setting `source_bins` leaves the production auto selector enabled.

The sweep used Cartesian and polar grids at
`16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256` bins.  The source points
include caustic offsets from 0 through 30 source radii plus field points.
The raw summaries are `discovery-summary.json` and `validation-summary.json`.

Reproduction commands:

```bash
PYTHONPATH=build python tests/diagnostics/triple_grid_calibration.py \
  --output-dir .calibration-runs/triple-grid-discovery-fixed-20260724/data \
  --lens-cases 256 --points-per-case 12 --limb-coefficients 0,.5,.8 \
  --caustic-bins 1200 --timeout 15 --seed 20260724

PYTHONPATH=build python tests/diagnostics/triple_grid_calibration_analyze.py \
  .calibration-runs/triple-grid-discovery-fixed-20260724/data
```

## Polar-grid result

The ordinary fixed-grid sweep is insufficient to validate the actual
high-magnification mode: mode 4 changes the polar grid ratio.  Therefore
`triple_grid_targeted_validation.py` re-ran the high-magnification,
`distance/rho >= 3` candidates using the exact production auto settings and a
Cartesian 256/384/512-bin reference sequence with longer timeouts.

Of 78 trusted targeted rows, the old polar implementation failed 15.  The
failures came from five geometries and all three limb coefficients, and could
not be separated by point magnification or caustic distance.  The cause was
not the polar grid selector: polar seeded its flood fill only with physical
images of the source centre.  A small fold-image component can intersect the
finite source without existing at its centre, causing clean but incomplete
convergence at every resolution.

Polar now uses the same centre, nearby-caustic, and 64-position source-boundary
seed augmentation as Cartesian.  `triple_polar_seed_replay.py` replayed the
original frozen rows at production resolution and grid ratio 12.  All 78
trusted rows passed; the maximum error was 0.82% of the allowed tolerance.
Median polar time was 212 ms against 894 ms for Cartesian 256, and polar won
75/78 rows.

An independent-seed validation selected 30 high-magnification rows from the
128-geometry holdout and recomputed Cartesian 256/384/512 references.  Of the
22 references completing within 120 seconds, all passed; maximum error was
0.76% of tolerance.  The other eight were reference timeouts and are not
counted as successes or failures.  Median trusted-row time was 127 ms for
polar and 542 ms for Cartesian 256; polar won 20/22.

**Frozen grid rule:** triple auto selects polar when `A_point >= 100` and the
refined caustic distance is at least `3 rho`.  Auto uses at least grid ratio 12
and the fixed auto resolution described below.  Inside `3 rho`, the
topology-aware Cartesian/source-plane routes remain mandatory.  Explicit
polar retains its close-caustic Cartesian fallback.

## `nbin` result

`triple_grid_nbin_calibration.py` fitted an exploratory quantile regression to
the fixed Cartesian sequences.  It produced zero label underpredictions on
9,213 discovery rows and 4,608 holdout rows (`nbin-fit.json`).  Direct runtime
auditing then found the fit unsuitable for production: the sampler's
polygonal caustic distance is not bit-identical to runtime's refined caustic
distance, so identical physical samples can enter different buckets.

**Rejected rule:** distance-feature quantile regression plus contact-band
floors.  It is retained only as a diagnostic artifact, not compiled into the
library.

**Frozen `nbin` rule:** triple `nbin='auto'` uses 256 bins, capped by the
caller-provided `max_source_bins`.  This is the largest fixed resolution in
the convergence sweep, has no runtime probe/refinement, and does not depend on
the mismatched distance proxy.  When `nbin` is explicitly fixed, the selector
is not invoked.

## Fast-path result: point source, hexadecapole, and Cartesian fallback

The triple fast-path policy was separately calibrated using exact production
mode-4 routing against forced-Cartesian 256- and 512-bin references.  A row is
trusted only when those two references agree; the same tolerance as above is
then applied to the selected fast-path value.  The discovery run covers 591
rows and the independent holdout covers 252 rows, with requested caustic
offsets from 3 to 20 source radii.

The original derivative-based point-source shortcut failed 21 trusted
discovery rows and 12 trusted holdout rows.  The original hexadecapole guard
at 3 source radii failed 3 discovery rows; a 4-radius replay still failed 6
holdout rows, all in the narrow 4.87--4.97-radius band.  Its self-reported
error can be essentially zero there, so it cannot be trusted as the sole
gate.

**Frozen fast-path rule:** for a triple lens, no derivative/point-source
shortcut is used inside the existing 20-radius point-source domain.  The
hexadecapole self-consistency check is eligible only at a caustic distance of
at least `max(hexadecapole_threshold, 5) * rho`; otherwise auto uses Cartesian
inverse ray.  The existing point-source exit beyond the 20-radius domain is
unchanged; a separate 20--60-radius replay found zero violations in 225
discovery and 132 holdout rows, with converged Cartesian 256/512 references in
every row.

Final replay with the 5-radius rule completed all reference-backed rows:

| set | rows | Cartesian | hexadecapole | trusted fast-path failures |
| --- | ---: | ---: | ---: | ---: |
| discovery | 510 | 444 | 66 | 0 |
| independent holdout | 225 | 207 | 18 | 0 |

The replay script is
`tests/diagnostics/triple_fast_path_replay.py`; the original collector and
its frozen references are produced by
`tests/diagnostics/triple_fast_path_calibration.py`.  Both run each source in
an isolated process with a timeout and atomically checkpoint every row.

## Topology-aware contact, grazing, and tangent rule

Distance-only Cartesian fallback is not sufficient close to a triple caustic.
In the outside-limb grazing regime, both Cartesian and polar image-plane grids
can converge to the same biased value because both miss the same sub-cell-thin
image finger.  Conversely, low-order source-plane rules can falsely converge
when the source actually crosses a tiny caustic sliver.  The production rule
therefore uses the already cached caustic branches and the refined distance,
not a named binary topology transplanted to a triple lens.

The frozen mode-4 rule is:

1. Perform no topology branch scan at `d_caustic >= 2 rho`; the existing
   point/hex/Cartesian paths have no added overhead there.
2. Below `2 rho`, scan branch segments once and classify whether the caustic
   enters the source disk.  Either refined distance `< rho`, an inside branch
   vertex, or a branch crossing forces Cartesian inverse ray.
3. For an outside disk with `A_point < 100`, cross-check a 64-order radial rule
   against a structurally independent 64-order Gauss--Legendre chord rule.
   The low-order result is accepted only when their difference is below
   `target_error / 40`.  The factor 40 is the widest zero-violation discovery
   boundary; it is frozen before holdout.
4. A rejected low-order pair escalates to chord orders 160/256.  If those
   disagree it always evaluates 400/512 and uses that pair for the final
   convergence decision.  Order 512 is a hard runtime ceiling.  A finite value
   at the ceiling is returned with `converged=false` rather than silently
   substituting the known biased Cartesian value.
5. `A_point >= 100` remains Cartesian in this band.  The discovery data showed
   low-order source-plane aliasing around triple cusps at high magnification,
   while the Cartesian and polar tails agreed there.

The topology discovery covers 3,045 rows between 0.5 and 2 source radii; its
independent holdout contains 1,536 rows.  Every changed grazing result received
an independent source-plane chord reference, not an image-plane reference:

| set | outside grazing rows | trusted 160/256 | trusted 400/512 | tolerance failures | unresolved at 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| discovery | 1,019 | 951 | 43 | 0 | 24 |
| independent holdout | 531 | 518 | 9 | 0 | 4 |

One discovery high-tail row lacked a complete trusted classification and is
not counted as either trusted or unresolved.  The unresolved rows are retained
as explicit non-convergence, not counted as successes.  On holdout, 353/531
grazing rows stopped at the low cross-check, 165 at 160/256, and 13 reached
400/512; 527 reported convergence and 4 reported non-convergence.  Median
production time over these grazing rows was 120 ms, versus 4.11 s for the
calibration-only fixed 256-order chord reference.  An exploratory 800/1024
tail did not complete even one parallel row in two minutes and was rejected as
a runtime strategy.

The checkpointed programs are
`triple_topology_calibration.py`, `triple_topology_replay.py`,
`triple_topology_reference_enrich.py`, and
`triple_topology_tail_escalation.py`.  The separate deep-tail diagnostic records
the rejected 800/1024 experiment.

## Implementation checks

The production implementation is in
`src/lcbinint/magnification/finite_source_magnifier.cpp`:

- `calibrated_triple_resolution()` returns `min(256, max_source_bins)` and
  leaves the binary resolution selector's `prefer_polar` flag false; the
  separately calibrated triple grid selector owns the polar decision.
- The selector is called only when automatic source bins are enabled.
- Triple auto polar uses the `A_point >= 100`, `distance >= 3 rho` rule and
  the seed-complete polar integrator; explicit polar uses the same seed set.
- Triple hexadecapole is blocked inside five source radii, and the former
  derivative-based point-source shortcut has been removed.
- Triple auto classifies caustic crossing versus outside-limb grazing from
  refined distance plus cached branch geometry, and uses bounded dual-rule
  source-plane convergence for the grazing case.

Executed after the final implementation:

```bash
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
PYTHONPATH=build pytest -q
git diff --check
```

Results: CTest `1/1` passed, pytest `153` passed (`3` skipped), and `git diff --check`
passed.  A direct Python check also confirmed that Cartesian `nbin='auto'` and
an explicit 256-bin Cartesian run produce identical values for a finite-source
regression case.
