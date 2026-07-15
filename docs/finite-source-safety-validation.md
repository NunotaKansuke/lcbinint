# Finite-source fast-path safety validation

## Scope

`tests/diagnostics/finite_source_safety_sweep.py` validates binary-lens method
selection over the 2018 VBBinaryLensing test domain:

- separation `s`: 0.1 to 4
- mass ratio `q`: 1e-6 to 1
- source radius `rho`: 1e-4 to 0.1
- absolute tolerance: 1e-4 to 1e-2

Lens cases combine boundary anchors with log-uniform random samples.  Source
positions are concentrated between `0.5 rho` and `50 rho` from sampled caustic
points, with additional far-field positions.  Every result is compared with a
uniform-source VBMicrolensing calculation using a reference tolerance no
larger than `1e-6` and at least 30 times tighter than the requested tolerance.

The primary failure condition is a `point_source` or `hexadecapole` result
whose absolute error exceeds the requested tolerance.  The sweep also reports
selection recall: the fraction of reference-safe point-source positions that
the implementation actually keeps on the point-source path.

## Current result

The long validation used nine independent seeds, each with 96 lens cases and
12 source positions per case.  It covered 10,368 source positions across
anchored and random lens cases.  The combined method counts were:

| Method | Count |
| --- | ---: |
| point source | 2,198 |
| hexadecapole | 1,226 |
| cartesian inverse ray | 5,406 |
| source-plane quadrature | 1,538 |

There were 3,424 point-source or hexadecapole fast-path results:

- tolerance violations: 0
- results above half the requested tolerance: 0
- maximum error / tolerance: 0.162
- unconverged fast-path results: 0
- evaluation exceptions: 0

Of 3,896 reference-safe point-source positions, 2,198 were selected as point
source, for a recall of 56.4 percent.  The remaining positions were handled
conservatively by hexadecapole or a full finite-source method.

## High-magnification reference audit

The same sweep records raw differences for every selected method, although
only a fast path exceeding tolerance makes the diagnostic command fail.  With
the cartesian inverse-ray backend deliberately selected, it initially reported
large numbers of full-method differences from VBMicrolensing.  Those counts
must not be interpreted as verified lcbinint failures.

| Method | Samples | Above tolerance | Unconverged | Maximum error / tolerance |
| --- | ---: | ---: | ---: | ---: |
| point source | 2,198 | 0 | 0 | 0.162 |
| hexadecapole | 1,226 | 0 | 0 | 0.036 |
| cartesian inverse ray | 5,406 | 2,913 | 4,529 | 7.64e8 |
| source-plane quadrature | 1,538 | 136 | 245 | 3.31e4 |

The worst raw discrepancy had a VBMicrolensing finite-source value near
389,888.  Dense source-plane integration of the point-source solutions from
both lcbinint and VBMicrolensing instead converged near 9,000; lcbinint's
automatic polar result was 8,997.  The two libraries also agreed on every
point-source image used in that check.  VBMicrolensing's finite-source result
is therefore not a reliable reference for this extreme-magnification case.

Forced cartesian mode had a separate, real error at the same position: it
returned 5,739.  The image seeds were present, but each image-area walk stopped
at the hard `source_bins^2 * 2000` step limit and silently returned partial
area.  The guard now scales from each seed's `1 / |J|` image-area estimate and
returns a numerical failure rather than partial area if it is ever exhausted.
The corrected 40-bin result is 8,951.  The default automatic selector was
already unaffected: it selected the polar method directly and returned 8,997.
The forced-cartesian case is retained as a regression using the independently
integrated value near 9,000 rather than the pathological finite-source
reference.

High-magnification validation must therefore use agreement among point-source
image integration, resolution/phase convergence, and per-image contributions.
VBMicrolensing remains useful in ordinary regimes but is not treated as the
sole oracle in this corner of the domain.  The raw full-method table above is
retained to document what triggered the audit, not as an accuracy verdict.

## Safety coefficient check

The local safety coefficients were evaluated on a grid around the initial
values `(quadrupole+cusp, ghost, planetary) = (6, 2, 2)`.

- The initial `(6, 2, 2)` local tests alone admitted one point whose
  point-source error was about 129 times its requested tolerance.  The
  independent caustic-distance guard rejected it, so the actual method switch
  remained within tolerance.
- Increasing the ghost coefficient from 2 to 3 removed all local-test
  violations in the long sweep.  At fixed quadrupole+cusp and planetary
  coefficients, its local-test recall changed from 58.9 to 57.2 percent.
- A ghost coefficient of 1 admitted between five and nine failures across the
  tested quadrupole+cusp and planetary coefficient combinations.
- Reducing the quadrupole+cusp coefficient could improve recall, but the
  observed gain does not justify weakening it while the sweep remains
  randomized rather than exhaustive.
- Varying the planetary coefficient between 1 and 4 produced no fast-path
  failure in this sample, but did not provide enough benefit to override the
  established safety value.

The implementation therefore uses `(6, 3, 2)`.  Both dangerous ghost-margin
cases are retained as regression tests.  The actual method selection retains
the caustic-distance and bounding-box checks as independent safety layers.

## Running the sweep

From the repository root with an in-tree build:

```sh
PYTHONPATH=build:python python tests/diagnostics/finite_source_safety_sweep.py \
  --lens-cases 96 \
  --points-per-case 12 \
  --seed 731 \
  --output /tmp/finite-source-safety.json
```

The script exits non-zero if a fast path violates tolerance or if an evaluation
raises an exception.  It also reports method-level errors for the slower
backends without treating them as failures of the fast-path selector.

## Limitations and next checks

This stratified randomized sweep is not an exhaustive proof.  A next
validation should check method-switch continuity along trajectories tangent
to folds and cusps and should target the transition surfaces of each safety
test directly.  Non-convergence of the inverse-ray path is reported
separately and is not used to judge the point-source and hexadecapole safety
tests.
