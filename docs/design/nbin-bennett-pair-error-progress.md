# Signed Bennett endpoint-pair errors (2026-10-13)

## Implemented

The isolated experiment adds `boundary_pair_error`. Given corrected contributions
Q1, Q2 and valid area intervals A1 in [L1,U1], A2 in [L2,U2], it computes with
outward rounding

    R1 = Q1 - [L1,U1], R2 = Q2 - [L2,U2]
    Rpair = R1 + R2
    error_upper = max(abs(Rpair.lower),abs(Rpair.upper)).

The preceding approach took absolute values before summation. The new approach
preserves the signed offsets and only takes an absolute value after combining
both contributions. It reuses all area computations; it performs no lens-map,
polynomial-root, or boundary-refinement calls. It remains conditional on the
supplied area intervals being valid for the intended physical integration domain.

Actual Eq. (15) endpoint pair IDs are recorded directly in the experimental
integration trace, using the run indices selected by the existing stencil.
Pairs are not inferred from point proximity or ordering. The existing stencil
requires at least four runs, so its two lower and two upper endpoint samples
do not overlap. Ordinary integration calls allocate no pair diagnostics.

## Same-domain comparison

Three geometries, nbin 16/64, two lower endpoint pairs per evaluation: 12 actual
pairs / 24 rectangles. These are the previous v3 cap rectangles with saved
independent reference areas; no repeat lens evaluation is needed for the bound.

| Existing cap budget per row | Mean paired upper / separate upper | Range |
| --- | ---: | ---: |
| 8 | 92.6% | 75.5–100% |
| 32 | 55.1% | 45.5–82.1% |
| 128 | 50.6% | 45.3–56.3% |

All 36 paired signed intervals contained the independent reference residual
(including quadrature change and the rounding of saved reference summaries).
Recorded corrected contributions match the saved ones exactly. Lower endpoint
pairs only were measured here; each evaluation also applies two upper pairs,
which this saved-input comparison does not audit.

This demonstrates a real loss from taking absolute values too early. It is
NOT yet the remainder of the complete Eq. (15) physical endpoint panel: the
original two row rectangles omit exterior cap area and have differing x extents.
The pair helper's total measured cost over 12 calls was about 11–13 microseconds
at the 32/128 settings, including Python overhead. Single pilot timings do not
establish total solver speedup. Input area computation costs are unchanged.

Additional checks: 44 constant-width full-endpoint cases, 300 randomized
high-precision interval arithmetic checks, exact/opposite residual cases, and
invalid interval input. Existing unit_core and unit_boundary_support pass.
For a constant-width endpoint, w1+w2=1.5+eta. Its true area includes the exterior
eta*h below the first sample and h/2 above the second; pairing preserves the
cancellation down to arithmetic enclosure width. This analytic check includes
the cap and does not equate the endpoint with two midpoint-height rows.

## Exterior-cap experiment: unresolved

A second audit encloses one rectangle spanning both samples, widened to their
combined x extent and extended below the first sample by one full h (rather
than h/2). It integrates the actual source preimage within that supplied
rectangle; it does not assert that this is the complete physical endpoint.
All 24 area/error intervals (12 panels, budgets 32/128) passed independent
60-digit sextic-slice quadrature checks.

However, the larger domain worsens the area enclosure width: its mean width
is 30.5% of reference area at 32 columns and 20.7% at 128. Moreover, none of the
12 panels has certified outside lower/left/right faces. Independent 60-digit
point evaluations actually find inside points on each of these faces for every
panel. Thus this is a geometric extent failure, not merely interval inflation.
The bound is valid only for its rectangle, not the whole endpoint.

The recorded ratio to the previous separate-row upper bounds reaches 2.14 in
one case, but those bounds have DIFFERENT target domains. That ratio must not
be interpreted as a controlled accuracy comparison. Merely enlarging rectangles
and increasing column count does not solve complete endpoint error estimation.

## Decision and next work

Keep cheap signed pair aggregation. Do not promote either rectangle audit to a
global tolerance gate and do not reduce production nbin.

The remaining estimator must follow the actual connected boundary into the
endpoint and bound the corrected panel residual, including its exterior cap.
It should use existing boundary positions/derivatives and local continuation;
repeatedly enlarging and densely certifying entire rectangles would lose the
performance goal. A derivation for the combined stencil remainder is needed
before deciding its useful local refinement budget. An assumption of statistically
random cancellation or a fitted discount factor is not a mathematical guarantee.

Independent intervals still discard correlation in the uncertain boundary
shape. Signed pairing recovers cancellation of the known residual offsets; it
does not yet exploit that remaining shape correlation.

Evidence and reproducible snapshots:

- `tests/diagnostics/results/recal2026/boundary-pair-20261013-v2/`
- `tests/diagnostics/results/recal2026/boundary-endpoint-panel-20261013-v3/`

Both include input/native hashes and saved results. Shared reference quadrature
uses sextic candidate roots plus high-precision refinement, not interval-certified
quadrature. Main master kernels, Bennett weights, and resolution law are unchanged.
