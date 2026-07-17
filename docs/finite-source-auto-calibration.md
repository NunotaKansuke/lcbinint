# Calibration of automatic finite-source resolution and backend recommendations

## Scope and design constraint

This calibration answers two separate questions for a binary lens at one source
position:

1. Which image-plane grid and source resolution (`nbin`) should lcbinint use?
2. Under which measured conditions is an external contour integrator likely to
   be both accurate and faster?

The two answers are deliberately independent.  Automatic `nbin` is an lcbinint
runtime feature.  The VBM result is an internal recommendation flag: it does
not link, call, copy, or automatically dispatch to VBM.  This separation
also prevents an external result from silently becoming the numerical oracle.

The runtime rule is a one-shot preselection from quantities already computed at
the source position.  It performs no trial integration, convergence retry, or
fallback, so its overhead is a few scalar operations.

## Parameter-space experiment

The discovery sweep contains 256 binary-lens configurations and 6,144
source/profile rows.  It spans separation `s = 0.1 ... 4`, mass ratio
`q = 1e-6 ... 1`, source radius `rho = 1e-5 ... 0.1`, uniform sources, and
linear limb darkening `c = 0.5`.  Source positions include offsets from every
sampled caustic component and general field positions.  For every row, both
Cartesian and polar calculations were made independently at

`nbin = 16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256`.

This is 147,456 fixed-resolution lcbinint evaluations before auxiliary point,
automatic-mode, and VBM measurements.  A hard per-evaluation timeout prevented
a small set of extreme cases from monopolizing the sweep; timeout-censored rows
are marked in the artifact and are not treated as successful convergence.

The independent validation sweep uses a different seed, 128 configurations,
4,608 source/profile rows, and adds strong limb darkening `c = 0.8`.  No model
coefficient or threshold was refitted on validation data.

Finally, 854 difficult or high-magnification rows were rerun on both grids at
`nbin = 64, 100, 160, 256, 400`, with a 60-second timeout per evaluation.  This
targeted the regime where a contour result is specifically not assumed to be
reliable.

All configurations, seeds, timeout values, and bin lists are serialized in
[`summary.json`](../tests/diagnostics/results/finite-source-auto-20260716/summary.json).
Per-source results are in
[`source-profile-results.csv.gz`](../tests/diagnostics/results/finite-source-auto-20260716/source-profile-results.csv.gz).
The complete original JSON—including all individual bin values, timings,
timeouts, and comparison-engine outputs—is preserved in the three archives
under [`raw/`](../tests/diagnostics/results/finite-source-auto-20260716/raw/),
with hashes in `SHA256SUMS`.

## Reference construction

The accuracy target is

`|A - A_ref| <= 1e-4 + 1e-3 max(|A_ref|, 1)`.

A reference is accepted only when a high-resolution lcbinint sequence is stable
and corroborated by the other image-plane grid.  One-grid references require
an ordinary-magnification VBM agreement; VBM is never accepted as the sole
oracle.  Grid-independent point-source and hexadecapole paths are identified
separately.  In the discovery sweep, 6,115 of 6,144 rows obtained trusted
references (6,114 from two-grid agreement); 29 extreme rows remained untrusted
and were excluded rather than assigned a convenient reference.

For a trusted reference, the required `nbin` on a grid is the first tested bin
for which that result and every higher finite result remain within tolerance.
Thus isolated lucky crossings of the reference do not count as convergence.

## Automatic `nbin` rule

The Cartesian resolution model is an upper quantile regression for
`log2(required_nbin)`.  It uses only hot-path quantities:

- `log10(max(A_point, 1))`
- `log10(rho)`
- `log10(q_small)`, where `q_small = min(q, 1/q)`
- `log10(max(d_caustic/rho, 1e-3))`
- near-caustic strength `max(0, 2 - min(d_caustic/rho, 2))`
- companion-resolution risk `max(0, log10(max(4 rho/q_small, 1)))`
- linear limb-darkening coefficient

Local caustic-component size was investigated but excluded because obtaining it
on the runtime hot path would add avoidable work.  The frozen model is the 0.98
quantile fit on 4,787 uncensored discovery rows.  Its raw prediction is
multiplied by 1.10 and rounded upward to the next supported bucket.  The exact
means, standard deviations, and coefficients are stored—not rounded—in
[`calibrated-rules.json`](../tests/diagnostics/results/finite-source-auto-20260716/calibrated-rules.json).

Two physically localized floors cover sparse training tails:

- `100` bins when `0.9 < d_caustic/rho < 1.1` (source limb tangent to a caustic)
- `80` bins when `4 rho/q_small > 50` (small-companion resolution risk)

On the independent validation set, the selected grid/bin rule underpredicted
zero of 3,655 evaluable rows.  Median selected `nbin` is 50, mean is 57.51;
35.24% of rows use fewer than 50 bins and 46.79% use more.  This is intentional:
the result preserves a familiar median while spending resolution only where
the measured geometry requires it.

## Cartesian/polar selection

The frozen one-position rule selects polar integration when

- `A_point >= 300`, or
- `A_point >= 100` and `d_caustic/rho < 0.3`.

These polar cases use `nbin = 64`.  All other cases use the Cartesian quantile
model above.  On independent validation rows with matched finite timing
measurements, this boundary reduced aggregate measured time from 463.58 s to
451.21 s; the per-row timing oracle was 417.35 s.  The rule captures a useful
fraction of the available gain without a learned classifier or a runtime probe.

The long-timeout sweep is the important safety check.  It produced 843
two-grid-stable references and 11 additional polar-tail references, with no
numerical errors.  Among 716 evaluated rows with `A_point >= 1000`, polar
`nbin = 64` had zero tolerance violations.  Polar `nbin = 100` had zero
violations across all 854 difficult rows.  This supports using polar resolution
directly in the high-magnification regime rather than trusting a contour value
there.

## External-contour recommendation

The recommendation is intentionally conservative about correctness, not merely
about speed.  It is evaluated per source position because magnification,
caustic distance, limb contact, and runtime change along a light curve.

For a uniform source, recommend VBM contour integration only when

`A_point < 1000` and the source is outside the tangent band
`0.9 < d_caustic/rho < 1.1`.

For a limb-darkened source, recommend VBM dark integration only when

`A_point < 5` and `d_caustic/rho > 1.05`.

The excluded tangent band is empirical and important: failures occurred even
at low point-source magnification when the source limb was tangent to a thin
caustic.  High magnification is excluded because the targeted tests show that
lcbinint's polar integration remains stable there while a contour result must
not be promoted to an oracle.

Independent validation selected 1,289 uniform rows and 972 limb-darkened rows.
There were zero inaccurate or failed recommended VBM results.  Where both
engines were accurate, VBM was faster in 1,273/1,277 uniform rows and 885/972
limb-darkened rows.  These figures justify an internal backend-routing hint,
but not an automatic public dispatcher.

## Numerical defect found during calibration

The forced-Cartesian sweep exposed a fold-image walk exhausting its step guard.
The guard had been sized from the local seed magnification alone (about 17.9),
while the connected fold image carried a finite-source magnification near
3,954.  The fix sizes the guard from the larger of the seed Jacobian estimate
and the already available global point-source magnification hint.  It does not
retry with another method or grid.  The formerly failing sequence now converges
smoothly through `nbin = 200`, and the high-difficulty sweep completed with zero
numerical errors.

## Reproduction and limitations

The resumable generators are:

- `tests/diagnostics/finite_source_calibration_sweep.py`
- `tests/diagnostics/finite_source_high_magnification_sweep.py`
- `tests/diagnostics/finite_source_calibration_analyze.py`
- `tests/diagnostics/finite_source_calibration_finalize.py`

The frozen tolerance is relative `1e-3` plus absolute `1e-4`.  Tighter requested
tolerances require an explicit conservative resolution scaling in the runtime;
they are not claimed to have the same zero-violation validation result.  The
calibration covers binary lenses and linear limb darkening over the ranges
above.  Extrapolation outside those ranges is capped by the configured maximum
bin count and should be validated as new data become available.
