# Calibration of automatic finite-source resolution and VBM-routing rule

> **Status, August 2026.** The binary resolution rule described below is what the
> runtime ships today, and this document remains its specification. It has since
> been re-measured against the certified algorithm, and the measurement says the
> shipping rule is heavily over-resolved — see
> [Recalibration against the certified algorithm](#recalibration-against-the-certified-algorithm-august-2026)
> below. That recalibration has **not** been applied to the runtime. Nothing in
> the older sections has been retracted; they were correct for the algorithm they
> were fitted to, which is the point the new campaign turns on.

## Scope and design constraint

This calibration originally answered two separate questions for a binary lens
at one source position:

1. Which image-plane grid and source resolution (`nbin`) should lcbinint use?
2. Under which measured conditions is an external contour integrator likely to
   be both accurate and faster?

The two answers were deliberately independent, and remain so: automatic
`nbin` is an lcbinint runtime feature and is documented in full below. The
VBM-routing rule that answered question 2 has since moved out of lcbinint
entirely — lcbinint has no concept of VBM or backend routing today, and
exposes only engine-neutral geometry/diagnostic primitives
(`lcbi_finite_source_geometry[_array]`, and the `separation`/`mass_ratio`/
`caustic_distance` fields on `lcbi_result`) for an external host to implement
its own routing on top of. moasarc is the current owner of the routing rule
derived in this document; see its own router documentation for the ported
rule and its runtime dispatch. The calibration methodology and results below
are kept as the historical record of how that rule was derived.

The binary runtime `nbin` rule starts with a calibrated preselection from
quantities already computed at the source position. Cartesian integration then
compares its independent area-error estimate with the requested tolerance. A
mismatch increases only that evaluation to the smallest supported bucket
implied by the measured shortfall, up to `max_source_bins`; fixed integer
`nbin` never retries.

The post-check is order-aware: the edge-corrected Cartesian scan uses a
second-order boundary estimate when there are no fold seeds and row-to-row
jumps remain small. Fold or large-jump topology warnings retain their original
first-order scale. This prevents smooth high-area images from turning the
feedback step into a blanket resolution increase.

## Triple-lens calibration

The binary calibration below must not be applied to a triple lens.  Triple
polar originally used only the physical images of the source centre as flood-
fill seeds.  A finite source can contain a small fold-image component absent
at its centre, so 15 of 78 trusted high-magnification rows converged to an
incomplete area.  Polar now shares Cartesian's centre, nearby-caustic, and
64-position source-boundary seed augmentation.  Replaying the original 78
trusted rows produced zero violations; an independent 30-row holdout produced
22 converged Cartesian 256/384/512 references and zero violations, while eight
references timed out and were not counted.  Triple auto therefore selects
polar for point magnification at least 100 and refined caustic distance at
least `3 rho`, with a minimum angular grid-ratio of 12.  The inner three-source-
radius band remains on the topology-aware Cartesian/source-plane path.

Triple `nbin='auto'` consequently uses a fixed 256-bin Cartesian resolution,
capped by `max_source_bins`; it does not use the binary quantile model below.
The full triple calibration record and reproduction commands are in
[`triple-grid-auto-20260724`](../tests/diagnostics/results/triple-grid-auto-20260724/README.md).

Triple fast paths were calibrated independently.  Within the existing
20-source-radius point-source domain, auto does not take the derivative-based
point-source shortcut.  Hexadecapole is permitted only at least
`max(hexadecapole_threshold, 5) * rho` from a caustic and only after its normal
self-consistency check; nearer rows use Cartesian inverse ray.  Forced
256/512-bin Cartesian replay found zero fast-path tolerance violations in 510
discovery and 225 independent holdout rows.  A separate 20--60-radius replay
also found zero point-source violations in 225 discovery and 132 holdout rows.

Below two source radii, triple auto additionally scans the cached caustic
branches and combines that result with refined distance.  A caustic entering
the source disk forces Cartesian inverse ray.  An outside-limb grazing source
with point magnification below 100 uses independent radial/chord source-plane
rules: a 64-order pair needs a calibrated 40x safety margin, otherwise it
escalates through 160/256 and, when needed, 400/512.  No topology scan is added
at distances of two source radii or more.  Independent source-plane references
gave zero tolerance violations in 994 trusted discovery rows and 527 trusted
holdout rows.  Rows still unresolved at the 512-order ceiling return the best
finite source-plane value with `finite_source_converged=false`; they are not
silently promoted to converged Cartesian results.

Binary topology proxies are likewise no longer allowed to expand an expensive
region across disconnected caustics.  Once the refined segment distance is at
least `20 rho`, it supersedes the local ghost and planetary proxies; point
source still has to pass its tolerance-aware derivative check, otherwise hex
still has to pass its self-consistency check.  The threshold had zero point-
source tolerance violations in 968 discovery and 726 independent holdout rows.
On the Roman planetary fixture this reduced Cartesian routing from 1,372 to
393 epochs while retaining zero `1e-3` discrepancies against a `Tol=1e-5` VBM
curve reference.

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

## VBM-routing rule (moved to moasarc)

This section is retained as the historical record of the rule's derivation.
The rule itself, and the runtime code that applies it, now live entirely in
moasarc — lcbinint exposes only the neutral geometry primitives it needs
(see "Scope and design constraint" above) and contains no VBM-routing logic
of its own.

The recommendation is intentionally conservative about correctness, not merely
about speed.  It is evaluated per source position because magnification,
caustic distance, limb contact, and runtime change along a light curve.

For a uniform source, the production recommendation is

`A_point < 1000` and the source is outside the tangent band
`0.95 < d_caustic/rho < 1.05`.

The initial frozen calibration used `0.9..1.1`. A speed-focused replay of both
stored datasets narrowed the band to `0.95..1.05`: discovery selected 2,609
uniform rows (2,589 also accurate in lcbinint; VBM faster in 2,580), and
independent validation selected 1,314 (1,301 also accurate in lcbinint; VBM
faster in 1,297), with zero inaccurate or failed selected VBM results in both.

For a limb-darkened source, recommend VBM dark integration only when

`A_point < 5` and `d_caustic/rho > 1.05`.

The excluded tangent band is empirical and important: failures occurred even
at low point-source magnification when the source limb was tangent to a thin
caustic.  High magnification is excluded because the targeted tests show that
lcbinint's polar integration remains stable there while a contour result must
not be promoted to an oracle.

Independent validation selected 1,314 uniform rows and 972 limb-darkened rows.
There were zero inaccurate or failed recommended VBM results.  Where both
engines were accurate, VBM was faster in 1,297/1,301 uniform rows and 885/972
limb-darkened rows. These figures support moasarc's router; lcbinint itself
remains independent of any VBM or contour implementation.

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

## Recalibration against the certified algorithm (August 2026)

The quantile model above was fitted before the component certificate existed.
At that time a bin count was buying two things at once: quadrature accuracy, and
enough resolution that the flood fill would not miss a disconnected image
component.  The certificate now proves disk support and topology directly.  It
does **not** prove that the quadrature error meets `reltol` — that remains the
grid's job — so the question the recalibration asks is how much of the old bin
count was paying for the part the certificate has taken over.

The measurement is a fresh 160-case, 2880-row binary corpus (seed 20260803),
timed per 24-epoch block rather than per epoch, with cost always read at
*delivered* accuracy rather than at a requested tolerance.  Full method, tables,
holdout coverage, and reproduction commands are in
[`results/recal2026/README.md`](../tests/diagnostics/results/recal2026/README.md);
the fitted artifacts are `nbin_rule.json`, `grid_switch_rule.json`, and
`route_audit.json` beside it.

Three findings bear on this document.

**The resolution rule is over-resolved by one to two orders of magnitude.**
Against the bins the corpus actually requires, the shipping rule spends 25× to
278× the work depending on grid and tolerance.  A *constant* bin count per
tolerance — no regression, no features — covers 99.3–100% of an independent
holdout, the single exception being Cartesian at `1e-4`, where a rho-dependent
linear rule is needed to reach 99.8%.  The seven-feature quantile model is
therefore not carrying its own weight under the certified algorithm.  Adopting a
constant would be a runtime change with a real correctness surface (the required
counts are right-skewed, p99 is three to ten times the median), so it is recorded
here as a measurement and not applied.

**The Cartesian/polar boundary moves, and moves down.**  The frozen rule selects
polar at `A_point >= 300`, or at `A_point >= 100` with `d_caustic/rho < 0.3`.
Re-derived on measured corpus time against a per-block oracle, the optimum is a
single condition, `A_point > 200`, joint-optimal across all six
profile × tolerance cells and flat over the decade 100–500.  The second clause
buys nothing measurable, and adding a rho condition does not improve the fit.
Always-Cartesian costs 1.29–1.45× the oracle; `A_point > 200` costs 1.07–1.26×.

**The fast-path boundaries hold, with two exceptions at `1e-4`.**  Routing was
audited by grouping delivered error by the route the pipeline actually took.
The point-source boundary misses on 0% of blocks at every accuracy and both
profiles, and every pure inverse-ray route likewise.  Two routes miss at `1e-4`:
hexadecapole on 7.6–8.8% of blocks (worst error 1.5× target) and
`inverse_ray_polar + source_plane_quadrature` on 8.0–11.1% (worst 2.6× target).
No cheap predicate separates either cleanly — the tightest bounding box in
`(rho, A)` that catches every hexadecapole miss also rejects 31.5% of legitimate
hits, which would move those blocks from 0.14 ms/epoch to roughly 80 ms/epoch.
The recommendation is to scale the hexadecapole acceptance test with rho, which
is a code change needing its own study, rather than to tighten the cut.

The quadrature misses are the more interesting of the two: all of them sit at
`A >= 39.8` with `d_caustic/rho` between 0.95 and 1.7 — the tangency regime,
where the source limb grazes a fold.  This is the same regime the frozen
VBM-routing rule excluded by hand as a tangent band, and it corroborates a
separately flagged `d/rho = 1.001` disagreement.  It is the subject of its own
study and is not resolved here.
