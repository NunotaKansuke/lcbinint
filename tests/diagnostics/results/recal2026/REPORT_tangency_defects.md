# Defect report — the tangency band, `d/rho ≈ 1`

August 2026 recalibration campaign. This file is separate from `REPORT_speed.md`
on purpose: the results below are **correctness findings, not performance
findings**, and two of the three are new.

They were found while auditing the references that the speed campaign rests on.
The audit was prompted by `README.md` Stage 3, where all six `1e-4` quadrature
misses turned out to sit at `A ≥ 39.8` with `d/rho ∈ [0.95, 1.7]` — the regime
where the source limb grazes a fold.

## Summary

| # | defect | status | cause | severity |
|---|---|---|---|---|
| A | Cartesian row scan assumes one interval per row | **new, confirmed** | **identified** — `LegacyImageAreaScratch` holds one `xmin`/`xmax` per row (§3) | silent wrong answers up to **2.2e-2** relative, `certified = true` |
| B | Both lcbinint grids wrong *together* near `d/rho ≈ 1.01–1.1` | **new, confirmed** | **not isolated** — three ranked candidates, all upstream of the lattice (§4) | ~1e-3 relative, invisible to any lcbinint-internal cross-check, **implicates the reference construction** |
| C | Flood fill is not seed-order independent | known, restated | **identified** — a seed below `kFoldJacobianThreshold` claims cells it does not count (§5) | probe rings currently mask it; blocks a planned optimisation; leading hypothesis for B |

All three fail **open**: the answer is wrong and the certificate says it is
proven. None of them is caught by the component certificate, and that is
expected rather than surprising — the certificate proves disk support and
topology. It does not prove the quadrature.

---

## 1. Method

### The three-witness scan

`tests/diagnostics/recal2026/tangency_scan.py`, results in `tangency_scan/`.

Three independent witnesses are evaluated at each position:

* **Cartesian**, forced to the inverse-ray Cartesian grid at nbin 400,
* **polar**, forced to the inverse-ray polar grid at nbin 400,
* **contour**, VBMicrolensing at `RelTol = 1e-8`.

At each position the median of the three is taken and the witness furthest from
it is named the outlier, with the gap recorded relative to the median. A
position is a **dissent** when that gap exceeds 1e-4 relative — four times looser
than the tightest tolerance the campaign asks for anywhere, so a dissent is
always a real disagreement rather than a rounding artefact.

The scan swept 26 caustic distances (`d/rho` from 0.30 to 5.00) over the campaign
geometry families. **115 cases completed, 35,880 positions.**

An important detail, because it silently broke the first version of this scan:
within about one source radius of the caustic, an explicit request for the polar
grid falls back to Cartesian. Left uncorrected, that gives lcbinint's *one* grid
two of the three votes exactly where the disagreements live, and the median then
cannot dissent against it. The polar witness is dropped on epochs where the
method audit reports the fallback.

### The arbiter

`tests/diagnostics/recal2026/tangency_arbiter.py`, results in
`tangency_arbitration.json` / `.log` and `tangency_arbitration_vbm.json` / `.log`.

A majority of two is not a proof, and defect B below is precisely the case where
the majority is wrong. So disputed positions are re-decided by brute force:
composite Gauss–Legendre integration of `W(r) · A_point(x, y)` over the source
disk on a `48 / 96 / 192 / 384` panel ladder, up to 9,437,184 samples per
position and up to 770 s per position.

The arbiter **shares only the root solver with lcbinint and nothing at all with
VBM**. It has no grid, no flood fill, no seeding, no contour. Its own
convergence gap between the last two ladder rungs is reported with every verdict
and is the criterion for accepting it (`VERDICT_MARGIN = 0.1`, i.e. the verdict
stands only when the nearest witness is at least ten times closer to the arbiter
than the furthest).

## 2. The scan result

Dissents above 1e-4 relative, by distance band. `n` is positions scanned.

| d/rho band | n | Cartesian | polar | contour | worst Cartesian | worst polar | worst contour |
|---|---|---|---|---|---|---|---|
| 0.00–0.50 | 5264 | 17 | 27 | 33 | 2.28e-04 | 2.02e-04 | 5.00e+00 |
| 0.50–0.70 | 3028 | 9 | 14 | 19 | 2.70e-04 | 1.54e-04 | 1.82e+01 |
| 0.70–0.85 | 4358 | 11 | 41 | 32 | 9.40e-03 | 2.44e-04 | 4.76e+01 |
| 0.85–0.95 | 3292 | 12 | 42 | 38 | **2.21e-02** | 2.40e-04 | 4.65e+02 |
| **0.95–1.05** | **5288** | **90** | 51 | **225** | **1.53e-02** | 2.36e-04 | **6.13e+05** |
| 1.05–1.20 | 3984 | 0 | 0 | 11 | — | — | 5.07e-04 |
| 1.20–1.40 | 2556 | 0 | 0 | 2 | — | — | 5.10e-04 |
| 1.40–1.70 | 2290 | 0 | 0 | 0 | — | — | — |
| 1.70–2.20 | 2256 | 0 | 16 | 0 | — | 1.94e-04 | — |
| 2.20–3.00 | 1274 | 0 | 22 | 0 | — | 2.22e-04 | — |
| 3.00–inf | 2290 | 0 | 28 | 0 | — | 1.88e-04 | — |

Totals: Cartesian 139, polar 241, contour 360, over 35,880 positions.

Three structures are visible and they are different phenomena.

**The Cartesian column switches on and off with the caustic.** Zero dissents at
every band beyond `d/rho > 1.05`, then 90 in the single band straddling 1.0. Its
magnitude is also the largest of the three lcbinint-side columns by two orders
of magnitude. This is defect A.

**The polar column is flat and small.** It has a floor of 16–28 dissents even in
the far field where nothing is happening, all of them 1.9–2.4e-4. That is the
polar grid's ordinary quadrature error at nbin 400 in the far field, not a
defect — the reference there is simply tighter than the witness.

**The contour column is bimodal.** 11–13 dissents beyond `d/rho > 1.05` at
5e-4, and 225 in the `0.95–1.05` band with a worst case of `6.13e+05` relative
(at `q ≈ 2.4e-6`, `rho ≈ 3.0e-4`, `s = 0.268`, `A_point ≈ 3319`). §4 shows that
most of this column is **not** a VBM failure.

### The Cartesian dissents are all certified

Of the 139 Cartesian dissents, **136 carry `certified = true`** and every one of
the eight worst also carries `converged = true`. The self-reported error
estimate understates the true gap on 59 of the 139, by factors from 1.3 to
**411**:

| true gap | self-estimate | understated by | d/rho | certified | converged |
|---|---|---|---|---|---|
| 2.21e-02 | 2.45e-04 | 90× | 0.935 | yes | yes |
| 1.66e-02 | 2.94e-04 | 56× | 0.935 | yes | yes |
| 1.59e-02 | 3.88e-05 | 411× | 0.907 | yes | yes |
| 1.56e-02 | 4.65e-05 | 335× | 0.907 | yes | yes |
| 1.53e-02 | 6.23e-05 | 245× | 0.950 | yes | yes |
| 1.38e-02 | 1.78e-04 | 78× | 0.970 | yes | yes |

68 of the 139 exceed 1e-3. **A caller who asked for 1e-3 and checked the
returned estimate would have been told the answer was good to 2e-4 when it was
wrong by 2e-2.** That is a fail-open path, and it is the reason this file exists.

## 3. Defect A — the Cartesian row scan assumes one interval per row

**Confirmed.** 11 of 11 disputed epochs arbitrated; Cartesian was the party in
error in all 11.

### Evidence

`tangency_arbitration.log`, the 11 lcbinint-dissent epochs. Representative:

```
[1/11] s=1.2 q=1.0 rho=2.72e-01 d/rho~1.0 uniform   witness spread 9.50e-03
    lcbinint_cartesian     1.73084582857
    lcbinint_polar         1.75931816534
    vbm_contour            1.75938028916
    arbiter (384 panels)   1.75934221825   self gap 1.86e-05  -> decided
      lcbinint_polar         1.367e-05
      vbm_contour            2.164e-05
      lcbinint_cartesian     1.620e-02      <-- 750x the next worst
```

The arbiter agrees with the polar grid to 1.4e-5 and with VBM to 2.2e-5, while
the Cartesian grid is off by 1.6e-2. The pattern repeats across all 11: polar
and VBM cluster at 1e-6 to 2e-4 from the arbiter, Cartesian sits at 3.7e-3 to
1.6e-2. The routed pipeline inherits the error at both 1e-3 and 1e-4, because
the route it picks in this band *is* the Cartesian grid.

The geometries span `s ∈ [0.22, 1.2]`, `q ∈ [8.5e-5, 1.0]`,
`rho ∈ [1.5e-4, 0.27]` and both profiles. It is not a corner of parameter space.

### Mechanism

`src/lcbinint/magnification/finite_source_magnifier.cpp:1484`:

```cpp
struct LegacyImageAreaScratch {
    std::vector<double> xmin;
    std::vector<double> xmax;
    ...
};
```

One `xmin` and one `xmax` per row. **A row is a single interval by
construction** — there is nowhere to record a second one. When an image
component's intersection with a scan row is genuinely two disjoint intervals,
the area between them is counted as image.

The repair that exists (`finite_source_magnifier.cpp:3697–3790`, the
`dxmax > 1.1 * incr` family) detects a *between-row* jump in the interval
endpoints and re-seeds. It cannot help here: it fixes the case where the single
interval moved too far between adjacent rows, not the case where a single row
needs two intervals.

The tangency band is where a row needs two intervals, because that is where the
source limb grazes a fold and the image component develops a waist that pinches
through a scan row before the component itself separates. That is exactly the
`d/rho ∈ [0.85, 1.05]` window in the table, and exactly why the Cartesian column
is zero on both sides of it.

### Consequences

* The Cartesian grid is **not** a safe reference in `d/rho ∈ [0.85, 1.05]`, at
  any bin count. Raising nbin does not converge it — the defect is topological.
* The polar grid is unaffected. It is not the shape of the components that saves
  it but the scan direction, and the empirical record is 0 dissents above 3e-4
  in the entire tangency band against 90 for Cartesian.

### Options

1. **Structural fix.** Let a row hold a list of intervals rather than a pair of
   doubles. This is the correct fix and it touches the hot loop, the scratch
   layout, and the certificate's interaction with it.
2. **Route around it.** `d/rho ∈ [0.88, 1.02] → polar`. The caustic distance is
   already computed before the route is chosen, so this is implementable today,
   and §4 of `REPORT_speed.md` says polar is the cheaper grid at high
   magnification anyway. But it does not fix defect B, and B lives in an
   overlapping band.

No fix is applied in this campaign. Everything here is a measurement.

## 4. Defect B — both grids wrong together

**New, confirmed, and the more serious of the two**, because no internal
cross-check can see it.

### How it surfaced

The scan's contour column has 225 dissents in the `0.95–1.05` band. The obvious
reading is that VBM fails at tangency. The 12 largest were arbitrated to check
that reading, and **it was overturned in 9 of the 12.**

`tangency_arbitration_vbm.log`, case 4 of 12:

```
[4/12] s=0.3 q=1.0e-02 rho=2.01e-02 d/rho~1.1 uniform  witness spread 2.26e-03
    lcbinint_cartesian     50.8857655822
    lcbinint_polar         50.8857655822      <-- identical to 12 figures
    vbm_contour            50.7707268507
    arbiter (384 panels)   50.770722594   self gap 1.21e-06  -> decided
      vbm_contour            8.384e-08        <-- VBM was right
      lcbinint_cartesian     2.266e-03
      lcbinint_polar         2.266e-03
```

VBM agrees with the arbiter to eight significant figures. **Both lcbinint grids
are wrong by 2.3e-3, and they are wrong by the same amount, to twelve figures.**

Verdicts over the 12:

| verdict | count | gap: lcbinint | gap: VBM |
|---|---|---|---|
| **VBM right, both lcbinint grids wrong together** | **9** | 7.7e-4 – 2.3e-3 | 8.4e-08 – 6.5e-06 |
| VBM genuinely wrong | 3 | 3.0e-5 – 1.5e-4 | 3.5e-2 / 5.6e-2 / 8.4e-1 |

All 3 genuine VBM failures share one geometry: `s = 1.5`, `q = 1e-6`,
`d/rho ≈ 1.0` — a very low mass ratio at exact tangency. Those are real and are
the reason the contour witness needs the same scepticism as the others.

The 9 overturned positions sit at `d/rho ≈ 1.01–1.1`, across
`s ∈ {0.3, 0.9, 1.5, 2.0}`, `q ∈ [1e-5, 1e-2]`, `rho ∈ [2.0e-2, 7.4e-2]`, both
profiles.

### What it means

The two grids share the seeding stage, the component certificate, and the flood
fill. They differ only in the scan lattice. **A defect that produces the same
answer to twelve significant figures on both cannot live in the lattice — it is
upstream, in the shared part.** The candidates, in order of suspicion:

1. the flood fill claiming or failing to claim a component (see defect C, which
   is a proven order dependence in exactly that stage),
2. the seed set missing a component that both lattices then miss identically,
3. the certificate proving support for a region whose boundary is misplaced by
   the same amount on both lattices.

The error is ~1e-3 relative, i.e. *below* the 1e-2 accuracy the campaign's
loosest tolerance asks for and *above* the 1e-4 it asks for at its tightest. It
is therefore invisible at 1e-2, marginal at 1e-3, and a genuine failure at 1e-4.

### The consequence for the campaign's references

This is the part that must not be lost.

The speed campaign's per-block references are built by taking the finest
certified Cartesian value. **Defect A says that construction is unreliable in
`d/rho ∈ [0.85, 1.05]`. Defect B says that even cross-checking it against the
polar grid would not have caught it, because both grids fail together in an
overlapping band.** The reference is a party to the dispute, not a judge of it.

The scan's own contour column is likewise unreliable: 9 of 12 sampled dissents
were the *other* two witnesses being wrong in unison. The "contour 360 dissents"
figure in §2 should be read as *"360 positions where the two lcbinint grids
agreed with each other and VBM did not"*, which is a different and much weaker
statement.

What survives:

* **Aggregates over ~1300 blocks are not moved by ~140 disputed positions.**
  Every table in `REPORT_speed.md` stands.
* **No single block in `d/rho ∈ [0.85, 1.05]` should be quoted** from any file in
  this directory without arbitration.
* The `README.md` Stage 3 route audit's 1e-4 quadrature misses — 6 blocks, all
  at `A ≥ 39.8`, `d/rho ∈ [0.95, 1.7]` — are **the same phenomenon** and are now
  explained. They should be reclassified from "route boundary too loose" to
  "reference and grid both suspect in this band".

## 5. Defect C — the flood fill is not seed-order independent

Known, restated here because §4 makes it a suspect rather than a curiosity.
Committed as `891ca5e`.

The comment claiming the claimed-cell registry makes the filled area independent
of the seed set is **false**. A seed below `kFoldJacobianThreshold` has its fill
pinned to one side of the critical curve and still claims every cell it counts,
so *adding* a seed can *lower* the answer. 24 of 911 rows are non-monotonic in
the certified ladder's depth, invisibly to the certificate.

The heuristic probe rings mask this by claiming cells first, which is why
`REPORT_speed.md` §10 could delete only the interior rings and not the rest:
**dropping the remaining rings for speed has to wait for the fill's order
dependence to be fixed.** The rings are not buying correctness; they are buying
a particular order.

The overlap with defect B is the reason this belongs in the same file. Both live
in the shared stage upstream of the lattice; both produce answers the
certificate signs off on.

## 6. Recommended order of work

**B is the priority, but C is the way in.** B is the defect that matters — it is
upstream, it corrupts the references the rest of the campaign is measured
against, and it is invisible to every check currently in the code. But B's cause
is not yet isolated, and C is a *proven* order dependence in the single most
suspected stage. So:

1. **Fix defect C first**, as the entry point into B. It is already
   characterised, its mechanism is known, and if the fill's order dependence is
   B's cause then B disappears with it. Re-arbitrate the 9 overturned positions
   afterwards to find out.
2. **Then defect B**, on whatever survives. The 9 positions in
   `tangency_arbitration_vbm.json` are a ready-made regression corpus with an
   independent ground truth attached, so this is a bisection over the remaining
   two candidates (seed set, certificate boundary) rather than a search.
3. **Defect A last**, as a structural fix. It is well understood, it is
   contained, and `d/rho ∈ [0.88, 1.02] → polar` routes around it in the
   meantime at no measured cost.
4. **Then re-run the tangency scan**, which stopped at 115 of 120 cases, and
   re-derive the `d/rho ∈ [0.85, 1.05]` rows of the speed tables against
   references that are no longer party to the dispute.

For the paper, all three are honest limitations of the current implementation,
and the first two are limitations of the *certificate's scope* rather than of the
algorithm: it proves support and topology and it does what it claims. What is
missing is a check on the stage between the seeds and the lattice.

## 7. Reproduction

```bash
# three-witness scan
taskset -c 12-39 python -m tests.diagnostics.recal2026.tangency_scan \
    --output tests/diagnostics/results/recal2026/tangency_scan \
    --cases 60 --workers 28

# arbitrate the positions where lcbinint stood alone
python -m tests.diagnostics.recal2026.tangency_arbiter \
    --from-scan tests/diagnostics/results/recal2026/tangency_scan \
    --party cartesian --threshold 1e-3 --limit 11 \
    --out tests/diagnostics/results/recal2026/tangency_arbitration.json

# arbitrate the positions where VBM stood alone -- this is the one that
# overturned the scan
python -m tests.diagnostics.recal2026.tangency_arbiter \
    --from-scan tests/diagnostics/results/recal2026/tangency_scan \
    --party contour --threshold 1e-4 --limit 12 \
    --out tests/diagnostics/results/recal2026/tangency_arbitration_vbm.json
```

The arbiter is slow by design: 9.4M samples per position, 30 s to 770 s each.
Budget an hour per dozen positions.
