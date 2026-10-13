# Vertical local quadrature: required additional caution

## Observed result

Native outer errors now expose cap signed residual/remainder, smooth-interior remainder, and junction/fallback residual/remainder separately through the opt-in retained-row trace. Physical flux and acceptance arithmetic remain unchanged. 166 selected adaptive calls,148 unchanged square adaptive calls and547 fixed calls reproduce saved numerical results exactly. Native unit suites pass2/2.

Three admission models were compared on the same166 saved independent-reference calls:

1. **Slack**: add a local quadrature indicator to the whole old component error, allocating only one eighth of remaining tolerance to panels (previous prototype).
2. **Signed transport**: update the component centre from S to S−Δ and retain R; the local quadrature defect is used for scheduling, not added again to admission.
3. **Absolute transport**: retain the old absolute indicator B, admit only if B+Σcomponent|Δ|≤tolerance. First try sparse panels; refine only when this admission fails. No external reference enters scheduling/admission.

Δ is the change in the stored numerical core integral, including the removal/restoration of the horizontal Bennett phase. Physical tip integration remains unchanged. The old core sum is available from retained rows; no comparison of whole magnifications at successive nbin is used.

Final replay v3: all three methods pass166/166 actual accuracy targets and estimated acceptance checks, with zero false acceptance. Uniform and linear LD use identical scheduling/admission code.

| tolerance | profile | slack core-row reduction | absolute transport core-row reduction |
|---|---|---:|---:|
|1e-3|uniform|38.0%|69.8%|
|1e-3|c=.5|9.3%|49.6%|
|1e-3|c=1|9.0%|35.0%|
|1e-4|uniform|38.2%|53.1%|
|1e-4|c=.5|14.1%|34.8%|
|1e-4|c=1|3.6%|5.9%|

At1e-4 absolute-transport maximum actual errors are9.53702e-5,9.56062e-5,9.86504e-5 respectively. Sparse-first retries occur in12/30,14/28,24/25 calls. The last profile has little affordable coarse-core perturbation at this tolerance.

These are **offline node-demand figures**, not measured runtime gains, lens-map savings or cold-start adaptive savings. Every underlying fine row was already produced by the native baseline. Tip/ineligible rows are excluded from core-row reductions. Retained fine sums cannot be assumed available before a live adaptive calculation has actually produced them. Runtime implementation and matched fixed comparisons remain outstanding.

## What is mathematically justified

Let F_old be a retained numerical integral and E_old=F_true−F_old. Change its stored value by a known numerical Δ:

E_new = E_old − Δ.

If E_old is enclosed by [S−R,S+R], the new enclosure is [S−Δ−R,S−Δ+R]. A further local quadrature-error allowance would double count uncertainty already enclosed by the old interval. Only numerical uncertainty in computing Δ and rounding of the interval arithmetic must be added. The native helper `retained_flux_envelope.hpp` implements this transport with outward rounding after each interval operation; invalid/nonfinite inputs fail closed. Its test checks10000 deterministic random enclosures against long-double endpoints, invalid cases and cancellation.

If only |E_old|≤B is known, the guaranteed absolute transport is |E_new|≤B+|Δ|. This extra |Δ| is necessary in the absence of sign information. It is generally smaller/more relevant than adding an unrelated Simpson smoothness defect; Δ is a change in the represented flux itself. Cancellation between panel changes inside one component is valid only when their total Δ is actually computed and retained, rather than guessed.

**These proofs transport an existing valid bound; they do not establish the original bound.** The helper is not yet wired into native mixed-spacing quadrature.

## Why the signed model is not a certificate

The original [S−R,S+R] misses the reference residual on25/166 calls (uniform6,c=.5 thirteen,c=1 six). Maximum envelope excess:7.74958e-5 at tolerance1e-3; at1e-4 maximum4.97443e-6. Reversing the residual sign makes agreement much worse, so this is not a simple sign-convention mismatch.

The original absolute indicator also underestimates actual error on three calls:

| profile | tolerance | actual error | original indicator |
|---|---:|---:|---:|
|uniform|1e-4|7.23135e-5|7.20656e-5|
|c=1|1e-3|5.10431e-4|4.32935e-4|
|c=1|1e-4|7.55782e-5|7.17884e-5|

All remain inside their requested tolerances. These are estimator-undercoverage cases, not newly introduced tolerance failures. We do not fit additional factors to these geometries or claim a rigorous certificate from the166 successes. The observed matrix supports target accuracy; a universal mathematical guarantee remains unavailable.

A bound B cannot be called guaranteed when it already undercovers. Likewise, signed transport cannot be promoted merely because its final replay outputs pass. A scalar adversarial perturbation can turn an inherited undercoverage into false acceptance; this is independent of whether that particular perturbation occurs in this replay.

## Root cause and next native design

Removing the old smooth-interior remainder alone is insufficient: at1e-4 its share of the positive inner+cap+smooth+junction remainders is about4.8% for uniform and1.9–2.0% for LD. The larger terms are horizontal, topology and physical-cap uncertainties. Those cannot be discarded because an interior panel changed.

There is also a fundamental limitation: B_new=B_old+|Δ| cannot make an initially unaccepted component (B_old>tolerance) acceptable. It safely compresses already accepted retained information, but cannot replace refinement of an unresolved component. Therefore this replay is not a shortcut to declaring live adaptive integration complete.

The live path must:

- Partition at actual retained row interfaces. Keep Bennett weights exclusively at physical tips; an artificial core/band boundary is not a new image tip.
- Retain the old **interior** quadrature and its own uncertainty, rather than the whole component B.
- Recompute error terms only for physical cap/junction bands that were refined; replace their prior contribution instead of adding it a second time.
- Carry the horizontal error of retained rows with correct physical y weights. LD horizontal moment uncertainty survives vertical changes.
- Apply interval transport only to an actually known old/new local numerical integral difference. Unknown new row contributions still need their own model/enclosure.
- Reuse all old mapped points, roots, panels and stored sums on partial failure. Never obtain the missing fine core sum by secretly scanning the whole refined component.

The decomposition and bound-transport helper are implemented. Native mixed-grid integration and general error certification are unfinished. No production/default promotion, commit or push.

## Reproduction

Single-thread commands, project-local build artifacts:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support \
/home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.vertical_error_accounting_audit --output vertical-error-accounting-v3
```

Final:`tests/diagnostics/results/recal2026/vertical-error-accounting-v3/{summary.json,replay.json,source/}`. Regression/native unit results and full envelope-undercoverage audit:`vertical-error-accounting-v2/`. First additive/signed/absolute variants retained in v1/v2. Native hash and source hashes are saved. No builds during runtime comparisons; no runtime comparison was claimed here.
