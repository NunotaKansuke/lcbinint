# General local radial-pair prototype (experimental; not adopted)

## Result

A case-independent detector and local integration prototype now reproduce the
previous oracle-map improvement. All runs use mu_atol=mu_rtol=1e-3. It does
not add a derivative tolerance or use measured derivative error for adaptation.
The default gradient module and notebook were not changed in this investigation.

|t (days)|existing u0 gradient|candidate|independent VBM FD median|
|---|---|---|---|
|-.7|-5.7882218|.3932438|.3932103|
|-.696|-9.7891368|.3930472|.3943238|
|+.78|4.4205790|.4319449|.4319976|

FD step ranges and rho/q/s comparisons are in comparison.csv. These are
comparisons, not gradient-error certifications. The existing strict native
FD had nonconverged flags and is not used as certified truth.

A scan of 89 epochs covers -.706 to -.690 with .00025-day spacing, .775 to
.785 with .0005-day spacing, plus three control epochs. The final candidate
has 623/623 finite gradients. The original u0 gradient ranges from about
-10.95 to +2.35 around -.7; the candidate removes those large spikes.
Finiteness and apparent smoothness do not prove accuracy at every scan point.
An attempted dense VBM FD reference was stopped after about four CPU minutes
without a completed output. No dense independent-FD accuracy claim is made.
Completed independent references at the seven original epochs are preserved.

## Method

Reusable implementation:

- benchmarks/holonomic/fused_gradient/fused_geometry.hpp
- benchmarks/holonomic/fused_gradient/fused_geometry_pair.inc
- benchmarks/holonomic/fused_gradient/build_geometry_overlay.py
- benchmarks/holonomic/fused_gradient/build_geometry.sh

1. Finish the ordinary value mesh and observed-count repairs first.
2. Within occupied bands isolate real stationary points of D14(R^2), using
   interval Bernstein/Descartes subdivision, binary128 retry, and a bounded
   Sturm fallback. This can find a narrow pair without a node inside it.
3. Propose two roots from the quadratic normal form; polish the actual D14
   roots and require interval Rouche dominance for an isolated two-root disk.
   This certifies the disk's root count, not complete image support or every
   downstream gradient error. Existing physical fold pairs are currently
   skipped. Higher-order clusters remain outside this normal form.
4. Add both missing real folds with the existing physical local fold solve.
   Reintegrate only when a new dangerous pair is found. Complex pairs use
   R=Rc+b*sinh(y) on the affected occupied band, with its exact Jacobian.
5. Between two newly inserted real folds, use a seven-node mixed moment rule
   exact for A(x)+sqrt(1-x*x)*B(x), with A,B polynomials through degree three.
   It integrates the regular background and the fold square root together.
   The same rule applies to every input; its weights are mathematical moments,
   not fitted to these epochs. Its absolute weight sum is about 2.03034.
6. Retain the all-span material derivative, including moving physical folds
   and cancellation at artificial cuts. No smoothing or clipping of output
   gradients is used. DERIVATION.md states the differentiation identity.

The seven mixed nodes are cos(k*pi/8). Symmetry enforces odd moments; the
remaining weights solve the exact moments 2, 2/3, pi/2, pi/8 for
1, x^2, sqrt(1-x^2), x^2*sqrt(1-x^2). See mixed_rule_check.json.

## Controlled failures that informed the final prototype

- Local Newton proposals alone missed the target pair; retained snapshots
  document this. Interval stationary-root isolation replaced them.
- Early detection allocated work to folds that the ordinary value repair
  would discover itself. Moving detection after value repair fixed the
  current-setting invalid point at t=-.004.
- High-order fold-mapped Gauss nodes approached real endpoints too closely.
  Five dense-scan epochs failed InnerPhiNonpositive, not detector isolation.
  The mixed rule keeps its physical nodes away from the rounded endpoint.
- A square-root-only split rule returned finite gradients but biased the
  regular background (e.g. -.696 u0 became .34264). This is NOT adopted.
  Its artifacts are explicitly prefixed split_trial_. The mixed rule restored
  .393047 while keeping all scan epochs finite.

## Speed and acceptance

Pinned to one allowed CPU, O2, same native archives, same inputs/tolerances,
seed preparation and API mapping included. Warmup then three timed repetitions,
401 epochs per curve, median wall time:

|case|existing|candidate|ratio|
|---|---:|---:|---:|
|historical q=1e-4, rho=.05|234.82 ms|376.25 ms|1.60|
|current q=1e-3, rho=.0005|58.54 ms|125.25 ms|2.14|

Both settings: 401/401 values converge, 2807/2807 gradients finite. The current
setting's 401 gradients are bitwise unchanged; its added cost is detection.
This does NOT meet the previous roughly-equal-speed target. The prototype is
therefore kept separate from the default module. No precision/support flags
are promoted and no production default is changed.

Remaining accuracy: -.704 still has ~.00167 u0 error, -.696 ~.00128, and -.68
retains the original ~.0218 error because its pair is already in the topology
and currently skipped. The local normal form is useful but not a full repair.

Next technical work is to use the accepted mesh's root geometry to reuse or
exclude already-resolved pairs, and to subtract/integrate the isolated singular
term while leaving the regular background on the original value mesh. This
must cover already-known close folds as well as missing folds. Global per-epoch
stationary enumeration is the main avoidable cost on the current setting.
A complete cosh/sin continuation or analytic limiting form at exact coalescence
is still needed; an arbitrary positive floor for b is not an acceptable fix.

## Reproduction

From the specified existing checkout (no new worktree):

    bash benchmarks/holonomic/fused_gradient/build_geometry.sh
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 evidence/holonomic/gradient_geometry_20261015/evaluate.py reference
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 evidence/holonomic/gradient_geometry_20261015/evaluate.py candidate
    PYTHONDONTWRITEBYTECODE=1 python3 evidence/holonomic/gradient_geometry_20261015/analyze.py

The experimental Python module is in
build-algebraic-boundary/gradient-probe/geometry-python; the default stays in
build-algebraic-boundary/fused-gradient/python. Compiler scratch is project-local.
Do not set LCBININT_RUN_EXPENSIVE_FD unless intentionally running the additional
VBM step study; bound that separate run and use its partial checkpoints.
