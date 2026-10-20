# Return computed derivative candidates independently of certification

## Change

The lcbinint C++ implementation in
`src/lcbinint/magnification/algebraic_boundary/fused_gradient.cpp` now retains
finite computed regular-radial derivative candidates even when the geometry
guard cannot certify them. `gradient_valid` remains false, and the stop text
records the regular-radial candidate, original failure and unresolved reason.
The candidate's value error is retained with its recovered value rather than
leaving the original infinite error attached to it.

The same candidate-export rule also covers the contact-retry rejection branch.
The prepared-support early return now preserves existing numerical derivatives,
value error and diagnostics instead of constructing a fresh zero-filled record.
No constant, artificial nonzero gradient or finite-difference derivative was
introduced. Genuine computed zero derivatives are not prohibited. This is an
export-policy correction, not a certificate of derivative accuracy or a solver
wide audit of every possible failure.

## Reproduced result

At the previously zero-gradient static point, the local derivatives are now:

```
xs           42552153555.27184
ys           -9190385671.338346
rho          19319360648.88178
q_internal   11.177209980384188
separation   -1148496.3863479905
```

Magnification remains 161148.03302349045. `gradient_valid` remains false because
`unresolved_outer_edge_pair` is still true. The fallback's finite estimated value
error is now preserved, so value_converged is true; this does not validate the
gradient or certify support. JAX reverse-mode differentiation returns the same
rho candidate, 19319360648.88178, rather than zero. Its local API and transformed
physical-coordinate gradients were both checked.

## Validation and build

- C++ unit_core, including the original four value-repair cases and the new
  finite/nonzero-but-unreliable gradient case: passed.
- All 38 public unified-algebraic regression tests: passed, including the new
  native/public/JAX export test.
- Scoped `git diff --check`: passed.

The separate build is
`/moao38_7/nunota/binfit/work/MB23169/runs/nuts_static_500w100s_depth10/gradient_candidate_repair_build/lcbinint`.
Native SHA256:
`80db156d472e6d8f965fbc8c93c7c4519d69e5a3b1f714212ba4eeaf77bb3389`.
Both original and value-only-repair diagnostic binaries were preserved.
No default package switch, commit, push or NUTS relaunch was performed.
No claim is made that the event's nearby finite-difference check or production
NUTS convergence has passed; scientific_acceptance remains false.

Reproduce with a new output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=true \
python tool/validate_gradient_candidate_export.py \
  --package /moao38_7/nunota/binfit/work/MB23169/runs/nuts_static_500w100s_depth10/gradient_candidate_repair_build/lcbinint \
  --output NEW_OUTPUT_DIRECTORY
```

The first validation attempt omitted `with_diagnostics=True` in the new test
and runner, so accessing the intentionally omitted diagnostics failed. That
test-harness mistake was fixed without modifying the solver; the successful
run and all hashes are in this directory's summary.json and the parent
gradient_candidate_repair_checked.log. The earlier failed log is preserved.
