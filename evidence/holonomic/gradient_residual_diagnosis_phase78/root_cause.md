# Phase 78: caustic LD gradient residual diagnosis

## Finding

For reference case row 17 (`caustic-cross`, `xs=0.12`, `ys=0`, `rho=0.02`, `q=0.5`, `a=1`, non-barycentric), the remaining X-gradient error is dominated by the fixed-R angular derivative quadrature in radial cell 6. This is a local inner-integration resolution problem; it is not explained by the radial scheduler, a missing projective-fold map, or the analytic `dphi/dX` formula.

Cell 6 spans `R=[0.84972672439230301, 0.85720378575239498]`, ending at the certified projective fold. Across its 1,536 radial nodes, the K-rule rejects all 1,536 candidates for the existing 8-vs-16 **value** disagreement gate. None fail the V-floor or TMax checks. The derivative therefore uses the fixed 64-node angular rescue throughout this cell.

That rescue integrates `phi_X/(2 sqrt(phi))` over a linear angular interval using 64 nodes and estimates inner error from 64-vs-32. Near an angular boundary, `phi` vanishes linearly, so this derivative integrand has an integrable inverse-square-root endpoint singularity. The low-order rescue converges slowly there. The value integrand `sqrt(phi)` is much smoother, which is why value convergence does not imply gradient convergence.

## Matched radial-node angular-resolution experiment

The diagnostic radial grid was fixed at GL192 with 8 subdivisions per cell. Event cuts, fold maps, endpoint/root construction, radial nodes, and all cells except cell 6 were held fixed. Only cell 6's angular derivative integration was recomputed with more nodes. The right column replaces the incumbent cell-6 contribution in the all-cell pure-LD radial-gradient sum; this is a matched quadrature diagnostic, not an adaptive whole-epoch timing or status result.

| Angular nodes in cell 6 | All-cell dmu/dX on the fixed radial grid |
|---:|---:|
| 32 | 2.6301069678 |
| 64 (incumbent) | 2.1432404680 |
| 128 | 2.0214117163 |
| 256 | 1.9909474690 |
| 512 | 1.9833309652 |
| 1024 | 1.9814268116 |
| 2048 | 1.9809507714 |

The independent high-resolution value finite difference gives pure-LD `dmu/dX = 1.9807892100` (`GL160 x 8 radial`, 512 angular nodes, `h/rho=2.5e-4`). The 64-node result is high by `0.162451258` (8.20%). Replacing only cell 6 with 2048 angular nodes reduces the difference to `0.000161561` (0.00816%). A diagnostic second-order Richardson extrapolation of the 1024/2048 sequence is `1.9807920914`, within `2.9e-6` of the independent FD. The extrapolation is diagnostic evidence, not a production error certificate.

For mixed `u=0.5`, the normalized gradient is `0.6*g_uniform + 0.4*g_pureLD`. The uniform component independently agrees at about `1e-9`; the LD component explains the observed mixed discrepancy. The 64-vs-32 whole-gradient difference in this cell is `0.486866500`, while the 64-node error against the independent FD is `0.162451258`: a ratio of `2.997`. This agrees with an approximately second-order angular tail and shows that the current inner ledger is conservative rather than falsely certifying the result. The solver correctly returns `FiniteUncertified` / `InnerAccuracyLimited` for the gradient.

The earlier QF fixed-R radial diagnostic with 128 angular nodes gave `2.0802322`; it was not angularly converged and must not be treated as the final derivative oracle. The high-resolution value FD is the independent target here.

## Why the other probes did not fix it

- Continuing radial-only refinement reduced radial error to about `1.6e-5`, but left the pure-LD gradient at `2.1432405` and inner error near `0.4869`.
- The 24-node K-Jacobian shadow made no material change: this cell fails the preceding value 8-vs-16 gate, so it never enters the K-Jacobian derivative path.
- The uniform gradient is already accurate; the residual is in the limb-darkening angular derivative.
- No production tolerance, quality gate, router, or evaluator was changed in this diagnosis.

## Interpretation and next research target

The analytic fixed-R derivative is consistent with the independent pointwise derivative checks away from the extreme endpoint. The remaining error comes from under-resolving its endpoint singularity in the 64-node fallback. The current 32-vs-64 estimate detects the problem and correctly prevents `ToleranceMet`; it does not make the finite returned value accurate enough by itself.

A focused next experiment should replace the fixed low-order rescue only on nodes/cells where the K-rule is rejected, using an endpoint-aware angular representation or a locally increased rule with an explicit nested error gate. The tests should preserve the incumbent value path and compare against the same independent high-resolution FD. A global 2048-node rule is only a diagnostic reference, not a proposed production setting. The sine-map/low-order Gauss-Legendre shadow tested here did not converge cheaply enough near the closest fold nodes, so that version is not yet a solution.

This finding is established for the caustic-cross X-gradient case only; it does not establish that every remaining gradient residual has the same cause.

## Reproduction

Run from the isolated worktree. This builds the diagnostic-only shadow, pins it to CPU 0, and writes the raw TSV files in this directory. No runtime `HOLO_*`/`D14_*` environment overrides were set for the recorded run.

```bash
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc \
  -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION \
  -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF \
  -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE \
  -DHOLO_ADAPTIVE_SPARSE_ANCHOR -DHOLO_ADAPTIVE_SAMPLE_GROWTH \
  -DHOLO_MV_RATIONAL_MIDPOINT -DHOLO_QUARTIC_WARM_SOLVE_TOL=1e-11 \
  -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED \
  -DHOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH \
  -march=native -funroll-loops -ffp-contract=fast -fno-math-errno \
  evidence/holonomic/gradient_residual_diagnosis_phase78/direct_radial_gradient_ld_x.cpp \
  -lquadmath -o /tmp/phase78_cell6_diag

taskset -c 0 /tmp/phase78_cell6_diag \
  evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv \
  evidence/holonomic/gradient_residual_diagnosis_phase78/direct_radial_gradient_profile.tsv
```

Primary raw artifacts:

- `component_adaptive.tsv`
- `component_radial_only.tsv`
- `projective_fd_highres.tsv`
- `direct_radial_gradient_profile.tsv`
- `direct_radial_gradient_profile.tsv.cells.tsv`
- `direct_radial_gradient_profile.tsv.events.tsv`
- `direct_radial_gradient_profile.tsv.cell6_local.tsv`
- `direct_radial_gradient_profile.tsv.cell6_angular.tsv`
- `direct_radial_gradient_profile.tsv.cell6_integral.tsv`
- `direct_radial_gradient_profile.tsv.cell6_sine_integral.tsv`
