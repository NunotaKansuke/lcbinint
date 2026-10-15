# Large-source angular aliasing regression

## Reproduction

Base commit 1d6984f7501066f22ff0143d69c1cd1c45b04d06, branch experiment/algebraic_boundary. Parameters: s=.6, input q=.001, rho=.5, t0=0, tE=10, u0=-1e-5, alpha=.5, linear limb darkening=.5. Coordinates vbm; nbin auto; algebraic_boundary; atol=reltol=.001; 401 uniform epochs from -.8 to .8.

The derivative spike is exactly at t=-.48000000000000004 on this grid. All values had convergence flags and all gradients were finite even before the fix. No close-pair replay occurs in the neighboring 15 epochs, and the radial mesh has the same 52 evaluated nodes at every epoch. The source of this spike is distinct from the preceding close-D14-pair bug.

## Cause

One radial node, R=1.6095516093978897, takes the non-reciprocal angular chart with m=-.014217861161051171 and v=.18079566508308553. The nested angular rule starts by comparing one and three GC2 points. At this epoch those results accidentally agree:

| Angular order | Integral |
|---|---:|
| 1 | 2.1304521868868762 |
| 3 | 2.1304509553598034 |
| 7 | 2.1362042071645790 |
| 15 | 2.1362118117698992 |

The 1/3 difference satisfies the existing 5e-5 relative angular gate although the three-point result is wrong by about .27%. Neighboring epochs reject three points and use 15, making the derivative discontinuous. Forcing a seven-point minimum fixes this specific example, but the adopted fix does not unconditionally increase the minimum order.

See angular_baseline.log and angular_min7.log for the controlled intervention, and local_pairs.tsv.nodes for the offending node. Earlier diagnostic compilation attempts and the stopped full-grid VBM FD run do not contribute to the conclusions.

## Fix and mathematical scope

Only the three-point early exit receives an extra conservative check, based on the analytic structure of the existing angular integrand:

f(x)=sqrt(S(x))/(A(x)^(3/2)*sqrt(B(x))),

where S,A,B are quadratic in x and positive at x=0. Outward interval coefficients select a dyadic r>1 such that each factor on the complex circle |x|=r is p0*(1+e), |e|<=1/2. This excludes zeros of the factors throughout the disk and bounds |f| by M. Cauchy's coefficient bound gives a degree-five Taylor remainder at most M*r^-6/(1-r^-1) on [-1,1]. GC2(3) is exact through degree five, and both its positive weight sum and the integration measure have mass pi/2, so the quadrature error is at most pi times that remainder. The physical 2*v/rho prefactor is included.

The early exit is accepted only when this bound also meets the existing value tolerance gate. An inconclusive bound continues to the existing 7/15/31 ladder. It does not clip, interpolate, or smooth derivatives and has no time/parameter-specific switch. There is no new gradient tolerance or gradient-error-driven adaptation.

This bounds analytic truncation for the represented deflated integrand, not endpoint/quartic errors or the full numerical derivative. Later ladder levels retain their existing embedded estimator and fallback. FiniteUncertified status is preserved. The one-half factor bound is an internal enclosure choice, not an empirical gradient threshold.

## Results at unchanged tolerances

| t | s derivative before | s derivative after |
|---|---:|---:|
| -.484 | 1.82051943e-5 | 1.82051943e-5 |
| -.480 | 7.17547970e-4 | 1.83770499e-5 |
| -.476 | 1.85481590e-5 | 1.85481590e-5 |

Largest absolute second difference of the s derivative over all 401 epochs falls from .00139834 to 2.41212e-9. This is a discontinuity diagnostic, not an error bound. The gradient call's magnification at -.48 changes from 4.3887573706 to 4.3890762628, by .000318892; the former error could fit the loose flux tolerance while corrupting a small derivative.

The tighter-tolerance fused result is s derivative 1.98586082e-5 at tol=reltol=1e-6, leaving an absolute difference of about 1.48e-6 (7.46% of this very small component). Thus the isolated spike is removed; full derivative accuracy is not certified. Native value finite differences at strict tolerance have nonconverged flags and cannot certify that reference. VBM finite differences also vary strongly with step, including outliers even at Tol=RelTol=1e-10 (vbm_s_reference.csv); they are not treated as ground truth.

## Speed and regression scope

One pinned CPU, O2, identical workload, warmup and five repetitions, median API milliseconds for 401 epochs including seed preparation and parameter mapping:

| Case | Previous | Fixed |
|---|---:|---:|
| Requested rho=.5 | 342.77 | 341.62 |
| Ordinary | 58.83 | 60.09 |
| Current small-source notebook | 77.93 | 79.35 |
| Earlier broad-source problem | 450.31 | 451.40 |

The requested case has essentially unchanged runtime. Ordinary overhead is about 2%. At the six previously checked difficult historical epochs, u0 and s derivatives change by less than 4e-13 and 2e-14 respectively. Maximum magnification change in the other three curves is 3.44e-11.

All four 401-point curves have converged values and 2807 finite gradient components. Root screening on/off gives bitwise identical values and derivatives. The expanded matrix covers 120 configurations, including rho=.5 and t=-.48; all 120 values converge and all 600 native gradient components are finite, with no on/off differences.

The bound was checked on 10000 randomized deflated integrands: 8263 finite bounds contain the observed GC2(3)-versus-GC2(256) discrepancy allowing explicit floating-point evaluation noise; 1737 are inconclusive. The specific alias shortcut is excluded. See check_short_bound.cpp and bound_check.txt. This numerical check supports the bound implementation and does not establish a global solver guarantee.

## Files and build

The change is in benchmarks/holonomic/fused_gradient/fused_k.hpp. The default module has been rebuilt using:

```bash
bash benchmarks/holonomic/fused_gradient/build_algebraic.sh
```

Restart a running Python/Jupyter kernel to load it. The notebook was not edited by this task.

- reproduce.py: saved pre-fix module by default; `screen` loads the rebuilt default. The retained previous module is under gradient-probe/geometry-python; its hash is in provenance.json.
- evaluate.py: previous/screen/screen_off modes for all four curves.
- matrix.py: requires probe_large_fix compiled from gradient_screen_20261015/probe.cpp with the current API include tree, ONLY_VALUE_MESH, O2, the two existing python-api archives, GSL and quadmath (as in the prior run_final.sh).
- check_short_bound.cpp: include epoch_jacobian.hpp before fused_k.hpp as in the kernel; compile against fused-gradient/api-include, src, include, and quadmath.
- reference_s.py, diagnose_python.py: bounded FD/tolerance diagnostics, including their limitations.
- summary.json and provenance.json: results, source/module hashes and build identity. This is a dirty source change beyond the recorded base commit.
