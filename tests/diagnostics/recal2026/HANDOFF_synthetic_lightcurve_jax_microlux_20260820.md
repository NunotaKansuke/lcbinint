# Handoff: compiled synthetic light-curve comparison

Date: 2026-08-20
Primary result: [value_and_grad results](../results/recal2026/synthetic_lightcurve_jax_microlux_value_and_grad_final_20260820/results.json)
Generated report: [REPORT.md](../results/recal2026/synthetic_lightcurve_jax_microlux_value_and_grad_final_20260820/REPORT.md)
Benchmark harness: [benchmark_synthetic_lightcurve_jax_microlux.py](benchmark_synthetic_lightcurve_jax_microlux.py)

This is the paper-facing handoff for the six synthetic binary-lens light-curve
cases. The final result was regenerated after adding a true model-only
value-and-gradient measurement. Native lcbinint, JAX no-warm, JAX warm,
microLUX, and the VBM reference were all handled by one final harness
invocation; no timing lane was copied from an earlier run.

## Executive summary

The comparison is geometry- and profile-dependent. JAX is not uniformly faster
for the forward magnification, but its compiled reverse-mode model-gradient
throughput is consistently favorable in this corpus.

- Forward T_A: warm JAX is faster than microLUX in 3/6 Uniform and 5/6
  Linear-LD cases. The median t_microLUX/t_JAX,warm ratios are 0.9104 and 2.141.
- Full physical Jacobian T_J: warm JAX is faster in 4/6 Uniform and 5/6
  Linear-LD cases. The median ratios are 2.350 and 3.149.
- Primary value-plus-gradient metric T_VG: warm JAX is faster in 4/6 Uniform
  and 6/6 Linear-LD cases. The median ratios are 5.374 and 4.930.
- The corrected warm-up route is faster than the no-warm automatic route in
  all 12 conditions for T_A, T_J, T_VG, and the retained diagnostic VJP.

T_VG returns both a scalar model contraction and its parameter gradient in one
jax.value_and_grad call. It is therefore a better model-level proxy for a
value-plus-gradient evaluation than the earlier VJP-only timing. It is still
not an end-to-end HMC or likelihood timing because the scalar is a
data-independent model contraction rather than a log likelihood.

## Scope and common protocol

The six cases and time grids are inherited from the paper-facing synthetic
benchmark:

| case | time interval | epochs |
|---|---:|---:|
| resonant_high_mag | [-0.4, 0.4] | 240 |
| resonant_large_source | [-0.4, 0.4] | 240 |
| close_binary | [-0.5, 0.5] | 240 |
| high_q | [-1.0, 1.0] | 240 |
| wide_planet | [-2.8, 0.8] | 600 |
| close_secondary_caustics | [-1.2, 1.2] | 400 |

Both source profiles were evaluated:

| profile | limb-darkening coefficient |
|---|---:|
| Uniform | c=0 |
| Linear LD | c=0.5 |

The common numerical target was tol = retol = 1e-3. microLUX used the
event-specific annulus counts selected by the VBM reference run. Uniform
curves use one n_annuli=10 group. Linear-LD curves are grouped by equal
annulus count and each static group is compiled in an isolated worker; group
steady-state costs are summed to form one complete light curve.

The JAX no-warm lane includes its first-call XLA compilation in the reported
cold timing, but not in the steady-state median. The warm lane obtains its
route and nbin proposal from the automatic JAX dispatcher, certifies the fixed
plan against the native self-converged reference, primes the support cache, and
compiles the fixed-plan transforms before timing. Warm-up and compilation are
excluded from all steady-state values.

## Definitions of the measurements

Let A_i(theta)=A(t_i; theta), i=1,...,N, be the model magnification curve.

### Forward timing T_A

This is one complete batched evaluation of A=(A_1,...,A_N). The report
normalizes this timing to ms/epoch. It is not a Python loop that invokes an
integration routine once per epoch: both backends receive arrays of epochs,
although their internal batching and integration graphs differ.

### Full physical Jacobian T_J

G_ik = d A_i / d theta_k, with G in R^(N x 7), and physical parameter order

    (s, q, rho, u0, alpha, t0, tE)

Both backends use jax.jacfwd for this diagnostic. The timing is for one
complete N x 7 Jacobian curve, not one epoch and not one scalar derivative.
It measures all seven parameter columns explicitly and is useful for AD
throughput diagnostics, but it is not the most inference-like reverse-mode
operation.

### Primary value-plus-gradient timing T_VG

To avoid observational data while measuring a value and a gradient together,
define a deterministic model-space cotangent

w_i proportional to sin(phi_i) + 0.37 cos(3 phi_i) + 0.19 sin(7 phi_i),
phi_i = 2 pi (i+1/2)/N.

The timed function is

S(eta) = w^T A(eta),
(S, grad_eta S) = value_and_grad(S),

in sampler coordinates

    (log_s, log_q, u0, alpha, log_rho, log_tE, t0).

The cotangent is generated from the time-grid length only and lies outside the
timed region. No data, uncertainty, residual, or likelihood is used. This
metric includes the scalar value and gradient together, but should be
described as a model-only value-plus-gradient proxy rather than as the full
HMC cost.

### Retained diagnostic T_VJP

The old VJP timing measures only

J_eta^T w = grad_eta [w^T A(eta)],

without returning the scalar S. It is retained in REPORT.md for implementation
diagnostics, but it is not the primary derivative-performance claim. T_VG
supersedes it for the paper-facing value-plus-gradient comparison.

## Forward steady-state result T_A

Times are ms/epoch. The native column is the warm native reference; JAX and
microLUX values are compiled steady-state medians.

| case | profile | native warm | JAX no | JAX warm | microLUX | n_annuli max |
|---|---|---:|---:|---:|---:|---:|
| resonant high-mag | Uniform | 0.094016 | 3.7012 | 2.4286 | 0.55491 | 10 |
| resonant high-mag | Linear LD | 0.10271 | 4.1763 | 2.8371 | 3.0398 | 21 |
| resonant large-source | Uniform | 0.13535 | 5.1970 | 3.4059 | 0.86947 | 10 |
| resonant large-source | Linear LD | 0.14452 | 6.6407 | 4.3228 | 7.3122 | 20 |
| close binary | Uniform | 0.015831 | 0.32036 | 0.21425 | 0.14429 | 10 |
| close binary | Linear LD | 0.016734 | 0.56351 | 0.37708 | 0.18814 | 5 |
| high q | Uniform | 0.016104 | 0.19914 | 0.13301 | 0.18546 | 10 |
| high q | Linear LD | 0.016091 | 0.21118 | 0.14707 | 0.38112 | 14 |
| wide planet | Uniform | 0.0046868 | 0.056394 | 0.037807 | 0.043377 | 10 |
| wide planet | Linear LD | 0.0048382 | 0.059954 | 0.041813 | 0.14564 | 17 |
| close-secondary caustics | Uniform | 0.0085564 | 0.13754 | 0.081252 | 0.10480 | 10 |
| close-secondary caustics | Linear LD | 0.0085925 | 0.14920 | 0.091935 | 0.40737 | 21 |

Aggregate warm-lane result:

| profile | JAX faster than microLUX | median t_microLUX/t_JAX,warm |
|---|---:|---:|
| Uniform | 3/6 | 0.9104 |
| Linear LD | 5/6 | 2.141 |

## Full-Jacobian result T_J

Times are ms per complete N x 7 Jacobian curve.

| case | profile | JAX no | JAX warm | microLUX | micro/no | micro/warm |
|---|---|---:|---:|---:|---:|---:|
| resonant high-mag | Uniform | 1821.8 | 1360.9 | 559.99 | 0.3074 | 0.4115 |
| resonant high-mag | Linear LD | 2767.8 | 1960.5 | 1996.8 | 0.7214 | 1.019 |
| resonant large-source | Uniform | 2507.5 | 1874.9 | 684.25 | 0.2729 | 0.3650 |
| resonant large-source | Linear LD | 4656.5 | 3163.7 | 4688.1 | 1.007 | 1.482 |
| close binary | Uniform | 135.43 | 109.37 | 175.96 | 1.299 | 1.609 |
| close binary | Linear LD | 382.18 | 281.89 | 204.27 | 0.5345 | 0.7246 |
| high q | Uniform | 79.200 | 62.798 | 246.71 | 3.115 | 3.929 |
| high q | Linear LD | 115.79 | 86.031 | 414.38 | 3.579 | 4.817 |
| wide planet | Uniform | 56.393 | 44.750 | 138.37 | 2.454 | 3.092 |
| wide planet | Linear LD | 84.017 | 62.351 | 303.96 | 3.618 | 4.875 |
| close-secondary caustics | Uniform | 81.800 | 58.963 | 212.42 | 2.597 | 3.603 |
| close-secondary caustics | Linear LD | 126.57 | 88.116 | 565.84 | 4.471 | 6.422 |

Aggregate warm-lane result:

| profile | JAX faster, no warm | JAX faster, warm | median micro/no | median micro/warm |
|---|---:|---:|---:|---:|
| Uniform | 4/6 | 4/6 | 1.876 | 2.350 |
| Linear LD | 4/6 | 5/6 | 2.293 | 3.149 |

## Primary value-plus-gradient result T_VG

Times are ms per complete model curve, with both S and grad_eta S returned
by one value_and_grad transform.

| case | profile | JAX no | JAX warm | microLUX | micro/no | micro/warm |
|---|---|---:|---:|---:|---:|---:|
| resonant high-mag | Uniform | 1825.1 | 1357.8 | 526.65 | 0.2886 | 0.3879 |
| resonant high-mag | Linear LD | 2770.8 | 1959.3 | 2037.4 | 0.7353 | 1.040 |
| resonant large-source | Uniform | 2509.9 | 1872.7 | 645.11 | 0.2570 | 0.3445 |
| resonant large-source | Linear LD | 4656.2 | 3162.3 | 4891.5 | 1.051 | 1.547 |
| close binary | Uniform | 138.22 | 109.08 | 399.04 | 2.887 | 3.658 |
| close binary | Linear LD | 383.52 | 281.80 | 470.47 | 1.227 | 1.670 |
| high q | Uniform | 80.839 | 62.225 | 441.19 | 5.458 | 7.090 |
| high q | Linear LD | 117.18 | 86.069 | 704.93 | 6.016 | 8.190 |
| wide planet | Uniform | 61.501 | 44.843 | 881.93 | 14.34 | 19.67 |
| wide planet | Linear LD | 86.708 | 62.196 | 1027.0 | 11.84 | 16.51 |
| close-secondary caustics | Uniform | 84.160 | 58.678 | 665.34 | 7.906 | 11.34 |
| close-secondary caustics | Linear LD | 129.57 | 87.834 | 1085.7 | 8.379 | 12.36 |

Aggregate T_VG result:

| profile | JAX faster, no warm | JAX faster, warm | median micro/no | median micro/warm |
|---|---:|---:|---:|---:|
| Uniform | 4/6 | 4/6 | 4.172 | 5.374 |
| Linear LD | 5/6 | 6/6 | 3.621 | 4.930 |

A ratio larger than one means that microLUX takes longer than JAX. The
Linear-LD result is particularly clear: after warm-up, all six microLUX/JAX
ratios exceed one for this model-only value-plus-gradient operation.

## Diagnostic VJP result

The VJP table remains in REPORT.md for debugging and comparison with the
earlier benchmark, but it is not the primary paper metric because it returns
only the pullback and omits the scalar value.

| profile | JAX faster, no warm | JAX faster, warm | median micro/no | median micro/warm |
|---|---:|---:|---:|---:|
| Uniform | 4/6 | 4/6 | 4.214 | 5.399 |
| Linear LD | 5/6 | 6/6 | 3.611 | 4.929 |

The close agreement between VJP and T_VG is expected: the same model
contraction and cotangent are used, while T_VG additionally returns the scalar
value. The paper should cite T_VG, not the VJP-only numbers, for
value-plus-gradient performance.

## Why the backend ranking differs by metric

The two implementations are compared at matched numerical tolerance, not under
an artificially identical integration graph.

### lcbinint JAX

The public JAX light-curve call receives the full epoch array and constructs a
vectorized source trajectory. The automatic dispatcher proposes a per-epoch
finite-source route and resolution. The warm-up certifies that proposal
against the native self-converged reference and then compiles the fixed plan.
Repeated calls reuse a static, batched graph rather than repeating route
selection or convergence work.

The derivative transform is compiled around this same batched plan. For reverse
mode, value_and_grad traverses the whole model curve graph once and returns the
scalar contraction and its sampler-coordinate pullback together.

### microLUX

The harness calls the public trajectory-array API for each static annulus group;
it is not a Python loop that invokes the integration routine once per epoch.
However, the microLUX graph includes mapped epoch work, additional annulus
mapping for limb darkening, and an adaptive contour-integral state. Reverse-mode
AD differentiates that mapped adaptive graph, which need not have the same cost
profile as its forward evaluation or as the batched lcbinint FFI path.

This explains why a backend can be competitive for forward magnification yet
slower for a model gradient. It also explains why Linear-LD behaves
differently: its event-specific annulus groups and extra source-profile work
change the compiled graph. There is no single caustic/non-caustic rule that
predicts the winner.

Relevant implementation locations:

- lcbinint JAX batching/routing: [python/lcbinint/jax_backend.py](../../../python/lcbinint/jax_backend.py)
- lcbinint fixed warm-up plan: [python/lcbinint/warmup.py](../../../python/lcbinint/warmup.py)
- microLUX trajectory wrapper: [/rogue1_8/nunota/microlux/src/microlux/trajectory_model.py](/rogue1_8/nunota/microlux/src/microlux/trajectory_model.py)
- microLUX contour implementation: [/rogue1_8/nunota/microlux/src/microlux/countour.py](/rogue1_8/nunota/microlux/src/microlux/countour.py)

## Accuracy audit

The speed comparison is separate from the forward accuracy audit against VBM.

| profile | JAX no-warm passes | JAX warm passes | microLUX passes |
|---|---:|---:|---:|
| Uniform | 6/6 | 6/6 | 6/6 |
| Linear LD | 4/6 | 6/6 | 4/6 |

The Linear-LD JAX no-warm misses are resonant_large_source and close_binary;
both have invalid epochs in the un-warmed automatic route. The warm route
passes all six Linear-LD cases. The microLUX Linear-LD misses are
resonant_high_mag and resonant_large_source; their curves remain finite but
exceed the 1e-3 maximum-relative-error threshold at a small number of epochs.
These rows should not be silently described as tolerance-qualified.

All 36 recorded T_VG gradients agree exactly with the corresponding retained
VJP vectors in the stored JSON. This checks the new timing observable's AD
result, but it is not an observational likelihood-gradient validation.

## Recommended paper wording

> We compared compiled steady-state evaluations of six synthetic binary-lens
> light curves using the JAX-enabled lcbinint backend and microLUX, for both a
> uniform source and linear limb darkening. In addition to the forward
> magnification curve, we measured the full physical-parameter Jacobian and a
> model-only value-plus-gradient operation. For the latter, a deterministic
> cotangent constructed solely from the time-grid size defines
> S(eta)=w^T A(eta), and one value_and_grad call returns both S and grad_eta S
> in log-transformed sampler coordinates. No observational data, uncertainties,
> residuals, or likelihood evaluation enter this diagnostic. After the
> corrected warm-up, JAX was faster than microLUX in 4/6 Uniform and 6/6
> Linear-LD value-plus-gradient cases, with median microLUX-to-JAX ratios of
> 5.37 and 4.93. The result characterizes compiled model throughput and should
> not be interpreted as an end-to-end HMC runtime.

The full physical Jacobian T_J can be reported as a secondary AD diagnostic.
The old VJP-only timing should be labelled diagnostic if retained; it should
not be presented as the complete value-plus-gradient cost.

## Reproduction artifacts

- Row-level result: [results.json](../results/recal2026/synthetic_lightcurve_jax_microlux_value_and_grad_final_20260820/results.json)
- Generated report: [REPORT.md](../results/recal2026/synthetic_lightcurve_jax_microlux_value_and_grad_final_20260820/REPORT.md)
- Benchmark harness: [benchmark_synthetic_lightcurve_jax_microlux.py](benchmark_synthetic_lightcurve_jax_microlux.py)
- Original six-case input corpus: [benchmark.json](../results/recal2026/synthetic_lightcurve_benchmark_narrow_windows_20260816/benchmark.json)

The result JSON contains the forward curves, VBM references, physical
Jacobians, value-plus-gradient outputs, diagnostic VJPs, per-lane timing
samples, warm-up metadata, and event-specific annulus plans.
