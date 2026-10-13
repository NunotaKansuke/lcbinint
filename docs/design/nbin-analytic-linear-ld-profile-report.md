# Linear LD: remove brightness quantization at its source

## Change and scope

The experimental overlay evaluates linear LD (d=0) analytically:

I(r²)=1-c*(1-sqrt(max(0,1-clamp(r²,0,1)))).

The shared brightness accessor, direct table fast paths, and Cartesian run-fill weights use the same analytic profile. The d=0 branch avoids the unused second square root. The former continuum nearest-bin bias scale is no longer added to the indicator for this profile, because the underlying quantization no longer exists. No geometry-dependent calibration or fitted safety factor was introduced. General two-coefficient d!=0 profiles retain the previous table behavior; the public experimental error entry supports linear LD only. Production master/default behavior is unchanged because the overlay remains opt-in.

The 20000-bin nearest-r² table produced a discontinuous profile. The quadrature residual model assumes smooth LD, and its continuum table-bias moment cannot describe discrete ray/table aliasing. Treating that moment as a signed correction previously failed; adding its magnitude caused a resolution-independent uncertainty floor. Direct analytic evaluation removes this separate error source instead of estimating it more tightly.

## Evidence: 82 matched saved-reference conditions

References are reused from the four final previous datasets, not recomputed with this implementation. Uniform has 32 conditions and LD has 50, with repeated geometry/coefficient/resolution combinations; these percentages are not operational occurrence rates.

| Indicator / actual error | Previous | Analytic LD |
|---|---:|---:|
| >=1.1 |21/82|15/82|
| >=1.5 |4/82|3/82|
| >=2 |3/82|2/82|
| >=3 |1/82|0/82|
| maximum |4.394876|2.431413|
| minimum |0.992494|0.993650|

All 82 indicator magnifications equal their ordinary scalar calls in the new build. Uniform outputs are also bit equal to the prior build. Extra error-estimation lens maps are zero; brightness-table error fields are zero. Unit core and boundary support pass 2/2. No sampled false tolerance prediction at absolute magnification tolerance 1e-5 or 1e-7.

Nine conditions slightly underestimate (eight LD); worst deficit is 0.635%. The estimator is still approximate, certified=false and complete_error_model=false.

The old maximum (s=1.1,q=1e-5,c=.5,nbin=512) changes from 4.394876 to 1.039538. Its actual error changes from 1.136956e-7 to 2.713102e-7; its indicator changes from 4.996779e-7 to 2.820372e-7. Thus the ratio improvement is partly removal of accidental cancellation in the old computed flux, not exclusively a smaller numerator. LD actual error improves in 31/50 conditions, not all 50. Largest relative LD magnification change is 1.448e-7.

The new maximum is the fresh geometry s=1.3106493282,q=.00012191597,c=1,nbin=128. Actual error 7.353993e-6; signed residual magnitude 8.578169e-6 plus model remainder 9.302426e-6. This remaining overestimate is not a brightness-table contribution.

## Runtime and adoption

Five changing pilot geometries, one thread, seven alternating plain/indicator batches, bins64/128/256/512. Two runs of the analytic candidate were made. Compared with saved v14 medians on this host, final indicator-enabled LD time is about +3.3%, +5.9%, +8.5%, +10.9%. Ordinary LD is about +2.7%, +6.3%, +11.6%, +14.9%. The extra indicator overhead relative to the new scalar is 7.2%,7.7%,5.6%,3.7%. These are sequential saved-baseline timing comparisons, not an interleaved old/new binary benchmark; small timing differences are not conclusive. Uniform is approximately unchanged, with observed indicator timing variation up to 1.9%.

Removing out-of-line table accessor calls in the hot run weights did not materially remove the total runtime increase. Exact brightness is therefore a valid experimental numerical candidate, but no production speed improvement or default promotion is claimed. The remaining speed issue is per-ray analytic brightness, not extra lens mappings or image root solves.

Next work should assess inexpensive continuous-profile evaluation against the analytic candidate, including a controlled brightness approximation remainder, while retaining the shared integration/error structure. Do not reintroduce nearest-bin quantization or a condition-fitted indicator multiplier. The remaining large local-model remainder also needs separate diagnosis.

## Artifacts and commands

- tests/diagnostics/results/recal2026/shared-flux-exact-ld-v1/: before/after source snapshots, provenance, 82-row results, build/audit/unit logs.
- shared-flux-error-exact-ld-cost-v1/: first analytic timing run.
- shared-flux-error-exact-ld-cost-v2/: final hot-path timing run.
- tests/diagnostics/recal2026/shared_flux_exact_ld_audit.py: matched-reference replay.

Build: cmake --build build-boundary-support --target lcbinint_python test_core -j 2.
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support and /home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.shared_flux_exact_ld_audit.
Timing: same environment, module shared_flux_error_audit --output shared-flux-error-exact-ld-cost-v2 (exclusive new output required).
Units: ctest --test-dir build-boundary-support -R 'unit_core|unit_boundary_support' --output-on-failure.
