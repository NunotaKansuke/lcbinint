# Selected triple algebraic checkpoint

The selected implementation transports flux and all eight canonical parameter derivatives together. The default gradient policy refines on flux error only; value-only integration evaluates no parameter directions. Strict derivative-error control remains an explicit diagnostic option.

Accepted changes include nested Patterson transport, moving real/complex radial charts, ordinary and critical angular sensitivity moments, retained nested packets, explicit real physical arithmetic, direct promoted harmonic rotation, shared positive embedded flux error estimation, and slope-only stationary discovery. Physical endpoint validation, complete support, and local precision repair remain active.

The subsequent partial-sector tracking and real-fold leading-term subtraction experiments did not outperform this selection. They are not enabled in this checkpoint. In particular, no leading-fold subtraction header is required by the selected source. Local investigation artifacts remain outside this commit.

## Validation

Release native build with `cmake --build build-triple-native -j2`; CTest 29 tests, including public API and first-order transport. Public independent physical FD gate: 88 components, maximum error/budget 0.253045. Saved selected C ABI gates: broad 120/120, saved 86/86, new holdout 24/24. Fresh independent five-point scalar FD: 32 gradient comparisons and 32 reference-half-budget checks, maximum combined error/budget 0.12223. Offline gradient budget is `1e-4 + 2e-4*abs(reference)`.

See the adjacent `selected-default-*.json` files. The library paths/hashes in those historical records identify the measured native binaries; binaries are not stored in Git. Numerical acceptance is bounded to the recorded cases, not universal gradient certification.

## Performance and remaining work

The selected version improved 1.9–5.5% against the preceding guarded-moments version in paired CPU0 thread-time tests (4 warmups and 24 rotated/reversed samples). CPU frequency was uncontrolled. Added gradient cost at rtol 1e-3 / 1e-4 was contact uniform 53.0% / 37.8%, large uniform 88.6% / 83.1%, contact LD 99.8% / 87.2%, large LD 176.6% / 107.2%. The requested 30% overhead target is **not achieved**. These data are not a new VBM comparison.

`gradient_valid` denotes a finite complete-support result; `gradient_reliable` and `gradient_precision_checked` remain false under the default value-mesh policy. Online derivative error estimates remain unchecked, consistently with the documented contract.
