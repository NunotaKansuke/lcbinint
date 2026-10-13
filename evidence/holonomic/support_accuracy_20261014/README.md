# Local-scale support verification

Date: 2026-10-14. Checkout: experiment/seeded-boundary-events, base HEAD 182edd845441e451c585328e66a7968982497723. Current changes are recorded in change.patch and source_and_backend_sha256.txt.

## Finding and change

The source-limb offset ladder can jump completely across a small caustic deep inside a much larger source disk. In the 45 previously unproven geometries, the failing extrema's original probes all returned three images. This is a support-verification limitation, rather than evidence that the value error estimator is wrong.

The resolver retains its image-count-transition predicate. Before exhausting the fine tail of a full eight-offset ray, it tries one local-scale position, using the nearest extremum separated by more than 32 times the polyline uncertainty. The new offset must be shorter than the smallest nominal offset, with a floor of four times the local polyline margin. A successful transition uses the existing per-ray early-out. An unsuccessful attempt leaves every original probe available. Still-unproven extrema receive a bounded local retry, interleaving four directions and stopping on a witness. Unusable probes and unresolved descriptors remain failures. Shortened diagnostic ladders do not gain an implicit full-resolution retry.

The seed build now explicitly selects this checkout's support header through a one-header include overlay. Other ABI headers and linked backend archives remain those used by the original benchmark. Backend archive hashes are recorded, so the numerical comparison uses identical linked backends; this does not retrospectively recover the exact source commit of the original archives.

## Full-corpus verification

Each run has the same 7,216 geometry/profile inputs, targets 1e-3 and 1e-4, three repetitions, and one thread: 14,432 output rows, matched by their complete keys.

| Status | Original | Current |
|---|---:|---:|
| OK | 14,248 | 14,388 |
| SEED_SUPPORT_UNPROVEN | 180 | 40 |
| TopologyUnresolved | 4 | 4 |

All 140 newly supported rows now have actual image-count witnesses. No previously OK row becomes non-OK. The 180 rows represent 45 geometries times two profiles and two tolerances; the remaining 40 represent 10 geometries in cases 38, 52 and 87.

All 14,432 magnifications, reported value errors, integration node counts and radial band counts are bitwise identical in the saved decimal outputs. Existing three saved-VBM discrepancies above their requested tolerance are unchanged. Coverage certification remains zero for every row; the support predicate does not establish global radial coverage. The four angular-integration TopologyUnresolved rows are unaffected.

## Timing

The final standard-runner comparison uses the retained original v45 executable and the executable rebuilt by the standard build script, with the same linked backend archives and timing boundaries. The current route includes seed construction and radial discovery/integration; caustic prebuild is excluded, consistently with the existing benchmark.

- Sum of row median total times, current/original: 0.999405281.
- Median paired total-time ratio: 0.995244681.
- Sum of row median seed times, current/original: 0.999786672.

Other full runs showed total-time differences around -0.06% to +0.41%. These small differences do not establish a speedup; the observed result is essentially unchanged runtime on this corpus. No claim is made about untested workloads.

## Independent checks and rejected alternative

The analytic test uses a closed circular boundary with an independently known image-count function (three outside, five inside), at radii 0.1, 1e-3, 1e-5 and 1e-7 inside a unit source. The original implementation fails at radius 1e-5; the current implementation passes all four. Constant image counts, unusable probes, unresolved descriptors, and a deliberately shortened ladder remain unproven. The regular circle does not exceed its nominal offered work. See analytic_baseline.txt and analytic_check.txt.

A rejected implementation replaced fine offsets globally. It rescued 140 rows but made 40 previously OK rows unproven. Its full table and transition audit are retained in candidate_full.tsv and audit.json; that implementation is not installed.

## Reproduction and remaining work

Run `bash benchmarks/holonomic/check_seed_support_local_scale.sh` from the checkout for the analytic checks. `bash evidence/holonomic/support_accuracy_20261014/reproduce.sh NEW_OUTPUT_DIRECTORY` verifies source/backend hashes, rebuilds, and repeats the standard old/current full benchmark. The supplied default output is inside this checkout and the script refuses to overwrite existing result tables.

The remaining ten support geometries still lack a witness. Their exact failure mechanism is not established by this change and their status is deliberately retained. Improving the local caustic representation/frame is a separate candidate; treating their small VBM differences as proof of support would weaken the present predicate.

Nominal `certified_offered` counts describe the original descriptor ladder; actual `certified_solves` also include local refinement calls. They need not retain an offered >= solved inequality.
