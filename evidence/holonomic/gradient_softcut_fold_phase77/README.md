# Phase 77: fold-adjacent physical-complex soft cut

This evidence compares the unmodified `cd1343cc` adaptive path with the
candidate that coalesces only same-topology arc cells whose soft cut lies
inside the configured Fejer-II first-node stride from a certified physical
fold.  The full source topology/events remain unchanged.

`reference_cases.tsv` is the 110-row corpus from
`evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv` at the base
commit. `corpus_baseline_final.tsv` and `corpus_candidate_final.tsv` contain
matched cold rows at value RelTol 1e-3/1e-4 with `None` and `ValueFirst`.
`rand035_*_custom_config.tsv` is the focused four-way-per-u run with the same
explicit ValueFirst gradient settings used during diagnosis. The independent
5-point FD references are listed in `independent_fd_reference.tsv`; raw
independent radial/angular integrations are in
`independent_rand035_fd_raw.tsv` and the fresh order-128/256 FD step study is
in `fd_reference.tsv`.

Reproduce from the repository root with `bash evidence/holonomic/gradient_softcut_fold_phase77/run_ab.sh`.
The script archives the parent commit for the baseline build and uses the
working tree for the candidate. Build flags are C++17, `-O3 -march=native
-funroll-loops -ffp-contract=fast -fno-math-errno`, linked with quadmath; runs
are pinned to CPU 0. Timing columns are retained as raw diagnostics and are
not used as an acceptance claim.
