# Seed frontier versus current inverse ray

Accepted current sources: source-final and manifest.json; shared diff includes preserved earlier research changes. No default dispatcher promotion; CALIBRATED=false.

Current candidate/D14 data: initial_final.tsv, fold_final.tsv, deeper_final.tsv. 754 D14-resolved comparisons, zero candidate failures; 142 unresolved references preserved and separated. Timing includes seed generation, with both full and caustic-excluded columns.

Current standard inverse-ray data: current_inverse_native.tsv (64/128/256/512), current_inverse_native_low.tsv (16/32). These use the current main checkout's Cartesian/polar kernels rebuilt at -O3 -march=native. The summary chooses the fastest accuracy-qualified grid per case without counting selection cost, deliberately favouring inverse ray. The normal matched set has 87 source/u conditions at tolerance1e-4. Accuracy is difference from D14, not an independent oracle.

fold_inverse_pilot.tsv is a bounded 10-condition supplement, at16/32/64/128, with fold_pilot_reference.tsv and fold_pilot_summary.json. It is not a complete fold campaign. fold_inverse_partial.tsv is a stopped, incomplete exploratory high-grid run. The high-grid polar planetary cases were expensive; no process remains running.

before_native_seed/ retains the prior measurements before matching the polynomial solver compilation flags. Other files are retained intermediate tests. deeper_yield.tsv has two unresolved early-yield cases and is not accepted. The final path reuses partial panels to obtain more witnesses when a count change cannot yet be localized. *_extended.tsv and source-extended record a rejected long-double assembly experiment with no observed point-count improvement. Generic-build current_inverse*.tsv without native in the name are superseded by the matched native build.

Re-run candidate, using a new output directory:
LCBININT_REPRESENTATIVE_PROBES=critical-seeds OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_seeded_radial_full evidence/holonomic/caustic_seeded_connection_v4/NAME_minimal_seeds.tsv OUTPUT/NAME_final.tsv 3 0
NAME=initial,fold,deeper.

Standard normal benchmark:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_inverse_ray_current evidence/holonomic/seeded_radial_full_v6/initial.tsv OUTPUT/current_inverse_native.tsv
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_inverse_ray_current evidence/holonomic/seeded_radial_full_v6/initial.tsv OUTPUT/current_inverse_native_low.tsv low

Summary:
python3 benchmarks/holonomic/summarize_seeded_inverse_comparison.py OUTPUT

Compiler, source and input hashes and timing definitions are in manifest.json. Libraries are built from the archived minimal seed source for the candidate, and current main source for standard inverse ray. Use matching headers for each library.
