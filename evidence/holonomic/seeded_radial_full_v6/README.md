# Full per-epoch comparison

Current: initial.tsv, fold.tsv, deeper.tsv, summary.json, manifest.json. The with_caustic_factor directory is a retained exploratory environment variant, not the reported measurement.

Each geometry/tolerance/u is independently prepared, 3 repetitions with alternating engine order, serial execution. D14 includes epoch_adaptive topology+integration. Candidate includes fresh minimal seed construction and frontier+integration. candidate_no_caustic_ms is candidate_ms minus that repetition's directly measured caustic construction/distance stage; centre roots and seed construction remain included. This assumes same-epoch caustic state is available for another purpose. It does not remove the caustic dependency from the algorithm.

D14-unresolved rows are retained and excluded from speed/accuracy acceptance summaries. CALIBRATED=false.

Reproduce after compiling against the archived minimal-seed source documented in manifest.json:

LCBININT_REPRESENTATIVE_PROBES=critical-seeds OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_seeded_radial_full evidence/holonomic/caustic_seeded_connection_v4/NAME_minimal_seeds.tsv evidence/holonomic/seeded_radial_full_v6/NAME.tsv 3

NAME = initial, fold, deeper. Use a new output directory to retain this checkpoint.
