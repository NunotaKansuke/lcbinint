#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
evidence="$repo_root/evidence/holonomic/gradient_softcut_fold_phase77"
baseline_commit=${BASELINE_COMMIT:-cd1343cc2bd7fc8bfde17c5556772ed6420f14b4}
cpu=${BENCH_CPU:-0}
cxx=${CXX:-c++}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/baseline"
git -C "$repo_root" archive "$baseline_commit" | tar -x -C "$tmp/baseline"

flags=(-std=gnu++17 -O3 -DNDEBUG -march=native -funroll-loops
       -ffp-contract=fast -fno-math-errno)
input="$evidence/reference_cases.tsv"
corpus_src="$evidence/softcut_corpus.cpp"
epoch_src="$evidence/softcut_fold_epoch.cpp"

"$cxx" "${flags[@]}" -I"$tmp/baseline/src" "$corpus_src" \
    -o "$tmp/corpus_baseline" -lquadmath
"$cxx" "${flags[@]}" -I"$tmp/baseline/src" "$epoch_src" \
    -o "$tmp/epoch_baseline" -lquadmath
"$cxx" "${flags[@]}" -I"$repo_root/src" -DSOFTCUTFOLD_CANDIDATE \
    "$corpus_src" -o "$tmp/corpus_candidate" -lquadmath
"$cxx" "${flags[@]}" -I"$repo_root/src" -DSOFTCUTFOLD_CANDIDATE \
    "$epoch_src" -o "$tmp/epoch_candidate" -lquadmath
"$cxx" "${flags[@]}" -I"$repo_root/src" \
    -I"$repo_root/tests/holonomic_cpp" \
    "$evidence/softcut_fold_reference.cpp" -o "$tmp/fd_reference" -lquadmath

taskset -c "$cpu" "$tmp/corpus_baseline" "$input" \
    > "$evidence/corpus_baseline_final.tsv"
taskset -c "$cpu" "$tmp/epoch_baseline" \
    > "$evidence/rand035_baseline_custom_config.tsv"
taskset -c "$cpu" "$tmp/corpus_candidate" "$input" \
    > "$evidence/corpus_candidate_final.tsv"
taskset -c "$cpu" "$tmp/epoch_candidate" \
    > "$evidence/rand035_candidate_custom_config.tsv"
taskset -c "$cpu" "$tmp/fd_reference" \
    > "$evidence/fd_reference.tsv"
python3 "$evidence/summarize.py" \
    "$evidence/corpus_baseline_final.tsv" \
    "$evidence/corpus_candidate_final.tsv" \
    "$evidence/rand035_baseline_custom_config.tsv" \
    "$evidence/rand035_candidate_custom_config.tsv" \
    "$evidence/independent_fd_reference.tsv" \
    "$evidence/independent_mu_reference.tsv" \
    > "$evidence/summary.json"
