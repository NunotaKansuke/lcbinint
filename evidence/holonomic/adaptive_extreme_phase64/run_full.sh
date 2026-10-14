set -eu
out=evidence/holonomic/adaptive_extreme_phase64/full
input=evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv
for item in base1 cache1 cache2 base2; do
    run=${item%?}; rep=${item: -1}
    taskset -c 0 "/tmp/p64_$run" "$input" warm-timing > "$out/${run}${rep}.tsv"
done
