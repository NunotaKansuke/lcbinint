set -eu
out=evidence/holonomic/adaptive_extreme_phase59
input=evidence/holonomic/adaptive_extreme_phase56/input_snapshot.tsv
for item in base1 cache1 cache2 base2; do
    run=${item%?}; rep=${item: -1}
    taskset -c 0 "/tmp/p59_$run" "$input" warm-timing > "$out/${run}${rep}.tsv"
done
