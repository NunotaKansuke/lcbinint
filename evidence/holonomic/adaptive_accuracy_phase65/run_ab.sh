set -eu
p=evidence/holonomic/adaptive_accuracy_phase65
input=evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv
taskset -c 0 /tmp/p65_ab "$input" "$p/base.tsv" 3
P65_SAFETY=1 taskset -c 0 /tmp/p65_ab "$input" "$p/safety1.tsv" 3
P65_INITIAL=4 taskset -c 0 /tmp/p65_ab "$input" "$p/initial15.tsv" 3
