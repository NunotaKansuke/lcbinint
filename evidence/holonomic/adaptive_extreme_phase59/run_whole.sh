set -eu
input=evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv
out=evidence/holonomic/adaptive_extreme_phase59
taskset -c 0 /tmp/p59_whole_base "$input" "$out/whole_baseline.tsv" 3
taskset -c 0 /tmp/p59_whole "$input" "$out/whole_candidate.tsv" 3
python benchmarks/holonomic/summarize_adaptive_reciprocal.py --baseline ../adaptive_extreme_phase59/whole_baseline.tsv --candidate ../adaptive_extreme_phase59/whole_candidate.tsv --output ../adaptive_extreme_phase59/whole_summary.json
