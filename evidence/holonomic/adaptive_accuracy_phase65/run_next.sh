set -eu
p=evidence/holonomic/adaptive_accuracy_phase65
timeout 120s python -u "$p/vbm_probe.py" > "$p/vbm_probe.log" 2>&1 || echo "VBM probe timeout/failure" >> "$p/vbm_probe.log"
bash "$p/build_k12.sh"
taskset -c 0 /tmp/p65_k12 evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv "$p/k12.tsv" 3
bash "$p/build_initial3.sh"
P65_INITIAL=2 taskset -c 0 /tmp/p65_initial3 evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv "$p/initial3.tsv" 3
python "$p/summarize_ab.py" > "$p/ab_summary.log"
