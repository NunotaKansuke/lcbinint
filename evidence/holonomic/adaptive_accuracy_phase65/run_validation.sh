set -eu
p=evidence/holonomic/adaptive_accuracy_phase65
timeout 120s python -u "$p/vbm_roundtrip.py" > "$p/vbm_roundtrip.log" 2>&1 || echo "VBM roundtrip probe timeout/failure" >> "$p/vbm_roundtrip.log"
bash "$p/validate_safety.sh"
bash "$p/validate_gradient.sh"
taskset -c 0 /tmp/p65_probe evidence/holonomic/adaptive_accuracy_phase65/rounded_input.tsv evidence/holonomic/adaptive_accuracy_phase65/rounded_v2.tsv 1
