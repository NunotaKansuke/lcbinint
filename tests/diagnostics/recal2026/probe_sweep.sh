#!/usr/bin/env bash
# Runs the hard corpus once per seeding policy, one process per core.
#
# The policy is read from the environment once per process, so a policy is a
# process; running them concurrently on separate cores is what keeps the set
# affordable.  Cores are taken from the low end because the campaign's timing
# sweep owns the high end and must not be perturbed.
#
# Only the full policy measures the 400-bin reference: a missing image
# component is a discrete drop in magnification and shows at every resolution,
# so paying for the reference in each ablation would buy nothing.
set -u

ROOT=/rogue1_8/nunota/lcbinint
OUT=${1:-$ROOT/tests/diagnostics/results/recal2026/probe}
mkdir -p "$OUT"
cd "$ROOT" || exit 1

run() {  # name core policy extra...
    local name=$1 core=$2 policy=$3
    shift 3
    LCBININT_PROBE_STATS=1 LCBININT_PROBE_POLICY="$policy" OMP_NUM_THREADS=1 \
        taskset -c "$core" python -m tests.diagnostics.recal2026.probe_study \
        --output "$OUT/$name.json" "$@" \
        > "$OUT/$name.log" 2>&1 &
    echo "$name -> core $core  policy='$policy'  pid $!"
}

# Certificate ladder calibration.  The retired ring and branch-grid heuristic
# cannot be restored through the runtime policy.
#   name          core  policy
run certificate   0     "offsets=8"
run cert_3        5     "offsets=3"                        --no-reference
run cert_2        6     "offsets=2"                        --no-reference
# Negative control.  A cusp's wedge opens along the tangent, so this one is
# expected to lose components; if the corpus does not notice, the corpus is
# not hard enough and nothing else in this study means anything.
run no_tangents   7     "tangents=0"                       --no-reference

wait
echo "all policies done"
