#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/.note/diagnostic_runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

EXTENSION="$ROOT/build/lcbinint.cpython-310-x86_64-linux-gnu.so"
export LCBININT_EXTENSION="$EXTENSION"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

PYTHONPATH_PREFIX="$OUT_DIR/pythonpath"
mkdir -p "$PYTHONPATH_PREFIX"
cat > "$PYTHONPATH_PREFIX/sitecustomize.py" <<'PY'
import sys
sys.meta_path = [
    finder for finder in sys.meta_path
    if finder.__class__.__module__ != "_lcbinint_editable"
]
PY
export PYTHONPATH="$PYTHONPATH_PREFIX:$ROOT/build:${PYTHONPATH:-}"

{
    echo "root=$ROOT"
    echo "out_dir=$OUT_DIR"
    echo "extension=$EXTENSION"
    echo "python=$(command -v python)"
    echo "started=$(date --iso-8601=seconds)"
    echo
    git -C "$ROOT" rev-parse --short HEAD
    git -C "$ROOT" status --short
} > "$OUT_DIR/metadata.txt" 2>&1

run_job() {
    local name="$1"
    shift
    echo "[$(date --iso-8601=seconds)] start $name" | tee -a "$OUT_DIR/status.log"
    "$@" > "$OUT_DIR/$name.log" 2>&1
    local status=$?
    echo "$status" > "$OUT_DIR/$name.status"
    echo "[$(date --iso-8601=seconds)] done $name status=$status" | tee -a "$OUT_DIR/status.log"
    return 0
}

run_job point_integration_benchmark \
    python "$ROOT/tests/diagnostics/point_integration_benchmark.py" \
        --source-bins 32,40,50,64,80 \
        --reltols 3e-3,1e-3,3e-4 \
        --max-bins 400 \
        --points-per-case 5 \
        --repeat 3 \
        --random 96 \
        --seed 20260624 \
        --reference-tol 1e-5 \
        --vbbl-tol 1e-3 \
        --top 30

run_job lightcurve_adaptive_sweep \
    python "$ROOT/tests/diagnostics/adaptive_source_bins_sweep.py" \
        --source-bins 50 \
        --max-bins 400 \
        --reltol 1e-3 \
        --random 96 \
        --seed 20260624 \
        --random-times 161 \
        --top 30

{
    echo "finished=$(date --iso-8601=seconds)"
    echo "point_status=$(cat "$OUT_DIR/point_integration_benchmark.status" 2>/dev/null || echo missing)"
    echo "lightcurve_status=$(cat "$OUT_DIR/lightcurve_adaptive_sweep.status" 2>/dev/null || echo missing)"
} >> "$OUT_DIR/metadata.txt"

echo "$OUT_DIR"
