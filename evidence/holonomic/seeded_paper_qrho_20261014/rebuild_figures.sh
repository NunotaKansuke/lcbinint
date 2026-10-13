#!/usr/bin/env bash
set -euo pipefail
cd /rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/evidence/holonomic/seeded_paper_qrho_20261014/.mplconfig"
/home/nunota/.miniconda3/envs/myenv/bin/python evidence/holonomic/seeded_paper_qrho_20261014/make_figures.py
