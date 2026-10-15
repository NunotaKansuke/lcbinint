#!/bin/bash
set -euo pipefail
export TMPDIR="$PWD/build-algebraic-boundary/fused-gradient/scratch"
g++ -std=gnu++17 -O2 -DNDEBUG -fext-numeric-literals -include cstdio -Ibuild-algebraic-boundary/stress-cause/include -Isrc -Iinclude -Ithird_party/skowron_gould -I/rogue1_8/nunota/local/gsl/include -c src/lcbinint/magnification/holonomic/fused_gradient.cpp -o build-algebraic-boundary/stress-cause/fused.o
g++ -std=gnu++17 -O2 -DNDEBUG -fext-numeric-literals -Isrc -Iinclude -Ithird_party/skowron_gould -I/rogue1_8/nunota/local/gsl/include evidence/holonomic/stress_cause_20261015/probe.cpp build-algebraic-boundary/stress-cause/fused.o -Wl,--start-group build-algebraic-boundary/python-api/liblcbinint_magnification.a build-algebraic-boundary/python-api/liblcbinint_lightcurve.a -Wl,--end-group -L/rogue1_8/nunota/local/gsl/lib -Wl,-rpath,/rogue1_8/nunota/local/gsl/lib -lgsl -lgslcblas -lquadmath -o build-algebraic-boundary/stress-cause/probe
