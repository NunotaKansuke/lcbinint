#!/usr/bin/env bash
set -eu
g++ -O3 -march=native -std=c++17 -fext-numeric-literals -Isrc evidence/holonomic/seed_count_v26/audit_count.cpp -lquadmath -o build-seeded-events/audit_count_v26
